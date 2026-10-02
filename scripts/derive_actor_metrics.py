#!/usr/bin/env python3
"""Derive 18 strictly prior actor/market metrics (Python 3.11+, standard library).

python3 scripts/derive_actor_metrics.py data/market_1897059 --out data/actor_metrics
python3 scripts/derive_actor_metrics.py --input-root data --out data/actor_metrics \
    --closed-positions closed_positions.jsonl --returns-file returns.jsonl

Optional --sft-dir adds the features to a NEW prepared dataset under OUT/sft.
Trade-only exports support four activity metrics. Performance metrics require
completed-position accounting; risk metrics require regular capital-adjusted
returns. Unknown values remain null. Current API snapshots are never read.
See docs/actor_metrics.md for input schemas, formulas and temporal guarantees.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

ADDRESS = re.compile(r'0x[0-9a-fA-F]{40}')
CONDITION = re.compile(r'0x[0-9a-fA-F]{64}')

# Reuse the standalone collector's inventory accounting in all feature paths.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from build_actor_dataset import InMarketPnL, pnl_prompt_fields, pnl_state_for_export, validate_pnl_features


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _bad_number(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


def loads(text):
    return json.loads(text, parse_float=Decimal, parse_constant=_bad_number,
                      object_pairs_hook=_unique_object)


def json_text(value):
    def encode(item):
        if isinstance(item, Decimal) and item.is_finite():
            return str(item)
        raise ValueError(f'Not a finite JSON value: {type(item).__name__}')
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False, default=encode)


def safe_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Missing or unsafe file: {path}')
    return path


def read_json(path):
    return loads(safe_file(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(value) + '\n', encoding='utf-8')


def iter_jsonl(path):
    path = safe_file(path)
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try:
                require(bool(line.strip()), 'Blank JSONL row')
                value = loads(line)
                require(isinstance(value, dict), 'JSONL row must be an object')
                yield value
            except (ValueError, TypeError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error


def sha256(path):
    digest = hashlib.sha256()
    with safe_file(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def timestamp_us(value):
    require(isinstance(value, str), 'Timestamps must be ISO strings with a UTC offset')
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError(f'Invalid timestamp: {value}') from error
    require(stamp.tzinfo is not None and stamp.utcoffset() is not None,
            f'Timestamp lacks UTC offset: {value}')
    delta = stamp.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def number(value, name):
    require(isinstance(value, (str, int, Decimal)) and not isinstance(value, bool),
            f'{name} must be a finite decimal string or JSON number')
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f'Invalid decimal for {name}') from error
    require(result.is_finite(), f'{name} must be finite')
    return result


def identity(row):
    actor, condition = row.get('actor_id'), row.get('condition_id')
    require(isinstance(actor, str) and ADDRESS.fullmatch(actor), 'Invalid actor_id')
    require(isinstance(condition, str) and CONDITION.fullmatch(condition), 'Invalid condition_id')
    return actor.lower(), condition.lower()


def load_closed_positions(path):
    groups, seen = defaultdict(list), set()
    if path is None:
        return groups
    for row in iter_jsonl(path):
        key = identity(row)
        position_id = row.get('position_id')
        require(isinstance(position_id, str) and position_id.strip(), 'Missing position_id')
        require((*key, position_id) not in seen, f'Duplicate completed position: {position_id}')
        seen.add((*key, position_id))
        opened, closed, known = (timestamp_us(row.get(name))
                                 for name in ('opened_at', 'closed_at', 'known_at'))
        require(opened <= closed <= known, 'Require opened_at <= closed_at <= known_at')
        cost, pnl = number(row.get('entry_cost'), 'entry_cost'), number(row.get('net_pnl'), 'net_pnl')
        require(cost > 0, 'Completed position entry_cost must be positive')
        groups[key].append({'opened_us': opened, 'closed_us': closed, 'known_us': known,
                            'entry_cost': cost, 'net_pnl': pnl})
    return groups


def load_returns(path):
    groups, seen = defaultdict(list), set()
    if path is None:
        return groups
    for row in iter_jsonl(path):
        key = identity(row)
        start, end, known = (timestamp_us(row.get(name))
                            for name in ('period_start', 'period_end', 'known_at'))
        require(start < end <= known, 'Require period_start < period_end <= known_at')
        require((*key, start) not in seen, 'Duplicate return period for actor/market')
        seen.add((*key, start))
        require(row.get('capital_flow_adjusted') is True,
                'Return observations must declare capital_flow_adjusted: true')
        value = number(row.get('period_return'), 'period_return')
        require(value >= -1, 'An unlevered capital return cannot be less than -1')
        groups[key].append({'start_us': start, 'end_us': end, 'known_us': known,
            'return': value, 'benchmark_return': number(row.get('benchmark_return', '0'), 'benchmark_return'),
            'target_return': number(row.get('target_return', '0'), 'target_return')})
    for key, rows in groups.items():
        rows.sort(key=lambda row: row['start_us'])
        for previous, current in zip(rows, rows[1:]):
            require(previous['end_us'] <= current['start_us'], f'Overlapping returns: {key}')
            require(previous['return'] != -1, 'Cannot extend a return series after its capital reaches zero')
    return groups


def discover_exports(exports, input_root):
    require(not (exports and input_root), 'Use export paths OR --input-root')
    if exports:
        paths = [Path(path).resolve() for path in exports]
    else:
        root = Path(input_root or 'data').resolve()
        require(root.is_dir(), f'Input root does not exist: {root}')
        candidates = [root] if (root / 'manifest.json').is_file() else sorted(root.iterdir())
        paths = []
        for path in candidates:
            if path.is_dir() and not path.is_symlink() and (path / 'manifest.json').is_file():
                if read_json(path / 'manifest.json').get('format') == 'actor_market_intervals_v1':
                    paths.append(path.resolve())
    require(paths and len(set(paths)) == len(paths), 'Supply distinct completed actor exports')
    sources, conditions, market_ids = [], set(), set()
    for path in paths:
        manifest, market = read_json(path / 'manifest.json'), read_json(path / 'market.json')
        require(manifest.get('format') == 'actor_market_intervals_v1', f'Unsupported export: {path}')
        condition = market.get('condition_id')
        require(isinstance(condition, str) and CONDITION.fullmatch(condition), 'Invalid market condition_id')
        market_id = str(market.get('market_id', ''))
        require(market_id and market_id != 'None', 'Missing market_id')
        require(str(manifest.get('market_id')) == market_id and manifest.get('condition_id') == condition,
                f'Manifest/market identity mismatch: {path}')
        require(condition.lower() not in conditions and market_id not in market_ids,
                'Duplicate market exports; choose one capture per market')
        conditions.add(condition.lower()); market_ids.add(market_id)
        actor_dir = path / 'actors'
        require(actor_dir.is_dir() and not actor_dir.is_symlink(), f'Missing or unsafe actors directory: {path}')
        outcomes = {item['outcome'] for item in market.get('tokens', [])}
        require(len(outcomes) == 2, 'Expected binary market outcomes')
        sources.append({'path': path, 'market_id': market_id, 'condition_id': condition.lower(),
                        'outcomes': outcomes, 'manifest': manifest})
    return sources


def actor_trade_groups(path, source):
    """Read complete raw pairs. Only trade labels become historical executions."""
    actor = path.name.removesuffix('.gz').removesuffix('.jsonl')
    require(ADDRESS.fullmatch(actor), f'Invalid actor filename: {path}')
    actor = actor.lower()
    rows = iter(iter_jsonl(path))
    previous, index, execution_count = None, 0, 0
    groups = []
    pnl = None
    while True:
        gap = next(rows, None)
        if gap is None:
            break
        trade = next(rows, None)
        require(trade is not None, f'Incomplete interval/trade pair: {path}')
        for offset, row in enumerate((gap, trade)):
            require(identity(row) == (actor, source['condition_id']) and
                    str(row.get('market_id')) == source['market_id'], f'Row identity mismatch: {path}')
            require(type(row.get('row_index')) is int and row['row_index'] == index + offset,
                    f'Non-contiguous row_index: {path}')
        require(gap.get('record_type') == 'interval' and gap.get('label') == {'action': 'NO_TRADE'},
                f'Invalid interval row: {path}')
        require(trade.get('record_type') == 'trade', f'Expected trade row: {path}')
        when = timestamp_us(trade.get('timestamp'))
        require(previous is None or previous < when, f'Actor timestamps must strictly increase: {path}')
        interval = gap.get('interval')
        require(isinstance(interval, dict) and interval == trade.get('context_interval') and
                timestamp_us(interval.get('end')) == when, f'Mismatched interval/trade timestamp: {path}')
        label = trade.get('label')
        require(isinstance(label, dict) and label.get('action') == 'TRADE' and
                isinstance(label.get('trades'), list) and label['trades'], f'Missing trade label: {path}')
        normalized, expected = [], []
        for item in label['trades']:
            require(isinstance(item, dict) and timestamp_us(item.get('time')) == when,
                    'Execution timestamp differs from its trade group')
            require(item.get('side') in ('BUY', 'SELL') and item.get('outcome') in source['outcomes'],
                    'Invalid execution side or outcome')
            shares, price = number(item.get('shares'), 'shares'), number(item.get('price'), 'price')
            require(shares > 0 and 0 <= price <= 1, 'Execution shares/price out of range')
            require(isinstance(item['shares'], str) and isinstance(item['price'], str),
                    'Actor export amounts must preserve decimal strings')
            normalized.append({'time_us': when, 'side': item['side'], 'shares': shares, 'price': price})
            expected.append({key: item[key] for key in ('side', 'outcome', 'shares', 'price')})
        if pnl is None:
            pnl = pnl_state_for_export(source['manifest'], gap)
        require(gap.get('market_context') == trade.get('market_context'), 'Adjacent market prices disagree')
        pnl_features = pnl.snapshot(trade.get('market_context'), trade['timestamp'])
        validate_pnl_features(gap, trade, pnl_features, source['manifest'])
        groups.append({'pnl_features': pnl_features,
                       'pnl_opening_history': gap.get('in_market_pnl_opening_history', []) if index == 0 else [],
                       'time_us': when, 'timestamp': trade['timestamp'], 'row_index': index + 1,
                       'trades': normalized, 'expected': expected})
        pnl.apply(expected)
        execution_count += len(normalized)
        previous, index = when, index + 2
    require(groups, f'Empty actor export: {path}')
    return actor, groups, {'actors': 1, 'rows': index, 'distinct_trade_times': len(groups),
                           'trade_observations': execution_count}


# All arithmetic remains decimal; no future row can enter these aggregates.
from decimal import Decimal, localcontext

ACTOR_METRIC_NAMES = (
    'net_realized_pnl', 'mean_realized_roi', 'win_rate',
    'average_winning_profit', 'average_losing_loss', 'payoff_ratio',
    'profit_factor', 'historical_expectancy', 'sharpe_ratio',
    'sortino_ratio', 'maximum_drawdown', 'return_volatility',
    'consecutive_loss_streak', 'average_holding_seconds',
    'average_execution_notional', 'execution_notional_cv',
    'executions_per_day', 'buy_notional_share',
)


def _metric_decimal_text(value):
    """JSON-safe decimal text, without float conversion or negative zero."""
    if not value.is_finite():
        raise ValueError('Metric arithmetic produced a non-finite number')
    if not value:
        return '0'
    result = format(value, 'f')
    return result.rstrip('0').rstrip('.') if '.' in result else result


def _metric_mean(values):
    return sum(values, Decimal(0)) / Decimal(len(values))


def _metric_stdev(values, *, sample):
    denominator = len(values) - (1 if sample else 0)
    if denominator <= 0:
        return None
    mean = _metric_mean(values)
    variance = sum(((value - mean) ** 2 for value in values), Decimal(0)) / Decimal(denominator)
    return variance.sqrt()


def compute_metrics(trades, closed, returns, query_us, lookback_seconds=None,
                    min_return_periods=30):
    """Compute 18 features strictly before one actor/market decision time.

    The caller validates source records and isolates actor and market identity.
    Trades describe captured executions, closed rows describe complete positions,
    and returns describe a capital-flow-adjusted periodic equity series. Current
    API snapshots and the current execution are never used.
    """
    if type(query_us) is not int:
        raise ValueError('query_us must be an integer number of microseconds')
    if type(min_return_periods) is not int or min_return_periods < 2:
        raise ValueError('min_return_periods must be at least 2')
    start_us = None
    if lookback_seconds is not None:
        duration = Decimal(str(lookback_seconds))
        if not duration.is_finite() or duration <= 0:
            raise ValueError('lookback_seconds must be finite and positive')
        duration_us = duration * Decimal(1000000)
        if duration_us != duration_us.to_integral_value():
            raise ValueError('lookback_seconds cannot have sub-microsecond precision')
        start_us = query_us - int(duration_us)

    prior_trades = [row for row in trades
                    if row['time_us'] < query_us
                    and (start_us is None or row['time_us'] >= start_us)]
    prior_closed = [row for row in closed
                    if row['closed_us'] < query_us and row['known_us'] < query_us
                    and (start_us is None or row['closed_us'] >= start_us)]
    prior_returns = [row for row in returns
                     if row['end_us'] < query_us and row['known_us'] < query_us
                     and (start_us is None or row['start_us'] >= start_us)]
    prior_closed.sort(key=lambda row: row['closed_us'])
    prior_returns.sort(key=lambda row: row['start_us'])

    values = dict.fromkeys(ACTOR_METRIC_NAMES)
    reasons = {}
    behavior_start_us = (start_us if start_us is not None else
                         min((row['time_us'] for row in prior_trades), default=None))
    with localcontext() as context:
        context.prec = 50

        def set_metric(name, value):
            values[name] = _metric_decimal_text(value)

        completed_names = (
            'net_realized_pnl', 'mean_realized_roi', 'win_rate',
            'average_winning_profit', 'average_losing_loss', 'payoff_ratio',
            'profit_factor', 'historical_expectancy', 'consecutive_loss_streak',
            'average_holding_seconds',
        )
        pnl = [row['net_pnl'] for row in prior_closed]
        wins = [value for value in pnl if value > 0]
        losses = [-value for value in pnl if value < 0]
        if not prior_closed:
            for name in completed_names:
                reasons[name] = 'no_eligible_completed_positions'
        else:
            count = Decimal(len(prior_closed))
            set_metric('net_realized_pnl', sum(pnl, Decimal(0)))
            set_metric('mean_realized_roi', _metric_mean([
                row['net_pnl'] / row['entry_cost'] for row in prior_closed]))
            set_metric('win_rate', Decimal(len(wins)) / count)
            set_metric('historical_expectancy', _metric_mean(pnl))
            set_metric('average_holding_seconds', _metric_mean([
                Decimal(row['closed_us'] - row['opened_us']) / Decimal(1000000)
                for row in prior_closed]))
            if wins:
                set_metric('average_winning_profit', _metric_mean(wins))
            else:
                reasons['average_winning_profit'] = 'no_profitable_completed_positions'
            if losses:
                set_metric('average_losing_loss', _metric_mean(losses))
                set_metric('profit_factor', sum(wins, Decimal(0)) / sum(losses, Decimal(0)))
            else:
                reasons['average_losing_loss'] = 'no_losing_completed_positions'
                reasons['profit_factor'] = 'no_losing_completed_positions'
            if wins and losses:
                set_metric('payoff_ratio', _metric_mean(wins) / _metric_mean(losses))
            else:
                reasons['payoff_ratio'] = 'requires_both_profitable_and_losing_completed_positions'

            # Count from the latest closure backwards. Tied all-loss closures
            # add to the streak; tied non-loss closures reset it. A mixed tie
            # can change the answer, so there is no invented internal ordering.
            index = len(prior_closed) - 1
            streak = 0
            ambiguous = False
            while index >= 0:
                instant = prior_closed[index]['closed_us']
                group = []
                while index >= 0 and prior_closed[index]['closed_us'] == instant:
                    group.append(prior_closed[index]['net_pnl'])
                    index -= 1
                loss_count = sum(value < 0 for value in group)
                if loss_count == len(group):
                    streak += loss_count
                    continue
                if loss_count:
                    ambiguous = True
                break
            if ambiguous:
                reasons['consecutive_loss_streak'] = 'ambiguous_close_order'
            else:
                values['consecutive_loss_streak'] = streak

        notionals = [row['shares'] * row['price'] for row in prior_trades]
        if not prior_trades:
            for name in ('average_execution_notional', 'execution_notional_cv', 'buy_notional_share'):
                reasons[name] = 'no_eligible_captured_executions'
        else:
            average_notional = _metric_mean(notionals)
            set_metric('average_execution_notional', average_notional)
            if len(notionals) < 2:
                reasons['execution_notional_cv'] = 'fewer_than_two_captured_executions'
            elif not average_notional:
                reasons['execution_notional_cv'] = 'zero_mean_execution_notional'
            else:
                set_metric('execution_notional_cv', _metric_stdev(notionals, sample=False) / average_notional)
            total_notional = sum(notionals, Decimal(0))
            if total_notional:
                buy_notional = sum((row['shares'] * row['price'] for row in prior_trades
                                    if row['side'] == 'BUY'), Decimal(0))
                set_metric('buy_notional_share', buy_notional / total_notional)
            else:
                reasons['buy_notional_share'] = 'zero_total_execution_notional'
        if behavior_start_us is not None and query_us > behavior_start_us:
            elapsed_days = Decimal(query_us - behavior_start_us) / Decimal(86400000000)
            set_metric('executions_per_day', Decimal(len(prior_trades)) / elapsed_days)
        else:
            reasons['executions_per_day'] = 'no_positive_observation_window'

        risk_names = ('sharpe_ratio', 'sortino_ratio', 'maximum_drawdown', 'return_volatility')
        series_problem = None
        if not prior_returns:
            series_problem = 'no_eligible_capital_adjusted_return_periods'
        else:
            lengths = {row['end_us'] - row['start_us'] for row in prior_returns}
            if len(lengths) != 1 or next(iter(lengths)) <= 0:
                series_problem = 'return_periods_have_inconsistent_duration'
            elif any(left['end_us'] != right['start_us']
                     for left, right in zip(prior_returns, prior_returns[1:])):
                series_problem = 'return_periods_are_not_contiguous'
        if series_problem:
            for name in risk_names:
                reasons[name] = series_problem
        else:
            wealth = Decimal(1)
            peak = Decimal(1)
            maximum_drawdown = Decimal(0)
            for row in prior_returns:
                wealth *= Decimal(1) + row['return']
                peak = max(peak, wealth)
                maximum_drawdown = max(maximum_drawdown, (peak - wealth) / peak)
            set_metric('maximum_drawdown', maximum_drawdown)
            if len(prior_returns) < min_return_periods:
                for name in ('sharpe_ratio', 'sortino_ratio', 'return_volatility'):
                    reasons[name] = 'insufficient_return_periods'
            else:
                raw_returns = [row['return'] for row in prior_returns]
                excess_returns = [row['return'] - row['benchmark_return'] for row in prior_returns]
                target_returns = [row['return'] - row['target_return'] for row in prior_returns]
                volatility = _metric_stdev(raw_returns, sample=True)
                set_metric('return_volatility', volatility)
                excess_volatility = _metric_stdev(excess_returns, sample=True)
                if excess_volatility:
                    set_metric('sharpe_ratio', _metric_mean(excess_returns) / excess_volatility)
                else:
                    reasons['sharpe_ratio'] = 'zero_excess_return_volatility'
                downside_deviation = _metric_mean([
                    min(value, Decimal(0)) ** 2 for value in target_returns]).sqrt()
                if downside_deviation:
                    set_metric('sortino_ratio', _metric_mean(target_returns) / downside_deviation)
                else:
                    reasons['sortino_ratio'] = 'zero_target_downside_deviation'

    return {
        'values': values,
        'unavailable_reasons': reasons,
        'sample_counts': {
            'captured_executions': len(prior_trades),
            'completed_positions': len(prior_closed),
            'profitable_completed_positions': len(wins),
            'losing_completed_positions': len(losses),
            'breakeven_completed_positions': len(prior_closed) - len(wins) - len(losses),
            'eligible_return_periods': len(prior_returns),
        },
        'window': {
            'start_us': start_us,
            'end_us': query_us,
            'start_inclusive': True,
            'end_inclusive': False,
            'execution_frequency_start_us': behavior_start_us,
            'return_window_rule': 'whole_period_start_at_or_after_window_start',
        },
        'minimum_return_periods': min_return_periods,
        'risk_ratios_annualized': False,
        'return_period_seconds': (None if series_problem else
            _metric_decimal_text(Decimal(prior_returns[0]['end_us'] - prior_returns[0]['start_us']) / Decimal(1000000))),
        'metric_scope': {
            'execution_metrics': 'captured_actor_market_executions',
            'performance_metrics': 'completed_positions_only',
            'risk_metrics': 'caller_supplied_capital_adjusted_returns',
        },
    }



def select_features(value):
    """Validate an optional comma-separated model feature subset."""
    if value is None:
        return list(ACTOR_METRIC_NAMES)
    names = [part.strip() for part in value.split(',')]
    require(names and all(names) and len(names) == len(set(names)),
            '--features must contain distinct comma-separated metric names')
    require(set(names) <= set(ACTOR_METRIC_NAMES),
            'Unknown features: ' + ', '.join(sorted(set(names) - set(ACTOR_METRIC_NAMES))))
    return [name for name in ACTOR_METRIC_NAMES if name in names]


def model_metric_fields(core, config):
    """Round derived prompt values only, retaining full precision in raw audit rows."""
    digits = config.get('metric_significant_digits', 10)
    require(type(digits) is int and 4 <= digits <= 16,
            '--metric-significant-digits must be between 4 and 16')
    names = config.get('selected_features', list(core['values']))
    require(isinstance(names, list) and names and len(names) == len(set(names))
            and set(names) <= set(core['values']), 'Invalid selected metric names')
    values = {}
    for name in names:
        value = core['values'][name]
        if value is None:
            continue
        if type(value) is int:
            values[name] = value
        else:
            value = number(value, name)
            values[name] = str(Decimal(format(value, f'.{digits}g')).normalize()) if value else '0'
    counts = core['sample_counts']
    if 'selected_features' in config:
        execution_names = {'average_execution_notional', 'execution_notional_cv',
                           'executions_per_day', 'buy_notional_share'}
        risk_names = {'sharpe_ratio', 'sortino_ratio', 'maximum_drawdown', 'return_volatility'}
        completed_names = set(ACTOR_METRIC_NAMES) - execution_names - risk_names
        keep_counts = set()
        for family, count in ((execution_names, 'captured_executions'),
                              (risk_names, 'eligible_return_periods'),
                              (completed_names, 'completed_positions')):
            if set(names) & family:
                keep_counts.add(count)
        counts = {name: count for name, count in counts.items() if name in keep_counts}
    result = {'values': values, 'sample_counts': counts}
    scope = config.get('history_scope')
    if scope in ('actor_and_binary_market', 'actor_across_all_markets'):
        result['scope'] = ('current_market' if scope == 'actor_and_binary_market' else 'global_wallet')
    if core.get('return_period_seconds') is not None and set(names) & {
            'sharpe_ratio', 'sortino_ratio', 'maximum_drawdown', 'return_volatility'}:
        result['return_period_seconds'] = core['return_period_seconds']
    return result


def enrich_sft(source_dir: Path, dest_dir: Path, index: dict, config: dict) -> dict:
    """Join already-causal features to exact SFT targets, never by nearest timestamp.

    The index is generated from raw histories before the query's executions are
    applied. Current position snapshots and labels are not metric inputs here.
    Original assistant messages and split assignments are preserved verbatim.
    """
    import copy
    import gzip
    import hashlib
    import shutil
    from datetime import datetime, timezone

    splits = ('train', 'validation', 'test')
    require(source_dir.is_dir() and not source_dir.is_symlink(),
            f'SFT source must be a real directory: {source_dir}')
    source_dir = source_dir.resolve()
    destination = dest_dir.resolve()
    require(not dest_dir.exists() and not dest_dir.is_symlink(),
            f'SFT output already exists: {dest_dir}')
    require(destination != source_dir and source_dir not in destination.parents
            and destination not in source_dir.parents,
            'SFT output must be separate from the source dataset')
    source_manifest = source_dir / 'manifest.json'
    require(source_manifest.is_file() and not source_manifest.is_symlink(),
            'SFT source manifest is missing or a symlink')
    source_manifest_digest = sha256(source_manifest)
    manifest = read_json(source_manifest)
    require(isinstance(manifest, dict)
            and manifest.get('format') == 'actor_market_trade_messages_v1',
            'Expected prepared actor_market_trade_messages_v1 SFT dataset')
    require('actor_metrics' not in manifest and 'derived_actor_metrics' not in manifest,
            'SFT source already declares actor metrics')
    require(isinstance(manifest.get('stats'), dict)
            and isinstance(manifest.get('split_sha256'), dict),
            'SFT manifest must declare split counts and hashes')
    enabled = manifest.get('enabled_splits', list(splits))
    require(isinstance(enabled, list) and enabled in (list(splits), list(splits[:2])),
            'SFT enabled_splits must be train/validation or train/validation/test')

    def checked_copy(source: Path, target: Path) -> None:
        require(not source.is_symlink(), f'Symlink in SFT auxiliary files: {source}')
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            for child in sorted(source.iterdir()):
                checked_copy(child, target / child.name)
        else:
            require(source.is_file(), f'Unexpected SFT auxiliary entry: {source}')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            require(sha256(source) == sha256(target), f'Failed SFT auxiliary copy: {source}')

    def source_hash(path: Path) -> str:
        digest = hashlib.sha256()
        opener = gzip.open if path.name.endswith('.gz') else open
        with opener(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    split_paths = {}
    split_digests = {}
    for split in splits:
        candidates = [source_dir / (split + suffix) for suffix in ('.jsonl', '.jsonl.gz')]
        existing = [path for path in candidates if path.exists() or path.is_symlink()]
        require(len(existing) == 1, f'SFT {split} needs exactly one .jsonl or .jsonl.gz file')
        path = existing[0]
        require(path.is_file() and not path.is_symlink(), f'SFT split is not a regular file: {path}')
        split_paths[split] = path
        split_digests[split] = source_hash(path)
        require(manifest['split_sha256'].get(split) == split_digests[split],
                f'SFT {split} hash differs from its manifest')

    counts = {}
    seen_queries = set()
    seen_sequences = set()
    seen_actor_markets = set()
    dest_dir.mkdir(parents=True)
    try:
        for split in splits:
            conversations = targets = executions = 0
            output_path = dest_dir / f'{split}.jsonl'
            with output_path.open('w', encoding='utf-8', newline='\n') as stream:
                for original in iter_jsonl(split_paths[split]):
                    require(isinstance(original, dict), f'{split}: invalid conversation')
                    record = copy.deepcopy(original)
                    interval_protocol = record.get('target_protocol') == 'observed_interval_and_execution_v1'
                    if interval_protocol:
                        # Shared validation checks alternation, equal-time endpoint
                        # pairing, gap continuity and both label counts.
                        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
                        from compare_actor_variants import conversation
                        conversation(record)
                    actor = record.get('actor_id')
                    market = record.get('market_id')
                    require(isinstance(actor, str) and bool(actor)
                            and isinstance(market, str) and bool(market),
                            f'{split}: conversation needs actor_id and market_id strings')
                    actor = actor.lower()
                    actor_market = (actor, market)
                    require(actor_market not in seen_actor_markets,
                            f'Duplicate actor/market conversation across SFT: {actor_market}')
                    seen_actor_markets.add(actor_market)
                    sequence = record.get('sequence_id')
                    require(isinstance(sequence, str) and bool(sequence)
                            and sequence not in seen_sequences,
                            f'{split}: missing or duplicate sequence_id')
                    seen_sequences.add(sequence)
                    messages = record.get('messages')
                    require(isinstance(messages, list) and len(messages) >= 3
                            and len(messages) % 2 == 1,
                            f'{split}: expected system then user/assistant message pairs')
                    require(all(isinstance(message, dict)
                                and isinstance(message.get('content'), str)
                                for message in messages), f'{split}: invalid SFT messages')
                    require(messages[0].get('role') == 'system',
                            f'{split}: first message must be system')
                    previous_query = None
                    target_count = execution_count = 0
                    for offset in range(1, len(messages), 2):
                        user, assistant = messages[offset:offset + 2]
                        require(user.get('role') == 'user' and assistant.get('role') == 'assistant',
                                f'{split}: SFT roles must strictly alternate user/assistant')
                        context = loads(user['content'])
                        label = loads(assistant['content'])
                        require(isinstance(context, dict), f'{split}: user context must be JSON object')
                        require('actor_metrics' not in context,
                                f'{split}: user context already contains actor_metrics')
                        compact_context = manifest.get('prompt_schema_version', 0) >= 2
                        if offset == 1 and not compact_context:
                            require(isinstance(context.get('actor_id'), str)
                                    and context['actor_id'].lower() == actor,
                                    f'{split}: context actor_id differs from conversation')
                            require(isinstance(context.get('market'), dict)
                                    and context['market'].get('market_id') == market,
                                    f'{split}: context market_id differs from conversation')
                        else:
                            require('actor_id' not in context
                                    or (isinstance(context['actor_id'], str)
                                        and context['actor_id'].lower() == actor),
                                    f'{split}: later context changes actor_id')
                            require('market' not in context
                                    or (isinstance(context['market'], dict)
                                        and (context['market'].get('market_id') == market
                                             or (compact_context and 'market_id' not in context['market']))),
                                    f'{split}: later context changes market_id')
                        query = timestamp_us(context.get('query_time'))
                        require(previous_query is None or query > previous_query or (interval_protocol and query == previous_query),
                                f'{split}: query times must strictly increase within an actor')
                        key = (actor, market, query)
                        is_interval = interval_protocol and 'interval' in context
                        unique_key = (*key, is_interval)
                        require(unique_key not in seen_queries, f'Duplicate SFT query: {key}')
                        seen_queries.add(unique_key)
                        require(key in index, f'No exact raw-history metrics match for SFT query: {key}')
                        entry = index[key]
                        require(isinstance(entry, dict) and isinstance(entry.get('actor_metrics'), dict)
                                and isinstance(entry.get('trades'), list) and entry['trades'],
                                f'Invalid derived-metrics index entry: {key}')
                        window = entry['actor_metrics'].get('window')
                        require(isinstance(window, dict)
                                and type(window.get('end_us')) is int
                                and window['end_us'] == query
                                and window.get('end_inclusive') is False,
                                f'Derived metrics do not have a strict prior cutoff at {key}')
                        expected = {'action': 'NO_TRADE'} if is_interval else {'action': 'TRADE', 'trades': entry['trades']}
                        require(label == expected, f'SFT assistant trades differ from the raw source at {key}')
                        core = entry['actor_metrics']
                        require(isinstance(core.get('values'), dict)
                                and isinstance(core.get('sample_counts'), dict),
                                f'Invalid metric values or sample counts at {key}')
                        prompt_metrics = model_metric_fields(core, config)
                        if 'pnl_features' in entry:
                            pnl_fields = pnl_prompt_fields(entry['pnl_features'])
                            if 'unrealized_in_market_pnl' in context:
                                require(all(context.get(k) == v for k, v in pnl_fields.items()),
                                        'SFT P&L differs from raw prior inventory')
                            else:
                                context.update(pnl_fields)
                                user['content'] = json_text(context)
                        # Preserve all existing context number types and literal text.
                        # The validated object is nonempty because query_time exists.
                        content = user['content'].rstrip()
                        require(content.endswith('}'), f'{split}: invalid user JSON object')
                        user['content'] = (content[:-1] + ',\"actor_metrics\":'
                                           + json_text(prompt_metrics) + '}')
                        previous_query = query
                        target_count += 1
                        execution_count += 0 if is_interval else len(entry['trades'])
                    require(type(record.get('target_count')) is int
                            and record['target_count'] == target_count,
                            f'{split}: conversation target_count mismatch')
                    require(type(record.get('execution_count')) is int
                            and record['execution_count'] == execution_count,
                            f'{split}: conversation execution_count mismatch')
                    stream.write(json_text(record) + '\n')
                    conversations += 1
                    targets += target_count
                    executions += execution_count
            counts[split] = {'conversations': conversations, 'targets': targets, 'executions': executions}
            declared = manifest['stats'].get(split)
            require(isinstance(declared, dict), f'SFT manifest lacks {split} stats')
            for name, observed in counts[split].items():
                # Empty disabled test splits are represented by {} in original exports.
                wanted = declared.get(name, 0)
                require(type(wanted) is int and wanted == observed,
                        f'SFT {split} {name} count differs from its manifest')
            require((conversations > 0) if split in enabled else (conversations == 0),
                    f'SFT {split} contents disagree with enabled_splits')
            require(source_hash(split_paths[split]) == split_digests[split],
                    f'SFT {split} changed during enrichment')

        for name in ('source_audit.jsonl', 'split_plan.json'):
            auxiliary = source_dir / name
            require(auxiliary.is_file() and not auxiliary.is_symlink(),
                    f'Missing or unsafe SFT auxiliary file: {auxiliary}')
            checked_copy(auxiliary, dest_dir / name)
        audit = source_dir / 'audit'
        if audit.exists() or audit.is_symlink():
            require(audit.is_dir() and not audit.is_symlink(), 'SFT audit must be a real directory')
            checked_copy(audit, dest_dir / 'audit')
        original_manifest_copy = dest_dir / 'audit' / 'metrics_source_manifest.json'
        require(not original_manifest_copy.exists(),
                'SFT already contains metrics_source_manifest.json; refusing repeated enrichment')
        original_manifest_copy.parent.mkdir(parents=True, exist_ok=True)
        checked_copy(source_manifest, original_manifest_copy)
        require(sha256(original_manifest_copy) == source_manifest_digest,
                'SFT source manifest changed during enrichment')
        output_manifest = copy.deepcopy(manifest)
        output_manifest['created_at'] = datetime.now(timezone.utc).isoformat()
        output_manifest['converter_sha256'] = sha256(Path(__file__))
        output_manifest['feature_variant'] = config.get('feature_variant', 'inmarket')
        output_manifest['token_lengths_checked'] = False
        output_manifest['max_length_checked'] = None
        for stats in output_manifest['stats'].values():
            require(isinstance(stats, dict), 'Invalid SFT split statistics')
            for name in ('tokens', 'max_tokens', 'p50_tokens', 'p95_tokens', 'loss_tokens'):
                stats.pop(name, None)
        output_manifest['split_sha256'] = {split: sha256(dest_dir / f'{split}.jsonl') for split in splits}
        output_manifest['actor_metrics'] = {
            'config': copy.deepcopy(config),
            'strict_prior': True,
            'source_dataset_manifest_sha256': sha256(original_manifest_copy),
            'source_split_sha256': split_digests,
            'join': 'exact_actor_market_query_time_and_matching_execution_labels',
            'history_scope': config.get('history_scope', 'actor_and_binary_market'),
            'actor_snapshots_used_as_model_input': False,
            'features_added_only_to_user_messages': True,
            'prompt_fields': (['values', 'sample_counts', 'scope', 'return_period_seconds_when_available']
                              if config.get('history_scope') in ('actor_and_binary_market', 'actor_across_all_markets')
                              else ['values', 'sample_counts']),
            'target_messages_unchanged': True,
            'unavailable_values': 'omitted_from_prompt_not_zero_retained_in_raw_audit',
        }
        write_json(dest_dir / 'manifest.json', output_manifest)
        return {'output': 'sft', 'splits': counts, 'token_lengths_checked': False,
                'strict_prior': True, 'source_dataset_manifest_sha256': sha256(original_manifest_copy)}
    except BaseException:
        shutil.rmtree(dest_dir)
        raise


def derive(args):
    require(args.min_return_periods >= 2, '--min-return-periods must be at least 2')
    require(args.lookback_seconds is None or args.lookback_seconds > 0, '--lookback-seconds must be positive')
    selected_features = select_features(getattr(args, 'features', None))
    significant_digits = getattr(args, 'metric_significant_digits', 10)
    require(type(significant_digits) is int and 4 <= significant_digits <= 16,
            '--metric-significant-digits must be between 4 and 16')
    sources = discover_exports(args.exports, args.input_root)
    output = args.out.resolve()
    require(not output.exists(), f'Output already exists: {output}. Choose a new --out.')
    for source in sources:
        path = source['path']
        require(output != path and not output.is_relative_to(path) and not path.is_relative_to(output),
                'Output must be separate from every actor export')
    if args.sft_dir:
        sft_path = args.sft_dir.resolve()
        require(output != sft_path and not output.is_relative_to(sft_path) and not sft_path.is_relative_to(output),
                'Output must be separate from the source SFT dataset')
    closed, returns = load_closed_positions(args.closed_positions), load_returns(args.returns_file)
    config = {'version': 2, 'strict_prior': True, 'history_scope': 'actor_and_binary_market',
              'feature_variant': 'inmarket', 'selected_features': selected_features,
              'metric_significant_digits': significant_digits,
              'lookback_seconds': args.lookback_seconds, 'min_return_periods': args.min_return_periods,
              'completed_position_ledger_supplied': args.closed_positions is not None,
              'capital_adjusted_returns_supplied': args.returns_file is not None,
              'risk_ratios_annualized': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='actor-metrics-', dir=output.parent))
    index, processed, reports = {}, set(), []
    total_counts = Counter()
    try:
        with (work / 'actor_index.jsonl').open('w', encoding='utf-8') as inventory:
            for source in sources:
                counts = Counter()
                actor_ids = set()
                source_inventory = hashlib.sha256()
                for path in sorted((source['path'] / 'actors').iterdir()):
                    require(path.is_file() and not path.is_symlink() and
                            (path.name.endswith('.jsonl') or path.name.endswith('.jsonl.gz')),
                            f'Unexpected actor file: {path}')
                    actor, groups, actor_counts = actor_trade_groups(path, source)
                    require(actor not in actor_ids, f'Duplicate actor file: {actor}')
                    actor_ids.add(actor)
                    key = (actor, source['condition_id'])
                    processed.add(key)
                    history = []
                    relative = f'markets/{source["condition_id"]}/actors/{actor}.jsonl'
                    destination = work / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('w', encoding='utf-8') as stream:
                        for group in groups:
                            metrics = compute_metrics(history, closed.get(key, []), returns.get(key, []),
                                group['time_us'], args.lookback_seconds, args.min_return_periods)
                            record = {'actor_id': actor, 'market_id': source['market_id'],
                                'condition_id': source['condition_id'], 'timestamp': group['timestamp'],
                                'source_trade_row_index': group['row_index'], 'actor_metrics': metrics,
                                **group['pnl_features']}
                            stream.write(json_text(record) + '\n')
                            if args.sft_dir:
                                index[(actor, source['market_id'], group['time_us'])] = {
                                    'actor_metrics': metrics, 'trades': group['expected'],
                                    'pnl_features': group['pnl_features']}
                            # Update only after producing this whole timestamp group's features.
                            history.extend(group['trades'])
                    file_hash = sha256(path)
                    source_inventory.update(json_text([path.name, file_hash]).encode('utf-8') + b'\n')
                    inventory.write(json_text({'actor_id': actor, 'market_id': source['market_id'],
                        'condition_id': source['condition_id'], 'path': relative, 'rows': len(groups),
                        'source_actor_sha256': file_hash, 'sha256': sha256(destination)}) + '\n')
                    counts.update(actor_counts)
                    if counts['actors'] % 1000 == 0:
                        print(f'Market {source["market_id"]}: derived {counts["actors"]:,} actors', flush=True)
                expected = source['manifest'].get('counts', {})
                require(counts['actors'] > 0, 'No actor files found')
                for name, value in counts.items():
                    require(type(expected.get(name)) is int and expected[name] == value,
                            f'{source["path"]}: {name} count differs from manifest')
                reports.append({'path': str(source['path']), 'market_id': source['market_id'],
                    'condition_id': source['condition_id'], 'counts': dict(counts),
                    'manifest_sha256': sha256(source['path'] / 'manifest.json'),
                    'market_sha256': sha256(source['path'] / 'market.json'),
                    'actor_inventory_sha256': source_inventory.hexdigest(),
                    'source_trade_coverage': source['manifest'].get('source')})
                total_counts.update(counts)
        for supplied, groups, name in ((args.closed_positions, closed, 'completed positions'),
                                       (args.returns_file, returns, 'returns')):
            require(supplied is None or not groups or any(key in processed for key in groups),
                    f'Supplied {name} do not match any exported actor/condition')
        supplement_sources = {}
        for name, path, groups in (('closed_positions', args.closed_positions, closed),
                                   ('returns', args.returns_file, returns)):
            if path:
                supplement_sources[name] = {'path': str(path.resolve()), 'sha256': sha256(path),
                    'ignored_out_of_scope_rows': sum(len(rows) for key, rows in groups.items() if key not in processed)}
        metadata = {'format': 'actor_prior_metrics_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
            'config': config, 'metric_names': list(ACTOR_METRIC_NAMES), 'sources': reports,
            'supplemental_sources': supplement_sources, 'counts': dict(total_counts),
            'feature_rows': total_counts['distinct_trade_times'], 'script_sha256': sha256(__file__),
            'current_actor_snapshots_used': False, 'current_execution_used': False,
            'source_timestamp_semantics': 'strictly before recorded execution timestamp; not verified order-submission time',
            'source_capture_complete': False, 'supplemental_accounting_and_availability': 'caller_supplied_not_independently_verified'}
        if args.sft_dir:
            metadata['sft'] = enrich_sft(args.sft_dir.resolve(), work / 'sft', index,
                {**config, 'provenance': {'raw_sources': reports, 'supplemental_sources': supplement_sources}})
        write_json(work / 'manifest.json', metadata)
        require(not output.exists(), 'Output appeared during processing')
        work.rename(output)
    finally:
        if work.exists():
            shutil.rmtree(work)
    print(json_text({'output': str(output), 'actors': total_counts['actors'],
                     'feature_rows': total_counts['distinct_trade_times'],
                     'sft_dataset': str(output / 'sft') if args.sft_dir else None}), flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exports', nargs='*', type=Path, help='Completed actor export directories')
    parser.add_argument('--input-root', type=Path, help='An actor export or parent containing completed exports')
    parser.add_argument('--out', type=Path, required=True, help='New metrics output directory')
    parser.add_argument('--closed-positions', type=Path, help='Historical completed-position JSONL ledger, optionally gzip')
    parser.add_argument('--returns-file', type=Path, help='Regular capital-adjusted return JSONL ledger, optionally gzip')
    parser.add_argument('--lookback-seconds', type=int, help='Use only this trailing window; default all prior supplied history')
    parser.add_argument('--min-return-periods', type=int, default=30, help='Minimum periods for Sharpe/Sortino/volatility (default: 30)')
    parser.add_argument('--sft-dir', type=Path, help='Prepared SFT dataset to enrich into OUT/sft, preserving splits and targets')
    parser.add_argument('--features', help='Comma-separated model feature subset; raw audit keeps all 18 metrics')
    parser.add_argument('--metric-significant-digits', type=int, default=10,
                        help='Significant digits for derived prompt metrics only, 4..16 (default: 10)')
    args = parser.parse_args(argv)
    try:
        return derive(args)
    except (ValueError, OSError, KeyError, TypeError, InvalidOperation) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    if sys.version_info < (3, 11):
        raise SystemExit('Python 3.11 or newer is required')
    main()
