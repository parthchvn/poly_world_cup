#!/usr/bin/env python3
"""Add prior wallet-wide metrics to the same actor decisions used by the baseline.

python3 scripts/derive_global_actor_metrics.py data/market_1897059 \
    --out data/market_1897059_global --http-transport curl
python3 scripts/derive_global_actor_metrics.py --input-root data \
    --wallet-trades wallet_trades.jsonl --out data/global_metrics \
    --sft-dir datasets/world_cup_sft

Without --wallet-trades, cache wallet executions across all API-served markets.
Historical wallet equity returns and completed-position accounting are optional
separate inputs. They are never reconstructed from today's portfolio snapshots.
All metrics precede the query strictly, including their known_at when supplied.
API executions use an explicitly documented execution-time availability proxy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import InvalidOperation
import hashlib
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

# Sibling scripts are shared implementations, even when imported by test tools.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import derive_actor_metrics as base


def actor_identity(value):
    base.require(isinstance(value, str) and base.ADDRESS.fullmatch(value), 'Invalid actor_id')
    return value.lower()


def normalize_wallet_row(row, *, api_proxy=False):
    """Validate an execution without equating collection time with event time.

Offline histories must provide real availability timestamps. Only the explicit
API adapter may use execution time as an unverified availability proxy.
"""
    actor, condition = base.identity(row)
    execution = row.get('execution_id')
    base.require(isinstance(execution, str) and execution.strip(), 'Missing execution_id')
    stamp = base.timestamp_us(row.get('timestamp'))
    if api_proxy:
        base.require(row.get('availability_semantics') ==
                     'execution_timestamp_proxy_not_verified_publication_time',
                     'API wallet history must declare its execution-time availability proxy')
        base.require('known_at' not in row, 'API proxy rows must not manufacture known_at')
        known = stamp
    else:
        known = base.timestamp_us(row.get('known_at'))
        base.require(stamp <= known, 'Wallet execution requires timestamp <= known_at')
    base.require(row.get('side') in ('BUY', 'SELL'), 'Wallet execution side must be BUY or SELL')
    shares, price = base.number(row.get('shares'), 'shares'), base.number(row.get('price'), 'price')
    base.require(shares > 0 and 0 <= price <= 1, 'Wallet execution shares/price out of range')
    return actor, {'condition_id': condition, 'execution_id': execution,
                   'time_us': stamp, 'known_us': known, 'side': row['side'],
                   'shares': shares, 'price': price}


def load_wallet_trades(path):
    """Small-file utility; the CLI stages offline histories on disk by actor."""
    groups, seen = defaultdict(list), set()
    for row in base.iter_jsonl(path):
        actor, execution = normalize_wallet_row(row)
        key = (actor, execution['execution_id'])
        base.require(key not in seen, f'Duplicate wallet execution: {key}')
        seen.add(key)
        groups[actor].append(execution)
    return groups


def load_wallet_returns(path):
    """Accept wallet-equity returns only; never concatenate market return series."""
    groups, seen = defaultdict(list), set()
    if path is None:
        return groups
    for row in base.iter_jsonl(path):
        actor = actor_identity(row.get('actor_id'))
        base.require(row.get('scope') == 'wallet' and 'condition_id' not in row
                     and 'market_id' not in row,
                     'Global returns require scope="wallet" and no market_id/condition_id')
        start, end, known = (base.timestamp_us(row.get(name))
                             for name in ('period_start', 'period_end', 'known_at'))
        base.require(start < end <= known, 'Require period_start < period_end <= known_at')
        base.require((actor, start) not in seen, 'Duplicate wallet return period')
        seen.add((actor, start))
        base.require(row.get('capital_flow_adjusted') is True,
                     'Wallet returns must declare capital_flow_adjusted: true')
        value = base.number(row.get('period_return'), 'period_return')
        base.require(value >= -1, 'An unlevered wallet capital return cannot be less than -1')
        groups[actor].append({'start_us': start, 'end_us': end, 'known_us': known,
            'return': value,
            'benchmark_return': base.number(row.get('benchmark_return', '0'), 'benchmark_return'),
            'target_return': base.number(row.get('target_return', '0'), 'target_return')})
    for actor, rows in groups.items():
        rows.sort(key=lambda row: row['start_us'])
        for previous, current in zip(rows, rows[1:]):
            base.require(previous['end_us'] <= current['start_us'], f'Overlapping wallet returns: {actor}')
            base.require(previous['return'] != -1,
                         'Cannot extend a wallet return series after its capital reaches zero')
    return groups


def load_global_closed_positions(path):
    result = defaultdict(list)
    for (actor, condition), rows in base.load_closed_positions(path).items():
        result[actor].extend({**row, 'condition_id': condition} for row in rows)
    return result


def compute_global_metrics(trades, closed, returns, query_us, lookback_seconds=None,
                           min_return_periods=30):
    # Shared arithmetic already checks strict event/known cutoffs for financial
    # rows. Wallet fills additionally carry known_us, unlike raw actor exports.
    eligible = [row for row in trades if row['known_us'] < query_us]
    metrics = base.compute_metrics(eligible, closed, returns, query_us,
                                   lookback_seconds, min_return_periods)
    metrics['metric_scope'] = {
        'execution_metrics': 'captured_actor_wallet_executions_across_markets',
        'performance_metrics': 'completed_positions_across_supplied_markets_only',
        'risk_metrics': 'caller_supplied_capital_adjusted_whole_wallet_returns',
    }
    return metrics


def collect_targets(sources):
    """Read the unchanged actor cohort and retain exact target labels for joins."""
    targets, reports = defaultdict(list), []
    total_counts = Counter()
    for source in sources:
        counts, actors = Counter(), set()
        inventory = hashlib.sha256()
        for path in sorted((source['path'] / 'actors').iterdir()):
            base.require(path.is_file() and not path.is_symlink() and
                         (path.name.endswith('.jsonl') or path.name.endswith('.jsonl.gz')),
                         f'Unexpected actor file: {path}')
            actor, groups, actor_counts = base.actor_trade_groups(path, source)
            base.require(actor not in actors, f'Duplicate actor file: {actor}')
            actors.add(actor)
            digest = base.sha256(path)
            inventory.update(base.json_text([path.name, digest]).encode('utf-8') + b'\n')
            targets[actor].append({'source': source, 'groups': groups,
                                   'source_actor_sha256': digest})
            counts.update(actor_counts)
        expected = source['manifest'].get('counts', {})
        base.require(counts['actors'] > 0, 'No actor files found')
        for name, value in counts.items():
            base.require(type(expected.get(name)) is int and expected[name] == value,
                         f'{source["path"]}: {name} count differs from manifest')
        reports.append({'path': str(source['path']), 'market_id': source['market_id'],
            'condition_id': source['condition_id'], 'counts': dict(counts),
            'manifest_sha256': base.sha256(source['path'] / 'manifest.json'),
            'market_sha256': base.sha256(source['path'] / 'market.json'),
            'actor_inventory_sha256': inventory.hexdigest(),
            'source_trade_coverage': source['manifest'].get('source')})
        total_counts.update(counts)
    return targets, reports, total_counts


def stage_wallet_trades(path, database_path, actors):
    """One file scan with disk staging, rather than all wallets in RAM at once."""
    digest = base.sha256(path)
    connection = sqlite3.connect(database_path)
    connection.execute('CREATE TABLE executions (actor TEXT, execution_id TEXT, condition_id TEXT, '
                       'time_us INTEGER, known_us INTEGER, side TEXT, shares TEXT, price TEXT, '
                       'PRIMARY KEY (actor, execution_id))')
    seen_actors = set()
    count = ignored = 0
    try:
        for raw in base.iter_jsonl(path):
            actor, row = normalize_wallet_row(raw)
            if actor not in actors:
                ignored += 1
                continue
            seen_actors.add(actor)
            try:
                connection.execute('INSERT INTO executions VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (actor, row['execution_id'], row['condition_id'], row['time_us'], row['known_us'],
                     row['side'], str(row['shares']), str(row['price'])))
            except sqlite3.IntegrityError as error:
                raise ValueError(f'Duplicate wallet execution for {actor}: {row["execution_id"]}') from error
            count += 1
        base.require(seen_actors == set(actors),
                     'Wallet history has no executions for target actors: ' +
                     ', '.join(sorted(set(actors) - seen_actors)[:10]))
        base.require(base.sha256(path) == digest, 'Wallet history changed during processing')
        connection.commit()
    finally:
        connection.close()
    return {'path': str(Path(path).resolve()), 'sha256': digest,
            'captured_executions': count, 'ignored_non_target_actor_rows': ignored,
            'coverage': 'caller_supplied_history_completeness_unverified',
            'availability_semantics': 'caller_supplied_known_at_not_independently_verified',
            'source_capture_complete': False}


def staged_actor_trades(database_path, actor):
    connection = sqlite3.connect(database_path)
    try:
        return [{'execution_id': identifier, 'condition_id': condition,
                 'time_us': stamp, 'known_us': known, 'side': side,
                 'shares': base.number(shares, 'shares'), 'price': base.number(price, 'price')}
                for identifier, condition, stamp, known, side, shares, price in
                connection.execute('SELECT execution_id, condition_id, time_us, known_us, side, shares, price '
                                   'FROM executions WHERE actor = ? ORDER BY time_us, execution_id', (actor,))]
    finally:
        connection.close()


def fetch_actor_history(args, actor, max_query_us, client):
    tool_dir = SCRIPT_DIR.parent / 'tools'
    if str(tool_dir) not in sys.path:
        sys.path.insert(0, str(tool_dir))
    import wallet_history
    manifest = wallet_history.ingest_wallet(client, actor_id=actor,
        output_dir=args.cache.resolve() / 'wallet_histories',
        end_seconds=max_query_us // 1_000_000, start_seconds=1,
        max_pages=args.wallet_max_pages, progress=True)
    capture = Path(manifest['capture_dir'])
    trades, seen = [], set()
    for row in wallet_history.iter_wallet_observations(capture):
        returned_actor, normalized = normalize_wallet_row(row, api_proxy=True)
        base.require(returned_actor == actor, 'Wallet capture returned another actor')
        base.require(normalized['execution_id'] not in seen, 'Duplicate execution in wallet capture')
        seen.add(normalized['execution_id'])
        trades.append(normalized)
    base.require(trades, f'No API wallet executions captured for target actor {actor}')
    return trades, {'actor_id': actor, 'capture_dir': str(capture),
                    'manifest_sha256': base.sha256(capture / 'manifest.json'),
                    'captured_executions': len(trades),
                    'coverage': 'all_markets_served_by_wallet_trade_endpoint_in_requested_window',
                    'availability_semantics': 'execution_timestamp_proxy_not_verified_publication_time',
                    'source_capture_complete': False}


def derive(args):
    base.require(args.min_return_periods >= 2, '--min-return-periods must be at least 2')
    base.require(args.lookback_seconds is None or args.lookback_seconds > 0,
                 '--lookback-seconds must be positive')
    base.require(args.wallet_max_pages is None or args.wallet_max_pages > 0,
                 '--wallet-max-pages must be positive')
    selected = base.select_features(args.features)
    base.require(type(args.metric_significant_digits) is int and 4 <= args.metric_significant_digits <= 16,
                 '--metric-significant-digits must be between 4 and 16')
    sources = base.discover_exports(args.exports, args.input_root)
    output = args.out.resolve()
    base.require(not output.exists(), f'Output already exists: {output}. Choose a new --out.')
    inputs = [source['path'] for source in sources]
    if args.sft_dir:
        inputs.append(args.sft_dir.resolve())
    for path in inputs:
        base.require(output != path and not output.is_relative_to(path) and not path.is_relative_to(output),
                     'Output must be separate from source actor exports and SFT dataset')
    split_validation = None
    if args.sft_dir:
        tool_dir = SCRIPT_DIR.parent / 'tools'
        if str(tool_dir) not in sys.path:
            sys.path.insert(0, str(tool_dir))
        from compare_actor_variants import validate_chronological_splits
        # Run before any network capture. A future training query must not use
        # earlier held-out target actions from another market in wallet history.
        split_validation = validate_chronological_splits(args.sft_dir.resolve())
    if not args.wallet_trades:
        cache = args.cache.resolve()
        base.require(not cache.is_relative_to(output) and not output.is_relative_to(cache),
                     'Global capture cache and output must be separate')
    targets, reports, counts = collect_targets(sources)
    closed = load_global_closed_positions(args.closed_positions)
    returns = load_wallet_returns(args.returns_file)
    for supplied, groups, name in ((args.closed_positions, closed, 'completed positions'),
                                   (args.returns_file, returns, 'wallet returns')):
        base.require(supplied is None or not groups or any(actor in targets for actor in groups),
                     f'Supplied {name} do not match any exported actor')
    supplements = {}
    for name, path, groups in (('closed_positions', args.closed_positions, closed),
                              ('returns', args.returns_file, returns)):
        if path:
            supplements[name] = {'path': str(path.resolve()), 'sha256': base.sha256(path),
                'ignored_non_target_actor_rows': sum(len(rows) for actor, rows in groups.items()
                                                     if actor not in targets)}
    availability = ('caller_supplied_known_at_not_independently_verified' if args.wallet_trades else
                    'execution_timestamp_proxy_not_verified_publication_time')
    config = {'version': 2, 'strict_prior': True, 'history_scope': 'actor_across_all_markets',
              'feature_variant': 'global', 'selected_features': selected,
              'metric_significant_digits': args.metric_significant_digits,
              'lookback_seconds': args.lookback_seconds, 'min_return_periods': args.min_return_periods,
              'completed_position_ledger_supplied': args.closed_positions is not None,
              'capital_adjusted_returns_supplied': args.returns_file is not None,
              'return_scope': 'whole_wallet_equity', 'risk_ratios_annualized': False,
              'execution_availability_semantics': availability,
              'converter_sha256': base.sha256(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='global-actor-metrics-', dir=output.parent))
    index, captures = {}, []
    database = work / 'wallet_history.sqlite'
    try:
        if args.wallet_trades:
            supplements['wallet_trades'] = stage_wallet_trades(args.wallet_trades, database, targets)
            client = None
        else:
            import build_actor_dataset as builder
            client = builder.HttpClient(args.cache.resolve() / 'http', compress=True,
                transport=args.http_transport, timeout=args.http_timeout, retries=args.http_retries,
                retry_delay=args.http_retry_delay, min_interval=args.http_min_interval, log_retries=True)
        with (work / 'actor_index.jsonl').open('w', encoding='utf-8') as inventory:
            for actor_number, actor in enumerate(sorted(targets), 1):
                queries = targets[actor]
                if args.wallet_trades:
                    trades = staged_actor_trades(database, actor)
                else:
                    max_query = max(group['time_us'] for target in queries for group in target['groups'])
                    trades, capture = fetch_actor_history(args, actor, max_query, client)
                    captures.append(capture)
                for target in queries:
                    source, groups = target['source'], target['groups']
                    relative = f'markets/{source["condition_id"]}/actors/{actor}.jsonl'
                    destination = work / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('w', encoding='utf-8') as stream:
                        for group in groups:
                            metrics = compute_global_metrics(trades, closed.get(actor, []), returns.get(actor, []),
                                group['time_us'], args.lookback_seconds, args.min_return_periods)
                            record = {'actor_id': actor, 'market_id': source['market_id'],
                                'condition_id': source['condition_id'], 'timestamp': group['timestamp'],
                                'source_trade_row_index': group['row_index'], 'actor_metrics': metrics}
                            stream.write(base.json_text(record) + '\n')
                            if args.sft_dir:
                                index[(actor, source['market_id'], group['time_us'])] = {
                                    'actor_metrics': metrics, 'trades': group['expected']}
                    inventory.write(base.json_text({'actor_id': actor, 'market_id': source['market_id'],
                        'condition_id': source['condition_id'], 'path': relative, 'rows': len(groups),
                        'source_actor_sha256': target['source_actor_sha256'],
                        'sha256': base.sha256(destination)}) + '\n')
                if actor_number % 100 == 0 or actor_number == len(targets):
                    print(f'Global metrics: {actor_number:,}/{len(targets):,} wallets', flush=True)
        if database.exists():
            database.unlink()
        metadata = {'format': 'actor_prior_metrics_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
            'config': config, 'metric_names': list(base.ACTOR_METRIC_NAMES), 'sources': reports,
            'supplemental_sources': supplements, 'wallet_captures': captures,
            'counts': dict(counts), 'unique_wallets': len(targets),
            'feature_rows': counts['distinct_trade_times'], 'script_sha256': base.sha256(__file__),
            'shared_metrics_sha256': base.sha256(base.__file__),
            'current_actor_snapshots_used': False, 'current_execution_used': False,
            'split_temporal_validation': split_validation,
            'source_timestamp_semantics': 'strictly before recorded execution timestamp; not verified order-submission time',
            'source_capture_complete': False,
            'supplemental_accounting_and_availability': 'caller_supplied_not_independently_verified'}
        if args.sft_dir:
            metadata['sft'] = base.enrich_sft(args.sft_dir.resolve(), work / 'sft', index,
                {**config, 'provenance': {'raw_sources': reports, 'supplemental_sources': supplements,
                                        'wallet_captures': captures}})
        base.write_json(work / 'manifest.json', metadata)
        base.require(not output.exists(), 'Output appeared during processing')
        work.rename(output)
    finally:
        if work.exists():
            shutil.rmtree(work)
    print(base.json_text({'output': str(output), 'actors': counts['actors'],
                         'unique_wallets': len(targets), 'feature_rows': counts['distinct_trade_times'],
                         'sft_dataset': str(output / 'sft') if args.sft_dir else None}), flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exports', nargs='*', type=Path, help='Completed actor export directories')
    parser.add_argument('--input-root', type=Path, help='An actor export or parent containing completed exports')
    parser.add_argument('--out', type=Path, required=True, help='New metrics output directory')
    parser.add_argument('--wallet-trades', type=Path,
                        help='Offline wallet execution JSONL(.gz), including known_at; default capture all-markets wallet API')
    parser.add_argument('--closed-positions', type=Path,
                        help='Historical completed positions across markets, JSONL(.gz) with condition_id')
    parser.add_argument('--returns-file', type=Path,
                        help='Whole-wallet capital-adjusted returns, JSONL(.gz) with scope=wallet and no condition_id')
    parser.add_argument('--lookback-seconds', type=int, help='Trailing window; default all prior supplied history')
    parser.add_argument('--min-return-periods', type=int, default=30)
    parser.add_argument('--sft-dir', type=Path, help='Prepared SFT dataset to enrich into OUT/sft, preserving targets/splits')
    parser.add_argument('--features', help='Comma-separated metric subset for model input; default all 18')
    parser.add_argument('--metric-significant-digits', type=int, default=10,
                        help='Significant digits for derived model features only (4..16, default 10)')
    parser.add_argument('--cache', type=Path, default=Path('data/market_actor_cache'), help='Resumable API history cache')
    parser.add_argument('--wallet-max-pages', type=int,
                        help='Stop at this page budget; partial captures are not exported as completed histories')
    parser.add_argument('--http-transport', choices=('urllib', 'curl'), default='urllib')
    parser.add_argument('--http-timeout', type=float, default=45)
    parser.add_argument('--http-retries', type=int, default=8)
    parser.add_argument('--http-retry-delay', type=float, default=2)
    parser.add_argument('--http-min-interval', type=float, default=0.1)
    args = parser.parse_args(argv)
    try:
        return derive(args)
    except (ValueError, OSError, KeyError, TypeError, InvalidOperation, sqlite3.Error) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    if sys.version_info < (3, 11):
        raise SystemExit('Python 3.11 or newer is required')
    main()
