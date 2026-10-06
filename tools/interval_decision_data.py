"""Causal, scheduled World Cup interval examples for SFT and XGBoost.

Only saved actor exports, price histories and ESPN events are read. No current
wallet snapshot, API call, future-defined trade gap, or class balancing is used.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
import copy
import hashlib
import heapq
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'scripts', ROOT / 'tools'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import actor_activity_common as activity
import build_actor_dataset as builder
import derive_actor_metrics as metrics

PROTOCOL = 'prospective_interval_activity_v1'
TRADE_DETAILS_PROTOCOL = 'prospective_interval_trade_details_v1'
TARGET_PROTOCOLS = {'activity': PROTOCOL, 'trade-details': TRADE_DETAILS_PROTOCOL}
SPLITS = ('train', 'validation', 'test')
SYSTEM = (
    'Predict whether this actor has at least one captured execution in the future '
    'interval [start,end), including start and excluding end. All observations supplied '
    'as context strictly precede start. Reply only with a JSON object: '
    '{"action":"TRADE"} or {"action":"NO_TRADE"}. '
    'Prior history and derived features concern this actor in this binary market only. '
    'Unrealized PnL values remaining captured inventory at earlier market prices, before fees; '
    'it is not accumulated historical PnL snapshots. Null means unavailable. '
    'Historical prices are sampled observations, not executable quotes. News timestamps '
    'are event-time proxies, not verified publication times. NO_TRADE means no captured '
    'execution; it does not identify unfilled orders or conscious intent.'
)

TRADE_DETAILS_SYSTEM = SYSTEM.replace(
    '{"action":"TRADE"} or {"action":"NO_TRADE"}. ',
    '{"action":"NO_TRADE"} when no execution occurs. Otherwise reply '
    '{"action":"TRADE","trades":[{"side":"BUY","outcome":"Yes",'
    '"price":"0.40","shares":"10"}]}. Predict the execution totals in the interval '
    'using trades entries with side BUY or SELL, outcome Yes or No, '
    'price in [0,1], and positive shares. Keep price and shares as decimal strings. '
    'For each side/outcome pair, combine every captured fill in the interval into one entry: '
    'shares is the total number of shares and price is their share-weighted mean execution price. '
    'Return at most four entries, sorted by side then outcome; do not predict timestamps. '
    'The example numbers describe only the output schema, not the present label. '
)


def interval_target(groups, query, horizon, target_mode='activity'):
    """Side/outcome aggregates over [query,query+horizon); tolerances are evaluation-only.

    Every captured fill contributes its shares and execution notional. The raw
    actor export is untouched; aggregate price uses the shared Decimal helper.
    """
    metrics.require(target_mode in TARGET_PROTOCOLS, 'Unknown target mode')
    times = [group['time_us'] for group in groups]
    selected = groups[bisect_left(times, query):bisect_left(times, query + horizon)]
    target = {'action': 'TRADE' if selected else 'NO_TRADE'}
    if selected and target_mode == 'trade-details':
        from interval_trade_tolerances import aggregate_trades
        target['trades'] = aggregate_trades([trade for group in selected for trade in group['expected']])
    return target


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True, allow_nan=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def number(value):
    if value is None:
        return None
    result = float(value)
    metrics.require(math.isfinite(result), 'Nonfinite derived feature')
    return round(result, 10)


def _price_at(timeline, outcome, query, max_age):
    index = bisect_left(timeline['price_times'][outcome], query) - 1
    if index < 0:
        return None
    instant, price = timeline['prices'][outcome][index]
    if query - instant > max_age * 1_000_000:
        return None
    return {'price': price, 'age_seconds': str((query - instant) / 1_000_000)}


def _kind_matches(kind, category):
    kind = str(kind or '').lower()
    if category == 'goal':
        return kind == 'goal' or kind.startswith('goal---') or kind.startswith('own-goal') or kind == 'penalty---scored'
    terms = {'card': ('card',), 'shot': ('shot',), 'substitution': ('substitution',),
             'foul': ('foul',), 'corner': ('corner',), 'delay': ('delay',),
             'phase': ('kickoff', 'half', 'end-regular', 'full-time')}
    return any(term in kind for term in terms[category])


def example_context(source, timeline, groups, query, horizon, *, history_groups=8,
                    news_seconds=1200, max_news_items=20, max_news_chars=300,
                    closed=(), returns=(), min_return_periods=30, actor_id=None):
    """Build both representations from a common strictly prior prefix.

    Current/future raw interval context is deliberately ignored. Full prefix
    statistics remain available when only the last history_groups are rendered.
    """
    end = bisect_left([group['time_us'] for group in groups], query)
    prior = groups[:end]
    metrics.require(prior, 'Actor must have a strictly prior execution before enrollment')
    history = [trade for group in prior for trade in group['trades']]
    expected = [trade for group in prior for trade in group['expected']]
    max_age = source['manifest']['market_price_max_age_seconds']
    marks = {o: _price_at(timeline, o, query, max_age) for o in ('yes', 'no')}
    pnl = metrics.pnl_state_for_export(source['manifest'], {
        'interval': {'start': source['manifest'].get('origin_utc')},
        'in_market_pnl_opening_history': groups[0].get('pnl_opening_history', [])})
    for group in prior:
        pnl.apply(group['expected'])
    snapshot = pnl.snapshot(marks, activity.utc(query))
    core = metrics.compute_metrics(history, list(closed), list(returns), query,
                                   min_return_periods=min_return_periods)
    features = {key: number(value) for key, value in core['values'].items()}
    features.update({
        'elapsed_since_kickoff_seconds': (query - source['kickoff']) / 1_000_000,
        'prediction_window_seconds': horizon / 1_000_000,
        'prior_execution_count': len(history), 'prior_execution_groups': len(prior),
        'seconds_since_first_execution': (query - prior[0]['time_us']) / 1_000_000,
        'seconds_since_last_execution': (query - prior[-1]['time_us']) / 1_000_000,
        'unrealized_in_market_pnl': number(snapshot['unrealized_in_market_pnl']),
        'unrealized_in_market_pnl_missing': int(snapshot['unrealized_in_market_pnl'] is None),
        'eligible_completed_positions': core['sample_counts']['completed_positions'],
        'eligible_return_periods': core['sample_counts']['eligible_return_periods'],
    })
    notionals = [float(t['shares'] * t['price']) for t in history]
    buys = [t for t in expected if t['side'] == 'BUY']
    total_notional = sum(notionals)
    features.update({
        'prior_total_execution_notional': number(total_notional),
        'prior_max_execution_notional': number(max(notionals)),
        'prior_buy_execution_fraction': len(buys) / len(expected),
        'prior_yes_execution_fraction': sum(t['outcome'] == 'Yes' for t in expected) / len(expected),
        'prior_net_cashflow_before_fees': number(sum(
            float(t['shares']) * float(t['price']) * (1 if t['side'] == 'SELL' else -1)
            for t in expected)),
        'last_group_execution_count': len(prior[-1]['expected']),
        'last_group_buy_fraction': sum(t['side'] == 'BUY' for t in prior[-1]['expected']) / len(prior[-1]['expected']),
        'last_group_yes_fraction': sum(t['outcome'] == 'Yes' for t in prior[-1]['expected']) / len(prior[-1]['expected']),
        'last_group_total_notional': number(sum(float(t['shares'] * t['price']) for t in prior[-1]['trades'])),
    })
    gaps = [(b['time_us'] - a['time_us']) / 1_000_000 for a, b in zip(prior, prior[1:])]
    features['mean_prior_interexecution_seconds'] = number(statistics.mean(gaps)) if gaps else None
    features['prior_interexecution_cv'] = number(statistics.pstdev(gaps) / statistics.mean(gaps)) if len(gaps) >= 2 else None
    for seconds in (300, 900, 3600):
        features[f'prior_execution_count_{seconds}s'] = sum(t['time_us'] >= query - seconds * 1_000_000 for t in history)
    holdings = snapshot['in_market_pnl_context']['holdings']
    for outcome in ('yes', 'no'):
        features[f'{outcome}_held_shares'] = number(holdings[outcome]['shares'])
        features[f'{outcome}_remaining_cost_basis'] = number(holdings[outcome]['remaining_cost_basis'])
        features[f'{outcome}_unrealized_pnl'] = number(holdings[outcome]['unrealized_pnl'])
        features[f'{outcome}_price'] = number(marks[outcome]['price']) if marks[outcome] else None
        features[f'{outcome}_price_age_seconds'] = number(marks[outcome]['age_seconds']) if marks[outcome] else None
        features[f'{outcome}_price_missing'] = int(marks[outcome] is None)
        for seconds in (300, 900):
            earlier = _price_at(timeline, outcome, query - seconds * 1_000_000, max_age)
            features[f'{outcome}_price_change_{seconds}s'] = (
                number(float(marks[outcome]['price']) - float(earlier['price']))
                if marks[outcome] and earlier else None)
        left = bisect_left(timeline['price_times'][outcome], query - 900_000_000)
        right = bisect_left(timeline['price_times'][outcome], query)
        prices = [float(row[1]) for row in timeline['prices'][outcome][left:right]]
        changes = [b-a for a, b in zip(prices, prices[1:])]
        features[f'{outcome}_price_sample_change_std_900s'] = number(statistics.stdev(changes)) if len(changes) >= 2 else None
    costs = [holdings[o]['remaining_cost_basis'] for o in ('yes', 'no')]
    basis = sum(float(c) for c in costs) if all(c is not None for c in costs) else None
    features['unrealized_return_on_remaining_cost'] = (
        number(float(snapshot['unrealized_in_market_pnl']) / basis)
        if basis and snapshot['unrealized_in_market_pnl'] is not None else None)
    news_end = bisect_left(timeline['news_times'], query)
    prior_news = timeline['news'][:news_end]
    for seconds in (300, 900, 3600):
        features[f'news_count_{seconds}s'] = news_end - bisect_left(timeline['news_times'], query - seconds * 1_000_000)
    features['seconds_since_latest_news'] = (query - prior_news[-1][0]) / 1_000_000 if prior_news else None
    for category in ('goal', 'card', 'shot', 'substitution', 'foul', 'corner', 'delay', 'phase'):
        typed = [instant for instant, event in prior_news if _kind_matches(event['type'], category)]
        features[f'news_{category}_count_900s'] = sum(instant >= query - 900_000_000 for instant in typed)
        features[f'seconds_since_latest_{category}'] = (query - typed[-1]) / 1_000_000 if typed else None
    features = {key: number(value) for key, value in sorted(features.items())}
    news_start = bisect_left(timeline['news_times'], query - news_seconds * 1_000_000)
    visible = [dict(event) for _, event in timeline['news'][news_start:news_end]]
    shown = visible[-max_news_items:]
    for event in shown:
        event['text'] = event['text'][:max_news_chars]
    context = {
        'market': {'question': source['market']['question'], 'outcomes': ['Yes', 'No'],
                   'kickoff': source['market']['kickoff_utc']},
        'query_time': activity.utc(query),
        'prediction_window': {'start': activity.utc(query), 'end': activity.utc(query + horizon)},
        'market_context': marks, 'news_window_start': activity.utc(query - news_seconds * 1_000_000),
        'news': shown, 'earlier_news_items_omitted': max(0, len(visible) - max_news_items),
        'prior_executions': [{'time': g['timestamp'], 'trades': g['expected']} for g in prior[-history_groups:]],
        'earlier_execution_groups_omitted': max(0, len(prior) - history_groups),
        'derived_features': features,
        **metrics.pnl_prompt_fields(snapshot),
    }
    if actor_id is not None:
        context['actor_id'] = 'actor_' + hashlib.sha256(actor_id.lower().encode()).hexdigest()[:24]
    return context, features, core['unavailable_reasons']


def _coverage(source, start, end, certificates):
    manifest = source['manifest']
    report = manifest.get('source', {})
    status = report.get('api_traversal_status')
    metrics.require(status != 'paused', f"{source['market_id']}: incomplete API capture; finish collection first")
    if status != 'exhausted':
        item = certificates.get(source['market_id'], {})
        metrics.require(item.get('complete') is True and isinstance(item.get('reason'), str) and item['reason'].strip(),
                        f"{source['market_id']}: source coverage uncertified. Supply a --coverage-file for file/SQLite captures")
        metrics.require(activity.timestamp_us(item['start']) <= start and activity.timestamp_us(item['end']) >= end,
                        f"{source['market_id']}: coverage certificate does not span the scheduled interval")
    metrics.require(activity.timestamp_us(manifest['created_at']) >= end,
                    f"{source['market_id']}: capture predates the scheduled window end")
    if manifest.get('origin_utc'):
        metrics.require(activity.timestamp_us(manifest['origin_utc']) <= start,
                        f"{source['market_id']}: export start cutoff follows the scheduled window start")


def prepare_interval_dataset(input_root, out, *, window_seconds=300, max_rows=200000,
        seed=42, match_minutes=150, pre_match_minutes=0, history_groups=8,
        news_seconds=1200, max_news_items=20, max_news_chars=300,
        max_trades_per_actor=20, validation_fraction=.1, test_fraction=.1,
        split_file=None, strict_chronology=True, include_actor_id=True,
        returns_file=None, closed_positions=None, min_return_periods=30, coverage_file=None,
        target_mode='activity', trade_tolerances=None):
    """Write a fresh, immutable split bundle; max_rows is a cap on interval targets."""
    metrics.require(target_mode in TARGET_PROTOCOLS, 'target_mode must be activity or trade-details')
    protocol = TARGET_PROTOCOLS[target_mode]
    normalized_tolerances = None
    if target_mode == 'trade-details':
        metrics.require(trade_tolerances is not None, 'trade-details requires explicit trade_tolerances')
        from interval_trade_tolerances import validate_tolerances
        normalized_tolerances = validate_tolerances(trade_tolerances)
    else:
        metrics.require(trade_tolerances is None, 'trade_tolerances apply only to trade-details targets')
    system = TRADE_DETAILS_SYSTEM if target_mode == 'trade-details' else SYSTEM
    out = Path(out)
    metrics.require(not out.exists(), f'Output exists: {out}; use a fresh directory')
    for name, value in [('window_seconds', window_seconds), ('max_rows', max_rows),
                        ('match_minutes', match_minutes), ('history_groups', history_groups),
                        ('news_seconds', news_seconds), ('max_news_items', max_news_items),
                        ('max_news_chars', max_news_chars)]:
        metrics.require(type(value) is int and value > 0, f'{name} must be a positive integer')
    metrics.require(type(pre_match_minutes) is int and pre_match_minutes >= 0, 'pre_match_minutes must be nonnegative')
    metrics.require(type(max_trades_per_actor) is int and max_trades_per_actor >= 0, 'Trade cap must be nonnegative')
    metrics.require(min_return_periods >= 2, 'min_return_periods must be at least 2')
    horizon = window_seconds * 1_000_000
    duration = (pre_match_minutes + match_minutes) * 60_000_000
    metrics.require(duration >= horizon and duration % horizon == 0,
                    'Scheduled duration must be an integer number of prediction windows')
    registry = {str(row['espn_event_id']): row for row in builder.BUNDLED_REGISTRY['fixtures']}
    certificates = metrics.read_json(Path(coverage_file)) if coverage_file else {}
    certificates = certificates.get('markets', certificates)
    sources = metrics.discover_exports([], Path(input_root))
    for source in sources:
        market = activity.native(metrics.read_json(source['path'] / 'market.json'))
        manifest = source['manifest']
        event = str(manifest.get('espn_event_id') or '')
        fixture = 'espn:' + event
        metrics.require(event in registry, f"{source['market_id']}: not a fixture in the bundled World Cup registry")
        metrics.require(market.get('fixture_id') in (None, '', fixture) and str(market.get('espn_event_id') or event) == event,
                        'Contradictory ESPN fixture identity')
        kickoff = activity.timestamp_us(market['kickoff_utc'])
        metrics.require(kickoff == activity.timestamp_us(registry[event]['kickoff_utc']), 'Kickoff differs from World Cup registry')
        metrics.require(source['outcomes'] == {'Yes', 'No'}, 'Require Yes/No binary contracts')
        metrics.require(isinstance(market.get('question'), str) and market['question'].strip(), 'Missing market question')
        metrics.require(manifest.get('market_context_version') == 2, 'Rebuild export with saved official price history')
        metrics.require(float(manifest.get('market_price_max_age_seconds', 0)) > 0, 'Missing price age policy')
        export_cap = manifest.get('max_trades_per_actor')
        metrics.require(export_cap is None or (max_trades_per_actor > 0 and max_trades_per_actor <= export_cap),
                        f"{source['market_id']}: cannot recover actors removed by source export cap {export_cap}")
        source.update(market=market, kickoff=kickoff, fixture_id=fixture,
                      scheduled_start=kickoff - pre_match_minutes * 60_000_000,
                      filtered_cohort=bool(export_cap or max_trades_per_actor or manifest.get('experiment_selection')))
        _coverage(source, source['scheduled_start'], kickoff + match_minutes * 60_000_000, certificates)
    sources.sort(key=lambda source: (source['kickoff'], source['market_id']))
    if split_file:
        mapping, split_method = builder.sft_assign_splits(sources, split_file=split_file,
            validation_fraction=validation_fraction, test_fraction=test_fraction)
    else:
        # Simultaneous matches must never straddle a chronological boundary.
        batches = {source['kickoff'] for source in sources}
        batch_sources = [{'fixture_id': str(kickoff), 'kickoff': kickoff} for kickoff in sorted(batches)]
        batch_mapping, _ = builder.sft_assign_splits(batch_sources,
            validation_fraction=validation_fraction, test_fraction=test_fraction)
        mapping = {source['fixture_id']: batch_mapping[str(source['kickoff'])] for source in sources}
        split_method = 'ordered_kickoff_batches_with_simultaneous_fixtures_kept_together'
    split_start = {split: min(s['scheduled_start'] for s in sources if mapping[s['fixture_id']] == split) for split in SPLITS}
    if strict_chronology:
        for left, right in zip(SPLITS, SPLITS[1:]):
            metrics.require(max(s['kickoff'] for s in sources if mapping[s['fixture_id']] == left) <
                            min(s['kickoff'] for s in sources if mapping[s['fixture_id']] == right),
                            'Strict chronology requires ordered matches and simultaneous kickoff fixtures in the same split')
    cutoffs = {'train': split_start['validation'], 'validation': split_start['test'], 'test': None}
    heap, totals, purged, audits, fingerprints = [], Counter(), Counter(), [], {}
    actor_split_sets = {split: set() for split in SPLITS}
    cap_exclusions = 0
    for source_index, source in enumerate(sources):
        split = mapping[source['fixture_id']]
        counts = Counter()
        seen = set()
        actor_hash = hashlib.sha256()
        for path in sorted((source['path'] / 'actors').glob('*.jsonl*')):
            actor, groups, actor_counts = metrics.actor_trade_groups(path, source)
            metrics.require(actor not in seen, 'Duplicate actor files in an export')
            seen.add(actor)
            counts.update(actor_counts)
            fingerprint = activity.sha(path)
            fingerprints[str(path)] = fingerprint
            actor_hash.update(dump([path.name, fingerprint]).encode())
            if max_trades_per_actor and actor_counts['trade_observations'] > max_trades_per_actor:
                cap_exclusions += 1
                continue
            for query in activity.eligible_queries(groups[0]['time_us'], source['scheduled_start'], duration, horizon):
                if strict_chronology and cutoffs[split] is not None and query + horizon > cutoffs[split]:
                    purged[split] += 1
                    continue
                identity = activity.candidate_id(actor, source['market_id'], query, horizon)
                candidate = (-activity.priority(seed, identity), identity, source_index, str(path), actor, query)
                totals[split] += 1
                actor_split_sets[split].add(actor)
                if len(heap) < max_rows:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    heapq.heapreplace(heap, candidate)
        for key in ('actors', 'rows', 'distinct_trade_times', 'trade_observations'):
            metrics.require(counts[key] == source['manifest']['counts'][key],
                            f"{source['market_id']}: incomplete/corrupt export: {key} count mismatch")
        audits.append({'market_id': source['market_id'], 'fixture_id': source['fixture_id'],
                       'source_path': str(source['path']), 'actor_files_sha256': actor_hash.hexdigest(),
                       'counts': dict(counts), 'retrospectively_filtered_actor_cohort': source['filtered_cohort']})
        print(f"Interval scan {source['market_id']}: {len(seen):,} actors; {sum(totals.values()):,} eligible windows", flush=True)
    metrics.require(heap, 'No eligible windows: actors need a prior execution inside the saved history')
    selected = sorted(heap, key=lambda item: (item[2], item[3], item[5]))
    selected_splits = Counter(mapping[sources[item[2]]['fixture_id']] for item in selected)
    metrics.require(all(selected_splits[s] for s in SPLITS),
                    'Each split needs selected windows; increase --max-rows or inspect split coverage')
    by_actor = defaultdict(list)
    for candidate in selected:
        by_actor[(candidate[2], candidate[3])].append(candidate)
    closed = metrics.load_closed_positions(Path(closed_positions) if closed_positions else None)
    returns = metrics.load_returns(Path(returns_file) if returns_file else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.intervals-', dir=out.parent))
    streams = {}
    counts = {split: Counter() for split in SPLITS}
    selected_actors = {split: set() for split in SPLITS}
    missing_counts, feature_names, timelines = Counter(), None, {}
    try:
        streams = {(split, kind): (work / (split + ('.features' if kind == 'features' else '') + '.jsonl')).open('w')
                   for split in SPLITS for kind in ('sft', 'features')}
        for (source_index, path), candidates in by_actor.items():
            source = sources[source_index]
            split = mapping[source['fixture_id']]
            # Keep a single market timeline resident as candidates are market-sorted.
            if source_index not in timelines:
                timelines = {source_index: activity.load_timeline(source)}
                audits[source_index]['files'] = timelines[source_index]['files']
            metrics.require(activity.sha(Path(path)) == fingerprints[path], 'Actor source changed while preparing')
            actor, groups, _ = metrics.actor_trade_groups(Path(path), source)
            ledger_key = (actor, source['condition_id'])
            for _, identity, _, _, _, query in candidates:
                context, features, missing = example_context(source, timelines[source_index], groups, query, horizon,
                    history_groups=history_groups, news_seconds=news_seconds, max_news_items=max_news_items,
                    max_news_chars=max_news_chars, closed=closed.get(ledger_key, []), returns=returns.get(ledger_key, []),
                    min_return_periods=min_return_periods, actor_id=actor if include_actor_id else None)
                target = interval_target(groups, query, horizon, target_mode)
                action = target['action']
                label = int(action == 'TRADE')
                metadata = {'row_id': identity, 'sequence_id': identity, 'actor_id': actor,
                    'market_id': source['market_id'], 'fixture_id': source['fixture_id'],
                    'target_protocol': protocol, 'query_time': activity.utc(query),
                    'interval_start': activity.utc(query), 'interval_end': activity.utc(query + horizon),
                    'interval_start_utc': activity.utc(query), 'interval_end_utc': activity.utc(query + horizon)}
                sft = {**metadata, 'target_count': 1, 'messages': [
                    {'role': 'system', 'content': system}, {'role': 'user', 'content': dump(context)},
                    {'role': 'assistant', 'content': dump(target)}]}
                feature_row = {**metadata, 'label': label, 'action': action, 'features': features}
                streams[(split, 'sft')].write(dump(sft) + '\n')
                streams[(split, 'features')].write(dump(feature_row) + '\n')
                counts[split][action] += 1
                selected_actors[split].add(actor)
                missing_counts.update(missing.keys())
                if feature_names is None:
                    feature_names = sorted(features)
                metrics.require(feature_names == sorted(features), 'Inconsistent derived feature schema')
        for stream in streams.values():
            stream.close()
        metrics.require(counts['train']['TRADE'] and counts['train']['NO_TRADE'],
                        'Training needs both labels; enlarge the cohort/window sample without balancing labels')
        split_plan = {'fixture_to_split': mapping, 'method': split_method,
                      'strict_chronology': strict_chronology,
                      'boundary_utc': {key: activity.utc(value) if value is not None else None for key, value in cutoffs.items()}}
        write_json(work / 'split_plan.json', split_plan)
        files = {path.name: activity.sha(path) for path in work.glob('*.json*')}
        manifest = {
            'format': protocol, 'target_protocol': protocol,
            'task': 'prospective_interval_trade_details' if target_mode == 'trade-details' else 'prospective_interval_activity',
            'target_mode': target_mode,
            'target_semantics': {'window': '[start,end)', 'trade_details': target_mode == 'trade-details',
                'fill_scope': 'all_captured_fills_in_interval',
                'amounts': 'total_shares_and_share_weighted_mean_price_by_side_outcome' if target_mode == 'trade-details' else None,
                'order': 'canonical_side_then_outcome' if target_mode == 'trade-details' else None,
                'evaluation_tolerances_modify_training_labels': False},
            'no_trade_targets': True, 'targets': len(selected), 'feature_names': feature_names,
            'files': files, 'sources': audits, 'fixture_to_split': mapping,
            'implementation_sha256': {name: activity.sha(ROOT / name) for name in
                ('tools/interval_decision_data.py', 'tools/actor_activity_common.py',
                 'scripts/build_actor_dataset.py', 'scripts/derive_actor_metrics.py')},
            'split_sha256': files['split_plan.json'],
            'splits': {split: {'rows': sum(counts[split].values()), 'action_counts': dict(counts[split]),
                'trade_prevalence': counts[split]['TRADE'] / sum(counts[split].values()),
                'actors': len(selected_actors[split]),
                'fixtures': sorted(key for key, value in mapping.items() if value == split)} for split in SPLITS},
            'sampling': {'seed': seed, 'window_seconds': window_seconds, 'max_rows': max_rows,
                'match_minutes': match_minutes, 'pre_match_minutes': pre_match_minutes,
                'eligible_windows': dict(totals), 'purged_boundary_windows': dict(purged),
                'method': 'uniform_bottom_k_hash_over_fixed_nonoverlapping_scheduled_windows',
                'label_balancing': False, 'enrollment': 'strictly_after_first_captured_execution_in_this_market'},
            'context': {'history_groups': history_groups, 'news_seconds': news_seconds,
                'max_news_items': max_news_items, 'max_news_chars': max_news_chars,
                'include_actor_id': include_actor_id, 'actor_id_representation': 'stable_sha256_prefix_24_hex',
                'scope': 'actor_and_current_binary_market', 'all_feature_observations_strictly_before_window_start': True},
            'global_query_time_separation_enforced': strict_chronology,
            'max_trades_per_actor': max_trades_per_actor or None,
            'additional_actors_removed_above_cap': cap_exclusions,
            'retrospectively_filtered_actor_cohort': any(s['filtered_cohort'] for s in sources),
            'test_actors_seen_in_train': len(selected_actors['test'] & selected_actors['train']),
            'validation_actors_seen_in_train': len(selected_actors['validation'] & selected_actors['train']),
            'optional_ledger_sha256': {name: activity.sha(Path(path)) if path else None
                for name, path in [('returns', returns_file), ('closed_positions', closed_positions), ('coverage', coverage_file)]},
            'min_return_periods': min_return_periods, 'metric_unavailable_counts': dict(missing_counts),
            'limitations': [
                'Labels describe captured executions; API exhaustion does not certify complete historical trading.',
                'Whole-export actor trade-count filtering is retrospective and does not prove an actor is human.',
                'Enrollment requires a prior captured execution; first trades and never-observed actors are not evaluated.',
                'ESPN times describe event occurrence and execution times describe block time, not verified decision-time knowledge.',
                'Predetermined match windows can include post-play inactivity; elapsed kickoff time is explicit.',
                'Overlapping actor windows and matches are dependent; evaluate uncertainty by match.',
                'Sharpe/Sortino require supplied regular capital-flow-adjusted returns and remain null otherwise.',
                'Inventory omits uncaptured transfers, splits, merges and redemptions; unknown holdings remain null.',
                'Base-model pretraining may already contain these historical match outcomes.',
            ]}
        if normalized_tolerances is not None:
            manifest['trade_tolerances'] = normalized_tolerances
            manifest['trade_detail_semantics'] = 'side_outcome_window_aggregates_v1'
            manifest['target_semantics']['aggregate_serialization_significant_digits'] = 40
            manifest['implementation_sha256']['tools/interval_trade_tolerances.py'] = activity.sha(
                ROOT / 'tools/interval_trade_tolerances.py')
        write_json(work / 'manifest.json', manifest)
        metrics.require(not out.exists(), f'Output appeared during preparation: {out}')
        work.rename(out)
    finally:
        for stream in streams.values():
            stream.close()
        if work.exists():
            shutil.rmtree(work)
    print(json.dumps({'output': str(out), 'targets': manifest['targets'], 'splits': manifest['splits']}, indent=2), flush=True)
    return manifest


def export_sft_variant(source, out, features):
    """Retain exact examples/labels while exposing a specified numerical subset.

    Shared PnL, prices, compact history and news remain in the common context.
    Thus this tests adding summaries, not removal of the underlying information.
    """
    source, out = Path(source), Path(out)
    manifest = json.loads((source / 'manifest.json').read_text())
    metrics.require(manifest.get('target_protocol') in TARGET_PROTOCOLS.values(), 'Unsupported dataset protocol')
    chosen = list(manifest['feature_names'] if features is None else features)
    metrics.require(len(chosen) == len(set(chosen)) and set(chosen) <= set(manifest['feature_names']), 'Unknown or duplicate feature')
    metrics.require(not out.exists(), f'Output exists: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.interval-variant-', dir=out.parent))
    try:
        for split in SPLITS:
            name = split + '.jsonl'
            metrics.require(activity.sha(source / name) == manifest['files'][name], 'Dataset checksum mismatch')
            with (work / name).open('w') as stream:
                for row in metrics.iter_jsonl(source / name):
                    context = json.loads(row['messages'][1]['content'])
                    context['derived_features'] = {key: context['derived_features'][key] for key in chosen}
                    row['messages'][1]['content'] = dump(context)
                    stream.write(dump(row) + '\n')
        shutil.copyfile(source / 'split_plan.json', work / 'split_plan.json')
        derived = copy.deepcopy(manifest)
        derived['parent_dataset'] = {'path': str(source.resolve()), 'manifest_sha256': activity.sha(source / 'manifest.json')}
        derived['selected_sft_features'] = chosen
        derived['files'] = {path.name: activity.sha(path) for path in work.glob('*.json*')}
        write_json(work / 'manifest.json', derived)
        work.rename(out)
    finally:
        if work.exists():
            shutil.rmtree(work)
    return derived
