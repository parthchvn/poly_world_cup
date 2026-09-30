"""Offline, paired activity probes. No network, weights or collection-time snapshots."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
import copy
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_world_cup_evaluation import references
import derive_actor_metrics as metrics
from compare_actor_variants import lines, timestamp_us, require
from world_cup_eval_common import dump, sha, write_json

FORMAT = 'world_cup_unseen_actor_activity_v1'
FEATURES = {'average_execution_notional', 'execution_notional_cv',
            'executions_per_day', 'buy_notional_share'}
SYSTEM = (
    'Predict whether this wallet has at least one captured execution in the specified future '
    'window [start,end): start included, end excluded. All supplied observations precede start. '
    'A = NO_TRADE (no captured execution in the window). B = TRADE (one or more captured '
    'executions). Reply with exactly A or B. Do not predict trade details. Prior executions '
    'describe one wallet in this binary market. Market prices are earlier sampled historical '
    'prices, not executable quotes or true winning probabilities; null means unavailable or '
    'stale. News times are recorded event occurrence times, not verified publication times. '
    'A no-execution label says nothing about unfilled orders or intent.'
)


def digest(value):
    return hashlib.sha256(dump(value).encode()).hexdigest()


def utc(instant):
    return datetime.fromtimestamp(instant / 1_000_000, timezone.utc).isoformat().replace('+00:00', 'Z')


def native(value):
    return json.loads(metrics.json_text(value))


def candidate_id(actor, market, query, horizon):
    return digest([actor.lower(), str(market), query, horizon])


def priority(seed, identity):
    return int(digest([seed, identity]), 16)


def eligible_queries(first_execution, kickoff, duration, horizon):
    """A scheduled grid, never an execution-centered or label-balanced sample."""
    return range(kickoff, kickoff + duration - horizon + 1, horizon) if first_execution < kickoff else range(
        kickoff + ((first_execution - kickoff) // horizon + 1) * horizon,
        kickoff + duration - horizon + 1, horizon)


def answer_at(groups, query, horizon):
    times = [g['time_us'] for g in groups]
    return int(bisect_left(times, query + horizon) > bisect_left(times, query))


def load_timeline(source):
    root, manifest = source['path'], source['manifest']
    path = root / 'market_price_history.jsonl'
    require(sha(path) == manifest['market_price_history']['sha256'], 'Price history checksum mismatch')
    prices = {'yes': [], 'no': []}
    for row in metrics.iter_jsonl(path):
        outcome = row.get('outcome')
        require(outcome in prices and row.get('source') == 'polymarket_clob_prices_history',
                'Require saved official price history, not execution-derived quotes')
        value = metrics.number(row['price'], 'price')
        require(0 <= value <= 1, 'Historical price outside [0,1]')
        prices[outcome].append((timestamp_us(row['observed_at']), str(value)))
    for values in prices.values():
        values.sort()
        require(len({r[0] for r in values}) == len(values), 'Duplicate price timestamps')
    news_path = root / 'espn_events.jsonl'
    news = []
    for row in metrics.iter_jsonl(news_path):
        instant = timestamp_us(row['time_utc'])
        require(row.get('timestamp_us', instant) == instant, 'ESPN timestamp fields disagree')
        news.append((instant, {'time': row['time_utc'], 'type': row.get('kind'), 'text': row['text']}))
    news.sort(key=lambda row: row[0])
    return {'prices': prices, 'price_times': {k: [v[0] for v in rows] for k, rows in prices.items()},
            'news': news, 'news_times': [v[0] for v in news],
            'files': {name: sha(root / name) for name in
                      ('manifest.json', 'market.json', 'espn_events.jsonl', 'market_price_history.jsonl')}}


def context_at(source, timeline, groups, query, horizon, config, history_groups, news_seconds):
    """Whitelist input fields. Never copy an interval's end-defined news/context."""
    end = bisect_left([g['time_us'] for g in groups], query)
    prior = groups[:end]
    require(prior, 'Wallet must have a strictly earlier execution before enrollment')
    history = [trade for group in prior for trade in group['trades']]
    prices = {}
    max_age = source['manifest']['market_price_max_age_seconds']
    for outcome in ('yes', 'no'):
        index = bisect_left(timeline['price_times'][outcome], query) - 1
        item = None
        if index >= 0:
            instant, value = timeline['prices'][outcome][index]
            age = (query - instant) / 1_000_000
            if age <= max_age:
                item = {'price': value, 'age_seconds': str(age)}
        prices[outcome] = item
    news_start = bisect_left(timeline['news_times'], query - news_seconds * 1_000_000)
    news_end = bisect_left(timeline['news_times'], query)
    market = source['market']
    context = {'market': {'question': market['question'], 'outcomes': ['Yes', 'No'],
                          'kickoff': market['kickoff_utc']},
        'query_time': utc(query), 'prediction_window': {'start': utc(query), 'end': utc(query + horizon)},
        'market_context': prices,
        'news_window_start': utc(query - news_seconds * 1_000_000),
        'news': [row[1] for row in timeline['news'][news_start:news_end]],
        'prior_executions': [{'time': g['timestamp'], 'trades': g['expected']} for g in prior[-history_groups:]],
        'earlier_execution_groups_omitted': max(0, len(prior) - history_groups)}
    enriched = copy.deepcopy(context)
    core = metrics.compute_metrics(history, [], [], query, config.get('lookback_seconds'),
                                   config.get('min_return_periods', 30))
    enriched['actor_metrics'] = metrics.model_metric_fields(core, config)
    # Exclude the enrollment event from this deliberately simple repeat-event rate.
    # Its observation exposure starts at enrollment and is known before the query.
    exposure = (query - prior[0]['time_us']) / 1_000_000
    baseline = -math.expm1(-(len(prior) - 1) * (horizon / 1_000_000) / exposure)
    return context, enriched, baseline


def prepare(args):
    require(not args.out.exists(), f'Output exists: {args.out}; use a fresh directory')
    require(args.targets > 0 and args.horizon_seconds > 0 and args.match_minutes > 0
            and args.history_groups > 0 and args.news_seconds > 0, 'Budgets must be positive')
    require(args.max_trades_per_actor >= 0, 'Actor trade cap must be nonnegative (0 disables it)')
    duration, horizon = args.match_minutes * 60_000_000, args.horizon_seconds * 1_000_000
    require(duration >= horizon, 'Horizon exceeds the scheduled match window')
    print('Checking both frozen training datasets and excluding train/validation wallets', flush=True)
    datasets, excluded_fixtures, excluded_markets, cutoff = references(args.basic_sft, args.inmarket_sft)
    excluded = set()
    reference = {}
    for variant, dataset in datasets.items():
        wallets = set()
        for split in ('train', 'validation'):
            wallets.update(row['actor_id'].lower() for _, row in lines(dataset['paths'][split]))
        require(all(metrics.ADDRESS.fullmatch(w) for w in wallets), 'Invalid training wallet address')
        excluded.update(wallets)
        reference[variant] = {'manifest_sha256': sha(dataset['root'] / 'manifest.json'),
            'source_sha256': {s: sha(dataset['paths'][s]) for s in ('train', 'validation')},
            'split_sha256': dataset['manifest']['split_sha256'],
            'fixtures': {s: dataset['splits'][s]['fixtures'] for s in ('train', 'validation')},
            'training_task': dataset['manifest'].get('task', 'conditional_execution'),
            'no_trade_targets': bool(dataset['manifest'].get('no_trade_targets')),
            'excluded_wallets': len(wallets)}
    config = native(datasets['inmarket']['manifest']['actor_metrics']['config'])
    require(config.get('selected_features') and set(config['selected_features']) <= FEATURES,
            'This test supports the four existing execution-history features only; no substitute metrics')
    require(not config.get('completed_position_ledger_supplied') and
            not config.get('capital_adjusted_returns_supplied'), 'External metric ledgers are unsupported')
    sources = metrics.discover_exports(args.exports, None if args.exports else args.input_root)
    eligible_sources, skipped_sources = [], []
    for source in sources:
        source['market'] = native(metrics.read_json(source['path'] / 'market.json'))
        market, manifest = source['market'], source['manifest']
        event = str(manifest.get('espn_event_id') or '')
        fixture = 'espn:' + event
        require(event.isdigit() and market.get('fixture_id') in (None, '', fixture)
                and str(market.get('espn_event_id') or event) == event, 'Contradictory or missing ESPN fixture identity')
        kickoff = timestamp_us(market['kickoff_utc'])
        if fixture in excluded_fixtures or source['market_id'] in excluded_markets or kickoff <= cutoff:
            skipped_sources.append(source['market_id'])
            continue
        require(set(source['outcomes']) == {'Yes', 'No'}, 'Invalid World Cup outcomes')
        require(isinstance(market.get('question'), str) and market['question'], 'Missing market question')
        require(manifest.get('market_context_version') == 2, 'Rebuild export with official price history')
        require(manifest.get('source', {}).get('api_traversal_status') == 'exhausted',
                f"{source['market_id']}: API traversal is incomplete or uncertified; do not label pending pages negative")
        require(timestamp_us(manifest['created_at']) >= kickoff + duration, 'Capture predates the evaluation window end')
        if manifest.get('origin_utc'):
            require(timestamp_us(manifest['origin_utc']) <= kickoff, 'Export start cutoff follows kickoff')
        export_cap = manifest.get('max_trades_per_actor')
        require(export_cap is None or (args.max_trades_per_actor > 0 and args.max_trades_per_actor <= export_cap),
                f"{source['market_id']}: source export removed wallets above {export_cap} trades; "
                'cannot recover them by raising/disabling the evaluation cap')
        filtered = bool(args.max_trades_per_actor or manifest.get('experiment_selection'))
        source.update(fixture_id=fixture, kickoff=kickoff, filtered_cohort=filtered)
        eligible_sources.append(source)
    require(eligible_sources, 'No later held-out exports; supply captures from matches after training/validation')
    eligible_sources.sort(key=lambda s: (s['kickoff'], s['market_id']))
    heap, total, actor_exclusions, cap_exclusions, eligible_wallets = [], 0, 0, 0, set()
    audit, actor_files = [], {}
    for source_index, source in enumerate(eligible_sources):
        counts = Counter()
        actor_hash = hashlib.sha256()
        files = sorted((source['path'] / 'actors').glob('*.jsonl*'))
        seen = set()
        for path in files:
            actor, groups, group_counts = metrics.actor_trade_groups(path, source)
            require(actor not in seen, 'Duplicate wallet files in one export')
            seen.add(actor)
            counts.update(group_counts)
            actor_files[str(path)] = sha(path)
            actor_hash.update(dump([path.name, actor_files[str(path)]]).encode())
            if actor in excluded:
                actor_exclusions += 1
                continue
            if args.max_trades_per_actor and group_counts['trade_observations'] > args.max_trades_per_actor:
                cap_exclusions += 1
                continue
            for query in eligible_queries(groups[0]['time_us'], source['kickoff'], duration, horizon):
                identity = candidate_id(actor, source['market_id'], query, horizon)
                # Bottom-k independent hash sampling is bounded in RAM and cannot use labels.
                candidate = (-priority(args.seed, identity), identity, source_index, str(path), actor, query)
                total += 1
                eligible_wallets.add(actor)
                if len(heap) < args.targets:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    heapq.heapreplace(heap, candidate)
        for name in ('actors', 'rows', 'distinct_trade_times', 'trade_observations'):
            require(counts[name] == source['manifest']['counts'][name],
                    f"{source['market_id']}: incomplete/corrupt export: {name} count mismatch")
        audit.append({'market_id': source['market_id'], 'fixture_id': source['fixture_id'],
            'source_path': str(source['path']), 'filtered_cohort': source['filtered_cohort'],
            'actor_files_sha256': actor_hash.hexdigest(), 'observations': dict(counts)})
        print(f"Scanned {source['market_id']}: {len(files):,} wallets; {total:,} eligible scheduled windows so far", flush=True)
    require(heap, 'No unseen wallets with prior executions and eligible scheduled windows remain')
    require(total >= args.targets, f'Only {total} eligible windows; choose --targets <= {total} before running')
    selected = sorted(heap, reverse=True)  # Seeded random order; prefixes are pilots, not the earliest actors.
    by_actor = defaultdict(list)
    for index, (_, identity, source_index, path, actor, query) in enumerate(selected):
        by_actor[(source_index, path)].append((index, identity, actor, query))
    rows = [None] * len(selected)
    timelines = {}
    for (source_index, path), queries in by_actor.items():
        source = eligible_sources[source_index]
        if source_index not in timelines:
            timelines[source_index] = load_timeline(source)
            audit[source_index]['files'] = timelines[source_index]['files']
        require(sha(Path(path)) == actor_files[path], 'Actor export changed while preparing; use immutable captures')
        _, groups, _ = metrics.actor_trade_groups(Path(path), source)
        for index, identity, actor, query in queries:
            b, m, rate = context_at(source, timelines[source_index], groups, query, horizon, config,
                                    args.history_groups, args.news_seconds)
            rows[index] = {'id': identity, 'actor_id': actor, 'market_id': source['market_id'],
                'fixture_id': source['fixture_id'], 'query_time': utc(query), 'end_time': utc(query + horizon),
                'label': answer_at(groups, query, horizon), 'prior_rate_score': rate,
                'basic': b, 'inmarket': m}
    identities = [[r['id'], r['label']] for r in rows]
    positives = sum(r['label'] for r in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.activity-', dir=args.out.parent))
    try:
        excluded_path = work / 'excluded_wallets.json'
        write_json(excluded_path, sorted(excluded))
        files = {'excluded_wallets': sha(excluded_path)}
        for name in ('basic', 'inmarket', 'labels'):
            path = work / (name + '.jsonl.gz')
            with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as stream:
                for row in rows:
                    value = ({k: v for k, v in row.items() if k not in ('basic', 'inmarket')}
                             if name == 'labels' else {'id': row['id'], 'messages': [
                                 {'role': 'system', 'content': SYSTEM},
                                 {'role': 'user', 'content': dump(row[name])}]})
                    stream.write((dump(value) + '\n').encode())
            files[name] = sha(path)
        manifest = {'format': FORMAT, 'targets': len(rows), 'positive_windows': positives,
            'negative_windows': len(rows) - positives, 'target_sha256': digest(identities),
            'files': files, 'reference': reference, 'fixtures': sorted({r['fixture_id'] for r in rows}),
            'markets': sorted({r['market_id'] for r in rows}), 'sources': audit,
            'excluded_fixtures': sorted(excluded_fixtures), 'excluded_markets': sorted(excluded_markets),
            'after_query_us': cutoff, 'excluded_wallet_count': len(excluded), 'excluded_actor_market_files': actor_exclusions,
            'eligible_wallets': len(eligible_wallets), 'selected_wallets': len({r['actor_id'] for r in rows}),
            'actor_overlap_train_validation': 0, 'eligible_windows': total, 'skipped_seen_or_earlier_markets': skipped_sources,
            'max_trades_per_actor': args.max_trades_per_actor or None,
            'additional_actor_files_excluded_above_cap': cap_exclusions,
            'cohort_rule': 'at most N captured executions across the exported actor/market history, not a market-maker classifier',
            'sampling': {'seed': args.seed, 'horizon_seconds': args.horizon_seconds, 'match_minutes': args.match_minutes,
                         'history_groups': args.history_groups, 'news_seconds': args.news_seconds,
                         'method': 'uniform_hash_sample_without_replacement_from_scheduled_eligible_wallet_windows',
                         'label_balancing': False, 'enrollment': 'strictly_after_first_captured_execution_in_this_market'},
            'inmarket_config': config, 'prospective_time_selection': True,
            'retrospectively_filtered_actor_cohort': any(s['filtered_cohort'] for s in eligible_sources),
            'task': 'zero_shot_captured_execution_occurrence_diagnostic',
            'label_semantics': 'any captured execution in [query_time,end_time); no order/intent claim',
            'limitations': ['API traversal exhaustion does not prove historical archive completeness.',
                'The requested whole-market trade-count cohort can use later counts; inputs and metrics still exclude future observations.',
                'ESPN occurrence times and execution block times are not verified publication/decision times.',
                'Only previously active wallets are enrolled; this does not test their first trade.',
                'Existing trade-only or interval-reconstruction adapters were not trained on this future-window task.',
                'Unseen means absent from both supplied SFT train/validation files; base pretraining is unknown.',
                'Repeated windows within wallets/matches are correlated; one match cannot establish generalization.']}
        write_json(work / 'manifest.json', manifest)
        # Independent read-back also audits every prompt cutoff and all actor exclusions.
        read_bundle(work)
        work.rename(args.out)
    finally:
        if work.exists():
            shutil.rmtree(work)
    report = {k: manifest[k] for k in ('targets', 'positive_windows', 'negative_windows', 'selected_wallets',
                                      'actor_overlap_train_validation', 'fixtures', 'retrospectively_filtered_actor_cohort')}
    report.update(output=str(args.out), note='No labels were used to choose windows. Few positives means wide uncertainty.')
    print(json.dumps(report, indent=2), flush=True)
    return manifest


def read_bundle(root):
    root = Path(root)
    meta = json.loads((root / 'manifest.json').read_text())
    require(meta.get('format') == FORMAT, 'Unsupported activity bundle')
    paths = {name: root / (name + '.jsonl.gz') for name in ('basic', 'inmarket', 'labels')}
    paths['excluded_wallets'] = root / 'excluded_wallets.json'
    for name, path in paths.items():
        require(sha(path) == meta['files'][name], f'{name}: checksum mismatch')
    excluded = set(json.loads(paths['excluded_wallets'].read_text()))
    records = {name: [native(r) for _, r in lines(path)] for name, path in paths.items() if name != 'excluded_wallets'}
    require(all(len(rows) == meta['targets'] for rows in records.values()), 'Activity row count mismatch')
    ids = set()
    for basic, inmarket, label in zip(records['basic'], records['inmarket'], records['labels']):
        require(basic['id'] == inmarket['id'] == label['id'] and label['id'] not in ids, 'Unpaired/duplicate windows')
        ids.add(label['id'])
        require(type(label['label']) is int and label['label'] in (0, 1), 'Invalid activity label')
        require(label['actor_id'].lower() not in excluded, 'Test wallet overlaps train/validation')
        require(label['fixture_id'] not in meta['excluded_fixtures'] and
                label['market_id'] not in meta['excluded_markets'], 'Test market/fixture overlaps training/validation')
        query, end = timestamp_us(label['query_time']), timestamp_us(label['end_time'])
        require(query > meta['after_query_us'] and end - query == meta['sampling']['horizon_seconds'] * 1_000_000,
                'Invalid query cutoff/horizon')
        require(label['id'] == candidate_id(label['actor_id'], label['market_id'], query, end-query), 'Bad window ID')
        for record in (basic, inmarket):
            require(set(record) == {'id', 'messages'} and len(record['messages']) == 2
                    and record['messages'][0] == {'role': 'system', 'content': SYSTEM}
                    and record['messages'][1]['role'] == 'user', 'Activity prompt must contain no assistant labels')
        b, m = (json.loads(r['messages'][1]['content']) for r in (basic, inmarket))
        require('actor_metrics' not in b and 'actor_metrics' in m, 'Wrong feature variant')
        m.pop('actor_metrics')
        require(b == m, 'Basic/In-market common contexts differ')
        require(b['query_time'] == label['query_time'] and b['prediction_window'] ==
                {'start': label['query_time'], 'end': label['end_time']}, 'Prompt query differs from label window')
        require(b['prior_executions'] and all(timestamp_us(g['time']) < query for g in b['prior_executions']),
                'Current/future execution leaked into prompt')
        require(all(timestamp_us(e['time']) < query for e in b['news']), 'Current/future news leaked into prompt')
        require(all(v is None or float(v['age_seconds']) > 0 for v in b['market_context'].values()),
                'Current/future price leaked into prompt')
    require(digest([[r['id'], r['label']] for r in records['labels']]) == meta['target_sha256'], 'Labels changed')
    require(sum(r['label'] for r in records['labels']) == meta['positive_windows'], 'Label class counts differ')
    return meta, records


def summarize(labels, scores):
    require(len(labels) == len(scores) > 0, 'Need aligned, nonempty labels/scores')
    require(all(type(y) is int and y in (0, 1) for y in labels), 'Invalid binary labels')
    require(all(math.isfinite(s) and 0 <= s <= 1 for s in scores), 'Invalid finite scores')
    tp = sum(y == 1 and s >= .5 for y, s in zip(labels, scores))
    fp = sum(y == 0 and s >= .5 for y, s in zip(labels, scores))
    positives, n = sum(labels), len(labels)
    fn, tn = positives - tp, n - positives - fp
    recall = tp / positives if positives else None
    specificity = tn / (n - positives) if n > positives else None
    # Average precision with complete tied-score groups; constant score = prevalence.
    ap = None
    if positives:
        by_score = defaultdict(list)
        for y, s in zip(labels, scores):
            by_score[s].append(y)
        cumulative_tp = cumulative_n = 0
        ap = 0.0
        for s in sorted(by_score, reverse=True):
            new_tp = sum(by_score[s])
            cumulative_tp += new_tp
            cumulative_n += len(by_score[s])
            ap += new_tp / positives * cumulative_tp / cumulative_n
    return {'windows': n, 'trade_windows': positives, 'no_trade_windows': n - positives,
        'prevalence': positives / n, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'threshold': .5, 'accuracy': (tp + tn) / n,
        'trade_precision': tp / (tp + fp) if tp + fp else None, 'trade_recall': recall,
        'no_trade_recall': specificity, 'trade_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        'balanced_accuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        'average_precision': ap, 'brier_score': sum((s - y) ** 2 for y, s in zip(labels, scores)) / n,
        'both_classes_present': 0 < positives < n}
