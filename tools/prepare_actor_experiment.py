#!/usr/bin/env python3
"""Freeze one shared chronological actor cohort for the three RunPod models.

The target budget counts predicted execution timestamps, not raw interval rows
or JSONL conversations. Entire actor/market conversations are kept together.
Original market captures are resumable; the prepared bundle is published once.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location('_experiment_actor_builder', ROOT / 'scripts/build_actor_dataset.py')
builder = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = builder
_SPEC.loader.exec_module(builder)
sys.path.insert(0, str(ROOT / 'tools'))
from compare_actor_variants import scan_dataset, validate_chronological_splits

SPLITS = ('train', 'validation', 'test')
require = builder.sft_require
sha256 = builder.sft_sha
read_json = builder.sft_read_json


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def candidate_plan():
    """One draw contract per match, with large gaps between split candidates."""
    registry = builder.BUNDLED_REGISTRY
    fixtures = sorted(registry['fixtures'], key=lambda f: (f['kickoff_utc'], f['fixture_id']))
    contracts = defaultdict(list)
    for contract in registry['contracts']:
        if contract.get('market_slug', '').endswith('-draw'):
            contracts[contract['fixture_id']].append(contract)
    result = {}
    for split, positions in (('train', (0, 24)), ('validation', (48, 60)), ('test', (72, 84))):
        result[split] = []
        for fixture in fixtures[slice(*positions)]:
            matches = contracts[fixture['fixture_id']]
            require(len(matches) == 1, f"Expected one registered draw contract for {fixture['fixture_id']}")
            result[split].append({'market_id': str(matches[0]['market_id']),
                'fixture_id': fixture['fixture_id'], 'kickoff_utc': fixture['kickoff_utc']})
        require(result[split], f'No bundled candidates for {split}')
    return result


def request_config(args, plan):
    return {'version': 1, 'targets': args.targets, 'validation_targets': args.validation_targets,
            'test_targets': args.test_targets, 'seed': args.seed, 'max_trades_per_actor': 20,
            'selection': 'registered_fixture_order_then_sha256_seed_sequence_id',
            'complete_conversations_only': True, 'strict_split_chronology': True,
            'candidate_plan': plan}


def selection_key(seed, sequence_id):
    return hashlib.sha256(f'{seed}:{sequence_id}'.encode('utf-8')).hexdigest()


def inspect_export(path, expected):
    """Validate every actor before using full-capture counts for cohort filtering."""
    require(path.is_dir() and not path.is_symlink(), f'Invalid market export: {path}')
    source = builder.sft_discover([path], None)[0]
    require(str(source['market']['market_id']) == expected['market_id']
            and source['fixture_id'] == expected['fixture_id'], f'Cached export identity differs: {path}')
    require(builder.sft_market_context_version(source['manifest']) == 2,
            f'Cached export lacks official historical prices: {path}. Use a new experiment root.')
    require(source['manifest'].get('max_trades_per_actor') == 20,
            f'Cached export must use the full-capture 20-execution actor filter: {path}')
    require(source['manifest'].get('actor_snapshots') is None,
            f'Cached export contains collection-time snapshots: {path}. Use a new experiment root.')
    candidates, counts = [], Counter()
    inventory = hashlib.sha256()
    seen = set()
    for actor_file in sorted((path / 'actors').iterdir()):
        require(actor_file.is_file() and not actor_file.is_symlink()
                and actor_file.name.endswith(('.jsonl', '.jsonl.gz')), f'Unexpected actor file: {actor_file}')
        record, audit, actor_counts = builder.sft_convert_actor(actor_file, source)
        require(record['actor_id'] not in seen, f'Duplicate actor: {actor_file}')
        seen.add(record['actor_id'])
        digest = sha256(actor_file)
        inventory.update(builder.sft_compact([actor_file.name, digest]).encode() + b'\n')
        candidates.append({'actor_id': record['actor_id'], 'sequence_id': record['sequence_id'],
            'path': actor_file, 'sha256': digest, 'counts': dict(actor_counts),
            'target_count': record['target_count'], 'execution_count': record['execution_count'],
            'first_query_time': audit['first_query_time'], 'last_query_time': audit['last_query_time']})
        counts.update(actor_counts)
        counts['actors'] += 1
    require(candidates, f'No eligible actor conversations: {path}')
    require(dict(counts) == source['manifest'].get('counts'), f'Full export counts disagree: {path}')
    report = {'source_export': str(path), 'market_id': expected['market_id'],
              'fixture_id': expected['fixture_id'], 'manifest_sha256': sha256(path / 'manifest.json'),
              'market_sha256': sha256(path / 'market.json'),
              'actor_inventory_sha256': inventory.hexdigest(), 'full_export_counts': dict(counts)}
    return source, candidates, report


def choose_conversations(candidates, *, remaining, seed, after=None):
    """Eligibility depends on a whole conversation's first query, never its label."""
    chosen = []
    count = rejected = 0
    for item in sorted(candidates, key=lambda x: (selection_key(seed, x['sequence_id']), x['sequence_id'])):
        if count >= remaining:
            break
        if after is not None and builder.sft_instant(item['first_query_time']) <= after:
            rejected += 1
            continue
        chosen.append(item)
        count += item['target_count']
    return chosen, rejected


def collect_export(args, expected, path, cache):
    if path.exists():
        print(f'Reusing complete experiment capture: {path}', flush=True)
        return
    if not args.skip_network_check and not getattr(args, '_network_checked', False):
        from check_collection_network import require_network
        event = expected['fixture_id'].split(':')[-1]
        snapshots = cache / 'espn_snapshots'
        saved_espn = any((snapshots / ('fifa.world_' + event + suffix)).is_file()
                         for suffix in ('.json', '.json.gz'))
        require_network(expected['market_id'], include_espn=not saved_espn)
        args._network_checked = True
    builder.collect_main([expected['market_id'], '--out', str(path), '--cache', str(cache),
        '--http-transport', args.http_transport, '--http-timeout', str(args.http_timeout),
        '--http-retries', str(args.http_retries), '--http-retry-delay', str(args.http_retry_delay),
        '--http-retry-budget', str(args.http_retry_budget),
        '--http-min-interval', str(args.http_min_interval), '--max-trades-per-actor', '20',
        '--skip-actor-snapshots', '--gzip'])


def copy_selected_export(source, chosen, destination, report, split):
    destination.mkdir(parents=True)
    (destination / 'actors').mkdir()
    counts, coverage = Counter(), Counter()
    inventory = hashlib.sha256()
    for name in ('market.json', 'market_price_history.jsonl', 'espn_events.jsonl',
                 'espn_unplaced_events.jsonl', 'espn_sources.json'):
        origin = source['path'] / name
        if origin.exists():
            require(origin.is_file() and not origin.is_symlink(), f'Invalid audit source: {origin}')
            shutil.copyfile(origin, destination / name)
    with (destination / 'actor_index.jsonl').open('w', encoding='utf-8') as index:
        for item in sorted(chosen, key=lambda x: x['path'].name):
            origin = item['path']
            target = destination / 'actors' / origin.name
            require(sha256(origin) == item['sha256'], f'Actor export changed during selection: {origin}')
            shutil.copyfile(origin, target)
            require(sha256(target) == item['sha256'], f'Actor copy differs: {target}')
            counts.update(item['counts'])
            counts['actors'] += 1
            inventory.update(builder.sft_compact([origin.name, item['sha256']]).encode() + b'\n')
            index.write(builder.sft_compact({'actor_id': item['actor_id'], 'path': 'actors/' + origin.name,
                                            **item['counts']}) + '\n')
            opener = gzip.open if target.name.endswith('.gz') else open
            with opener(target, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    row = builder.sft_loads(line)
                    if row['record_type'] != 'trade':
                        continue
                    context = row['market_context']
                    coverage['trade_rows'] += 1
                    coverage['both_outcomes_available'] += int(all(context[o] is not None for o in ('yes', 'no')))
                    for outcome in ('yes', 'no'):
                        available = context[outcome] is not None
                        coverage[outcome + ('_available' if available else '_missing')] += 1
                        if not available:
                            coverage[outcome + '_' + context['missing_reasons'][outcome]] += 1
    manifest = copy.deepcopy(source['manifest'])
    manifest['counts'] = dict(counts)
    manifest['market_price_coverage'] = dict(coverage)
    manifest['price_context_complete_for_exported_rows'] = coverage['both_outcomes_available'] == coverage['trade_rows']
    manifest['experiment_selection'] = {'split': split, 'whole_actor_conversations': True,
        'original_export': str(source['path']), 'original_manifest_sha256': report['manifest_sha256'],
        'original_full_export_counts': report['full_export_counts'],
        'selected_actor_inventory_sha256': inventory.hexdigest(),
        'original_source_capture_statistics_retained': True}
    write_json(destination / 'manifest.json', manifest)
    return {'market_id': str(source['market']['market_id']), 'fixture_id': source['fixture_id'],
            'split': split, 'counts': dict(counts), 'actor_inventory_sha256': inventory.hexdigest(),
            'manifest_sha256': sha256(destination / 'manifest.json'),
            'market_sha256': sha256(destination / 'market.json')}


def inventory_digest(path):
    digest = hashlib.sha256()
    for actor_file in sorted((path / 'actors').iterdir()):
        require(actor_file.is_file() and not actor_file.is_symlink(), f'Unsafe selected actor: {actor_file}')
        digest.update(builder.sft_compact([actor_file.name, sha256(actor_file)]).encode() + b'\n')
    return digest.hexdigest()


def prepared_result(bundle, config=None):
    selection = read_json(bundle / 'selection_manifest.json')
    if config is not None:
        require(selection.get('request') == config, 'Prepared experiment settings differ. Use a new --out directory.')
    require(selection.get('preparer_sha256') == sha256(__file__) and
            selection.get('builder_sha256') == sha256(ROOT / 'scripts/build_actor_dataset.py'),
            'Prepared experiment code differs. Use a new --out directory.')
    dataset = scan_dataset(bundle / 'basic')
    validate_chronological_splits(bundle / 'basic')
    require(dataset['manifest']['split_sha256'] == selection['split_sha256'], 'Frozen basic split hash changed')
    for report in selection['selected_exports']:
        path = bundle / 'selected_exports' / ('market_' + report['market_id'])
        require(sha256(path / 'manifest.json') == report['manifest_sha256']
                and sha256(path / 'market.json') == report['market_sha256']
                and inventory_digest(path) == report['actor_inventory_sha256'],
                f'Frozen selected export changed: {path}')
    return {'basic_path': str(bundle / 'basic'), 'exports_path': str(bundle / 'selected_exports'),
            'counts': dataset['manifest']['stats'], 'split_sha256': selection['split_sha256'],
            'selection_manifest': str(bundle / 'selection_manifest.json'),
            'selection_manifest_sha256': sha256(bundle / 'selection_manifest.json')}


def relocate_source_audits(basic, old_root, final_root):
    """Change generated audit paths only; never touch model messages or targets."""
    manifest = read_json(basic / 'manifest.json')
    for source in manifest['sources']:
        source['path'] = str(final_root / Path(source['path']).relative_to(old_root))
    write_json(basic / 'manifest.json', manifest)
    audit_path = basic / 'source_audit.jsonl'
    records = []
    for line in audit_path.read_text(encoding='utf-8').splitlines():
        row = builder.sft_loads(line)
        row['source_export'] = str(final_root / Path(row['source_export']).relative_to(old_root))
        records.append(builder.sft_compact(row) + '\n')
    audit_path.write_text(''.join(records), encoding='utf-8')


def prepare(args, *, plan=None):
    for name in ('targets', 'validation_targets', 'test_targets'):
        require(type(getattr(args, name)) is int and getattr(args, name) > 0, f'--{name.replace("_", "-")} must be positive')
    plan = candidate_plan() if plan is None else plan
    require(set(plan) == set(SPLITS) and all(plan[s] for s in SPLITS), 'Every split needs candidate markets')
    fixtures = [p['fixture_id'] for split in SPLITS for p in plan[split]]
    market_ids = [p['market_id'] for split in SPLITS for p in plan[split]]
    require(len(fixtures) == len(set(fixtures)) and len(market_ids) == len(set(market_ids)),
            'Candidate markets/fixtures must be distinct across splits')
    config = request_config(args, plan)
    root = args.out.resolve()
    require(not root.is_symlink(), 'Experiment output may not be a symlink')
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / 'prepared'
    if bundle.exists():
        return prepared_result(bundle, config)
    exports = root / 'exports'
    exports.mkdir(exist_ok=True)
    cache = args.cache.resolve() if args.cache else root / 'cache'
    require(not cache.is_relative_to(bundle), 'Cache must remain outside prepared data')
    budgets = {'train': args.targets, 'validation': args.validation_targets, 'test': args.test_targets}
    selections, source_reports, counts = [], [], {}
    boundary = None
    for split in SPLITS:
        targets = conversations = executions = excluded = 0
        earliest = latest = None
        for expected in plan[split]:
            path = exports / ('market_' + expected['market_id'])
            collect_export(args, expected, path, cache)
            source, candidates, report = inspect_export(path, expected)
            chosen, rejected = choose_conversations(candidates, remaining=budgets[split] - targets,
                                                     seed=args.seed, after=boundary)
            excluded += rejected
            source_reports.append({**report, 'split': split, 'chronology_exclusions_examined': rejected})
            if chosen:
                selections.append((source, chosen, report, split))
            for item in chosen:
                first, last = map(builder.sft_instant, (item['first_query_time'], item['last_query_time']))
                earliest = first if earliest is None else min(earliest, first)
                latest = last if latest is None else max(latest, last)
                targets += item['target_count']
                conversations += 1
                executions += item['execution_count']
            print(f'{split}: selected {targets:,}/{budgets[split]:,} targets in {conversations:,} whole conversations', flush=True)
            if targets >= budgets[split]:
                break
        require(targets >= budgets[split],
                f'{split}: only {targets:,} eligible targets after all {len(plan[split])} registered candidate markets. '
                f'Need {budgets[split]:,}. Captures are saved at {exports}; choose a smaller target budget in a new experiment root.')
        counts[split] = {'targets': targets, 'conversations': conversations, 'executions': executions,
                         'first_query_time': earliest.isoformat(), 'last_query_time': latest.isoformat(),
                         'chronology_exclusions_examined': excluded}
        boundary = latest
    staging = Path(tempfile.mkdtemp(prefix='.prepared-', dir=root))
    try:
        selected = staging / 'selected_exports'
        selected.mkdir()
        reports, fixture_to_split = [], {}
        for source, chosen, report, split in selections:
            market_id = str(source['market']['market_id'])
            reports.append(copy_selected_export(source, chosen, selected / ('market_' + market_id), report, split))
            fixture_to_split[source['fixture_id']] = split
        split_plan = staging / 'split_plan.json'
        write_json(split_plan, {'fixture_to_split': fixture_to_split})
        builder.prepare_main(['--input-root', str(selected), '--split-file', str(split_plan),
                              '--out', str(staging / 'basic')])
        validate_chronological_splits(staging / 'basic')
        relocate_source_audits(staging / 'basic', staging, bundle)
        manifest = read_json(staging / 'basic/manifest.json')
        for split in SPLITS:
            for field in ('targets', 'conversations', 'executions'):
                require(manifest['stats'][split][field] == counts[split][field], 'Prepared counts differ from selection')
        frozen = {'format': 'actor_experiment_selection_v1', 'request': config,
            'preparer_sha256': sha256(__file__), 'builder_sha256': sha256(ROOT / 'scripts/build_actor_dataset.py'),
            'counts': counts, 'sources': source_reports, 'selected_exports': reports,
            'split_sha256': manifest['split_sha256'],
            'financial_input_files_supplied': False,
            'raw_actor_files_unchanged': True, 'token_lengths_checked': False,
            'count_semantics': 'whole_conversations_until_at_least_requested_execution_timestamp_targets'}
        write_json(staging / 'selection_manifest.json', frozen)
        require(not bundle.exists(), 'Prepared experiment appeared while building. Use a single collector.')
        staging.rename(bundle)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return prepared_result(bundle, config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Shared experiment root containing original exports and frozen prepared bundle')
    parser.add_argument('--targets', type=int, default=40000, help='Minimum train execution targets, retaining whole conversations')
    parser.add_argument('--validation-targets', type=int, default=2000)
    parser.add_argument('--test-targets', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--http-transport', choices=('curl', 'urllib'), default='curl')
    parser.add_argument('--http-timeout', type=float, default=45)
    parser.add_argument('--http-retries', type=int, default=3)
    parser.add_argument('--http-retry-budget', type=float, default=90,
                        help='Maximum retry time per URL, including attempts and backoff')
    parser.add_argument('--http-retry-delay', type=float, default=2)
    parser.add_argument('--http-min-interval', type=float, default=1)
    parser.add_argument('--skip-network-check', action='store_true',
                        help='Skip live endpoint probes, for example when rebuilding entirely from cache')
    return parser.parse_args(argv)


def main(argv=None):
    try:
        result = prepare(parse_args(argv))
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f'Error: {error}') from error
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
