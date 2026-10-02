#!/usr/bin/env python3
"""Freeze identical Basic/In-market holdout targets on a CPU/Mac. No model needed."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from world_cup_eval_common import FORMAT, check_pair, dump, sha, targets, write_json
from compare_actor_variants import (scan_dataset, _chronology, _variant_manifest,
                                    lines, require, timestamp_us, _METRICS as metrics)


def references(basic_path, inmarket_path):
    datasets = {v: scan_dataset(p) for v, p in (('basic', basic_path), ('inmarket', inmarket_path))}
    for variant, dataset in datasets.items():
        _chronology(dataset)
        _variant_manifest(dataset, variant)
    for split in ('train', 'validation', 'test'):
        for pair in zip_longest(*(lines(d['paths'][split]) for d in datasets.values())):
            require(all(p is not None for p in pair), f'{split}: variants have different rows')
            check_pair(pair[0][1], pair[1][1])
    excluded_fixtures, excluded_markets = set(), set()
    last_query = 0
    for dataset in datasets.values():
        for split in ('train', 'validation'):
            last_query = max(last_query, dataset['splits'][split]['last_query_us'])
            for _, row in lines(dataset['paths'][split]):
                excluded_fixtures.add(row['fixture_id'])
                excluded_markets.add(row['market_id'])
    return datasets, excluded_fixtures, excluded_markets, last_query


def enrich_record(record, groups, config):
    result = copy.deepcopy(record)
    mixed = record.get('target_protocol') == 'observed_interval_and_execution_v1'
    require(len(groups) == record.get('trade_target_count', record['target_count']), 'Raw groups differ from conversation')
    history = []
    for group, offset in zip(groups, range(1, len(result['messages']), 4 if mixed else 2)):
        context = json.loads(result['messages'][offset]['content'])
        require(timestamp_us(context['query_time']) == group['time_us'], 'Feature query time mismatch')
        require(json.loads(result['messages'][offset + (3 if mixed else 1)]['content'])['trades'] == group['expected'],
                'Feature execution labels differ')
        core = metrics.compute_metrics(history, [], [], group['time_us'],
                    config.get('lookback_seconds'), config.get('min_return_periods', 30))
        context['actor_metrics'] = metrics.model_metric_fields(core, config)
        result['messages'][offset]['content'] = metrics.json_text(context)
        if mixed:
            execution_context = json.loads(result['messages'][offset + 2]['content'])
            execution_context['actor_metrics'] = metrics.model_metric_fields(core, config)
            result['messages'][offset + 2]['content'] = metrics.json_text(execution_context)
        history.extend(group['trades'])  # Add this execution only AFTER constructing its input.
    check_pair(record, result)
    return result


def collect_new(args, datasets, excluded_fixtures, excluded_markets, cutoff):
    from prepare_actor_experiment import builder, inspect_export, collect_export
    config = datasets['inmarket']['manifest']['actor_metrics']['config']
    supported = {'average_execution_notional', 'execution_notional_cv', 'executions_per_day', 'buy_notional_share'}
    require(set(config.get('selected_features', [])) <= supported and config.get('selected_features'),
            'Fresh collection supports the four execution-history features; other features need their source ledgers')
    require(not config.get('completed_position_ledger_supplied') and
            not config.get('capital_adjusted_returns_supplied'), 'External metric ledgers are unsupported here')
    fixtures = {f['fixture_id']: f for f in builder.BUNDLED_REGISTRY['fixtures']}
    contracts = {str(c['market_id']): c for c in builder.BUNDLED_REGISTRY['contracts']}
    plan = []
    for market in args.market_ids:
        require(market in contracts, f'Unknown World Cup market: {market}')
        fixture_id = contracts[market]['fixture_id']
        require(market not in excluded_markets and fixture_id not in excluded_fixtures,
                f'Market {market} overlaps a training/validation MATCH')
        fixture = fixtures[fixture_id]
        require(timestamp_us(fixture['kickoff_utc']) > cutoff,
                f'Market {market} is not chronologically later than training/validation queries')
        plan.append({'market_id': market, 'fixture_id': fixture_id, 'kickoff_utc': fixture['kickoff_utc']})
    for expected in plan:
        path = args.capture_root / ('market_' + expected['market_id'])
        path.parent.mkdir(parents=True, exist_ok=True)
        collect_export(args, expected, path, args.cache)
        source, candidates, _ = inspect_export(path, expected)
        metric_source = metrics.discover_exports([path], None)[0]
        for item in candidates:
            basic, audit, _ = builder.sft_convert_actor(item['path'], source)
            if timestamp_us(audit['first_query_time']) <= cutoff:
                args.excluded_early += 1
                continue  # Whole conversation excluded by time only; no label-based selection.
            _, groups, _ = metrics.actor_trade_groups(item['path'], metric_source)
            yield basic, enrich_record(basic, groups, config)


def prepare(args):
    require(not args.out.exists(), f'Output exists: {args.out}; choose a fresh directory')
    datasets, excluded_fixtures, excluded_markets, cutoff = references(args.basic_sft, args.inmarket_sft)
    args.excluded_early = 0
    if args.market_ids:
        require(len(set(args.market_ids)) == len(args.market_ids), 'Duplicate market IDs')
        pairs = list(collect_new(args, datasets, excluded_fixtures, excluded_markets, cutoff))
    else:
        pairs = [(a[1], b[1]) for a, b in zip(lines(datasets['basic']['paths']['test']),
                                               lines(datasets['inmarket']['paths']['test']))]
    require(bool(pairs), 'No held-out conversations remain')
    pairs.sort(key=lambda pair: (pair[0]['fixture_id'], pair[0]['sequence_id']))
    fixtures, market_ids, identities, seen, actor_markets = set(), set(), [], set(), set()
    for b, m in pairs:
        check_pair(b, m)
        require(b['fixture_id'] not in excluded_fixtures and b['market_id'] not in excluded_markets,
                'Held-out match overlaps train/validation')
        require(b['sequence_id'] not in seen and (b['actor_id'], b['market_id']) not in actor_markets,
                'Duplicate held-out conversation')
        seen.add(b['sequence_id'])
        actor_markets.add((b['actor_id'], b['market_id']))
        for t in targets(b):
            require(timestamp_us(t['query_time']) > cutoff, 'Held-out query precedes split cutoff')
            identities.append([t['id'], t['answer']])
        fixtures.add(b['fixture_id'])
        market_ids.add(b['market_id'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='eval-', dir=args.out.parent))
    try:
        files = {}
        for index, variant in enumerate(('basic', 'inmarket')):
            path = work / f'{variant}.jsonl.gz'
            with path.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as f:
                for pair in pairs:
                    f.write((dump(pair[index]) + '\n').encode())
            files[variant] = sha(path)
        reference = {}
        for variant, d in datasets.items():
            reference[variant] = {'manifest_sha256': sha(d['root'] / 'manifest.json'),
                'source_sha256': {s: sha(d['paths'][s]) for s in ('train', 'validation')},
                'split_sha256': d['manifest']['split_sha256'],
                'fixtures': {s: d['splits'][s]['fixtures'] for s in ('train', 'validation')}}
        meta = {'format': FORMAT, 'task': 'conditional_execution',
            'source': 'fresh_markets' if args.market_ids else 'frozen_test_split',
            'conversations': len(pairs), 'targets': len(identities), 'fixtures': sorted(fixtures),
            'markets': sorted(market_ids), 'excluded_early_conversations': args.excluded_early,
            'after_query_us': cutoff, 'excluded_fixtures': sorted(excluded_fixtures),
            'files': files, 'reference': reference,
            'target_sha256': hashlib.sha256(dump(identities).encode()).hexdigest(),
            'inmarket_config': datasets['inmarket']['manifest']['actor_metrics']['config'],
            'history_protocol': 'observed_prior_actions_only; current_and_future_answers_excluded',
            'base_model_pretraining_contamination': 'unknown; fine_tuning_holdout_only'}
        if all(b.get('target_protocol') == 'observed_interval_and_execution_v1' for b, _ in pairs):
            meta.update(task='observed_interval_and_execution_reconstruction', no_trade_targets=True,
                        prospective_trade_timing_benchmark=False)
        write_json(work / 'manifest.json', meta)
        work.rename(args.out)
    finally:
        if work.exists():
            shutil.rmtree(work)
    print(json.dumps({'output': str(args.out), 'targets': len(identities),
                      'fixtures': sorted(fixtures)}, indent=2))
    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--basic-sft', required=True, type=Path)
    p.add_argument('--inmarket-sft', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--market-ids', nargs='+', help='Optional NEW matches; otherwise reuse frozen test split')
    p.add_argument('--capture-root', type=Path, default=Path('data/evaluation_exports'))
    p.add_argument('--cache', type=Path, default=Path('data/market_actor_cache'))
    p.add_argument('--http-transport', choices=('curl', 'urllib'), default='curl')
    p.add_argument('--http-timeout', type=float, default=30)
    p.add_argument('--http-retries', type=int, default=3)
    p.add_argument('--http-retry-delay', type=float, default=2)
    p.add_argument('--http-retry-budget', type=float, default=90)
    p.add_argument('--http-min-interval', type=float, default=.25)
    p.add_argument('--skip-network-check', action='store_true')
    return p.parse_args(argv)


if __name__ == '__main__':
    try:
        prepare(parse_args())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'Error: {exc}')
