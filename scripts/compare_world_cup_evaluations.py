#!/usr/bin/env python3
"""CPU-only paired comparison of completed Basic and In-market evaluations."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from world_cup_eval_common import METRICS, dump, require, score, sha, summarize, write_json


def load_result(path, variant):
    identity = json.loads((path / 'identity.json').read_text())
    summary = json.loads((path / 'summary.json').read_text())
    require(summary['status'] == 'completed' and summary['variant'] == variant and identity['variant'] == variant,
            f'Expected completed {variant} predictions')
    require(sha(path / 'identity.json') == summary['identity_sha256'] and
            sha(path / 'predictions.jsonl') == summary['predictions_sha256'], 'Evaluation files changed')
    rows = [json.loads(line) for line in (path / 'predictions.jsonl').read_text().splitlines()]
    require(len(rows) == identity['selected_targets'] and len({r['id'] for r in rows}) == len(rows),
            'Duplicate or missing prediction targets')
    require(hashlib.sha256(dump([r['id'] for r in rows]).encode()).hexdigest() == identity['selected_ids_sha256'],
            'Prediction target order changed')
    return identity, rows


def compare(basic_path, inmarket_path):
    bi, basic = load_result(basic_path, 'basic')
    mi, inmarket = load_result(inmarket_path, 'inmarket')
    for key in ('bundle_sha256', 'target_sha256', 'decoding', 'selected_ids_sha256', 'selected_targets',
                'limit', 'history_protocol', 'tokenizer_sha256', 'versions', 'evaluator_sha256', 'common_sha256'):
        require(bi[key] == mi[key], f'Cannot fairly compare: {key} differs')
    require(len(basic) == len(inmarket), 'Evaluation target counts differ')
    by_fixture = defaultdict(lambda: {'basic': [], 'inmarket': []})
    for left, right in zip(basic, inmarket):
        for key in ('id', 'sequence_id', 'actor_id', 'fixture_id', 'market_id', 'query_time', 'answer'):
            require(left[key] == right[key], f'Paired target {key} differs')
        by_fixture[left['fixture_id']]['basic'].append(left)
        by_fixture[right['fixture_id']]['inmarket'].append(right)
    summaries = {'basic': summarize(basic), 'inmarket': summarize(inmarket)}
    differences = {k: summaries['inmarket'][k] - summaries['basic'][k] for k in METRICS}
    per_fixture = {f: {v: summarize(rows) for v, rows in variants.items()}
                   for f, variants in sorted(by_fixture.items())}
    # Confidence intervals resample MATCHES, not correlated decisions as if IID.
    ci = None
    if len(by_fixture) >= 2:
        rng = random.Random(42)
        fs = sorted(by_fixture)
        boot = {k: [] for k in METRICS}
        for _ in range(2000):
            sampled = [rng.choice(fs) for _ in fs]
            total = sum(per_fixture[f]['basic']['targets'] for f in sampled)
            for k in METRICS:
                boot[k].append(sum((per_fixture[f]['inmarket'][k] - per_fixture[f]['basic'][k]) *
                                  per_fixture[f]['basic']['targets'] for f in sampled) / total)
        ci = {k: [sorted(v)[49], sorted(v)[1949]] for k, v in boot.items()}
    paired_numeric = {'targets': 0, 'trades': 0, 'basic': defaultdict(float), 'inmarket': defaultdict(float)}
    for b, m in zip(basic, inmarket):
        scores = {'basic': score(b['answer'], b['prediction']), 'inmarket': score(m['answer'], m['prediction'])}
        if all(s['side_outcome_multiset_correct'] for s in scores.values()):
            paired_numeric['targets'] += 1
            paired_numeric['trades'] += scores['basic']['numeric_matched_trades']
            for v, s in scores.items():
                for k in ('price', 'shares', 'notional'):
                    paired_numeric[v][k + '_mae'] += s[k + '_abs_error_sum']
    for variant in ('basic', 'inmarket'):
        paired_numeric[variant] = {k: value / paired_numeric['trades']
                                  for k, value in paired_numeric[variant].items()}
    excluded = {'data', 'model', 'max_length'}
    settings_differ = {k: {'basic': bi['training_signature'].get(k), 'inmarket': mi['training_signature'].get(k)}
        for k in set(bi['training_signature']) | set(mi['training_signature'])
        if k not in excluded and bi['training_signature'].get(k) != mi['training_signature'].get(k)}
    report = {'task': 'conditional_execution', 'metrics': summaries,
        'delta_inmarket_minus_basic': differences, 'by_fixture': per_fixture,
        'match_cluster_bootstrap_95ci': ci,
        'paired_numeric_errors_on_jointly_correct_categories': paired_numeric,
        'training_setting_differences': settings_differ,
        'training_max_lengths': {v: ident['training_signature']['max_length'] for v, ident in
                                (('basic', bi), ('inmarket', mi))},
        'pilot': bool(bi['limit']),
        'notes': ['Basic means the BASIC fine-tuned adapter, not the untrained base model.',
            'All labels are TRADE. Scores do not measure trade timing, abstention, profit or forecasting.',
            'Prior observed actions are provided; predictions are not rolled into later history.',
            'Exact trade metrics ignore within-timestamp ordering and preserve duplicates.',
            'Numeric errors are conditional; use joint categorical accuracy and coverage alongside them.',
            'One match cannot establish across-match generalization. Even multiple-match intervals are exploratory.',
            'Match bootstrap does not account for wallet dependence across different matches.',
            'Base-model pretraining contamination cannot be ruled out by fine-tuning split checks.']}
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--basic', required=True, type=Path)
    p.add_argument('--inmarket', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args(argv)
    report = compare(args.basic, args.inmarket)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    require(not args.out.exists(), 'Report already exists; choose another --out')
    write_json(args.out, report)
    print('Metric                                  Basic      In-market')
    for k in METRICS:
        print(f'{k:38} {report["metrics"]["basic"][k]:9.3%} {report["metrics"]["inmarket"][k]:12.3%}')
    print(f'Matches: {len(report["by_fixture"])}. Full report: {args.out}')
    if report['training_setting_differences']:
        print('WARNING: Training settings differ beyond feature context; inspect the report.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'Error: {exc}')
