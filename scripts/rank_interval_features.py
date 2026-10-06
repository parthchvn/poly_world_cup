#!/usr/bin/env python3
"""Fit a TRADE/NO_TRADE interval classifier and rank features on validation data.

Input: {train,validation,test}.features.jsonl[.gz], one numeric feature dictionary
per independently defined interval. ``train`` never opens test. Install optional
training dependencies with: pip install 'numpy>=1.23' 'xgboost>=1.7,<4'.
All importance/selection numbers are development diagnostics: validation is also
used for early stopping and feature selection. Run ``evaluate`` once, after the
experiment is frozen, for a held-out test estimate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile

FORMAT = 'world_cup_interval_xgboost_v1'
FORBIDDEN_FEATURES = {'actor_id', 'wallet_id', 'fixture_id', 'match_id', 'market_id',
                      'label', 'target', 'trade_count', 'future_trade_count', 'row_id'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def feature_file(dataset, split):
    path = Path(dataset) / f'{split}.features.jsonl'
    if path.is_file():
        return path
    zipped = Path(str(path) + '.gz')
    if zipped.is_file():
        return zipped
    raise ValueError(f'Missing interval features: {path}[.gz]')


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    rows, row_ids = [], set()
    with opener(path, 'rt') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            location = f'{path}:{line_number}'
            if not isinstance(row, dict):
                raise ValueError(f'{location}: expected an object')
            row_id = row.get('row_id')
            if not isinstance(row_id, str) or not row_id or row_id in row_ids:
                raise ValueError(f'{location}: missing or duplicate row_id')
            row_ids.add(row_id)
            fixture = row.get('fixture_id', row.get('match_id'))
            if fixture is None or str(fixture) == '':
                raise ValueError(f'{location}: fixture_id or match_id is required')
            row['fixture_id'] = str(fixture)
            label = row.get('label')
            if label in ('TRADE', 'NO_TRADE'):
                label = int(label == 'TRADE')
            if type(label) not in (int, bool) or label not in (0, 1):
                raise ValueError(f'{location}: label must be 0/1 or TRADE/NO_TRADE')
            row['label'] = int(label)
            features = row.get('features')
            if not isinstance(features, dict) or not features:
                raise ValueError(f'{location}: nonempty numeric features dictionary required')
            for name, value in features.items():
                if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name)
                        or name in FORBIDDEN_FEATURES):
                    raise ValueError(f'{location}: unsupported/leaking feature name {name!r}')
                if value is not None and (not isinstance(value, (int, float))
                                          or not math.isfinite(value)):
                    raise ValueError(f'{location}: feature {name} must be finite numeric or null')
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no rows')
    return rows


def assert_disjoint(left, right):
    common_rows = {r['row_id'] for r in left} & {r['row_id'] for r in right}
    common_fixtures = {r['fixture_id'] for r in left} & {r['fixture_id'] for r in right}
    if common_rows or common_fixtures:
        raise ValueError('Split leakage: overlapping row_ids or fixtures: '
                         f'{sorted(common_rows)[:3]}, {sorted(common_fixtures)[:3]}')


def _validated_pairs(labels, probabilities):
    labels, probabilities = list(labels), list(probabilities)
    if len(labels) != len(probabilities) or not labels:
        raise ValueError('Metrics need equally sized, nonempty labels and probabilities')
    if any(v not in (0, 1) for v in labels):
        raise ValueError('Labels must be binary')
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError('Probabilities must be finite and in [0, 1]')
    return [(int(y), float(p)) for y, p in zip(labels, probabilities)]


def binary_metrics(labels, probabilities, threshold=0.5, bins=10):
    """Dependency-free binary metrics; discriminative ranking is null for one class."""
    pairs = _validated_pairs(labels, probabilities)
    if not math.isfinite(threshold) or bins < 1:
        raise ValueError('Invalid threshold or calibration bin count')
    n, positives = len(pairs), sum(y for y, _ in pairs)
    negatives = n - positives
    tp = sum(y == 1 and p >= threshold for y, p in pairs)
    fp = sum(y == 0 and p >= threshold for y, p in pairs)
    fn, tn = positives - tp, negatives - fp
    eps = 1e-15
    loss = -sum(y * math.log(max(eps, min(1 - eps, p))) +
                (1 - y) * math.log(max(eps, min(1 - eps, 1 - p))) for y, p in pairs) / n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else None
    groups = defaultdict(lambda: [0, 0])
    for y, p in pairs:
        groups[p][y] += 1
    roc, ap = None, None
    if positives and negatives:
        # Each positive beats all lower-scoring negatives; ties count one half.
        lower_negatives, wins = 0, 0.0
        for p in sorted(groups):
            neg, pos = groups[p]
            wins += pos * (lower_negatives + neg / 2)
            lower_negatives += neg
        roc = wins / (positives * negatives)
        accumulated_pos, accumulated_n, ap = 0, 0, 0.0
        for p in sorted(groups, reverse=True):
            neg, pos = groups[p]
            accumulated_pos += pos
            accumulated_n += pos + neg
            ap += (pos / positives) * (accumulated_pos / accumulated_n)
    buckets = [[] for _ in range(bins)]
    for y, p in pairs:
        buckets[min(bins - 1, int(p * bins))].append((y, p))
    calibration, ece = [], 0.0
    for index, bucket in enumerate(buckets):
        predicted = sum(p for _, p in bucket) / len(bucket) if bucket else None
        observed = sum(y for y, _ in bucket) / len(bucket) if bucket else None
        if bucket:
            ece += len(bucket) / n * abs(predicted - observed)
        calibration.append({'lower': index / bins, 'upper': (index + 1) / bins,
                            'count': len(bucket), 'mean_probability': predicted,
                            'observed_rate': observed})
    return {'n': n, 'positives': positives, 'prevalence': positives / n,
            'log_loss': loss, 'brier_score': sum((p - y) ** 2 for y, p in pairs) / n,
            'average_precision': ap, 'roc_auc': roc, 'threshold': threshold,
            'accuracy': (tp + tn) / n, 'precision': precision, 'recall': recall,
            'specificity': tn / negatives if negatives else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            'ece': ece, 'calibration': calibration}


def choose_threshold(labels, probabilities):
    """Validation F1 optimum, handling equal scores together; ties prefer abstention."""
    pairs = _validated_pairs(labels, probabilities)
    positives = sum(y for y, _ in pairs)
    if positives in (0, len(pairs)):
        return 0.5
    groups = defaultdict(lambda: [0, 0])
    for y, p in pairs:
        groups[p][y] += 1
    tp, fp, best_f1, threshold = 0, 0, 0.0, math.nextafter(max(groups), math.inf)
    for score in sorted(groups, reverse=True):
        neg, pos = groups[score]
        tp += pos
        fp += neg
        f1 = 2 * tp / (positives + tp + fp)
        if f1 > best_f1:
            best_f1, threshold = f1, score
    return threshold


def coverage(rows, names):
    out = {}
    for name in names:
        values = [r['features'].get(name) for r in rows]
        present = [value for value in values if value is not None]
        out[name] = {'present': len(present), 'missing': len(rows) - len(present),
                     'coverage': len(present) / len(rows),
                     'minimum': min(present) if present else None,
                     'maximum': max(present) if present else None}
    return out


def training_schema(rows):
    names = sorted({name for row in rows for name in row['features']})
    counts = coverage(rows, names)
    dropped, retained = {}, []
    for name in names:
        item = counts[name]
        if not item['present']:
            dropped[name] = 'all_missing_in_train'
        elif item['minimum'] == item['maximum'] and not item['missing']:
            dropped[name] = 'constant_in_train'
        else:
            retained.append(name)
    if not retained:
        raise ValueError('No variable training features remain after removing constants/all-missing columns')
    return retained, dropped


def dependencies():
    try:
        import numpy as np
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError("Install ranking dependencies: python -m pip install 'numpy>=1.23' 'xgboost>=1.7,<4'") from error
    if tuple(int(part) for part in xgb.__version__.split('.')[:2]) < (1, 7):
        raise RuntimeError('XGBoost >= 1.7 is required')
    return np, xgb


def matrix(rows, names, np):
    return np.asarray([[row['features'].get(name) if row['features'].get(name) is not None
                        else np.nan for name in names] for row in rows], dtype=np.float32)


def fit_model(train_x, train_y, val_x, val_y, names, args, xgb):
    training = xgb.DMatrix(train_x, label=train_y, feature_names=names, nthread=args.n_jobs)
    validation = xgb.DMatrix(val_x, label=val_y, feature_names=names, nthread=args.n_jobs)
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist',
              'max_depth': args.max_depth, 'eta': args.learning_rate,
              'min_child_weight': args.min_child_weight, 'subsample': 0.8,
              'colsample_bytree': 0.9, 'lambda': 1.0, 'seed': args.seed,
              'nthread': args.n_jobs, 'base_score': sum(train_y) / len(train_y)}
    # No balancing/resampling: probability estimation keeps the eligible population prevalence.
    history = {}
    model = xgb.train(params, training, num_boost_round=args.max_rounds,
                      evals=[(training, 'train'), (validation, 'validation')],
                      early_stopping_rounds=args.early_stopping_rounds,
                      evals_result=history, verbose_eval=False)
    return model, history


def predict(model, values, names, xgb, n_jobs=1):
    data = xgb.DMatrix(values, feature_names=names, nthread=n_jobs)
    best_iteration = model.attr('best_iteration')
    if best_iteration is not None:
        return model.predict(data, iteration_range=(0, int(best_iteration) + 1)).tolist()
    return model.predict(data).tolist()


def feature_group(name):
    if any(part in name for part in ('pnl', 'profit', 'drawdown', 'sharpe', 'sortino', 'return')):
        return 'performance'
    if any(part in name for part in ('inventory', 'cost_basis', 'position', 'holding', 'shares')):
        return 'inventory'
    if any(part in name for part in ('news', 'goal', 'card', 'score', 'event', 'shot', 'corner')):
        return 'news_and_match_state'
    if any(part in name for part in ('price', 'spread', 'volume', 'momentum', 'liquidity')):
        return 'market_state'
    if any(part in name for part in ('time', 'minute', 'hour', 'second', 'horizon', 'kickoff')):
        return 'timing'
    return 'actor_history'


def permutation_ranking(model, values, labels, names, args, np, xgb, baseline):
    """Unconditional permutations are predictive diagnostics, not causal effects."""
    rng = np.random.default_rng(args.seed)
    gains = model.get_score(importance_type='gain')
    ranked, grouped = [], []
    groups = defaultdict(list)
    for index, name in enumerate(names):
        groups[feature_group(name)].append(index)
    items = [(False, name, [i]) for i, name in enumerate(names)]
    items += [(True, group, indices) for group, indices in groups.items()]
    for is_group, label, indices in items:
        deltas = []
        for _ in range(args.permutation_repeats):
            shuffled = values.copy()
            permutation = rng.permutation(len(values))
            shuffled[:, indices] = values[permutation][:, indices]
            probabilities = predict(model, shuffled, names, xgb, args.n_jobs)
            deltas.append(binary_metrics(labels, probabilities)['log_loss'] - baseline)
        item = {'group' if is_group else 'feature': label,
                'mean_log_loss_increase': statistics.mean(deltas),
                'std_log_loss_increase': statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                'repeats': len(deltas), 'repeat_log_loss_increases': deltas}
        if not is_group:
            item['gain'] = float(gains.get(label, 0.0))
            item['group'] = feature_group(label)
            ranked.append(item)
        else:
            item['features'] = [names[i] for i in indices]
            grouped.append(item)
    ranked.sort(key=lambda item: (-item['mean_log_loss_increase'], -item['gain'], item['feature']))
    grouped.sort(key=lambda item: -item['mean_log_loss_increase'])
    for rank, item in enumerate(ranked, 1):
        item['rank'] = rank
    return ranked, grouped


def grouped_metrics(rows, probabilities, threshold, key):
    groups = defaultdict(lambda: ([], []))
    for row, probability in zip(rows, probabilities):
        labels, scores = groups[str(row.get(key, 'unknown'))]
        labels.append(row['label'])
        scores.append(probability)
    reports = {name: binary_metrics(labels, scores, threshold) for name, (labels, scores) in groups.items()}
    keys = ('log_loss', 'brier_score', 'average_precision', 'roc_auc', 'accuracy', 'f1')
    macro = {}
    for metric in keys:
        values = [report[metric] for report in reports.values() if report[metric] is not None]
        macro[metric] = statistics.mean(values) if values else None
    return {'groups': len(reports), 'macro': macro, 'per_group': reports}


def save_model(model, path):
    path = Path(path)
    temporary = path.with_name(path.stem + '.tmp.json')
    model.save_model(temporary)
    temporary.replace(path)


def train(args):
    inputs = {split: feature_file(args.dataset_dir, split) for split in ('train', 'validation')}
    config = {key: value for key, value in vars(args).items() if key not in ('command', 'dataset_dir', 'out')}
    versions = {}
    for package in ('numpy', 'xgboost'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    signature = {'format': FORMAT, 'input_sha256': {key: digest(path) for key, path in inputs.items()},
                 'script_sha256': digest(__file__), 'packages': versions, 'config': config}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'run_state.json'
    required = ('report.json', 'selected_features.json', 'schema.json', 'model.json',
                'selected_model.json', 'feature_importance.csv', 'feature_importance.json')
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('signature') != signature:
            raise ValueError('Output belongs to different data/settings. Use a new --out directory.')
        if state.get('status') == 'completed' and all((out / name).is_file() for name in required):
            if state.get('output_sha256') != {name: digest(out / name) for name in required}:
                raise ValueError('Ranking artifacts changed since completion. Use a new --out directory.')
            print(f'Existing complete ranking verified: {out}', flush=True)
            return json.loads((out / 'report.json').read_text())
    elif any(out.iterdir()):
        raise ValueError('Nonempty output directory has no run_state.json. Use a new --out.')
    atomic_json(state_path, {'signature': signature, 'status': 'running'})
    training, validation = (read_rows(inputs[split]) for split in ('train', 'validation'))
    assert_disjoint(training, validation)
    if len({row['label'] for row in training}) != 2:
        raise ValueError('Training needs both TRADE and NO_TRADE examples')
    names, dropped = training_schema(training)
    np, xgb = dependencies()
    train_x, val_x = matrix(training, names, np), matrix(validation, names, np)
    train_y, val_y = [r['label'] for r in training], [r['label'] for r in validation]
    print(f'Training XGBoost: {len(training)} train, {len(validation)} validation, {len(names)} features; test unopened.', flush=True)
    model, history = fit_model(train_x, train_y, val_x, val_y, names, args, xgb)
    probabilities = predict(model, val_x, names, xgb, args.n_jobs)
    threshold = choose_threshold(val_y, probabilities)
    validation_metrics = binary_metrics(val_y, probabilities, threshold)
    ranked, grouped = permutation_ranking(model, val_x, val_y, names, args, np, xgb,
                                          validation_metrics['log_loss'])
    selected = [item['feature'] for item in ranked if item['mean_log_loss_increase'] > 0][:args.top_k]
    selected_indices = [names.index(name) for name in selected]
    prevalence = sum(train_y) / len(train_y)
    if not selected:
        selected_model, selected_history = None, {}
    elif set(selected) == set(names):
        # Preserve full model's column order; XGBoost feature order is part of the model.
        selected, selected_indices = names[:], list(range(len(names)))
        selected_model, selected_history = model, history
    else:
        selected_model, selected_history = fit_model(train_x[:, selected_indices], train_y,
            val_x[:, selected_indices], val_y, selected, args, xgb)
    selected_probabilities = (predict(selected_model, val_x[:, selected_indices], selected, xgb, args.n_jobs)
                              if selected else [prevalence] * len(val_y))
    selected_threshold = choose_threshold(val_y, selected_probabilities)
    warnings = ['Validation is reused for early stopping, permutation importance, feature selection and threshold selection; report final generalization on untouched test.',
                'Unconditional permutations can create implausible correlated-feature combinations; group permutations and optional retraining ablations are supplementary.',
                'Importance describes this fitted classifier, not causal effects or guaranteed SFT improvement.']
    if len(set(val_y)) < 2:
        warnings.append('Validation has one class: ranking metrics are null and threshold defaults to 0.5.')
    if not any(item['mean_log_loss_increase'] > 0 for item in ranked):
        warnings.append('No feature had positive permutation importance. selected_features is empty; selected XGBoost uses the training-prevalence baseline. SFT should retain only its base context.')
    train_coverage = coverage(training, sorted(set(names) | set(dropped)))
    val_coverage = coverage(validation, sorted(set(names) | set(dropped)))
    for item in ranked:
        item['train_coverage'] = train_coverage[item['feature']]['coverage']
        item['validation_coverage'] = val_coverage[item['feature']]['coverage']
    ablations = []
    if args.drop_group_ablation:
        for group in sorted({feature_group(name) for name in names}):
            indices = [i for i, name in enumerate(names) if feature_group(name) != group]
            if not indices:
                continue
            kept = [names[i] for i in indices]
            ablated, _ = fit_model(train_x[:, indices], train_y, val_x[:, indices], val_y, kept, args, xgb)
            ablated_probabilities = predict(ablated, val_x[:, indices], kept, xgb, args.n_jobs)
            metrics = binary_metrics(val_y, ablated_probabilities, threshold)
            ablations.append({'removed_group': group, 'removed_features': [name for name in names if name not in kept],
                              'log_loss_increase': metrics['log_loss'] - validation_metrics['log_loss'],
                              'metrics': metrics})
    schema = {'format': FORMAT, 'features': names, 'selected_features': selected, 'dropped_features': dropped,
              'train_fixtures': sorted({r['fixture_id'] for r in training}),
              'validation_fixtures': sorted({r['fixture_id'] for r in validation}),
              'train_actor_ids': sorted({str(r.get('actor_id', 'unknown')) for r in training}),
              'training_prevalence': prevalence, 'threshold': threshold,
              'selected_threshold': selected_threshold, 'test_used': False,
              'input_sha256': signature['input_sha256']}
    report = {'format': FORMAT, 'test_used': False, 'label': {'0': 'NO_TRADE', '1': 'TRADE'},
              'training_rows': len(training), 'validation_rows': len(validation),
              'training_prevalence': prevalence, 'feature_count': len(names), 'warnings': warnings,
              'threshold_selection': 'maximum validation F1; ties prefer higher threshold',
              'validation': validation_metrics,
              'validation_at_0_5': binary_metrics(val_y, probabilities, 0.5),
              'selected_validation': binary_metrics(val_y, selected_probabilities, selected_threshold),
              'baselines': {'training_prevalence': binary_metrics(val_y, [prevalence] * len(val_y)),
                            'always_no_trade': binary_metrics(val_y, [0.0] * len(val_y))},
              'coverage': {'train': train_coverage, 'validation': val_coverage},
              'validation_extra_features_ignored': sorted({name for row in validation for name in row['features']} - set(names) - set(dropped)),
              'group_permutation_importance': grouped, 'drop_group_ablations': ablations,
              'best_iteration': int(model.attr('best_iteration')),
              'selected_best_iteration': int(selected_model.attr('best_iteration')) if selected_model else None,
              'training_history': history, 'selected_training_history': selected_history,
              'packages': {'numpy': np.__version__, 'xgboost': xgb.__version__},
              'config': config}
    save_model(model, out / 'model.json')
    if selected_model:
        save_model(selected_model, out / 'selected_model.json')
    else:
        atomic_json(out / 'selected_model.json', {'constant_probability': prevalence})
    atomic_json(out / 'schema.json', schema)
    atomic_json(out / 'selected_features.json', {'features': selected, 'test_used': False,
                'selection': f'up to {args.top_k} features with positive validation permutation mean log-loss increase; development shortlist, not significance test',
                'positive_importance_features': [item['feature'] for item in ranked if item['mean_log_loss_increase'] > 0]})
    atomic_json(out / 'feature_importance.json', ranked)
    atomic_json(out / 'report.json', report)
    with (out / 'feature_importance.csv').open('w', newline='') as stream:
        fields = ['rank', 'feature', 'group', 'mean_log_loss_increase', 'std_log_loss_increase',
                  'gain', 'train_coverage', 'validation_coverage']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(ranked)
    atomic_json(state_path, {'signature': signature, 'status': 'completed',
                            'output_sha256': {name: digest(out / name) for name in required}})
    print(f'Ranking complete: {out / "feature_importance.csv"}; selected {len(selected)} features.', flush=True)
    return report


def evaluate(args):
    state = json.loads((args.model_dir / 'run_state.json').read_text())
    if state.get('status') != 'completed':
        raise ValueError('Ranking run is incomplete')
    for name, expected in state.get('output_sha256', {}).items():
        if digest(args.model_dir / name) != expected:
            raise ValueError(f'Ranking artifact changed since completion: {name}')
    schema = json.loads((args.model_dir / 'schema.json').read_text())
    if schema.get('format') != FORMAT:
        raise ValueError('Unsupported ranking schema')
    # Verify development identities before opening test. No refit or threshold tuning here.
    for split in ('train', 'validation'):
        if digest(feature_file(args.dataset_dir, split)) != schema['input_sha256'][split]:
            raise ValueError(f'{split} features differ from the data used for model selection')
    test_path = feature_file(args.dataset_dir, 'test')
    rows = read_rows(test_path)
    test_fixtures = {row['fixture_id'] for row in rows}
    if test_fixtures & set(schema['train_fixtures'] + schema['validation_fixtures']):
        raise ValueError('Split leakage: test fixture overlaps train/validation')
    np, xgb = dependencies()
    selected = args.model_variant == 'selected'
    names = schema['selected_features'] if selected else schema['features']
    threshold = schema['selected_threshold'] if selected else schema['threshold']
    if selected and not names:
        probabilities = [schema['training_prevalence']] * len(rows)
    else:
        model = xgb.Booster()
        model.load_model(args.model_dir / ('selected_model.json' if selected else 'model.json'))
        probabilities = predict(model, matrix(rows, names, np), names, xgb, args.n_jobs)
    labels = [row['label'] for row in rows]
    seen = set(schema['train_actor_ids'])
    cohorts = {}
    for cohort, predicate in (('seen_actors', lambda r: str(r.get('actor_id', 'unknown')) in seen),
                              ('unseen_actors', lambda r: str(r.get('actor_id', 'unknown')) not in seen)):
        items = [(row['label'], p) for row, p in zip(rows, probabilities) if predicate(row)]
        cohorts[cohort] = binary_metrics([y for y, _ in items], [p for _, p in items], threshold) if items else None
    report = {'format': FORMAT, 'split': 'test', 'model_variant': args.model_variant,
              'test_sha256': digest(test_path), 'threshold_frozen_from_validation': threshold,
              'metrics': binary_metrics(labels, probabilities, threshold),
              'metrics_at_0_5': binary_metrics(labels, probabilities, 0.5),
              'actor_cohorts': cohorts,
              'by_fixture': grouped_metrics(rows, probabilities, threshold, 'fixture_id'),
              'by_actor': grouped_metrics(rows, probabilities, threshold, 'actor_id'),
              'baselines': {'training_prevalence': binary_metrics(labels, [schema['training_prevalence']] * len(labels)),
                            'always_no_trade': binary_metrics(labels, [0.0] * len(labels))},
              'coverage': coverage(rows, names)}
    destination = args.out or args.model_dir / f'test_{args.model_variant}_report.json'
    atomic_json(destination, report)
    print(f'Frozen-model test report: {destination}', flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train', help='Rank with train/validation only; never open test')
    fit.add_argument('--dataset-dir', type=Path, required=True)
    fit.add_argument('--out', type=Path, required=True)
    fit.add_argument('--top-k', type=int, default=12)
    fit.add_argument('--permutation-repeats', type=int, default=3)
    fit.add_argument('--max-rounds', type=int, default=500)
    fit.add_argument('--early-stopping-rounds', type=int, default=30)
    fit.add_argument('--max-depth', type=int, default=4)
    fit.add_argument('--learning-rate', type=float, default=0.05)
    fit.add_argument('--min-child-weight', type=float, default=5.0)
    fit.add_argument('--drop-group-ablation', action='store_true')
    fit.add_argument('--n-jobs', '--threads', type=int, default=min(8, os.cpu_count() or 1))
    fit.add_argument('--seed', type=int, default=42)
    test = sub.add_parser('evaluate', help='Explicitly open test with frozen features/model/threshold')
    test.add_argument('--dataset-dir', type=Path, required=True)
    test.add_argument('--model-dir', type=Path, required=True)
    test.add_argument('--model-variant', choices=('full', 'selected'), default='selected')
    test.add_argument('--out', type=Path)
    test.add_argument('--n-jobs', '--threads', type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    positive = ('n_jobs', 'top_k', 'permutation_repeats', 'max_rounds', 'early_stopping_rounds', 'max_depth', 'learning_rate')
    for key in positive:
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if getattr(args, 'min_child_weight', 0) < 0:
        parser.error('--min-child-weight must be nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return train(args) if args.command == 'train' else evaluate(args)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
