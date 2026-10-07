#!/usr/bin/env python3
"""Fit a frozen-LLM binary activity probe and compare development baselines.

Only train and validation are opened. All logistic fits use natural-frequency,
unweighted binary labels, training-only preprocessing, and a fixed C grid.
Validation selects C and a diagnostic F1 threshold, so reported validation
performance is a development estimate, not a new held-out result.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rank_interval_features as ranking

FORMAT = 'frozen_activity_probe_fit_v1'
SPLITS = ('train', 'validation')
C_VALUES = (0.01, 0.1, 1.0)
RECENT_FEATURES = ('seconds_since_last_execution', 'seconds_since_first_execution',
    'prior_execution_count', 'prior_execution_count_300s',
    'prior_execution_count_900s', 'prior_execution_count_3600s')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sft_file(directory, split):
    require(split in SPLITS, 'This experiment only permits train and validation')
    path = Path(directory) / f'{split}.jsonl'
    if path.is_file():
        return path
    path = Path(str(path) + '.gz')
    require(path.is_file(), f'Missing SFT split {path}')
    return path


def read_sft_metadata(path):
    """Discard large prompt bodies after validating each gold binary action."""
    result, seen = [], set()
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get('row_id')
            require(isinstance(row_id, str) and row_id and row_id not in seen,
                    f'Missing or duplicate SFT row_id at {path}:{number}')
            seen.add(row_id)
            messages = row.get('messages', [])
            require(len(messages) == 3 and [m.get('role') for m in messages] ==
                    ['system', 'user', 'assistant'], f'Unexpected messages in {row_id}')
            action = json.loads(messages[-1]['content']).get('action')
            require(action in ('TRADE', 'NO_TRADE'), f'Invalid action in {row_id}')
            require(all(isinstance(row.get(key), str) and row[key]
                        for key in ('fixture_id', 'actor_id')), f'Missing identity in {row_id}')
            result.append({'row_id': row_id, 'fixture_id': row['fixture_id'],
                           'actor_id': row['actor_id'], 'label': int(action == 'TRADE')})
    require(result, f'Empty SFT file: {path}')
    return result


def align_feature_rows(metadata, rows):
    by_id = {row['row_id']: row for row in rows}
    require(len(by_id) == len(rows), 'Duplicate feature row IDs')
    require(set(by_id) == {row['row_id'] for row in metadata},
            'Feature/SFT row IDs differ; exact coverage required')
    aligned = []
    for expected in metadata:
        row = by_id[expected['row_id']]
        for key in ('label', 'actor_id', 'fixture_id'):
            require(row.get(key) == expected[key],
                    f'Feature/SFT {key} mismatch for {expected["row_id"]}')
        aligned.append(row)
    return aligned


def alignment_digest(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps([row[k] for k in ('row_id', 'label', 'fixture_id', 'actor_id')],
                             separators=(',', ':')) + '\n').encode())
    return h.hexdigest()


def load_embeddings(directory, split, metadata, identity, np):
    """Load disjoint shards, restoring exact source order and checking identities."""
    require(split in SPLITS, 'Embedding loading only permits train and validation')
    require(identity['splits'][split]['rows'] == len(metadata), 'Embedding row count differs from SFT')
    index = {row['row_id']: i for i, row in enumerate(metadata)}
    require(len(index) == len(metadata), 'Duplicate source row IDs')
    shards = sorted(path for path in Path(directory).glob('shard*') if path.is_dir())
    require(shards, 'No embedding shards found')
    result, seen, files = None, set(), {}
    for shard in shards:
        path = shard / f'{split}.npz'
        require(path.is_file(), f'Missing completed embedding shard: {path}')
        completed = read_json(shard / 'complete.json')
        require(completed.get('test_used') is False and
                completed.get('identity_sha256') == ranking.digest(Path(directory) / 'identity.json'),
                f'Embedding completion identity mismatch: {shard}')
        files[str(path.relative_to(directory))] = ranking.digest(path)
        require(completed.get('sha256', {}).get(split) == files[str(path.relative_to(directory))],
                f'Embedding completion checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            require({'X', 'row_ids', 'labels', 'fixture_ids', 'actor_ids'} <= set(data.files),
                    f'Incomplete embedding arrays: {path}')
            x = data['X']
            ids, labels = data['row_ids'], data['labels']
            fixtures, actors = data['fixture_ids'], data['actor_ids']
            require(x.ndim == 2 and x.shape[1] > 0 and x.dtype.kind == 'f' and np.isfinite(x).all(),
                    f'Nonfinite or malformed embeddings: {path}')
            require(ids.ndim == labels.ndim == fixtures.ndim == actors.ndim == 1 and
                    len(ids) == len(labels) == len(fixtures) == len(actors) == len(x),
                    f'Embedding array shapes differ: {path}')
            require(ids.dtype.kind in 'US' and fixtures.dtype.kind in 'US' and actors.dtype.kind in 'US',
                    'Embedding IDs must be string arrays')
            require(labels.dtype.kind in 'biu' and np.isin(labels, [0, 1]).all(), 'Invalid embedding labels')
            if result is None:
                result = np.empty((len(metadata), x.shape[1]), dtype=np.float32)
            require(x.shape[1] == result.shape[1] == completed.get('hidden_size'),
                    'Embedding dimensions differ between shards/completion marker')
            positions = []
            for row_id, label, fixture, actor in zip(ids.tolist(), labels.tolist(),
                                                    fixtures.tolist(), actors.tolist()):
                require(row_id in index and row_id not in seen,
                        f'Unknown or duplicate embedding row_id: {row_id}')
                expected = metadata[index[row_id]]
                require(label == expected['label'] and fixture == expected['fixture_id'] and
                        actor == expected['actor_id'], f'Embedding gold/identity mismatch: {row_id}')
                seen.add(row_id)
                positions.append(index[row_id])
            result[positions] = x
    require(len(seen) == len(metadata), f'Incomplete embedding coverage for {split}: {len(seen)}/{len(metadata)}')
    return result, files


def validate_xgboost(directory, input_hashes, rows):
    state = read_json(directory / 'run_state.json')
    require(state.get('status') == 'completed', 'Existing XGBoost fit is not complete')
    hashes = state.get('output_sha256', {})
    require({'model.json', 'schema.json', 'report.json'} <= set(hashes), 'XGBoost lacks artifact hashes')
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), 'Invalid XGBoost artifact path')
        require(path.is_file() and ranking.digest(path) == expected, f'Changed XGBoost artifact: {name}')
    schema = read_json(directory / 'schema.json')
    require(schema.get('format') == ranking.FORMAT and schema.get('test_used') is False,
            'Unrecognized or test-contaminated XGBoost schema')
    require(schema.get('input_sha256') == input_hashes and
            state.get('signature', {}).get('input_sha256') == input_hashes,
            'XGBoost was fitted on different feature data')
    for split in SPLITS:
        require(set(schema[f'{split}_fixtures']) == {r['fixture_id'] for r in rows[split]},
                f'XGBoost {split} fixtures mismatch')
    expected_names, dropped = ranking.training_schema(rows['train'])
    require(schema['features'] == expected_names and schema['dropped_features'] == dropped,
            'XGBoost feature schema differs from training features')
    prevalence = sum(r['label'] for r in rows['train']) / len(rows['train'])
    require(abs(schema['training_prevalence'] - prevalence) < 1e-12, 'XGBoost training prevalence mismatch')
    return schema, {'run_state_sha256': ranking.digest(directory / 'run_state.json'),
                    'artifact_sha256': hashes}


def metrics(labels, scores, threshold=0.5):
    result = ranking.binary_metrics(labels, scores, threshold)
    if result['confusion']['tp'] + result['confusion']['fp'] == 0:
        result['precision'] = None
    return result


def recent_arrays(train_rows, val_rows, np):
    # Fixed simple recency/activity baseline: no data-dependent feature search.
    names = list(RECENT_FEATURES)
    require(all(name in row['features'] for row in train_rows + val_rows for name in names),
            'Missing required recent-activity feature')
    train_x = ranking.matrix(train_rows, names, np)
    val_x = ranking.matrix(val_rows, names, np)
    require(np.all(train_x[np.isfinite(train_x)] >= 0) and np.all(val_x[np.isfinite(val_x)] >= 0),
            'Recency/activity features must be nonnegative')
    # log1p is a predetermined transform, with medians fitted on training only.
    train_x, val_x = np.log1p(train_x), np.log1p(val_x)
    medians = np.asarray([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
                          for col in train_x.T], dtype=np.float32)
    def transform(values):
        missing = np.isnan(values)
        filled = np.where(missing, medians, values)
        return np.concatenate((filled, missing.astype(np.float32)), axis=1)
    return transform(train_x), transform(val_x), {'features': names, 'transform': 'log1p_then_train_median_and_missing_indicators',
                                                  'medians': medians}


def fit_logistic_grid(train_x, train_y, val_x, val_y, name, max_iter, np):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    # copy=False keeps the embedding matrix from being duplicated unnecessarily.
    scaler = StandardScaler(copy=False)
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    candidates, best, best_score = [], None, float('-inf')
    for c in C_VALUES:
        print(f'{name}: fitting C={c:g}; train={len(train_y):,}, dimensions={train_x.shape[1]:,}', flush=True)
        began = time.monotonic()
        model = LogisticRegression(C=c, solver='lbfgs', max_iter=max_iter,
            tol=1e-4, class_weight=None, fit_intercept=True, random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(train_x, train_y)
        require(not any(issubclass(item.category, ConvergenceWarning) for item in caught),
                f'{name} C={c:g} did not converge within {max_iter} iterations; '
                'do not use partial fits; rerun with larger --max-iter and a fresh output directory')
        scores = model.predict_proba(val_x)[:, 1]
        result = metrics(val_y, scores)
        elapsed = time.monotonic() - began
        candidates.append({'C': c, 'iterations': int(model.n_iter_.max()),
                           'seconds': elapsed, 'validation': result})
        # The grid is ordered strongest to weakest regularization; ties keep smaller C.
        ap = result['average_precision']
        selection_score = ap if ap is not None else -result['log_loss']
        print(f'{name}: C={c:g}, validation AP={ap}, logloss={result["log_loss"]:.6f}, {elapsed:.1f}s', flush=True)
        if selection_score > best_score:
            best_score = selection_score
            best = {'C': c, 'probabilities': scores.copy(), 'coef': model.coef_.copy(),
                    'intercept': model.intercept_.copy(), 'mean': scaler.mean_.copy(),
                    'scale': scaler.scale_.copy()}
    best['candidates'] = candidates
    return best


def probability_report(rows, scores, train_actors, threshold):
    y = [r['label'] for r in rows]
    groups = {}
    for fixture in sorted({r['fixture_id'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['fixture_id'] == fixture]
        groups[fixture] = metrics([y[i] for i in indices], [scores[i] for i in indices], threshold)
    cohorts = {}
    for name, seen in (('seen_actors', True), ('unseen_actors', False)):
        indices = [i for i, row in enumerate(rows) if (row['actor_id'] in train_actors) == seen]
        cohorts[name] = metrics([y[i] for i in indices], [scores[i] for i in indices], threshold) if indices else None
    return {'at_0_5': metrics(y, scores), 'at_validation_f1_threshold': metrics(y, scores, threshold),
            'validation_f1_threshold': threshold, 'by_fixture': groups, 'actor_cohorts': cohorts}


def save_npz(path, np, **arrays):
    temporary = path.with_suffix('.npz.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def plot_validation(labels, probabilities, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, scores in probabilities.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = metrics(labels, scores)['average_precision']
        axes[0].step(recall, precision, where='post', label=f'{name} (AP={ap:.4f})' if ap is not None else name)
        observed, predicted = calibration_curve(labels, scores, n_bins=10, strategy='quantile')
        axes[1].plot(predicted, observed, marker='o', label=name)
    axes[0].set(xlabel='Recall', ylabel='Precision', title='Validation precision–recall', xlim=(0, 1), ylim=(0, 1))
    axes[1].plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    axes[1].set(xlabel='Mean predicted probability', ylabel='Observed trade frequency',
                title='Validation calibration (quantile bins)', xlim=(0, 1), ylim=(0, 1))
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    fig.suptitle('Development diagnostics: validation used for model and threshold selection', fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def prepare_output(out, signature):
    """Reuse verified complete fits or redo an interrupted fit of identical inputs."""
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'run_state.json'
    if state_path.exists():
        state = read_json(state_path)
        require(state.get('signature') == signature and state.get('test_used') is False,
                'Output belongs to different inputs/settings; use a fresh --out')
        if state.get('status') == 'completed':
            hashes = state.get('output_sha256', {})
            require({'report.json', 'selected_params.json', 'val_predictions.csv', 'frozen_probe.npz',
                     'recent_activity_probe.npz', 'validation_pr_calibration.png'} <= set(hashes),
                    'Completed comparison lacks artifact hashes')
            for name, expected in hashes.items():
                path = (out / name).resolve()
                require(path.is_relative_to(out.resolve()) and path.is_file() and
                        ranking.digest(path) == expected, f'Completed comparison artifact changed: {name}')
            return read_json(out / 'report.json')
        require(state.get('status') == 'running', 'Unsupported comparison status')
        print('Re-fitting interrupted CPU comparison with identical inputs/settings', flush=True)
    else:
        require(not any(out.iterdir()), 'Nonempty output has no provenance; use a fresh --out')
    ranking.atomic_json(state_path, {'status': 'running', 'signature': signature, 'test_used': False})
    return None


def fit(args):
    import numpy as np
    import xgboost as xgb
    from threadpoolctl import threadpool_limits

    require(args.threads > 0 and args.max_iter > 0, 'Threads and max iterations must be positive')
    identity_path = args.embeddings_dir / 'identity.json'
    identity = read_json(identity_path)
    require(identity.get('test_used') is False, 'Embeddings must declare test_used=false')
    sft_manifest = read_json(args.dataset_dir / 'manifest.json')
    feature_manifest = read_json(args.features_dir / 'manifest.json')
    metadata, feature_rows, sft_hashes, feature_hashes = {}, {}, {}, {}
    for split in SPLITS:
        sft_path = sft_file(args.dataset_dir, split)
        feature_path = ranking.feature_file(args.features_dir, split)
        sft_hashes[split], feature_hashes[split] = ranking.digest(sft_path), ranking.digest(feature_path)
        require(sft_manifest.get('files', {}).get(sft_path.name) == sft_hashes[split], f'{split} SFT checksum mismatch')
        require(feature_manifest.get('files', {}).get(feature_path.name) == feature_hashes[split], f'{split} feature checksum mismatch')
        require(identity.get('input_sha256', {}).get(split) == sft_hashes[split], f'{split} embedding source checksum mismatch')
        metadata[split] = read_sft_metadata(sft_path)
        feature_rows[split] = align_feature_rows(metadata[split], ranking.read_rows(feature_path))
    ranking.assert_disjoint(metadata['train'], metadata['validation'])
    require(len({r['label'] for r in metadata['train']}) == 2, 'Training requires both classes')
    schema, xgb_identity = validate_xgboost(args.xgb_dir, feature_hashes, feature_rows)
    arrays, embedding_hashes = {}, {}
    for split in SPLITS:
        print(f'Loading and aligning {split} embeddings', flush=True)
        arrays[split], hashes = load_embeddings(args.embeddings_dir, split, metadata[split], identity, np)
        embedding_hashes.update(hashes)
    require(arrays['train'].shape[1] == arrays['validation'].shape[1], 'Train/validation embedding dimensions differ')
    signature = {'format': FORMAT, 'sft_sha256': sft_hashes, 'feature_sha256': feature_hashes,
        'embedding_identity_sha256': ranking.digest(identity_path), 'embedding_files_sha256': embedding_hashes,
        'alignment_sha256': {split: alignment_digest(metadata[split]) for split in SPLITS},
        'xgboost': xgb_identity, 'C_values': list(C_VALUES), 'max_iter': args.max_iter,
        'script_sha256': ranking.digest(__file__), 'ranking_helper_sha256': ranking.digest(ranking.__file__),
        'packages': {'numpy': np.__version__, 'scikit-learn': importlib.metadata.version('scikit-learn'),
                     'xgboost': xgb.__version__}}
    existing = prepare_output(args.out, signature)
    if existing is not None:
        print(f'Completed CPU comparison verified and reused: {args.out}', flush=True)
        return existing
    labels = {split: np.asarray([r['label'] for r in metadata[split]], dtype=np.int64) for split in SPLITS}
    with threadpool_limits(limits=args.threads):
        probe = fit_logistic_grid(arrays.pop('train'), labels['train'], arrays.pop('validation'),
                                 labels['validation'], 'frozen_9b_probe', args.max_iter, np)
        recent_train, recent_val, recent_preprocess = recent_arrays(feature_rows['train'], feature_rows['validation'], np)
        recent = fit_logistic_grid(recent_train, labels['train'], recent_val, labels['validation'],
                                  'recent_activity_logistic', args.max_iter, np)
        booster = xgb.Booster(params={'nthread': args.threads})
        booster.load_model(args.xgb_dir / 'model.json')
        require(booster.feature_names == schema['features'], 'XGBoost model feature order differs from schema')
        xgb_scores = ranking.predict(booster, ranking.matrix(feature_rows['validation'], schema['features'], np),
                                     schema['features'], xgb, args.threads)
    prevalence = float(labels['train'].mean())
    probabilities = {'training_prevalence': np.full(len(labels['validation']), prevalence),
        'recent_activity_logistic': recent['probabilities'], 'xgboost_full': xgb_scores,
        'frozen_9b_probe': probe['probabilities']}
    seen = {row['actor_id'] for row in metadata['train']}
    reports, selected = {}, {}
    for name, scores in probabilities.items():
        threshold = ranking.choose_threshold(labels['validation'], scores)
        reports[name] = probability_report(metadata['validation'], scores, seen, threshold)
        selected[name] = {'threshold': threshold, 'threshold_selection': 'maximum validation F1; development diagnostic'}
    for name, fitted, filename in (('frozen_9b_probe', probe, 'frozen_probe.npz'),
                                   ('recent_activity_logistic', recent, 'recent_activity_probe.npz')):
        reports[name]['C_candidates'] = fitted['candidates']
        selected[name].update(C=fitted['C'], C_selection='maximum validation AP; smaller C breaks ties')
        saved = {key: fitted[key] for key in ('coef', 'intercept', 'mean', 'scale')}
        saved.update(C=np.asarray(fitted['C']), threshold=np.asarray(selected[name]['threshold']),
                     classes=np.asarray([0, 1], dtype=np.int64), format=np.asarray(FORMAT))
        if name == 'recent_activity_logistic':
            saved.update(features=np.asarray(recent_preprocess['features']), medians=recent_preprocess['medians'],
                         transform=np.asarray(recent_preprocess['transform']))
        save_npz(args.out / filename, np, **saved)
    with (args.out / 'val_predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row_id', 'fixture_id', 'actor_id', 'label', *probabilities])
        for i, row in enumerate(metadata['validation']):
            writer.writerow([row[k] for k in ('row_id', 'fixture_id', 'actor_id', 'label')] +
                            [float(scores[i]) for scores in probabilities.values()])
    plot_validation(labels['validation'], probabilities, args.out / 'validation_pr_calibration.png')
    warnings_out = ['Validation selects regularization and thresholds; these are development diagnostics, not unbiased held-out estimates.',
        'The previous SFT adapter already used validation for monitoring; this probe does not create a fresh holdout.',
        'Many windows share actors and match context. Evaluate frozen settings on additional untouched matches.',
        'No balancing, class weighting, oversampling, test fitting, or price/size targets were used.',
        'Portable probe: sigmoid(((X - mean) / scale) @ coef.T + intercept); probabilities are not automatically calibrated.']
    if len({row['fixture_id'] for row in metadata['validation']}) < 2:
        warnings_out.append('Validation contains only one match; cross-match generalization cannot be established.')
    if len(set(labels['validation'])) < 2:
        warnings_out.append('Validation has one class: AP and ROC AUC are null; C fallback selects minimum validation log loss.')
    report = {'format': FORMAT, 'status': 'completed', 'test_used': False,
        'training_rows': len(metadata['train']), 'validation_rows': len(metadata['validation']),
        'training_prevalence': prevalence, 'validation_prevalence': float(labels['validation'].mean()),
        'train_fixtures': sorted({r['fixture_id'] for r in metadata['train']}),
        'validation_fixtures': sorted({r['fixture_id'] for r in metadata['validation']}),
        'models': reports, 'selected_parameters': selected, 'warnings': warnings_out, 'signature': signature}
    ranking.atomic_json(args.out / 'selected_params.json', selected)
    ranking.atomic_json(args.out / 'report.json', report)
    artifacts = ['report.json', 'selected_params.json', 'val_predictions.csv', 'frozen_probe.npz',
                 'recent_activity_probe.npz', 'validation_pr_calibration.png']
    ranking.atomic_json(args.out / 'run_state.json', {'status': 'completed', 'signature': signature,
        'test_used': False, 'output_sha256': {name: ranking.digest(args.out / name) for name in artifacts}})
    print(f'COMPLETE. Development comparison: {args.out / "report.json"}', flush=True)
    for name, result in reports.items():
        metric = result['at_0_5']
        tuned = result['at_validation_f1_threshold']
        print(f'{name}: AP={metric["average_precision"]}, ROC AUC={metric["roc_auc"]}, '
              f'logloss={metric["log_loss"]:.6f}, validation-selected F1={tuned["f1"]:.4f}', flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('embeddings-dir', 'dataset-dir', 'features-dir', 'xgb-dir', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-iter', type=int, default=1000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(args.threads))
    try:
        fit(args)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        raise SystemExit(f'Error: {error}') from error


if __name__ == '__main__':
    main()
