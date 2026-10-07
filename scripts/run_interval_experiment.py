#!/usr/bin/env python3
"""Prepare causal interval data, rank features, and launch overnight QLoRA.

The test split is frozen during preparation; evaluation is an explicit next-day
command. Existing completed actor exports are the input, not old SFT JSONL.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
from train_world_cup_multigpu import atomic_json, sha256_file
from run_world_cup_experiments import complete_checkpoint, install_dependencies, verify_weights


def log(message):
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def lock(path):
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another interval runner already owns this output directory') from None
        yield


def file_inventory(directory):
    """Hash artifacts for resumability; no test predictions or scores are computed."""
    return {str(p.relative_to(directory)): sha256_file(p) for p in sorted(Path(directory).rglob('*'))
            if p.is_file()}


def source_identity(args):
    # Detect ordinary source edits on resume without rereading large actor corpora.
    # Preparation records the content hashes used to produce the frozen dataset.
    records = []
    for path in sorted(args.input_root.rglob('*')):
        if path.is_file() and ('actor_snapshots' not in path.parts):
            stat = path.stat()
            records.append([str(path.relative_to(args.input_root)), stat.st_size, stat.st_mtime_ns])
    require(records, f'No source files under {args.input_root}')
    return hashlib.sha256(json.dumps(records, separators=(',', ':')).encode()).hexdigest()


def configuration(args):
    ignored = {'background', 'resume', 'install_deps', 'install_xgb', 'prepare_only', 'rank_only'}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ignored}
    code = {}
    for folder in ('scripts', 'tools', 'poly_world_cup'):
        for path in sorted((ROOT / folder).glob('*.py')):
            code[str(path.relative_to(ROOT))] = sha256_file(path)
    config.update(code=code, source_inventory_sha256=source_identity(args))
    for name in ('split_file', 'returns_file', 'closed_positions', 'coverage_file'):
        value = getattr(args, name)
        if value:
            config[name + '_sha256'] = sha256_file(value)
    return config


def stage(args, name, destination, function):
    marker = args.out / 'stages' / (name + '.json')
    if marker.exists():
        require(args.resume, f'{name} already exists; use --resume with the original options')
        require(read_json(marker)['files'] == file_inventory(destination),
                f'{name}: completed outputs changed; use a fresh experiment directory')
        log(f'Reusing completed {name}')
        return
    # XGBoost has its own data/code/version lease and can safely restart an
    # interrupted fit. Atomic dataset preparation never publishes partial data.
    resumable_ranking = name == 'xgboost' and args.resume and (destination / 'run_state.json').is_file()
    require(not destination.exists() or resumable_ranking,
            f'Unfinished {name} at {destination}; preserve or move it before retrying')
    function()
    require(destination.is_dir(), f'{name} did not produce {destination}')
    atomic_json(marker, {'files': file_inventory(destination)})


def command(args, name, values):
    logfile = args.out / 'logs' / (name + '.log')
    log(f'{name}: {shlex.join([str(v) for v in values])}')
    log(f'Live log: {logfile}')
    with logfile.open('a') as stream:
        result = subprocess.run([str(v) for v in values], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = logfile.read_text(errors='replace').splitlines()[-30:]
        raise RuntimeError(f'{name} exited {result.returncode}; {logfile}\n' + '\n'.join(tail))


def xgb_command(args, dataset, out):
    return [sys.executable, '-u', ROOT / 'scripts/rank_interval_features.py', 'train',
            '--dataset-dir', dataset, '--out', out, '--top-k', args.top_k,
            '--permutation-repeats', args.permutation_repeats, '--max-rounds', args.xgb_rounds,
            '--early-stopping-rounds', args.xgb_early_stopping, '--threads', args.xgb_threads,
            '--seed', args.seed]


def train_command(args, dataset, out, checkpoint=None):
    cmd = [sys.executable, '-u', ROOT / 'scripts/train_world_cup_multigpu.py',
           '--gpus', args.gpus, '--gpu-ids', args.gpu_ids, '--model', args.model,
           '--dataset-dir', dataset, '--out', out, '--cache-dir', args.out / 'token_cache',
           '--epochs', args.epochs, '--learning-rate', args.learning_rate, '--seed', args.seed,
           '--rank', args.rank, '--alpha', args.alpha, '--max-length', args.max_length,
           '--micro-batch', args.micro_batch, '--global-batch', args.global_batch,
           '--logging-steps', 1, '--save-steps', args.save_steps, '--eval-steps', args.eval_steps,
           '--smoke-then-full']
    if checkpoint:
        cmd += ['--resume', checkpoint]
    return cmd


def train_variant(args, name, dataset):
    output = args.out / 'runs' / name
    metadata = output / 'training_metadata.json'
    marker = args.out / 'stages' / ('train_' + name + '.json')
    if metadata.exists() and read_json(metadata).get('status') == 'completed':
        require(args.resume, f'{name}: training already completed; use --resume to reuse it')
        require((output / 'adapter/adapter_model.safetensors').is_file(), f'{name}: final adapter missing')
        identity = {'adapter': file_inventory(output / 'adapter'), 'metadata': sha256_file(metadata)}
        if marker.exists():
            require(read_json(marker) == identity, f'{name}: completed adapter/metadata changed')
        else:
            atomic_json(marker, identity)  # Recover a controller interruption after trainer completion.
        log(f'{name}: completed adapter retained')
        return
    checkpoint = complete_checkpoint(output, args.gpus) if output.exists() else None
    if output.exists():
        require(args.resume, f'{name}: unfinished run exists; use --resume')
        if checkpoint is None:
            backup = output.with_name(name + '.incomplete.' + str(time.time_ns()))
            output.rename(backup)
            log(f'{name}: no complete checkpoint; preserved old attempt at {backup}')
    command(args, 'train_' + name, train_command(args, dataset, output, checkpoint))
    require(metadata.exists() and read_json(metadata).get('status') == 'completed',
            f'{name}: trainer exited without a completed adapter')
    atomic_json(marker, {'adapter': file_inventory(output / 'adapter'), 'metadata': sha256_file(metadata)})


def dependencies(args):
    if args.install_xgb:
        # No upgrades of working packages and no GPU package dependency.
        missing = []
        for package, spec in (('numpy', 'numpy>=1.23'), ('xgboost', 'xgboost>=1.7,<4')):
            try:
                importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                missing.append(spec)
        if missing:
            subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
    if not args.prepare_only:
        try:
            import numpy  # noqa: F401
            import xgboost  # noqa: F401
        except ImportError as error:
            raise ValueError('Install XGBoost with --install-xgb, or python -m pip install "xgboost>=1.7,<4" numpy') from error
    if not (args.prepare_only or args.rank_only):
        install_dependencies(args)
        require(args.model.is_dir(), f'Local model directory missing: {args.model}')
        verify_weights(args.model)
        probe = ('import sys; sys.path.insert(0, sys.argv[1]); '
                 'from run_world_cup_experiments import probe_environment; '
                 'probe_environment(sys.argv[2], sys.argv[3])')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_ids)
        subprocess.run([sys.executable, '-c', probe, str(ROOT / 'scripts'), str(args.model), args.gpu_ids],
                       cwd=ROOT, env=env, check=True)


def run(args):
    from interval_decision_data import prepare_interval_dataset, export_sft_variant
    dependencies(args)
    identity = configuration(args)
    config_path = args.out / 'config.json'
    if config_path.exists():
        require(args.resume, 'Experiment exists; use --resume with the original options')
        require(read_json(config_path) == identity, 'Data, code or experiment options changed; use a fresh --out')
    else:
        atomic_json(config_path, identity)
    dataset = args.out / 'dataset'
    prepare_options = {key: getattr(args, key) for key in (
        'window_seconds', 'max_rows', 'seed', 'match_minutes', 'pre_match_minutes', 'history_groups',
        'news_seconds', 'max_news_items', 'max_news_chars', 'max_trades_per_actor',
        'validation_fraction', 'test_fraction', 'split_file', 'include_actor_id', 'returns_file',
        'closed_positions', 'min_return_periods', 'coverage_file', 'target_mode', 'trade_tolerances')}
    stage(args, 'prepare', dataset,
          lambda: prepare_interval_dataset(args.input_root, dataset, strict_chronology=True, **prepare_options))
    if args.prepare_only:
        log(f'Dataset ready at {dataset}. No ranking, training or testing performed.')
        return
    ranking = args.out / 'xgboost'
    stage(args, 'xgboost', ranking, lambda: command(args, 'xgboost', xgb_command(args, dataset, ranking)))
    selected = read_json(ranking / 'selected_features.json')['features']
    log('Activity-ranking shortlist: ' + ', '.join(selected))
    if args.target_mode == 'trade-details':
        log('Detailed targets predict total shares and weighted mean prices. XGBoost ranks activity only; '
            f'SFT summary policy: {args.sft_features}. Frozen tolerances: {args.trade_tolerances}')
    sft = args.out / 'sft'
    selected_sft = sft / 'selected'
    features = selected if args.sft_features == 'selected' else None
    stage(args, 'sft_selected', selected_sft, lambda: export_sft_variant(dataset, selected_sft, features))
    if args.compare_basic:
        stage(args, 'sft_basic', sft / 'basic', lambda: export_sft_variant(dataset, sft / 'basic', []))
    if args.rank_only:
        log(f'Ranking and SFT files ready at {args.out}; no training or testing performed.')
        return
    train_variant(args, 'selected', selected_sft)
    if args.compare_basic:
        train_variant(args, 'basic', sft / 'basic')
    atomic_json(args.out / 'completed.json', {'status': 'completed', 'test_evaluated': False,
                'dataset': str(selected_sft), 'run_dir': str(args.out / 'runs/selected'),
                'activity_ranking_features': selected, 'sft_features_policy': args.sft_features,
                'target_mode': args.target_mode, 'trade_tolerances': args.trade_tolerances,
                'window_seconds': args.window_seconds})
    log('Training complete. Test evaluation has not been run.')
    evaluator = ('scripts/evaluate_interval_trade_details.py' if args.target_mode == 'trade-details'
                 else 'scripts/evaluate_interval_decisions.py')
    log('Tomorrow: ' + shlex.join([sys.executable, evaluator,
        '--dataset-dir', str(selected_sft), '--run-dir', str(args.out / 'runs/selected'),
        '--model', str(args.model), '--out', str(args.out / 'evaluation/selected')]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True, help='Parent of completed raw market actor exports')
    p.add_argument('--out', type=Path, required=True, help='Fresh experiment directory on persistent storage')
    p.add_argument('--model', type=Path, default=Path('/workspace/models/Qwen3.6-27B'))
    p.add_argument('--window-seconds', type=int, default=300)
    p.add_argument('--target-mode', choices=('activity', 'trade-details'), default='activity',
                   help='Binary activity, or activity plus side/outcome totals and weighted mean prices')
    p.add_argument('--price-delta', help='Required for details: absolute price error tolerance, e.g. 0.02 (two cents)')
    p.add_argument('--shares-relative-delta', help='Required for details: share error fraction of observed shares, e.g. 0.20')
    p.add_argument('--shares-absolute-delta', default=None, help='Optional absolute share-error floor (default 0 for details)')
    p.add_argument('--max-rows', '--targets', type=int, default=200000, help='Label-independent cap across all splits')
    p.add_argument('--match-minutes', type=int, default=150)
    p.add_argument('--pre-match-minutes', type=int, default=0)
    p.add_argument('--history-groups', type=int, default=8)
    p.add_argument('--news-seconds', type=int, default=1200)
    p.add_argument('--max-news-items', type=int, default=20)
    p.add_argument('--max-news-chars', type=int, default=300)
    p.add_argument('--max-trades-per-actor', type=int, default=20)
    p.add_argument('--validation-fraction', type=float, default=.1)
    p.add_argument('--test-fraction', type=float, default=.1)
    p.add_argument('--split-file', type=Path)
    p.add_argument('--returns-file', type=Path)
    p.add_argument('--closed-positions', type=Path)
    p.add_argument('--coverage-file', type=Path)
    p.add_argument('--min-return-periods', type=int, default=30)
    p.add_argument('--include-actor-id', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--top-k', type=int, default=12)
    p.add_argument('--permutation-repeats', type=int, default=3)
    p.add_argument('--xgb-rounds', type=int, default=500)
    p.add_argument('--xgb-early-stopping', type=int, default=30)
    p.add_argument('--xgb-threads', type=int, default=8)
    p.add_argument('--sft-features', choices=('selected', 'all'), default=None,
                   help='Default selected for activity, all for details: activity ranks do not measure size/price usefulness')
    p.add_argument('--compare-basic', action='store_true', help='Also train a fresh basic adapter after the selected adapter')
    p.add_argument('--gpus', type=int, default=2)
    p.add_argument('--gpu-ids', default=None)
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--max-length', type=int, default=8192)
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--global-batch', type=int, default=8)
    p.add_argument('--save-steps', type=int, default=100)
    p.add_argument('--eval-steps', type=int, default=277)
    p.add_argument('--install-xgb', action='store_true', help='Install missing CPU XGBoost/numpy packages')
    p.add_argument('--install-deps', action='store_true', help='Use existing torch2.8/CUDA12.8 pinned SFT dependency installer')
    stop = p.add_mutually_exclusive_group()
    stop.add_argument('--prepare-only', action='store_true')
    stop.add_argument('--rank-only', action='store_true')
    p.add_argument('--background', action='store_true', help='Detach runner and write logs under --out/logs')
    p.add_argument('--resume', action='store_true', help='Reuse verified stages and resume a complete training checkpoint')
    args = p.parse_args(argv)
    if args.target_mode == 'trade-details':
        from interval_trade_tolerances import validate_tolerances
        require(args.price_delta is not None and args.shares_relative_delta is not None,
                '--target-mode trade-details requires explicit --price-delta and --shares-relative-delta')
        args.trade_tolerances = validate_tolerances({'price_delta': args.price_delta,
            'shares_relative_delta': args.shares_relative_delta,
            'shares_absolute_delta': args.shares_absolute_delta or '0'})
    else:
        require(all(getattr(args, name) is None for name in
                    ('price_delta', 'shares_relative_delta', 'shares_absolute_delta')),
                'Numeric tolerances require --target-mode trade-details; activity has no price/size output')
        args.trade_tolerances = None
    args.sft_features = args.sft_features or ('all' if args.target_mode == 'trade-details' else 'selected')
    for name in ('input_root', 'out', 'model', 'split_file', 'returns_file', 'closed_positions', 'coverage_file'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    require(args.input_root.is_dir(), f'Input root not found: {args.input_root}')
    require(args.input_root != args.out and args.input_root not in args.out.parents,
            '--out must be outside --input-root so outputs cannot enter their own source inventory')
    args.gpu_ids = args.gpu_ids or ','.join(str(i) for i in range(args.gpus))
    require(args.gpus > 0 and len(args.gpu_ids.split(',')) == args.gpus
            and len(set(args.gpu_ids.split(','))) == args.gpus
            and all(i.isdigit() for i in args.gpu_ids.split(',')), '--gpu-ids must list --gpus distinct indices')
    for key in ('window_seconds', 'max_rows', 'match_minutes', 'history_groups', 'news_seconds',
                'max_news_items', 'max_news_chars', 'min_return_periods', 'top_k', 'permutation_repeats',
                'xgb_rounds', 'xgb_early_stopping', 'xgb_threads', 'epochs', 'learning_rate',
                'rank', 'alpha', 'max_length', 'micro_batch', 'global_batch', 'save_steps', 'eval_steps'):
        require(getattr(args, key) > 0, f'--{key.replace("_", "-")} must be positive')
    require(args.global_batch % (args.gpus * args.micro_batch) == 0,
            '--global-batch must be divisible by --gpus * --micro-batch')
    return args


def main(argv=None):
    original = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parse_args(original)
        for folder in ('logs', 'stages'):
            (args.out / folder).mkdir(parents=True, exist_ok=True)
        if args.background:
            original.remove('--background')
            path = args.out / 'logs' / ('runner-' + str(time.time_ns()) + '.log')
            with path.open('a') as stream:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), *original],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            log(f'Background PID {process.pid}; follow: tail -f {shlex.quote(str(path))}')
            return 0
        with lock(args.out / 'runner.lock'):
            atomic_json(args.out / 'status.json', {'status': 'running', 'pid': os.getpid(), 'test_evaluated': False})
            try:
                run(args)
            except BaseException as error:
                atomic_json(args.out / 'status.json', {'status': 'failed', 'error': str(error), 'test_evaluated': False})
                raise
            atomic_json(args.out / 'status.json', {'status': 'prepared' if args.prepare_only else
                'ranked' if args.rank_only else 'completed', 'test_evaluated': False})
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        log(f'ERROR: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
