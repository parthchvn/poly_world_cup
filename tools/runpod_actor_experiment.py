#!/usr/bin/env python3
"""Run one of three independent two-GPU experiments on a shared /workspace.

Prefer --dataset-dir with a completed, transferred variant. Collection on GPU
pods is explicit via --collect. No packages are installed or upgraded.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
if str(ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tools'))
import derive_actor_metrics as metrics
from compare_actor_variants import scan_dataset, validate_chronological_splits

FEATURES = ('average_execution_notional', 'execution_notional_cv',
            'executions_per_day', 'buy_notional_share')
CODE_FILES = ('scripts/build_actor_dataset.py', 'scripts/derive_actor_metrics.py',
              'scripts/derive_global_actor_metrics.py', 'scripts/train_world_cup_multigpu.py',
              'tools/prepare_actor_experiment.py', 'tools/runpod_actor_experiment.py',
              'tools/compare_actor_variants.py', 'tools/wallet_history.py')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def paths(args):
    return {'common': args.root / 'common', 'ready': args.root / 'common_ready.json',
            'failed': args.root / 'producer_failed.json',
            'features': args.root / 'features' / args.variant,
            'run': args.root / 'runs' / args.variant,
            'token_cache': args.root / 'token_cache' / args.variant,
            'receipt': args.root / 'receipts' / (args.variant + '.json')}


def experiment_config(args):
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'datasets', 'bitsandbytes',
                 'tilelang', 'flash-linear-attention', 'fla-core', 'causal-conv1d'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'version': 1, 'training_targets': args.targets,
            'validation_targets': args.validation_targets, 'test_targets': args.test_targets,
            'seed': args.seed, 'model': str(args.model),
            'model_config_sha256': metrics.sha256(args.model / 'config.json'),
            'features': list(FEATURES),
            'package_versions': versions,
            'code_sha256': {name: metrics.sha256(ROOT / name) for name in CODE_FILES}}


def preflight(args):
    """Check the existing successful training environment without loading weights."""
    metrics.require(args.model.is_dir(), f'Model directory is missing: {args.model}')
    metrics.require(sys.version_info >= (3, 11), 'Python 3.11+ is required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ.setdefault('FLA_TILELANG', '1')
    os.environ.setdefault('FLA_DISABLE_BACKEND_DISPATCH', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    try:
        import importlib
        for name in ('torch', 'transformers', 'peft', 'accelerate', 'datasets',
                     'bitsandbytes', 'fla.ops.gated_delta_rule', 'causal_conv1d', 'tilelang'):
            importlib.import_module(name)
        import torch
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
        del Qwen3_5ForCausalLM
        metrics.require(torch.cuda.device_count() >= 2, 'This launcher needs two visible CUDA GPUs')
        for device in (0, 1):
            with torch.cuda.device(device):
                metrics.require(torch.cuda.is_bf16_supported(), f'GPU {device} lacks BF16 support')
        AutoConfig.from_pretrained(args.model, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
        metrics.require(tokenizer.is_fast and tokenizer.chat_template, 'Need the official fast tokenizer/chat template')
    except (ImportError, RuntimeError, OSError) as error:
        raise ValueError('Training environment preflight failed. Use the same RunPod image/packages '
                         f'as your successful Qwen smoke test. No packages were changed. Details: {error}') from error
    indexes = list(args.model.glob('*.safetensors.index.json'))
    weights = set()
    for index in indexes:
        data = metrics.read_json(index)
        weights.update(data.get('weight_map', {}).values())
    if weights:
        for name in weights:
            path = args.model / name
            metrics.require(not Path(name).is_absolute() and '..' not in Path(name).parts
                            and path.is_file(), f'Missing model weight shard: {name}')
    else:
        metrics.require(any(args.model.glob('*.safetensors')), 'No safetensors weights found in model directory')
    print('Preflight passed: model files, tokenizer, existing packages and two CUDA GPUs.', flush=True)


def source_inventory(exports):
    """Fingerprint the selected actor rows, including market and manifest identity."""
    inventory = hashlib.sha256()
    sources = metrics.discover_exports([], exports)
    for source in sorted(sources, key=lambda item: item['market_id']):
        for name in ('market.json', 'manifest.json'):
            inventory.update(metrics.json_text([source['market_id'], name,
                                               metrics.sha256(source['path'] / name)]).encode() + b'\n')
        for actor_file in sorted((source['path'] / 'actors').iterdir()):
            inventory.update(metrics.json_text([source['market_id'], actor_file.name,
                                               metrics.sha256(actor_file)]).encode() + b'\n')
    return inventory.hexdigest()


def publish_ready(args, prepared):
    basic = Path(prepared['basic_path']).resolve()
    exports = Path(prepared['exports_path']).resolve()
    for path in (basic, exports):
        metrics.require(path.is_relative_to(args.root), 'Prepared data must be inside the experiment root')
    dataset = scan_dataset(basic)
    ranges = validate_chronological_splits(basic)
    required = {'train': args.targets, 'validation': args.validation_targets, 'test': args.test_targets}
    for split, minimum in required.items():
        count = dataset['splits'][split]['targets']
        metrics.require(minimum <= count < minimum + 20,
                        f'{split}: expected {minimum} to {minimum + 19} complete-conversation targets; got {count}')
    ready = {'status': 'ready', 'created_at': datetime.now(timezone.utc).isoformat(),
             'config': experiment_config(args), 'basic_path': str(basic), 'exports_path': str(exports),
             'split_sha256': dataset['manifest']['split_sha256'],
             'manifest_sha256': metrics.sha256(basic / 'manifest.json'),
             'actor_inventory_sha256': source_inventory(exports),
             'counts': dataset['manifest']['stats'], 'ranges': ranges}
    atomic_json(paths(args)['ready'], ready)
    return ready


def validate_ready(args, ready):
    metrics.require(ready.get('status') == 'ready', 'Shared cohort is not ready')
    metrics.require(ready.get('config') == experiment_config(args),
                    'Shared experiment configuration/code/model/packages differ. Use the same pinned checkout, '
                    'model, package environment and flags on all pods, or choose a fresh experiment --root.')
    basic, exports = Path(ready['basic_path']).resolve(), Path(ready['exports_path']).resolve()
    metrics.require(basic.is_relative_to(args.root) and exports.is_relative_to(args.root),
                    'Shared data paths escape the experiment root')
    data = scan_dataset(basic)
    metrics.require(metrics.sha256(basic / 'manifest.json') == ready['manifest_sha256']
                    and data['manifest']['split_sha256'] == ready['split_sha256'],
                    'Shared basic dataset changed after it was frozen')
    metrics.require(source_inventory(exports) == ready['actor_inventory_sha256'],
                    'Selected actor exports changed after the cohort was frozen')
    validate_chronological_splits(basic)
    return ready


def wait_ready(args, timeout=86400):
    files = paths(args)
    started = time.monotonic()
    while not files['ready'].is_file():
        if files['failed'].is_file():
            failed = metrics.read_json(files['failed'])
            raise ValueError('Base preparation failed: ' + str(failed.get('error')) +
                             '. Fix/restart the Base pod, then rerun this launcher.')
        metrics.require(time.monotonic() - started < timeout,
                        'Timed out waiting for the Base pod. Check the shared volume and Base log.')
        print('Waiting for the Base pod to freeze the common dataset...', flush=True)
        time.sleep(30)
    return validate_ready(args, metrics.read_json(files['ready']))


@contextmanager
def variant_lock(args):
    import fcntl
    directory = args.root / 'locks'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (args.variant + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Another {args.variant} launcher is active on this shared volume') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def choose_resume(run_dir):
    valid = []
    required = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'adapter_config.json',
                'adapter_model.safetensors', 'complete.json', 'rng_state_0.pth', 'rng_state_1.pth')
    for path in run_dir.glob('checkpoint-*'):
        suffix = path.name.removeprefix('checkpoint-')
        if path.is_dir() and suffix.isdigit() and all((path / name).is_file() for name in required):
            if metrics.read_json(path / 'complete.json').get('gpus') == 2:
                valid.append((int(suffix), path))
    metrics.require(valid, f'No complete two-GPU checkpoint in {run_dir}; use a fresh --root to restart.')
    return max(valid)[1]


def training_command(args, dataset):
    locations = paths(args)
    command = [sys.executable, str(ROOT / 'scripts/train_world_cup_multigpu.py'),
               '--gpus', '2', '--gpu-ids', '0,1', '--model', str(args.model),
               '--dataset-dir', str(dataset), '--out', str(locations['run']),
               '--cache-dir', str(locations['token_cache']), '--seed', str(args.seed),
               '--epochs', '1', '--global-batch', '8', '--micro-batch', '1',
               '--max-length', '8192', '--learning-rate', '0.0001', '--rank', '16', '--alpha', '32',
               '--no-group-by-length', '--smoke-then-full']
    if args.resume:
        command.extend(['--resume', str(choose_resume(locations['run']))])
    if getattr(args, 'allow_trade_only', False):
        command.append('--allow-trade-only')
    return command


def enrich(args, ready):
    if args.variant == 'basic':
        return Path(ready['basic_path'])
    location = paths(args)['features']
    sft = location / 'sft'
    if location.exists():
        manifest = metrics.read_json(sft / 'manifest.json')
        metrics.require(manifest.get('feature_variant') == args.variant and
                        manifest.get('actor_metrics', {}).get('source_split_sha256') == ready['split_sha256'],
                        'Existing enriched data uses a different cohort or variant')
        config = manifest['actor_metrics']['config']
        metrics.require(config.get('selected_features') == metrics.select_features(','.join(FEATURES)),
                        'Existing enriched feature selection differs')
        scan_dataset(sft)
        return sft
    name = 'derive_actor_metrics.py' if args.variant == 'inmarket' else 'derive_global_actor_metrics.py'
    command = [sys.executable, str(ROOT / 'scripts' / name), '--input-root', ready['exports_path'],
               '--sft-dir', ready['basic_path'], '--out', str(location),
               '--features', ','.join(FEATURES)]
    if args.variant == 'global':
        command += ['--http-transport', 'curl', '--cache', str(args.root / 'wallet_cache'),
                    '--wallet-workers', str(args.wallet_workers),
                    '--max-runtime-seconds', str(args.max_runtime_seconds),
                    '--max-cache-gib', str(args.max_cache_gib)]
    subprocess.run(command, check=True)
    scan_dataset(sft)
    return sft


def run(args):
    if args.dataset_dir:
        return run_prepared(args)
    metrics.require(args.collect,
                    'No dataset supplied. Use --dataset-dir with completed Basic/In-market/Global SFT data. '
                    'To explicitly collect on this GPU pod, add --collect after checking the network and collection estimate.')
    args.root.mkdir(parents=True, exist_ok=True)
    files = paths(args)
    with variant_lock(args):
        try:
            preflight(args)
            if args.variant == 'basic':
                files['failed'].unlink(missing_ok=True)
                if files['ready'].is_file():
                    ready = validate_ready(args, metrics.read_json(files['ready']))
                else:
                    import prepare_actor_experiment as preparation
                    prep_args = preparation.parse_args([
                        '--out', str(files['common']), '--cache', str(args.cache),
                        '--targets', str(args.targets), '--validation-targets', str(args.validation_targets),
                        '--test-targets', str(args.test_targets), '--seed', str(args.seed),
                        '--http-transport', 'curl'])
                    ready = publish_ready(args, preparation.prepare(prep_args))
            else:
                ready = wait_ready(args)
        except BaseException as error:
            if args.variant == 'basic' and not files['ready'].exists():
                atomic_json(files['failed'], {'error': str(error), 'time': datetime.now(timezone.utc).isoformat()})
            raise
        print(json.dumps({'variant': args.variant, 'shared_counts': ready['counts'],
                          'shared_split_sha256': ready['split_sha256']}, indent=2), flush=True)
        metadata = files['run'] / 'training_metadata.json'
        if metadata.is_file() and metrics.read_json(metadata).get('status') == 'completed':
            print(f'Already completed: {files["run"] / "adapter"}', flush=True)
            return
        if files['run'].exists() and any(files['run'].iterdir()) and not args.resume:
            raise ValueError(f'Training output is nonempty: {files["run"]}. Rerun with --resume for '
                             'the latest complete checkpoint, or choose a new experiment --root.')
        dataset = enrich(args, ready)
        # Recheck the immutable source before loading the model.
        validate_ready(args, ready)
        command = training_command(args, dataset)
        atomic_json(files['receipt'], {'variant': args.variant, 'basic_split_sha256': ready['split_sha256'],
                    'dataset': str(dataset), 'dataset_split_sha256': metrics.read_json(dataset / 'manifest.json')['split_sha256'],
                    'command': command, 'config': ready['config']})
        print('Launching independent adapter training: ' + str(files['run']), flush=True)
        subprocess.run(command, check=True)
        print('Finished. Adapter: ' + str(files['run'] / 'adapter'), flush=True)


def run_prepared(args):
    """Training-only path: never calls the market collector or waits on a pod."""
    dataset = args.dataset_dir
    data = scan_dataset(dataset)
    validate_chronological_splits(dataset)
    actual = data['manifest'].get('feature_variant', 'basic')
    metrics.require(actual == args.variant, f'Dataset variant is {actual}, requested {args.variant}')
    args.root.mkdir(parents=True, exist_ok=True)
    with variant_lock(args):
        files = paths(args)
        config = experiment_config(args)
        identity = {'variant': args.variant, 'dataset': str(dataset),
                    'dataset_split_sha256': data['manifest']['split_sha256'], 'config': config}
        if files['receipt'].is_file():
            old = metrics.read_json(files['receipt'])
            metrics.require(all(old.get(key) == value for key, value in identity.items()),
                            'Run receipt differs from dataset/settings/code. Choose a fresh --root.')
        metadata = files['run'] / 'training_metadata.json'
        if metadata.is_file() and metrics.read_json(metadata).get('status') == 'completed':
            metrics.require(files['receipt'].is_file(), 'Completed run lacks its dataset receipt')
            print(f'Already completed: {files["run"] / "adapter"}', flush=True)
            return
        if files['run'].exists() and any(files['run'].iterdir()) and not args.resume:
            raise ValueError(f'Training output is nonempty: {files["run"]}. Use --resume or a fresh --root.')
        preflight(args)
        command = training_command(args, dataset)
        atomic_json(files['receipt'], {**identity, 'command': command})
        print(f'Training prepared {args.variant} data; no API collection: {dataset}', flush=True)
        subprocess.run(command, check=True)
        print('Finished. Adapter: ' + str(files['run'] / 'adapter'), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=('basic', 'inmarket', 'global'))
    parser.add_argument('--root', type=Path, default=Path('/workspace/world_cup_40k_v1'))
    parser.add_argument('--model', type=Path, default=Path('/workspace/models/Qwen3.6-27B'))
    parser.add_argument('--cache', type=Path, default=Path('/workspace/world_cup_actor_data/data/market_actor_cache'))
    parser.add_argument('--targets', type=int, default=40000)
    parser.add_argument('--validation-targets', type=int, default=2000)
    parser.add_argument('--test-targets', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='Resume the latest complete checkpoint for this variant')
    parser.add_argument('--dataset-dir', type=Path, help='Completed variant SFT directory; train without any API collection')
    parser.add_argument('--allow-trade-only', action='store_true', help='Explicit legacy reproduction without NO_TRADE targets')
    parser.add_argument('--collect', action='store_true', help='Explicitly allow legacy collection/waiting on GPU pods')
    parser.add_argument('--wallet-workers', type=int, default=4)
    parser.add_argument('--max-runtime-seconds', type=float, default=7200)
    parser.add_argument('--max-cache-gib', type=float, default=20)
    args = parser.parse_args(argv)
    for name in ('root', 'model', 'cache'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.dataset_dir:
        args.dataset_dir = args.dataset_dir.expanduser().resolve()
    metrics.require(not (args.dataset_dir and args.collect), 'Choose --dataset-dir or --collect')
    metrics.require(min(args.targets, args.validation_targets, args.test_targets) > 0, 'Target counts must be positive')
    metrics.require(not args.model.is_relative_to(args.root), 'Keep model weights outside the experiment output')
    return args


def main(argv=None):
    try:
        run(parse_args(argv))
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'Error: {error}') from error


if __name__ == '__main__':
    main()
