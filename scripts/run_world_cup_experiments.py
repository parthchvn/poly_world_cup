#!/usr/bin/env python3
"""Upload prepared Basic/In-market SFT data, then train, evaluate and compare.

No market/wallet collection is performed. See docs/automated_experiments.md.
The standard-library controller never loads model weights. GPU work runs in
separate processes, with one reserved GPU group per experiment at a time.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from decimal import Decimal
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import queue
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'scripts'))
from world_cup_eval_common import sha, read_bundle, targets
from compare_actor_variants import split_file
from train_world_cup_multigpu import atomic_json

VARIANTS = ('basic', 'inmarket')
CODE = (
    'scripts/run_world_cup_experiments.py', 'scripts/train_world_cup_multigpu.py',
    'scripts/evaluate_world_cup.py', 'scripts/prepare_world_cup_evaluation.py',
    'scripts/compare_world_cup_evaluations.py', 'scripts/plot_training_losses.py',
    'scripts/derive_actor_metrics.py', 'tools/world_cup_eval_common.py',
    'tools/compare_actor_variants.py',
)
PINS = {
    'transformers': '5.17.0', 'tokenizers': '0.23.2', 'accelerate': '1.15.0',
    'bitsandbytes': '0.50.2', 'datasets': '5.0.1', 'peft': '0.21.0',
    'tilelang': '0.1.14', 'fla-core': '0.5.2', 'flash-linear-attention': '0.5.2',
    'causal-conv1d': '1.7.0',
}
PACKAGES = ('torch', 'triton', *PINS, 'matplotlib')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def file_lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise ValueError(f'Another runner owns {path}; do not launch the same job twice.') from None
        yield


def inventory(dataset):
    files = [dataset / 'manifest.json'] + [split_file(dataset, s) for s in ('train', 'validation', 'test')]
    return {p.name: sha(p) for p in files}


def unpack(archive, destination, max_bytes):
    """Check gzip CRC, bound expansion, and reject links/path traversal before publication."""
    digest = sha(archive)
    final = destination / digest
    if (final / '.archive.json').is_file():
        require(read_json(final / '.archive.json')['sha256'] == digest, 'Archive receipt changed')
        return final
    destination.mkdir(parents=True, exist_ok=True)
    expanded = 0
    with gzip.open(archive, 'rb') as source:
        while chunk := source.read(1024 * 1024):
            expanded += len(chunk)
            require(expanded <= max_bytes, 'Archive exceeds --max-extract-gib; upload only prepared SFT files.')
    require(shutil.disk_usage(destination).free > expanded + 512 * 1024**2,
            'Not enough disk space to extract the uploaded archive')
    work = Path(tempfile.mkdtemp(prefix='extract-', dir=destination))
    try:
        seen, extracted_bytes = set(), 0
        with tarfile.open(archive, 'r:gz') as source:
            for index, member in enumerate(source):
                require(index < 10000, 'Archive has too many entries; upload only prepared SFT files')
                name = PurePosixPath(member.name)
                require(not name.is_absolute() and '..' not in name.parts and '\\' not in member.name,
                        f'Unsafe archive path: {member.name}')
                require(member.isfile() or member.isdir(), f'Archive links/devices are not accepted: {member.name}')
                extracted_bytes += member.size
                require(not member.sparse and extracted_bytes <= max_bytes,
                        'Sparse/oversized archive member is not accepted')
                if str(name) == '.':
                    continue
                require(str(name) not in seen, f'Duplicate archive path: {member.name}')
                seen.add(str(name))
                target = work.joinpath(*name.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as inp, target.open('xb') as out:
                        shutil.copyfileobj(inp, out)
        atomic_json(work / '.archive.json', {'sha256': digest, 'expanded_bytes': expanded})
        work.rename(final)
    finally:
        if work.exists():
            shutil.rmtree(work)
    return final


def discover(inputs, archive_root, max_bytes):
    roots = []
    for source in inputs:
        require(source.exists(), f'Uploaded data not found: {source}')
        if source.is_file():
            roots.append(unpack(source, archive_root, max_bytes))
        else:
            roots.append(source)
            # Only top-level archives, never recursively unpack raw wallet caches.
            for archive in sorted(source.glob('*.tar.gz')):
                roots.append(unpack(archive, archive_root, max_bytes))
    found = {}
    for root in roots:
        for current, directories, names in os.walk(root, followlinks=False):
            here = Path(current)
            directories[:] = [d for d in directories if not d.startswith('.') and
                               d not in ('actors', 'wallet_cache', 'market_cache', 'token_cache', 'models', 'runs')]
            if len(here.relative_to(root).parts) >= 6:
                directories[:] = []
            if 'manifest.json' not in names:
                continue
            meta = read_json(here / 'manifest.json')
            if meta.get('format') != 'actor_market_trade_messages_v1':
                continue
            variant = meta.get('feature_variant')
            if variant not in VARIANTS:
                continue
            require(variant not in found or found[variant].resolve() == here.resolve(),
                    f'Multiple {variant} datasets found. Pass only the intended dataset directories with --data.')
            found[variant] = here
    require('basic' in found,
            'Need a prepared basic dataset with manifest.json and train/validation/test.jsonl[.gz]. '
            f'Found: {sorted(found)}. Raw actor exports are not prepared SFT datasets.')
    return found


def derive_inmarket(basic, destination):
    """The four execution features are computable from the already captured Basic conversations.

    No wallet/API fetch, position snapshot, current execution or future answer is
    used in a prediction's metrics. Only whole, untruncated conversations qualify.
    """
    from compare_actor_variants import scan_dataset, _chronology, _variant_manifest, lines
    from prepare_world_cup_evaluation import enrich_record
    from derive_actor_metrics import timestamp_us
    source = scan_dataset(basic)
    _chronology(source)
    _variant_manifest(source, 'basic')
    require(source['manifest'].get('targets_truncated_or_dropped') == 0,
            'Automatic In-market features require complete Basic conversations (targets_truncated_or_dropped=0).')
    config = {'version': 2, 'strict_prior': True, 'history_scope': 'actor_and_binary_market',
              'feature_variant': 'inmarket', 'selected_features': [
                  'average_execution_notional', 'execution_notional_cv', 'executions_per_day', 'buy_notional_share'],
              'lookback_seconds': None, 'min_return_periods': 30, 'metric_significant_digits': 10,
              'completed_position_ledger_supplied': False, 'capital_adjusted_returns_supplied': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='inmarket-', dir=destination.parent))
    digests = {}
    try:
        for split, path in source['paths'].items():
            digest = hashlib.sha256()
            with (work / f'{split}.jsonl.gz').open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as output:
                for _, record in lines(path):
                    groups = []
                    for offset in range(1, len(record['messages']), 2):
                        context = json.loads(record['messages'][offset]['content'])
                        trades = json.loads(record['messages'][offset + 1]['content'])['trades']
                        when = timestamp_us(context['query_time'])
                        normalized = [{**t, 'time_us': when, 'shares': Decimal(t['shares']),
                                       'price': Decimal(t['price'])} for t in trades]
                        groups.append({'time_us': when, 'expected': trades, 'trades': normalized})
                    enriched = enrich_record(record, groups, config)
                    line = (json.dumps(enriched, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
                    output.write(line)
                    digest.update(line)
            digests[split] = digest.hexdigest()
        manifest = copy.deepcopy(source['manifest'])
        manifest.update(feature_variant='inmarket', token_lengths_checked=False, max_length_checked=None,
                        split_sha256=digests, converter_sha256=sha(Path(__file__)))
        for stats in manifest['stats'].values():
            for key in ('tokens', 'max_tokens', 'p50_tokens', 'p95_tokens', 'loss_tokens'):
                stats.pop(key, None)
        manifest['actor_metrics'] = {'config': config, 'strict_prior': True,
            'source_dataset_manifest_sha256': sha(basic / 'manifest.json'),
            'source_split_sha256': source['manifest']['split_sha256'],
            'history_scope': 'actor_and_binary_market', 'target_messages_unchanged': True,
            'features_added_only_to_user_messages': True, 'actor_snapshots_used_as_model_input': False,
            'source': 'prior_executions_in_complete_basic_conversation'}
        atomic_json(work / 'manifest.json', manifest)
        scan_dataset(work)
        work.rename(destination)
    finally:
        if work.exists():
            shutil.rmtree(work)
    log('Built In-market features from earlier Basic trades; no API calls.')


def stage_data(args):
    staged = {v: args.out / 'data' / v for v in VARIANTS}
    if not args.data:
        require((staged['basic'] / 'manifest.json').is_file(),
                'First launch needs --data DIRECTORY_OR_TAR_GZ (repeat for separate uploads).')
        if not staged['inmarket'].exists():
            derive_inmarket(staged['basic'], staged['inmarket'])
        return staged
    sources = discover(args.data, args.out / 'uploads', int(args.max_extract_gib * 1024**3))
    for variant, source in sources.items():
        identity = inventory(source)
        destination = staged[variant]
        if destination.exists():
            require(inventory(destination) == identity,
                    f'{variant} data differs from this experiment. Use a fresh --out; existing data was not overwritten.')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = sum((source / name).stat().st_size for name in identity)
        require(shutil.disk_usage(destination.parent).free > total + 512 * 1024**2,
                'Not enough disk space to stage prepared datasets')
        work = Path(tempfile.mkdtemp(prefix=variant + '-', dir=destination.parent))
        try:
            for name in identity:
                shutil.copyfile(source / name, work / name)
            require(inventory(work) == identity, 'Uploaded data changed while being copied; retry after upload finishes')
            work.rename(destination)
        finally:
            if work.exists():
                shutil.rmtree(work)
    if 'inmarket' not in sources and not staged['inmarket'].exists():
        derive_inmarket(staged['basic'], staged['inmarket'])
    return staged


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def install_dependencies(args):
    if not args.install_deps:
        return
    # Never replace a working torch/CUDA installation or silently change existing versions.
    with file_lock(Path(tempfile.gettempdir()) / 'world-cup-package-install.lock'):
        code = ('import torch; assert torch.__version__.split("+")[0] == "2.8.0" '
                'and torch.version.cuda == "12.8", '
                '"--install-deps requires the PyTorch 2.8.0 / CUDA 12.8 RunPod image"')
        subprocess.run([sys.executable, '-c', code], check=True)
        versions = package_versions()
        require(versions['triton'] == '3.4.0', '--install-deps expects the existing triton==3.4.0')
        conflicts = {k: versions[k] for k, expected in PINS.items() if versions[k] not in (None, expected)}
        require(not conflicts, f'Existing packages differ from the known working environment: {conflicts}. '
                'No packages changed. Use the successful training image, or omit --install-deps to validate your own environment.')
        missing = [f'{k}=={v}' for k, v in PINS.items() if versions[k] is None and k != 'causal-conv1d']
        if versions['matplotlib'] is None:
            missing.append('matplotlib')
        with tempfile.TemporaryDirectory(prefix='wc-deps-') as temp:
            constraint = Path(temp) / 'constraints.txt'
            constraint.write_text(f'torch=={versions["torch"]}\ntriton==3.4.0\n' +
                                  ''.join(f'{key}=={value}\n' for key, value in PINS.items()))
            if missing or versions['causal-conv1d'] is None:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-c', str(constraint),
                                'setuptools', 'wheel', 'packaging', 'ninja', *missing], check=True)
            if versions['causal-conv1d'] is None:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
                                '-c', str(constraint), 'causal-conv1d==1.7.0'],
                               env={**os.environ, 'MAX_JOBS': '4'}, check=True)


def probe_environment(model, group):
    """Runs in a short-lived child so the controller keeps no CUDA context."""
    import importlib
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'datasets', 'bitsandbytes',
                 'fla.ops.gated_delta_rule', 'causal_conv1d', 'tilelang', 'matplotlib'):
        importlib.import_module(name)
    import torch
    from transformers import AutoTokenizer, AutoConfig, Qwen3_5ForCausalLM
    del Qwen3_5ForCausalLM
    AutoConfig.from_pretrained(model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    require(tokenizer.is_fast and tokenizer.chat_template, 'Missing official fast tokenizer/chat template')
    require(torch.cuda.device_count() == len(group.split(',')), 'Requested GPU group is not fully visible')
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            require(torch.cuda.is_bf16_supported(), f'GPU {index} does not support BF16')
    log('Environment, tokenizer and BF16 GPU checks passed; no model weights loaded.')


def verify_weights(model):
    require((model / 'config.json').is_file(), f'Missing model: {model}')
    shards = set()
    for index in model.glob('*.safetensors.index.json'):
        shards.update(read_json(index).get('weight_map', {}).values())
    if not shards:
        shards.update(p.name for p in model.glob('*.safetensors'))
    require(bool(shards), f'No model weight files in {model}')
    for name in shards:
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and
                (model / name).is_file() and (model / name).stat().st_size > 0,
                f'Missing/empty model shard: {name}')


def gpu_info(group):
    result = subprocess.run(['nvidia-smi', f'--id={group}', '--query-gpu=uuid,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = [(r[0].strip(), float(r[1])) for r in csv.reader(result.stdout.splitlines()) if r]
    require(len(rows) == len(group.split(',')), f'Cannot identify GPU group {group}')
    return rows


def wait_for_memory(group, minimum_gib, timeout, stop):
    deadline = time.monotonic() + timeout
    last_log = 0
    while True:
        rows = gpu_info(group)
        if all(free >= minimum_gib * 1024 for _, free in rows):
            return
        require(time.monotonic() < deadline, f'GPU group {group} is busy or has insufficient free VRAM: {rows}. '
                f'Need {minimum_gib:g} GiB free per GPU. No process was killed. Rerun when the GPUs are free.')
        if time.monotonic() - last_log >= 15:
            log(f'Waiting for GPU group {group} to become free: {rows}')
            last_log = time.monotonic()
        if stop.wait(1):
            raise InterruptedError('Runner interrupted')


def complete_checkpoint(run, gpus):
    """Select the newest nonempty checkpoint with marker, optimizer, scheduler and all RNG states."""
    candidates = sorted((p for p in run.glob('checkpoint-*') if p.name[11:].isdigit()),
                        key=lambda p: int(p.name[11:]), reverse=True)
    rng = ['rng_state.pth'] if gpus == 1 else [f'rng_state_{i}.pth' for i in range(gpus)]
    required = ['complete.json', 'trainer_state.json', 'optimizer.pt', 'scheduler.pt',
                'adapter_config.json', 'adapter_model.safetensors', *rng]
    for path in candidates:
        if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
            continue
        try:
            marker, state = read_json(path / 'complete.json'), read_json(path / 'trainer_state.json')
            step = int(path.name[11:])
            if marker.get('step') == state.get('global_step') == step and marker.get('gpus') == gpus:
                return path
        except (ValueError, OSError):
            continue
    return None


class Controller:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.active = {}
        self.mutex = threading.Lock()

    def command(self, command, name, group=None):
        if self.stop.is_set():
            raise InterruptedError('Runner interrupted')
        path = self.args.out / 'logs' / (name + '.log')
        path.parent.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
                       'FLA_TILELANG': '1', 'FLA_DISABLE_BACKEND_DISPATCH': '0',
                       'OMP_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1'}
        if group is not None:
            environment['CUDA_VISIBLE_DEVICES'] = group
        log(f'{name}: starting; log: {path}')
        with path.open('a', buffering=1) as stream:
            stream.write(f'\n[{now()}] {json.dumps(command)}\n')
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment, cwd=ROOT, start_new_session=True)
            with self.mutex:
                self.active[process.pid] = process
            try:
                heartbeat = time.monotonic()
                while process.poll() is None:
                    if self.stop.wait(1):
                        raise InterruptedError('Runner interrupted')
                    if time.monotonic() - heartbeat >= 60:
                        log(f'{name}: running; follow {path}')
                        heartbeat = time.monotonic()
                if process.returncode:
                    with path.open('rb') as source:
                        source.seek(max(0, path.stat().st_size - 3000))
                        tail = source.read().decode(errors='replace')
                    raise RuntimeError(f'{name} exited {process.returncode}. Log: {path}\n{tail}')
            finally:
                if process.poll() is None:
                    terminate_group(process)
                with self.mutex:
                    self.active.pop(process.pid, None)

    def cancel(self):
        self.stop.set()
        with self.mutex:
            processes = list(self.active.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def train_command(args, variant, group, run, dataset):
    return [sys.executable, str(ROOT / 'scripts/train_world_cup_multigpu.py'),
            '--gpus', str(len(group.split(','))), '--gpu-ids', group,
            '--model', str(args.model), '--dataset-dir', str(dataset), '--out', str(run),
            '--cache-dir', str(args.out / 'token_cache' / variant), '--seed', str(args.seed),
            '--epochs', str(args.epochs), '--global-batch', str(args.global_batch),
            '--micro-batch', '1', '--max-length', str(args.max_length),
            '--learning-rate', str(args.learning_rate), '--rank', str(args.rank),
            '--alpha', str(args.alpha), '--save-steps', str(args.save_steps),
            '--logging-steps', '1', '--no-group-by-length', '--smoke-then-full']


def check_test_lengths(args):
    from transformers import AutoTokenizer
    from train_world_cup_multigpu import CHAT_KWARGS
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    _, records = read_bundle(args.out / 'bundle')
    maximum = {}
    for variant in VARIANTS:
        lengths = []
        for record in records[variant]:
            for target in targets(record):
                rendered = tokenizer.apply_chat_template(target['messages'], tokenize=False,
                    add_generation_prompt=True, **CHAT_KWARGS)
                length = len(tokenizer(rendered, add_special_tokens=False, truncation=False)['input_ids'])
                require(length + args.max_new_tokens <= args.max_context,
                        f'{variant} test prompt {target["id"]}: {length} + {args.max_new_tokens} exceeds '
                        f'--max-context {args.max_context}. No targets removed.')
                lengths.append(length)
        maximum[variant] = max(lengths)
    log(f'Test prompt lengths checked before training (no generation): {maximum}')
    atomic_json(args.out / 'test_lengths.json', maximum)


def initialize(args, controller):
    from prepare_world_cup_evaluation import prepare, parse_args as eval_args, references
    from evaluate_world_cup import validate_run
    install_dependencies(args)
    versions = package_versions()
    require(all(versions.values()), 'Missing packages: ' + ', '.join(k for k, v in versions.items() if not v) +
            '. On the PyTorch 2.8.0/CUDA 12.8 image, rerun with --install-deps.')
    verify_weights(args.model)
    log('Checking uploaded datasets and paired splits; this uses CPU only.')
    with file_lock(args.out / 'locks/prepare.lock'):
        datasets = stage_data(args)
        config = {'format': 'world_cup_experiment_v1', 'model': str(args.model),
            'model_config_sha256': sha(args.model / 'config.json'), 'packages': versions,
            'data': {v: inventory(p) for v, p in datasets.items()},
            'code': {p: sha(ROOT / p) for p in CODE},
            'settings': {k: getattr(args, k) for k in ('seed', 'epochs', 'global_batch', 'max_length',
                'learning_rate', 'rank', 'alpha', 'save_steps', 'max_context', 'max_new_tokens')},
            'training_gpus': len(args.gpu_group[0].split(',')),
            'runs': {v: str(getattr(args, v + '_run') or args.out / 'runs' / v) for v in VARIANTS}}
        identity = args.out / 'experiment.json'
        if identity.exists():
            require(read_json(identity) == config,
                    'Experiment data/code/model/packages/settings changed. Restore the original checkout and flags '
                    'to resume, or use a fresh --out. Nothing was retrained or overwritten.')
        else:
            require(not (args.out / 'runs').exists() and not (args.out / 'evaluation').exists(),
                    'This output already contains runs/results without an experiment identity. '
                    'Use a fresh --out and --basic-run/--inmarket-run for completed external runs.')
            atomic_json(identity, config)
        bundle = args.out / 'bundle'
        if not bundle.exists():
            prepare(eval_args(['--basic-sft', str(datasets['basic']), '--inmarket-sft', str(datasets['inmarket']),
                              '--out', str(bundle)]))
        else:
            meta, _ = read_bundle(bundle)
            checked, _, _, _ = references(datasets['basic'], datasets['inmarket'])
            for variant in VARIANTS:
                for split in ('train', 'validation'):
                    require(meta['reference'][variant]['source_sha256'][split] == sha(checked[variant]['paths'][split]),
                            'Existing evaluation bundle differs from these training datasets')
                require(meta['reference'][variant]['manifest_sha256'] == sha(datasets[variant] / 'manifest.json'),
                        'Existing evaluation bundle manifest differs')
        meta, _ = read_bundle(bundle)
        for variant in VARIANTS:
            external = getattr(args, variant + '_run')
            if external:
                validate_run(external, args.model, variant, meta)
        # Both training datasets and both test prompt lengths must pass before any expensive training.
        if not (args.out / 'preparation_complete.json').exists():
            for variant in VARIANTS:
                if not getattr(args, variant + '_run'):
                    command = train_command(args, variant, args.gpu_group[0], args.out / 'runs' / variant, datasets[variant])
                    controller.command([*command, '--prepare-only'], f'prepare_{variant}')
            controller.command([sys.executable, str(Path(__file__).resolve()), '--out', str(args.out),
                '--model', str(args.model), '--max-context', str(args.max_context),
                '--max-new-tokens', str(args.max_new_tokens), '--internal', 'test-lengths'], 'check_test_lengths')
            atomic_json(args.out / 'preparation_complete.json', {'completed_at': now(), 'experiment_sha256': sha(identity)})
        require(read_json(args.out / 'preparation_complete.json')['experiment_sha256'] == sha(identity),
                'Preparation receipt differs from this experiment')
    return datasets, config


def status(args, variant, stage, **extra):
    atomic_json(args.out / 'status' / (variant + '.json'),
                {'variant': variant, 'stage': stage, 'updated_at': now(), **extra})
    log(f'{variant}: {stage}')


def run_variant(args, controller, variant, group, datasets, config):
    from evaluate_world_cup import validate_run
    from compare_world_cup_evaluations import load_result
    run = Path(config['runs'][variant])
    output = args.out / 'evaluation' / variant
    owned = False
    try:
        with file_lock(args.out / 'locks' / (variant + '.lock'), blocking=False):
            owned = True
            bundle, _ = read_bundle(args.out / 'bundle')
            completed = (run / 'training_metadata.json').is_file() and read_json(run / 'training_metadata.json').get('status') == 'completed'
            if not completed:
                require(not getattr(args, variant + '_run'), 'Reused runs must be completed; this runner does not resume external training.')
                command = train_command(args, variant, group, run, datasets[variant])
                checkpoint = complete_checkpoint(run, len(group.split(',')))
                if checkpoint:
                    command.extend(['--resume', str(checkpoint)])
                    log(f'{variant}: resuming {checkpoint}')
                elif run.exists() and any(run.iterdir()):
                    backup = args.out / 'unfinished_attempts' / f'{variant}-{time.time_ns()}'
                    backup.parent.mkdir(exist_ok=True)
                    run.rename(backup)
                    log(f'{variant}: no complete checkpoint. Preserved unfinished attempt at {backup}; restarting training.')
                wait_for_memory(group, args.min_free_gib, args.gpu_wait_seconds, controller.stop)
                status(args, variant, 'training', log=str(args.out / 'logs' / f'train_{variant}.log'))
                controller.command(command, f'train_{variant}', group)
            validate_run(run, args.model, variant, bundle)
            status(args, variant, 'training_completed', adapter=str(run / 'adapter'))
            # Make the finished training curve available before the longer generation evaluation.
            # Matplotlib has process-global state: render in an isolated CPU subprocess.
            controller.command([sys.executable, str(ROOT / 'scripts/plot_training_losses.py'),
                                '--run-dir', str(run), '--output', str(args.out / 'plots' / f'{variant}_loss.png'),
                                '--title', variant], f'plot_{variant}')
            if (output / 'summary.json').is_file():
                identity, _ = load_result(output, variant)
                require(identity['bundle_sha256'] == sha(args.out / 'bundle/manifest.json') and
                        identity['adapter_sha256'] == sha(run / 'adapter/adapter_model.safetensors') and
                        identity['training_metadata_sha256'] == sha(run / 'training_metadata.json'),
                        'Saved evaluation no longer matches the bundle/adapter/training run')
                require(identity['limit'] == 0 and identity['selected_targets'] == bundle['targets'] and
                        identity['evaluator_sha256'] == sha(ROOT / 'scripts/evaluate_world_cup.py') and
                        identity['common_sha256'] == sha(ROOT / 'tools/world_cup_eval_common.py') and
                        identity['decoding'] == {'do_sample': False, 'num_beams': 1,
                            'max_new_tokens': args.max_new_tokens, 'max_context': args.max_context,
                            'seed': 42, 'attention': 'sdpa'} and
                        all(config['packages'].get(k) == v for k, v in identity['versions'].items()),
                        'Saved evaluation code/settings/packages/target count differ from this experiment')
                log(f'{variant}: completed predictions verified and reused')
            else:
                # Training has exited before evaluation is started. Reserve the entire group until evaluation exits.
                wait_for_memory(group, args.min_free_gib, args.gpu_wait_seconds, controller.stop)
                status(args, variant, 'evaluating', log=str(args.out / 'logs' / f'evaluate_{variant}.log'))
                controller.command([sys.executable, str(ROOT / 'scripts/evaluate_world_cup.py'),
                    '--bundle', str(args.out / 'bundle'), '--variant', variant, '--run-dir', str(run),
                    '--model', str(args.model), '--out', str(output), '--gpu', '0',
                    '--max-context', str(args.max_context), '--max-new-tokens', str(args.max_new_tokens), '--resume'],
                    f'evaluate_{variant}', group.split(',')[0])
                load_result(output, variant)
            status(args, variant, 'completed', adapter=str(run / 'adapter'), results=str(output))
        return True
    except Exception as error:
        if owned:
            status(args, variant, 'failed', error=str(error))
        log(f'{variant}: {error}\nRerun the same command to resume; completed work is retained.')
        return False


def write_report(args, config):
    from compare_world_cup_evaluations import compare
    from plot_training_losses import read_rows, loss_series
    if not all((args.out / 'evaluation' / v / 'summary.json').is_file() for v in VARIANTS):
        log('The paired report will be built automatically by the last experiment to finish.')
        return False
    with file_lock(args.out / 'locks/report.lock'):
        report = compare(args.out / 'evaluation/basic', args.out / 'evaluation/inmarket')
        atomic_json(args.out / 'comparison.json', report)
        with (args.out / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['metric', 'basic', 'inmarket'])
            for key in report['metrics']['basic']:
                writer.writerow([key, *(report['metrics'][v].get(key) for v in VARIANTS)])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        try:
            for variant, color in zip(VARIANTS, ('#2563eb', '#ea580c')):
                series = loss_series(read_rows(Path(config['runs'][variant]) / 'metrics.jsonl'))
                for ax, key in zip(axes, ('loss', 'eval_loss')):
                    if series[key]:
                        x, y = zip(*series[key])
                        ax.plot(x, y, label=variant, color=color, linewidth=1.2,
                                marker='o' if key == 'eval_loss' else None, markersize=3)
                if series['smoke_eval_loss']:
                    x, y = zip(*series['smoke_eval_loss'])
                    axes[1].scatter(x, y, color=color, marker='D', label=variant + ': smoke subset')
            for ax, title in zip(axes, ('Training loss', 'Validation loss')):
                ax.set(title=title, xlabel='Optimizer step', ylabel='Assistant-token cross-entropy')
                ax.grid(alpha=0.2)
                ax.legend()
            figure.tight_layout()
            (args.out / 'plots').mkdir(exist_ok=True)
            for suffix in ('png', 'pdf'):
                figure.savefig(args.out / 'plots' / ('loss_comparison.' + suffix), dpi=180)
        finally:
            plt.close(figure)
        rows = ['# World Cup experiment comparison', '',
                '| Metric | Basic | In-market |', '|---|---:|---:|']
        for key in report['delta_inmarket_minus_basic']:
            rows.append(f'| {key} | {report["metrics"]["basic"][key]:.2%} | {report["metrics"]["inmarket"][key]:.2%} |')
        rows += ['', '![Loss curves](plots/loss_comparison.png)', '',
                 'See comparison.json for paired numeric errors on the same correctly classified targets,',
                 'per-match scores, training-setting differences and any match-cluster intervals.', '',
                 *['- ' + note for note in report['notes']]]
        (args.out / 'REPORT.md').write_text('\n'.join(rows) + '\n')
        atomic_json(args.out / 'completed.json', {'status': 'completed', 'completed_at': now(),
                    'comparison_sha256': sha(args.out / 'comparison.json')})
        log(f'All experiments complete. Report: {args.out / "REPORT.md"}')
    return True


def run(args):
    controller = Controller(args)
    try:
        datasets, config = initialize(args, controller)
        if args.check_only:
            log('Data, identities and all train/validation/test lengths passed. No weights loaded.')
            return 0
        todo = queue.Queue()
        for variant in VARIANTS if args.variant == 'both' else (args.variant,):
            todo.put(variant)

        def worker(group):
            from contextlib import ExitStack
            results = []
            # Local physical-UUID locks also protect against other runner output directories.
            with ExitStack() as locks:
                requested = VARIANTS if args.variant == 'both' else (args.variant,)
                needs_gpu = not all((args.out / 'evaluation' / v / 'summary.json').exists() and
                    (Path(config['runs'][v]) / 'training_metadata.json').exists() and
                    read_json(Path(config['runs'][v]) / 'training_metadata.json').get('status') == 'completed'
                    for v in requested)
                if needs_gpu:
                    for uuid, _ in sorted(gpu_info(group)):
                        locks.enter_context(file_lock(Path(tempfile.gettempdir()) / f'world-cup-{uuid}.lock', blocking=False))
                    controller.command([sys.executable, str(Path(__file__).resolve()), '--out', str(args.out),
                                        '--model', str(args.model), '--gpu-group', group,
                                        '--global-batch', str(args.global_batch), '--internal', 'probe'],
                                       'probe_' + group.replace(',', '_'), group)
                while not controller.stop.is_set():
                    try:
                        variant = todo.get_nowait()
                    except queue.Empty:
                        break
                    results.append(run_variant(args, controller, variant, group, datasets, config))
            return all(results)

        executor = ThreadPoolExecutor(max_workers=len(args.gpu_group))
        try:
            futures = [executor.submit(worker, group) for group in args.gpu_group]
            results = [f.result() for f in as_completed(futures)]
        except BaseException:
            controller.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        report_ready = write_report(args, config)
        return 0 if all(results) and (report_ready or args.variant != 'both') else 1
    except BaseException:
        controller.cancel()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, action='append', help='Prepared Basic data directory or .tar.gz; optionally repeat for an existing In-market upload')
    parser.add_argument('--out', type=Path, required=True, help='Persistent experiment directory; same path/flags to resume')
    parser.add_argument('--model', type=Path, default=Path('/workspace/models/Qwen3.6-27B'))
    parser.add_argument('--variant', choices=('both', *VARIANTS), default='both')
    parser.add_argument('--gpu-group', action='append', help='Physical GPU IDs for one experiment, e.g. 0,1; repeat for parallel groups')
    parser.add_argument('--basic-run', type=Path, help='Reuse an already COMPLETED Basic training run')
    parser.add_argument('--inmarket-run', type=Path, help='Reuse an already COMPLETED In-market training run')
    parser.add_argument('--install-deps', action='store_true', help='Install missing pinned packages on torch2.8/CUDA12.8; never replace existing versions')
    parser.add_argument('--background', action='store_true', help='Detach from terminal; logs persist under --out/logs')
    parser.add_argument('--status', action='store_true', help='Print saved stage status; no GPU/packages/data required')
    parser.add_argument('--check-only', action='store_true', help='Validate data and tokenize all splits; do not load weights')
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--global-batch', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=16384)
    parser.add_argument('--max-context', type=int, default=32768)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=int, default=32)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--min-free-gib', type=float, default=60, help='Free VRAM gate per GPU; default targets idle 80GB GPUs')
    parser.add_argument('--gpu-wait-seconds', type=float, default=60, help='Bounded wait for GPU memory release; never kill other jobs')
    parser.add_argument('--max-extract-gib', type=float, default=10)
    parser.add_argument('--internal', choices=('probe', 'test-lengths'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ('out', 'model', 'basic_run', 'inmarket_run'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    args.data = [p.expanduser().resolve() for p in args.data or []]
    groups = args.gpu_group or ['0,1']
    parsed = [group.split(',') for group in groups]
    require(all(all(x.isdigit() for x in group) for group in parsed), 'GPU groups must contain comma-separated physical integer IDs')
    parsed = [[str(int(x)) for x in group] for group in parsed]
    flat = [x for group in parsed for x in group]
    require(len(flat) == len(set(flat)), 'GPU groups overlap or repeat a device')
    require(len({len(group) for group in parsed}) == 1, 'Use the same GPU count for both training experiments')
    require(len(groups) <= 2 and (args.variant == 'both' or len(groups) == 1), 'Use at most one GPU group per requested experiment')
    args.gpu_group = [','.join(group) for group in parsed]
    for key in ('epochs', 'global_batch', 'max_length', 'max_context', 'max_new_tokens', 'learning_rate',
                'rank', 'alpha', 'save_steps', 'min_free_gib', 'max_extract_gib'):
        require(math.isfinite(getattr(args, key)) and getattr(args, key) > 0, f'--{key.replace("_", "-")} must be positive and finite')
    require(math.isfinite(args.gpu_wait_seconds) and args.gpu_wait_seconds >= 0, 'Invalid GPU wait budget')
    require(args.max_context > args.max_new_tokens, 'Generation budget must be below max context')
    require(args.global_batch % len(parsed[0]) == 0, '--global-batch must be divisible by GPUs per training group')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.internal == 'probe':
        probe_environment(args.model, args.gpu_group[0])
        return 0
    if args.internal == 'test-lengths':
        check_test_lengths(args)
        return 0
    if args.status:
        for path in sorted((args.out / 'status').glob('controller-*.json')):
            print(json.dumps(read_json(path), indent=2))
        for variant in VARIANTS:
            path = args.out / 'status' / (variant + '.json')
            print(json.dumps(read_json(path) if path.exists() else {'variant': variant, 'stage': 'not_started'}, indent=2))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.background:
        original = list(sys.argv[1:] if argv is None else argv)
        original.remove('--background')
        path = args.out / 'logs' / f'runner-{time.time_ns()}.log'
        path.parent.mkdir(exist_ok=True)
        with path.open('a') as output:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), *original],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        log(f'Background runner PID {process.pid}. Follow: tail -f {path}')
        return 0
    with file_lock(args.out / 'locks' / f'controller-{args.variant}.lock', blocking=False):
        path = args.out / 'status' / f'controller-{args.variant}.json'
        state = {'variant': args.variant, 'pid': os.getpid(), 'host': os.uname().nodename}
        atomic_json(path, {**state, 'stage': 'preparing', 'updated_at': now()})
        try:
            result = run(args)
        except BaseException as error:
            atomic_json(path, {**state, 'stage': 'failed', 'updated_at': now(),
                              'error': str(error) or type(error).__name__})
            raise
        stage = 'checked' if args.check_only else 'completed' if (args.out / 'completed.json').exists() else 'waiting_for_other_variant'
        atomic_json(path, {**state, 'stage': stage if result == 0 else 'failed', 'updated_at': now(), 'exit_code': result})
        return result


if __name__ == '__main__':
    def stop_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_signal)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log('Interrupted. Complete checkpoints and saved predictions are retained. Rerun the same command.')
        sys.exit(130)
    except (ValueError, OSError, RuntimeError, EOFError, tarfile.TarError, subprocess.CalledProcessError) as error:
        log(f'ERROR: {error}')
        sys.exit(1)
