#!/usr/bin/env python3
"""Resumable, sharded extraction of frozen SFT context vectors, then CPU probes.

No test data, generated answers, price targets, or gradients enter extraction.
The last input token's final hidden state is the single predefined representation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_interval_decisions import read_records, require, validate_run
from evaluate_interval_trade_details import encode_generation_prompt, frozen_tolerances
from train_world_cup_multigpu import (INTERVAL_DETAILS_PROTOCOL, atomic_json,
                                    kernel_probe, sha256_file, split_path)

SPLITS = ('train', 'validation')


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + ' | ' + message, flush=True)


def save_npz(path, **arrays):
    import numpy as np
    temporary = path.with_suffix('.npz.partial')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def shard_records(path, index, count):
    return [row for i, row in enumerate(read_records(path)) if i % count == index]


def identifiers(records):
    import numpy as np
    return {
        'row_ids': np.asarray([r['row_id'] for r in records], dtype=str),
        'labels': np.asarray([int(json.loads(r['messages'][-1]['content'])['action'] == 'TRADE')
                              for r in records], dtype=np.int8),
        'fixture_ids': np.asarray([str(r['fixture_id']) for r in records], dtype=str),
        'actor_ids': np.asarray([str(r['actor_id']) for r in records], dtype=str),
    }


def validate_chunk(path, expected, hidden_size):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        require(saved['X'].shape == (len(expected), hidden_size), f'Wrong embedding shape: {path}')
        require(np.isfinite(saved['X']).all(), f'Nonfinite embeddings: {path}')
        for key, value in identifiers(expected).items():
            require(np.array_equal(saved[key], value), f'Embedding {key} mismatch: {path}')
        return saved['X'].copy()


def final_prefix_vectors(decoder, sequences, pad_id, device, torch):
    """Right padding; select the final real input token, never an answer token."""
    lengths = [len(s) for s in sequences]
    require(all(lengths), 'Empty prompt')
    width = max(lengths)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, :len(seq)] = 1
    result = decoder(input_ids=ids, attention_mask=mask, use_cache=False,
                     output_hidden_states=False, output_attentions=False, return_dict=True)
    hidden = result.last_hidden_state
    vectors = hidden[torch.arange(len(lengths), device=device),
                     torch.tensor(lengths, device=device) - 1]
    require(bool(torch.isfinite(vectors).all()), 'Nonfinite frozen representations')
    return vectors.float().cpu().numpy()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'peft', 'bitsandbytes', 'tokenizers',
             'fla-core', 'causal-conv1d', 'numpy')}


def build_identity(args):
    metadata, adapter = validate_run(args.run_dir, args.dataset_dir, args.model,
                                     'validation', protocol=INTERVAL_DETAILS_PROTOCOL)
    frozen_tolerances(json.loads((args.dataset_dir / 'manifest.json').read_text()))
    counts = {s: int(metadata['data'][s]['conversations']) for s in SPLITS}
    require(all(n >= len(args.gpu_ids) for n in counts.values()), 'Too few rows for extraction shards')
    return {
        'format': 'frozen_activity_embeddings_v1', 'test_used': False,
        'input_sha256': {s: sha256_file(split_path(args.dataset_dir, s)) for s in SPLITS},
        'splits': {s: {'rows': counts[s]} for s in SPLITS},
        'dataset_manifest_sha256': sha256_file(args.dataset_dir / 'manifest.json'),
        'metadata_sha256': sha256_file(args.run_dir / 'training_metadata.json'),
        'adapter_sha256': sha256_file(adapter / 'adapter_model.safetensors'),
        'adapter_config_sha256': sha256_file(adapter / 'adapter_config.json'),
        'base_config_sha256': sha256_file(args.model / 'config.json'),
        'base_model': str(args.model), 'run_dir': str(args.run_dir),
        'dataset_dir': str(args.dataset_dir),
        'scripts': {p: sha256_file(ROOT / 'scripts' / p) for p in
                    ('run_frozen_activity_probe.py', 'evaluate_interval_trade_details.py',
                     'evaluate_interval_decisions.py', 'train_world_cup_multigpu.py')},
        'scorer_sha256': sha256_file(ROOT / 'tools/interval_trade_tolerances.py'),
        'versions': versions(), 'batch_size': args.batch_size, 'chunk_size': args.chunk_size,
        'max_length': args.max_length, 'shards': len(args.gpu_ids),
        'representation': 'final_decoder_hidden_state_at_last_generation_prefix_token',
        'model_frozen': True, 'adapter_frozen': True, 'quantization': 'NF4_double_BF16',
        'padding': 'right', 'attention': 'sdpa', 'feature_dtype': 'float16',
        'padding_parity_relative_l2_limit': 0.02,
    }


def extract(args):
    import numpy as np
    identity_path = args.out / 'embeddings/identity.json'
    identity = json.loads(identity_path.read_text())
    require(0 <= args.worker < identity['shards'], 'Invalid extraction shard')
    require(all(getattr(args, key) == identity[key] for key in ('batch_size', 'chunk_size', 'max_length'))
            and len(args.gpu_ids) == identity['shards'], 'Worker settings differ from saved identity')
    require(identity['scripts']['run_frozen_activity_probe.py'] == sha256_file(__file__),
            'Extractor changed during run')
    shard_dir = args.out / 'embeddings' / f'shard{args.worker}'
    shard_dir.mkdir(parents=True, exist_ok=True)
    with (shard_dir / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Re-read only the two development files. Test is never opened.
        records = {}
        for split in SPLITS:
            log(f'shard {args.worker}: checking and reading {split} input')
            path = split_path(args.dataset_dir, split)
            require(sha256_file(path) == identity['input_sha256'][split], 'Source changed during extraction')
            records[split] = shard_records(path, args.worker, identity['shards'])
            expected_n = len(range(args.worker, identity['splits'][split]['rows'], identity['shards']))
            require(len(records[split]) == expected_n, 'Source row count differs from metadata')
            log(f'shard {args.worker}: {split} {expected_n:,} contexts')
        completed = shard_dir / 'complete.json'
        if completed.exists():
            state = json.loads(completed.read_text())
            require(state['identity_sha256'] == sha256_file(identity_path), 'Shard identity changed')
            for split in SPLITS:
                path = shard_dir / f'{split}.npz'
                require(state['sha256'][split] == sha256_file(path), 'Completed embedding file changed')
                validate_chunk(path, records[split], state['hidden_size'])
            log(f'shard {args.worker}: completed extraction verified and reused')
            return

        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM, set_seed
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Worker needs exactly one visible GPU')
        torch.cuda.set_device(0)
        require(torch.cuda.is_bf16_supported(), 'BF16 required')
        adapter = args.run_dir / 'adapter'
        require(sha256_file(adapter / 'adapter_model.safetensors') == identity['adapter_sha256'], 'Adapter changed')
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, use_fast=True)
        metadata = json.loads((args.run_dir / 'training_metadata.json').read_text())
        require(tokenizer.is_fast and tokenizer.chat_template, 'Need the saved fast chat tokenizer')
        require(hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() ==
                metadata['signature']['data']['tokenizer'], 'Tokenizer changed')
        require(tokenizer.chat_template == metadata['signature']['data']['template'], 'Chat template changed')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        set_seed(42)
        log(f'shard {args.worker}: checking CUDA kernels')
        kernel_probe(torch, 0)
        torch.cuda.empty_cache()
        log(f'shard {args.worker}: loading frozen base and saved adapter')
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = Qwen3_5ForCausalLM.from_pretrained(args.model, local_files_only=True,
            dtype=torch.bfloat16, quantization_config=quant, device_map={'': 0}, attn_implementation='sdpa')
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
        model.requires_grad_(False)
        model.eval()
        require(not any(p.requires_grad for p in model.parameters()), 'All parameters must be frozen')
        require(set(model.peft_config) == {'default'} and
                str(model.peft_config['default'].peft_type).split('.')[-1] == 'LORA', 'Only a single LoRA adapter is supported')
        # The LoRA layers are inside this decoder. Calling it avoids the vocabulary
        # projection entirely, unlike output_hidden_states on the CausalLM wrapper.
        decoder = model.get_base_model().model
        require(type(model.get_base_model()).__name__ == 'Qwen3_5ForCausalLM' and
                type(decoder).__name__ == 'Qwen3_5TextModel', 'Unexpected Qwen backbone')
        require(not getattr(model.peft_config['default'], 'alora_invocation_tokens', None),
                'Activated LoRA is not supported')
        lora_layers = [m for m in decoder.modules() if hasattr(m, 'lora_A') and 'default' in m.lora_A]
        require(lora_layers and all(not m.disable_adapters and 'default' in m.active_adapters
                                   for m in lora_layers), 'Saved adapter is not active')
        hidden_size = int(decoder.config.hidden_size)
        log(f'shard {args.worker}: frozen model loaded; hidden size {hidden_size}; no LM logits')
        # Hybrid attention/convolution padding must preserve each real prefix.
        # Use short/long contexts from a fixed training-only sample and compare
        # batched extraction with independent unpadded forward passes.
        if args.batch_size > 1:
            log(f'shard {args.worker}: checking padded versus individual context vectors')
            candidates = [encode_generation_prompt(r, tokenizer, args.max_length + 2048, 2048)
                          for r in records['train'][:16]]
            sequences = [min(candidates, key=len), max(candidates, key=len)]
            require(len(sequences[0]) != len(sequences[1]), 'Padding check needs unequal prompt lengths; use --batch-size 1')
            with torch.inference_mode():
                batched = final_prefix_vectors(decoder, sequences, tokenizer.pad_token_id, 'cuda:0', torch)
                individual = np.concatenate([final_prefix_vectors(decoder, [s], tokenizer.pad_token_id,
                                                                    'cuda:0', torch) for s in sequences])
            relative = np.linalg.norm(batched - individual, axis=1) / np.maximum(np.linalg.norm(individual, axis=1), 1e-8)
            limit = identity['padding_parity_relative_l2_limit']
            require(bool(np.all(relative <= limit)),
                    f'Padding parity failed ({relative}); rerun with --batch-size 1 and a new --out')
            atomic_json(shard_dir / 'padding_check.json', {'relative_l2': relative.tolist(), 'limit': limit,
                                                         'lengths': list(map(len, sequences)), 'test_used': False})
            log(f'shard {args.worker}: padding check passed, relative L2 {relative.tolist()}')
        started, extracted = time.monotonic(), 0
        for split in SPLITS:
            rows = records[split]
            chunks_dir = shard_dir / f'{split}_chunks'
            chunks_dir.mkdir(exist_ok=True)
            chunks = []
            for start in range(0, len(rows), args.chunk_size):
                batch_rows = rows[start:start + args.chunk_size]
                path = chunks_dir / f'{start:08d}.npz'
                if path.exists():
                    validate_chunk(path, batch_rows, hidden_size)
                else:
                    vectors = []
                    for offset in range(0, len(batch_rows), args.batch_size):
                        current = batch_rows[offset:offset + args.batch_size]
                        sequences = [encode_generation_prompt(r, tokenizer,
                            args.max_length + 2048, 2048) for r in current]
                        with torch.inference_mode():
                            vectors.append(final_prefix_vectors(decoder, sequences,
                                tokenizer.pad_token_id, 'cuda:0', torch).astype(np.float16))
                        extracted += len(current)
                    save_npz(path, X=np.concatenate(vectors), **identifiers(batch_rows))
                chunks.append(path)
                n = start + len(batch_rows)
                rate = extracted / max(time.monotonic() - started, 1e-9)
                log(f'shard {args.worker} {split}: {n:,}/{len(rows):,} ({100*n/len(rows):.1f}%); {rate:.2f} new contexts/s')
                atomic_json(shard_dir / 'progress.json', {'split': split, 'done': n,
                    'total': len(rows), 'new_contexts_per_second': rate, 'test_used': False})
            arrays = [validate_chunk(p, rows[i*args.chunk_size:(i+1)*args.chunk_size], hidden_size)
                      for i, p in enumerate(chunks)]
            save_npz(shard_dir / f'{split}.npz', X=np.concatenate(arrays), **identifiers(rows))
            del arrays
        atomic_json(completed, {'identity_sha256': sha256_file(identity_path), 'hidden_size': hidden_size,
            'sha256': {s: sha256_file(shard_dir / f'{s}.npz') for s in SPLITS}, 'test_used': False})
        log(f'shard {args.worker}: COMPLETE')


def ensure_gpus_free(ids):
    """Check memory, not the often-empty container process table. Never stop jobs."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    usage = {}
    for line in result.stdout.splitlines():
        index, memory = line.split(',')
        usage[index.strip()] = int(memory.strip())
    require(all(i in usage for i in ids), 'Requested GPU is not present')
    busy = {i: usage[i] for i in ids if usage[i] > 512}
    require(not busy, f'Requested GPUs have memory allocated (MiB): {busy}. No jobs stopped or launched.')


def supervise(args):
    require('0' not in args.gpu_ids, 'GPU 0 is reserved for the ongoing 0.8B job')
    # One host-local lock per physical GPU protects overlapping launcher sets,
    # even before the first model allocation becomes visible to nvidia-smi.
    with ExitStack() as locks:
        for gpu in sorted(args.gpu_ids, key=int):
            lock = locks.enter_context((Path('/tmp') / f'wc_frozen_probe_gpu{gpu}.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / 'launcher.lock').open('a') as out_lock:
            fcntl.flock(out_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            log('Verifying completed adapter and frozen train/validation files; test unopened')
            identity = build_identity(args)
            embeddings = args.out / 'embeddings'
            embeddings.mkdir(exist_ok=True)
            path = embeddings / 'identity.json'
            if path.exists():
                require(json.loads(path.read_text()) == identity, 'Embedding identity differs. Use original settings or a new --out.')
            else:
                require(not any(embeddings.iterdir()), 'Unidentified extraction output')
                atomic_json(path, identity)
            logs = args.out / 'logs'
            logs.mkdir(exist_ok=True)
            pending = [i for i in range(len(args.gpu_ids)) if not (embeddings / f'shard{i}/complete.json').exists()]
            if pending:
                ensure_gpus_free([args.gpu_ids[i] for i in pending])
            jobs = []
            def stop_workers(signum, frame):
                for job in jobs:
                    if job.poll() is None:
                        job.terminate()
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, stop_workers)
            try:
                for shard, gpu in enumerate(args.gpu_ids):
                    command = [sys.executable, '-u', str(Path(__file__).resolve()),
                        '--dataset-dir', str(args.dataset_dir), '--features-dir', str(args.features_dir),
                        '--xgb-dir', str(args.xgb_dir), '--run-dir', str(args.run_dir),
                        '--model', str(args.model), '--out', str(args.out),
                        '--gpu-ids', ','.join(args.gpu_ids), '--batch-size', str(args.batch_size),
                        '--chunk-size', str(args.chunk_size), '--max-length', str(args.max_length), '--worker', str(shard)]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1',
                        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', FLA_TILELANG='1',
                        FLA_DISABLE_BACKEND_DISPATCH='0', TRITON_CACHE_DIR=str(Path.home() / '.cache' / f'triton_probe_gpu{gpu}'))
                    with (logs / f'gpu{gpu}.log').open('a') as stream:
                        jobs.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                            env=env, stdin=subprocess.DEVNULL))
                    log(f'GPU {gpu}: worker {jobs[-1].pid}; log {logs / ("gpu"+gpu+".log")}')
                while any(j.poll() is None for j in jobs):
                    require(all(j.poll() in (None, 0) for j in jobs), 'Extraction worker failed; inspect GPU logs, then rerun to resume saved chunks')
                    time.sleep(2)
                require(all(j.returncode == 0 for j in jobs), 'Extraction failed; inspect GPU logs')
            finally:
                for job in jobs:
                    if job.poll() is None:
                        job.terminate()
                for job in jobs:
                    try:
                        job.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        job.kill()
                        job.wait()
            log('Extraction complete. Fitting CPU classifiers; GPU workers have exited.')
            command = [sys.executable, '-u', str(ROOT / 'scripts/fit_frozen_activity_probe.py'),
                '--embeddings-dir', str(embeddings), '--dataset-dir', str(args.dataset_dir),
                '--features-dir', str(args.features_dir), '--xgb-dir', str(args.xgb_dir),
                '--out', str(args.out / 'comparison')]
            fit_job = subprocess.Popen(command)
            jobs.append(fit_job)
            try:
                require(fit_job.wait() == 0, 'CPU comparison failed; see launcher log')
            finally:
                if fit_job.poll() is None:
                    fit_job.terminate()
                    try:
                        fit_job.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        fit_job.kill()
                        fit_job.wait()
            log(f'COMPLETE: {args.out / "comparison/report.json"}; validation only, not a new held-out result')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, default=Path('/workspace/wc_shared_intervals_v2/sft/selected'))
    p.add_argument('--features-dir', type=Path, default=Path('/workspace/wc_shared_intervals_v2/dataset'))
    p.add_argument('--xgb-dir', type=Path, default=Path('/workspace/wc_shared_intervals_v2/xgboost'))
    p.add_argument('--run-dir', type=Path, default=Path('/workspace/runs/wc_first_5gpu_v1/Qwen3.5-9B_retry1'))
    p.add_argument('--model', type=Path, default=Path('/workspace/models/Qwen3.5-9B'))
    p.add_argument('--out', type=Path, default=Path('/workspace/runs/wc_frozen9b_probe_v1'))
    p.add_argument('--gpu-ids', default='1,2,3,4')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--chunk-size', type=int, default=256)
    p.add_argument('--max-length', type=int, default=8192)
    p.add_argument('--worker', type=int, default=-1, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.gpu_ids = a.gpu_ids.split(',')
    require(all(x.isdecimal() for x in a.gpu_ids) and len(set(a.gpu_ids)) == len(a.gpu_ids), 'Use distinct numeric GPU IDs')
    require(a.batch_size > 0 and a.chunk_size > 0 and a.max_length > 0, 'Positive sizes required')
    for key in ('dataset_dir', 'features_dir', 'xgb_dir', 'run_dir', 'model', 'out'):
        setattr(a, key, getattr(a, key).expanduser().resolve())
    return a


if __name__ == '__main__':
    args = parse_args()
    try:
        extract(args) if args.worker >= 0 else supervise(args)
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
