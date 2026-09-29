#!/usr/bin/env python3
"""Evaluate one completed Basic or In-market adapter on the paired holdout.

Run separately on each pod/GPU. No fitting, API calls, package installation,
model downloads, truncation, or feeding the current target to generation.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'scripts'))
from world_cup_eval_common import dump, sha, write_json, read_bundle, targets, summarize, score, require
from train_world_cup_multigpu import CHAT_KWARGS, kernel_probe


def validate_run(run, model_path, variant, bundle):
    metadata = json.loads((run / 'training_metadata.json').read_text())
    # --smoke-then-full is a full run with an early validation check. The trainer
    # preserves that mode name after all optimization and final saving complete.
    status, mode = metadata.get('status'), metadata.get('mode')
    require(status == 'completed' and mode in ('full', 'smoke_then_full'),
            'Use a COMPLETED full or smoke-then-full training run, not a smoke '
            f'run/checkpoint or an active run (status={status!r}, mode={mode!r})')
    require(metadata.get('test_used') is False, 'Training run does not declare test_used=false')
    signature = metadata['signature']
    reference = bundle['reference'][variant]
    for split in ('train', 'validation'):
        require(signature['data']['sources'][split]['sha256'] == reference['source_sha256'][split],
                f'{variant} {split} file differs from the data used by this adapter')
        require(metadata['data'][split]['sha256'] == reference['split_sha256'][split],
                f'{variant} {split} uncompressed contents differ from training')
        require(set(metadata['data'][split]['fixtures']).isdisjoint(bundle['fixtures']),
                'Evaluation fixture was seen during training/validation')
    require(sha(model_path / 'config.json') == signature['data']['model_config_sha256'],
            'Base model configuration differs from training')
    adapter = run / 'adapter'
    for file in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json'):
        require((adapter / file).is_file(), f'Missing final adapter file: {adapter / file}')
    return metadata, adapter


def select_targets(records, limit):
    result = [t for record in records for t in targets(record)]
    return result[:limit] if limit else result


def recover_predictions(path, expected, resume):
    if not path.exists():
        return []
    require(resume, 'Predictions already exist; use --resume or a new --out')
    raw = path.read_bytes()
    # Only discard an incomplete LAST journal line left by a killed process.
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n') + 1]
        with path.open('wb') as f:
            f.write(raw)
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) <= len(expected), 'Journal has more targets than this evaluation')
    for row, target in zip(rows, expected):
        require(row['id'] == target['id'] and row['answer'] == target['answer'] and
                row['prompt_sha256'] == hashlib.sha256(dump(target['messages']).encode()).hexdigest(),
                'Journal target or prompt mismatch')
    return rows


def run(args):
    require(args.max_context > args.max_new_tokens > 0 and args.limit >= 0, 'Invalid token limits')
    require(not os.environ.get('WORLD_SIZE') or os.environ['WORLD_SIZE'] == '1',
            'Use python, not torchrun: one inference process per invocation')
    # The GPU index is relative to this process's currently visible devices.
    os.environ.setdefault('FLA_TILELANG', '1')
    os.environ.setdefault('FLA_DISABLE_BACKEND_DISPATCH', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    bundle, records = read_bundle(args.bundle)
    metadata, adapter = validate_run(args.run_dir, args.model, args.variant, bundle)
    selected = select_targets(records[args.variant], args.limit)
    require(bool(selected), 'No evaluation targets')
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig, Qwen3_5ForCausalLM, set_seed
    from peft import PeftModel, prepare_model_for_kbit_training
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, use_fast=True)
    require(tokenizer.is_fast and tokenizer.chat_template, 'Need the saved fast chat tokenizer')
    tokenizer_sha = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    require(tokenizer_sha == metadata['signature']['data']['tokenizer'], 'Tokenizer differs from training')
    require(tokenizer.chat_template == metadata['signature']['data']['template'], 'Chat template differs from training')
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    require(eos is not None and tokenizer.convert_ids_to_tokens(eos) == '<|im_end|>', 'Missing Qwen EOS')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos
    print(f'Checking {len(selected):,} prompt lengths before loading weights', flush=True)
    encoded = []
    for target in selected:
        # Generation receives ONLY the prefix ending in the current USER query.
        text = tokenizer.apply_chat_template(target['messages'], tokenize=False,
                                            add_generation_prompt=True, **CHAT_KWARGS)
        ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        require(len(ids) + args.max_new_tokens <= args.max_context,
                f"Target {target['id']}: {len(ids)} prompt tokens + {args.max_new_tokens} generation "
                f"budget exceeds --max-context {args.max_context}; no targets were dropped")
        encoded.append(ids)
    print(f'Max prompt: {max(map(len, encoded)):,} tokens. No weights loaded yet.', flush=True)
    if args.check_only:
        return
    require(torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count(), 'Requested CUDA GPU unavailable')
    torch.cuda.set_device(args.gpu)
    require(torch.cuda.is_bf16_supported(), 'BF16 CUDA GPU required')
    versions = {n: importlib.metadata.version(n) for n in
                ('torch', 'transformers', 'peft', 'bitsandbytes', 'tokenizers', 'fla-core', 'causal-conv1d')}
    identity = {'format': 'world_cup_predictions_v1', 'variant': args.variant,
        'bundle_sha256': sha(args.bundle / 'manifest.json'), 'target_sha256': bundle['target_sha256'],
        'adapter_sha256': sha(adapter / 'adapter_model.safetensors'),
        'adapter_config_sha256': sha(adapter / 'adapter_config.json'),
        'training_metadata_sha256': sha(args.run_dir / 'training_metadata.json'),
        'training_signature': metadata['signature'],
        'evaluator_sha256': sha(Path(__file__)),
        'common_sha256': sha(ROOT / 'tools/world_cup_eval_common.py'),
        'tokenizer_sha256': tokenizer_sha, 'versions': versions,
        'decoding': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': args.max_new_tokens,
                     'max_context': args.max_context, 'seed': 42, 'attention': 'sdpa'},
        'limit': args.limit, 'selected_targets': len(selected),
        'selected_ids_sha256': hashlib.sha256(dump([t['id'] for t in selected]).encode()).hexdigest(),
        'fixtures': bundle['fixtures'], 'history_protocol': bundle['history_protocol']}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another evaluator is using this output directory')
        identity_path = args.out / 'identity.json'
        if identity_path.exists():
            require(args.resume and json.loads(identity_path.read_text()) == identity,
                    'Output exists or identity differs. Use --resume with identical settings or a new --out')
        else:
            require(not any(p.name != 'evaluation.lock' for p in args.out.iterdir()), 'Output directory is not empty')
            write_json(identity_path, identity)
        journal = args.out / 'predictions.jsonl'
        rows = recover_predictions(journal, selected, args.resume)
        if len(rows) < len(selected):
            kernel_probe(torch, args.gpu)
            torch.cuda.empty_cache()
            set_seed(42)
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            print('Loading frozen base weights and final adapter once', flush=True)
            base = Qwen3_5ForCausalLM.from_pretrained(args.model, local_files_only=True,
                        dtype=torch.bfloat16, quantization_config=quant,
                        device_map={'': args.gpu}, attn_implementation='sdpa')
            # Match the nonquantized parameter casting performed by the trainer.
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
            model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
            model.eval()
            model.config.use_cache = True
            generation_config = GenerationConfig(do_sample=False, num_beams=1,
                num_return_sequences=1, max_new_tokens=args.max_new_tokens,
                eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id, repetition_penalty=1.0, use_cache=True)
            device = torch.device('cuda', args.gpu)
            start = time.monotonic()
            initial = len(rows)
            with journal.open('a', buffering=1) as stream:
                for index in range(initial, len(selected)):
                    target, ids = selected[index], encoded[index]
                    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                    with torch.inference_mode():
                        output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                            generation_config=generation_config, logits_to_keep=1)
                    new_ids = output[0, len(ids):].tolist()
                    prediction = tokenizer.decode(new_ids, skip_special_tokens=True)
                    row = {k: v for k, v in target.items() if k != 'messages'}
                    row.update(prediction=prediction, prompt_tokens=len(ids), generated_tokens=len(new_ids),
                        prompt_sha256=hashlib.sha256(dump(target['messages']).encode()).hexdigest(),
                        hit_generation_limit=len(new_ids) == args.max_new_tokens and new_ids[-1] != eos)
                    # Validate gold immediately; invalid model output remains a scored failure.
                    score(row['answer'], prediction)
                    stream.write(dump(row) + '\n')
                    stream.flush()
                    rows.append(row)
                    if (index + 1) % 10 == 0 or index + 1 == len(selected):
                        os.fsync(stream.fileno())
                        elapsed = time.monotonic() - start
                        rate = elapsed / (index + 1 - initial)
                        print(f'{index + 1}/{len(selected)} targets; {rate:.2f}s/target; '
                              f'rough remaining {(len(selected)-index-1)*rate/60:.1f} min', flush=True)
        summary = {'status': 'completed', 'variant': args.variant,
            'pilot': len(selected) < bundle['targets'], 'metrics': summarize(rows),
            'predictions_sha256': sha(journal), 'identity_sha256': sha(identity_path)}
        write_json(args.out / 'summary.json', summary)
        print(json.dumps(summary, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--variant', required=True, choices=('basic', 'inmarket'))
    p.add_argument('--run-dir', required=True, type=Path, help='Completed run containing adapter/ and training_metadata.json')
    p.add_argument('--model', type=Path, default=Path('/workspace/models/Qwen3.6-27B'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max-context', type=int, default=32768)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--limit', type=int, default=0, help='0 evaluates all; positive value is a pilot prefix')
    p.add_argument('--check-only', action='store_true', help='Check identities and tokenize; do not load weights')
    p.add_argument('--resume', action='store_true')
    return p.parse_args(argv)


if __name__ == '__main__':
    try:
        run(parse_args())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'Error: {exc}')
