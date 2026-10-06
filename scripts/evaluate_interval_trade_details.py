#!/usr/bin/env python3
"""Generate interval actions and trade aggregates; score frozen numeric tolerances.

For a prospective [start,end) interval, predict NO_TRADE or each side/outcome's
total shares and share-weighted mean execution price. Tolerances are acceptance
criteria for these point predictions, read from the manifest frozen at training.
No test-time tolerance choices, random decoding, truncation, or future context.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
from train_world_cup_multigpu import (CHAT_KWARGS, INTERVAL_DETAILS_PROTOCOL, atomic_json,
    encode_conversation, kernel_probe, sha256_file, split_path, validate_interval_record)
from evaluate_interval_decisions import dump, read_records, require, validate_run
from interval_trade_tolerances import (score_trade_details, summarize_trade_details,
                                       validate_tolerances)

DETAIL_SEMANTICS = 'side_outcome_window_aggregates_v1'


def frozen_tolerances(manifest):
    require(manifest.get('trade_detail_semantics') == DETAIL_SEMANTICS,
            'Expected side/outcome window aggregates in the frozen manifest')
    require(manifest.get('implementation_sha256', {}).get('tools/interval_trade_tolerances.py') ==
            sha256_file(ROOT / 'tools/interval_trade_tolerances.py'),
            'Trade-details scoring implementation differs from the frozen dataset manifest')
    return validate_tolerances(manifest['trade_tolerances'])


def encode_generation_prompt(record, tokenizer, max_context, max_new_tokens):
    require(record.get('target_protocol') == INTERVAL_DETAILS_PROTOCOL,
            'This evaluator requires prospective trade-details targets')
    validate_interval_record(record)
    messages = record['messages'][:-1]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, **CHAT_KWARGS)
    ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
    require(len(ids) + max_new_tokens <= max_context,
            f"{record['row_id']}: prompt plus generation budget exceeds --max-context; no truncation allowed")
    # Check the saved target's template for alignment only. No answer token enters
    # the returned prompt or model.generate, and its length does not set the budget.
    training, _ = encode_conversation(record, tokenizer, max_context, record['row_id'])
    start = next(i for i, label in enumerate(training['labels']) if label != -100)
    require(ids == training['input_ids'][:start],
            'Generation prompt does not match the conditioning prefix used during training')
    return ids


def score_prediction(gold, prediction, tolerances, terminated):
    if not terminated:
        result = score_trade_details(gold, '__UNTERMINATED_GENERATION__', tolerances)
        result['parse_error'] = 'Generation did not terminate with the assistant end marker'
        return result
    return score_trade_details(gold, prediction, tolerances)


def recover_predictions(path, records, resume, tolerances):
    if not path.exists():
        return []
    require(resume, 'Predictions exist; use --resume or a new output directory')
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n') + 1]
        path.write_bytes(raw)
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) <= len(records), 'Journal contains excess predictions')
    for row, record in zip(rows, records):
        require(row['row_id'] == record['row_id'] and
            row['prompt_sha256'] == hashlib.sha256(dump(record['messages'][:-1]).encode()).hexdigest(),
            'Prediction journal does not match evaluation prompts')
        require(row['gold'] == json.loads(record['messages'][-1]['content']),
                'Prediction journal does not match evaluation labels')
        rescored = score_prediction(row['gold'], row['prediction'], tolerances, row['terminated_with_eos'])
        require(row['score'] == rescored, 'Prediction journal score differs from the frozen scorer/tolerances')
    return rows


def summarize(rows, tolerances):
    result = {'model': summarize_trade_details([row['score'] for row in rows]),
        'baselines': {'always_no_trade': summarize_trade_details([
            score_trade_details(row['gold'], {'action': 'NO_TRADE'}, tolerances) for row in rows])}}
    for key in ('actor_id', 'fixture_id', 'actor_seen_in_training'):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row['score'])
        result['by_' + key] = {name: summarize_trade_details(scores) for name, scores in sorted(groups.items())}
        if key in ('actor_id', 'fixture_id'):
            result['macro_' + key] = {}
            for metric in ('action_accuracy', 'exact_interval_match_rate', 'trade_window_exact_match_rate', 'trade_f1'):
                values = [group[metric] for group in result['by_' + key].values() if group.get(metric) is not None]
                result['macro_' + key][metric] = sum(values)/len(values) if values else None
    result['terminated_with_eos_rate'] = sum(row['terminated_with_eos'] for row in rows)/len(rows)
    result['generation_limit_rate'] = sum(row['hit_generation_limit'] for row in rows)/len(rows)
    return result


def run(args):
    require(args.max_context > args.max_new_tokens > 0 and args.limit >= 0, 'Invalid token limits')
    require(os.environ.get('WORLD_SIZE', '1') == '1', 'Run with python, not torchrun')
    metadata, adapter = validate_run(args.run_dir, args.dataset_dir, args.model, args.split,
                                     protocol=INTERVAL_DETAILS_PROTOCOL)
    manifest = json.loads((args.dataset_dir / 'manifest.json').read_text())
    tolerances = frozen_tolerances(manifest)
    evaluation_path = split_path(args.dataset_dir, args.split)
    all_records = list(read_records(evaluation_path))
    require(bool(all_records), 'Empty evaluation split')
    excluded = set(metadata['data']['train']['fixtures'])
    if args.split == 'test':
        excluded |= set(metadata['data']['validation']['fixtures'])
    row_ids = set()
    for record in all_records:
        require(record.get('target_protocol') == INTERVAL_DETAILS_PROTOCOL, 'Mixed evaluation protocols')
        validate_interval_record(record)
        require(record['row_id'] not in row_ids, 'Duplicate evaluation row_id')
        row_ids.add(record['row_id'])
        require(str(record['fixture_id']) not in excluded, 'Evaluation match overlaps training/validation')
    records = all_records[:args.limit] if args.limit else all_records
    train_actors = {str(row['actor_id']) for row in read_records(split_path(args.dataset_dir, 'train'))}
    os.environ.setdefault('FLA_TILELANG', '1')
    os.environ.setdefault('FLA_DISABLE_BACKEND_DISPATCH', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, use_fast=True)
    require(tokenizer.is_fast and tokenizer.chat_template, 'Need the saved fast chat tokenizer')
    require(hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() ==
            metadata['signature']['data']['tokenizer'], 'Tokenizer differs from training')
    require(tokenizer.chat_template == metadata['signature']['data']['template'], 'Chat template differs from training')
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    require(eos is not None and tokenizer.convert_ids_to_tokens(eos) == '<|im_end|>', 'Missing assistant end marker')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos
    print(f'Checking {len(records):,} generation prefixes before loading weights', flush=True)
    maximum = 0
    for record in records:
        ids = encode_generation_prompt(record, tokenizer, args.max_context, args.max_new_tokens)
        maximum = max(maximum, len(ids))
    print(f'Maximum prompt: {maximum:,} tokens; generation budget: {args.max_new_tokens:,}', flush=True)
    if args.check_only:
        return
    import torch
    from transformers import BitsAndBytesConfig, GenerationConfig, Qwen3_5ForCausalLM, set_seed
    from peft import PeftModel, prepare_model_for_kbit_training
    require(torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count(), 'Requested CUDA GPU unavailable')
    torch.cuda.set_device(args.gpu)
    require(torch.cuda.is_bf16_supported(), 'BF16 CUDA GPU required')
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'tokenizers', 'fla-core', 'causal-conv1d'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity = {'protocol': INTERVAL_DETAILS_PROTOCOL, 'trade_detail_semantics': DETAIL_SEMANTICS,
        'trade_tolerances': tolerances, 'split': args.split, 'source_sha256': sha256_file(evaluation_path),
        'manifest_sha256': sha256_file(args.dataset_dir / 'manifest.json'),
        'adapter_sha256': sha256_file(adapter / 'adapter_model.safetensors'),
        'training_metadata_sha256': sha256_file(args.run_dir / 'training_metadata.json'),
        'evaluator_sha256': sha256_file(__file__), 'trainer_sha256': sha256_file(ROOT / 'scripts/train_world_cup_multigpu.py'),
        'scorer_sha256': sha256_file(ROOT / 'tools/interval_trade_tolerances.py'), 'versions': versions,
        'selected_ids_sha256': hashlib.sha256(dump([row['row_id'] for row in records]).encode()).hexdigest(),
        'decoding': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': args.max_new_tokens,
            'max_context': args.max_context, 'seed': 42, 'attention': 'sdpa'}, 'limit': args.limit}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another evaluator is using this output directory')
        identity_path = args.out / 'identity.json'
        if identity_path.exists():
            require(args.resume and json.loads(identity_path.read_text()) == identity,
                    'Output identity differs; use an empty directory or identical --resume settings')
        else:
            require(not any(p.name != 'evaluation.lock' for p in args.out.iterdir()), 'Output directory is not empty')
            atomic_json(identity_path, identity)
        journal = args.out / 'predictions.jsonl'
        rows = recover_predictions(journal, records, args.resume, tolerances)
        if len(rows) < len(records):
            kernel_probe(torch, args.gpu)
            torch.cuda.empty_cache()
            set_seed(42)
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            base = Qwen3_5ForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16,
                quantization_config=quant, device_map={'': args.gpu}, attn_implementation='sdpa')
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
            model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
            model.eval()
            model.config.use_cache = True
            generation = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
                max_new_tokens=args.max_new_tokens, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id, repetition_penalty=1.0, use_cache=True)
            device = torch.device('cuda', args.gpu)
            initial, started = len(rows), time.monotonic()
            with journal.open('a', buffering=1) as stream:
                for index in range(initial, len(records)):
                    record = records[index]
                    ids = encode_generation_prompt(record, tokenizer, args.max_context, args.max_new_tokens)
                    inputs = torch.tensor([ids], dtype=torch.long, device=device)
                    with torch.inference_mode():
                        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            generation_config=generation, logits_to_keep=1)
                    new_ids = output[0, len(ids):].tolist()
                    terminated = bool(new_ids and new_ids[-1] == eos)
                    prediction = tokenizer.decode(new_ids, skip_special_tokens=True)
                    gold = json.loads(record['messages'][-1]['content'])
                    row = {key: record[key] for key in ('row_id', 'actor_id', 'fixture_id', 'market_id', 'interval_start', 'interval_end')}
                    row.update(gold=gold, prediction=prediction,
                        score=score_prediction(gold, prediction, tolerances, terminated),
                        terminated_with_eos=terminated, hit_generation_limit=len(new_ids) >= args.max_new_tokens and not terminated,
                        generated_tokens=len(new_ids), prompt_tokens=len(ids),
                        actor_seen_in_training=str(record['actor_id']) in train_actors,
                        prompt_sha256=hashlib.sha256(dump(record['messages'][:-1]).encode()).hexdigest())
                    stream.write(dump(row) + '\n')
                    rows.append(row)
                    if (index+1) % 10 == 0 or index+1 == len(records):
                        stream.flush()
                        os.fsync(stream.fileno())
                        print(f'{index+1}/{len(records)} intervals; {(time.monotonic()-started)/(index+1-initial):.2f}s/interval', flush=True)
        summary = {'status': 'completed', 'protocol': INTERVAL_DETAILS_PROTOCOL, 'split': args.split,
            'trade_detail_semantics': DETAIL_SEMANTICS, 'trade_tolerances': tolerances,
            'pilot': len(records) < len(all_records), 'metrics': summarize(rows, tolerances),
            'predictions_sha256': sha256_file(journal)}
        atomic_json(args.out / 'summary.json', summary)
        print(json.dumps(summary['metrics']['model'], indent=2))
        print(f'Frozen-tolerance metrics, baseline and groups: {args.out / "summary.json"}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-context', type=int, default=8192)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    try:
        run(parse_args())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(str(exc))
