#!/usr/bin/env python3
"""Score a completed interval SFT adapter, explicitly and separately from training.

For each prospective [start, end) interval, compute the likelihood of BOTH
canonical assistant completions (including their EOS). Normalize those two
likelihoods to obtain P(TRADE | prompt, answer is one of these two completions).
No length normalization, generation sampling, future context, or test fitting.
The result measures captured execution activity, not conscious abstention.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_cup_multigpu import (CHAT_KWARGS, INTERVAL_PROTOCOL, atomic_json,
    encode_conversation, kernel_probe, selected_positions, sha256_file,
    split_path, validate_interval_record)

ACTIONS = ("NO_TRADE", "TRADE")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_answer(action):
    return json.dumps({"action": action}, separators=(",", ":"))


def read_records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def normalized_trade_probability(no_trade_loglik, trade_loglik):
    require(math.isfinite(no_trade_loglik) and math.isfinite(trade_loglik),
            "Nonfinite completion log likelihood")
    difference = trade_loglik - no_trade_loglik
    if difference >= 0:
        return 1.0 / (1.0 + math.exp(-difference))
    exponential = math.exp(difference)
    return exponential / (1.0 + exponential)


def encode_candidates(record, tokenizer, max_context):
    """Reuse the TRAINER's exact chat template and assistant/EOS token mask."""
    require(record.get('target_protocol') == INTERVAL_PROTOCOL,
            'Binary likelihood scoring requires activity targets; use evaluate_interval_trade_details.py for fill targets')
    validate_interval_record(record)
    gold = json.loads(record['messages'][-1]['content'])['action']
    require(record['messages'][-1]['content'].strip() == canonical_answer(gold),
            'Interval targets must use the canonical compact action JSON used by the builder')
    result = []
    for action in ACTIONS:
        candidate = {**record, "messages": [*record['messages'][:-1],
            {"role": "assistant", "content": canonical_answer(action)}]}
        encoded, count = encode_conversation(candidate, tokenizer, max_context,
                                             f"{record['row_id']} {action}")
        require(count == 1, 'Exactly one assistant target is required')
        result.append(encoded)
    # Both candidates must share exactly the same conditioning token sequence.
    prefixes = [x['input_ids'][:next(i for i, y in enumerate(x['labels']) if y != -100)]
                for x in result]
    require(prefixes[0] == prefixes[1], 'Candidate tokenization changes the conditioning prefix')
    return result


def score_candidates(model, encoded, tokenizer, torch, device):
    """One forward pass for the two candidates, projecting only target positions."""
    length = max(len(row['input_ids']) for row in encoded)
    ids, masks, labels = [], [], []
    for row in encoded:
        padding = length - len(row['input_ids'])
        ids.append(row['input_ids'] + [tokenizer.pad_token_id] * padding)
        masks.append(row['attention_mask'] + [0] * padding)
        labels.append(row['labels'] + [-100] * padding)
    inputs = torch.tensor(ids, dtype=torch.long, device=device)
    attention = torch.tensor(masks, dtype=torch.long, device=device)
    targets = torch.tensor(labels, dtype=torch.long, device=device)
    positions = selected_positions(targets)
    shifted = targets[:, 1:].index_select(1, positions)
    with torch.inference_mode():
        output = model(input_ids=inputs, attention_mask=attention,
                       logits_to_keep=positions, use_cache=False)
        require(output.logits.shape[:2] == shifted.shape, 'Likelihood token alignment failure')
        losses = torch.nn.functional.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
            shifted.reshape(-1), ignore_index=-100, reduction='none').reshape(shifted.shape)
        likelihoods = (-losses.sum(dim=1)).cpu().tolist()
    return likelihoods


def validate_run(run_dir, dataset_dir, model_path, split='test', protocol=INTERVAL_PROTOCOL):
    metadata = json.loads((run_dir / 'training_metadata.json').read_text())
    require(metadata.get('status') == 'completed' and metadata.get('mode') in ('full', 'smoke_then_full'),
            'Use a completed full training run with its final adapter')
    require(metadata.get('test_used') is False, 'Training must declare test_used=false')
    manifest_path = dataset_dir / 'manifest.json'
    require(sha256_file(manifest_path) == metadata['signature']['data'].get('manifest_sha256'),
            'Dataset manifest differs from the frozen manifest recorded before training')
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('target_protocol') == protocol, f'Manifest does not use the requested {protocol} protocol')
    evaluation_path = split_path(dataset_dir, split)
    require(sha256_file(evaluation_path) == manifest.get('files', {}).get(evaluation_path.name),
            f'{split} file differs from the frozen manifest recorded before training')
    for split in ('train', 'validation'):
        require(set(metadata['data'][split].get('protocol_counts', {})) == {protocol},
                f'{split} was not trained with the prospective interval protocol')
        actual = sha256_file(split_path(dataset_dir, split))
        require(actual == metadata['signature']['data']['sources'][split]['sha256'],
                f'{split} differs from the training run; use its original selected dataset directory')
    require(sha256_file(model_path / 'config.json') == metadata['signature']['data']['model_config_sha256'],
            'Base model config differs from training')
    adapter = run_dir / 'adapter'
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json'):
        require((adapter / name).is_file(), f'Missing final adapter file {name}')
    return metadata, adapter


def recover_predictions(path, records, resume):
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
                'Prediction journal does not match these evaluation prompts')
        require(row['label'] == int(json.loads(record['messages'][-1]['content'])['action'] == 'TRADE'),
                'Prediction journal does not match these evaluation labels')
    return rows


def summarize(rows, train_prevalence, threshold):
    from rank_interval_features import binary_metrics
    labels = [row['label'] for row in rows]
    probabilities = [row['probability_trade'] for row in rows]
    result = {'model': binary_metrics(labels, probabilities, threshold=threshold),
              'baselines': {
                  'always_no_trade': binary_metrics(labels, [0.] * len(rows), threshold=threshold),
                  'train_frequency': binary_metrics(labels, [train_prevalence] * len(rows), threshold=threshold)}}
    for key in ('actor_id', 'fixture_id', 'actor_seen_in_training'):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        result['by_' + key] = {group: binary_metrics([r['label'] for r in members],
            [r['probability_trade'] for r in members], threshold=threshold) for group, members in sorted(groups.items())}
        if key in ('actor_id', 'fixture_id'):
            # Mean per-person/per-match loss gives equal weight to each group.
            result['macro_' + key] = {metric: sum(values) / len(values) if values else None
                for metric in ('log_loss', 'brier_score', 'accuracy', 'f1')
                for values in [[m[metric] for m in result['by_' + key].values() if m.get(metric) is not None]]}
    return result


def run(args):
    require(args.max_context > 0 and args.limit >= 0 and 0 < args.threshold < 1, 'Invalid limits/threshold')
    require(os.environ.get('WORLD_SIZE', '1') == '1', 'Run one evaluator with python, not torchrun')
    metadata, adapter = validate_run(args.run_dir, args.dataset_dir, args.model, args.split)
    evaluation_path = split_path(args.dataset_dir, args.split)
    all_records = list(read_records(evaluation_path))
    require(bool(all_records), 'Empty evaluation split')
    identifiers = set()
    train_fixtures = set(metadata['data']['train']['fixtures'])
    validation_fixtures = set(metadata['data']['validation']['fixtures'])
    for record in all_records:
        validate_interval_record(record)
        require(record['row_id'] not in identifiers, 'Duplicate evaluation row_id')
        identifiers.add(record['row_id'])
        excluded = train_fixtures | (validation_fixtures if args.split == 'test' else set())
        require(str(record['fixture_id']) not in excluded, 'Evaluation match was present in training/validation')
    records = all_records[:args.limit] if args.limit else all_records
    train_actors = {str(row['actor_id']) for row in read_records(split_path(args.dataset_dir, 'train'))}
    counts = metadata['data']['train']['action_counts']
    train_prevalence = counts.get('TRADE', 0) / sum(counts.values())
    os.environ.setdefault('FLA_TILELANG', '1')
    os.environ.setdefault('FLA_DISABLE_BACKEND_DISPATCH', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, use_fast=True)
    require(tokenizer.is_fast and tokenizer.chat_template, 'Need the saved fast chat tokenizer')
    require(hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() ==
            metadata['signature']['data']['tokenizer'], 'Tokenizer differs from training')
    require(tokenizer.chat_template == metadata['signature']['data']['template'], 'Chat template differs from training')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    print(f'Checking both completion lengths for {len(records):,} intervals before loading weights', flush=True)
    maximum = 0
    for record in records:
        encoded = encode_candidates(record, tokenizer, args.max_context)
        maximum = max(maximum, *(len(row['input_ids']) for row in encoded))
    print(f'Maximum candidate sequence: {maximum:,} tokens; no truncation', flush=True)
    if args.check_only:
        return
    import torch
    from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM, set_seed
    from peft import PeftModel, prepare_model_for_kbit_training
    require(torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count(), 'Requested CUDA GPU unavailable')
    torch.cuda.set_device(args.gpu)
    require(torch.cuda.is_bf16_supported(), 'BF16 CUDA GPU required')
    require('logits_to_keep' in inspect.signature(Qwen3_5ForCausalLM.forward).parameters,
            'The installed model class must support selected logits, as used by the trainer')
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'tokenizers', 'fla-core', 'causal-conv1d'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity = {'protocol': INTERVAL_PROTOCOL, 'split': args.split, 'source_sha256': sha256_file(evaluation_path),
        'adapter_sha256': sha256_file(adapter / 'adapter_model.safetensors'),
        'training_metadata_sha256': sha256_file(args.run_dir / 'training_metadata.json'),
        'evaluator_sha256': sha256_file(__file__), 'trainer_sha256': sha256_file(Path(__file__).with_name('train_world_cup_multigpu.py')),
        'metrics_sha256': sha256_file(Path(__file__).with_name('rank_interval_features.py')), 'versions': versions,
        'selected_ids_sha256': hashlib.sha256(dump([row['row_id'] for row in records]).encode()).hexdigest(),
        'threshold': args.threshold, 'limit': args.limit, 'max_context': args.max_context,
        'candidate_actions': list(ACTIONS), 'candidate_scoring': 'sum_assistant_and_EOS_log_likelihood_then_softmax',
        'conditioning': 'restricted_to_the_two_canonical_completions', 'train_prevalence': train_prevalence}
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
        rows = recover_predictions(journal, records, args.resume)
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
            device = torch.device('cuda', args.gpu)
            initial, started = len(rows), time.monotonic()
            with journal.open('a', buffering=1) as stream:
                for index in range(initial, len(records)):
                    record = records[index]
                    encoded = encode_candidates(record, tokenizer, args.max_context)
                    likelihoods = score_candidates(model, encoded, tokenizer, torch, device)
                    probability = normalized_trade_probability(*likelihoods)
                    row = {key: record[key] for key in ('row_id', 'actor_id', 'fixture_id', 'market_id', 'interval_start', 'interval_end')}
                    row.update(label=int(json.loads(record['messages'][-1]['content'])['action'] == 'TRADE'),
                        probability_trade=probability, predicted_action='TRADE' if probability >= args.threshold else 'NO_TRADE',
                        completion_log_likelihood=dict(zip(ACTIONS, likelihoods)),
                        actor_seen_in_training=str(record['actor_id']) in train_actors,
                        prompt_sha256=hashlib.sha256(dump(record['messages'][:-1]).encode()).hexdigest())
                    stream.write(dump(row) + '\n')
                    rows.append(row)
                    if (index + 1) % 25 == 0 or index + 1 == len(records):
                        stream.flush()
                        os.fsync(stream.fileno())
                        print(f'{index+1}/{len(records)} intervals; {(time.monotonic()-started)/(index+1-initial):.2f}s/interval', flush=True)
        summary = {'status': 'completed', 'protocol': INTERVAL_PROTOCOL, 'split': args.split,
            'pilot': len(records) < len(all_records), 'threshold': args.threshold,
            'probability_semantics': identity['conditioning'], 'metrics': summarize(rows, train_prevalence, args.threshold),
            'predictions_sha256': sha256_file(journal)}
        atomic_json(args.out / 'summary.json', summary)
        print(json.dumps({key: summary[key] for key in ('status', 'split', 'pilot', 'threshold')}, indent=2))
        print(json.dumps(summary['metrics']['model'], indent=2))
        print(f'Complete metrics, baselines and per-actor/per-match results: {args.out / "summary.json"}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True, help='Same selected SFT dataset used in training')
    parser.add_argument('--run-dir', type=Path, required=True, help='Completed run containing adapter/ and training_metadata.json')
    parser.add_argument('--model', type=Path, required=True, help='Same local Qwen model directory used in training')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-context', type=int, default=8192)
    parser.add_argument('--threshold', type=float, default=.5, help='Fix before opening test; defaults to 0.5')
    parser.add_argument('--limit', type=int, default=0, help='Positive count is an explicitly labeled pilot prefix')
    parser.add_argument('--check-only', action='store_true', help='Validate/tokenize; do not load model weights')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    try:
        run(parse_args())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(str(exc))
