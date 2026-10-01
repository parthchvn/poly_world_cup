#!/usr/bin/env python3
"""Prepare on a Mac, then test unseen-wallet activity on an idle GPU. No API fetches.

Subcommands: prepare, evaluate, run (both sequentially), report (CPU only), compare.
This future-window diagnostic is a different task from predicting trade details.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'scripts'))
import actor_activity_common as common
from actor_activity_common import dump, sha, write_json, require, digest, read_bundle, summarize, probability
from evaluate_world_cup import validate_run
from train_world_cup_multigpu import CHAT_KWARGS, kernel_probe


def encode_prompts(tokenizer, records, max_context):
    # One token per answer eliminates variable answer-length and generation-limit effects.
    choices = [tokenizer(answer, add_special_tokens=False)['input_ids'] for answer in ('A', 'B')]
    require(all(len(c) == 1 for c in choices) and choices[0] != choices[1],
            'Saved tokenizer must encode A and B as distinct single tokens')
    encoded = []
    for row in records:
        prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                                               add_generation_prompt=True, **CHAT_KWARGS)
        ids = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
        require(len(ids) <= max_context, f"{row['id']}: {len(ids)} > --max-context {max_context}; no rows truncated")
        encoded.append(ids)
    return encoded, [c[0] for c in choices]


def recover(path, prompts, labels, resume):
    if not path.exists():
        return []
    require(resume, 'Existing predictions; use --resume or a fresh output')
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n') + 1]
        path.write_bytes(raw)
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) <= len(labels), 'Journal exceeds selected targets')
    for row, prompt, label in zip(rows, prompts, labels):
        require(row['id'] == label['id'] and row['label'] == label['label']
                and row['prompt_sha256'] == digest(prompt['messages']), 'Resume prompt/label mismatch')
        require(math.isfinite(row['trade_score']) and 0 <= row['trade_score'] <= 1, 'Invalid journal score')
    return rows


def write_summary(meta, rows, variant, out, *, offline=False):
    """Reporting never changes the predictions or their original inference identity."""
    truth = [r['label'] for r in rows]
    summary = {'status': 'completed', 'variant': variant, 'pilot': len(rows) < meta['targets'],
        'task': meta['task'], 'metrics': summarize(truth, [r['trade_score'] for r in rows]),
        'baselines': {'always_no_trade': summarize(truth, [0.0] * len(rows)),
                      'prior_repeat_event_rate': summarize(truth, [r['prior_rate_score'] for r in rows])},
        'mean_unconstrained_choice_mass': sum(probability(r['unconstrained_choice_mass']) for r in rows) / len(rows),
        'predictions_sha256': sha(out / 'predictions.jsonl'), 'identity_sha256': sha(out / 'identity.json'),
        'probability_calibrated': False, 'limitations': meta['limitations'],
        'retrospectively_filtered_actor_cohort': meta['retrospectively_filtered_actor_cohort'],
        'postprocessing': {'offline_recovery': offline, 'script_sha256': sha(Path(__file__)),
                           'common_sha256': sha(ROOT / 'tools/actor_activity_common.py')}}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def report(args):
    """Finalize complete legacy/current journals without torch, tokenizers or weights.

    Inference code hashes remain untouched. We validate the frozen bundle and
    every saved target/prompt, then write a separately versioned report.
    """
    meta, records = read_bundle(args.bundle)
    paths = ([args.results] if (args.results / 'identity.json').is_file() else
             [args.results / v for v in ('basic', 'inmarket') if (args.results / v / 'identity.json').is_file()])
    require(paths, f'No saved activity prediction identities under {args.results}')
    completed, pending = {}, []
    print('Reporting saved predictions on CPU. No model packages, weights or GPU will be loaded.', flush=True)
    for path in paths:
        with (path / 'evaluation.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f'{path}: evaluator is still active; left untouched', flush=True)
                pending.append(str(path))
                continue
            identity = json.loads((path / 'identity.json').read_text())
            variant = identity.get('variant')
            require(identity.get('format') == 'world_cup_activity_scores_v1' and variant in ('basic', 'inmarket'),
                    f'{path}: unsupported inference identity')
            require(variant not in completed, 'Duplicate variant result directories')
            require(identity.get('bundle_sha256') == sha(args.bundle / 'manifest.json') and
                    identity.get('target_sha256') == meta['target_sha256'], 'Saved predictions belong to a different bundle')
            limit = identity.get('limit')
            require(type(limit) is int and 0 <= limit <= meta['targets'], 'Invalid saved evaluation limit')
            labels, prompts = records['labels'][:limit or meta['targets']], records[variant][:limit or meta['targets']]
            require(identity.get('selected_ids_sha256') == digest([r['id'] for r in labels]), 'Saved target selection changed')
            require(identity.get('score_protocol') ==
                    'single_forward_next_token_P(B)/(P(A)+P(B)); A=NO_TRADE; B=TRADE' and identity.get('threshold') == .5,
                    'Unsupported saved scoring protocol; refusing to reinterpret predictions')
            journal = path / 'predictions.jsonl'
            raw = journal.read_bytes() if journal.exists() else b''
            if raw and not raw.endswith(b'\n'):
                print(f'{variant}: unfinished last journal line; resume inference with its original checkout first', flush=True)
                pending.append(str(path))
                continue
            rows = [json.loads(line) for line in raw.splitlines()]
            require(len(rows) <= len(labels), 'Journal exceeds selected target count')
            for row, prompt, target in zip(rows, prompts, labels):
                require(type(row.get('label')) is int and all(row.get(k) == target[k] for k in
                        ('id', 'actor_id', 'market_id', 'fixture_id', 'query_time', 'end_time', 'label')),
                        'Saved prediction target/label mismatch')
                require(row.get('prompt_sha256') == digest(prompt['messages']), 'Saved prediction prompt mismatch')
                require(probability(row['prior_rate_score']) == target['prior_rate_score'], 'Saved baseline score mismatch')
                p = probability(row['trade_score'])
                probability(row['unconstrained_choice_mass'])
                require(row.get('prediction') == ('TRADE' if p >= .5 else 'NO_TRADE'), 'Prediction disagrees with saved score')
            if len(rows) < len(labels):
                print(f'{variant}: {len(rows)}/{len(labels)} predictions saved; no complete summary written', flush=True)
                pending.append(str(path))
                continue
            write_summary(meta, rows, variant, path, offline=True)
            completed[variant] = path
    if set(completed) == {'basic', 'inmarket'}:
        compare(argparse.Namespace(basic=completed['basic'], inmarket=completed['inmarket'],
                                   out=args.results / 'comparison.json'))
    return {'completed': {k: str(v) for k, v in completed.items()}, 'pending': pending}


def evaluate(args):
    require(args.limit >= 0 and args.max_context > 0 and args.min_free_gib > 0, 'Invalid limits')
    require(os.environ.get('WORLD_SIZE', '1') == '1', 'Use python, not torchrun')
    for name, value in {'FLA_TILELANG': '1', 'FLA_DISABLE_BACKEND_DISPATCH': '0',
                        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '2'}.items():
        os.environ.setdefault(name, value)
    meta, records = read_bundle(args.bundle)
    training, adapter = validate_run(args.run_dir, args.model, args.variant, meta)
    limit = args.limit or meta['targets']
    require(limit <= meta['targets'], '--limit exceeds bundle size')
    prompts, labels = records[args.variant][:limit], records['labels'][:limit]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, use_fast=True)
    require(tokenizer.is_fast and tokenizer.chat_template, 'Need saved fast chat tokenizer')
    tokenizer_sha = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    require(tokenizer_sha == training['signature']['data']['tokenizer'] and
            tokenizer.chat_template == training['signature']['data']['template'], 'Tokenizer differs from training')
    encoded, choice_ids = encode_prompts(tokenizer, prompts, args.max_context)
    print(f"{args.variant}: {len(prompts):,} windows; {sum(y['label'] for y in labels):,} positives; "
          f"max prompt {max(map(len, encoded)):,} tokens. Actor overlap: 0. No weights loaded yet.", flush=True)
    print('New future-window task: existing adapters are being tested zero-shot. A=NO_TRADE, B=TRADE.', flush=True)
    if args.check_only:
        return
    import torch
    from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM, set_seed
    from peft import PeftModel, prepare_model_for_kbit_training
    require(torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count(), 'Requested CUDA GPU unavailable')
    torch.cuda.set_device(args.gpu)
    require(torch.cuda.is_bf16_supported(), 'BF16 CUDA GPU required')
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'bitsandbytes',
                'tokenizers', 'fla-core', 'causal-conv1d', 'tilelang')}
    code = {name: sha(ROOT / name) for name in ('scripts/test_actor_activity.py',
        'tools/actor_activity_common.py', 'scripts/evaluate_world_cup.py',
        'tools/world_cup_eval_common.py', 'scripts/train_world_cup_multigpu.py')}
    identity = {'format': 'world_cup_activity_scores_v1', 'variant': args.variant,
        'bundle_sha256': sha(args.bundle / 'manifest.json'), 'target_sha256': meta['target_sha256'],
        'selected_ids_sha256': digest([r['id'] for r in labels]), 'limit': args.limit,
        'adapter_sha256': sha(adapter / 'adapter_model.safetensors'),
        'adapter_config_sha256': sha(adapter / 'adapter_config.json'),
        'training_metadata_sha256': sha(args.run_dir / 'training_metadata.json'),
        'tokenizer_sha256': tokenizer_sha, 'code': code, 'versions': versions,
        'score_protocol': 'single_forward_next_token_P(B)/(P(A)+P(B)); A=NO_TRADE; B=TRADE',
        'probability_calibrated': False, 'threshold': .5, 'max_context': args.max_context, 'seed': 42,
        'attention': 'sdpa', 'training_task': meta['reference'][args.variant]['training_task']}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another evaluator is writing to this output directory') from exc
        identity_path = args.out / 'identity.json'
        if identity_path.exists():
            require(args.resume and json.loads(identity_path.read_text()) == identity,
                    'Existing output/settings/code differ; use identical --resume or a fresh --out')
        else:
            require(not any(p.name != 'evaluation.lock' for p in args.out.iterdir()), 'Output directory is not empty')
            write_json(identity_path, identity)
        journal = args.out / 'predictions.jsonl'
        rows = recover(journal, prompts, labels, args.resume)
        if len(rows) < len(labels):
            free, capacity = torch.cuda.mem_get_info(args.gpu)
            require(free >= args.min_free_gib * 1024**3 and free >= .80 * capacity,
                    f'GPU {args.gpu} has only {free / 1024**3:.1f} GiB free. Use an idle GPU; '
                    'do not share with training. No weights loaded.')
            kernel_probe(torch, args.gpu)
            torch.cuda.empty_cache()
            set_seed(42)
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            print('Loading frozen base and adapter once. Each window needs one forward pass, no long generation.', flush=True)
            base = Qwen3_5ForCausalLM.from_pretrained(args.model, local_files_only=True,
                dtype=torch.bfloat16, quantization_config=quant, device_map={'': args.gpu}, attn_implementation='sdpa')
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
            model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
            model.eval()
            device = torch.device('cuda', args.gpu)
            initial, started = len(rows), time.monotonic()
            with journal.open('a', buffering=1) as stream:
                for index in range(initial, len(labels)):
                    ids = torch.tensor([encoded[index]], dtype=torch.long, device=device)
                    with torch.inference_mode():
                        output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       use_cache=False, logits_to_keep=1)
                        logits = output.logits[0, -1].float()
                        p = torch.softmax(logits[choice_ids], dim=0)[1].item()
                        choice_mass = torch.softmax(logits, dim=0)[choice_ids].sum().item()
                    require(math.isfinite(p) and math.isfinite(choice_mass), 'Nonfinite model output; no score written')
                    row = {**labels[index], 'trade_score': p, 'prediction': 'TRADE' if p >= .5 else 'NO_TRADE',
                        'unconstrained_choice_mass': choice_mass, 'prompt_tokens': len(encoded[index]),
                        'prompt_sha256': digest(prompts[index]['messages'])}
                    stream.write(dump(row) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows.append(row)
                    del output, logits, ids
                    if (index + 1) % 10 == 0 or index + 1 == len(labels):
                        seconds = (time.monotonic() - started) / (index + 1 - initial)
                        print(f'{index+1}/{len(labels)} windows; {seconds:.2f}s/window; '
                              f'rough remaining {(len(labels)-index-1)*seconds/60:.1f} min', flush=True)
            del model, base
            gc.collect()
            torch.cuda.empty_cache()
        return write_summary(meta, rows, args.variant, args.out)


def compare(args):
    rows, identities, summaries = {}, {}, {}
    for name, path in (('basic', args.basic), ('inmarket', args.inmarket)):
        summary = summaries[name] = json.loads((path / 'summary.json').read_text())
        require(summary['status'] == 'completed' and summary['variant'] == name, 'Need completed matching variants')
        require(sha(path / 'identity.json') == summary['identity_sha256'] and
                sha(path / 'predictions.jsonl') == summary['predictions_sha256'], 'Evaluation output checksum mismatch')
        identities[name] = json.loads((path / 'identity.json').read_text())
        rows[name] = [json.loads(line) for line in (path / 'predictions.jsonl').read_text().splitlines()]
    for field in ('bundle_sha256', 'target_sha256', 'selected_ids_sha256', 'score_protocol', 'max_context', 'seed', 'code', 'versions'):
        require(identities['basic'][field] == identities['inmarket'][field], f'Comparison differs: {field}')
    require([(r['id'], r['label']) for r in rows['basic']] == [(r['id'], r['label']) for r in rows['inmarket']],
            'Models did not receive identical target windows')
    models = {name: summarize([r['label'] for r in values], [r['trade_score'] for r in values]) for name, values in rows.items()}
    models.update(summaries['basic']['baselines'])
    result = {'status': 'completed', 'paired_windows': len(rows['basic']), 'models': models,
        'task': summaries['basic']['task'], 'pilot': summaries['basic']['pilot'],
        'retrospectively_filtered_actor_cohort': summaries['basic']['retrospectively_filtered_actor_cohort'],
        'interpretation': 'No test-set threshold fitting. Scores are forced-choice preferences, not calibrated probabilities. '
                          'Windows are clustered by wallet/match; this table is descriptive, not a significance test.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def run_both(args):
    # Validate both runs/tokenizers/data before spending GPU time on either adapter.
    for check_only in ([True] if args.check_only else [True, False]):
        for variant in ('basic', 'inmarket'):
            child = copy.copy(args)
            child.variant = variant
            child.run_dir = getattr(args, variant + '_run_dir')
            child.out = args.out / variant
            child.check_only = check_only
            evaluate(child)
    if not args.check_only:
        compare(argparse.Namespace(basic=args.out / 'basic', inmarket=args.out / 'inmarket', out=args.out / 'comparison.json'))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Offline CPU/Mac preparation, no model/packages/API required')
    prep.add_argument('--basic-sft', type=Path, required=True)
    prep.add_argument('--inmarket-sft', type=Path, required=True)
    group = prep.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-root', type=Path, help='Directory of full saved raw market exports')
    group.add_argument('--exports', nargs='+', type=Path, help='Explicit saved raw market export directories')
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--targets', type=int, default=2000)
    prep.add_argument('--horizon-seconds', type=int, default=60)
    prep.add_argument('--match-minutes', type=int, default=120)
    prep.add_argument('--history-groups', type=int, default=20)
    prep.add_argument('--news-seconds', type=int, default=1200)
    prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--max-trades-per-actor', type=int, default=20,
                      help='Keep wallets with at most N captured executions per market (default 20); 0 requires unfiltered exports')
    prep.set_defaults(func=common.prepare)
    for name, func in (('evaluate', evaluate), ('run', run_both)):
        parser = sub.add_parser(name, help='One adapter' if name == 'evaluate' else 'Check both, then evaluate sequentially and compare')
        parser.add_argument('--bundle', type=Path, required=True)
        if name == 'evaluate':
            parser.add_argument('--variant', choices=('basic', 'inmarket'), required=True)
            parser.add_argument('--run-dir', type=Path, required=True)
        else:
            parser.add_argument('--basic-run-dir', type=Path, required=True)
            parser.add_argument('--inmarket-run-dir', type=Path, required=True)
        parser.add_argument('--model', type=Path, default=Path('/workspace/models/Qwen3.6-27B'))
        parser.add_argument('--out', type=Path, required=True)
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--max-context', type=int, default=16384)
        parser.add_argument('--min-free-gib', type=float, default=55)
        parser.add_argument('--limit', type=int, default=0, help='0=all; prefix of seeded random order is a pilot')
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--check-only', action='store_true', help='Tokenizers/identities/lengths only; no weights')
        parser.set_defaults(func=func)
    comparison = sub.add_parser('compare')
    comparison.add_argument('--basic', type=Path, required=True)
    comparison.add_argument('--inmarket', type=Path, required=True)
    comparison.add_argument('--out', type=Path, required=True)
    comparison.set_defaults(func=compare)
    recovery = sub.add_parser('report', help='CPU-only summary recovery from complete saved journals; no inference')
    recovery.add_argument('--bundle', type=Path, required=True)
    recovery.add_argument('--results', type=Path, required=True, help='One result directory, or parent containing basic/ and inmarket/')
    recovery.set_defaults(func=report)
    return p.parse_args(argv)


if __name__ == '__main__':
    try:
        args = parse_args()
        args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'Error: {exc}')
