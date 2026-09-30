#!/usr/bin/env python3
"""Single-node, N-GPU supervised QLoRA for the existing World Cup conversations.

Usage on RunPod: python train_world_cup_multigpu.py --gpus 2
Use --help for smoke tests, benchmarks, checkpoint resume and alternate paths.
The launcher starts one torchrun process per GPU. No package installs or downloads.
Only assistant JSON and its end marker receive loss. Test data is never opened.
Based on this project's train_market_qlora.py; see MULTIGPU_README.md for sources.
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
import re
import sys
import math
import inspect
import shutil
import subprocess
import time
import statistics
import functools
import tempfile
from datetime import datetime, timezone


CHAT_KWARGS = {"enable_thinking": False, "preserve_thinking": True}
MESSAGE_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.S)
END = "<|im_end|>"


def encode_conversation(record, tokenizer, max_length, location):
    """Apply the official template, then identify assistant-only loss by offsets."""
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{location}: expected a messages conversation")
    roles = []
    for index, message in enumerate(messages):
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise ValueError(f"{location}: only text system/user/assistant messages are supported")
        if any(token in content for token in ("<|im_start|>", END, "<think>", "</think>")):
            raise ValueError(f"{location}: message {index} contains reserved chat markers")
        if role == "assistant":
            if not roles or roles[-1] != "user":
                raise ValueError(f"{location}: each assistant target must follow a user query")
            if not isinstance(json.loads(content), dict):
                raise ValueError(f"{location}: each assistant target must be a JSON object")
        roles.append(role)
    expected_roles = (["system"] if roles[0] == "system" else [])
    remaining = len(roles) - len(expected_roles)
    expected_roles += ["user", "assistant"] * (remaining // 2)
    if roles != expected_roles:
        raise ValueError(f"{location}: expected optional system then user/assistant pairs")
    targets = roles.count("assistant")
    if record.get("target_count", targets) != targets:
        raise ValueError(f"{location}: target_count disagrees with assistant messages")

    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **CHAT_KWARGS
    )
    chunks = list(MESSAGE_RE.finditer(rendered))
    if [chunk.group(1) for chunk in chunks] != roles:
        raise ValueError(f"{location}: unsupported chat template; expected Qwen ChatML messages")
    spans = []
    for message, chunk in zip(messages, chunks):
        if message["role"] != "assistant":
            continue
        content = message["content"].strip()
        body = chunk.group(2)
        if not body.endswith(content):
            raise ValueError(f"{location}: chat template changed an assistant target")
        prefix = body[:len(body) - len(content)]
        if prefix not in ("", "<think>\n\n</think>\n\n"):
            raise ValueError(f"{location}: unexpected assistant reasoning/template prefix")
        begin = chunk.end(2) - len(content)
        spans.append((begin, chunk.end()))  # Include the assistant's <|im_end|>.

    encoded = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False, return_attention_mask=True,
    )
    input_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(input_ids) > max_length:
        raise ValueError(
            f"{location}: {len(input_ids)} tokens exceeds --max-length {max_length}. "
            "Increase the limit if GPU memory permits, or rebuild shorter actor "
            "sequences with explicit earlier context. No targets were truncated."
        )
    labels = [-100] * len(input_ids)
    cursor = 0
    span_token_counts = [0] * targets
    eos_id = tokenizer.convert_tokens_to_ids(END)
    eos_count = 0
    for index, (left, right) in enumerate(offsets):
        if left == right:
            continue
        while cursor < len(spans) and left >= spans[cursor][1]:
            cursor += 1
        if cursor == len(spans):
            break
        begin, end = spans[cursor]
        if right <= begin:
            continue
        if left < begin or right > end:
            raise ValueError(f"{location}: tokenizer crosses an assistant target boundary")
        labels[index] = input_ids[index]
        span_token_counts[cursor] += 1
        eos_count += int(input_ids[index] == eos_id)
    if not all(count > 1 for count in span_token_counts) or eos_count != targets:
        raise ValueError(f"{location}: incomplete assistant target/EOS token mask")
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }, targets


def read_split(path, tokenizer, max_length):
    rows, fixtures = [], set()
    stats = {"conversations": 0, "targets": 0, "tokens": 0, "loss_tokens": 0, "max_tokens": 0}
    digest = hashlib.sha256()
    action_counts = {}
    open_source = gzip.open if path.suffix == ".gz" else open
    with open_source(path, "rb") as source:
        for number, line in enumerate(source, 1):
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('target_protocol') == 'observed_interval_and_execution_v1':
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
                from compare_actor_variants import conversation
                conversation(record)
            for message in record['messages']:
                if message['role'] == 'assistant':
                    action = json.loads(message['content']).get('action', 'UNKNOWN')
                    action_counts[action] = action_counts.get(action, 0) + 1
            encoded, targets = encode_conversation(record, tokenizer, max_length, f"{path}:{number}")
            if not record.get("fixture_id"):
                raise ValueError(f"{path}:{number}: fixture_id metadata is required")
            fixtures.add(str(record["fixture_id"]))
            rows.append(encoded)
            length = len(encoded["input_ids"])
            stats["conversations"] += 1
            stats["targets"] += targets
            stats["tokens"] += length
            stats["loss_tokens"] += sum(value != -100 for value in encoded["labels"])
            stats["max_tokens"] = max(stats["max_tokens"], length)
    if not rows:
        raise ValueError(f"{path}: no conversations")
    stats.update(sha256=digest.hexdigest(), sha256_scope="uncompressed_jsonl_bytes", fixtures=sorted(fixtures),
                 action_counts=action_counts)
    return rows, stats


def split_path(directory, split):
    """Prefer an unpacked split if present, otherwise read the published gzip."""
    plain = directory / f"{split}.jsonl"
    return plain if plain.exists() else directory / f"{split}.jsonl.gz"


def validate_target_counts(stats, allow_trade_only=False):
    counts = stats.get('action_counts', {})
    if not counts.get('NO_TRADE') and not allow_trade_only:
        raise ValueError('SFT contains no NO_TRADE targets. Rebuild from saved raw actor exports with '
                         'build_actor_dataset.py prepare. Use --allow-trade-only only for explicit legacy reproduction.')


DEFAULT_DATASET = "/workspace/world_cup_15k_qlora/datasets/world_cup_2026_pilot_15k"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def make_loss_callback(output_dir):
    """Persist Trainer's reduced losses on rank zero, independently of checkpoints."""
    from transformers import TrainerCallback

    directory = Path(output_dir)
    fields = ("timestamp_utc", "event", "step", "epoch", "loss", "eval_loss",
              "smoke_eval_loss", "learning_rate", "grad_norm")

    class LossLogger(TrainerCallback):
        def write(self, state, logs, event="log"):
            if not state.is_world_process_zero:
                return
            directory.mkdir(parents=True, exist_ok=True)
            row = {**logs, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "event": event, "step": state.global_step,
                   "epoch": logs.get("epoch", state.epoch)}
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as target:
                target.write(json.dumps(row) + "\n")
                target.flush()
                os.fsync(target.fileno())
            if event == "train_begin" or any(key in logs for key in ("loss", "eval_loss", "smoke_eval_loss")):
                with (directory / "losses.csv").open("a", newline="", encoding="utf-8") as target:
                    writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
                    if target.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                    target.flush()
                    os.fsync(target.fileno())

        def on_train_begin(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                # A killed process can leave an incomplete final record. Drop only
                # that uncommitted tail before appending the new resume marker.
                for name in ("metrics.jsonl", "losses.csv"):
                    path = directory / name
                    if path.exists():
                        with path.open("rb+") as target:
                            target.seek(0, os.SEEK_END)
                            end = target.tell()
                            while end:
                                target.seek(end - 1)
                                if target.read(1) == b"\n":
                                    break
                                end -= 1
                            target.truncate(end)
            # Trainer has restored global_step here. Plotting uses this marker
            # to discard the abandoned tail after a checkpoint rewind.
            self.write(state, {}, event="train_begin")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                self.write(state, logs)

    return LossLogger()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_plan(gpus, micro_batch, global_batch):
    minimum = gpus * micro_batch
    effective = math.ceil((global_batch or max(8, minimum)) / minimum) * minimum
    if global_batch is not None and effective != global_batch:
        raise ValueError(
            f"--global-batch must be divisible by gpus × micro-batch ({minimum}); "
            f"for example {effective}. No silent batch-size changes."
        )
    return {"gpus": gpus, "micro_batch": micro_batch, "effective_batch": effective,
            "gradient_accumulation": effective // minimum}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--gpus", type=int, required=True, help="GPUs on ONE machine/Pod")
    p.add_argument("--gpu-ids", help="Explicit visible GPU IDs, e.g. 0,1; overrides inherited CUDA_VISIBLE_DEVICES")
    p.add_argument("--model", type=Path, default=Path("/workspace/models/Qwen3.6-27B"))
    p.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET))
    p.add_argument("--out", type=Path, help="Fresh directory; by default creates a timestamped /workspace/runs directory")
    p.add_argument("--cache-dir", type=Path, default=Path("/workspace/cache/world_cup_sft"))
    p.add_argument("--micro-batch", type=int, default=1, help="Conversations per GPU per forward pass")
    p.add_argument("--global-batch", type=int, help="Conversations per optimizer update; default smallest achievable >=8")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max-steps", type=int, default=-1, help="Optimizer steps; -1 uses epochs")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=8192, help="Fail on longer conversations; never silently truncate")
    p.add_argument("--eval-steps", type=int, default=277)
    p.add_argument("--logging-steps", type=int, default=1,
                   help="Save live training loss every N optimizer updates to metrics.jsonl and losses.csv")
    p.add_argument("--save-steps", type=int, default=100, help="Keep optimizer/scheduler/RNG for interruption recovery")
    p.add_argument("--eval-batch", type=int, default=1)
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers PER GPU; tokenization is cached")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-ratio", type=float, default=0.0, help="0 preserves the earlier run; optional 0.03 for a new experiment")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--selected-logits", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply vocabulary head only at positions that predict supervised tokens")
    p.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=True,
                   help="Bucket complete conversations by length to reduce padding and GPU stragglers")
    p.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    p.add_argument("--kernel-check", action=argparse.BooleanOptionalAction, default=True,
                   help="Check installed Qwen FLA/causal-conv CUDA forward and backward before loading weights")
    p.add_argument("--smoke", action="store_true", help="10 optimizer steps plus up to 32 validation conversations")
    p.add_argument('--allow-trade-only', action='store_true',
                   help='Explicit legacy reproduction: permit training without NO_TRADE targets')
    p.add_argument("--smoke-then-full", action="store_true",
                   help="Check up to 32 validation conversations after 10 steps, then continue the same full run without reloading weights")
    p.add_argument("--benchmark", action="store_true", help="20 optimizer steps, no evaluation or checkpoint writes; no final adapter")
    p.add_argument("--prepare-only", action="store_true", help="Tokenize/cache without CUDA or weights")
    p.add_argument("--dry-run", action="store_true", help="Show batch plan and paths without imports, files or GPU work")
    p.add_argument("--resume", type=Path, help="Resume a checkpoint-N created by THIS script; retain its --out and settings")
    p.add_argument("--init-adapter", type=Path, help="Start a NEW run from an existing adapter, with a fresh optimizer")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--local-rank", "--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    p.add_argument("--prepared-cache", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if min(args.gpus, args.micro_batch, args.rank, args.alpha, args.max_length,
           args.eval_steps, args.save_steps, args.eval_batch, args.logging_steps) < 1:
        p.error("GPU counts, batches, lengths and intervals must be positive")
    if args.global_batch is not None and args.global_batch < 1:
        p.error("--global-batch must be positive")
    if args.workers < 0 or args.epochs <= 0 or args.learning_rate <= 0:
        p.error("workers must be nonnegative; epochs/rate must be positive")
    if not 0 <= args.dropout < 1 or not 0 <= args.warmup_ratio < 1:
        p.error("dropout and warmup ratio must be in [0,1)")
    if args.max_steps == 0 or args.max_steps < -1:
        p.error("max-steps must be -1 or positive")
    if sum((args.smoke, args.benchmark, args.smoke_then_full)) > 1 or (args.resume and (args.smoke or args.benchmark)):
        p.error("Choose one of --smoke, --benchmark, --smoke-then-full; --resume is supported only for full training")
    if args.resume and args.init_adapter:
        p.error("Use --resume OR --init-adapter, not both")
    if args.smoke:
        args.max_steps = 10
    if args.benchmark:
        args.max_steps = 20
    args.batch = batch_plan(args.gpus, args.micro_batch, args.global_batch)
    args.model = args.model.expanduser().resolve()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.resume:
        args.resume = args.resume.expanduser().resolve()
        args.out = args.out or args.resume.parent
    if args.out is None:
        tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        suffix = "benchmark" if args.benchmark else "smoke" if args.smoke else "smoke_then_full" if args.smoke_then_full else "full"
        args.out = Path(f"/workspace/runs/world_cup_{tag}_{args.gpus}gpu_{suffix}")
    args.out = args.out.expanduser().resolve()
    if args.init_adapter:
        args.init_adapter = args.init_adapter.expanduser().resolve()
    return args


def prepare_cache(args):
    """Launcher does CPU tokenization once; workers memory-map the same Arrow cache."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError("Need the model's official chat template and a fast tokenizer")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    paths = {split: split_path(args.dataset_dir, split) for split in ("train", "validation")}
    identity = {
        "script": sha256_file(__file__), "max_length": args.max_length,
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "template": tokenizer.chat_template, "chat_kwargs": CHAT_KWARGS,
        "tokenizer": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "sources": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()},
        "versions": {key: importlib.metadata.version(key) for key in ("transformers", "tokenizers", "datasets")},
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = args.cache_dir / key
    if (destination / "prepared.json").is_file():
        for stats in json.loads((destination / 'prepared.json').read_text())['splits'].values():
            validate_target_counts(stats, args.allow_trade_only)
        print(f"Reusing tokenized cache: {destination}", flush=True)
        return destination
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=key + ".", dir=args.cache_dir))
    stats = {}
    try:
        for split, path in paths.items():
            print(f"Tokenizing {split} once (all actor conversations retained)", flush=True)
            rows, stats[split] = read_split(path, tokenizer, args.max_length)
            validate_target_counts(stats[split], args.allow_trade_only)
            manifest_path = args.dataset_dir / 'manifest.json'
            if manifest_path.is_file() and json.loads(manifest_path.read_text()).get('no_trade_targets'):
                counts = stats[split]['action_counts']
                if not (counts.get('NO_TRADE', 0) == counts.get('TRADE', 0) > 0):
                    raise ValueError(f'{split}: missing NO_TRADE interval supervision: {counts}')
            lengths = sorted(len(row["input_ids"]) for row in rows)
            stats[split]["p50_tokens"] = lengths[len(lengths) // 2]
            stats[split]["p95_tokens"] = lengths[min(len(lengths)-1, int(.95 * len(lengths)))]
            for row in rows:
                row["length"] = len(row["input_ids"])
            Dataset.from_list(rows).save_to_disk(str(temp / split))
            del rows
        overlap = set(stats["train"]["fixtures"]) & set(stats["validation"]["fixtures"])
        if overlap:
            raise ValueError(f"Train/validation fixture overlap: {sorted(overlap)}")
        tokenizer.save_pretrained(temp / "tokenizer")
        atomic_json(temp / "prepared.json", {"identity": identity, "splits": stats, "test_used": False})
        # Concurrent independent launches may have prepared the same immutable cache.
        try:
            temp.rename(destination)
        except OSError:
            if not (destination / "prepared.json").is_file():
                raise
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return destination


def selected_positions(labels):
    """Hidden state at t predicts label at t+1. Union supports micro-batch >1."""
    return (labels[:, 1:] != -100).any(dim=0).nonzero(as_tuple=True)[0]


def assistant_loss(outputs, labels, num_items_in_batch=None, *, selected=True):
    """Sum token CE; Trainer supplies accumulation- AND DDP-wide denominator.

    Trainer also compensates for DDP's gradient averaging. Do not divide again
    by GPU count or gradient accumulation steps here.
    """
    import torch.nn.functional as F
    shifted = labels[:, 1:]
    logits = outputs.logits
    if selected:
        shifted = shifted.index_select(1, selected_positions(labels))
    else:
        logits = logits[:, :-1, :]
    if logits.shape[:2] != shifted.shape:
        raise RuntimeError("Selected logits and next-token labels do not align")
    denominator = num_items_in_batch if num_items_in_batch is not None else (shifted != -100).sum()
    # FP32 CE is intentional; the model and matrix multiplies use BF16.
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           shifted.reshape(-1), ignore_index=-100, reduction="sum") / denominator


def make_trainer_class():
    from transformers import Trainer

    class AssistantTrainer(Trainer):
        def __init__(self, *a, selected_logits=True, smoke_check_step=None, smoke_reporter=None, **kw):
            self.selected_logits = selected_logits
            self.smoke_check_step = smoke_check_step
            self.smoke_check_result = None
            self.smoke_reporter = smoke_reporter
            kw["compute_loss_func"] = functools.partial(assistant_loss, selected=selected_logits)
            super().__init__(*a, **kw)
            # We remove labels before model.forward via compute_loss_func. This
            # prevents PEFT kwargs from sending an unused num_items to the backbone.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            if self.selected_logits:
                inputs["logits_to_keep"] = selected_positions(inputs["labels"])
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kw):
            # Trainer calls this normally after a callback requests evaluation.
            # Every rank follows this path; never evaluate on rank zero alone.
            check_due = (self.smoke_check_step is not None and self.is_in_train
                         and self.state.global_step >= self.smoke_check_step
                         and self.smoke_check_result is None and eval_dataset is None)
            if not check_due:
                return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                        metric_key_prefix=metric_key_prefix, **kw)
            count = min(32, len(self.eval_dataset))
            metrics = super().evaluate(eval_dataset=self.eval_dataset.select(range(count)),
                                       ignore_keys=ignore_keys, metric_key_prefix="smoke_eval", **kw)
            loss = metrics.get("smoke_eval_loss")
            if loss is None or not math.isfinite(float(loss)):
                raise RuntimeError(f"Initial smoke check failed: nonfinite or missing validation loss ({loss})")
            self.smoke_check_result = {"status": "passed", "step": self.state.global_step,
                                       "validation_conversations": count, "metrics": metrics}
            if self.smoke_reporter:
                self.smoke_reporter(self.smoke_check_result)
            if self.is_world_process_zero():
                print(f"Smoke check passed at step {self.state.global_step} on {count} validation conversations. "
                      "Continuing the same run; model weights and optimizer stay loaded.", flush=True)
            return metrics

    return AssistantTrainer


def bound_backend(fn):
    for _ in range(12):
        if inspect.isfunction(fn):
            implementation = inspect.getclosurevars(fn).nonlocals.get("implementation")
            if callable(implementation):
                return implementation.__module__ + "." + implementation.__name__
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is None:
            break
        fn = wrapped
    return getattr(fn, "__module__", "?") + "." + getattr(fn, "__name__", "?")


def kernel_probe(torch, local_rank):
    """Test the bound Transformers kernels, not just whether packages import."""
    import importlib
    for module in ("fla.ops.gated_delta_rule", "causal_conv1d", "tilelang"):
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise RuntimeError(f"Missing {module}. Reuse your working FLA/TileLang environment; see README.") from error
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    delta = getattr(qwen, "torch_chunk_gated_delta_rule", None)
    if delta is None:
        raise RuntimeError("Installed Qwen implementation lacks the expected delta-rule API; see README.")
    backends = {"delta": bound_backend(delta), "convolution": bound_backend(qwen.causal_conv1d_fn)}
    print(f"[GPU {local_rank}] Bound backends: {backends}", flush=True)
    if not backends["delta"].startswith("fla.") or not backends["convolution"].startswith("causal_conv1d."):
        raise RuntimeError("Qwen's FLA/causal-conv fast paths did not bind. Restore the working environment; "
                           "--no-kernel-check explicitly allows other implementations, possibly much slower.")
    device = torch.device("cuda", local_rank)
    print(f"[GPU {local_rank}] Checking Qwen CUDA kernels; initial compilation may take minutes", flush=True)
    q, k, v = [torch.randn(1, 128, 2, 64, device=device, dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3)]
    g = torch.full((1, 128, 2), -.1, device=device, dtype=torch.float32, requires_grad=True)
    beta = torch.full((1, 128, 2), .5, device=device, dtype=torch.bfloat16, requires_grad=True)
    output, _ = delta(q, k, v, g=g, beta=beta, output_final_state=False, use_qk_l2norm_in_kernel=True)
    output.float().square().mean().backward()
    tensors = [output, q.grad, k.grad, v.grad, g.grad, beta.grad]
    x = torch.randn(1, 16, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(16, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
    conv = qwen.causal_conv1d_fn(x, weight, bias=None, activation="silu")
    conv.float().square().mean().backward()
    tensors.extend((conv, x.grad, weight.grad))
    if any(value is None or not torch.isfinite(value).all().item() for value in tensors):
        raise RuntimeError("Kernel check found missing/nonfinite outputs or gradients")
    torch.cuda.synchronize(device)
    print(f"[GPU {local_rank}] CUDA forward/backward passed (a finite-value check, not a numerical proof)", flush=True)


def run_signature(args, prepared):
    signature = {"model": str(args.model), "data": prepared["identity"],
            "batch": args.batch, "max_length": args.max_length, "epochs": args.epochs,
            "max_steps": args.max_steps, "lr": args.learning_rate, "rank": args.rank,
            "alpha": args.alpha, "dropout": args.dropout, "seed": args.seed,
            "warmup_ratio": args.warmup_ratio, "group_by_length": args.group_by_length,
            "gradient_checkpointing": args.gradient_checkpointing,
            "selected_logits": args.selected_logits, "attention": args.attention,
            "init_adapter": str(args.init_adapter) if args.init_adapter else None}
    # Preserve signatures for ordinary runs/checkpoints made before this option.
    if args.smoke_then_full:
        signature["smoke_then_full"] = True
    return signature


def launch(args):
    print(json.dumps({"batch_plan": args.batch, "model": str(args.model),
                      "dataset": str(args.dataset_dir), "output": str(args.out)}, indent=2), flush=True)
    if args.dry_run:
        return
    if args.worker:
        raise ValueError("--worker is internal to torchrun")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Launch with python SCRIPT --gpus N; it runs torchrun for you")
    if args.gpu_ids:
        ids = args.gpu_ids.split(",")
        if len(ids) != args.gpus or len(set(ids)) != len(ids) or any(not x.strip() for x in ids):
            raise ValueError("--gpu-ids must list exactly --gpus distinct IDs")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # These retain the environment that fixed the earlier H100 backward failure.
    os.environ.setdefault("FLA_TILELANG", "1")
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if not args.prepare_only:
        import torch
        available = torch.cuda.device_count()
        if available < args.gpus:
            raise ValueError(
                f"Requested {args.gpus} GPUs, but Python sees {available}. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}. "
                "On a multi-GPU Pod, use --gpu-ids 0,1 for two GPUs, or clear the old restriction."
            )
    prepared_cache = prepare_cache(args)
    prepared = json.loads((prepared_cache / "prepared.json").read_text())
    print(json.dumps(prepared["splits"], indent=2), flush=True)
    if args.prepare_only:
        print(f"Prepared cache: {prepared_cache}. No weights loaded or training run.")
        return
    signature = run_signature(args, prepared)
    if args.resume:
        if args.resume.parent != args.out:
            raise ValueError("--resume must be a checkpoint-N directly inside this run's --out directory")
        old = json.loads((args.out / "training_metadata.json").read_text())
        # An initialized adapter's identity is inherited from the interrupted run.
        signature["init_adapter"] = old["signature"]["init_adapter"]
        if old.get("signature") != signature:
            raise ValueError("Resume settings/data differ from this run. Restore the original flags, including --gpus; "
                             "use --init-adapter for a NEW run with changed settings.")
        rng_files = ["rng_state.pth"] if args.gpus == 1 else [f"rng_state_{r}.pth" for r in range(args.gpus)]
        required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json",
                    "adapter_model.safetensors", "complete.json", *rng_files]
        if not all((args.resume / name).is_file() for name in required):
            raise ValueError("Incomplete checkpoint. Choose an earlier complete checkpoint-N, not adapter/.")
        if json.loads((args.resume / "complete.json").read_text())["gpus"] != args.gpus:
            raise ValueError("Checkpoint GPU count does not match --gpus")
    elif args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"Output is not empty: {args.out}. Choose a fresh --out or use --resume.")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        atomic_json(args.out / "training_metadata.json", {
            "status": "prepared", "signature": signature, "data": prepared["splits"],
            "task": ("observed_interval_and_execution_reconstruction"
                     if prepared['splits']['train'].get('action_counts', {}).get('NO_TRADE') else "conditional_execution"), "test_used": False,
            "script_sha256": sha256_file(__file__), "prepared_cache": str(prepared_cache),
            "mode": "benchmark" if args.benchmark else "smoke" if args.smoke else "smoke_then_full" if args.smoke_then_full else "full",
            "smoke_check": {"status": "pending"} if args.smoke_then_full else None,
        })
    # Capture software versions for recreating the environment; never install or upgrade it.
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if freeze.returncode == 0:
        (args.out / f"requirements_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.txt").write_text(freeze.stdout)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc-per-node={args.gpus}", "--max-restarts=0", str(Path(__file__).resolve()),
               *sys.argv[1:], "--worker", "--prepared-cache", str(prepared_cache), "--out", str(args.out)]
    atomic_json(args.out / "launch_command.json", command)
    print(f"Launching {args.gpus} training processes. Log: {args.out / 'run.log'}", flush=True)
    with (args.out / "run.log").open("a", buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Training exited with code {code}. Full log: {args.out / 'run.log'}")


def train_worker(args):
    import torch
    import torch.distributed as dist
    from datasets import load_from_disk
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Qwen3_5ForCausalLM, TrainerCallback, TrainingArguments, set_seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != args.gpus:
        raise ValueError("torchrun world size does not match --gpus")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This script requires CUDA GPUs with BF16 support (e.g. H100)")
    if args.selected_logits and "logits_to_keep" not in inspect.signature(Qwen3_5ForCausalLM.forward).parameters:
        raise RuntimeError("Model lacks tensor logits_to_keep support; use --no-selected-logits or the recorded Transformers version")
    if args.kernel_check:
        kernel_probe(torch, local_rank)
        torch.cuda.empty_cache()
    set_seed(args.seed)
    train_data = load_from_disk(str(args.prepared_cache / "train"))
    val_data = load_from_disk(str(args.prepared_cache / "validation"))
    if args.smoke:
        val_data = val_data.select(range(min(32, len(val_data))))
    tokenizer = AutoTokenizer.from_pretrained(args.prepared_cache / "tokenizer", local_files_only=True)
    tokenizer.padding_side = "right"
    steps_per_epoch = math.ceil(math.ceil(len(train_data) / (args.gpus * args.micro_batch)) /
                                args.batch["gradient_accumulation"])
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.epochs * steps_per_epoch)
    cadence = min(args.eval_steps, total_steps)
    save_cadence = min(args.save_steps, total_steps)
    kwargs = dict(
        output_dir=str(args.out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch, per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.batch["gradient_accumulation"], learning_rate=args.learning_rate,
        bf16=True, tf32=True, gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch_fused",
        lr_scheduler_type="cosine", warmup_steps=math.ceil(total_steps * args.warmup_ratio),
        eval_strategy="no" if args.benchmark else "steps", eval_steps=cadence,
        save_strategy="no" if args.benchmark else "steps", save_steps=save_cadence,
        save_total_limit=2, save_only_model=False, load_best_model_at_end=False,
        logging_strategy="steps", logging_steps=args.logging_steps, logging_first_step=True,
        logging_nan_inf_filter=False,
        report_to="none", prediction_loss_only=True, remove_unused_columns=False,
        label_names=["labels"], average_tokens_across_devices=True,
        dataloader_num_workers=args.workers, dataloader_pin_memory=True,
        ddp_find_unused_parameters=False, ddp_broadcast_buffers=False, ddp_timeout=7200,
        seed=args.seed, data_seed=args.seed,
    )
    parameters = inspect.signature(TrainingArguments).parameters
    # Transformers renamed this option; support both public APIs explicitly.
    if "train_sampling_strategy" in parameters:
        kwargs["train_sampling_strategy"] = "group_by_length" if args.group_by_length else "random"
    elif "group_by_length" in parameters:
        kwargs["group_by_length"] = args.group_by_length
    else:
        raise RuntimeError("Unsupported TrainingArguments sampling API")
    training_args = TrainingArguments(**kwargs)
    if training_args.world_size != args.gpus:
        raise RuntimeError("Trainer did not initialize the requested distributed world")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = Qwen3_5ForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, quantization_config=quant,
        device_map={"": local_rank}, attn_implementation=args.attention,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=args.gradient_checkpointing,
                                          gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.resume:
        # Trainer restores weights and optimizer; instantiate the original adapter shape first.
        adapter_config = PeftConfig.from_pretrained(args.resume)
        adapter_config.inference_mode = False
        model = get_peft_model(base, adapter_config)
    elif args.init_adapter:
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=args.alpha,
                              target_modules="all-linear", lora_dropout=args.dropout,
                              bias="none", task_type="CAUSAL_LM"))
    # Quantized DDP holds one entire model per process; never device_map='auto'.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable:
        raise RuntimeError("No trainable adapter parameters")
    metadata_path = args.out / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    main = training_args.process_index == 0
    if main:
        model.print_trainable_parameters()
        metadata.update(status="running", started_at_utc=datetime.now(timezone.utc).isoformat(),
                        gpu=torch.cuda.get_device_name(local_rank), gpus=args.gpus,
                        trainable_parameters=trainable,
                        actual_lora_config=model.peft_config["default"].to_dict(),
                        versions={name: importlib.metadata.version(name) for name in
                                  ("torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes")},
                        requested_training=kwargs, estimated_optimizer_steps=total_steps)
        atomic_json(metadata_path, metadata)
        print(f"Live losses: {args.out / 'losses.csv'} (all metrics: {args.out / 'metrics.jsonl'})", flush=True)

    class RunProgress(TrainerCallback):
        def __init__(self):
            self.step_start = None
            self.times = []

        def on_step_begin(self, args_, state, control, **kw):
            self.step_start = time.perf_counter()

        def on_step_end(self, args_, state, control, **kw):
            if args.benchmark:
                torch.cuda.synchronize(local_rank)
            self.times.append(time.perf_counter() - self.step_start)
            if state.global_step >= state.max_steps:
                control.should_log = True  # Include a final partial logging window.
            if args.smoke_then_full and trainer.smoke_check_result is None:
                control.should_log = True
                if state.global_step >= min(10, total_steps):
                    control.should_evaluate = True
            return control

        def on_save(self, args_, state, control, **kw):
            # At on_save, every rank has finished writing its own RNG state.
            if dist.is_initialized():
                dist.barrier()
            if main:
                checkpoint = args.out / f"checkpoint-{state.global_step}"
                atomic_json(checkpoint / "complete.json", {"step": state.global_step, "gpus": args.gpus})

    progress = RunProgress()
    basic_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
                                           pad_to_multiple_of=8, label_pad_token_id=-100)

    def collate(rows):
        # 'length' serves the sampler only, never the model or tokenizer.pad.
        return basic_collator([{k: v for k, v in row.items() if k != "length"} for row in rows])

    TrainerClass = make_trainer_class()
    def record_smoke_check(report):
        if main:
            metadata["smoke_check"] = report
            atomic_json(metadata_path, metadata)

    trainer = TrainerClass(model=model, args=training_args, train_dataset=train_data,
                           eval_dataset=val_data, processing_class=tokenizer,
                           data_collator=collate, selected_logits=args.selected_logits,
                           smoke_check_step=min(10, total_steps) if args.smoke_then_full else None,
                           smoke_reporter=record_smoke_check,
                           callbacks=[progress, make_loss_callback(args.out)])
    torch.cuda.reset_peak_memory_stats(local_rank)
    try:
        result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        validation = None
        if not args.benchmark:
            # Reuse the final step's evaluation if it already happened.
            validation = next((row for row in reversed(trainer.state.log_history)
                               if "eval_loss" in row and row.get("step") == trainer.state.global_step), None)
            if validation is None:
                validation = trainer.evaluate()
            trainer.save_model(str(args.out / "adapter"))
            if main:
                tokenizer.save_pretrained(args.out / "adapter")
                (args.out / "adapter" / "base_model_path.txt").write_text(str(args.model) + "\n")
        local_stats = {"rank": local_rank, "peak_allocated_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved(local_rank) / 2**30,
                       "median_step_seconds_after_first_3": statistics.median(progress.times[3:]) if len(progress.times) > 3 else None}
        all_stats = [None] * args.gpus
        if dist.is_initialized():
            dist.all_gather_object(all_stats, local_stats)
        else:
            all_stats = [local_stats]
        if main:
            metadata.update(status="benchmark_completed" if args.benchmark else "completed",
                            completed_at_utc=datetime.now(timezone.utc).isoformat(),
                            completed_steps=trainer.state.global_step, training_metrics=result.metrics,
                            validation_metrics=validation, per_gpu=all_stats,
                            adapter=None if args.benchmark else str(args.out / "adapter"),
                            selection="last trained step; validation does not select a different checkpoint")
            atomic_json(metadata_path, metadata)
            print(json.dumps({"status": metadata["status"], "adapter": metadata["adapter"],
                              "per_gpu": all_stats, "metadata": str(metadata_path)}, indent=2), flush=True)
    except BaseException as error:
        if main:
            metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error=f"{type(error).__name__}: {error}")
            atomic_json(metadata_path, metadata)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        arguments = parse_args()
        if arguments.worker:
            train_worker(arguments)
        else:
            launch(arguments)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
