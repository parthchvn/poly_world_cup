#!/usr/bin/env python3
"""Train a text-only Qwen3.6 LoRA with Hugging Face libraries on one CUDA GPU.

Only assistant JSON targets and their end-of-message tokens contribute loss.
No model training algorithm is implemented here: PEFT adds LoRA and Transformers
Trainer runs optimization. Training reads train/validation.jsonl, never test.jsonl.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
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
    open_source = gzip.open if path.suffix == ".gz" else open
    with open_source(path, "rb") as source:
        for number, line in enumerate(source, 1):
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
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
    stats.update(sha256=digest.hexdigest(), sha256_scope="uncompressed_jsonl_bytes", fixtures=sorted(fixtures))
    return rows, stats


def split_path(directory, split):
    """Prefer an unpacked split if present, otherwise read the published gzip."""
    plain = directory / f"{split}.jsonl"
    return plain if plain.exists() else directory / f"{split}.jsonl.gz"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/world_cup_2026_pilot_15k"))
    parser.add_argument("--model", default="/workspace/models/Qwen3.6-27B")
    parser.add_argument("--out", type=Path, default=Path("outputs/world_cup_qlora"))
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="10 for a short GPU training smoke run")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true", help="Tokenize and inspect loss masks without loading model weights or requiring CUDA")
    args = parser.parse_args()
    if min(args.max_length, args.batch_size, args.gradient_accumulation, args.rank, args.alpha, args.eval_steps) <= 0:
        parser.error("length, batch, accumulation, rank, alpha and eval steps must be positive")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("epochs/rate must be positive; max steps must be -1 or positive")
    return args


def main():
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.out}. Choose a fresh --out.")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install the packages listed in docs/world_cup_pilot_lora.md") from error
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required for exact assistant masks")
    if not tokenizer.chat_template:
        raise ValueError("The model folder must contain its official chat template")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Preparing train and validation conversations; test.jsonl is not opened", flush=True)
    train_rows, train_stats = read_split(split_path(args.dataset_dir, "train"), tokenizer, args.max_length)
    val_rows, val_stats = read_split(split_path(args.dataset_dir, "validation"), tokenizer, args.max_length)
    overlap = set(train_stats["fixtures"]) & set(val_stats["fixtures"])
    if overlap:
        raise ValueError(f"Train/validation fixture overlap: {sorted(overlap)}")
    metadata = {
        "status": "prepared", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "task": "conditional_execution", "dataset_dir": str(args.dataset_dir),
        "train": train_stats, "validation": val_stats, "test_used": False,
        "chat_template_kwargs": CHAT_KWARGS,
        "loss_scope": "assistant JSON content and end-of-message token only",
        "max_length": args.max_length, "seed": args.seed,
        "lora": {"rank": args.rank, "alpha": args.alpha, "target_modules": "all-linear", "dropout": 0.05},
        "quantization": "4-bit NF4 with double quantization, bfloat16 compute",
        "requested_training": {"epochs": args.epochs, "max_steps": args.max_steps, "learning_rate": args.learning_rate,
                               "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation},
    }
    print(json.dumps({"train": train_stats, "validation": val_stats}, indent=2), flush=True)
    if args.prepare_only:
        print("Preparation complete. No weights were loaded and no training was run.")
        return

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (BitsAndBytesConfig, DataCollatorForSeq2Seq,
                                  Qwen3_5ForCausalLM, Trainer, TrainingArguments, set_seed)
    except ImportError as error:
        raise RuntimeError("Install current Transformers, PEFT, datasets, accelerate, bitsandbytes and CUDA PyTorch; see docs/world_cup_pilot_lora.md") from error
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("This entry point requires a CUDA GPU supporting bfloat16; run it on the RunPod GPU")
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This entry point uses one GPU. Set CUDA_VISIBLE_DEVICES=0 and run python directly.")
    set_seed(args.seed)
    metadata["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes")
    }
    metadata["gpu"] = torch.cuda.get_device_name(0)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        quantization_config=quantization, device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    cadence = min(args.eval_steps, args.max_steps) if args.max_steps > 0 else args.eval_steps
    training_args = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", lr_scheduler_type="cosine", warmup_steps=0,
        eval_strategy="steps", eval_steps=cadence, save_strategy="steps", save_steps=cadence,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, logging_steps=min(10, cadence), report_to="none",
        prediction_loss_only=True, remove_unused_columns=False, seed=args.seed, data_seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows), processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
    )
    result = trainer.train()
    validation = trainer.evaluate()
    final_dir = args.out / "adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    training_metrics=result.metrics, validation_metrics=validation,
                    selected_checkpoint=trainer.state.best_model_checkpoint,
                    completed_steps=trainer.state.global_step, adapter=str(final_dir))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved adapter and tokenizer: {final_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
