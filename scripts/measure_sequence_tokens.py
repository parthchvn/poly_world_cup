#!/usr/bin/env python3
"""Measure complete conversation rows using a pinned real tokenizer.

No model weights, truncation, GPU measurement or training-speed estimate.
The optional prefix baseline compares repeated serialization of these same
conversation prefixes, not a different dataset release.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re

REFERENCE_MODEL = "Qwen/Qwen3-0.6B"
REFERENCE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data/tokenizer_reference/qwen3_06b"
REFERENCE_CHAT_TEMPLATE = Path(__file__).resolve().parents[1] / "configs/actor_sequence_chat_template.jinja"
CHAT_OPTIONS = {"add_generation_prompt": False, "enable_thinking": False}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_tokenizer(path: str | Path = REFERENCE_PATH, chat_template: str | Path | None = None):
    """Load a local tokenizer only; never download or load model weights."""
    os.environ.setdefault("USE_TORCH", "0")
    os.environ.setdefault("USE_TF", "0")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers==4.57.6 and jinja2==3.1.6 to measure tokens") from exc
    path = Path(path)
    if not (path / "tokenizer_config.json").is_file():
        raise ValueError(f"Missing local tokenizer at {path}; see docs/actor_sequence_training.md")
    manifest_path = path / "reference_manifest.json"
    if manifest_path.exists():
        for name, expected in json.loads(manifest_path.read_text()).get("files", {}).items():
            if file_sha256(path / name) != expected["sha256"]:
                raise ValueError(f"Tokenizer checksum mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if chat_template is not None:
        tokenizer.chat_template = Path(chat_template).read_text(encoding="utf-8")
    if not tokenizer.chat_template:
        raise ValueError("Choose a tokenizer with an explicit training chat template")
    return tokenizer


def validate_messages(messages: object) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("A row must contain a nonempty messages list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message at index {index}")
        if not isinstance(message.get("content"), str):
            raise ValueError("This dataset requires string message contents")
        if message.get("weight", 1) != 1:
            raise ValueError("Per-message loss weights require a separate verified masking implementation")
    if messages[-1]["role"] != "assistant":
        raise ValueError("A training sequence must end in an assistant target")
    return messages


def count_chat_tokens(tokenizer, messages: list[dict]) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, truncation=False, padding=False, **CHAT_OPTIONS))


def assistant_mask_probe(tokenizer, messages: list[dict]) -> dict:
    """Probe native template masks without inventing a loss-mask fallback."""
    if not re.search(r"\{%[-+]?\s*generation\b", tokenizer.get_chat_template()):
        return {"status": "unsupported_native_template", "reason": "No generation blocks; assistant-only loss is not verified."}
    try:
        result = tokenizer.apply_chat_template(messages, tokenize=True, truncation=False, padding=False,
            return_dict=True, return_assistant_tokens_mask=True, **CHAT_OPTIONS)
        mask = result.get("assistant_masks")
        if mask is None or len(mask) != len(result["input_ids"]) or not any(mask):
            return {"status": "invalid_native_mask", "reason": "Missing, empty or wrong-length mask"}
        if any(value not in (0, 1) for value in mask):
            return {"status": "invalid_native_mask", "reason": "Nonbinary mask"}
        spans, start = [], None
        for index, value in enumerate([*mask, 0]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                spans.append(tokenizer.decode(result["input_ids"][start:index], skip_special_tokens=False))
                start = None
        expected = [message["content"] + (tokenizer.eos_token or "")
                    for message in messages if message["role"] == "assistant"]
        return {"status": "assistant_content_and_end_tokens_verified" if spans == expected else "native_mask_returned_not_semantically_verified",
                "masked_tokens": sum(mask), "total_tokens": len(mask), "supervised_spans": len(spans),
                "assistant_turns": len(expected),
                "note": "Probe applies only to this row/template; inspect actual trainer labels before training."}
    except (ValueError, TypeError, RuntimeError) as exc:
        return {"status": "native_mask_error", "reason": str(exc)}


def distribution(histogram: Counter) -> dict:
    count = sum(histogram.values())
    if not count:
        return {"count": 0, "total": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(histogram.items())
    def quantile(fraction):
        rank, cumulative = max(1, math.ceil(count * fraction)), 0
        for value, frequency in ordered:
            cumulative += frequency
            if cumulative >= rank:
                return value
    total = sum(value * frequency for value, frequency in ordered)
    return {"count": count, "total": total, "min": ordered[0][0], "p50": quantile(.5),
            "p95": quantile(.95), "max": ordered[-1][0], "mean": total / count}


def input_paths(inputs: list[str | Path]) -> list[Path]:
    found = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            found.update(p.resolve() for p in path.rglob("*.jsonl.gz"))
            found.update(p.resolve() for p in path.rglob("*.jsonl"))
        elif path.is_file():
            found.add(path.resolve())
        else:
            raise ValueError(f"Missing input: {path}")
    if not found:
        raise ValueError("No JSONL input files found")
    return sorted(found)


def rows(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank line at {path}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object row at {path}:{number}")
            yield number, row


def measure_sequences(paths: list[Path], tokenizer, *, limit_per_group: int = 0,
                      prefix_baseline_limit_per_group: int = 100, max_tokens: int = 8192) -> dict:
    if min(limit_per_group, prefix_baseline_limit_per_group) < 0 or max_tokens <= 0:
        raise ValueError("Limits must be nonnegative and max_tokens positive")
    groups, inputs = {}, []
    for path in paths:
        path = Path(path)
        inputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        for line_number, row in rows(path):
            key = (str(row.get("profile", "unspecified")), str(row.get("split", "unspecified")))
            if key not in groups:
                groups[key] = {"rows_seen": 0, "rows_measured": 0, "chat": Counter(), "assistant": Counter(),
                    "turns": Counter(), "over_limit": 0, "over_limit_examples": [], "baseline_chat": Counter(),
                    "baseline_prefix": Counter(), "baseline_rows": 0, "mask_probe": None}
            group = groups[key]
            group["rows_seen"] += 1
            if limit_per_group and group["rows_measured"] >= limit_per_group:
                continue
            messages = validate_messages(row.get("messages"))
            token_count = count_chat_tokens(tokenizer, messages)
            group["rows_measured"] += 1
            group["chat"][token_count] += 1
            indices = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
            group["turns"][len(indices)] += 1
            for index in indices:
                group["assistant"][len(tokenizer.encode(messages[index]["content"], add_special_tokens=False))] += 1
            if token_count > max_tokens:
                group["over_limit"] += 1
                if len(group["over_limit_examples"]) < 10:
                    group["over_limit_examples"].append({"file": str(path), "line": line_number,
                        "sequence_id": row.get("sequence_id"), "tokens": token_count})
            if group["mask_probe"] is None:
                group["mask_probe"] = assistant_mask_probe(tokenizer, messages)
            if group["baseline_rows"] < prefix_baseline_limit_per_group:
                group["baseline_rows"] += 1
                group["baseline_chat"][token_count] += 1
                for index in indices:
                    group["baseline_prefix"][count_chat_tokens(tokenizer, messages[:index + 1])] += 1
    results = []
    for (profile, split), group in sorted(groups.items()):
        chat = distribution(group["chat"])
        base_chat, base_prefix = distribution(group["baseline_chat"]), distribution(group["baseline_prefix"])
        results.append({"profile": profile, "split": split, "rows_seen": group["rows_seen"],
            "measurement_scope": "all_rows" if chat["count"] == group["rows_seen"] else "first_rows_per_group",
            "conversation_tokens": chat, "assistant_content_tokens_per_turn": distribution(group["assistant"]),
            "assistant_turns_per_row": distribution(group["turns"]), "rows_over_limit": group["over_limit"],
            "over_limit_examples": group["over_limit_examples"], "assistant_mask_probe": group["mask_probe"],
            "repeated_prefix_comparison": {"scope": "first_rows_per_group", "rows": group["baseline_rows"],
                "conversation_tokens": base_chat, "independent_prefix_tokens_per_target": base_prefix,
                "prefix_to_conversation_token_ratio": base_prefix["total"] / base_chat["total"] if base_chat["total"] else None,
                "not_an_estimate_of_gpu_speedup": True, "not_a_comparison_to_the_v2_export": True}})
    if not results:
        raise ValueError("No training rows found")
    return {"schema": "actor_sequence_token_measurement_v1", "inputs": inputs, "chat_options": dict(CHAT_OPTIONS),
        "truncation": False, "max_tokens": max_tokens, "limit_per_group": limit_per_group,
        "prefix_baseline_limit_per_group": prefix_baseline_limit_per_group,
        "token_limit_passed_for_measured_rows": all(g["rows_over_limit"] == 0 for g in results),
        "all_rows_measured": all(g["measurement_scope"] == "all_rows" for g in results),
        "quantile_method": "nearest_rank", "groups": results, "training_or_gpu_benchmark_performed": False,
        "loss_mask_counts_verified": False,
        "assistant_content_counts_exclude_control_tokens_and_are_not_loss_mask_counts": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSONL(.gz) files or profile directories, excluding audit files")
    parser.add_argument("--tokenizer", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--chat-template", type=Path, default=REFERENCE_CHAT_TEMPLATE)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit-per-group", type=int, default=0, help="0=all; positive=first N per profile/split")
    parser.add_argument("--prefix-baseline-limit-per-group", type=int, default=100, help="0 disables slower prefix comparison")
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    tokenizer = load_reference_tokenizer(args.tokenizer, args.chat_template)
    report = measure_sequences(input_paths(args.inputs), tokenizer, limit_per_group=args.limit_per_group,
        prefix_baseline_limit_per_group=args.prefix_baseline_limit_per_group, max_tokens=args.max_tokens)
    manifest = args.tokenizer / "reference_manifest.json"
    report["tokenizer"] = {"local_path": str(args.tokenizer.resolve()), "class": type(tokenizer).__name__,
        "transformers_version": importlib.metadata.version("transformers"), "tokenizers_version": importlib.metadata.version("tokenizers"),
        "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
        "chat_template_path": str(args.chat_template.resolve()),
        "files": {p.name: file_sha256(p) for p in sorted(args.tokenizer.iterdir()) if p.is_file()},
        "reference_manifest": json.loads(manifest.read_text()) if manifest.exists() else None}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.report), "rows_measured": sum(g["conversation_tokens"]["count"] for g in report["groups"]),
        "all_rows_measured": report["all_rows_measured"], "token_limit_passed_for_measured_rows": report["token_limit_passed_for_measured_rows"]}))
    return 0 if report["token_limit_passed_for_measured_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
