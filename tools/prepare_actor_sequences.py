#!/usr/bin/env python3
"""Build a new actor-conversation release without modifying earlier datasets."""
from __future__ import annotations
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.actor_sequences import SequencePolicy, export_actor_sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path("datasets/world_cup_2026_tournament_lt20_v2_evidence"))
    parser.add_argument("--split-policy", type=Path, default=Path("configs/tournament_sft_v1.json"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer_reference/qwen3_06b"))
    parser.add_argument("--chat-template", type=Path, default=Path("configs/actor_sequence_chat_template.jinja"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--actor-limit", type=int, help="Explicitly marked smoke sample; not a complete release")
    defaults = SequencePolicy()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    if args.workers < 1 or (args.actor_limit is not None and args.actor_limit < 1):
        parser.error("workers and actor-limit must be positive")
    policy = SequencePolicy(**{f.name: getattr(args, f.name) for f in fields(defaults)})
    report = export_actor_sequences(args.source, args.output, args.evidence,
        json.loads(args.split_policy.read_text()), policy, args.tokenizer, args.chat_template,
        workers=args.workers, actor_limit=args.actor_limit)
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
