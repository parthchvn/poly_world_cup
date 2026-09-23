#!/usr/bin/env python3
"""Prepare auditable retrospective SFT JSONL profiles from a tournament cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.sft import export_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-policy", type=Path, required=True)
    parser.add_argument("--shard-rows", type=int, default=20_000)
    parser.add_argument("--news-limit", type=int, default=8)
    parser.add_argument("--deduplicate-headlines", action="store_true",
                        help="Fill the bounded prompt with distinct headlines; preserve all source evidence")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Explicitly permit the incomplete captured corpus; readiness flags remain false")
    args = parser.parse_args()
    policy = json.loads(args.split_policy.read_text(encoding="utf-8"))
    report = export_sft(args.database, args.output, policy, shard_rows=args.shard_rows,
                        news_limit=args.news_limit, allow_partial=args.allow_partial,
                        deduplicate_headlines=args.deduplicate_headlines,
                        progress=lambda message: print(message, file=sys.stderr, flush=True))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
