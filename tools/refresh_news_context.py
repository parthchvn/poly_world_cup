#!/usr/bin/env python3
"""Rebuild news attribution in a separate copy of a tournament cohort."""
import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.news_context import replace_news_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--news", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, help="Historically verified contract evidence JSONL")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.news.suffix == ".gz" else open
    with opener(args.news, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    contracts = None if args.contracts is None else [json.loads(line) for line in args.contracts.read_text().splitlines()]
    report = replace_news_context(args.database, args.output,
        json.loads(args.registry.read_text()), records, contract_records=contracts,
        progress=lambda message: print(message, flush=True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
