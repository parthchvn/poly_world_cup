#!/usr/bin/env python3
"""Report tournament capture and direct match-news coverage; exit 1 until ready."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.completion import assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--split-policy", type=Path)
    parser.add_argument("--deduplicate-headlines", action="store_true",
                        help="Use distinct normalized headlines in pre-export coverage (release uses its manifest)")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refusing to overwrite an existing report")
    report = assess(args.database, args.registry, args.release, split_policy=args.split_policy,
                    deduplicate_headlines=args.deduplicate_headlines,
                    progress=lambda message: print(message, file=sys.stderr, flush=True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in (
        "all104_fixture_context_ready", "fixtures_with_eligible_prior_direct_news",
        "fixtures_with_exported_prior_direct_news", "failed_checks")}, indent=2))
    return 0 if report["all104_fixture_context_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
