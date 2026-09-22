"""Command line entry points. No credentials or runtime dependencies required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import audit_registry
from .http import HttpClient
from .io import write_json, write_jsonl


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="Discover all 104 fixtures and map match-result contracts")
    discover.add_argument("--output", type=Path, default=Path("data/registry"))
    discover.add_argument("--cache", type=Path, default=Path("data/cache"))
    discover.add_argument("--refresh", action="store_true", help="Capture current responses while preserving prior raw bodies")

    ingest = commands.add_parser("ingest", help="Collect observed wallet-side fills with resumable cursor traversal")
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--condition", action="append", help="Condition ID; repeat to select multiple contracts")
    source.add_argument("--registry", type=Path, help="Registry JSON containing contracts to collect")
    ingest.add_argument("--output", type=Path, default=Path("data/trades"))
    ingest.add_argument("--cache", type=Path, default=Path("data/cache"))
    ingest.add_argument("--limit", type=int, default=1000, help="Rows per page (maximum 1000)")
    bound = ingest.add_mutually_exclusive_group()
    bound.add_argument("--max-pages", type=int, default=2, help="Maximum new pages per condition this invocation (default 2)")
    bound.add_argument("--all-pages", action="store_true", help="Continue each condition until API exhaustion")
    ingest.add_argument("--max-conditions", type=int, help="Limit conditions for a deterministic smoke run")

    audit = commands.add_parser("audit", help="Report mapping, ingestion coverage, and blockers to training")
    audit.add_argument("--registry", type=Path, default=Path("data/registry/registry.json"))
    audit.add_argument("--trades-root", type=Path, default=Path("data/trades"))
    audit.add_argument("--output", type=Path, default=Path("data/audit.json"))
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "discover":
            from .registry import discover_registry
            result = discover_registry(HttpClient(args.cache, refresh=args.refresh), year=2026)
            write_json(args.output / "registry.json", result)
            write_jsonl(args.output / "fixtures.jsonl", result["fixtures"])
            write_jsonl(args.output / "contracts.jsonl", result["contracts"])
            write_json(args.output / "discovery_report.json", result["report"])
            print(json.dumps(result["report"], indent=2, sort_keys=True))
            return 0 if len(result["fixtures"]) == 104 and result["report"].get("coverage_complete") is True else 2
        if args.command == "ingest":
            from .trades import ingest_condition
            if args.max_conditions is not None and args.max_conditions < 1:
                raise ValueError("--max-conditions must be positive")
            conditions = args.condition
            if args.registry:
                registry = json.loads(args.registry.read_text())
                conditions = [row["condition_id"] for row in registry["contracts"]]
            conditions = sorted(set(conditions))
            if args.max_conditions is not None:
                conditions = conditions[:args.max_conditions]
            if not conditions:
                raise ValueError("No conditions selected")
            for condition in conditions:
                manifest = ingest_condition(
                    HttpClient(args.cache), condition_id=condition,
                    output_dir=args.output, limit=args.limit,
                    max_pages=None if args.all_pages else args.max_pages,
                )
                print(json.dumps(manifest, sort_keys=True), flush=True)
            return 0
        report = audit_registry(json.loads(args.registry.read_text()), args.trades_root)
        write_json(args.output, report)
        print(json.dumps({key: value for key, value in report.items() if key not in {"ingestion_manifests", "missing_condition_ids"}}, indent=2, sort_keys=True))
        return 0 if report["structural_audit_passed"] else 2
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
