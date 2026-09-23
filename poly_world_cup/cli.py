"""Command line entry points. No credentials or runtime dependencies required."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    ingest.add_argument("--minimum-size", default="0.01", help="Positive requested TOKENS threshold; zero means the provider default")
    ingest.add_argument("--compress", action="store_true", help="Gzip raw bodies and normalized observation pages")

    batch = commands.add_parser("collect-tournament", help="Collect every mapped condition with bounded concurrency and resumable progress")
    batch.add_argument("--registry", type=Path, default=Path("data/registry/registry.json"))
    batch.add_argument("--output", type=Path, default=Path("data/full/trades"))
    batch.add_argument("--cache", type=Path, default=Path("data/full/cache"))
    batch.add_argument("--workers", type=int, default=12)
    batch.add_argument("--requests-per-second", type=float, default=4.0)
    batch.add_argument("--limit", type=int, default=1000)
    batch.add_argument("--max-pages", type=int, help="Optional new-page bound per condition; omitted means traverse to API exhaustion")
    batch.add_argument("--minimum-size", default="0.000001", help="Requested positive TOKENS threshold; provider completeness remains unverified")

    news = commands.add_parser("collect-news", help="Collect retrospective ESPN metadata and candidate fixture links")
    news.add_argument("--registry", type=Path, default=Path("data/registry/registry.json"))
    news.add_argument("--output", type=Path, default=Path("data/news"))
    news.add_argument("--cache", type=Path, default=Path("data/news/cache"))
    news.add_argument("--workers", type=int, default=6)
    news.add_argument("--max-archive-pages", type=int, default=100)
    news.add_argument("--window-start", default="2026-01-01T00:00:00Z")
    news.add_argument("--window-end")

    archive = commands.add_parser("archive-news", help="Recover separately verified archived headline versions")
    archive.add_argument("--news", type=Path, default=Path("data/news/news.jsonl"))
    archive.add_argument("--output", type=Path, default=Path("data/news_archive"))
    archive.add_argument("--start", default="20260101")
    archive.add_argument("--cutoff", default="20260720235959")
    archive.add_argument("--workers", type=int, default=4)
    archive.add_argument("--max-articles", type=int)

    attribute = commands.add_parser("attribute", help="Build a queryable SQLite index from audited trades and news")
    attribute.add_argument("--registry", type=Path, default=Path("data/registry/registry.json"))
    attribute.add_argument("--trades-root", type=Path, default=Path("data/full/trades"))
    attribute.add_argument("--news", type=Path, default=Path("data/news/news.jsonl"))
    attribute.add_argument("--archive-news", type=Path, action="append", default=[])
    attribute.add_argument("--database", type=Path, default=Path("data/full/attribution.sqlite"))
    attribute.add_argument("--report", type=Path, default=Path("data/full/attribution_report.json"))
    attribute.add_argument("--allow-partial", action="store_true", help="Explicitly allow missing or paused conditions; corrupt collections still fail")

    inspect = commands.add_parser("inspect-context", help="Inspect one observation and its separately qualified context")
    inspect.add_argument("--database", type=Path, default=Path("data/full/attribution.sqlite"))
    inspect.add_argument("--row", type=int, default=1)
    inspect.add_argument("--output", type=Path)

    reconcile = commands.add_parser("reconcile-sample", help="Check a bounded fixture-stratified sample against Polygon receipts")
    reconcile.add_argument("--registry", type=Path, default=Path("data/registry/registry.json"))
    reconcile.add_argument("--trades-root", type=Path, default=Path("data/full/trades"))
    reconcile.add_argument("--output", type=Path, default=Path("data/full/reconciliation"))
    reconcile.add_argument("--max-transactions", type=int, default=104)
    reconcile.add_argument("--rpc-url", default="https://polygon.drpc.org")
    reconcile.add_argument("--requests-per-second", type=float, default=2.0)

    provenance = commands.add_parser("verify-provenance", help="Replay normalized observations from immutable raw source captures")
    provenance.add_argument("--registry", type=Path, default=Path("data/registry/registry.json"))
    provenance.add_argument("--trades-root", type=Path, default=Path("data/full/trades"))
    provenance.add_argument("--cache-root", type=Path, default=Path("data/full/cache"))
    provenance.add_argument("--cache-layout", choices=["per_condition", "shared"], default="per_condition")
    provenance.add_argument("--output", type=Path, default=Path("data/full/provenance_report.json"))

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
                    HttpClient(args.cache, compress=args.compress), condition_id=condition,
                    output_dir=args.output, limit=args.limit,
                    max_pages=None if args.all_pages else args.max_pages,
                    compress=args.compress, minimum_size=args.minimum_size,
                )
                print(json.dumps(manifest, sort_keys=True), flush=True)
            return 0
        if args.command == "collect-tournament":
            from .batch import run_batch
            registry = json.loads(args.registry.read_text())
            if not audit_registry(registry)["structural_audit_passed"]:
                raise ValueError("Resolve registry mapping errors before tournament collection")
            last_print = 0.0
            def progress(value):
                nonlocal last_print
                now = time.monotonic()
                if value["status"] != "running" or now - last_print >= 10:
                    print(json.dumps({key: value[key] for key in (
                        "updated_at", "status", "status_counts", "committed_observation_count",
                        "committed_page_count", "validated_observation_count",
                    )}), flush=True)
                    last_print = now
            report = run_batch(
                [row["condition_id"] for row in registry["contracts"]],
                output_dir=args.output, cache_dir=args.cache,
                workers=args.workers, requests_per_second=args.requests_per_second,
                limit=args.limit, max_pages_per_condition=args.max_pages,
                compress=True, minimum_size=args.minimum_size, on_progress=progress,
            )
            return 2 if report["status_counts"]["failed"] else 0
        if args.command == "collect-news":
            from .news import collect_news
            report = collect_news(
                HttpClient(args.cache), json.loads(args.registry.read_text()), args.output,
                workers=args.workers, max_archive_pages=args.max_archive_pages,
                window_start=args.window_start, window_end=args.window_end,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 2 if report.get("errors") else 0
        if args.command == "archive-news":
            from .news_archive import ArchiveCollector
            records = [json.loads(line) for line in args.news.read_text().splitlines() if line.strip()]
            # Current publication claims can reflect later republication. The
            # historical capture cutoff, not that claim, bounds availability.
            report = ArchiveCollector(args.output, cutoff=args.cutoff, start=args.start,
                                      workers=args.workers).collect(records, max_articles=args.max_articles)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "attribute":
            from .attribution import build_attribution_index
            registry = json.loads(args.registry.read_text())
            audit = audit_registry(registry, args.trades_root)
            if not audit["structural_audit_passed"]:
                raise ValueError("Collection audit failed: " + "; ".join(audit["structural_errors"][:5]))
            if not args.allow_partial and (audit["conditions_without_manifests"] or
                    audit["manifest_status_counts"].get("exhausted", 0) != audit["contract_count"]):
                raise ValueError("Attribution requires exhausted API traversals for every condition; use --allow-partial to explicitly inspect a partial corpus")
            pages = [args.trades_root / manifest["condition_id"] / page["file"]
                     for manifest in audit["ingestion_manifests"] for page in manifest["pages"]]
            records = [json.loads(line) for path in [args.news, *args.archive_news]
                       for line in path.read_text().splitlines() if line.strip()]
            report = build_attribution_index(registry=registry, trade_pages=pages,
                                             news_records=records, output_path=args.database)
            report["api_traversals_exhausted"] = audit["manifest_status_counts"].get("exhausted", 0)
            report["registry_condition_count"] = audit["contract_count"]
            report["partial_collection_allowed"] = args.allow_partial
            write_json(args.report, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "inspect-context":
            from .attribution import read_trade_context
            result = read_trade_context(args.database, args.row)
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "reconcile-sample":
            from .reconcile import run_sample
            report = run_sample(
                json.loads(args.registry.read_text()), args.trades_root, args.output,
                max_transactions=args.max_transactions, rpc_url=args.rpc_url,
                requests_per_second=args.requests_per_second,
            )
            print(json.dumps({key: value for key, value in report.items()
                              if key != "results"}, indent=2, sort_keys=True))
            return 0
        if args.command == "verify-provenance":
            from .provenance import verify_raw_provenance
            registry = json.loads(args.registry.read_text())
            def save_progress(value):
                write_json(args.output.with_name(args.output.stem + ".progress.json"), value)
            report = verify_raw_provenance(
                args.trades_root, args.cache_root,
                condition_ids=[row["condition_id"] for row in registry["contracts"]],
                cache_layout=args.cache_layout, on_progress=save_progress,
            )
            write_json(args.output, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["raw_provenance_verified"] else 2
        report = audit_registry(json.loads(args.registry.read_text()), args.trades_root)
        write_json(args.output, report)
        print(json.dumps({key: value for key, value in report.items() if key not in {"ingestion_manifests", "missing_condition_ids"}}, indent=2, sort_keys=True))
        return 0 if report["structural_audit_passed"] else 2
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
