#!/usr/bin/env python3
"""Recover the exactly-20 cohort omitted by the original strict-<20 archive.

Each wallet/contract is a disjoint capture, never merged with another capture
of that pair. Exhaustion proves only equality to the published API count ledger;
it does not prove on-chain completeness, historical observability or inactivity.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly_world_cup.batch import RequestPacer
from poly_world_cup.http import FetchResult, HttpClient, request_url
from poly_world_cup.trades import (
    ADDRESS, API_URL, HEX_32, TradeIngestionError, _atomic_json,
    _atomic_write, _json_bytes, _validated_page, _writer_lock,
)


def parameters(wallet: str, condition_id: str) -> dict:
    if not ADDRESS.fullmatch(wallet) or not HEX_32.fullmatch(condition_id):
        raise TradeIngestionError("Malformed requested wallet or condition")
    return {"condition": condition_id, "user": wallet, "filter_type": "TOKENS",
            "filter_amount": "0.000001", "taker_only": False, "limit": 1000}


def validate_response(result, wallet: str, condition_id: str, token_ids: set[str]) -> list[dict]:
    expected_url = request_url(API_URL, parameters(wallet, condition_id))
    if result.url != expected_url:
        raise TradeIngestionError("Response URL does not match the requested pair and filters")
    rows, cursor, has_more = _validated_page(result, condition_id, None, [])
    if has_more or cursor is not None or len(rows) != 20:
        raise TradeIngestionError("Exactly-20 query must return 20 observations and terminate")
    if any(row["proxy_wallet"] != wallet for row in rows):
        raise TradeIngestionError("Query returned a different wallet")
    if any(row["token_id"] not in token_ids for row in rows):
        raise TradeIngestionError("Query returned a token absent from the contract registry")
    if any(Decimal(row["size"]) < Decimal("0.000001") for row in rows):
        raise TradeIngestionError("Query returned a below-threshold observation")
    return rows


def recover_pair(wallet: str, condition_id: str, *, output_dir: Path,
                 cache_dir: Path, token_ids: set[str], pacer=None) -> dict:
    pair_dir = output_dir / condition_id / wallet
    pair_cache = cache_dir / condition_id / wallet
    pair_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pair_dir / "manifest.json"
    params = parameters(wallet, condition_id)
    expected_url = request_url(API_URL, params)
    request_hash = hashlib.sha256(expected_url.encode()).hexdigest()
    with _writer_lock(pair_dir):
        # A saved result must replay its original raw cache, never silently fetch
        # a replacement response after provenance files have gone missing.
        if manifest_path.exists() and not (pair_cache / "requests" / f"{request_hash}.json").is_file():
            raise TradeIngestionError("Saved pair has no original raw request cache")
        if pacer is not None:
            pacer.acquire()
        client = HttpClient(pair_cache, compress=True)
        capture = client.get_json(API_URL, params)
        result = client.get_json(API_URL, params)
        if (not result.from_cache or result.url != capture.url
                or result.body_sha256 != capture.body_sha256
                or result.retrieved_at != capture.retrieved_at):
            raise TradeIngestionError("Raw response cache does not replay the captured response")
        rows = validate_response(result, wallet, condition_id, token_ids)
        normalized = b"".join(_json_bytes(row) for row in rows)
        compressed = gzip.compress(normalized, compresslevel=6, mtime=0)
        times = [row["block_timestamp"] for row in rows]
        manifest = {
            "schema_version": 1, "source_scope": "exactly20_wallet_condition_pair",
            "wallet": wallet, "condition_id": condition_id, "parameters": params,
            "request_url": expected_url, "body_sha256": result.body_sha256,
            "retrieved_at": result.retrieved_at,
            "raw_cache_relative_path": f"{condition_id}/{wallet}",
            "raw_request_relative_path": f"requests/{request_hash}.json",
            "normalized_file": "observations.jsonl.gz",
            "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "row_count": len(rows), "expected_ledger_count": 20,
            "earliest_block_timestamp": min(times), "latest_block_timestamp": max(times),
            "api_traversal_status": "exhausted", "raw_replay_verified": True,
            "training_coverage_certified": False,
            "limitations": [
                "API exhaustion does not establish on-chain completeness or zero trading",
                "Historical public availability and canonical fill identity are unknown",
                "This capture replaces only a pair absent from the original strict-<20 archive",
            ],
        }
        data_path = pair_dir / manifest["normalized_file"]
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            if saved != manifest or data_path.read_bytes() != compressed:
                raise TradeIngestionError("Saved pair differs from the original raw response replay")
        else:
            _atomic_write(data_path, compressed)
            _atomic_json(manifest_path, manifest)
        return {"wallet": wallet, "condition_id": condition_id, "row_count": len(rows),
                "manifest": str(manifest_path.relative_to(output_dir)),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}


def verify_saved_pair(root: Path, cache: Path, wallet: str, condition: str,
                      tokens: set[str]) -> tuple[list[dict], dict]:
    """Read-only raw replay for the mixed-source SQLite importer. Never fetch."""
    root, cache = Path(root), Path(cache)
    pair_dir = root / condition / wallet
    manifest_path = pair_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    params = parameters(wallet, condition)
    expected_url = request_url(API_URL, params)
    request_hash = hashlib.sha256(expected_url.encode()).hexdigest()
    pair_cache = cache / condition / wallet
    metadata_path = pair_cache / "requests" / f"{request_hash}.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["url"] != expected_url or metadata.get("body_compression") != "gzip":
        raise TradeIngestionError("Raw request cache does not bind the expected pair query")
    raw_path = pair_cache / "bodies" / f"{metadata['body_sha256']}.json.gz"
    raw = gzip.decompress(raw_path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != metadata["body_sha256"]:
        raise TradeIngestionError("Raw response checksum mismatch")
    result = FetchResult(json.loads(raw, parse_float=Decimal), expected_url,
                         metadata["retrieved_at"], metadata["body_sha256"], True)
    rows = validate_response(result, wallet, condition, set(tokens))
    normalized = b"".join(_json_bytes(row) for row in rows)
    normalized_path = pair_dir / "observations.jsonl.gz"
    saved_compressed = normalized_path.read_bytes()
    if gzip.decompress(saved_compressed) != normalized:
        raise TradeIngestionError("Normalized observations differ from raw replay")
    expected = {
        "schema_version": 1, "source_scope": "exactly20_wallet_condition_pair",
        "wallet": wallet, "condition_id": condition, "parameters": params,
        "request_url": expected_url, "body_sha256": result.body_sha256,
        "retrieved_at": result.retrieved_at,
        "raw_cache_relative_path": f"{condition}/{wallet}",
        "raw_request_relative_path": f"requests/{request_hash}.json",
        "normalized_file": "observations.jsonl.gz",
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "compressed_sha256": hashlib.sha256(saved_compressed).hexdigest(),
        "row_count": 20, "expected_ledger_count": 20,
        "earliest_block_timestamp": min(r["block_timestamp"] for r in rows),
        "latest_block_timestamp": max(r["block_timestamp"] for r in rows),
        "api_traversal_status": "exhausted", "raw_replay_verified": True,
        "training_coverage_certified": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise TradeIngestionError("Saved manifest differs from verified raw pair replay")
    return rows, {
        "page": {key: manifest[key] for key in (
            "normalized_sha256", "request_url", "body_sha256", "retrieved_at")},
        "normalized_path": str(normalized_path), "raw_path": str(raw_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def selected_pairs(evidence: Path) -> list[tuple[str, str]]:
    replacements = {row["condition_id"] for row in json.loads(
        (evidence / "replacement_collection.json").read_text())["conditions"]}
    pairs = []
    for path in sorted((evidence / "trade_count_ledger").glob("part-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["observation_count"] == 20 and row["condition_id"] not in replacements:
                    pairs.append((row["wallet"], row["condition_id"]))
    if len(set(pairs)) != len(pairs):
        raise TradeIngestionError("Duplicate exactly-20 pairs in the ledger")
    return sorted(pairs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("datasets/world_cup_2026_tournament_lt20_v2_evidence"))
    parser.add_argument("--output", type=Path, default=Path("data/sequence_v3_recovery/exact20"))
    parser.add_argument("--cache", type=Path, default=Path("data/sequence_v3_recovery/exact20_cache"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--requests-per-second", type=float, default=6)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 64:
        parser.error("workers must be between 1 and 64")
    pacer = RequestPacer(args.requests_per_second)
    registry = json.loads((args.evidence / "registry.json").read_text())
    tokens = {c["condition_id"]: {t["token_id"] for t in c["tokens"]} for c in registry["contracts"]}
    pairs = selected_pairs(args.evidence)
    if not pairs:
        raise TradeIngestionError("No exactly-20 pairs were selected from the ledger")
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"status": "collecting", "expected_pairs": len(pairs), "expected_rows": len(pairs) * 20}), flush=True)
    results, failures = [], []
    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(recover_pair, wallet, condition, output_dir=args.output,
                             cache_dir=args.cache, token_ids=tokens[condition], pacer=pacer):
                 (wallet, condition) for wallet, condition in pairs}
        for completed, future in enumerate(as_completed(tasks), 1):
            wallet, condition = tasks[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failure = {"wallet": wallet, "condition_id": condition, "error": str(exc)}
                failures.append(failure)
                print(json.dumps({"failed_pair": failure}), flush=True)
            if completed % 100 == 0 or completed == len(tasks):
                print(json.dumps({"completed": completed, "expected": len(tasks), "failures": len(failures)}), flush=True)
    report = {"schema_version": 1, "status": "complete" if not failures else "failed",
              "started_at": started,
              "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "expected_pair_count": len(pairs), "recovered_pair_count": len(results),
              "recovered_observation_count": sum(row["row_count"] for row in results),
              "training_coverage_certified": False,
              "pairs": sorted(results, key=lambda row: (row["wallet"], row["condition_id"])),
              "failures": failures}
    report_path = args.output / ("failed_collection_report.json" if failures else "collection_report.json")
    if failures:
        (args.output / "collection_report.json").unlink(missing_ok=True)
    _atomic_json(report_path, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"pairs", "failures"}}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
