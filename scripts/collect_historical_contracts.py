#!/usr/bin/env python3
"""Collect immutable initial question and fixture metadata for the registry.

Each request is cached with a SHA-256 checked response, making retries resumable.
Inputs are untouched. RPC evidence is independently decoded on every rebuild.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.historical_contracts import (ADAPTER, CHAIN_ID, QUESTION_TOPIC, MARKET_TOPIC,
    SOURCE_URL, build_contract_record, mapping_call_specs, validate_contract_record)
from poly_world_cup.io import atomic_write, write_json, write_jsonl
from poly_world_cup.registry import canonical_team


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class RPC:
    def __init__(self, url, cache):
        self.url, self.cache = url, Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._rate_lock = threading.Lock()
        self._next_request = 0.0

    def _pace(self, call_count):
        # Count individual JSON-RPC operations, not only HTTP batches.
        with self._rate_lock:
            delay = max(0.0, self._next_request - time.monotonic())
            self._next_request = max(self._next_request, time.monotonic()) + call_count / 2
        if delay:
            time.sleep(delay)

    def batch(self, specs):
        results = [None] * len(specs)
        missing = []
        for index, spec in enumerate(specs):
            key = hashlib.sha256(canonical({"url": self.url, **spec})).hexdigest()
            path = self.cache / (key + ".json")
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["request"] != spec or saved.get("rpc_url") != self.url or saved["response_sha256"] != hashlib.sha256(canonical(saved["result"])).hexdigest():
                    raise ValueError("RPC cache integrity failure")
                results[index] = saved["result"]
            else:
                missing.append((index, spec, path))
        if missing:
            payload = [{"jsonrpc": "2.0", "id": index, **spec} for index, spec, _ in missing]
            error = None
            for attempt in range(7):
                try:
                    self._pace(len(payload))
                    req = urllib.request.Request(self.url, data=canonical(payload),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=45) as response:
                        received = json.load(response)
                    if not isinstance(received, list) or len(received) != len(missing):
                        raise ValueError("Incomplete RPC batch")
                    by_id = {row["id"]: row for row in received}
                    if len(by_id) != len(missing):
                        raise ValueError("Duplicate RPC response IDs")
                    for index, spec, path in missing:
                        result = by_id[index]
                        if "error" in result or "result" not in result:
                            raise ValueError(f"RPC {spec['method']} failed: {result.get('error')}")
                        results[index] = result["result"]
                        write_json(path, {"request": spec, "result": result["result"],
                                         "response_sha256": hashlib.sha256(canonical(result["result"])).hexdigest(),
                                         "rpc_url": self.url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat()})
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    if attempt < 6:
                        if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                            retry = exc.headers.get("Retry-After", "")
                            delay = min(55, max(5, int(retry) if retry.isdigit() else 5 * (attempt + 1)))
                            print(f"RPC rate limit: retrying after {delay}s", flush=True)
                        else:
                            delay = 1 + attempt
                        time.sleep(delay)
            if error is not None:
                raise error
        return results

    def one(self, method, params):
        return self.batch([{"method": method, "params": params}])[0]


def _parallel_batches(rpc, specs, *, size=18, workers=4, label="RPC"):
    result = []
    chunks = [specs[i:i + size] for i in range(0, len(specs), size)]
    indexed = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(rpc.batch, chunk): i for i, chunk in enumerate(chunks)}
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                indexed[futures[future]] = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            print(f"{label}: {completed}/{len(chunks)} batches", flush=True)
    for index in range(len(chunks)):
        result.extend(indexed[index])
    return result


def candidate_questions(registry, cache):
    wanted = {row["condition_id"] for row in registry["contracts"]}
    found = {}
    for path in sorted(Path(cache).glob("*.json")):
        raw = path.read_bytes()
        # This is the existing content-addressed Gamma capture format.
        if hashlib.sha256(raw).hexdigest() != path.stem:
            raise ValueError(f"Registry source body hash mismatch: {path.name}")
        page = json.loads(raw)
        if not isinstance(page, list):
            continue
        for event in page:
            for market in event.get("markets", []) if isinstance(event, dict) else []:
                condition = str(market.get("conditionId", "")).lower()
                if condition not in wanted:
                    continue
                if market.get("negRisk") is not True:
                    raise ValueError("Only audited negative-risk contracts are supported")
                question_id = str(market.get("questionID", "")).lower()
                if condition in found and found[condition]["question_id"] != question_id:
                    raise ValueError("Conflicting question IDs in original sources")
                found[condition] = {"question_id": question_id, "source_body_sha256": path.stem}
    if found.keys() != wanted:
        raise ValueError(f"Missing question IDs: {len(wanted - found.keys())}")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--gamma-cache-bodies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rpc-url", default="https://polygon.gateway.tenderly.co")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    args.output.mkdir(parents=True, exist_ok=True)
    registry = json.loads(args.registry.read_text())
    candidates = candidate_questions(registry, args.gamma_cache_bodies)
    write_json(args.output / "question_candidates.json", candidates)
    rpc = RPC(args.rpc_url, args.output / "rpc_cache")
    if int(rpc.one("eth_chainId", []), 16) != CHAIN_ID:
        raise ValueError("RPC is not Polygon mainnet")
    # This cached block freezes all log scans to one reproducible snapshot.
    snapshot = rpc.one("eth_blockNumber", [])
    question_ids = sorted({row["question_id"] for row in candidates.values()})
    market_ids = sorted({qid[:-2] + "00" for qid in question_ids})
    logs = {}
    for label, ids, topic, filter_index in [("question", question_ids, QUESTION_TOPIC, 2),
                                          ("market", market_ids, MARKET_TOPIC, 1)]:
        specs = []
        for start in range(0, len(ids), 24):
            topics = [topic, ids[start:start + 24]] if filter_index == 1 else [topic, None, ids[start:start + 24]]
            specs.append({"method": "eth_getLogs", "params": [{"fromBlock": hex(50_505_403),
                "toBlock": snapshot, "address": ADAPTER, "topics": topics}]})
        pages = _parallel_batches(rpc, specs, size=1, workers=args.workers, label=f"{label} events")
        keyed = {}
        for page in pages:
            if not isinstance(page, list):
                raise ValueError("Malformed event page")
            for log in page:
                key = log["topics"][filter_index]
                if key not in ids or key in keyed:
                    raise ValueError("Unexpected or duplicate initialization event")
                keyed[key] = log
        if keyed.keys() != set(ids):
            raise ValueError(f"Missing {label} initialization events: {len(set(ids) - keyed.keys())}")
        logs[label] = keyed
        print(f"Verified event identity count: {label}={len(keyed)}", flush=True)
    blocks = sorted({log["blockNumber"] for values in logs.values() for log in values.values()}, key=lambda x: int(x, 16))
    block_values = _parallel_batches(rpc, [{"method": "eth_getBlockByNumber", "params": [number, False]}
                                         for number in blocks], workers=args.workers, label="block headers")
    headers = dict(zip(blocks, block_values))
    all_specs = []
    for question in question_ids:
        all_specs.extend(mapping_call_specs(question, logs["question"][question]["blockNumber"]))
    # Historical eth_call is substantially slower than headers. Small batches
    # avoid public RPC timeout limits while the content cache makes retry cheap.
    all_results = _parallel_batches(rpc, all_specs, size=3, workers=args.workers, label="historical condition/token mappings")
    mappings = {question: [{**spec, "result": value} for spec, value in
                           zip(all_specs[index * 3:index * 3 + 3], all_results[index * 3:index * 3 + 3])]
                for index, question in enumerate(question_ids)}
    records = []
    fixtures = {row["fixture_id"]: row for row in registry["fixtures"]}
    title_mismatches = []
    for contract in registry["contracts"]:
        qid = candidates[contract["condition_id"]]["question_id"]
        qlog, mlog = logs["question"][qid], logs["market"][qid[:-2] + "00"]
        evidence = {"chain_id": CHAIN_ID, "adapter": ADAPTER, "rpc_url": args.rpc_url,
                    "source_code_url": SOURCE_URL, "snapshot_block_number": snapshot,
                    "question_log": qlog, "market_log": mlog,
                    "question_block": headers[qlog["blockNumber"]], "market_block": headers[mlog["blockNumber"]],
                    "mapping_calls": mappings[qid]}
        row = build_contract_record(evidence, fixture_id=contract["fixture_id"])
        if row["condition_id"] != contract["condition_id"]:
            raise ValueError("On-chain condition disagrees with registry")
        outcomes = {token["outcome"]: token["token_id"] for token in contract["tokens"]}
        if row["yes_token_id"] != outcomes["Yes"] or row["no_token_id"] != outcomes["No"]:
            raise ValueError("On-chain token labels disagree with registry")
        if row["gamma_market_id"] != contract["market_id"] or row["gamma_event_id"] != contract["event_id"]:
            raise ValueError("On-chain original metadata ID disagrees with registry")
        fixture = fixtures[contract["fixture_id"]]
        expected_teams = sorted([fixture["home_team"]["canonical_name"], fixture["away_team"]["canonical_name"]])
        actual_teams = sorted(map(canonical_team, row["fixture_title"].split(" vs. ")))
        if actual_teams != expected_teams:
            raise ValueError("Historical fixture title does not identify the audited pair")
        if row["question"] != contract["question"]:
            title_mismatches.append({"condition_id": row["condition_id"], "initial": row["question"],
                                     "current": contract["question"]})
        validate_contract_record(row)
        records.append(row)
    records.sort(key=lambda row: row["condition_id"])
    write_jsonl(args.output / "contract_evidence.jsonl", records)
    compressed = gzip.compress(b"".join(canonical(row) + b"\n" for row in records), mtime=0)
    if len(gzip.decompress(compressed).splitlines()) != len(records):
        raise ValueError("Contract evidence gzip verification failure")
    compressed_path = args.output / "contract_evidence.jsonl.gz"
    atomic_write(compressed_path, compressed)
    persisted = compressed_path.read_bytes()
    if hashlib.sha256(persisted).digest() != hashlib.sha256(compressed).digest():
        raise ValueError("Persisted contract evidence differs from written bytes")
    decoded = gzip.decompress(persisted).splitlines()
    if len(decoded) != len(records):
        raise ValueError("Persisted contract evidence is incomplete")
    for line in decoded:
        validate_contract_record(json.loads(line))
    fixture_rows = []
    for fixture_id in sorted(fixtures):
        related = [row for row in records if row["fixture_id"] == fixture_id]
        if len(related) != 3 or len({row["fixture_title"] for row in related}) != 1:
            raise ValueError("Fixture must have three consistently named initial contracts")
        first = min(related, key=lambda row: row["fixture_initialized_at_utc"])
        fixture_rows.append({"fixture_id": fixture_id, "fixture_title": first["fixture_title"],
            "pairing_known_at_utc": first["fixture_initialized_at_utc"],
            "historical_pairing_verified": True, "evidence_kind": "polygon_market_prepared",
            "evidence_id": first["evidence_id"], "condition_id": first["condition_id"],
            "canonical_teams": sorted([fixtures[fixture_id]["home_team"]["canonical_name"],
                                        fixtures[fixture_id]["away_team"]["canonical_name"]])})
    write_jsonl(args.output / "fixture_known_at.jsonl", fixture_rows)
    write_json(args.output / "report.json", {"schema_version": 1, "contract_count": len(records),
        "fixture_count": len(fixture_rows), "token_count": 2 * len(records),
        "snapshot_block_number": int(snapshot, 16), "rpc_url": args.rpc_url,
        "source_code_url": SOURCE_URL, "current_vs_initial_question_differences": title_mismatches,
        "complete": len(records) == 312 and len(fixture_rows) == 104,
        "limitations": ["RPC is trusted for canonical chain data; this is not a consensus proof",
                        "Initial metadata does not certify later rule clarifications or actor exposure",
                        "Current Gamma metadata used only to discover candidate IDs, independently checked on chain"]})
    print(f"Complete: {len(records)} contracts / {len(fixture_rows)} fixtures / {2 * len(records)} tokens", flush=True)


if __name__ == "__main__":
    main()
