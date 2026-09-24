#!/usr/bin/env python3
"""Prepare, but never install, a source-reconstructed conversation shard.

Intact neighboring shards establish exact sequence boundaries. Every surviving
row and actor-index hash is checked. Missing original witnesses are reported,
not invented. The immutable source and unchanged exporter rebuild that gap.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
from itertools import groupby
import json
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import time
import zlib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from poly_world_cup.actor_sequences import (
    ConversationBuilder, SequencePolicy, keep_negative, normalize_observations, scheduled_events,
)
from poly_world_cup.sequence_context import ContextCatalog
from poly_world_cup.sequence_tokens import TokenBudget
from poly_world_cup.sft import _Shards, _identity, _json, _reject_wal, _sha
from scripts.measure_sequence_tokens import load_reference_tokenizer


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_sha(row):
    return hashlib.sha256(_json(row).encode()).hexdigest()


def read_prefix(path, *, allow_truncated=False):
    """Keep complete JSON lines yielded before a damaged gzip footer/tail."""
    before = _identity(path)
    result, error = [], None
    decoder = zlib.decompressobj(31)
    body = decoder.decompress(path.read_bytes())
    if not decoder.eof:
        error = "EOFError: compressed file ends before gzip footer"
    require(not decoder.unused_data, "Unexpected extra gzip member or trailing bytes")
    partial_bytes = 0
    for line in body.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            partial_bytes = len(line)
            error = error or "EOFError: partial final JSONL row"
            break
        value = json.loads(line)
        require(isinstance(value, dict), "Non-object source row")
        result.append(value)
    if error and not allow_truncated:
        raise EOFError(error)
    require(_identity(path) == before, "Input shard changed during read: " + str(path))
    return result, {"path": str(path), **before, "sha256": _sha(path), "complete_rows": len(result),
        "decompressed_prefix_bytes": len(body), "decompressed_prefix_sha256": hashlib.sha256(body).hexdigest(),
        "partial_final_row_bytes": partial_bytes, "error": error}


def index_witnesses(dataset, relative, through):
    entries, scans = {}, []
    needle = relative.encode()
    for number in range(1, through + 1):
        path = dataset / "actor_index" / f"part-{number:05d}.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        before = _identity(path)
        count, error = 0, None
        decoder = zlib.decompressobj(31)
        body = decoder.decompress(path.read_bytes())
        require(not decoder.unused_data, "Unexpected actor-index gzip member or trailing bytes")
        if not decoder.eof:
            error = "EOFError: compressed file ends before gzip footer"
        for line_number, line in enumerate(body.splitlines(keepends=True), 1):
            if not line.endswith(b"\n"):
                error = error or "EOFError: partial final JSONL row"
                break
            count += 1
            if needle not in line:
                continue
            row = json.loads(line)
            if row["path"] == relative:
                require(row["line"] not in entries, "Duplicate original index line")
                entries[row["line"]] = row
        require(_identity(path) == before, "Actor index changed during read")
        scans.append({"path": str(path.relative_to(dataset)), **before, "sha256": _sha(path),
                      "complete_rows": count, "error": error})
    return entries, scans


def actor_rows(actor, raw, profile, split, catalog, coverage, split_policy, policy, budget):
    normalized = normalize_observations(raw, catalog, split_policy)
    source_valid = [r for r in normalized if not (set(r["errors"]) - {"fixture_time_split_mismatch"})]
    history = [r for r in source_valid if r["history_split"] == split]
    if not history:
        return []
    builder = ConversationBuilder(actor, profile, split, history, catalog, policy, budget)
    if profile == "conditional_trades":
        for query, values in groupby((r for r in history if r["split"] == split), key=lambda r: r["query_us"]):
            targets = list(values)
            builder.add(query, query, {r["condition_id"] for r in targets}, targets)
    else:
        invalid = {r["condition_id"] for r in normalized if set(r["errors"]) - {"fixture_time_split_mismatch"}}
        for query, end, conditions, targets in scheduled_events(history, coverage, split, split_policy, policy, invalid):
            if targets or keep_negative(actor, split, query, policy):
                builder.add(query, end, conditions, targets)
    builder.flush()
    return builder.records


def reconstruct(args):
    dataset, source, output_root = args.dataset.resolve(), args.source.resolve(), args.output_root.resolve()
    relative = PurePosixPath(args.relative_shard)
    require(len(relative.parts) == 3 and relative.parts[0] in ("conditional_trades", "scheduled_windows") and
            relative.parts[1] in ("train", "validation", "test"), "Expected a conversation shard path")
    profile, split, filename = relative.parts
    number = int(filename.removeprefix("part-").removesuffix(".jsonl.gz"))
    require(filename == f"part-{number:05d}.jsonl.gz" and number > 1, "Recovery requires an existing preceding shard")
    require(not output_root.is_relative_to(dataset), "Never write into the active dataset")
    destination = output_root / profile / split / f"part-{number:05d}"
    require(not destination.exists(), "Use a fresh private repair directory")
    destination.mkdir(parents=True)
    damaged = dataset / relative
    previous = dataset / profile / split / f"part-{number-1:05d}.jsonl.gz"
    following = dataset / profile / split / f"part-{number+1:05d}.jsonl.gz"
    survivors, damaged_info = read_prefix(damaged, allow_truncated=True)
    require(damaged_info["error"] is not None, "Requested shard is not demonstrably truncated")
    before_rows, before_info = read_prefix(previous)
    after_rows, after_info = read_prefix(following)
    require(len(before_rows) == args.expected_rows and after_rows, "Neighboring shard bounds are incomplete")
    before, after = before_rows[-1], after_rows[0]
    lower, upper = before["actor_id"], after["actor_id"]
    require(lower <= upper, "Reversed actor boundaries")
    witnesses, index_scans = index_witnesses(dataset, str(relative), args.closed_index_through)
    require(all(1 <= line <= args.expected_rows for line in witnesses), "Index witness outside shard extent")
    for line, row in enumerate(survivors, 1):
        if line in witnesses:
            require(canonical_sha(row) == witnesses[line]["sha256"], "Surviving data/index witnesses disagree")
    (destination / "original_index_witnesses.json").write_text(_json({str(k): v for k, v in witnesses.items()}) + "\n")
    (destination / "surviving_rows.jsonl").write_text("".join(_json(row) + "\n" for row in survivors))
    _reject_wal(source)
    source_identity = _identity(source)
    expected_source = json.loads(args.recovery_report.read_text())["source_database_sha256"]
    cache_path = output_root / "verified_source_identity.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cached.get("identity") == source_identity and cached.get("sha256") == expected_source:
        source_hash = cached["sha256"]
        source_hash_method = "reused_same_inode_size_mtime_identity_from_prior_full_hash"
    else:
        source_hash = _sha(source)
        source_hash_method = "full_sha256_read"
        require(source_hash == expected_source, "Immutable source SHA differs from recovery report")
        cache_path.write_text(_json({"identity": source_identity, "sha256": source_hash}) + "\n")
    require(source_hash == expected_source, "Incorrect source hash")
    code_paths = [REPO / p for p in ("poly_world_cup/actor_sequences.py", "poly_world_cup/sequence_context.py",
        "poly_world_cup/sequence_tokens.py", "poly_world_cup/sft.py", "scripts/measure_sequence_tokens.py")]
    code_hashes = {str(p.relative_to(REPO)): _sha(p) for p in code_paths}
    policy = SequencePolicy(**json.loads((dataset / "policy.json").read_text()))
    split_policy = json.loads((dataset / "split_policy.json").read_text())
    catalog = ContextCatalog(args.evidence, policy.initial_news_per_scope)
    tokenizer = load_reference_tokenizer(args.tokenizer, chat_template=args.chat_template)
    budget = TokenBudget(tokenizer, policy.max_tokens)
    db = sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    coverage = {r["condition_id"]: dict(r) for r in db.execute("SELECT * FROM condition_coverage")}
    extra = "" if policy.filter_scope == "actor_market" else (
        " AND NOT EXISTS (SELECT 1 FROM wallet_market_counts z WHERE z.wallet=t.wallet AND z.observation_count>?)")
    sql = """SELECT t.* FROM trades t JOIN wallet_market_counts w
        ON w.wallet=t.wallet AND w.condition_id=t.condition_id
        WHERE t.wallet>=? AND t.wallet<=? AND w.observation_count<=?""" + extra + " ORDER BY t.wallet,t.query_us,t.observation_id,t.trade_row_id"
    parameters = [lower, upper, policy.max_trades_per_market] + ([policy.max_trades_per_market] if extra else [])
    rebuilt, actor_count = [], 0
    start = time.monotonic()
    try:
        for actor, rows in groupby(db.execute(sql, parameters), key=lambda r: r["wallet"]):
            rebuilt.extend(actor_rows(actor, [dict(r) for r in rows], profile, split,
                                      catalog, coverage, split_policy, policy, budget))
            actor_count += 1
            if actor_count % 100 == 0:
                print(f"{relative}: actors={actor_count} reconstructed={len(rebuilt)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    finally:
        db.close()
    positions = {r["sequence_id"]: i for i, r in enumerate(rebuilt)}
    require(len(positions) == len(rebuilt), "Duplicate regenerated sequence IDs")
    require(before["sequence_id"] in positions and after["sequence_id"] in positions, "Regenerated neighbors not found")
    begin, end = positions[before["sequence_id"]], positions[after["sequence_id"]]
    require(canonical_sha(rebuilt[begin]) == canonical_sha(before) and canonical_sha(rebuilt[end]) == canonical_sha(after),
            "Regenerated neighboring boundary row changed")
    replacement = rebuilt[begin + 1:end]
    require(len(replacement) == args.expected_rows, f"Boundary interval has {len(replacement)} rows, expected {args.expected_rows}")
    verified = set()
    for line, row in enumerate(survivors, 1):
        require(row == replacement[line-1], f"Surviving original row changed at line {line}")
        verified.add(line)
    new_index = []
    for line, row in enumerate(replacement, 1):
        entry = {k: row[k] for k in ("sequence_id", "actor_id", "profile", "split", "chunk_index")}
        entry.update(path=str(relative), line=line, sha256=canonical_sha(row))
        new_index.append(entry)
        if line in witnesses:
            require(entry == witnesses[line], f"Original actor-index witness differs at line {line}")
            verified.add(line)
    writer = _Shards(destination / "replacement", args.expected_rows)
    writer.number = number - 1
    for row in replacement:
        writer.write(row)
    writer.close()
    prepared = destination / "replacement" / filename
    reread, prepared_info = read_prefix(prepared)
    require(reread == replacement and len(reread) == args.expected_rows, "Prepared gzip failed independent decode comparison")
    require(prepared.read_bytes().startswith(damaged.read_bytes()),
            "Prepared deterministic gzip differs from the original damaged compressed prefix")
    require(_identity(source) == source_identity, "Source changed during reconstruction")
    _reject_wal(source)
    require(all(_sha(REPO / name) == digest for name, digest in code_hashes.items()), "Reconstruction code changed while running")
    require(_identity(damaged) == {k: damaged_info[k] for k in ("bytes", "mtime_ns", "inode", "device")}, "Damaged source was modified")
    (destination / "reconstructed_index_entries.jsonl").write_text("".join(_json(row) + "\n" for row in new_index))
    report = {"status": "verified_private_replacement_not_installed", "relative_shard": str(relative),
        "prepared": str(prepared), "prepared_bytes": prepared_info["bytes"], "prepared_sha256": prepared_info["sha256"],
        "rows": len(replacement), "reconstructed_actors": actor_count, "actor_range": [lower, upper],
        "source_sha256": source_hash, "source_hash_method": source_hash_method, "source_identity": source_identity,
        "code_sha256": code_hashes, "original_damaged_shard": damaged_info,
        "original_index_witness_count": len(witnesses), "surviving_original_rows_compared": len(survivors),
        "rows_with_direct_original_witness": len(verified), "rows_without_direct_original_witness": sorted(set(range(1,args.expected_rows+1))-verified),
        "boundary_rows_compared": 2, "boundary_sequence_ids": [before["sequence_id"], after["sequence_id"]],
        "original_compressed_prefix_matches": True,
        "intact_neighbors": [before_info, after_info], "actor_index_scans": index_scans,
        "elapsed_seconds": time.monotonic()-start,
        "limitations": ["No installed or published data was modified", "Missing original row/index witnesses are reconstructed from the immutable source between independently verified intact neighboring sequences", "Whole-release independent validation must run after installation"]}
    (destination / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("status","relative_shard","prepared","prepared_sha256","rows","rows_with_direct_original_witness","original_index_witness_count")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-shard", required=True)
    parser.add_argument("--dataset", type=Path, default=REPO / "datasets/world_cup_2026_actor_sequences_v3")
    parser.add_argument("--source", type=Path, default=REPO / "data/sequence_v3_recovery/source.sqlite")
    parser.add_argument("--evidence", type=Path, default=REPO / "datasets/world_cup_2026_tournament_lt20_v2_evidence")
    parser.add_argument("--recovery-report", type=Path, default=REPO / "data/sequence_v3_recovery/recovery_report.json")
    parser.add_argument("--output-root", type=Path, default=REPO / "data/sequence_v3_repairs/prepared")
    parser.add_argument("--tokenizer", type=Path, default=REPO / "data/tokenizer_reference/qwen3_06b")
    parser.add_argument("--chat-template", type=Path, default=REPO / "configs/actor_sequence_chat_template.jinja")
    parser.add_argument("--closed-index-through", type=int, required=True)
    parser.add_argument("--expected-rows", type=int, default=2000)
    reconstruct(parser.parse_args())


if __name__ == "__main__":
    main()
