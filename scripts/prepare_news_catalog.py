#!/usr/bin/env python3
"""Prepare reproducible direct-news additions and separate relevance evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.completion import coverage_for_records, eligible_target_times
from poly_world_cup.news_relevance import build_news_relevance


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def jsonl(path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path, rows):
    with path.open("xb") as raw:
        if path.name.endswith(".gz"):
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
                for row in rows:
                    stream.write(encoded(row))
        else:
            for row in rows:
                raw.write(encoded(row))


def prepare(database, registry, gdelt_news, output, *, split_policy, contract_evidence=None,
            target_times_cache=None, reviewed_links=None):
    database, gdelt_news, output = Path(database).resolve(), Path(gdelt_news).resolve(), Path(output).absolute()
    if os.path.lexists(output):
        raise FileExistsError("Refusing to replace an existing news catalog")
    source_sha, gdelt_sha = sha(database), sha(gdelt_news)
    policy_sha = hashlib.sha256(encoded(split_policy)).hexdigest()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        original = [json.loads(row[0]) for row in db.execute("SELECT record_json FROM news ORDER BY news_id")]
    additions = jsonl(gdelt_news)
    versions = set()
    for row in original + additions:
        key = (row.get("news_item_id", row["news_id"]), row.get("version_rank", 0))
        if (not isinstance(key[0], str) or not key[0] or type(key[1]) is not int or key[1] < 0
                or key in versions):
            raise ValueError("Merged news item/version identity is invalid or duplicated")
        versions.add(key)
    original_ids = {row["news_id"] for row in original}
    result = build_news_relevance(original + additions, registry, contract_records=contract_evidence,
                                 reviewed_links=reviewed_links)
    gdelt_direct_ids = {row["news_id"] for row in result["direct_links"] if row["news_id"] not in original_ids}
    sft_records = []
    for supplied in result["news_records"]:
        if supplied["news_id"] not in original_ids and supplied["news_id"] not in gdelt_direct_ids:
            continue
        row = dict(supplied)
        if row["news_id"] not in original_ids:
            # Preserve broad discovery in the evidence sidecar, while avoiding
            # redundant new global prefixes in the training context stream.
            row["context_scope"] = "fixture"
            row["historical_tournament_scope_verified"] = False
            row["tournament_scope_availability_upper_utc"] = None
            row["tournament_scope_availability_evidence"] = []
        sft_records.append(row)
    sft_records.sort(key=lambda row: row["news_id"])
    if target_times_cache and Path(target_times_cache).exists():
        with gzip.open(target_times_cache, "rt", encoding="utf-8") as stream:
            cache = json.load(stream)
        if cache.get("source_database_sha256") != source_sha or cache.get("split_policy_sha256") != policy_sha:
            raise ValueError("Target-time cache belongs to a different cohort or split policy")
        target_times = cache["target_times"]
    else:
        target_times = eligible_target_times(database, split_policy)
        if target_times_cache:
            target_times_cache = Path(target_times_cache)
            target_times_cache.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(target_times_cache, [{"source_database_sha256": source_sha,
                "split_policy_sha256": policy_sha, "target_times": target_times}])
    coverage = coverage_for_records(target_times, sft_records,
        fixture_ids=[row["fixture_id"] for row in registry["fixtures"]], news_limit=8,
        deduplicate_headlines=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".news-catalog-", dir=output.parent) as temporary:
        stage = Path(temporary)
        write_jsonl(stage / "news_for_sft.jsonl", sft_records)
        write_jsonl(stage / "all_news_candidates.jsonl.gz", sorted(result["news_records"], key=lambda row: row["news_id"]))
        write_jsonl(stage / "team_background_links.jsonl.gz", sorted(result["team_background_links"], key=lambda row: (row["fixture_id"], row["news_id"])))
        (stage / "fixture_pairing_evidence.json").write_bytes(encoded(result["fixture_pairing_evidence"]))
        if reviewed_links is not None:
            (stage / "reviewed_links.json").write_bytes(encoded(reviewed_links))
        (stage / "fixture_coverage.json").write_bytes(encoded(coverage))
        report = {**result["report"], "source_database_sha256": source_sha,
                  "input_gdelt_news_sha256": gdelt_sha, "split_policy_sha256": policy_sha,
                  "original_news_records_preserved": len(original), "added_direct_news_records": len(gdelt_direct_ids),
                  "all_candidate_item_version_identities_unique": True,
                  "sft_news_records": len(sft_records), "added_global_news_records": 0,
                  "news_limit": 8, "deduplicate_headlines": True,
                  "eligible_target_count": sum(row["eligible_targets"] for row in coverage),
                  "eligible_targets_with_selected_prior_direct_fixture_news": sum(row["eligible_targets_with_selected_prior_direct_fixture_news"] for row in coverage),
                  "fixtures_with_selected_prior_direct_fixture_news": sum(row["eligible_targets_with_selected_prior_direct_fixture_news"] > 0 for row in coverage),
                  "coverage_cohort_note": "Coverage applies to the input cohort; recompute after replacing incomplete trade captures",
                  "source_database_modified": False,
                  "artifacts": [{"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size}
                                for path in sorted(stage.iterdir())]}
        (stage / "report.json").write_bytes(encoded(report))
        if sha(database) != source_sha or sha(gdelt_news) != gdelt_sha:
            raise ValueError("News or cohort input changed during preparation")
        # Claim the new directory exclusively, then publish the report last.
        # On failure remove only files this invocation created.
        output.mkdir()
        created = []
        try:
            for path in sorted(stage.iterdir(), key=lambda path: (path.name == "report.json", path.name)):
                destination = output / path.name
                os.link(path, destination)
                created.append(destination)
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            try:
                output.rmdir()
            except OSError:
                pass
            raise
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("database", "registry", "gdelt-news", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--split-policy", type=Path, default=Path("configs/tournament_sft_v1.json"))
    parser.add_argument("--contract-evidence", type=Path)
    parser.add_argument("--target-times-cache", type=Path)
    parser.add_argument("--reviewed-links", type=Path)
    args = parser.parse_args()
    report = prepare(args.database, json.loads(args.registry.read_text()), args.gdelt_news, args.output,
        split_policy=json.loads(args.split_policy.read_text()),
        contract_evidence=jsonl(args.contract_evidence) if args.contract_evidence else None,
        target_times_cache=args.target_times_cache,
        reviewed_links=json.loads(args.reviewed_links.read_text()) if args.reviewed_links else None)
    print(json.dumps({key: value for key, value in report.items() if key not in ("fixtures", "artifacts")}, indent=2))


if __name__ == "__main__":
    main()
