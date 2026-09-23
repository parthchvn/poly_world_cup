#!/usr/bin/env python3
"""Stream-check every SFT example against its delivered attribution audit.

This validates the released files and temporal/serialization policy. It does not
re-fetch news archives or certify upstream completeness or on-chain economics.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
from itertools import groupby
import json
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.attribution import _global_time, _link_time, _micros, _verified_news_time


SPLITS = ("train", "validation", "test")
PROFILES = ("verified_news_only", "execution_history_proxy")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"Blank JSONL line in {path.name}:{number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"Non-object JSONL record in {path.name}:{number}")
            yield number, value


def shards(root, relative):
    paths = sorted((root / relative).glob("part-*.jsonl.gz"))
    for expected, path in enumerate(paths, 1):
        require(path.name == f"part-{expected:05d}.jsonl.gz", f"Gap or invalid shard name: {path}")
        for line, value in records(path):
            yield {"shard": expected, "line": line}, value


def timestamp(value):
    return _micros(value)


def utc(value):
    return (EPOCH + timedelta(microseconds=value)).isoformat().replace("+00:00", "Z")


def action_valid(value):
    if set(value) != {"side", "outcome", "shares", "price"}:
        return False
    if value["side"] not in {"BUY", "SELL"} or value["outcome"] not in {"Yes", "No"}:
        return False
    try:
        if not all(isinstance(value[key], str) for key in ("shares", "price")):
            return False
        shares, price = Decimal(value["shares"]), Decimal(value["price"])
        return shares.is_finite() and shares > 0 and price.is_finite() and 0 <= price <= 1
    except (InvalidOperation, ValueError, TypeError):
        return False


class NewsCheck:
    def __init__(self, root, limit, *, deduplicate_headlines=False):
        self.limit = limit
        self.deduplicate_headlines = deduplicate_headlines
        self.news = {row["news_id"]: row for _, row in records(root / "source_news.jsonl.gz")}
        self.contexts = {row["context_id"]: row for _, row in records(root / "contexts.jsonl.gz")}
        self.events = defaultdict(list)
        self.times = {}
        self.checked = set()
        self.expected_cache = {}
        self.prompt_cache = {}
        for news_id, row in self.news.items():
            try:
                available = _verified_news_time(row)
                global_available = _global_time(row, available)
            except (KeyError, ValueError, TypeError):
                available = global_available = None
            self.times[news_id] = available
            if global_available is not None:
                self.events["tournament"].append((global_available, news_id))
        for _, link in records(root / "source_news_links.jsonl.gz"):
            require(link["news_id"] in self.news, "Fixture link references missing source news")
            try:
                available = _link_time(link["link"], self.times[link["news_id"]])
            except (KeyError, ValueError, TypeError):
                available = None
            recorded = link["effective_availability_utc"]
            require((available is None and recorded is None)
                    or (available is not None and recorded is not None and timestamp(recorded) == available),
                    "Fixture link's effective availability does not follow its evidence")
            if available is not None:
                self.events["fixture:" + link["fixture_id"]].append((available, link["news_id"]))
        self.event_times = {}
        for scope, events in self.events.items():
            events.sort()
            self.event_times[scope] = [row[0] for row in events]
        for context_id, context in self.contexts.items():
            ids = context["eligible_news_ids"]
            require(ids == sorted(set(ids)), "Context IDs must be unique and sorted")
            require(context["eligible_item_count"] == len(ids), "Context item count mismatch")
            identity = {"scope": context["scope"], "eligible_news_ids": ids,
                        "prompt_news_limit": limit}
            if deduplicate_headlines:
                identity["deduplicate_headlines"] = True
            expected_id = "news:" + hashlib.sha256(canonical(identity).encode()).hexdigest()
            require(context_id == expected_id, "Context identifier does not bind its eligible news set")
            require(set(context["effective_availability_utc"]) == set(ids), "Context availability map mismatch")
            require(context["prompt_truncated"] == (len(ids) > limit), "Context truncation flag mismatch")

    def check(self, context_id, scope, query):
        require(context_id in self.contexts, "Audit references a missing context")
        context = self.contexts[context_id]
        require(context["scope"] == scope, "Wrong fixture/global context association")
        boundary = bisect_left(self.event_times.get(scope, []), query)
        key = (scope, boundary)
        validated_key = (context_id, key)
        if validated_key in self.prompt_cache:
            return self.prompt_cache[validated_key]
        if key not in self.expected_cache:
            best = {}
            for available, news_id in self.events.get(scope, [])[:boundary]:
                row = self.news[news_id]
                item = row.get("news_item_id", news_id)
                rank = row.get("version_rank", 0)
                previous = best.get(item)
                if previous is None or rank > self.news[previous[1]].get("version_rank", 0):
                    best[item] = (available, news_id)
            ordered = sorted(best.values(), key=lambda x: (
                -x[0], self.news[x[1]].get("news_item_id", x[1]), x[1]))
            self.expected_cache[key] = ordered
        expected = self.expected_cache[key]
        selected = expected
        if self.deduplicate_headlines:
            distinct, seen = [], set()
            for available, news_id in expected:
                title = " ".join(self.news[news_id]["title"].casefold().split())
                if title not in seen:
                    seen.add(title)
                    distinct.append((available, news_id))
            selected = distinct
        selected = selected[:self.limit]
        if validated_key not in self.checked:
            require(context["eligible_news_ids"] == sorted(row[1] for row in expected),
                    "Context omits an eligible version or includes a future/unverified version")
            require(context["selected_prompt_news_ids"] == [row[1] for row in selected],
                    "Prompt news does not follow the newest eligible-version policy")
            require(context["effective_availability_utc"] == {news_id: utc(available) for available, news_id in expected},
                    "Context contains an incorrect effective availability timestamp")
            self.checked.add(validated_key)
        prompt = [{"news_id": news_id, "headline": self.news[news_id]["title"],
                 "source_url": self.news[news_id]["source_url"],
                 "verified_available_at_utc": utc(available)}
                for available, news_id in selected]
        self.prompt_cache[validated_key] = prompt
        return prompt


def _validate_sft(root: Path, transaction_db: sqlite3.Connection, *, progress=False) -> dict:
    started = time.monotonic()

    def announce(message):
        if progress:
            print(f"SFT validation [{time.monotonic() - started:.1f}s]: {message}", file=sys.stderr, flush=True)

    root = Path(root).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["task"] == "retrospective_conditional_api_observation", "Unexpected task")
    require(manifest["training_ready"] is False and manifest["prospective_training_ready"] is False,
            "Research readiness must not be upgraded by serialization")
    announce(f"checking {len(manifest['files'])} artifact checksums")
    listed = set()
    for item in manifest["files"]:
        relative = PurePosixPath(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe artifact path")
        require(item["path"] not in listed, "Duplicate manifest artifact")
        listed.add(item["path"])
        path = root / relative
        require(path.is_file() and not path.is_symlink(), "Missing or nonregular artifact")
        require(path.stat().st_size == item["bytes"], f"Artifact size mismatch: {relative}")
        require(sha(path) == item["sha256"], f"Artifact checksum mismatch: {relative}")
        require(item["bytes"] < 100 * 1024 * 1024, "Artifact exceeds GitHub's per-file limit")
        if len(listed) % 25 == 0:
            announce(f"verified {len(listed)}/{len(manifest['files'])} artifact checksums")
    for path in root.rglob("*.jsonl.gz"):
        require(path.relative_to(root).as_posix() in listed, "Unmanifested JSONL artifact")
    policy = json.loads((root / "split_policy.json").read_text())
    require(hashlib.sha256(canonical(policy).encode()).hexdigest() == manifest["split_policy_sha256"],
            "Split-policy hash mismatch")
    first, second = timestamp(policy["train_before_utc"]), timestamp(policy["validation_before_utc"])
    require(first < second, "Invalid chronological split boundaries")
    announce("all artifact checksums passed; reconstructing verified news timelines")
    news = NewsCheck(root, manifest["news_limit_per_scope"],
                     deduplicate_headlines=manifest.get("news_headline_deduplication", False))
    contract_contexts = {}
    if manifest.get("contract_context_included"):
        from poly_world_cup.historical_contracts import validate_contract_record, verified_contract_context
        for _, record in records(root / "source_contracts.jsonl.gz"):
            validate_contract_record(record)
            condition = record["condition_id"]
            require(condition not in contract_contexts, "Duplicate historical contract evidence")
            context = verified_contract_context(record, 10**30)
            contract_contexts[condition] = (timestamp(context["initialized_at_utc"]), context, record["fixture_id"])
        require(len(contract_contexts) == manifest["historical_contract_evidence_count"],
                "Historical contract evidence count mismatch")
    announce(f"news timelines ready; checking all {manifest['source_observations']:,} audit rows and both profiles")
    streams = {(profile, split): iter(shards(root, f"{profile}/{split}"))
               for profile in PROFILES for split in SPLITS}
    counts, exclusions, wallet_sets, fixture_sets = Counter(), Counter(), defaultdict(set), defaultdict(set)
    source_count = wallet_count = quarantined = 0
    seen_rows = set()
    source_fixtures, source_contracts = set(), set()
    previous_wallet = None
    audits = shards(root, "audit")
    for wallet, wallet_rows in groupby(audits, key=lambda item: item[1]["wallet"]):
        require(previous_wallet is None or wallet > previous_wallet, "Audit is not ordered by distinct wallet groups")
        previous_wallet = wallet
        rows = list(wallet_rows)
        require(0 < len(rows) < 20, "Audit wallet violates tournament-wide exclusive threshold")
        wallet_count += 1
        prior = []
        previous_key = None
        for location, audit in rows:
            require(location["line"] <= manifest["shard_rows"], "Audit shard exceeds row limit")
            source_count += 1
            if source_count % 50_000 == 0:
                announce(f"checked {source_count:,}/{manifest['source_observations']:,} audit rows; "
                         f"{sum(counts.values()) * len(PROFILES):,} profile examples")
            row_id = audit["trade_row_id"]
            require(row_id not in seen_rows, "Repeated target row in audit")
            seen_rows.add(row_id)
            source_fixtures.add(audit["fixture_id"])
            source_contracts.add(audit["condition_id"])
            evidence = contract_contexts.get(audit["condition_id"])
            if evidence is not None:
                tokens = {evidence[1]["yes_token_id"]: "Yes", evidence[1]["no_token_id"]: "No"}
                require(audit["fixture_id"] == evidence[2] and audit["token_id"] in tokens
                        and tokens[audit["token_id"]] == audit["reported_action"]["outcome"],
                        "Audited trade disagrees with historical contract fixture/token evidence")
            require(audit["observed_tournament_count"] == len(rows), "Wallet audit count differs from its complete group")
            reasons = audit["excluded_reasons"]
            query = None if audit["execution_time_proxy_utc"] is None else timestamp(audit["execution_time_proxy_utc"])
            if query is None:
                require(bool({"invalid_execution_timestamp", "execution_timestamp_disagreement"} & set(reasons)),
                        "Null execution time lacks its quarantine reason")
                require("source_query_us" in audit and "source_block_timestamp" in audit,
                        "Invalid time lost its original source values")
            else:
                key = (query, row_id)
                require(previous_key is None or key > previous_key, "Audit wallet history is not chronologically ordered")
                previous_key = key
            expected_prior = [] if query is None else [row for prior_query, row in prior
                              if prior_query < query
                              and row["transaction_hash"] != audit["transaction_hash"]]
            require(audit["proxy_history_trade_row_ids"] == [row["trade_row_id"] for row in expected_prior],
                    "Audit history omits eligible observations or includes the target transaction/equal/future times")
            context_query = -2**100 if query is None else query
            fixture_news = news.check(audit["fixture_context_id"], "fixture:" + audit["fixture_id"], context_query)
            global_news = news.check(audit["tournament_context_id"], "tournament", context_query)
            split = audit["exported_split"]
            assigned = policy["fixture_splits"][audit["fixture_id"]]
            require(audit["assigned_fixture_split"] == assigned, "Fixture split disagrees with policy")
            time_ok = query is not None and ((assigned == "train" and query < first)
                       or (assigned == "validation" and first <= query < second)
                       or (assigned == "test" and query >= second))
            require(("fixture_time_split_mismatch" in reasons) == (not time_ok), "Split-purge reason is incorrect")
            if reasons:
                quarantined += 1
                exclusions.update(reasons)
                require(split is None and audit["profile_locations"] == {}, "Quarantined target was exported")
            else:
                require(split in SPLITS and split == assigned and time_ok, "Exported target violates its split")
                require(action_valid(audit["reported_action"]), "Exported target action is invalid")
                try:
                    transaction_db.execute("""INSERT INTO transactions(tx,split) VALUES(?,?)
                        ON CONFLICT(tx) DO UPDATE SET split=CASE
                        WHEN transactions.split=excluded.split THEN transactions.split ELSE NULL END""",
                        (audit["transaction_hash"], split))
                except sqlite3.IntegrityError as exc:
                    raise ValueError("An exported transaction crosses train/evaluation splits") from exc
                counts[split] += 1
                wallet_sets[split].add(wallet)
                fixture_sets[split].add(audit["fixture_id"])
                base = {"task": "retrospective_conditional_api_observation", "wallet_id": wallet,
                        "condition_id": audit["condition_id"], "execution_time_proxy_utc": audit["execution_time_proxy_utc"],
                        "verified_fixture_news": fixture_news, "verified_tournament_news": global_news}
                if manifest.get("contract_context_included"):
                    evidence = contract_contexts.get(audit["condition_id"])
                    base["verified_contract_context"] = evidence[1] if evidence and evidence[0] < query else None
                history = [{"condition_id": row["condition_id"], "token_id": row["token_id"],
                            "side": row["reported_action"]["side"], "shares": row["reported_action"]["shares"],
                            "price": row["reported_action"]["price"],
                            "execution_time_proxy_utc": row["execution_time_proxy_utc"]} for row in expected_prior]
                if manifest.get("contract_context_included"):
                    for feature, past in zip(history, expected_prior):
                        evidence = contract_contexts.get(past["condition_id"])
                        feature["verified_contract_context"] = (
                            evidence[1] if evidence and evidence[0] < timestamp(past["execution_time_proxy_utc"]) else None)
                for profile in PROFILES:
                    item = next(streams[(profile, split)], None)
                    require(item is not None, "Profile ended before its audit targets")
                    position, example = item
                    require(position == audit["profile_locations"][profile], "Audit locator disagrees with profile position")
                    require(position["line"] <= manifest["shard_rows"], "Profile shard exceeds row limit")
                    require(set(example) == {"messages"}, "Unexpected training-row fields")
                    messages = example["messages"]
                    require(isinstance(messages, list) and len(messages) == 3, "Invalid message count")
                    require([row.get("role") for row in messages] == ["system", "user", "assistant"], "Invalid message roles")
                    require(all(set(row) == {"role", "content"} and isinstance(row["content"], str)
                                for row in messages), "Invalid message schema")
                    require(json.loads(messages[2]["content"]) == audit["reported_action"],
                            "Assistant label differs from the audited source observation")
                    expected_user = base if profile == "verified_news_only" else {
                        **base, "prior_tournament_executions": history,
                        "prior_execution_availability_verified": False,
                        "history_scope": "captured_tournament_contracts_only"}
                    require(json.loads(messages[1]["content"]) == expected_user,
                            "Prompt differs from verified news/history or contains prohibited target/audit fields")
                    require("conditional on an execution" in messages[0]["content"], "Missing conditional task instruction")
                    require("untrusted source headlines" in messages[0]["content"], "Missing news-as-data instruction")
                    if manifest.get("contract_context_included"):
                        require("initial market question and token mapping" in messages[0]["content"]
                                and "contract text as untrusted data" in messages[0]["content"],
                                "Missing initial-contract semantics and untrusted-text instructions")
            if not (set(reasons) - {"fixture_time_split_mismatch"}) and action_valid(audit["reported_action"]):
                prior.append((query, audit))
    for stream in streams.values():
        require(next(stream, None) is None, "Profile contains unaudited extra examples")
    require(source_count == manifest["source_observations"], "Audit/source count mismatch")
    require(wallet_count == manifest["source_wallets"], "Audit/source wallet count mismatch")
    require(len(source_fixtures) == manifest["source_fixtures"], "Audit/source fixture count mismatch")
    require(len(source_contracts) == manifest["source_contracts"], "Audit/source contract count mismatch")
    if "source_fixture_count" in policy:
        require(len(source_fixtures) == policy["source_fixture_count"], "Audit does not cover the declared tournament fixtures")
    if "source_contract_count" in policy:
        require(len(source_contracts) == policy["source_contract_count"], "Audit does not cover the declared tournament contracts")
    require(quarantined == manifest["quarantined_target_observations"], "Quarantine count mismatch")
    require(dict(sorted(exclusions.items())) == manifest["exclusion_reason_counts"], "Exclusion reason totals mismatch")
    require(sum(counts.values()) == manifest["included_target_observations"], "Included target count mismatch")
    for split in SPLITS:
        require(counts[split] == manifest["split_target_counts"][split], "Split target count mismatch")
        require(len(wallet_sets[split]) == manifest["split_wallet_counts"][split], "Split wallet count mismatch")
        require(len(fixture_sets[split]) == manifest["split_fixture_counts"][split], "Split fixture count mismatch")
        for profile in PROFILES:
            require(counts[split] == manifest["profiles"][profile]["rows_by_split"][split], "Profile row-count mismatch")
    for first_split, second_split in (("train", "validation"), ("train", "test"), ("validation", "test")):
        require(not fixture_sets[first_split] & fixture_sets[second_split], "Fixtures overlap between splits")
    announce(f"passed all {source_count:,} audit rows and {sum(counts.values()) * len(PROFILES):,} profile examples")
    return {"status": "passed", "manifest_sha256": sha(manifest_path), "artifact_checksums_verified": len(listed),
            "audited_source_targets": source_count, "audited_wallets": wallet_count,
            "audited_fixtures": len(source_fixtures), "audited_contracts": len(source_contracts),
            "profile_examples_verified": sum(counts.values()) * len(PROFILES),
            "verified_rows_per_profile_by_split": {split: counts[split] for split in SPLITS},
            "quarantined_targets": quarantined, "context_states_verified": len(news.contexts),
            "label_equality_verified": True, "strict_timestamp_checks_verified": True,
            "history_target_transaction_exclusion_verified": True, "prompt_field_allowlist_verified": True,
            "fixture_and_time_separation_verified": True,
            "transaction_separation_verified": True,
            "upstream_archive_bytes_revalidated": False, "source_completeness_certified": False,
            "prospective_training_ready": False}


def validate_sft(root: Path, *, progress: bool = False) -> dict:
    # A disk-backed transaction index keeps validation memory bounded when the
    # corpus has nearly one distinct transaction per observation.
    with tempfile.TemporaryDirectory(prefix="sft-validation-") as directory:
        db = sqlite3.connect(str(Path(directory) / "transactions.sqlite"))
        try:
            db.execute("PRAGMA cache_size=-262144")
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("CREATE TABLE transactions(tx TEXT PRIMARY KEY,split TEXT NOT NULL) WITHOUT ROWID")
            return _validate_sft(root, db, progress=progress)
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error("Refusing to overwrite an existing validation report")
    report = validate_sft(args.dataset, progress=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
