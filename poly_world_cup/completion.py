"""Fail-closed tournament coverage checks for a retrospective SFT release.

This gate tests a finite, filtered API capture and direct fixture-news coverage.
It does not certify chain completeness, news-feed completeness, actor exposure,
decision timing, prospective cohort selection, or human identity.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .attribution import _link_time, _micros, _verified_news_time
from .sft import (_Timeline, _assigned_split, _cross_split_transactions, _identity,
                  _json, _reject_wal, _sha, _split_policy, _target_errors, _utc)


def _load(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _policy(value=None):
    return _load(value or Path(__file__).resolve().parents[1] / "configs/tournament_sft_v1.json")


def _connect(path):
    path = Path(path).resolve(strict=True)
    _reject_wal(path)
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-65536")
    return db


def _eligible_targets(db, split_policy):
    fixtures = {row[0] for row in db.execute("SELECT DISTINCT fixture_id FROM trades")}
    assignments, first, second = _split_policy(split_policy, fixtures)
    duplicates = {row[0] for row in db.execute(
        "SELECT observation_id FROM trades GROUP BY observation_id HAVING COUNT(*)>1")}
    cross = _cross_split_transactions(db, assignments, first, second)
    for row in db.execute("SELECT * FROM trades"):
        if (_target_errors(row) or row["observation_id"] in duplicates
                or row["transaction_hash"] in cross):
            continue
        split = _assigned_split(row["fixture_id"], row["query_us"], assignments, first, second)
        if split is not None:
            yield row, split


def eligible_target_times(database, split_policy=None) -> dict[str, list[int]]:
    """Cache exact SFT-eligible query times once for iterative news discovery.

    Uses the exporter's target, duplicate, transaction, and fixture/time rules.
    Lists contain one entry per eligible observation, including equal times.
    No source database is changed. The default policy is the frozen v1 policy.
    """
    db = _connect(database)
    try:
        result = defaultdict(list)
        for row, _ in _eligible_targets(db, _policy(split_policy)):
            result[row["fixture_id"]].append(row["query_us"])
        return {fixture: sorted(times) for fixture, times in result.items()}
    finally:
        db.close()


def is_direct_fixture_link(link: dict) -> bool:
    """Recognize explicit fixture evidence; never promote team background."""
    if link.get("link_type") in {"team_background", "tournament"}:
        return False
    return (link.get("relationship") == "direct_match"
            or link.get("relevance_basis") == "archived_direct_game_url"
            or (link.get("relationship") == "explicit_captured_matchup"
                and link.get("link_type") == "direct_fixture"
                and link.get("direct_fixture_relevance_verified") is True))


def _verified_events(records, links=None):
    records = list(records)
    by_id, times = {}, {}
    for value in records:
        record = dict(value)
        identity = record["news_id"]
        if identity in by_id:
            raise ValueError("Duplicate news identity: " + identity)
        record.setdefault("news_item_id", identity)
        record.setdefault("version_rank", 0)
        try:
            available = _verified_news_time(record)
        except (KeyError, TypeError, ValueError):
            available = None
        if (type(record["version_rank"]) is not int or record["version_rank"] < 0
                or not isinstance(record.get("title"), str) or not record["title"].strip()
                or not isinstance(record.get("source_url"), str)
                or not record["source_url"].startswith(("https://", "http://"))):
            available = None
        by_id[identity], times[identity] = record, available
    if links is None:
        links = ({"news_id": record["news_id"], "fixture_id": link["fixture_id"], "link": link}
                 for record in records for link in record.get("fixture_links", []))
    events, direct, all_links = defaultdict(list), {}, {}
    for value in links:
        fixture, identity, link = value["fixture_id"], value["news_id"], value["link"]
        if identity not in by_id:
            raise ValueError("Fixture link references absent news: " + identity)
        if link.get("fixture_id", fixture) != fixture:
            raise ValueError("Fixture link identity mismatch")
        try:
            available = _link_time(link, times[identity])
        except (KeyError, TypeError, ValueError):
            available = None
        if available is None:
            continue
        key = (fixture, identity)
        if key in all_links:
            raise ValueError("Duplicate fixture/news link")
        all_links[key] = available
        events[fixture].append((available, by_id[identity]))
        if is_direct_fixture_link(link):
            direct[key] = available
    return events, direct, all_links


def coverage_for_records(target_times: dict[str, list[int]], news_records, *,
                         fixture_links=None, fixture_ids=None, news_limit=8,
                         deduplicate_headlines=False) -> list[dict]:
    """Measure usable direct news with exact version selection and prompt limit.

    ``target_times`` can be reused across archive retrieval iterations. News rows
    may carry fixture_links, or callers may provide normalized link descriptors.
    Merely having earlier news outside the selected prompt does not count.
    """
    events, direct, _ = _verified_events(news_records, fixture_links)
    fixtures = set(fixture_ids or ()) | set(target_times) | set(events)
    result = []
    for fixture in sorted(fixtures):
        times = sorted(target_times.get(fixture, []))
        timeline = _Timeline("fixture:" + fixture, events.get(fixture, []), news_limit,
                             **({"deduplicate_headlines": True} if deduplicate_headlines else {}))
        covered = 0
        for index, start in enumerate(timeline.times):
            state = timeline.states[index]
            if not any((fixture, row["news_id"]) in direct for row in state["prompt"]):
                continue
            lower = bisect_right(times, start)
            upper = (bisect_right(times, timeline.times[index + 1])
                     if index + 1 < len(timeline.times) else len(times))
            covered += upper - lower
        availability = [time for (identity, _), time in direct.items() if identity == fixture]
        result.append({
            "fixture_id": fixture, "eligible_targets": len(times),
            "eligible_targets_with_selected_prior_direct_fixture_news": covered,
            "eligible_targets_without_selected_prior_direct_fixture_news": len(times) - covered,
            "eligible_direct_fixture_news_coverage_fraction": covered / len(times) if times else 0.0,
            "verified_direct_fixture_news_versions": len(availability),
            "earliest_direct_news_available_utc": _utc(min(availability)) if availability else None,
            "latest_direct_news_available_utc": _utc(max(availability)) if availability else None,
            "earliest_eligible_target_utc": _utc(times[0]) if times else None,
            "latest_eligible_target_utc": _utc(times[-1]) if times else None,
        })
    return result


def _database_news(db):
    records = []
    for row in db.execute("SELECT news_id,item_id,version_rank,record_json FROM news"):
        record = json.loads(row["record_json"])
        if (record.get("news_id") != row["news_id"]
                or record.get("news_item_id", record["news_id"]) != row["item_id"]
                or record.get("version_rank", 0) != row["version_rank"]):
            raise ValueError("News record identity disagrees with database")
        records.append(record)
    links = [{"fixture_id": row["fixture_id"], "news_id": row["news_id"],
              "link": json.loads(row["link_json"])} for row in db.execute("SELECT * FROM fixture_news")]
    return records, links


def _jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Incomplete JSONL record: " + str(path))
            yield json.loads(line)


def assess(database, registry, release=None, *, expected_fixtures=104,
           expected_contracts=312, expected_tokens=624, split_policy=None,
           news_limit=8, deduplicate_headlines=False, progress=None) -> dict:
    """Assess coverage; return explicit failures instead of broad readiness claims.

    A release directory is required to pass ``all104_fixture_context_ready``.
    Before export, eligible source coverage is useful for archive densification.
    This gate complements the full SFT format/content validator; it does not
    replace it. Callback ``progress(message)`` is optional for long scans.
    """
    announce = progress or (lambda message: None)
    database = Path(database).resolve(strict=True)
    before = _identity(database)
    registry = _load(registry)
    manifest = _load(Path(release) / "manifest.json") if release is not None else None
    policy = _policy(split_policy or (Path(release) / "split_policy.json" if release else None))
    checks, details = {}, {}
    fixtures = [row["fixture_id"] for row in registry["fixtures"]]
    contracts = registry["contracts"]
    conditions = [row["condition_id"].lower() for row in contracts]
    tokens = [str(token["token_id"]) for row in contracts for token in row["tokens"]]
    fixture_set, condition_set = set(fixtures), set(conditions)
    multiplicity = Counter(row["fixture_id"] for row in contracts)
    checks["registry_universe_valid"] = (
        len(fixtures) == len(fixture_set) == expected_fixtures
        and len(conditions) == len(condition_set) == expected_contracts
        and len(tokens) == len(set(tokens)) == expected_tokens
        and set(multiplicity) == fixture_set
        and expected_contracts % expected_fixtures == 0
        and set(multiplicity.values()) == {expected_contracts // expected_fixtures}
        and all(len(row["tokens"]) == 2 and {t["outcome"] for t in row["tokens"]} == {"Yes", "No"}
                for row in contracts))
    details["registry"] = {"fixtures": len(fixtures), "unique_fixtures": len(fixture_set),
                           "contracts": len(conditions), "unique_contracts": len(condition_set),
                           "tokens": len(tokens), "unique_tokens": len(set(tokens))}
    db = _connect(database)
    try:
        metadata = {row[0]: json.loads(row[1]) for row in db.execute("SELECT * FROM metadata")}
        source_report = metadata.get("report", {})
        checks["all_contract_api_traversals_exhausted"] = (
            source_report.get("partial_api_collection") is False
            and source_report.get("api_traversals_exhausted") == expected_contracts
            and source_report.get("registry_condition_count") == expected_contracts)
        replacement = source_report.get("replacement_contract_count", 0)
        checks["replacement_capture_checks_passed"] = (not replacement or all(
            source_report.get(key) is True for key in (
                "old_wallet_contract_counts_monotone_verified", "old_retained_multiset_containment_verified")))
        ledger = metadata.get("contract_exhaustion_ledger")
        if ledger is not None:
            checks["contract_capture_ledger_valid"] = (
                isinstance(ledger, list) and len(ledger) == expected_contracts
                and {row.get("condition_id") for row in ledger} == condition_set
                and all(row.get("status") == "api_exhausted"
                        and row.get("provenance_scope") in {"inherited", "fresh"}
                        and all(isinstance(row.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", row[key])
                                for key in ("manifest_sha256", "checkpoint_sha256")) for row in ledger))
        else:
            checks["contract_capture_ledger_valid"] = False
        details["api_capture"] = {
            "api_traversals_exhausted": source_report.get("api_traversals_exhausted"),
            "inherited_exhausted_contracts": source_report.get("old_exhausted_contracts_inherited"),
            "replacement_contracts": replacement, "per_contract_ledger_present": ledger is not None,
            "scope": "finite minimum-size-filtered API traversals; not canonical chain completeness",
            "requested_minimum_size_tokens": source_report.get("requested_minimum_size_tokens"),
        }
        db.execute("CREATE TEMP TABLE expected_tokens(condition_id TEXT,token_id TEXT,fixture_id TEXT,outcome TEXT)")
        db.executemany("INSERT INTO expected_tokens VALUES(?,?,?,?)", (
            (row["condition_id"].lower(), str(token["token_id"]), row["fixture_id"], token["outcome"])
            for row in contracts for token in row["tokens"]))
        db.execute("CREATE INDEX expected_tokens_key ON expected_tokens(condition_id,token_id)")
        invalid_mapping = db.execute("""SELECT COUNT(*) FROM trades t WHERE NOT EXISTS (
            SELECT 1 FROM expected_tokens e WHERE e.condition_id=t.condition_id AND e.token_id=t.token_id
            AND e.fixture_id=t.fixture_id AND e.outcome=t.token_outcome)""").fetchone()[0]
        actual_fixtures = {row[0] for row in db.execute("SELECT DISTINCT fixture_id FROM trades")}
        actual_contracts = {row[0] for row in db.execute("SELECT DISTINCT condition_id FROM trades")}
        checks["selected_rows_cover_registry"] = actual_fixtures == fixture_set and actual_contracts == condition_set
        checks["selected_token_mappings_valid"] = invalid_mapping == 0
        details["selected_missing_fixture_ids"] = sorted(fixture_set - actual_fixtures)
        details["selected_missing_condition_ids"] = sorted(condition_set - actual_contracts)
        details["invalid_token_mapping_rows"] = invalid_mapping
        announce("Checking full count ledger and retained wallet histories")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        count_columns = {row[1] for row in db.execute("PRAGMA table_info(wallet_counts)")}
        count_column = next((name for name in ("observed_count", "observation_count", "tournament_observation_count")
                             if name in count_columns), None)
        if count_column is None:
            raise ValueError("Missing wallet count column")
        checks["global_under20_filter_declared"] = (
            source_report.get("threshold_scope") == "tournament"
            and source_report.get("threshold_exclusive") == 20
            and source_report.get("full_wallet_history_included") is True)
        invalid_counts = db.execute(f"SELECT COUNT(*) FROM wallet_counts WHERE typeof({count_column})<>'integer' OR {count_column}<1").fetchone()[0]
        count_mismatch = db.execute(f"""WITH actual AS (SELECT wallet,COUNT(*) n FROM trades GROUP BY wallet)
            SELECT COUNT(*) FROM wallet_counts w LEFT JOIN actual a USING(wallet)
            WHERE (w.{count_column}<20 AND COALESCE(a.n,0)<>w.{count_column})
               OR (w.{count_column}>=20 AND COALESCE(a.n,0)<>0)""").fetchone()[0]
        unknown_wallets = db.execute("SELECT COUNT(*) FROM trades t LEFT JOIN wallet_counts w USING(wallet) WHERE w.wallet IS NULL").fetchone()[0]
        checks["retained_wallet_histories_match_full_counts"] = invalid_counts == count_mismatch == unknown_wallets == 0
        counted_wallets, counted_observations = db.execute(
            f"SELECT COUNT(*),COALESCE(SUM({count_column}),0) FROM wallet_counts").fetchone()
        selected_observations = db.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        checks["counts_match_capture_report"] = (
            counted_wallets == source_report.get("source_wallets")
            and counted_observations == source_report.get("source_observations")
            and selected_observations == source_report.get("observation_count"))
        checks["all_wallet_contract_count_ledger_valid"] = False
        if "wallet_market_counts" in tables:
            db.execute("CREATE TEMP TABLE ledger_totals AS SELECT wallet,SUM(observation_count) n FROM wallet_market_counts GROUP BY wallet")
            db.execute("CREATE UNIQUE INDEX ledger_totals_key ON ledger_totals(wallet)")
            mismatch = db.execute(f"""SELECT COUNT(*) FROM wallet_counts w LEFT JOIN ledger_totals l USING(wallet)
                WHERE l.n IS NULL OR w.{count_column}<>l.n""").fetchone()[0]
            extra = db.execute("SELECT COUNT(*) FROM ledger_totals l LEFT JOIN wallet_counts w USING(wallet) WHERE w.wallet IS NULL").fetchone()[0]
            invalid = db.execute("SELECT COUNT(*) FROM wallet_market_counts WHERE typeof(observation_count)<>'integer' OR observation_count<1").fetchone()[0]
            ledger_conditions = {row[0] for row in db.execute("SELECT DISTINCT condition_id FROM wallet_market_counts")}
            selected_pair_mismatch = db.execute("""SELECT COUNT(*) FROM (
                SELECT wallet,condition_id,COUNT(*) n FROM trades GROUP BY wallet,condition_id
                ) t LEFT JOIN wallet_market_counts w USING(wallet,condition_id)
                WHERE w.observation_count IS NULL OR w.observation_count<>t.n""").fetchone()[0]
            checks["all_wallet_contract_count_ledger_valid"] = (
                mismatch == extra == invalid == selected_pair_mismatch == 0 and ledger_conditions == condition_set)
        details["wallet_counts"] = {"invalid_counts": invalid_counts, "retained_count_mismatches": count_mismatch,
                                    "unknown_selected_wallet_rows": unknown_wallets,
                                    "source_wallets": counted_wallets, "source_observations": counted_observations,
                                    "selected_observations": selected_observations,
                                    "full_contract_ledger_present": "wallet_market_counts" in tables}
        announce("Computing eligible targets with the frozen split policy")
        contract_records, contract_errors = {}, []
        contract_coverage = defaultdict(Counter)
        registry_contracts = {record["condition_id"].lower(): record for record in contracts}
        if "contract_evidence" in tables:
            from .historical_contracts import validate_contract_record, verified_contract_context
            for value in db.execute("SELECT condition_id,record_json FROM contract_evidence"):
                try:
                    record = validate_contract_record(json.loads(value["record_json"]))
                    if record["condition_id"] != value["condition_id"]:
                        raise ValueError("Contract evidence identity disagrees with database")
                    expected = registry_contracts.get(record["condition_id"])
                    if expected is None or record["fixture_id"] != expected["fixture_id"]:
                        raise ValueError("Historical contract fixture mapping disagrees with registry")
                    token_mapping = {token["outcome"]: str(token["token_id"]) for token in expected["tokens"]}
                    if record["yes_token_id"] != token_mapping["Yes"] or record["no_token_id"] != token_mapping["No"]:
                        raise ValueError("Historical token mapping disagrees with registry")
                    contract_records[value["condition_id"]] = record
                except (KeyError, TypeError, ValueError) as exc:
                    contract_errors.append({"condition_id": value["condition_id"], "error": str(exc)})
        checks["provided_historical_contract_records_valid"] = not contract_errors
        db.execute("CREATE TEMP TABLE eligible_targets(trade_row_id INTEGER PRIMARY KEY,split TEXT NOT NULL)")
        target_times = defaultdict(list)
        batch, eligible_count = [], 0
        for row, split in _eligible_targets(db, policy):
            batch.append((row["trade_row_id"], split))
            target_times[row["fixture_id"]].append(row["query_us"])
            record = contract_records.get(row["condition_id"])
            if record is not None and verified_contract_context(record, row["query_us"]) is not None:
                contract_coverage[row["fixture_id"]]["eligible_targets_with_verified_contract_question"] += 1
            eligible_count += 1
            if len(batch) >= 10000:
                db.executemany("INSERT INTO eligible_targets VALUES(?,?)", batch)
                batch.clear()
            if eligible_count % 100000 == 0:
                announce(f"Checked {eligible_count:,} eligible source targets")
        db.executemany("INSERT INTO eligible_targets VALUES(?,?)", batch)
        records, links = _database_news(db)
        rows = coverage_for_records(target_times, records, fixture_links=links,
                                    fixture_ids=fixture_set,
                                    news_limit=news_limit if manifest is None else manifest["news_limit_per_scope"],
                                    deduplicate_headlines=deduplicate_headlines if manifest is None else manifest.get("news_headline_deduplication", False))
        for row in rows:
            covered = contract_coverage[row["fixture_id"]]["eligible_targets_with_verified_contract_question"]
            row["eligible_targets_with_verified_contract_question"] = covered
            row["eligible_targets_without_verified_contract_question"] = row["eligible_targets"] - covered
        details["historical_contract_questions"] = {
            "verified_contract_records": len(contract_records), "invalid_contract_records": contract_errors,
            "eligible_targets_with_verified_question": sum(row["eligible_targets_with_verified_contract_question"] for row in rows),
            "eligible_targets_without_verified_question": sum(row["eligible_targets_without_verified_contract_question"] for row in rows),
            "coverage_is_reported_separately_from_fixture_news_gate": True,
        }
        checks["all_fixtures_have_eligible_targets"] = all(row["eligible_targets"] > 0 for row in rows)
        checks["all_fixtures_have_prior_direct_news_for_eligible_target"] = all(
            row["eligible_targets_with_selected_prior_direct_fixture_news"] > 0 for row in rows)
        checks["release_present"] = release is not None
        if release is not None:
            announce("Checking released audit rows and selected fixture news")
            release_stats = _check_release(db, Path(release), manifest, records, links, rows, contract_records, announce)
            checks.update(release_stats.pop("checks"))
            details["release"] = release_stats
            checks["release_source_database_hash_matches"] = manifest.get("source_database_sha256") == _sha(database)
        _reject_wal(database)
        checks["source_database_unchanged"] = _identity(database) == before
        failed = sorted(name for name, value in checks.items() if not value)
        return {
            "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "all104_fixture_context_ready": not failed,
            "gate_scope": "retrospective under-20 API-observation SFT coverage across the configured fixture universe",
            "prospective_training_ready": False, "prospective_fully_ready": False,
            "canonical_chain_completeness_certified": False, "all_news_feed_completeness_certified": False,
            "actor_news_exposure_verified": False, "every_target_has_direct_fixture_news": all(
                row.get("exported_targets_without_selected_prior_direct_fixture_news", row["eligible_targets_without_selected_prior_direct_fixture_news"]) == 0
                for row in rows),
            "empty_news_before_first_verified_capture_is_valid": True,
            "checks": checks, "failed_checks": failed, "details": details,
            "expected": {"fixtures": expected_fixtures, "contracts": expected_contracts, "tokens": expected_tokens},
            "fixtures_with_eligible_prior_direct_news": sum(row["eligible_targets_with_selected_prior_direct_fixture_news"] > 0 for row in rows),
            "fixtures_with_exported_prior_direct_news": sum(row.get("exported_targets_with_selected_prior_direct_fixture_news", 0) > 0 for row in rows),
            "fixture_ids_missing_eligible_prior_direct_news": [row["fixture_id"] for row in rows if not row["eligible_targets_with_selected_prior_direct_fixture_news"]],
            "fixtures": rows,
            "requires_separate_full_sft_integrity_validation": True,
        }
    finally:
        db.close()


def _check_release(db, release, manifest, records, links, fixture_rows, contract_records, announce):
    _, direct, all_links = _verified_events(records, links)
    contexts, invalid_contexts = {}, 0
    for context in _jsonl(release / "contexts.jsonl.gz"):
        identity = {key: context[key] for key in ("scope", "eligible_news_ids", "prompt_news_limit")}
        if context.get("deduplicate_headlines") is True:
            identity["deduplicate_headlines"] = True
        valid = (context["context_id"] == "news:" + hashlib.sha256(_json(identity).encode()).hexdigest()
                 and context["selected_prompt_news_ids"] == list(dict.fromkeys(context["selected_prompt_news_ids"]))
                 and set(context["selected_prompt_news_ids"]).issubset(context["eligible_news_ids"])
                 and context["prompt_news_limit"] == manifest["news_limit_per_scope"]
                 and context.get("deduplicate_headlines", False) == manifest.get("news_headline_deduplication", False)
                 and len(context["selected_prompt_news_ids"]) <= manifest["news_limit_per_scope"])
        invalid_contexts += not valid or context["context_id"] in contexts
        contexts[context["context_id"]] = context
    db.execute("""CREATE TEMP TABLE release_audits(trade_row_id INTEGER PRIMARY KEY,fixture_id TEXT,
        condition_id TEXT,wallet TEXT,observation_id TEXT,query_us INTEGER,exported_split TEXT,
        token_id TEXT,transaction_hash TEXT,side TEXT,outcome TEXT,shares TEXT,price TEXT,
        source_page_id INTEGER,source_line INTEGER,source_query_us,source_block_timestamp TEXT)""")
    per_fixture = {row["fixture_id"]: row for row in fixture_rows}
    contract_times = {condition: _micros(record["initialized_at_utc"])
                      for condition, record in contract_records.items()}
    totals, split_counts, fixture_counts = Counter(), Counter(), defaultdict(Counter)
    batch = []
    for path in sorted((release / "audit").glob("part-*.jsonl.gz")):
        for audit in _jsonl(path):
            totals["audit_rows"] += 1
            query = _micros(audit["execution_time_proxy_utc"]) if audit["execution_time_proxy_utc"] is not None else None
            split, fixture = audit["exported_split"], audit["fixture_id"]
            action = audit["reported_action"]
            batch.append((audit["trade_row_id"], fixture, audit["condition_id"], audit["wallet"],
                          audit["observation_id"], query, split, audit["token_id"], audit["transaction_hash"],
                          action["side"], action["outcome"], action["shares"], action["price"],
                          audit["source_page_id"], audit["source_line"],
                          audit.get("source_query_us"), audit.get("source_block_timestamp")))
            if len(batch) >= 10000:
                db.executemany("INSERT INTO release_audits VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                batch.clear()
            if split is None:
                fixture_counts[fixture]["quarantined_targets"] += 1
                continue
            split_counts[split] += 1
            fixture_counts[fixture]["exported_targets"] += 1
            contract_time = contract_times.get(audit["condition_id"])
            if (manifest.get("contract_context_included") is True and contract_time is not None
                    and query is not None and contract_time < query):
                fixture_counts[fixture]["exported_targets_with_verified_contract_question"] += 1
            if audit["excluded_reasons"] or set(audit["profile_locations"]) != set(manifest["profiles"]):
                totals["invalid_export_descriptors"] += 1
            context = contexts.get(audit["fixture_context_id"])
            if context is None or context["scope"] != "fixture:" + fixture:
                totals["invalid_fixture_context_references"] += 1
                continue
            selected = context["selected_prompt_news_ids"]
            if query is None or any((fixture, news_id) not in all_links or all_links[(fixture, news_id)] >= query
                                    for news_id in selected):
                totals["invalid_selected_news_availability"] += 1
                continue
            has_direct = any((fixture, news_id) in direct and direct[(fixture, news_id)] < query for news_id in selected)
            fixture_counts[fixture]["exported_targets_with_selected_prior_direct_fixture_news"] += has_direct
        announce(f"Checked {totals['audit_rows']:,} release audit rows")
    db.executemany("INSERT INTO release_audits VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
    def source_timestamp_valid(query_us, block_timestamp):
        if type(query_us) is not int:
            return False
        try:
            return _micros(block_timestamp) == query_us
        except (TypeError, ValueError, OverflowError):
            return False
    db.create_function("source_timestamp_valid", 2, source_timestamp_valid)
    mismatched = db.execute("""SELECT COUNT(*) FROM release_audits a LEFT JOIN trades t USING(trade_row_id)
        WHERE t.trade_row_id IS NULL OR a.fixture_id IS NOT t.fixture_id OR a.condition_id IS NOT t.condition_id
        OR a.wallet IS NOT t.wallet OR a.observation_id IS NOT t.observation_id
        OR a.token_id IS NOT t.token_id OR a.transaction_hash IS NOT t.transaction_hash
        OR a.side IS NOT t.side OR a.outcome IS NOT t.token_outcome
        OR a.shares IS NOT t.shares OR a.price IS NOT t.price
        OR a.source_page_id IS NOT t.source_page_id OR a.source_line IS NOT t.source_line
        OR (a.query_us IS NOT NULL AND a.query_us IS NOT t.query_us)
        OR (a.query_us IS NULL AND (a.source_query_us IS NOT t.query_us
            OR a.source_block_timestamp IS NOT t.block_timestamp
            OR source_timestamp_valid(t.query_us,t.block_timestamp)))""").fetchone()[0]
    missing = db.execute("SELECT COUNT(*) FROM trades t LEFT JOIN release_audits a USING(trade_row_id) WHERE a.trade_row_id IS NULL").fetchone()[0]
    eligibility_mismatch = db.execute("""SELECT COUNT(*) FROM release_audits a
        LEFT JOIN eligible_targets e USING(trade_row_id) WHERE a.exported_split IS NOT e.split""").fetchone()[0]
    for fixture, row in per_fixture.items():
        counts = fixture_counts[fixture]
        row.update({key: counts[key] for key in ("exported_targets", "quarantined_targets", "exported_targets_with_selected_prior_direct_fixture_news")})
        row["exported_targets_without_selected_prior_direct_fixture_news"] = counts["exported_targets"] - counts["exported_targets_with_selected_prior_direct_fixture_news"]
        row["exported_direct_fixture_news_coverage_fraction"] = counts["exported_targets_with_selected_prior_direct_fixture_news"] / counts["exported_targets"] if counts["exported_targets"] else 0.0
        row["exported_targets_with_verified_contract_question"] = counts["exported_targets_with_verified_contract_question"]
        row["exported_targets_without_verified_contract_question"] = counts["exported_targets"] - counts["exported_targets_with_verified_contract_question"]
    checks = {
        "release_audit_exactly_covers_source": mismatched == missing == 0,
        "release_export_matches_eligible_targets": eligibility_mismatch == 0,
        "release_context_descriptors_valid": invalid_contexts == 0,
        "release_selected_news_strictly_prior_and_verified": not any(totals[key] for key in (
            "invalid_export_descriptors", "invalid_fixture_context_references", "invalid_selected_news_availability")),
        "release_manifest_target_totals_match": (dict(split_counts) == {k: v for k, v in manifest["split_target_counts"].items() if v}
            and sum(split_counts.values()) == manifest["included_target_observations"]
            and totals["audit_rows"] == manifest["source_observations"]),
        "all_fixtures_have_exported_prior_direct_news": all(row["exported_targets_with_selected_prior_direct_fixture_news"] > 0 for row in fixture_rows),
    }
    return {"checks": checks, **totals, "source_mismatched_audits": mismatched,
            "missing_source_audits": missing, "target_eligibility_mismatches": eligibility_mismatch,
            "invalid_contexts": invalid_contexts, "split_target_counts": dict(split_counts)}
