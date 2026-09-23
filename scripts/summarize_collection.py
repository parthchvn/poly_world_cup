#!/usr/bin/env python3
"""Compose a strict, compact collection overview from finalized reports.

No trade-table scan is needed. Completion means the specified public API
traversals and local integrity checks finished, never complete on-chain history.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.io import atomic_write, write_json


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value, label: str) -> int:
    _require(type(value) is int and value >= 0, label + " must be a nonnegative integer")
    return value


def _time(value: str) -> datetime:
    _require(isinstance(value, str), "Missing timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(result.tzinfo is not None, "Timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def _utc(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None


def _bounds(rows: list[dict]) -> tuple[str | None, str | None]:
    positive = [row for row in rows if row["row_count"] > 0]
    if not positive:
        return None, None
    starts, ends = [], []
    for row in positive:
        first, last = _time(row.get("earliest_block_timestamp")), _time(row.get("latest_block_timestamp"))
        _require(first <= last, "Condition observation bounds are reversed")
        starts.append(first)
        ends.append(last)
    return _utc(min(starts)), _utc(max(ends))


def _receipt_summary(report: dict, *, require_completed: bool) -> dict:
    if require_completed:
        _require(report.get("sample_collection_status") == "completed", "Receipt sample is still running")
    rows = report.get("results")
    _require(isinstance(rows, list), "Receipt report lacks observation results")
    counts = dict(sorted(Counter(row.get("status", "unknown") for row in rows).items()))
    if "status_counts" in report:
        _require(report["status_counts"] == counts, "Receipt status counts disagree with result rows")
    return {"scope": report.get("scope", "bounded_receipt_probe"), "observation_checks": len(rows),
            "distinct_observation_ids": len({row.get("observation_id") for row in rows}),
            "status_counts": counts, "fixture_count_requested": report.get("fixture_count"),
            "fixture_count_sampled": report.get("selected_fixture_count"),
            "missing_fixture_ids": report.get("missing_fixture_ids", []),
            "selection": report.get("selection", "earlier diagnostic probe"),
            "full_history_reconciled": False, "canonical_fill_identity_upgraded": False}


def compose_summary(*, registry: dict, batch: dict, provenance: dict, attribution: dict,
                    news_coverage: dict, archive_report: dict, current_news: list[dict],
                    archived_news: list[dict], receipt_sample: dict, earlier_probe: dict,
                    code_revision: str | None = None, input_files: dict | None = None) -> dict:
    """Validate all final counters before returning a publishable overview."""
    if code_revision is not None:
        _require(bool(re.fullmatch(r"[0-9a-f]{40}", code_revision)), "Code revision must be an actual full Git SHA")
    fixtures, contracts = registry["fixtures"], registry["contracts"]
    fixture_map = {row["fixture_id"]: row for row in fixtures}
    contract_map = {row["condition_id"]: row for row in contracts}
    _require(len(fixtures) == len(fixture_map) == 104, "World Cup registry must contain exactly 104 unique fixtures")
    _require(len(contracts) == len(contract_map) == 312, "World Cup registry must contain exactly 312 unique contracts")
    _require(all(row.get("mapping_status") == "matched" for row in fixtures), "Registry contains unresolved fixture mappings")
    _require(set(row["fixture_id"] for row in contracts) == set(fixture_map), "Registry contract-to-fixture mapping disagrees")
    contracts_per_fixture = Counter(row["fixture_id"] for row in contracts)
    _require(set(contracts_per_fixture.values()) == {3}, "Each fixture must have the three audited result contracts")
    tokens = [token["token_id"] for row in contracts for token in row["tokens"]]
    _require(len(tokens) == len(set(tokens)) == 624, "Registry must contain exactly 624 distinct outcome tokens")

    conditions = batch.get("conditions", [])
    condition_map = {row["condition_id"]: row for row in conditions}
    _require(len(conditions) == len(condition_map) == 312 and set(condition_map) == set(contract_map), "Batch conditions do not match all registry contracts")
    _require(batch.get("status") == "api_exhausted" and batch.get("finished_at"), "Tournament API collection has not finished")
    _require(batch.get("condition_count") == 312, "Batch condition total disagrees")
    _require(batch.get("status_counts", {}).get("exhausted") == 312 and
             all(value == 0 for key, value in batch.get("status_counts", {}).items() if key != "exhausted"),
             "Some batch conditions are not exhausted")
    for row in conditions:
        _require(row.get("status") == row.get("api_traversal_status") == "exhausted"
                 and row.get("integrity_validated") is True, "An exhausted condition lacks validated integrity")
        _integer(row.get("row_count"), "Condition row count")
        _integer(row.get("page_count"), "Condition page count")
    observation_count = sum(row["row_count"] for row in conditions)
    page_count = sum(row["page_count"] for row in conditions)
    _require(batch.get("committed_observation_count") == batch.get("validated_observation_count") == observation_count,
             "Batch observation totals disagree")
    _require(batch.get("committed_page_count") == page_count, "Batch page totals disagree")
    expected_condition_hash = hashlib.sha256("\n".join(sorted(contract_map)).encode()).hexdigest()
    _require(batch.get("condition_set_sha256") == expected_condition_hash, "Batch condition identity hash disagrees")
    try:
        minimum_size = Decimal(str(batch["requested_minimum_size_tokens"]))
    except (KeyError, InvalidOperation) as error:
        raise ValueError("Invalid requested minimum trade size") from error
    _require(minimum_size.is_finite() and minimum_size > 0, "Requested minimum trade size must be positive")

    _require(provenance.get("raw_provenance_verified") is True and provenance.get("errors") == [], "Raw provenance verification did not pass")
    _require(provenance.get("condition_selection") == "explicit", "Provenance verification must use explicit registry conditions")
    for field in ("requested_condition_count", "verified_condition_count", "normalized_integrity_verified_conditions", "api_exhausted_conditions"):
        _require(provenance.get(field) == 312, "Provenance condition total disagrees: " + field)
    _require(provenance.get("verified_observation_count") == observation_count, "Provenance observation total disagrees")
    _require(provenance.get("committed_page_count") == provenance.get("verified_raw_page_count") == page_count,
             "Provenance page totals disagree")
    _require(attribution.get("registry_condition_count") == attribution.get("api_traversals_exhausted") == 312
             and attribution.get("partial_collection_allowed") is False, "Attribution does not establish all registry traversals")
    _require(attribution.get("observation_count") == observation_count, "Attribution observation total disagrees")
    _require(attribution.get("source_page_count") == page_count, "Attribution source page total disagrees")
    _require(sum(attribution.get("token_mapping_counts", {}).values()) == observation_count, "Attribution token mapping counts disagree")
    _require(0 <= _integer(attribution.get("distinct_wallet_count"), "Distinct wallet count") <= observation_count,
             "Distinct wallet total exceeds observations")

    current_ids = {row["news_id"] for row in current_news}
    archived_ids = {row["news_id"] for row in archived_news}
    _require(len(current_ids) == len(current_news) and len(archived_ids) == len(archived_news)
             and not current_ids.intersection(archived_ids), "News catalogs contain duplicate version identities")
    _require(news_coverage.get("metadata_versions") == len(current_news), "Current news coverage count disagrees with catalog")
    _require(news_coverage.get("unique_articles") == len({row.get("news_item_id", row["news_id"]) for row in current_news}),
             "Current news unique article count disagrees")
    _require(news_coverage.get("in_window_metadata_versions") == sum(row.get("within_study_window") is True for row in current_news),
             "Current news in-window count disagrees")
    _require(archive_report.get("completed_articles") == archive_report.get("requested_articles")
             == sum(archive_report.get("statuses", {}).values()), "Historical archive lookup has not finished or counts disagree")
    _require(archive_report.get("verified_headline_versions") == len(archived_news), "Archive verified headline count disagrees with catalog")
    _require(attribution.get("news_count") == len(current_news) + len(archived_news), "Attribution news count disagrees with catalogs")
    _require(attribution.get("rejected_historical_availability_claims", 0) == 0, "Attribution rejected historical availability claims")
    coverage_rows = news_coverage.get("fixture_coverage", [])
    coverage_map = {row["fixture_id"]: row for row in coverage_rows}
    _require(len(coverage_rows) == 104 and set(coverage_map) == set(fixture_map), "News coverage must include each fixture exactly once")
    actual_candidates, archived_fixture_claims = Counter(), Counter()
    background_candidates = Counter()
    global_current_count = 0
    for row in current_news:
        _require(set(row.get("fixture_ids", [])) <= set(fixture_map), "Current news references an unknown fixture")
        global_current_count += row.get("context_scope") == "tournament"
        if row.get("within_study_window"):
            actual_candidates.update(row.get("fixture_ids", []))
            background_candidates.update(link["fixture_id"] for link in row.get("fixture_links", [])
                                         if link.get("relationship", "").endswith("_background"))
    for row in archived_news:
        _require(row.get("historical_availability_verified") is True, "Archived headline lacks a historical availability claim")
        _require(set(row.get("fixture_ids", [])) <= set(fixture_map), "Archived news references an unknown fixture")
        archived_fixture_claims.update(link["fixture_id"] for link in row.get("fixture_links", [])
                                      if link.get("historical_link_verified") is True)
    for fid, row in coverage_map.items():
        _require(row.get("candidate_news_versions") == actual_candidates[fid], "Per-fixture candidate news count disagrees: " + fid)
    _require(archive_report.get("fixture_links_verified") == sum(archived_fixture_claims.values()), "Archive fixture-link total disagrees")

    per_fixture_conditions = defaultdict(list)
    for condition, row in condition_map.items():
        per_fixture_conditions[contract_map[condition]["fixture_id"]].append(row)
    fixture_coverage = []
    for fixture in sorted(fixtures, key=lambda x: (_time(x["kickoff_utc"]), x["fixture_id"])):
        fid = fixture["fixture_id"]
        rows = per_fixture_conditions[fid]
        earliest, latest = _bounds(rows)
        fixture_coverage.append({"fixture_id": fid, "home_team": fixture["home_team"]["name"],
            "away_team": fixture["away_team"]["name"], "stage": fixture.get("stage"),
            "kickoff_utc": fixture["kickoff_utc"], "contracts": len(rows),
            "observation_count": sum(row["row_count"] for row in rows), "page_count": sum(row["page_count"] for row in rows),
            "earliest_block_timestamp": earliest, "latest_block_timestamp": latest,
            "api_traversals_exhausted": len(rows), "retrospective_news_candidates": actual_candidates[fid],
            "premarket_background_candidates": background_candidates[fid],
            "claimed_pregame_news_versions": coverage_map[fid].get("claimed_pregame_versions"),
            "archive_verified_fixture_link_claims": archived_fixture_claims[fid],
            "news_source_status": coverage_map[fid].get("source_status"),
            "fixture_metadata_historically_verified": False, "news_coverage_complete": False})
    nonempty_fixtures = sum(row["observation_count"] > 0 for row in fixture_coverage)
    _require(attribution.get("fixtures_with_observations") == nonempty_fixtures, "Attribution observed-fixture count disagrees")
    first_trade, last_trade = _bounds(conditions)
    news_start, news_end = news_coverage["window_start_utc"], news_coverage["window_end_utc"]
    _require(_time(news_start) < _time(news_end), "News study window bounds are invalid")
    news_window_covers_trades = bool(first_trade and _time(news_start) <= _time(first_trade)
                                   and _time(news_end) > _time(last_trade))
    receipts = {"tournament_sample": _receipt_summary(receipt_sample, require_completed=True),
                "earlier_diagnostic_probe": _receipt_summary(earlier_probe, require_completed=False),
                "full_history_reconciled": False,
                "interpretation": "Bounded diagnostic observation checks; ambiguous logs and economic mismatches are not resolved by API exhaustion."}
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_revision": code_revision, "completion_status": "requested_api_traversals_and_local_integrity_checks_completed",
        "scope": {"tournament": "2026 FIFA World Cup", "fixtures": 104, "contracts": 312, "outcome_tokens": 624,
                  "market_type": "mapped regulation-time match-result contracts", "providers": ["Polymarket Data API v2", "ESPN", "Internet Archive"]},
        "trades": {"observation_count": observation_count, "distinct_public_wallets": attribution["distinct_wallet_count"],
            "page_count": page_count, "conditions_exhausted": 312, "fixtures_with_observations": nonempty_fixtures,
            "earliest_block_timestamp": first_trade, "latest_block_timestamp": last_trade,
            "requested_minimum_size_tokens": str(minimum_size), "requested_taker_only": False,
            "count_semantics": "Public wallet-side API observations; not unique canonical on-chain fills or proven human decisions",
            "duplicate_observation_id_groups": attribution.get("duplicate_observation_id_groups"),
            "token_mapping_counts": attribution["token_mapping_counts"],
            "query_time_semantics": "execution_block_timestamp_proxy_not_decision_time",
            "raw_provenance_verified": True, "source_completeness_certified": False,
            "collection_started_at": batch.get("started_at"), "collection_finished_at": batch["finished_at"]},
        "news": {"current_metadata_versions": len(current_news), "current_unique_articles": news_coverage["unique_articles"],
            "current_versions_with_publication_claim_in_window": news_coverage["in_window_metadata_versions"],
            "current_tournament_scope_versions": global_current_count,
            "provider_archive_pages": news_coverage.get("archive_pages"), "provider_archive_status": news_coverage.get("archive_status"),
            "window_start_utc": news_start, "window_end_utc": news_end, "window_covers_observed_trade_interval": news_window_covers_trades,
            "premarket_background_lookback_days": news_coverage.get("premarket_background_lookback_days"),
            "candidate_fixture_links": sum(actual_candidates.values()),
            "fixtures_with_candidate_news": sum(value > 0 for value in actual_candidates.values()),
            "current_source_errors": news_coverage.get("fixture_source_errors"),
            "archive_requested_articles": archive_report["requested_articles"], "archive_completed_articles": archive_report["completed_articles"],
            "archive_lookup_status_counts": archive_report.get("statuses", {}), "archived_verified_headline_versions": len(archived_news),
            "archive_capture_lookup_start": archive_report.get("lookup_start"), "archive_capture_lookup_cutoff": archive_report.get("lookup_cutoff"),
            "archive_search_scope": archive_report.get("content_scope"),
            "archive_prefix_capture_collapse": archive_report.get("prefix_capture_collapse"),
            "archive_exact_lookup_candidate_limit": archive_report.get("exact_lookup_candidate_limit"),
            "historically_eligible_fixture_news_links": attribution.get("historically_eligible_fixture_news_links"),
            "historically_eligible_global_news_versions": attribution.get("historically_eligible_global_news_versions"),
            "observations_with_verified_prior_fixture_news": attribution.get("observations_with_verified_prior_news"),
            "observations_with_verified_prior_global_news": attribution.get("observations_with_verified_prior_global_news"),
            "raw_article_text_included": False, "complete_news_coverage": False,
            "limitations": news_coverage.get("limitations", [])},
        "reconciliation": receipts, "fixture_coverage": fixture_coverage,
        "integrity": {"batch_counts_match_provenance_and_attribution": True, "registry_condition_hash": expected_condition_hash,
            "raw_provenance_scope": provenance.get("integrity_scope"), "raw_provenance_checked_at": provenance.get("checked_at_utc"),
            "input_files": input_files or {}},
        "training_ready": False, "limitations": [
            "An exhausted filtered public API traversal does not prove complete exchange or on-chain trade history.",
            "API observations do not supply a unique canonical fill identity; do not add duplicated receipt emissions as extra trades.",
            "One publisher's retrieved archive is not all tournament news; team/headline matching is retrospective relevance, not actor exposure.",
            "Current headline publication claims do not prove historical content versions; recovered archive headlines use conservative capture bounds.",
            "Archive lookup limits and capture-collapse heuristics can miss older versions; broad tournament relevance is not direct fixture relevance.",
            "Current fixture metadata, especially knockout participants, cannot be assumed known before it became public.",
            "Wallet history covers tournament contracts only; holdings, private beliefs, other markets, and decision timing remain unobserved.",
            "Do not infer no-trade labels from missing data. SFT still requires label-coverage validation and baseline evaluation."]}


def render_readme(summary: dict) -> str:
    trades, news = summary["trades"], summary["news"]
    receipt = summary["reconciliation"]["tournament_sample"]
    probe = summary["reconciliation"]["earlier_diagnostic_probe"]
    counts = ", ".join(f"{name}: {count}" for name, count in receipt["status_counts"].items()) or "no results"
    probe_counts = ", ".join(f"{name}: {count}" for name, count in probe["status_counts"].items()) or "no results"
    lines = ["# 2026 World Cup observation dataset", "",
        f"The collection covers **104 fixtures, 312 regulation-result contracts, and 624 outcome tokens**. "
        f"It contains **{trades['observation_count']:,} public wallet-side API observations** from "
        f"**{trades['distinct_public_wallets']:,} distinct wallet addresses**, across {trades['page_count']:,} source pages.", "",
        f"Observed execution block timestamps run from `{trades['earliest_block_timestamp']}` to `{trades['latest_block_timestamp']}`. "
        f"The requested minimum trade-size filter was `{trades['requested_minimum_size_tokens']}` tokens, with both sides requested.", "",
        "All 312 requested API traversals exhausted, and collection, immutable raw-source replay, and SQLite attribution counts agree. "
        "These are API observations, not a certified set of unique exchange fills or human decisions. Upstream completeness remains unverified.", "",
        "## Open the data", "",
        "Extract `world_cup_corpus.tar.gz` and open `attribution.sqlite` with a SQLite browser or Python's built-in `sqlite3`. "
        "The database holds every normalized observation; `context_samples.jsonl` provides a small browsing sample. "
        "Use `registry.json` to look up fixture and contract IDs, and `MANIFEST.json` to verify member hashes.", "",
        "```sql", "SELECT fixture_id, COUNT(*) AS observations", "FROM trades GROUP BY fixture_id ORDER BY observations DESC;", "",
        "SELECT trade_row_id, wallet, fixture_id, side, shares, price, block_timestamp", "FROM trades ORDER BY trade_row_id LIMIT 20;", "```", "",
        "Shares, prices, and token IDs remain text to preserve exact decimals and identities. "
        "The optional separate `trade_provenance.tar.gz` contains source trade pages and captures. News article bodies and HTML are excluded.", "",
        "## News and context", "",
        f"The ESPN catalog contains {news['current_metadata_versions']:,} current metadata versions, including "
        f"{news['current_versions_with_publication_claim_in_window']:,} whose publication claims lie in "
        f"`[{news['window_start_utc']}, {news['window_end_utc']})`. Its team/background links provide candidates for "
        f"{news['fixtures_with_candidate_news']} fixtures. Current publication dates do not verify earlier headline versions.", "",
        f"Archive lookup completed for {news['archive_completed_articles']:,} requested articles and recovered "
        f"{news['archived_verified_headline_versions']:,} separate historical headline versions. The attribution index accepts "
        f"{news['historically_eligible_fixture_news_links']:,} historical fixture links and "
        f"{news['historically_eligible_global_news_versions']:,} broad World Cup headline versions. "
        "Archive capture time is a conservative availability bound. Broad news relevance does not establish direct fixture relevance or wallet exposure.", "",
        "## Receipt checks", "",
        f"The bounded tournament diagnostic checked {receipt['observation_checks']} observations ({counts}). "
        f"The earlier diagnostic checked {probe['observation_checks']} observations ({probe_counts}). "
        "These samples do not reconcile the full history or resolve every maker/taker and aggregate-log ambiguity.", "",
        "## Research limits", "",
        "**This is not yet an SFT-ready dataset.** Wallet histories cover these tournament markets only. "
        "Holdings, other-market trades, private beliefs, actor exposure, and actual order-decision timestamps remain unobserved. "
        "No-trade labels cannot be inferred from gaps. Final knockout team identities cannot be backfilled into earlier context. "
        "News coverage is limited to retrieved sources and bounded archive lookup heuristics.", "",
        "`tournament_collection.json` contains the full compact overview and one coverage row per fixture. "
        "See the repository's `docs/dataset_access.md` for SQL, context-inspection commands, and hash verification.", ""]
    if not news["window_covers_observed_trade_interval"]:
        lines.extend(["The configured news collection interval does not cover the full observed trade timestamp range. See the exact bounds in the JSON report.", ""])
    if summary.get("code_revision"):
        lines.extend([f"Code revision: `{summary['code_revision']}`.", ""])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {"registry": "data/registry/registry.json", "batch": "data/full/trades/batch_progress.json",
        "provenance": "data/full/provenance_report.json", "attribution": "data/full/attribution_report.json",
        "news_coverage": "data/news/coverage.json", "archive_report": "data/news_archive/report.json",
        "current_news": "data/news/news.jsonl", "archived_news": "data/news_archive/news_archive.jsonl",
        "receipt_sample": "data/full/reconciliation/sample_report.json", "earlier_probe": "data/full/reconciliation_probe/report.json"}
    for name, default in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(default))
    parser.add_argument("--code-revision")
    parser.add_argument("--output", type=Path, default=Path("reports/tournament_collection.json"))
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inputs, sources = {}, {}
        for name in defaults:
            path = getattr(args, name)
            data = path.read_bytes()
            sources[name] = {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            inputs[name] = ([json.loads(line) for line in data.splitlines() if line.strip()]
                            if name in {"current_news", "archived_news"} else json.loads(data))
        result = compose_summary(**inputs, code_revision=args.code_revision, input_files=sources)
        # No output is touched until every consistency check passes.
        write_json(args.output, result)
        atomic_write(args.readme, render_readme(result))
        print(json.dumps({"output": str(args.output), "readme": str(args.readme),
                          "observations": result["trades"]["observation_count"], "training_ready": False}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
