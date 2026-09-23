"""Compact, conservative links between observed executions and public context.

This is an audit index, not an SFT exporter. It does not establish what a wallet
saw, when an order was decided, or why an execution occurred. All news and
registry links are retrospective unless their historical availability is proved.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from io import BufferedReader
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Iterable, Mapping

from .temporal import parse_utc

SCHEMA_VERSION = 2
_GLOBAL_SCOPE = "@global:world-cup"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _micros(value: str) -> int:
    delta = parse_utc(value) - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _evidence_valid(evidence: Any, *, upper: int, content_hash: str | None = None) -> bool:
    """Validate evidence descriptors, not the truth of the upstream assertion."""
    if not isinstance(evidence, list):
        return False
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if item.get("kind") not in {"archive_snapshot", "contemporaneous_capture"}:
            continue
        digest = item.get("content_sha256")
        if (not isinstance(digest, str) or not _SHA256.fullmatch(digest)
                or not isinstance(item.get("source_url"), str)
                or not item["source_url"].startswith(("https://", "http://"))):
            continue
        if content_hash is not None and content_hash != digest:
            continue
        try:
            if _micros(item["captured_at_utc"]) <= upper:
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def _verified_news_time(record: Mapping[str, Any]) -> int | None:
    if record.get("historical_availability_verified") is not True:
        return None
    if (record.get("version_rank", 0) != 0
            and record.get("version_order_historically_verified") is not True):
        return None
    upper = record.get("availability_upper_utc")
    digest = record.get("historical_content_sha256")
    if not isinstance(upper, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        return None
    value = _micros(upper)
    if not _evidence_valid(record.get("availability_evidence"), upper=value, content_hash=digest):
        return None
    return value


def _link_time(link: Mapping[str, Any], news_time: int | None) -> int | None:
    if news_time is None or link.get("historical_link_verified") is not True:
        return None
    upper = link.get("link_availability_upper_utc")
    if not isinstance(upper, str):
        return None
    value = _micros(upper)
    if not _evidence_valid(link.get("link_availability_evidence"), upper=value):
        return None
    return max(value, news_time)


def _conflicting_tournament_scope(title: str) -> bool:
    """Conservatively reject explicit other editions, age groups, or sports."""
    if any(year != "2026" for year in re.findall(r"\b(?:19|20)\d{2}\b", title)):
        return True
    return bool(re.search(
        r"\b(?:club|women(?:['’]s)?|womens|girls|youth|under[ -]?\d{1,2}|u[ -]?\d{1,2}"
        r"|rugby|cricket|hockey|basketball|futsal|beach|ski(?:ing)?|esports?)\b",
        title, re.I,
    ))


def is_2026_world_cup_headline(title: str) -> bool:
    """Conservative global-scope heuristic for an already verified headline.

    A literal World Cup reference with no explicit conflicting edition is a
    broad relevance cue, not proof that all of its content concerns this event.
    Historical comparisons containing other years fail closed deliberately.
    This helper never verifies historical availability or the article version.
    """
    return (isinstance(title, str) and not _conflicting_tournament_scope(title)
            and bool(re.search(r"\bworld\s+cup\b", title, re.I)))


def _global_time(record: Mapping[str, Any], news_time: int | None) -> int | None:
    """Require historical evidence for broad World Cup relevance, not final teams."""
    if news_time is None or record.get("context_scope") != "tournament":
        return None
    # This must be the archived headline tied to historical_content_sha256,
    # never today's headline grafted onto an old archive timestamp.
    title = str(record.get("title", ""))
    if _conflicting_tournament_scope(title):
        return None
    if is_2026_world_cup_headline(title):
        return news_time
    if record.get("historical_tournament_scope_verified") is not True:
        return None
    upper = record.get("tournament_scope_availability_upper_utc")
    if not isinstance(upper, str):
        return None
    instant = _micros(upper)
    if not _evidence_valid(record.get("tournament_scope_availability_evidence"), upper=instant):
        return None
    return max(instant, news_time)


def _contract_index(registry: Mapping[str, Any]) -> tuple[set[str], dict[str, dict]]:
    fixtures = {row["fixture_id"] for row in registry["fixtures"]}
    if len(fixtures) != len(registry["fixtures"]):
        raise ValueError("Duplicate fixture identity in registry")
    result: dict[str, dict] = {}
    for contract in registry["contracts"]:
        condition = contract["condition_id"].lower()
        if condition in result:
            raise ValueError("Ambiguous condition identity in registry")
        if contract["fixture_id"] not in fixtures:
            raise ValueError("Contract references an unknown fixture")
        tokens = {str(token["token_id"]): token["outcome"] for token in contract["tokens"]}
        if len(tokens) != len(contract["tokens"]):
            raise ValueError("Duplicate token identity in contract")
        result[condition] = {"fixture_id": contract["fixture_id"],
                             "selection": contract["selection"], "tokens": tokens}
    return fixtures, result


def _create_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        CREATE TABLE news(
            news_id TEXT PRIMARY KEY, item_id TEXT NOT NULL, version_rank INTEGER NOT NULL,
            published_us INTEGER, verified_availability_us INTEGER, record_json TEXT NOT NULL,
            UNIQUE(item_id, version_rank)
        );
        CREATE TABLE fixture_news(
            fixture_id TEXT NOT NULL, news_id TEXT NOT NULL REFERENCES news(news_id),
            eligible_us INTEGER, link_json TEXT NOT NULL, PRIMARY KEY(fixture_id, news_id)
        );
        CREATE TABLE global_news(
            news_id TEXT PRIMARY KEY REFERENCES news(news_id), eligible_us INTEGER NOT NULL
        );
        CREATE TABLE context_states(
            context_state_id TEXT PRIMARY KEY, fixture_id TEXT NOT NULL,
            event_count INTEGER NOT NULL, newest_event_us INTEGER, prefix_news_id TEXT
        );
        CREATE TABLE source_pages(
            source_page_id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE,
            uncompressed_sha256 TEXT NOT NULL, row_count INTEGER NOT NULL
        );
        CREATE TABLE trades(
            trade_row_id INTEGER PRIMARY KEY, observation_id TEXT NOT NULL,
            source_page_id INTEGER NOT NULL REFERENCES source_pages(source_page_id),
            source_line INTEGER NOT NULL, wallet TEXT NOT NULL, condition_id TEXT NOT NULL,
            token_id TEXT NOT NULL, side TEXT NOT NULL, shares TEXT NOT NULL, price TEXT NOT NULL,
            query_us INTEGER NOT NULL, block_timestamp TEXT NOT NULL, transaction_hash TEXT NOT NULL,
            fixture_id TEXT, selection TEXT, token_outcome TEXT, token_mapping_status TEXT NOT NULL,
            context_state_id TEXT, eligible_context_event_count INTEGER NOT NULL,
            retrospective_candidate_count INTEGER NOT NULL,
            global_context_state_id TEXT NOT NULL, global_eligible_context_event_count INTEGER NOT NULL,
            UNIQUE(source_page_id, source_line)
        );
    """)


def _normalized_page_expectations(values: Mapping[Path | str, Any] | None, *,
                                  kind: str) -> dict[Path, Any] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise ValueError(f"Expected page {kind} must be a path-keyed mapping")
    result = {}
    for supplied_path, value in values.items():
        path = Path(supplied_path).resolve()
        if path in result:
            raise ValueError(f"Duplicate resolved path in expected page {kind}: {path}")
        if kind == "hashes":
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"Expected page hash must be lowercase SHA-256: {path}")
        elif type(value) is not int or value < 0:
            raise ValueError(f"Expected page row count must be a nonnegative integer: {path}")
        result[path] = value
    return result


def build_attribution_index(*, registry: Mapping[str, Any], trade_pages: Iterable[Path],
                            news_records: Iterable[Mapping[str, Any]], output_path: Path,
                            expected_page_hashes: Mapping[Path | str, str] | None = None,
                            expected_page_row_counts: Mapping[Path | str, int] | None = None) -> dict:
    """Stream normalized JSONL[.gz] pages into an atomically replaced SQLite index.

    The caller must supply the exact committed pages from audited collection
    manifests. This function records their uncompressed checksums but cannot
    certify completeness or compare them to an omitted manifest. It retains
    every input observation, including equal economic fields or observation IDs.
    Optional expected_page_hashes / expected_page_row_counts bind this build to
    the exact pages of an independently verified manifest snapshot. Each supplied
    mapping must name exactly the input page set, using resolved paths. Hashes
    cover uncompressed JSONL bytes. The already-streamed digest and row count are
    checked before publishing; any mismatch preserves the previous database.
    These expectations do not themselves establish source truth or completeness.
    News volume is held in memory; trade volume is streamed to disk.
    """
    fixtures, contracts = _contract_index(registry)
    expected_hashes = _normalized_page_expectations(expected_page_hashes, kind="hashes")
    expected_counts = _normalized_page_expectations(expected_page_row_counts, kind="row counts")
    if (expected_hashes is not None and expected_counts is not None
            and expected_hashes.keys() != expected_counts.keys()):
        raise ValueError("Expected page hashes and row counts name different path sets")
    observed_paths: set[Path] = set()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp",
                                     dir=output_path.parent)
    os.close(fd)
    db = sqlite3.connect(temporary)
    try:
        db.execute("PRAGMA foreign_keys=ON")
        # Keep multi-million-row index builds independent of available RAM.
        # SQLite may use temporary disk files for index/grouping operations.
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-32768")
        _create_schema(db)
        eligible_events: dict[str, list[tuple[int, str, dict, dict]]] = {f: [] for f in fixtures}
        eligible_events[_GLOBAL_SCOPE] = []
        candidates: dict[str, list[int]] = {f: [] for f in fixtures | {_GLOBAL_SCOPE}}
        blocked_claims = 0
        news_count = 0
        link_count = 0
        for supplied in news_records:
            record = dict(supplied)
            news_id = record.get("news_id")
            if not isinstance(news_id, str) or not news_id:
                raise ValueError("News requires a stable nonempty news_id")
            item_id = record.get("news_item_id", news_id)
            rank = record.get("version_rank", 0)
            if not isinstance(item_id, str) or not item_id or type(rank) is not int or rank < 0:
                raise ValueError("Invalid news item/version identity")
            published = record.get("published_at_utc")
            published_us = _micros(published) if published is not None else None
            verified_time = _verified_news_time(record)
            blocked_claims += int(record.get("historical_availability_verified") is True
                                  and verified_time is None)
            db.execute("INSERT INTO news VALUES(?,?,?,?,?,?)",
                       (news_id, item_id, rank, published_us, verified_time, _json(record)))
            news_count += 1
            global_time = _global_time(record, verified_time)
            if global_time is not None:
                db.execute("INSERT INTO global_news VALUES(?,?)", (news_id, global_time))
                eligible_events[_GLOBAL_SCOPE].append((global_time, news_id, record,
                    {"scope": "world_cup_global_public_context", "direct_fixture_relevance_verified": False}))
            links = record.get("fixture_links", [])
            if not isinstance(links, list) or any(not isinstance(link, dict) for link in links):
                raise ValueError("fixture_links must be an array of objects")
            links_by_fixture = {}
            for link in links:
                fixture = link.get("fixture_id")
                if fixture in links_by_fixture:
                    raise ValueError("Duplicate news-to-fixture link")
                links_by_fixture[fixture] = link
            linked = record.get("fixture_ids", list(links_by_fixture))
            if not isinstance(linked, list) or len(set(linked)) != len(linked):
                raise ValueError("fixture_ids must contain distinct fixture identities")
            if not set(links_by_fixture).issubset(linked):
                raise ValueError("fixture_links and fixture_ids disagree")
            for fixture in linked:
                if fixture not in fixtures:
                    raise ValueError("News references an unknown fixture")
                link = links_by_fixture.get(fixture, {"fixture_id": fixture,
                    "relationship": "retrospective_unspecified", "historical_link_verified": False})
                eligible_time = _link_time(link, verified_time)
                db.execute("INSERT INTO fixture_news VALUES(?,?,?,?)",
                           (fixture, news_id, eligible_time, _json(link)))
                link_count += 1
                if eligible_time is not None:
                    eligible_events[fixture].append((eligible_time, news_id, record, link))
                elif published_us is not None:
                    candidates[fixture].append(published_us)

        states: dict[str, list[str]] = {}
        event_times: dict[str, list[int]] = {}
        for fixture in sorted(eligible_events):
            digest = hashlib.sha256(_json(["context-v1", fixture]).encode()).hexdigest()
            states[fixture] = [digest]
            event_times[fixture] = []
            db.execute("INSERT INTO context_states VALUES(?,?,?,?,?)", (digest, fixture, 0, None, None))
            events = sorted(eligible_events[fixture], key=lambda item: (item[0], item[1]))
            for count, (instant, news_id, record, link) in enumerate(events, 1):
                digest = hashlib.sha256(_json([digest, news_id, record.get("historical_content_sha256"),
                                              record.get("version_rank", 0), instant, link]).encode()).hexdigest()
                states[fixture].append(digest)
                event_times[fixture].append(instant)
                db.execute("INSERT INTO context_states VALUES(?,?,?,?,?)",
                           (digest, fixture, count, instant, news_id))
            candidates[fixture].sort()

        # Executions share block times; bound this cache independently of the
        # corpus size and retain exactly the same strict timestamp parser.
        trade_timestamp = lru_cache(maxsize=8192)(_micros)
        count = 0
        mapping_counts: Counter[str] = Counter()
        for supplied_path in trade_pages:
            path = Path(supplied_path).resolve()
            for kind, expected in (("hash", expected_hashes), ("row count", expected_counts)):
                if expected is not None and path not in expected:
                    raise ValueError(f"Missing expected page {kind}: {path}")
            observed_paths.add(path)
            digest = hashlib.sha256()
            cursor = db.execute("INSERT INTO source_pages(path,uncompressed_sha256,row_count) VALUES(?,?,0)",
                                (str(path), "pending"))
            page_id = cursor.lastrowid
            opener = gzip.open if path.name.endswith(".gz") else open
            page_count = 0
            with BufferedReader(opener(path, "rb"), buffer_size=256 * 1024) as stream:
                for line_number, line in enumerate(stream, 1):
                    digest.update(line)
                    # The normalized page contract is UTF-8 JSONL. Explicit
                    # decoding avoids repeated generic JSON encoding detection.
                    row = json.loads(line.decode("utf-8"))
                    condition = row["condition_id"].lower()
                    contract = contracts.get(condition)
                    fixture = contract["fixture_id"] if contract else None
                    token = row["token_id"]
                    if not isinstance(token, str):
                        raise ValueError("Token IDs must remain strings")
                    outcome = contract["tokens"].get(token) if contract else None
                    status = ("exact_current_registry_token" if outcome is not None else
                              "unresolved_token" if contract else "unmapped_condition")
                    query_us = trade_timestamp(row["block_timestamp"])
                    prefix = bisect_left(event_times[fixture], query_us) if fixture else 0
                    retrospective_count = bisect_left(candidates[fixture], query_us) if fixture else 0
                    global_prefix = bisect_left(event_times[_GLOBAL_SCOPE], query_us) if fixture else 0
                    if row["side"] not in {"BUY", "SELL"}:
                        raise ValueError("Observed side must remain BUY or SELL")
                    if not isinstance(row["size"], str) or not isinstance(row["price"], str):
                        raise ValueError("Use normalized exact decimal strings for shares and price")
                    db.execute("""INSERT INTO trades(
                        observation_id,source_page_id,source_line,wallet,condition_id,token_id,
                        side,shares,price,query_us,block_timestamp,transaction_hash,fixture_id,
                        selection,token_outcome,token_mapping_status,context_state_id,
                        eligible_context_event_count,retrospective_candidate_count,
                        global_context_state_id,global_eligible_context_event_count
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (row["observation_id"], page_id, line_number, row["proxy_wallet"], condition,
                         token, row["side"], row["size"], row["price"], query_us,
                         row["block_timestamp"], row["transaction_hash"], fixture,
                         contract["selection"] if contract else None, outcome, status,
                         states[fixture][prefix] if fixture else None, prefix, retrospective_count,
                         states[_GLOBAL_SCOPE][global_prefix], global_prefix))
                    page_count += 1
                    count += 1
                    mapping_counts[status] += 1
            checksum = digest.hexdigest()
            if expected_hashes is not None and checksum != expected_hashes[path]:
                raise ValueError(f"Page checksum differs from verified expectation: {path}")
            if expected_counts is not None and page_count != expected_counts[path]:
                raise ValueError(f"Page row count differs from verified expectation: {path}")
            db.execute("UPDATE source_pages SET uncompressed_sha256=?,row_count=? WHERE source_page_id=?",
                       (checksum, page_count, page_id))
        for kind, expected in (("hashes", expected_hashes), ("row counts", expected_counts)):
            if expected is not None and observed_paths != expected.keys():
                raise ValueError(f"Expected page {kind} contain paths absent from input")
        db.executescript("""
            CREATE INDEX trades_wallet_time ON trades(wallet,query_us);
            CREATE INDEX trades_fixture_time ON trades(fixture_id,query_us);
            CREATE INDEX trades_observation ON trades(observation_id);
            CREATE INDEX fixture_news_time ON fixture_news(fixture_id,eligible_us);
            CREATE INDEX global_news_time ON global_news(eligible_us);
        """)
        report = {
            "schema_version": SCHEMA_VERSION,
            "expected_page_hashes_verified": expected_hashes is not None,
            "expected_page_row_counts_verified": expected_counts is not None,
            "observation_count": count,
            "source_page_count": db.execute("SELECT COUNT(*) FROM source_pages").fetchone()[0],
            "distinct_wallet_count": db.execute("SELECT COUNT(DISTINCT wallet) FROM trades").fetchone()[0],
            "fixtures_with_observations": db.execute("SELECT COUNT(DISTINCT fixture_id) FROM trades").fetchone()[0],
            "news_count": news_count, "fixture_news_link_count": link_count,
            "historically_eligible_fixture_news_links": sum(len(eligible_events[f]) for f in fixtures),
            "historically_eligible_global_news_versions": len(eligible_events[_GLOBAL_SCOPE]),
            "rejected_historical_availability_claims": blocked_claims,
            "observations_with_verified_prior_news": db.execute(
                "SELECT COUNT(*) FROM trades WHERE eligible_context_event_count>0").fetchone()[0],
            "observations_with_verified_prior_global_news": db.execute(
                "SELECT COUNT(*) FROM trades WHERE global_eligible_context_event_count>0").fetchone()[0],
            "token_mapping_counts": dict(sorted(mapping_counts.items())),
            "duplicate_observation_id_groups": db.execute(
                "SELECT COUNT(*) FROM (SELECT observation_id FROM trades GROUP BY observation_id HAVING COUNT(*)>1)"
            ).fetchone()[0],
            "wallet_history_scope": "tournament_only",
            "holdings_reconstructed": False,
            "trade_amounts_semantics": "provider_reported_not_chain_reconciled",
            "chain_reconciliation_verified": False,
            "query_time_semantics": "execution_block_timestamp_proxy_not_decision_time",
            "source_completeness_certified": False,
            "training_ready": False,
            "interpretation": "Public-context relevance only; actor exposure and causation unknown",
        }
        db.execute("INSERT INTO metadata VALUES(?,?)", ("report", _json(report)))
        db.execute("INSERT INTO metadata VALUES(?,?)", ("registry_feature_eligible", "false"))
        db.commit()
        db.close()
        os.replace(temporary, output_path)
        return report
    finally:
        db.close()
        Path(temporary).unlink(missing_ok=True)


def read_trade_context(database_path: Path, trade_row_id: int, *,
                       retrospective_limit: int = 20, wallet_history_limit: int = 20,
                       global_context_limit: int = 20) -> dict:
    """Inspect one observation, verified context, and explicitly retrospective history.

    Earlier wallet executions exclude the entire target transaction and equal
    timestamps. They remain ineligible as model input because API block time
    does not establish historical public availability or order decision time.
    """
    if type(trade_row_id) is not int or trade_row_id < 1:
        raise ValueError("trade_row_id must be a positive integer")
    if (type(retrospective_limit) is not int or retrospective_limit < 0
            or type(wallet_history_limit) is not int or wallet_history_limit < 0
            or type(global_context_limit) is not int or global_context_limit < 0):
        raise ValueError("Display limits must be nonnegative integers")
    db = sqlite3.connect(Path(database_path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        row = db.execute("SELECT * FROM trades WHERE trade_row_id=?", (trade_row_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown trade row {trade_row_id}")
        trade = dict(row)
        eligible = db.execute("""
            SELECT record_json FROM (
                SELECT n.record_json,n.item_id,
                    ROW_NUMBER() OVER (PARTITION BY n.item_id ORDER BY n.version_rank DESC) AS rank
                FROM news n JOIN fixture_news f USING(news_id)
                WHERE f.fixture_id=? AND f.eligible_us<?
            ) WHERE rank=1 ORDER BY item_id
        """, (trade["fixture_id"], trade["query_us"])).fetchall()
        global_query_us = trade["query_us"] if trade["fixture_id"] is not None else -2**63
        global_context_count = db.execute("""
            SELECT COUNT(DISTINCT n.item_id) FROM news n JOIN global_news g USING(news_id)
            WHERE g.eligible_us<?
        """, (global_query_us,)).fetchone()[0]
        global_context = db.execute("""
            SELECT record_json FROM (
                SELECT n.record_json,n.item_id,g.eligible_us,
                    ROW_NUMBER() OVER (PARTITION BY n.item_id ORDER BY n.version_rank DESC) AS rank
                FROM news n JOIN global_news g USING(news_id) WHERE g.eligible_us<?
            ) WHERE rank=1 ORDER BY eligible_us DESC,item_id LIMIT ?
        """, (global_query_us, global_context_limit)).fetchall()
        retrospective = db.execute("""
            SELECT n.record_json FROM news n JOIN fixture_news f USING(news_id)
            WHERE f.fixture_id=? AND f.eligible_us IS NULL AND n.published_us<?
            ORDER BY n.published_us DESC,n.news_id LIMIT ?
        """, (trade["fixture_id"], trade["query_us"], retrospective_limit)).fetchall()
        past = db.execute("""SELECT trade_row_id,observation_id,fixture_id,condition_id,token_id,
                                   side,shares,price,block_timestamp,transaction_hash
            FROM trades WHERE wallet=? AND query_us<? AND transaction_hash<>?
            ORDER BY query_us DESC,observation_id,trade_row_id LIMIT ?""",
            (trade["wallet"], trade["query_us"], trade["transaction_hash"], wallet_history_limit)).fetchall()
        return {
            "trade": trade,
            "verified_public_context_at_execution_proxy": [json.loads(row[0]) for row in eligible],
            "verified_global_public_context": [json.loads(row[0]) for row in global_context],
            "verified_global_public_context_item_count": global_context_count,
            "global_public_context_display_truncated": len(global_context) < global_context_count,
            "global_context_direct_fixture_relevance_verified": False,
            "retrospective_news_candidates": [json.loads(row[0]) for row in retrospective],
            "retrospective_prior_tournament_executions": [dict(row) for row in past],
            "wallet_history_feature_eligible": False,
            "registry_metadata_feature_eligible": False,
            "wallet_history_scope": "tournament_only",
            "holdings": None,
            "trade_amounts_semantics": "provider_reported_not_chain_reconciled",
            "chain_reconciliation_verified": False,
            "actor_exposure_verified": False,
            "decision_time_verified": False,
            "causal_attribution": False,
            "training_ready": False,
        }
    finally:
        db.close()
