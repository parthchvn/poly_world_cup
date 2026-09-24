"""Market lookup and streaming captured trade inputs for actor interval exports.

The live path uses the repository's resumable cursor-based v2 collector. It
never substitutes the old offset-limited endpoint. API exhaustion describes
the API traversal, not complete on-chain history. No actor filtering happens
here: the caller counts each actor's observations before applying its cutoff.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator
from urllib.parse import quote

from .http import HttpClient
from .trades import ingest_condition

GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
_CONDITION = re.compile(r"^0x[0-9a-fA-F]{64}$")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "datasets/world_cup_2026_tournament_lt20_v2_evidence/registry.json"


def utc_time(time_us: int) -> str:
    return (_EPOCH + timedelta(microseconds=time_us)).isoformat().replace("+00:00", "Z")


def timestamp_us(value: Any) -> int:
    """Parse an explicitly zoned ISO timestamp or epoch seconds, without floats."""
    if isinstance(value, bool) or value is None:
        raise ValueError("A trade timestamp is required")
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        number = Decimal(text)
        # Existing captures sometimes export epoch milliseconds/microseconds.
        scale = 1 if number >= Decimal("100000000000000") else (1000 if number >= Decimal("100000000000") else 1000000)
        return int(number * scale)
    instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError(f"Timestamp needs a timezone: {text}")
    delta = instant.astimezone(timezone.utc) - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def _array(value: Any) -> list:
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, list) else []


def _matches(row: dict, supplied: str) -> bool:
    return supplied.casefold() in {
        str(row.get(key, "")).casefold()
        for key in ("id", "market_id", "conditionId", "condition_id", "slug", "market_slug")
    }


def _metadata(raw: dict, fixture: dict | None = None) -> dict:
    fixture = fixture or {}
    condition = str(raw.get("condition_id") or raw.get("conditionId") or "").lower()
    if not _CONDITION.fullmatch(condition):
        raise ValueError("Market metadata does not identify one binary condition")
    tokens = raw.get("tokens")
    if not isinstance(tokens, list):
        outcomes, ids = _array(raw.get("outcomes")), _array(raw.get("clobTokenIds"))
        if len(outcomes) != len(ids):
            raise ValueError("Market outcome and token mappings have different lengths")
        tokens = [{"token_id": str(token), "outcome": str(outcome), "outcome_index": index}
                  for index, (outcome, token) in enumerate(zip(outcomes, ids))]
    if len(tokens) != 2:
        raise ValueError("This exporter currently requires one binary market, not an event containing several markets")
    events = raw.get("events") or []
    event = events[0] if len(events) == 1 and isinstance(events[0], dict) else {}
    teams = [fixture.get(side, {}).get("name") for side in ("home_team", "away_team")]
    teams = [team for team in teams if team]
    title = raw.get("fixture_title") or (" vs. ".join(teams) if teams else event.get("title"))
    if not teams and title:
        teams = [text.strip() for text in re.split(r"\s+(?:vs\.?|v\.)\s+", title) if text.strip()]
        if len(teams) != 2:
            teams = []
    opened = raw.get("accepting_orders_at") or raw.get("acceptingOrdersTimestamp")
    open_basis = "accepting_orders_timestamp" if opened else None
    kickoff = fixture.get("kickoff_utc") or raw.get("game_start_time") or raw.get("gameStartTime")
    kickoff_utc = utc_time(timestamp_us(kickoff)) if kickoff else None
    # Creation/start dates are metadata, never silently substituted for opening.
    return {
        "market_id": str(raw.get("market_id") or raw.get("id") or ""),
        "condition_id": condition,
        "market_slug": raw.get("market_slug") or raw.get("slug"),
        "question": raw.get("question"),
        "tokens": tokens,
        "token_outcomes": {str(token["token_id"]): str(token["outcome"]) for token in tokens},
        "fixture_id": fixture.get("fixture_id") or raw.get("fixture_id"),
        "fixture_title": title,
        "team_names": teams,
        "espn_event_id": fixture.get("espn_event_id") or raw.get("espn_event_id"),
        "kickoff_utc": kickoff_utc,
        "fixture_date": kickoff_utc[:10] if kickoff_utc else None,
        "market_open_utc": utc_time(timestamp_us(opened)) if opened else None,
        "market_open_basis": open_basis,
        "created_at_utc": raw.get("created_at") or raw.get("createdAt"),
        "rules_text": raw.get("rules_text") or raw.get("description"),
        "metadata_scope": "retrospective_market_metadata_not_time_verified_prompt_features",
        "source": raw.get("source"),
    }


def resolve_market(market_id: str, *, client: Any = None,
                   registry: Path | dict | None = None,
                   metadata_file: Path | None = None) -> dict:
    """Resolve a numeric market ID, condition ID or market slug.

    Prefer the bundled registry when available, including its ESPN event map.
    ``metadata_file`` accepts a Gamma market object or a registry-style contract
    with an optional ``fixture`` object, allowing fully offline operation.
    An event slug is deliberately not expanded into several binary markets.
    """
    supplied = str(market_id).strip()
    if not supplied:
        raise ValueError("market_id must not be empty")
    if metadata_file is not None:
        raw = json.loads(Path(metadata_file).read_text())
        if not isinstance(raw, dict) or not _matches(raw, supplied):
            raise ValueError("Metadata file does not match the requested market")
        return _metadata(raw, raw.get("fixture"))
    if registry is None and _DEFAULT_REGISTRY.exists():
        registry = _DEFAULT_REGISTRY
    if registry is not None:
        catalog = registry if isinstance(registry, dict) else json.loads(Path(registry).read_text())
        matches = [row for row in catalog.get("contracts", []) if _matches(row, supplied)]
        if len(matches) > 1:
            raise ValueError("The supplied market identifier is ambiguous in the registry")
        if matches:
            raw = matches[0]
            fixture = next((row for row in catalog.get("fixtures", [])
                            if row.get("fixture_id") == raw.get("fixture_id")), {})
            return _metadata(raw, fixture)
    if client is None:
        raise ValueError("Market not in the local registry; supply a client or --market-metadata for offline use")
    if supplied.isdigit():
        response = client.get_json(f"{GAMMA_MARKETS}/{quote(supplied, safe='')}")
        candidates = [response.data]
    else:
        params = {"condition_ids": supplied} if _CONDITION.fullmatch(supplied) else {"slug": supplied}
        response = client.get_json(GAMMA_MARKETS, params=params)
        candidates = response.data
    if not isinstance(candidates, list):
        raise ValueError("Unexpected Gamma market response")
    matches = [row for row in candidates if isinstance(row, dict) and _matches(row, supplied)]
    if len(matches) != 1:
        raise ValueError("Expected one binary market; use its numeric market_id or condition_id, not an event ID")
    raw = dict(matches[0])
    raw["source"] = {"url": response.url, "retrieved_at": response.retrieved_at,
                     "body_sha256": response.body_sha256}
    return _metadata(raw)


def _rows(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    suffix = path.with_suffix("").suffix if path.suffix == ".gz" else path.suffix
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        if suffix.lower() == ".csv":
            yield from csv.DictReader(stream)
        elif suffix.lower() == ".json":
            data = json.load(stream, parse_float=Decimal)
            rows = data.get("data", []) if isinstance(data, dict) else data
            if not isinstance(rows, list):
                raise ValueError(f"Expected a trade array: {path}")
            yield from rows
        else:
            for line in stream:
                if line.strip():
                    yield json.loads(line, parse_float=Decimal)


def _first(row: dict, *names: str) -> Any:
    return next((row[name] for name in names if row.get(name) is not None and row[name] != ""), None)


def _normalized(row: dict, market: dict, *, number: int, source: str, report: dict) -> dict | None:
    if not isinstance(row, dict):
        raise ValueError(f"Trade {number} in {source} is not an object")
    condition = _first(row, "condition_id", "conditionId")
    row_market = _first(row, "market_id", "marketId")
    if condition and str(condition).lower() != market["condition_id"]:
        report["other_market_rows_skipped"] += 1
        return None
    if not condition and row_market and str(row_market) not in {market["market_id"], market["condition_id"]}:
        report["other_market_rows_skipped"] += 1
        return None
    if not condition and not row_market:
        report["rows_with_assumed_market_identity"] += 1
    actor = _first(row, "actor_id", "wallet", "wallet_id", "proxy_wallet", "proxyWallet")
    if not isinstance(actor, str) or not re.fullmatch(r"0x[0-9a-fA-F]{40}", actor):
        raise ValueError(f"Trade {number} in {source} is missing a wallet address")
    instant = int(row["query_us"]) if row.get("query_us") is not None else timestamp_us(
        _first(row, "block_timestamp", "block_timestamp_seconds", "timestamp", "time", "execution_time_proxy_utc"))
    token = _first(row, "token_id", "tokenId", "asset")
    outcome = _first(row, "outcome", "token_outcome")
    mapped = market["token_outcomes"].get(str(token)) if token is not None else None
    if outcome is None:
        outcome = mapped
    if outcome is None and row.get("outcomeIndex") is not None:
        idx = int(row["outcomeIndex"])
        outcome = next((item["outcome"] for item in market["tokens"] if item.get("outcome_index") == idx), None)
    if outcome is None or (mapped is not None and str(outcome).casefold() != mapped.casefold()):
        raise ValueError(f"Missing or contradictory outcome mapping at {source}:{number}")
    side = str(row.get("side", "")).upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"Missing BUY/SELL side at {source}:{number}")
    shares, price = _first(row, "shares", "size"), row.get("price")
    if shares is None or price is None:
        raise ValueError(f"Missing shares or price at {source}:{number}")
    for value, label in ((shares, "shares"), (price, "price")):
        number_value = Decimal(str(value))
        if not number_value.is_finite() or number_value < 0 or (label == "shares" and number_value == 0):
            raise ValueError(f"Invalid {label} at {source}:{number}")
    identity = row.get("observation_id") or hashlib.sha256(
        json.dumps([source, number, row], sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
    report["observations_read"] += 1
    report["earliest_trade_us"] = min(instant, report.get("earliest_trade_us", instant))
    report["latest_trade_us"] = max(instant, report.get("latest_trade_us", instant))
    return {"actor_id": actor.lower(), "time_us": instant, "time": utc_time(instant),
            "trade": {"side": side, "outcome": str(outcome), "shares": str(shares), "price": str(price)},
            "observation_id": str(identity)}


def load_market_trades(market: dict, *, cache_dir: Path,
                       trades_file: Path | None = None,
                       sqlite_path: Path | None = None,
                       capture_dir: Path | None = None,
                       max_pages: int | None = None,
                       client: Any = None) -> tuple[Iterator[dict], dict]:
    """Return a streaming iterator and its provenance/count report.

    The report's row counts finalize only when the iterator is exhausted. File
    and SQLite inputs may contain other markets, which are excluded. Rows are
    not deduplicated: identical-looking executions may be separate observations.
    Selected-cohort SQLite archives cannot reconstruct excluded actors.
    """
    if trades_file is not None and sqlite_path is not None:
        raise ValueError("Choose either trades_file or sqlite_path")
    report: dict = {"condition_id": market["condition_id"], "observations_read": 0,
                    "other_market_rows_skipped": 0, "rows_with_assumed_market_identity": 0,
                    "canonical_history_complete": False,
                    "timestamp_semantics": "captured_execution_block_time_proxy_not_order_submission_time",
                    "actor_semantics": "API_reported_proxy_wallet_no_inferred_counterparty",
                    "duplicate_policy": "preserve_source_row_multiplicity"}
    if trades_file is not None:
        path = Path(trades_file)
        report.update(source_type="trade_file", source=str(path), source_scope="caller_supplied_captured_observations")
        def raw_rows() -> Iterator[dict]:
            yield from _rows(path)
    elif sqlite_path is not None:
        path = Path(sqlite_path).resolve()
        report.update(source_type="sqlite", source=str(path), source_scope="sqlite_captured_observations_scope_unknown")
        def raw_rows() -> Iterator[dict]:
            db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            db.row_factory = sqlite3.Row
            try:
                tables = {item[0] for item in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "metadata" in tables:
                    metadata = {item[0]: json.loads(item[1]) for item in db.execute("SELECT key,value_json FROM metadata")}
                    details = metadata.get("report", {})
                    report["source_scope"] = metadata.get("source_scope") or details.get("source_scope") or report["source_scope"]
                    report["source_maximum_observations_inclusive"] = details.get("maximum_observations_inclusive")
                    report["source_filter_limitation"] = "An already filtered archive cannot supply actors or rows previously excluded."
                for row in db.execute("SELECT * FROM trades WHERE condition_id = ?", (market["condition_id"],)):
                    yield dict(row)
            finally:
                db.close()
    else:
        root = Path(capture_dir) if capture_dir is not None else Path(cache_dir) / "trade_capture"
        client = client or HttpClient(Path(cache_dir) / "http", compress=True)
        state = ingest_condition(client, condition_id=market["condition_id"], output_dir=root,
                                 compress=True, minimum_size="0.000001", max_pages=max_pages)
        report.update(source_type="polymarket_data_api_v2", source=str(root / market["condition_id"]),
                      source_scope="cursor_traversal_captured_observations", api_traversal_status=state["api_traversal_status"],
                      page_count=state["page_count"], minimum_size_tokens="0.000001",
                      taker_only=False, history_window=state["history_window"],
                      limitations=state["coverage_limitations"])
        if state["api_traversal_status"] != "exhausted":
            raise ValueError("Trade capture is paused before API exhaustion. Resume it without --max-pages before exporting actor intervals.")
        def raw_rows() -> Iterator[dict]:
            for page in state["pages"]:
                yield from _rows(root / market["condition_id"] / page["file"])
    def normalized_rows() -> Iterator[dict]:
        for number, row in enumerate(raw_rows(), 1):
            result = _normalized(row, market, number=number, source=report["source"], report=report)
            if result is not None:
                yield result
    return normalized_rows(), report
