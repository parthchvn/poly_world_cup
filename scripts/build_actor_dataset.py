#!/usr/bin/env python3
"""Generate per-actor market datasets with a single file (Python 3.11+, Mac/Linux).

    python3 build_actor_dataset.py 1897059
    python3 build_actor_dataset.py MARKET_ID --out data/my_market

No pip packages, repository clone, API key, or manually downloaded ESPN file.
Germany-Curacao metadata and 93 saved ESPN events are included. Other matches
use a pinned World Cup registry and saved ESPN data when available, then live
public APIs. ESPN event ID / league can be supplied for other soccer leagues.

Output: actors/<wallet>.jsonl, one actor per file in chronological order.
Each execution timestamp has an open-interval NO_TRADE row and a TRADE row.
Both carry interval news. Simultaneous executions share one TRADE row.
Earlier rows are actor history; history is not copied into every later row.
Default: at most 20 captured executions per actor in the selected binary market.

NO_TRADE describes an observed gap, not a verified conscious decision. Interval
ends use the next observed trade, so these are retrospective sequence records.
ESPN wallclock is an occurrence proxy, not verified historical publication time.
Captured API coverage is recorded; it is not claimed to be full on-chain history.

Adapted from parthchvn/poly_world_cup's builder, HTTP, trades, market lookup,
and ESPN modules. All executable source is contained below; no code is fetched.
Reusing a cache resumes the same capture. Use a new --cache for a fresh capture.
Existing output directories are never overwritten.
"""
from __future__ import annotations
import sys
if sys.version_info < (3, 11):
    raise SystemExit('Python 3.11 or newer is required.')
import unicodedata


"""Small, deterministic file primitives shared by collection commands."""


import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write(path: Path, content: str | bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = content.encode("utf-8") if isinstance(content, str) else content
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def io_write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write(path, "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))



"""Public HTTP JSON retrieval with immutable response bodies and provenance.

A retrieval timestamp means 'we captured this now', never 'known then'. Cache
replays retain the original capture timestamp. Run one writer per cache directory.
"""


import hashlib
import gzip
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl
from urllib.request import Request, urlopen



@dataclass(frozen=True)
class FetchResult:
    data: Any
    url: str
    retrieved_at: str
    body_sha256: str
    from_cache: bool


def request_url(url: str, params: dict | None = None) -> str:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.netloc or parts.username or parts.password:
        raise ValueError("Only public HTTPS URLs without embedded credentials are supported")
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    for key, value in (params or {}).items():
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            pairs.append((key, str(item).lower() if isinstance(item, bool) else str(item)))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(sorted(pairs)), ""))


class HttpClient:
    def __init__(self, cache_dir: Path, refresh: bool = False, *, timeout: float = 30, retries: int = 3, compress: bool = False):
        self.cache_dir = Path(cache_dir)
        self.refresh = refresh
        self.timeout = timeout
        self.retries = retries
        self.compress = compress
        if retries < 0 or timeout <= 0:
            raise ValueError("retries must be nonnegative and timeout positive")

    def get_json(self, url: str, params: dict | None = None) -> FetchResult:
        full_url = request_url(url, params)
        key = hashlib.sha256(full_url.encode()).hexdigest()
        index_path = self.cache_dir / "requests" / f"{key}.json"
        if index_path.exists() and not self.refresh:
            metadata = json.loads(index_path.read_text())
            compressed = metadata.get("body_compression") == "gzip"
            suffix = ".json.gz" if compressed else ".json"
            body = (self.cache_dir / "bodies" / f"{metadata['body_sha256']}{suffix}").read_bytes()
            if compressed:
                body = gzip.decompress(body)
            if metadata["url"] != full_url or hashlib.sha256(body).hexdigest() != metadata["body_sha256"]:
                raise ValueError(f"Cache integrity failure: {index_path}")
            return FetchResult(json.loads(body, parse_float=Decimal), full_url, metadata["retrieved_at"], metadata["body_sha256"], True)

        request = Request(full_url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; PolyWorldCupResearch/0.1)",
            "Accept": "application/json",
        })
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    response_headers = {key: response.headers.get(key) for key in ("Date", "ETag", "Last-Modified")}
                break
            except (HTTPError, URLError, TimeoutError) as error:
                retryable = not isinstance(error, HTTPError) or error.code == 429 or error.code >= 500
                if not retryable or attempt == self.retries:
                    raise
                time.sleep(min(2 ** attempt, 8))

        # Invalid payloads never replace a valid cached response.
        # Amounts/prices must not pass through a binary floating-point round trip.
        data = json.loads(body, parse_float=Decimal)
        body_hash = hashlib.sha256(body).hexdigest()
        retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        suffix = ".json.gz" if self.compress else ".json"
        body_path = self.cache_dir / "bodies" / f"{body_hash}{suffix}"
        if not body_path.exists():
            atomic_write(body_path, gzip.compress(body, compresslevel=6, mtime=0) if self.compress else body)
        metadata = {
            "url": full_url, "retrieved_at": retrieved_at, "body_sha256": body_hash,
            "response_headers": response_headers,
            "historical_availability_verified": False,
            "body_compression": "gzip" if self.compress else None,
        }
        # Preserve every capture record even if a refresh replaces the request index.
        capture_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        io_write_json(self.cache_dir / "captures" / f"{capture_hash}.json", metadata)
        io_write_json(index_path, metadata)
        return FetchResult(data, full_url, retrieved_at, body_hash, False)



"""Resumable collection of *observations* from Polymarket's v2 trade API.

This API does not expose a canonical log/order identifier, omits subthreshold
trades, and serves a bounded history. Exhausting it never certifies a complete
training label window. A row is an API observation, not a reconstructed order.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import gzip
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid

API_URL = "https://data-api.polymarket.com/v2/trades"
SCHEMA_VERSION = 1
HEX_32 = re.compile(r"^0x[0-9a-fA-F]{64}$")
ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")


class TradeIngestionError(ValueError):
    """The source or saved state cannot be safely accepted."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, _json_bytes(value))


@contextmanager
def _writer_lock(directory: Path):
    # Advisory locks are released by the OS after a process crash, so recovery
    # does not require deleting stale marker files. This collector targets POSIX.
    import fcntl

    with (directory / ".writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TradeIngestionError("Another collector is writing this condition") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _decimal_string(value: Any, name: str, *, positive: bool = False,
                    maximum: Decimal | None = None) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise TradeIngestionError(f"Invalid {name}: expected a finite number")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise TradeIngestionError(f"Invalid {name}: expected a finite number") from exc
    if not number.is_finite() or number < 0 or (positive and number <= 0):
        raise TradeIngestionError(f"Invalid {name}: outside allowed range")
    if maximum is not None and number > maximum:
        raise TradeIngestionError(f"Invalid {name}: outside allowed range")
    result = format(number, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def _utc_seconds(value: Any) -> tuple[int, str]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TradeIngestionError("Invalid timestamp: expected nonnegative integer epoch seconds")
    try:
        instant = datetime.fromtimestamp(value, timezone.utc)
    except (ValueError, OverflowError, OSError) as exc:
        raise TradeIngestionError("Invalid timestamp: unsupported epoch seconds") from exc
    return value, instant.isoformat().replace("+00:00", "Z")


def normalize_observation(row: dict, *, condition_id: str, request_url: str,
                          body_sha256: str, retrieved_at: str,
                          row_index: int) -> dict:
    """Normalize one source row while retaining row multiplicity and provenance.

    Row identity depends on the source response and row index. It MUST NOT be
    interpreted as a canonical on-chain fill ID or used to merge two captures.
    """
    if not isinstance(row, dict):
        raise TradeIngestionError("Trade row must be an object")
    condition = row.get("condition_id")
    if not isinstance(condition, str) or condition.lower() != condition_id.lower():
        raise TradeIngestionError("Source returned a trade for a different condition")
    wallet = row.get("proxy_wallet")
    transaction = row.get("transaction_hash")
    token = row.get("token_id")
    if not isinstance(wallet, str) or not ADDRESS.fullmatch(wallet):
        raise TradeIngestionError("Invalid proxy_wallet")
    if not isinstance(transaction, str) or not HEX_32.fullmatch(transaction):
        raise TradeIngestionError("Invalid transaction_hash")
    if not isinstance(token, str) or not token.isascii() or not token.isdigit() or int(token) <= 0:
        raise TradeIngestionError("Invalid token_id: expected positive decimal string")
    if row.get("side") not in ("BUY", "SELL"):
        raise TradeIngestionError("Invalid side: expected BUY or SELL")
    seconds, utc = _utc_seconds(row.get("timestamp"))
    identity = _digest(_json_bytes([request_url, body_sha256, row_index]))
    return {
        "schema_version": SCHEMA_VERSION,
        "observation_id": identity,
        "identity_quality": "api_observation",
        "proxy_wallet": wallet.lower(),
        "condition_id": condition.lower(),
        "token_id": token,
        "side": row["side"],
        "size": _decimal_string(row.get("size"), "size", positive=True),
        "price": _decimal_string(row.get("price"), "price", maximum=Decimal(1)),
        "block_timestamp_seconds": seconds,
        "block_timestamp": utc,
        "transaction_hash": transaction.lower(),
        "maker_taker_role": None,
        "order_id": None,
        "log_index": None,
        "publicly_available_at_upper_bound": None,
        "source": {
            "provider": "polymarket_data_api_v2",
            "request_url": request_url,
            "body_sha256": body_sha256,
            "row_index": row_index,
            "retrieved_at": retrieved_at,
        },
        "quality_flags": [
            "not_a_canonical_fill_identifier",
            "block_time_is_not_decision_time",
            "historical_public_observability_unknown",
            "maker_taker_role_unknown",
        ],
    }


def _page_stem(index: int, cursor: str | None) -> str:
    return f"page-{index:08d}-{_digest(_json_bytes(cursor))[:16]}"


def _validated_page(result: Any, condition_id: str, requested_cursor: str | None,
                    previous_cursors: list[str | None]) -> tuple[list[dict], str | None, bool]:
    envelope = result.data
    if not isinstance(envelope, dict) or not isinstance(envelope.get("data"), list):
        raise TradeIngestionError("Expected a v2 {data, pagination} envelope")
    pagination = envelope.get("pagination")
    if not isinstance(pagination, dict) or type(pagination.get("has_more")) is not bool:
        raise TradeIngestionError("Missing or invalid pagination.has_more")
    has_more = pagination["has_more"]
    next_cursor = pagination.get("next_cursor")
    if has_more:
        if not isinstance(next_cursor, str) or not next_cursor:
            raise TradeIngestionError("has_more requires a nonempty next_cursor")
        if next_cursor in previous_cursors or next_cursor == requested_cursor:
            raise TradeIngestionError("Repeated pagination cursor; traversal cannot advance")
    elif next_cursor not in (None, ""):
        raise TradeIngestionError("Contradictory pagination: terminal page has next_cursor")
    else:
        next_cursor = None
    normalized = [normalize_observation(
        row, condition_id=condition_id, request_url=result.url,
        body_sha256=result.body_sha256, retrieved_at=result.retrieved_at,
        row_index=index,
    ) for index, row in enumerate(envelope["data"])]
    return normalized, next_cursor, has_more


def _initial_manifest(condition_id: str, parameters: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "condition_id": condition_id,
        "api_url": API_URL,
        "parameters": parameters,
        "api_traversal_status": "paused",
        "training_coverage_certified": False,
        "canonical_fill_identity_available": False,
        "history_window": "API condition query: fixed three years before retrieval",
        "minimum_size_filter": {"type": "TOKENS", "amount": parameters["filter_amount"]},
        "coverage_limitations": [
            "API excludes trades smaller than the minimum size filter",
            "API exhaustion does not establish archive completeness or zero trading",
            "Block timestamps do not establish order submission or public availability",
            "No canonical log index, order ID, or maker/taker role is exposed",
            "A live traversal is not a guaranteed database snapshot",
        ],
        "next_cursor": None,
        "page_count": 0,
        "row_count": 0,
        "earliest_block_timestamp": None,
        "latest_block_timestamp": None,
        "pages": [],
    }


def _validate_saved_page(page: dict, directory: Path, *, index: int,
                         cursor: str | None, condition_id: str,
                         parameters: dict, seen_cursors: list) -> None:
    """Check journal metadata against the committed normalized page."""
    stem = _page_stem(index, cursor)
    if (page["page_index"] != index or page["requested_cursor"] != cursor
            or page["condition_id"] != condition_id
            or page["parameters"] != parameters
            or page["file"] not in {f"pages/{stem}.jsonl", f"pages/{stem}.jsonl.gz"}):
        raise TradeIngestionError("Saved page does not match its manifest or cursor chain")
    if type(page["has_more"]) is not bool:
        raise TradeIngestionError("Invalid saved has_more")
    next_cursor = page["next_cursor"]
    if page["has_more"]:
        if (not isinstance(next_cursor, str) or not next_cursor
                or next_cursor == cursor or next_cursor in seen_cursors):
            raise TradeIngestionError("Invalid saved cursor progression")
    elif next_cursor is not None:
        raise TradeIngestionError("Saved terminal page has a next_cursor")
    content = (directory / page["file"]).read_bytes()
    if page["file"].endswith(".gz"):
        content = gzip.decompress(content)
    if _digest(content) != page["normalized_sha256"]:
        raise TradeIngestionError("Committed normalized page checksum mismatch")
    rows = [json.loads(line) for line in content.splitlines()]
    if (type(page["row_count"]) is not int or page["row_count"] != len(rows)
            or any(not isinstance(row, dict) or row.get("condition_id") != condition_id
                   for row in rows)):
        raise TradeIngestionError("Saved page row count or condition is inconsistent")
    timestamps = [row["block_timestamp"] for row in rows]
    if (page["earliest_block_timestamp"] != (min(timestamps) if timestamps else None)
            or page["latest_block_timestamp"] != (max(timestamps) if timestamps else None)):
        raise TradeIngestionError("Saved page timestamp summary is inconsistent")


def _load_manifest(path: Path, condition_id: str, parameters: dict) -> dict:
    try:
        state = json.loads(path.read_text())
        if (not isinstance(state, dict)
                or state["schema_version"] != SCHEMA_VERSION
                or state["condition_id"] != condition_id
                or state["api_url"] != API_URL
                or state["parameters"] != parameters
                or state["minimum_size_filter"] != {"type": "TOKENS", "amount": parameters["filter_amount"]}
                or state["training_coverage_certified"] is not False
                or state["api_traversal_status"] not in ("paused", "exhausted")
                or not isinstance(state["pages"], list)
                or type(state["page_count"]) is not int
                or state["page_count"] != len(state["pages"])
                or type(state["row_count"]) is not int
                or state["row_count"] != sum(p["row_count"] for p in state["pages"])):
            raise TradeIngestionError("Saved manifest is incompatible or inconsistent")
        previous_cursor = None
        seen_cursors = []
        for index, page in enumerate(state["pages"]):
            _validate_saved_page(page, path.parent, index=index,
                                 cursor=previous_cursor, condition_id=condition_id,
                                 parameters=parameters, seen_cursors=seen_cursors)
            if not page["has_more"] and index != len(state["pages"]) - 1:
                raise TradeIngestionError("Saved traversal continues after a terminal page")
            seen_cursors.append(previous_cursor)
            previous_cursor = page["next_cursor"]
        if previous_cursor != state["next_cursor"]:
            raise TradeIngestionError("Saved next_cursor is inconsistent")
        expected_status = ("exhausted" if state["pages"] and not state["pages"][-1]["has_more"]
                           else "paused")
        if state["api_traversal_status"] != expected_status:
            raise TradeIngestionError("Saved traversal status contradicts its last page")
        for key, reducer in (("earliest_block_timestamp", min), ("latest_block_timestamp", max)):
            values = [page[key] for page in state["pages"] if page[key] is not None]
            if state[key] != (reducer(values) if values else None):
                raise TradeIngestionError("Saved manifest timestamp summary is inconsistent")
        return state
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TradeIngestionError("Cannot safely resume saved manifest") from exc


def _commit_page(state: dict, page: dict, manifest_path: Path) -> None:
    state["pages"].append(page)
    state["page_count"] += 1
    state["row_count"] += page["row_count"]
    for key, reducer in (("earliest_block_timestamp", min), ("latest_block_timestamp", max)):
        values = [value for value in (state[key], page[key]) if value is not None]
        state[key] = reducer(values) if values else None
    state["next_cursor"] = page["next_cursor"]
    state["api_traversal_status"] = "paused" if page["has_more"] else "exhausted"
    _atomic_json(manifest_path, state)


def _query_parameters(condition_id: str, limit: int, minimum_size: str = "0.01") -> dict:
    if not isinstance(condition_id, str) or not HEX_32.fullmatch(condition_id):
        raise TradeIngestionError("condition_id must be a 32-byte 0x-prefixed hex string")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise TradeIngestionError("limit must be an integer between 1 and 1000")
    return {"condition": condition_id.lower(), "limit": limit, "taker_only": "false",
            "filter_type": "TOKENS", "filter_amount": _decimal_string(minimum_size, "minimum_size", positive=True)}


def validate_collection(output_dir: Path, *, condition_id: str) -> dict:
    """Read and verify an existing collection without HTTP calls or mutations.

    Check the supported query parameters, cursor chain, normalized page hashes,
    row counts, and timestamp summaries using the same verifier as resume.
    Uncommitted recovery pages are excluded until ingestion commits them.
    This verifies local integrity, not source truth or historical completeness.
    """
    # Validate before constructing any path from the condition identifier.
    _query_parameters(condition_id, 1)
    condition_id = condition_id.lower()
    path = Path(output_dir) / condition_id / "manifest.json"
    try:
        state = json.loads(path.read_text())
        parameters = _query_parameters(condition_id, state["parameters"]["limit"], state["minimum_size_filter"]["amount"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TradeIngestionError("Cannot validate collection manifest") from exc
    return _load_manifest(path, condition_id, parameters)


def ingest_condition(client: Any, *, condition_id: str, output_dir: Path,
                     limit: int = 1000, max_pages: int | None = None,
                     compress: bool = False, minimum_size: str = "0.01") -> dict:
    """Collect one condition, resumably; return its durable manifest.

    ``max_pages`` limits pages committed in this invocation (including recovered
    pages). ``None`` traverses to API exhaustion. The saved limit and filters
    must match on resume. Output lives in ``output_dir / condition_id``. The
    client implements ``get_json(url, params=...)`` returning attributes ``data``,
    ``url``, ``body_sha256``, and ``retrieved_at``. Use one run directory for a
    traversal; a later independent capture belongs in a different directory.

    POSIX advisory locking prevents concurrent writers. Page data and recovery
    metadata are durable before the manifest is advanced. Invalid source pages
    raise TradeIngestionError without committing partial observations.
    """
    parameters = _query_parameters(condition_id, limit, minimum_size)
    if max_pages is not None and (type(max_pages) is not int or max_pages < 1):
        raise TradeIngestionError("max_pages must be a positive integer or None")
    condition_id = condition_id.lower()
    directory = Path(output_dir) / condition_id
    pages_dir = directory / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    with _writer_lock(directory):
        if manifest_path.exists():
            state = _load_manifest(manifest_path, condition_id, parameters)
        else:
            state = _initial_manifest(condition_id, parameters)
            _atomic_json(manifest_path, state)
        committed = 0
        while state["api_traversal_status"] != "exhausted":
            if max_pages is not None and committed >= max_pages:
                break
            cursor = state["next_cursor"]
            index = state["page_count"]
            stem = _page_stem(index, cursor)
            metadata_path = pages_dir / f"{stem}.json"
            page_path = pages_dir / (f"{stem}.jsonl.gz" if compress else f"{stem}.jsonl")
            if metadata_path.exists():
                # A validated page survived a crash before its manifest commit.
                try:
                    page = json.loads(metadata_path.read_text())
                    _validate_saved_page(
                        page, directory, index=index, cursor=cursor,
                        condition_id=condition_id, parameters=parameters,
                        seen_cursors=[p["requested_cursor"] for p in state["pages"]],
                    )
                except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise TradeIngestionError("Cannot safely recover an uncommitted page") from exc
            else:
                params = dict(parameters)
                if cursor is not None:
                    params["cursor"] = cursor
                result = client.get_json(API_URL, params=params)
                rows, next_cursor, has_more = _validated_page(
                    result, condition_id, cursor,
                    [page["requested_cursor"] for page in state["pages"]],
                )
                content = b"".join(_json_bytes(row) for row in rows)
                timestamps = [row["block_timestamp"] for row in rows]
                page = {
                    "page_index": index,
                    "condition_id": condition_id,
                    "parameters": parameters,
                    "file": f"pages/{page_path.name}",
                    "normalized_sha256": _digest(content),
                    "row_count": len(rows),
                    "requested_cursor": cursor,
                    "next_cursor": next_cursor,
                    "has_more": has_more,
                    "request_url": result.url,
                    "body_sha256": result.body_sha256,
                    "retrieved_at": result.retrieved_at,
                    "earliest_block_timestamp": min(timestamps) if timestamps else None,
                    "latest_block_timestamp": max(timestamps) if timestamps else None,
                }
                _atomic_write(page_path, gzip.compress(content, compresslevel=6, mtime=0) if compress else content)
                _atomic_json(metadata_path, page)
            _commit_page(state, page, manifest_path)
            committed += 1
        return state



TEAM_ALIASES = {
    "caboverde": "capeverde",
    "capeverdeislands": "capeverde",
    "congodr": "drcongo",
    "cotedivoire": "ivorycoast",
    "iriran": "iran",
    "korearepublic": "southkorea",
    "bosniaandherzegovina": "bosniaherzegovina",
    "czechrepublic": "czechia",
    "turkey": "turkiye",
}


def canonical_team(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.casefold())
    key = "".join(character for character in text if character.isalnum())
    return TEAM_ALIASES.get(key, key)




"""Fetch ESPN soccer commentary once and retain its actual textual context.

An ESPN wallclock is an event-time proxy, not proof of the time the feed was
published. A match clock alone is never silently converted to a UTC timestamp.
No model, article-body scraper, or paid API is required.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any


SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
CORE = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _datetime(value: Any) -> datetime | None:
    """Read timezone-aware ISO or Unix seconds/milliseconds/microseconds."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float, Decimal)) or (
            isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)?", value)
        ):
            seconds = Decimal(str(value))
            if seconds >= Decimal("1e14"):
                seconds /= 1_000_000
            elif seconds >= Decimal("1e11"):
                seconds /= 1_000
            return EPOCH + timedelta(microseconds=int(seconds * 1_000_000))
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        return None


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _us(value: datetime) -> int:
    delta = value - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds)


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def _source(result: Any) -> dict:
    return {"url": result.url, "retrieved_at": result.retrieved_at,
            "body_sha256": result.body_sha256, "from_cache": result.from_cache}


def _read(path: Path) -> Any:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        if str(path).removesuffix(".gz").endswith(".jsonl"):
            return [json.loads(line, parse_float=Decimal) for line in stream if line.strip()]
        return json.load(stream, parse_float=Decimal)


def discover_espn_event(client: Any, *, fixture_date: str, teams: list[str] | tuple[str, str],
                        league: str = "fifa.world") -> tuple[str, dict, list[dict]]:
    """Require an exact two-team match on the requested UTC date (+/- one day)."""
    if len(teams) != 2:
        raise ValueError("ESPN discovery requires exactly two team names")
    date = datetime.fromisoformat(str(fixture_date)[:10]).date()
    dates = (date - timedelta(days=1)).strftime("%Y%m%d") + "-" + (date + timedelta(days=1)).strftime("%Y%m%d")
    response = client.get_json(f"{SITE}/{league}/scoreboard", {"dates": dates, "limit": 1000})
    wanted = {canonical_team(name) for name in teams}
    candidates = []
    for event in response.data.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        names = {canonical_team(str(c.get("team", {}).get("displayName", "")))
                 for c in competitions[0].get("competitors", [])}
        if names == wanted:
            candidates.append(event)
    if len(candidates) != 1:
        ids = [str(event.get("id")) for event in candidates]
        raise ValueError(f"ESPN match is ambiguous or missing ({ids}); provide --espn-event-id and --league")
    return str(candidates[0]["id"]), candidates[0], [_source(response)]


def _play_rows(payload: Any) -> list[dict]:
    """Unify summary commentary/keyEvents and the core plays endpoint."""
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = list(payload.get("commentary", [])) + list(payload.get("keyEvents", []))
        raw += list(payload.get("plays", [])) + list(payload.get("items", []))
        if not raw and isinstance(payload.get("events"), list):
            # Offline normalized context documents, not a scoreboard.
            raw = [item for item in payload["events"] if "text" in item or "headline" in item]
        if not raw and ("text" in payload or "headline" in payload):
            raw = [payload]
    else:
        raise ValueError("ESPN input must be an object or a list")
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        row = dict(item.get("play") or item)
        if item.get("text"):
            row["text"] = item["text"]
        if "clock" not in row and isinstance(item.get("time"), dict):
            row["clock"] = item["time"]
        for field in ("timestamp_us", "time_utc", "available_at_utc", "observed_at_utc", "published", "wallclock", "sequence"):
            if field in item and field not in row:
                row[field] = item[field]
        rows.append(row)
    return rows


def _article_rows(payload: Any, event_id: str, team_ids: set[str]) -> list[dict]:
    """Keep the match report and explicit team/match-related headline metadata.

    summary.news is a league widget: do not attribute every article to this match.
    Article bodies are deliberately not requested or copied.
    """
    if not isinstance(payload, dict):
        return []
    candidates = [payload.get("article", {})]
    widget = payload.get("news", {})
    if isinstance(widget, dict):
        candidates += widget.get("articles", [])
    candidates += payload.get("articles", [])
    selected = []
    for article in candidates:
        if not isinstance(article, dict) or not article.get("headline"):
            continue
        explicit_game = str(article.get("gameId", ""))
        category_teams = {str(c.get("teamId", c.get("team", {}).get("id", "")))
                          for c in article.get("categories", []) if isinstance(c, dict)}
        if explicit_game == event_id or (team_ids & category_teams):
            selected.append({**article, "_article": True})
    return selected


def _kind(row: dict) -> str:
    if row.get("_article") or row.get("headline"):
        return "headline"
    if row.get("kind"):
        return str(row["kind"])
    play_type = row.get("type", {})
    if isinstance(play_type, dict):
        return str(play_type.get("type") or play_type.get("text") or "commentary")
    return str(row.get("kind") or play_type or "commentary")


def _play_text(row: dict) -> str:
    text = _text(row.get("headline") or row.get("text") or row.get("shortText"))
    if text:
        return text
    names = [_text(p.get("athlete", {}).get("displayName"))
             for p in row.get("participants", []) if isinstance(p, dict)]
    names = [name for name in names if name]
    team = _text(row.get("team", {}).get("displayName"))
    kind = _kind(row)
    # Structured facts only, never an inferred rationale or invented incident.
    parts = [kind.replace("-", " ")]
    if names:
        parts.append(", ".join(names))
    if team:
        parts.append(team)
    return ": ".join(parts) if names or team else ""


def _raw_id(row: dict) -> str:
    if row.get("news_id") or row.get("id"):
        return str(row.get("news_id") or row["id"])
    identity = {key: row.get(key) for key in ("text", "headline", "period", "clock", "sequence")}
    return "text-" + hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _clock(row: dict) -> tuple[int | None, Decimal | None, str | None]:
    period = row.get("period", {})
    period = period.get("number") if isinstance(period, dict) else period
    clock = row.get("clock", {})
    if not isinstance(clock, dict):
        clock = {}
    try:
        period = int(period) if period is not None else None
        seconds = Decimal(str(clock["value"])) if clock.get("value") is not None else None
    except (ValueError, TypeError, InvalidOperation):
        return None, None, None
    return period, seconds, clock.get("displayValue")


def collect_espn_context(client: Any, *, event_id: str | None = None,
                         league: str = "fifa.world", fixture_date: str | None = None,
                         teams: list[str] | tuple[str, str] | None = None,
                         espn_files: list[str | Path] | None = None,
                         time_map: dict | str | Path | None = None,
                         time_policy: str = "provider", allow_clock_estimates: bool = False,
                         include_core_plays: bool = False) -> dict:
    """Return timed_events and untimed_events containing real ESPN text.

    ``time_map`` accepts ``{"PLAY_ID": "ISO_UTC"}`` or an object containing
    ``events`` with that mapping and optional ``period_anchors``. A period anchor
    is ``{"1": {"time_utc": "...Z", "clock_seconds": 0}}``. Clock estimates
    require the explicit opt-in and an anchor for that same period. Their timing
    remains approximate because match-clock stoppages are not reconstructed.

    Offline inputs can be saved summary JSON, core play-page JSON, JSONL, gzip,
    or normalized text events with ``time_utc``/``timestamp_us``. Their declared
    timestamps are carried through with an explicit provenance label.
    """
    if time_policy not in {"provider", "observed"}:
        raise ValueError("time_policy must be provider or observed")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", league):
        raise ValueError("Invalid ESPN league slug")
    if isinstance(time_map, (str, Path)):
        time_map = _read(Path(time_map))
    time_map = time_map or {}
    if not isinstance(time_map, dict):
        raise ValueError("The time map must be a JSON object")
    mapped_times = time_map.get("events", time_map)
    if not isinstance(mapped_times, dict):
        raise ValueError("time_map.events must be an event-ID to timestamp object")
    sources, payloads, warnings = [], [], []
    header: dict = {}
    if espn_files:
        for filename in espn_files:
            path = Path(filename)
            payloads.append(_read(path))
            sources.append({"path": str(path), "body_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "source_kind": "user_supplied_espn_file"})
    else:
        if not event_id:
            if not fixture_date or not teams:
                raise ValueError("Supply an ESPN event ID or a fixture date and two teams")
            event_id, _, discovery_sources = discover_espn_event(client, fixture_date=fixture_date, teams=teams, league=league)
            sources.extend(discovery_sources)
        event_id = str(event_id)
        if not event_id.isdigit():
            raise ValueError("ESPN event ID must be numeric")
        summary = client.get_json(f"{SITE}/{league}/summary", {"event": event_id})
        payloads.append(summary.data)
        sources.append(_source(summary))
        if include_core_plays:
            page = 1
            while True:
                result = client.get_json(f"{CORE}/{league}/events/{event_id}/competitions/{event_id}/plays",
                                         {"limit": 1000, "page": page})
                payloads.append(result.data)
                sources.append(_source(result))
                pages = int(result.data.get("pageCount", 1))
                if page >= pages:
                    break
                page += 1
                if page > 100:
                    raise ValueError("ESPN plays endpoint exceeded 100 pages")
    for payload in payloads:
        if isinstance(payload, dict) and payload.get("header"):
            header = payload["header"]
            break
    event_id = str(event_id or header.get("id") or "offline")
    competitions = header.get("competitions", [])
    competition = competitions[0] if competitions else {}
    kickoff = _datetime(competition.get("date") or header.get("date"))
    team_ids = {str(c.get("team", {}).get("id", "")) for c in competition.get("competitors", [])}
    rows: dict[str, dict] = {}
    for payload in payloads:
        for row in _play_rows(payload) + _article_rows(payload, event_id, team_ids):
            if row.get("valid") is False:
                continue
            identity = ("article:" if row.get("_article") else "play:") + _raw_id(row)
            previous = rows.get(identity, {})
            # Commentary has more readable text; supplement its missing fields
            # from the core feed without replacing it with low-level play text.
            merged = {**row, **previous}
            for key, value in row.items():
                if merged.get(key) in (None, "", {}):
                    merged[key] = value
            rows[identity] = merged
    anchors = dict(time_map.get("period_anchors", {}))
    for row in rows.values():
        period, seconds, _ = _clock(row)
        wallclock = _datetime(row.get("wallclock"))
        if period and seconds is not None and wallclock and _kind(row).lower() in {"kickoff", "start-period"}:
            if kickoff is None or kickoff - timedelta(hours=6) <= wallclock <= kickoff + timedelta(hours=12):
                anchors.setdefault(str(period), {"time_utc": _utc(wallclock), "clock_seconds": str(seconds)})
    observed = min((_datetime(source.get("retrieved_at")) for source in sources if source.get("retrieved_at")), default=None)
    timed, untimed = [], []
    for identity, row in rows.items():
        text = _play_text(row)
        if not text:
            continue
        raw_id = _raw_id(row)
        news_id = f"espn:{event_id}:{identity}"
        period, seconds, display = _clock(row)
        when, basis = None, None
        explicit = mapped_times.get(news_id, mapped_times.get(raw_id))
        if isinstance(explicit, dict):
            explicit = explicit.get("time_utc") or explicit.get("available_at_utc") or explicit.get("timestamp_us")
        if explicit is not None:
            when, basis = _datetime(explicit), "user_supplied_event_time"
            if when is None:
                raise ValueError(f"Invalid explicit timestamp for ESPN event {raw_id}")
        if when is None and time_policy == "observed":
            when, basis = _datetime(row.get("observed_at_utc")) or observed, "captured_at"
        if when is None and time_policy == "provider":
            for key in ("available_at_utc", "observed_at_utc", "time_utc", "timestamp_us"):
                if row.get(key) is not None:
                    when = _datetime(row[key])
                    if when:
                        basis = "input_" + key
                        break
            if when is None and _kind(row) == "headline":
                published = _datetime(row.get("published") or row.get("originallyPosted"))
                modified = _datetime(row.get("lastModified"))
                when = max(published, modified) if published and modified else published
                basis = "publisher_revision_time_proxy" if when else None
            if when is None:
                when = _datetime(row.get("wallclock"))
                basis = "provider_event_time_proxy" if when else None
                if when and kickoff and not kickoff - timedelta(hours=6) <= when <= kickoff + timedelta(hours=12):
                    warnings.append(f"Ignored implausible wallclock for {raw_id}: {_utc(when)}")
                    when, basis = None, None
        if when is None and allow_clock_estimates and period and seconds is not None:
            anchor = anchors.get(str(period), {})
            anchor_time = _datetime(anchor.get("time_utc"))
            if anchor_time is not None and anchor.get("clock_seconds") is not None:
                offset = seconds - Decimal(str(anchor["clock_seconds"]))
                when = anchor_time + timedelta(microseconds=int(offset * 1_000_000))
                basis = "estimated_from_period_anchor"
        event = {"news_id": news_id, "source": "ESPN", "event_id": event_id,
                 "source_event_id": raw_id, "kind": _kind(row), "text": text,
                 "match_clock": display, "period": period,
                 "time_utc": _utc(when) if when else None,
                 "timestamp_us": _us(when) if when else None,
                 "available_at_utc": _utc(when) if when else None,
                 "time_basis": basis or "no_absolute_time",
                 "historical_availability_verified": False}
        if when:
            timed.append(event)
        else:
            untimed.append(event)
    timed.sort(key=lambda event: (event["timestamp_us"], event["news_id"]))
    untimed.sort(key=lambda event: event["news_id"])
    return {"event_id": event_id, "league": league, "fixture_name": header.get("name"),
            "scheduled_kickoff_utc": _utc(kickoff) if kickoff else None,
            "timed_events": timed, "untimed_events": untimed, "sources": sources,
            "timing_counts": dict(Counter(event["time_basis"] for event in timed + untimed)),
            "warnings": warnings,
            "news_scope": "match_commentary_and_explicit_match_or_team_headlines_in_saved_responses",
            "complete_historical_espn_news_archive": False,
            "historical_availability_verified": False}



"""Market lookup and streaming captured trade inputs for actor interval exports.

The live path uses the repository's resumable cursor-based v2 collector. It
never substitutes the old offset-limited endpoint. API exhaustion describes
the API traversal, not complete on-chain history. No actor filtering happens
here: the caller counts each actor's observations before applying its cutoff.
"""

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


GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
_CONDITION = re.compile(r"^0x[0-9a-fA-F]{64}$")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_DEFAULT_REGISTRY = Path("__unused_registry__")


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
    if registry is None:
        if any(_matches(row, supplied) for row in BUNDLED_REGISTRY['contracts']):
            registry = BUNDLED_REGISTRY
        elif client is not None:
            try:
                registry = client.get_json(REPOSITORY_RAW + '/datasets/world_cup_2026_tournament_lt20_v2_evidence/registry.json').data
            except (HTTPError, URLError, TimeoutError) as error:
                print(f'Saved registry unavailable ({error}); resolving with Gamma.', file=sys.stderr)
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
    allowed_outcomes = {str(x).casefold(): str(x) for x in market['token_outcomes'].values()}
    if str(outcome).casefold() not in allowed_outcomes:
        raise ValueError(f'Unknown outcome at {source}:{number}')
    outcome = allowed_outcomes[str(outcome).casefold()]
    side = str(row.get("side", "")).upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"Missing BUY/SELL side at {source}:{number}")
    shares, price = _first(row, "shares", "size"), row.get("price")
    if shares is None or price is None:
        raise ValueError(f"Missing shares or price at {source}:{number}")
    for value, label in ((shares, "shares"), (price, "price")):
        number_value = Decimal(str(value))
        if not number_value.is_finite() or number_value < 0 or (label == "price" and number_value > 1) or (label == "shares" and number_value == 0):
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



REPOSITORY_RAW = "https://raw.githubusercontent.com/parthchvn/poly_world_cup/e3bb6e1485cd243853431709eae6629ab86547e2"


BUNDLED_REGISTRY = {'contracts': [{'accepting_orders_at': '2026-04-06T22:46:05Z', 'condition_id': '0x18f73aca12019d3fc2a03e7af28f6ebcec12634413819605d7cfa3db20073f26', 'created_at': '2026-04-06T22:19:08.375517Z', 'event_id': '351723', 'event_slug': 'fifwc-ger-kor-2026-06-14', 'feature_eligible': False, 'fixture_id': 'espn:760422', 'game_start_time': '2026-06-14T17:00:00Z', 'market_id': '1897059', 'market_slug': 'fifwc-ger-kor-2026-06-14-draw', 'question': 'Will Germany vs. Curaçao end in a draw?', 'regulation_time_explicit_in_rules': True, 'retrospective_metadata': True, 'rules_text': 'In the upcoming game, scheduled for June 14, 2026\nIf the game ends in a draw, this market will resolve to "Yes".\nOtherwise, this market will resolve to "No".\nIf the game is postponed, this market will remain open until the game has been completed.\nIf the game is canceled entirely, with no make-up game, this market will resolve to "Yes".\nThis market refers only to the outcome within the first 90 minutes of regular play plus stoppage time.\n\nThe primary resolution source for this market is the official statistics of the event as recognized by the governing body or event organizers. However, if the governing body or event organizers have not published final match statistics within 2 hours after the event\'s conclusion, a consensus of credible reporting may be used instead.', 'selection': 'draw', 'source': {'body_sha256': 'b68b5beadc959d2268e67d4fda13f02d00706a5bcc5c226530f9befb3a1fc92a', 'feature_eligible': False, 'retrieved_at': '2026-09-22T23:45:26.069177Z', 'retrospective_metadata': True, 'url': 'https://gamma-api.polymarket.com/events?ascending=true&limit=100&offset=0&order=id&series_id=11433'}, 'sports_market_type': 'moneyline', 'start_date': '2026-04-06T22:47:27.348152Z', 'tokens': [{'outcome': 'Yes', 'outcome_index': 0, 'token_id': '43065699114218738643061430368238869232700747238329067809024727726995177258597'}, {'outcome': 'No', 'outcome_index': 1, 'token_id': '70361345364071436882740201217779608050753190101468453278134436864619706671216'}]}], 'fixtures': [{'away_team': {'canonical_name': 'curacao', 'espn_team_id': '11678', 'name': 'Curaçao'}, 'candidate_event_ids': ['351723'], 'earliest_accepting_orders_at': '2026-04-06T22:45:57Z', 'espn_event_id': '760422', 'feature_eligible': False, 'fixture_id': 'espn:760422', 'gamma_kickoff_utc': '2026-06-14T17:00:00Z', 'home_team': {'canonical_name': 'germany', 'espn_team_id': '481', 'name': 'Germany'}, 'kickoff_difference_seconds': 0, 'kickoff_utc': '2026-06-14T17:00:00Z', 'mapping_status': 'matched', 'polymarket_event_id': '351723', 'polymarket_event_slug': 'fifwc-ger-kor-2026-06-14', 'retrospective_metadata': True, 'source': {'body_sha256': '028729fac08ae6ab2ccb9d417323c85f1a524b52d5d419729d945559eca262d5', 'feature_eligible': False, 'retrieved_at': '2026-09-22T23:45:21.143972Z', 'retrospective_metadata': True, 'url': 'https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=2026&limit=1000'}, 'stage': 'group-stage', 'year': 2026}]}


BUNDLED_ESPN = {'header': {'id': '760422', 'name': None, 'competitions': [{'id': '760422', 'date': '2026-06-14T17:00Z', 'competitors': [{'team': {'id': '481', 'displayName': 'Germany'}}, {'team': {'id': '11678', 'displayName': 'Curaçao'}}]}], 'date': None}, 'plays': [{'id': '49506820', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 47.0, 'displayValue': "1'"}, 'wallclock': '2026-06-14T17:02:36Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Aleksandar Pavlovic'}}, {'athlete': {'displayName': 'Tahith Chong'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Aleksandar Pavlovic, Tahith Chong.'}, {'id': '49506879', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 130.0, 'displayValue': "3'"}, 'wallclock': '2026-06-14T17:03:59Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Aleksandar Pavlovic'}}, {'athlete': {'displayName': 'Deveron Fonville'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Aleksandar Pavlovic, Deveron Fonville.'}, {'id': '49506937', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 252.0, 'displayValue': "5'"}, 'wallclock': '2026-06-14T17:06:01Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Juninho Bacuna'}}, {'athlete': {'displayName': 'Joshua Kimmich'}}], 'text': 'Event: Foul. Team: Curaçao. Players involved: Juninho Bacuna, Joshua Kimmich.'}, {'id': '49506990', 'type': {'id': '135', 'text': 'Shot Blocked', 'type': 'shot-blocked'}, 'period': {'number': 1}, 'clock': {'value': 304.0, 'displayValue': "6'"}, 'wallclock': '2026-06-14T17:06:54Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Jamal Musiala'}}, {'athlete': {'displayName': 'Nathaniel Brown'}}], 'text': 'Event: Shot Blocked. Team: Germany. Players involved: Jamal Musiala, Nathaniel Brown.'}, {'id': '49506991', 'type': {'id': '70', 'text': 'Goal', 'type': 'goal'}, 'period': {'number': 1}, 'clock': {'value': 317.0, 'displayValue': "6'"}, 'wallclock': '2026-06-14T17:07:07Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Felix Nmecha'}}, {'athlete': {'displayName': 'Florian Wirtz'}}], 'text': 'Event: Goal. Team: Germany. Players involved: Felix Nmecha, Florian Wirtz.'}, {'id': '49507011', 'type': {'id': '135', 'text': 'Shot Blocked', 'type': 'shot-blocked'}, 'period': {'number': 1}, 'clock': {'value': 405.0, 'displayValue': "7'"}, 'wallclock': '2026-06-14T17:08:35Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Jamal Musiala'}}], 'text': 'Event: Shot Blocked. Team: Germany. Players involved: Jamal Musiala.'}, {'id': '49507030', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 1}, 'clock': {'value': 488.0, 'displayValue': "9'"}, 'wallclock': '2026-06-14T17:09:57Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Felix Nmecha'}}, {'athlete': {'displayName': 'Jamal Musiala'}}], 'text': 'Event: Shot Off Target. Team: Germany. Players involved: Felix Nmecha, Jamal Musiala.'}, {'id': '49507115', 'type': {'id': '135', 'text': 'Shot Blocked', 'type': 'shot-blocked'}, 'period': {'number': 1}, 'clock': {'value': 588.0, 'displayValue': "10'"}, 'wallclock': '2026-06-14T17:11:37Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Florian Wirtz'}}], 'text': 'Event: Shot Blocked. Team: Germany. Players involved: Florian Wirtz.'}, {'id': '49507141', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 1}, 'clock': {'value': 621.0, 'displayValue': "11'"}, 'wallclock': '2026-06-14T17:12:10Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Leroy Sané'}}, {'athlete': {'displayName': 'Jamal Musiala'}}], 'text': 'Event: Shot Off Target. Team: Germany. Players involved: Leroy Sané, Jamal Musiala.'}, {'id': '49507172', 'type': {'id': '106', 'text': 'Shot On Target', 'type': 'shot-on-target'}, 'period': {'number': 1}, 'clock': {'value': 702.0, 'displayValue': "12'"}, 'wallclock': '2026-06-14T17:13:31Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Felix Nmecha'}}, {'athlete': {'displayName': 'Aleksandar Pavlovic'}}], 'text': 'Event: Shot On Target. Team: Germany. Players involved: Felix Nmecha, Aleksandar Pavlovic.'}, {'id': '49507190', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 723.0, 'displayValue': "13'"}, 'wallclock': '2026-06-14T17:13:53Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Jürgen Locadia'}}, {'athlete': {'displayName': 'Joshua Kimmich'}}], 'text': 'Event: Foul. Team: Curaçao. Players involved: Jürgen Locadia, Joshua Kimmich.'}, {'id': '49507236', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 1}, 'clock': {'value': 827.0, 'displayValue': "14'"}, 'wallclock': '2026-06-14T17:15:37Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Florian Wirtz'}}, {'athlete': {'displayName': 'Nathaniel Brown'}}], 'text': 'Event: Shot Off Target. Team: Germany. Players involved: Florian Wirtz, Nathaniel Brown.'}, {'id': '49507298', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 1019.0, 'displayValue': "17'"}, 'wallclock': '2026-06-14T17:18:48Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Jonathan Tah'}}, {'athlete': {'displayName': 'Jürgen Locadia'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Jonathan Tah, Jürgen Locadia.'}, {'id': '49507351', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 1067.0, 'displayValue': "18'"}, 'wallclock': '2026-06-14T17:19:37Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Jamal Musiala'}}, {'athlete': {'displayName': 'Tahith Chong'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Jamal Musiala, Tahith Chong.'}, {'id': '49507384', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 1}, 'clock': {'value': 1138.0, 'displayValue': "19'"}, 'wallclock': '2026-06-14T17:20:47Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Leandro Bacuna'}}, {'athlete': {'displayName': 'Sontje Hansen'}}], 'text': 'Event: Shot Off Target. Team: Curaçao. Players involved: Leandro Bacuna, Sontje Hansen.'}, {'id': '49507405', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 1221.0, 'displayValue': "21'"}, 'wallclock': '2026-06-14T17:22:10Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Felix Nmecha'}}, {'athlete': {'displayName': 'Juninho Bacuna'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Felix Nmecha, Juninho Bacuna.'}, {'id': '49507416', 'type': {'id': '70', 'text': 'Goal', 'type': 'goal'}, 'period': {'number': 1}, 'clock': {'value': 1244.0, 'displayValue': "21'"}, 'wallclock': '2026-06-14T17:22:33Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Livano Comenencia'}}], 'text': 'Event: Goal. Team: Curaçao. Players involved: Livano Comenencia.'}, {'id': '49507426', 'type': {'id': '129', 'text': 'Start Delay', 'type': 'start-delay'}, 'period': {'number': 1}, 'clock': {'value': 1361.0, 'displayValue': "23'"}, 'wallclock': '2026-06-14T17:24:30Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: Start Delay. Team: Germany. Reported incident: drinks break.'}, {'id': '49507432', 'type': {'id': '130', 'text': 'End Delay', 'type': 'end-delay'}, 'period': {'number': 1}, 'clock': {'value': 1483.0, 'displayValue': "25'"}, 'wallclock': '2026-06-14T17:26:33Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: End Delay. Team: Germany.'}, {'id': '49507451', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 1592.0, 'displayValue': "27'"}, 'wallclock': '2026-06-14T17:28:22Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Leandro Bacuna'}}, {'athlete': {'displayName': 'Jamal Musiala'}}], 'text': 'Event: Foul. Team: Curaçao. Players involved: Leandro Bacuna, Jamal Musiala.'}, {'id': '49507455', 'type': {'id': '106', 'text': 'Shot On Target', 'type': 'shot-on-target'}, 'period': {'number': 1}, 'clock': {'value': 1633.0, 'displayValue': "28'"}, 'wallclock': '2026-06-14T17:29:03Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Nico Schlotterbeck'}}, {'athlete': {'displayName': 'Joshua Kimmich'}}], 'text': 'Event: Shot On Target. Team: Germany. Players involved: Nico Schlotterbeck, Joshua Kimmich.'}, {'id': '49507454', 'type': {'id': '95', 'text': 'Corner Awarded', 'type': 'corner-awarded'}, 'period': {'number': 1}, 'clock': {'value': 1635.0, 'displayValue': "28'"}, 'wallclock': '2026-06-14T17:29:05Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: Corner Awarded. Team: Germany.'}, {'id': '49507471', 'type': {'id': '95', 'text': 'Corner Awarded', 'type': 'corner-awarded'}, 'period': {'number': 1}, 'clock': {'value': 1699.0, 'displayValue': "29'"}, 'wallclock': '2026-06-14T17:30:08Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: Corner Awarded. Team: Germany.'}, {'id': '49507491', 'type': {'id': '135', 'text': 'Shot Blocked', 'type': 'shot-blocked'}, 'period': {'number': 1}, 'clock': {'value': 1775.0, 'displayValue': "30'"}, 'wallclock': '2026-06-14T17:31:25Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Aleksandar Pavlovic'}}, {'athlete': {'displayName': 'Leroy Sané'}}], 'text': 'Event: Shot Blocked. Team: Germany. Players involved: Aleksandar Pavlovic, Leroy Sané.'}, {'id': '49507492', 'type': {'id': '95', 'text': 'Corner Awarded', 'type': 'corner-awarded'}, 'period': {'number': 1}, 'clock': {'value': 1777.0, 'displayValue': "30'"}, 'wallclock': '2026-06-14T17:31:26Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: Corner Awarded. Team: Germany.'}, {'id': '49507540', 'type': {'id': '135', 'text': 'Shot Blocked', 'type': 'shot-blocked'}, 'period': {'number': 1}, 'clock': {'value': 1919.0, 'displayValue': "32'"}, 'wallclock': '2026-06-14T17:33:48Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Leroy Sané'}}, {'athlete': {'displayName': 'Florian Wirtz'}}], 'text': 'Event: Shot Blocked. Team: Germany. Players involved: Leroy Sané, Florian Wirtz.'}, {'id': '49507566', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 2028.0, 'displayValue': "34'"}, 'wallclock': '2026-06-14T17:35:38Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Nico Schlotterbeck'}}, {'athlete': {'displayName': 'Jürgen Locadia'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Nico Schlotterbeck, Jürgen Locadia.'}, {'id': '49507607', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 1}, 'clock': {'value': 2104.0, 'displayValue': "36'"}, 'wallclock': '2026-06-14T17:36:53Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Juninho Bacuna'}}, {'athlete': {'displayName': 'Sontje Hansen'}}], 'text': 'Event: Shot Off Target. Team: Curaçao. Players involved: Juninho Bacuna, Sontje Hansen.'}, {'id': '49507655', 'type': {'id': '95', 'text': 'Corner Awarded', 'type': 'corner-awarded'}, 'period': {'number': 1}, 'clock': {'value': 2180.0, 'displayValue': "37'"}, 'wallclock': '2026-06-14T17:38:10Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: Corner Awarded. Team: Germany.'}, {'id': '49507667', 'type': {'id': '137', 'text': 'Goal - Header', 'type': 'goal---header'}, 'period': {'number': 1}, 'clock': {'value': 2250.0, 'displayValue': "38'"}, 'wallclock': '2026-06-14T17:39:20Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Nico Schlotterbeck'}}, {'athlete': {'displayName': 'Nathaniel Brown'}}], 'text': 'Event: Goal - Header. Team: Germany. Players involved: Nico Schlotterbeck, Nathaniel Brown.'}, {'id': '49507711', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 2333.0, 'displayValue': "39'"}, 'wallclock': '2026-06-14T17:40:42Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Leroy Sané'}}, {'athlete': {'displayName': 'Deveron Fonville'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Leroy Sané, Deveron Fonville.'}, {'id': '49507725', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 2385.0, 'displayValue': "40'"}, 'wallclock': '2026-06-14T17:41:34Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Juninho Bacuna'}}, {'athlete': {'displayName': 'Kai Havertz'}}], 'text': 'Event: Foul. Team: Curaçao. Players involved: Juninho Bacuna, Kai Havertz.'}, {'id': '49507834', 'type': {'id': '95', 'text': 'Corner Awarded', 'type': 'corner-awarded'}, 'period': {'number': 1}, 'clock': {'value': 2588.0, 'displayValue': "44'"}, 'wallclock': '2026-06-14T17:44:58Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: Corner Awarded. Team: Germany.'}, {'id': '49507840', 'type': {'id': '95', 'text': 'Corner Awarded', 'type': 'corner-awarded'}, 'period': {'number': 1}, 'clock': {'value': 2633.0, 'displayValue': "44'"}, 'wallclock': '2026-06-14T17:45:42Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: Corner Awarded. Team: Germany.'}, {'id': '49507865', 'type': {'id': '135', 'text': 'Shot Blocked', 'type': 'shot-blocked'}, 'period': {'number': 1}, 'clock': {'value': 2691.0, 'displayValue': "45'"}, 'wallclock': '2026-06-14T17:46:40Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Nathaniel Brown'}}], 'text': 'Event: Shot Blocked. Team: Germany. Players involved: Nathaniel Brown.'}, {'id': '49507866', 'type': {'id': '135', 'text': 'Shot Blocked', 'type': 'shot-blocked'}, 'period': {'number': 1}, 'clock': {'value': 2693.0, 'displayValue': "45'"}, 'wallclock': '2026-06-14T17:46:42Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Aleksandar Pavlovic'}}, {'athlete': {'displayName': 'Leroy Sané'}}], 'text': 'Event: Shot Blocked. Team: Germany. Players involved: Aleksandar Pavlovic, Leroy Sané.'}, {'id': '49507891', 'type': {'id': '135', 'text': 'Shot Blocked', 'type': 'shot-blocked'}, 'period': {'number': 1}, 'clock': {'value': 2696.0, 'displayValue': "45'"}, 'wallclock': '2026-06-14T17:46:45Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Kai Havertz'}}], 'text': 'Event: Shot Blocked. Team: Germany. Players involved: Kai Havertz.'}, {'id': '49507892', 'type': {'id': '106', 'text': 'Shot On Target', 'type': 'shot-on-target'}, 'period': {'number': 1}, 'clock': {'value': 2700.0, 'displayValue': "45'+2'"}, 'wallclock': '2026-06-14T17:48:08Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Sontje Hansen'}}, {'athlete': {'displayName': 'Livano Comenencia'}}], 'text': 'Event: Shot On Target. Team: Curaçao. Players involved: Sontje Hansen, Livano Comenencia.'}, {'id': '49507931', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 1}, 'clock': {'value': 2700.0, 'displayValue': "45'+3'"}, 'wallclock': '2026-06-14T17:48:58Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Riechedly Bazoer'}}, {'athlete': {'displayName': 'Felix Nmecha'}}], 'text': 'Event: Foul. Team: Curaçao. Players involved: Riechedly Bazoer, Felix Nmecha.'}, {'id': '49507949', 'type': {'id': '98', 'text': 'Penalty - Scored', 'type': 'penalty---scored'}, 'period': {'number': 1}, 'clock': {'value': 2700.0, 'displayValue': "45'+5'"}, 'wallclock': '2026-06-14T17:50:59Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Kai Havertz'}}], 'text': 'Event: Penalty - Scored. Team: Germany. Players involved: Kai Havertz.'}, {'id': '49508193', 'type': {'id': '76', 'text': 'Substitution', 'type': 'substitution'}, 'period': {'number': 2}, 'clock': {'value': 2700.0, 'displayValue': "45'"}, 'wallclock': '2026-06-14T18:07:56Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Jeremy Antonisse'}}, {'athlete': {'displayName': 'Sontje Hansen'}}], 'text': 'Event: Substitution. Team: Curaçao. Players involved: Jeremy Antonisse, Sontje Hansen.'}, {'id': '49508227', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 2750.0, 'displayValue': "46'"}, 'wallclock': '2026-06-14T18:09:22Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Riechedly Bazoer'}}, {'athlete': {'displayName': 'Jamal Musiala'}}], 'text': 'Event: Foul. Team: Curaçao. Players involved: Riechedly Bazoer, Jamal Musiala.'}, {'id': '49508258', 'type': {'id': '70', 'text': 'Goal', 'type': 'goal'}, 'period': {'number': 2}, 'clock': {'value': 2768.0, 'displayValue': "47'"}, 'wallclock': '2026-06-14T18:09:41Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Jamal Musiala'}}, {'athlete': {'displayName': 'Joshua Kimmich'}}], 'text': 'Event: Goal. Team: Germany. Players involved: Jamal Musiala, Joshua Kimmich.'}, {'id': '49508328', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 2949.0, 'displayValue': "50'"}, 'wallclock': '2026-06-14T18:12:41Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Deveron Fonville'}}, {'athlete': {'displayName': 'Leroy Sané'}}], 'text': 'Event: Foul. Team: Curaçao. Players involved: Deveron Fonville, Leroy Sané.'}, {'id': '49508344', 'type': {'id': '106', 'text': 'Shot On Target', 'type': 'shot-on-target'}, 'period': {'number': 2}, 'clock': {'value': 2972.0, 'displayValue': "50'"}, 'wallclock': '2026-06-14T18:13:04Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Felix Nmecha'}}, {'athlete': {'displayName': 'Joshua Kimmich'}}], 'text': 'Event: Shot On Target. Team: Germany. Players involved: Felix Nmecha, Joshua Kimmich.'}, {'id': '49508365', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 3015.0, 'displayValue': "51'"}, 'wallclock': '2026-06-14T18:13:48Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Tahith Chong'}}, {'athlete': {'displayName': 'Nico Schlotterbeck'}}], 'text': 'Event: Foul. Team: Curaçao. Players involved: Tahith Chong, Nico Schlotterbeck.'}, {'id': '49508404', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 3093.0, 'displayValue': "52'"}, 'wallclock': '2026-06-14T18:15:05Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Nathaniel Brown'}}, {'athlete': {'displayName': 'Tahith Chong'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Nathaniel Brown, Tahith Chong.'}, {'id': '49508421', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 3134.0, 'displayValue': "53'"}, 'wallclock': '2026-06-14T18:15:47Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Kai Havertz'}}, {'athlete': {'displayName': 'Leandro Bacuna'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Kai Havertz, Leandro Bacuna.'}, {'id': '49508452', 'type': {'id': '135', 'text': 'Shot Blocked', 'type': 'shot-blocked'}, 'period': {'number': 2}, 'clock': {'value': 3178.0, 'displayValue': "53'"}, 'wallclock': '2026-06-14T18:16:30Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Florian Wirtz'}}, {'athlete': {'displayName': 'Joshua Kimmich'}}], 'text': 'Event: Shot Blocked. Team: Germany. Players involved: Florian Wirtz, Joshua Kimmich.'}, {'id': '49508451', 'type': {'id': '95', 'text': 'Corner Awarded', 'type': 'corner-awarded'}, 'period': {'number': 2}, 'clock': {'value': 3179.0, 'displayValue': "53'"}, 'wallclock': '2026-06-14T18:16:31Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: Corner Awarded. Team: Germany.'}, {'id': '49508489', 'type': {'id': '135', 'text': 'Shot Blocked', 'type': 'shot-blocked'}, 'period': {'number': 2}, 'clock': {'value': 3218.0, 'displayValue': "54'"}, 'wallclock': '2026-06-14T18:17:10Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Kai Havertz'}}, {'athlete': {'displayName': 'Florian Wirtz'}}], 'text': 'Event: Shot Blocked. Team: Germany. Players involved: Kai Havertz, Florian Wirtz.'}, {'id': '49508487', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 2}, 'clock': {'value': 3222.0, 'displayValue': "54'"}, 'wallclock': '2026-06-14T18:17:14Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Aleksandar Pavlovic'}}, {'athlete': {'displayName': 'Florian Wirtz'}}], 'text': 'Event: Shot Off Target. Team: Germany. Players involved: Aleksandar Pavlovic, Florian Wirtz.'}, {'id': '49508501', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 3277.0, 'displayValue': "55'"}, 'wallclock': '2026-06-14T18:18:09Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Jamal Musiala'}}, {'athlete': {'displayName': 'Tahith Chong'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Jamal Musiala, Tahith Chong.'}, {'id': '49508571', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 3382.0, 'displayValue': "57'"}, 'wallclock': '2026-06-14T18:19:54Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Nathaniel Brown'}}, {'athlete': {'displayName': 'Livano Comenencia'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Nathaniel Brown, Livano Comenencia.'}, {'id': '49508668', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 3568.0, 'displayValue': "60'"}, 'wallclock': '2026-06-14T18:23:01Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Nico Schlotterbeck'}}, {'athlete': {'displayName': 'Tahith Chong'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Nico Schlotterbeck, Tahith Chong.'}, {'id': '49508677', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 3601.0, 'displayValue': "61'"}, 'wallclock': '2026-06-14T18:23:33Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Aleksandar Pavlovic'}}, {'athlete': {'displayName': 'Tahith Chong'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Aleksandar Pavlovic, Tahith Chong.'}, {'id': '49508705', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 3643.0, 'displayValue': "61'"}, 'wallclock': '2026-06-14T18:24:15Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Jonathan Tah'}}, {'athlete': {'displayName': 'Jürgen Locadia'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Jonathan Tah, Jürgen Locadia.'}, {'id': '49508710', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 2}, 'clock': {'value': 3712.0, 'displayValue': "62'"}, 'wallclock': '2026-06-14T18:25:24Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Leandro Bacuna'}}, {'athlete': {'displayName': 'Jeremy Antonisse'}}], 'text': 'Event: Shot Off Target. Team: Curaçao. Players involved: Leandro Bacuna, Jeremy Antonisse.'}, {'id': '49508722', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 2}, 'clock': {'value': 3753.0, 'displayValue': "63'"}, 'wallclock': '2026-06-14T18:26:06Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Leroy Sané'}}, {'athlete': {'displayName': 'Jonathan Tah'}}], 'text': 'Event: Shot Off Target. Team: Germany. Players involved: Leroy Sané, Jonathan Tah.'}, {'id': '49508782', 'type': {'id': '76', 'text': 'Substitution', 'type': 'substitution'}, 'period': {'number': 2}, 'clock': {'value': 3823.0, 'displayValue': "64'"}, 'wallclock': '2026-06-14T18:27:16Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Deniz Undav'}}, {'athlete': {'displayName': 'Jamal Musiala'}}], 'text': 'Event: Substitution. Team: Germany. Players involved: Deniz Undav, Jamal Musiala.'}, {'id': '49508785', 'type': {'id': '76', 'text': 'Substitution', 'type': 'substitution'}, 'period': {'number': 2}, 'clock': {'value': 3851.0, 'displayValue': "65'"}, 'wallclock': '2026-06-14T18:27:43Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Jearl Margaritha'}}, {'athlete': {'displayName': 'Jürgen Locadia'}}], 'text': 'Event: Substitution. Team: Curaçao. Players involved: Jearl Margaritha, Jürgen Locadia.'}, {'id': '49508818', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 3923.0, 'displayValue': "66'"}, 'wallclock': '2026-06-14T18:28:55Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Deniz Undav'}}, {'athlete': {'displayName': 'Leandro Bacuna'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Deniz Undav, Leandro Bacuna.'}, {'id': '49508826', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 2}, 'clock': {'value': 3964.0, 'displayValue': "67'"}, 'wallclock': '2026-06-14T18:29:37Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Livano Comenencia'}}], 'text': 'Event: Shot Off Target. Team: Curaçao. Players involved: Livano Comenencia.'}, {'id': '49508825', 'type': {'id': '68', 'text': 'Offside', 'type': 'offside'}, 'period': {'number': 2}, 'clock': {'value': 3966.0, 'displayValue': "67'"}, 'wallclock': '2026-06-14T18:29:38Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Livano Comenencia'}}], 'text': 'Event: Offside. Team: Curaçao. Players involved: Livano Comenencia.'}, {'id': '49508863', 'type': {'id': '173', 'text': 'Goal - Volley', 'type': 'goal---volley'}, 'period': {'number': 2}, 'clock': {'value': 4062.0, 'displayValue': "68'"}, 'wallclock': '2026-06-14T18:31:15Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Nathaniel Brown'}}, {'athlete': {'displayName': 'Deniz Undav'}}], 'text': 'Event: Goal - Volley. Team: Germany. Players involved: Nathaniel Brown, Deniz Undav.'}, {'id': '49508865', 'type': {'id': '129', 'text': 'Start Delay', 'type': 'start-delay'}, 'period': {'number': 2}, 'clock': {'value': 4139.0, 'displayValue': "69'"}, 'wallclock': '2026-06-14T18:32:31Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: Start Delay. Team: Germany. Reported incident: drinks break.'}, {'id': '49508867', 'type': {'id': '130', 'text': 'End Delay', 'type': 'end-delay'}, 'period': {'number': 2}, 'clock': {'value': 4282.0, 'displayValue': "72'"}, 'wallclock': '2026-06-14T18:34:54Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: End Delay. Team: Germany.'}, {'id': '49508870', 'type': {'id': '76', 'text': 'Substitution', 'type': 'substitution'}, 'period': {'number': 2}, 'clock': {'value': 4322.0, 'displayValue': "73'"}, 'wallclock': '2026-06-14T18:35:35Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Antonio Rüdiger'}}, {'athlete': {'displayName': 'Jonathan Tah'}}], 'text': 'Event: Substitution. Team: Germany. Players involved: Antonio Rüdiger, Jonathan Tah.'}, {'id': '49508869', 'type': {'id': '76', 'text': 'Substitution', 'type': 'substitution'}, 'period': {'number': 2}, 'clock': {'value': 4345.0, 'displayValue': "73'"}, 'wallclock': '2026-06-14T18:35:57Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Leon Goretzka'}}, {'athlete': {'displayName': 'Felix Nmecha'}}], 'text': 'Event: Substitution. Team: Germany. Players involved: Leon Goretzka, Felix Nmecha.'}, {'id': '49508876', 'type': {'id': '76', 'text': 'Substitution', 'type': 'substitution'}, 'period': {'number': 2}, 'clock': {'value': 4351.0, 'displayValue': "73'"}, 'wallclock': '2026-06-14T18:36:04Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'David Raum'}}, {'athlete': {'displayName': 'Nathaniel Brown'}}], 'text': 'Event: Substitution. Team: Germany. Players involved: David Raum, Nathaniel Brown.'}, {'id': '49508891', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 4378.0, 'displayValue': "73'"}, 'wallclock': '2026-06-14T18:36:30Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Aleksandar Pavlovic'}}, {'athlete': {'displayName': 'Tahith Chong'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Aleksandar Pavlovic, Tahith Chong.'}, {'id': '49508890', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 4419.0, 'displayValue': "74'"}, 'wallclock': '2026-06-14T18:37:11Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Aleksandar Pavlovic'}}, {'athlete': {'displayName': 'Tahith Chong'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Aleksandar Pavlovic, Tahith Chong.'}, {'id': '49508910', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 2}, 'clock': {'value': 4480.0, 'displayValue': "75'"}, 'wallclock': '2026-06-14T18:38:12Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Tahith Chong'}}, {'athlete': {'displayName': 'Leandro Bacuna'}}], 'text': 'Event: Shot Off Target. Team: Curaçao. Players involved: Tahith Chong, Leandro Bacuna.'}, {'id': '49508920', 'type': {'id': '117', 'text': 'Shot Off Target', 'type': 'shot-off-target'}, 'period': {'number': 2}, 'clock': {'value': 4520.0, 'displayValue': "76'"}, 'wallclock': '2026-06-14T18:38:52Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Jearl Margaritha'}}, {'athlete': {'displayName': 'Livano Comenencia'}}], 'text': 'Event: Shot Off Target. Team: Curaçao. Players involved: Jearl Margaritha, Livano Comenencia.'}, {'id': '49508963', 'type': {'id': '70', 'text': 'Goal', 'type': 'goal'}, 'period': {'number': 2}, 'clock': {'value': 4626.0, 'displayValue': "78'"}, 'wallclock': '2026-06-14T18:40:39Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Deniz Undav'}}, {'athlete': {'displayName': 'Joshua Kimmich'}}], 'text': 'Event: Goal. Team: Germany. Players involved: Deniz Undav, Joshua Kimmich.'}, {'id': '49509033', 'type': {'id': '95', 'text': 'Corner Awarded', 'type': 'corner-awarded'}, 'period': {'number': 2}, 'clock': {'value': 4852.0, 'displayValue': "81'"}, 'wallclock': '2026-06-14T18:44:24Z', 'team': {'displayName': 'Germany'}, 'text': 'Event: Corner Awarded. Team: Germany.'}, {'id': '49509035', 'type': {'id': '76', 'text': 'Substitution', 'type': 'substitution'}, 'period': {'number': 2}, 'clock': {'value': 4930.0, 'displayValue': "83'"}, 'wallclock': '2026-06-14T18:45:43Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Gervane Kastaneer'}}, {'athlete': {'displayName': 'Tahith Chong'}}], 'text': 'Event: Substitution. Team: Curaçao. Players involved: Gervane Kastaneer, Tahith Chong.'}, {'id': '49509037', 'type': {'id': '76', 'text': 'Substitution', 'type': 'substitution'}, 'period': {'number': 2}, 'clock': {'value': 4953.0, 'displayValue': "83'"}, 'wallclock': '2026-06-14T18:46:05Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Waldemar Anton'}}, {'athlete': {'displayName': 'Joshua Kimmich'}}], 'text': 'Event: Substitution. Team: Germany. Players involved: Waldemar Anton, Joshua Kimmich.'}, {'id': '49509047', 'type': {'id': '122', 'text': 'Handball', 'type': 'handball'}, 'period': {'number': 2}, 'clock': {'value': 4973.0, 'displayValue': "83'"}, 'wallclock': '2026-06-14T18:46:25Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Gervane Kastaneer'}}], 'text': 'Event: Handball. Team: Curaçao. Players involved: Gervane Kastaneer.'}, {'id': '49509062', 'type': {'id': '95', 'text': 'Corner Awarded', 'type': 'corner-awarded'}, 'period': {'number': 2}, 'clock': {'value': 5025.0, 'displayValue': "84'"}, 'wallclock': '2026-06-14T18:47:17Z', 'team': {'displayName': 'Curaçao'}, 'text': 'Event: Corner Awarded. Team: Curaçao.'}, {'id': '49509081', 'type': {'id': '106', 'text': 'Shot On Target', 'type': 'shot-on-target'}, 'period': {'number': 2}, 'clock': {'value': 5103.0, 'displayValue': "86'"}, 'wallclock': '2026-06-14T18:48:35Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'David Raum'}}, {'athlete': {'displayName': 'Deniz Undav'}}], 'text': 'Event: Shot On Target. Team: Germany. Players involved: David Raum, Deniz Undav.'}, {'id': '49509086', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 5129.0, 'displayValue': "86'"}, 'wallclock': '2026-06-14T18:49:01Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Aleksandar Pavlovic'}}, {'athlete': {'displayName': 'Juninho Bacuna'}}], 'text': 'Event: Foul. Team: Germany. Players involved: Aleksandar Pavlovic, Juninho Bacuna.'}, {'id': '49509096', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 5166.0, 'displayValue': "87'"}, 'wallclock': '2026-06-14T18:49:38Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Sherel Floranus'}}, {'athlete': {'displayName': 'Manuel Neuer'}}], 'text': 'Event: Foul. Team: Curaçao. Players involved: Sherel Floranus, Manuel Neuer.'}, {'id': '49509123', 'type': {'id': '70', 'text': 'Goal', 'type': 'goal'}, 'period': {'number': 2}, 'clock': {'value': 5263.0, 'displayValue': "88'"}, 'wallclock': '2026-06-14T18:51:15Z', 'team': {'displayName': 'Germany'}, 'participants': [{'athlete': {'displayName': 'Kai Havertz'}}, {'athlete': {'displayName': 'Deniz Undav'}}], 'text': 'Event: Goal. Team: Germany. Players involved: Kai Havertz, Deniz Undav.'}, {'id': '49509252', 'type': {'id': '66', 'text': 'Foul', 'type': 'foul'}, 'period': {'number': 2}, 'clock': {'value': 5400.0, 'displayValue': "90'+4'"}, 'wallclock': '2026-06-14T18:57:06Z', 'team': {'displayName': 'Curaçao'}, 'participants': [{'athlete': {'displayName': 'Livano Comenencia'}}, {'athlete': {'displayName': 'Aleksandar Pavlovic'}}], 'text': 'Event: Foul. Team: Curaçao. Players involved: Livano Comenencia, Aleksandar Pavlovic.'}, {'id': '49506808', 'type': {'id': '80', 'text': 'Kickoff', 'type': 'kickoff'}, 'period': {'number': 1}, 'clock': {'value': 0.0, 'displayValue': ''}, 'wallclock': '2026-06-14T17:01:49Z', 'text': 'Event: Kickoff.'}, {'id': '49507427', 'type': {'id': '129', 'text': 'Start Delay', 'type': 'start-delay'}, 'period': {'number': 1}, 'clock': {'value': 1361.0, 'displayValue': "23'"}, 'wallclock': '2026-06-14T17:24:31Z', 'team': {'id': '11678', 'displayName': 'Curaçao'}, 'text': 'Event: Start Delay. Team: Curaçao.'}, {'id': '49507431', 'type': {'id': '130', 'text': 'End Delay', 'type': 'end-delay'}, 'period': {'number': 1}, 'clock': {'value': 1483.0, 'displayValue': "25'"}, 'wallclock': '2026-06-14T17:26:33Z', 'team': {'id': '11678', 'displayName': 'Curaçao'}, 'text': 'Event: End Delay. Team: Curaçao.'}, {'id': '49507970', 'type': {'id': '81', 'text': 'Halftime', 'type': 'halftime'}, 'period': {'number': 1}, 'clock': {'value': 2700.0, 'displayValue': "45'+6'"}, 'wallclock': '2026-06-14T17:52:02Z', 'text': 'Event: Halftime.'}, {'id': '49508206', 'type': {'id': '82', 'text': 'Start 2nd Half', 'type': 'start-2nd-half'}, 'period': {'number': 2}, 'clock': {'value': 2700.0, 'displayValue': "45'"}, 'wallclock': '2026-06-14T18:08:32Z', 'text': 'Event: Start 2nd Half.'}, {'id': '49508866', 'type': {'id': '129', 'text': 'Start Delay', 'type': 'start-delay'}, 'period': {'number': 2}, 'clock': {'value': 4139.0, 'displayValue': "69'"}, 'wallclock': '2026-06-14T18:32:32Z', 'team': {'id': '11678', 'displayName': 'Curaçao'}, 'text': 'Event: Start Delay. Team: Curaçao.'}, {'id': '49508868', 'type': {'id': '130', 'text': 'End Delay', 'type': 'end-delay'}, 'period': {'number': 2}, 'clock': {'value': 4282.0, 'displayValue': "72'"}, 'wallclock': '2026-06-14T18:34:54Z', 'team': {'id': '11678', 'displayName': 'Curaçao'}, 'text': 'Event: End Delay. Team: Curaçao.'}, {'id': '49509261', 'type': {'id': '83', 'text': 'End Regular Time', 'type': 'end-regular-time'}, 'period': {'number': 2}, 'clock': {'value': 5400.0, 'displayValue': "90'+6'"}, 'wallclock': '2026-06-14T18:58:53Z', 'text': 'Event: End Regular Time.'}], 'source_provenance': {'provider': 'ESPN', 'source_url': 'https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event=760422', 'source_body_sha256': 'a3e1ddfdb1c5580bd898d32111331dde8776491060cce00b7f51cb93bc570c0e', 'captured_at_utc': '2026-09-24T20:27:43.578134+00:00', 'transformation': 'Structured event type, team, participants, period, clock, and wallclock retained; text rendered from these facts. No article bodies or original narrative commentary included.', 'scope': 'Unique structured events returned in commentary and keyEvents; this is a saved historical capture.'}}



def automatic_espn_file(cache, event_id, league):
    """Materialize saved data automatically. Never download Python source."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', league):
        raise ValueError('Invalid ESPN league')
    if event_id is None:
        return None
    if not str(event_id).isdigit():
        raise ValueError('ESPN event ID must be numeric')
    path = cache / 'espn_snapshots' / f'{league}_{event_id}.json.gz'
    if path.exists():
        json.loads(gzip.decompress(path.read_bytes()))
        return path
    if league == 'fifa.world' and str(event_id) == '760422':
        body = gzip.compress(json.dumps(BUNDLED_ESPN, ensure_ascii=False).encode(), mtime=0)
    else:
        url = REPOSITORY_RAW + f'/data_sources/espn/{league}_{event_id}.json.gz'
        try:
            with urlopen(Request(url, headers={'User-Agent': 'PolyWorldCupResearch/1.0'}), timeout=30) as response:
                body = response.read()
            json.loads(gzip.decompress(body))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            print(f'No usable saved ESPN snapshot ({error}); trying live ESPN.', flush=True)
            return None
    atomic_write(path, body)
    return path


#!/usr/bin/env python3
"""Export one market as per-actor interval / execution rows with ESPN text.

Usage: python3 build_actor_dataset.py 1897059 --out data/market_1897059

The two records for each distinct execution time are:
  (previous execution, current execution): news in the open interval, NO_TRADE
  current execution: the same news, and all captured executions at that time.

These are retrospective descriptions of observed gaps, not prospective
trade-occurrence targets. They never assert absence outside the supplied feed.
Only Python 3.11+ and its standard library are needed.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timezone
import gzip
from itertools import groupby
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
from urllib.error import HTTPError

ROOT = Path.cwd()



def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_events(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(compact(row) + "\n")


def key_event(event):
    """Optional text/type filter. The default retains every timed commentary."""
    value = (str(event.get("kind", "")) + " " + event["text"]).casefold()
    return bool(re.search(r"\b(goal|yellow|red card|second yellow|substitution|substitute|"
                          r"penalty|penalties|var|injur\w*|half[- ]?time|full[- ]?time|"
                          r"kick[- ]?off|disallow\w*|lineup|line-up)\b", value))


def news_feature(event):
    # The feature contains the actual text. IDs/URLs are kept in the shared
    # source files, never substituted for the news supplied to the model.
    # Match clocks and per-event timing provenance also remain in those files.
    return {"time": event["time_utc"], "type": event.get("kind"), "text": event["text"]}


def stage_trades(db, iterator):
    """Disk-backed sort: do not keep the full market or its wallets in RAM."""
    db.execute("CREATE TABLE trades (ordinal INTEGER PRIMARY KEY, actor TEXT, instant INTEGER, payload TEXT)")
    batch, count = [], 0
    for row in iterator:
        batch.append((row["actor_id"], row["time_us"], compact(row)))
        count += 1
        if len(batch) >= 10000:
            db.executemany("INSERT INTO trades(actor,instant,payload) VALUES (?,?,?)", batch)
            batch.clear()
    if batch:
        db.executemany("INSERT INTO trades(actor,instant,payload) VALUES (?,?,?)", batch)
    db.execute("CREATE INDEX actor_time ON trades(actor,instant,ordinal)")
    db.execute("CREATE TABLE actor_counts AS SELECT actor,COUNT(*) n FROM trades GROUP BY actor")
    db.execute("CREATE UNIQUE INDEX actor_count_id ON actor_counts(actor)")
    db.commit()
    return count


def actor_records(actor, trades, market, events, event_times, origin):
    """Yield exactly two records per distinct execution timestamp.

    News uses strict lower < news time < current execution time. Equal-time
    executions are grouped rather than assigned an invented internal order.
    A gap's news is deliberately repeated on the following trade row, matching
    the requested schema. No history is expanded again on subsequent rows.
    """
    previous = origin
    for index, (instant, executions) in enumerate(groupby(trades, lambda row: row["time_us"]), 1):
        lo = 0 if previous is None else bisect_right(event_times, previous)
        hi = bisect_left(event_times, instant)
        news = [news_feature(event) for event in events[lo:hi]]
        interval = {"start": utc_time(previous) if previous is not None else None,
                    "end": utc_time(instant), "start_inclusive": False, "end_inclusive": False}
        base = {"actor_id": actor, "market_id": market["market_id"], "condition_id": market["condition_id"]}
        yield {**base, "record_type": "interval", "row_index": 2 * index - 2,
               "interval": interval, "news": news, "label": {"action": "NO_TRADE"}}
        values = list(executions)
        yield {**base, "record_type": "trade", "row_index": 2 * index - 1,
               "timestamp": utc_time(instant), "context_interval": interval,
               "news": news, "label": {"action": "TRADE", "trades": [
                   {"time": value["time"], **value["trade"]} for value in values]}}
        previous = instant


def export(args):
    if args.max_trades_per_actor < 0:
        raise ValueError("--max-trades-per-actor must be nonnegative; 0 includes all actors")
    if args.trade_capture and (args.trades_file or args.sqlite):
        raise ValueError("Choose --trade-capture, --trades-file, or --sqlite, not several sources")
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", args.market_id)
    output = (args.out or ROOT / "data" / ("market_" + key)).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new --out directory.")
    # This exporter always creates a separate dataset. Never write inside a
    # previous prepared release, even if a new subdirectory was requested.
    for parent in (output, *output.parents):
        if (parent / "manifest.json").exists():
            raise ValueError("Choose an output directory outside existing dataset releases")
    cache = args.cache.resolve()
    if cache == output or cache.is_relative_to(output):
        raise ValueError("Keep --cache outside the new --out directory")
    client = HttpClient(cache / "http", compress=True)
    print(f"Resolving market {args.market_id}", flush=True)
    market = resolve_market(args.market_id, client=client, metadata_file=args.market_metadata)
    event_id = args.espn_event_id or market.get("espn_event_id")
    fixture_date = args.date or market.get("fixture_date")
    teams = args.teams or market.get("team_names")
    if not event_id and not args.espn_file and fixture_date and teams:
        event_id, _, _ = discover_espn_event(client, fixture_date=fixture_date, teams=teams, league=args.league)
    if not args.espn_file and not args.include_core_plays:
        snapshot = automatic_espn_file(cache, event_id, args.league)
        if snapshot:
            args.espn_file = [snapshot]
            print(f'Using saved ESPN events: {snapshot}', flush=True)
    print("Loading ESPN commentary once for this match", flush=True)
    context_options = dict(event_id=event_id, league=args.league,
        fixture_date=fixture_date, teams=teams, espn_files=args.espn_file, time_map=args.time_map,
        time_policy="provider", allow_clock_estimates=args.allow_clock_estimates,
        include_core_plays=args.include_core_plays)
    try:
        context = collect_espn_context(client, **context_options)
    except HTTPError as error:
        # Public ESPN availability can differ between networks. A previously
        # captured factual event file needs no live ESPN request or credentials.
        safe_id = str(event_id) if str(event_id).isdigit() else "unknown"
        snapshot = ROOT / "data_sources" / "espn" / f"{args.league}_{safe_id}.json.gz"
        if error.code == 403 and not args.espn_file and not args.include_core_plays and snapshot.is_file():
            print(f"ESPN returned HTTP 403; using saved match events: {snapshot}", flush=True)
            context_options.update(espn_files=[snapshot], include_core_plays=False)
            context = collect_espn_context(client, **context_options)
        else:
            raise ValueError(f"ESPN returned HTTP {error.code} for {error.url}. "
                "Use --espn-file PATH with a saved ESPN summary or event JSON/JSONL file. "
                "No trade collection or actor export has started.") from error
    events = [event for event in context["timed_events"] if not args.key_events_only or key_event(event)]
    events.sort(key=lambda event: (event["timestamp_us"], event["news_id"]))
    event_times = [event["timestamp_us"] for event in events]
    # Do not silently return an apparently news-enriched dataset whose entire
    # commentary could not be assigned to any real-world time interval.
    if not events:
        diagnostic = cache / "espn_unplaced" / (str(context["event_id"]) + ".json")
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        write_json(diagnostic, context)
        raise ValueError(f"No timestamped ESPN text is available for this selection. Saved source details: {diagnostic}. "
                         "Supply --espn-file or --time-map, or use --allow-clock-estimates with period anchors.")
    print(f"Loaded {len(events):,} timed ESPN items; {len(context['untimed_events']):,} items have no assignable UTC time", flush=True)
    iterator, source_report = load_market_trades(market, cache_dir=cache, client=client,
        trades_file=args.trades_file, sqlite_path=args.sqlite, capture_dir=args.trade_capture)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build in a sibling directory, then publish it with one rename. An error
    # leaves the existing datasets untouched, while the HTTP/trade cache stays.
    work = Path(tempfile.mkdtemp(prefix="market-actor-build-", dir=output.parent))
    try:
        db_path = work / "sort.sqlite"
        db = sqlite3.connect(db_path)
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            print("Reading captured trades and sorting actors on disk", flush=True)
            observations = stage_trades(db, iterator)
            if not observations:
                raise ValueError("No captured trades found for this market")
            minimum = db.execute("SELECT MIN(instant) FROM trades").fetchone()[0]
            origin = timestamp_us(args.start) if args.start else (
                timestamp_us(market["market_open_utc"]) if market.get("market_open_utc") else None)
            origin_basis = "user_supplied" if args.start else (market.get("market_open_basis") or "unbounded_source_history")
            if not args.start and origin is not None and origin > minimum:
                # A current opening field may refer to a reopening. Keep all
                # historical executions instead of silently cutting them away.
                origin, origin_basis = None, "opening_later_than_first_observation_unbounded"
            threshold = args.max_trades_per_actor
            query = """SELECT t.actor,t.payload FROM trades t JOIN actor_counts c ON c.actor=t.actor
                WHERE (?=0 OR c.n<=?) AND (? IS NULL OR t.instant>=?)
                ORDER BY t.actor,t.instant,t.ordinal"""
            rows = db.execute(query, (threshold, threshold, origin, origin))
            actors_dir = work / "actors"
            actors_dir.mkdir()
            counts = Counter()
            with (work / "actor_index.jsonl").open("w", encoding="utf-8") as index_stream:
                for actor, actor_rows in groupby(rows, lambda item: item[0]):
                    filename = actor + (".jsonl.gz" if args.gzip else ".jsonl")
                    path = actors_dir / filename
                    opener = gzip.open if args.gzip else open
                    actor_count = Counter()
                    trades = (json.loads(item[1]) for item in actor_rows)
                    with opener(path, "wt", encoding="utf-8") as stream:
                        for row in actor_records(actor, trades, market, events, event_times, origin):
                            stream.write(compact(row) + "\n")
                            actor_count["rows"] += 1
                            actor_count["news_entries"] += len(row["news"])
                            if row["record_type"] == "trade":
                                actor_count["distinct_trade_times"] += 1
                                actor_count["trade_observations"] += len(row["label"]["trades"])
                    index_stream.write(compact({"actor_id": actor, "path": "actors/" + filename, **dict(actor_count)}) + "\n")
                    counts.update(actor_count)
                    counts["actors"] += 1
                    if counts["actors"] % 1000 == 0:
                        print(f"Wrote {counts['actors']:,} actors / {counts['rows']:,} rows", flush=True)
            all_actors = db.execute("SELECT COUNT(*) FROM actor_counts").fetchone()[0]
            excluded = db.execute("SELECT COUNT(*) FROM actor_counts WHERE ?!=0 AND n>?", (threshold, threshold)).fetchone()[0]
        finally:
            db.close()
        db_path.unlink()
        write_json(work / "market.json", market)
        write_events(work / "espn_events.jsonl", context["timed_events"])
        write_events(work / "espn_unplaced_events.jsonl", context["untimed_events"])
        write_json(work / "espn_sources.json", {key: value for key, value in context.items()
                   if key not in {"timed_events", "untimed_events"}})
        manifest = {"format": "actor_market_intervals_v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "market_id": market["market_id"], "condition_id": market["condition_id"],
            "espn_event_id": context["event_id"], "counts": dict(counts),
            "source_trade_observations": observations, "source_actors": all_actors,
            "actors_excluded_above_trade_limit": excluded,
            "max_trades_per_actor": threshold or None,
            "filter_scope": "all_captured_observations_for_actor_in_this_binary_market_before_applying_start_cutoff",
            "origin_utc": utc_time(origin) if origin is not None else None, "origin_basis": origin_basis,
            "espn_timed_items": len(events), "espn_unplaced_items": len(context["untimed_events"]),
            "key_events_only": args.key_events_only, "source": source_report,
            "record_order": ["open_interval_before_trade", "trade_timestamp"],
            "equal_time_trades": "one_trade_row_containing_all_same_time_observations",
            "news_boundary_rule": "start < news_timestamp < trade_timestamp; equal-time news excluded",
            "news_repeated_on_adjacent_trade_row": True,
            "trailing_interval_after_last_trade": False,
            "no_trade_semantics": "zero_captured_observations_in_this_actor_market_open_interval",
            "task_semantics": "retrospective_gap_description; future_execution_defines_interval_end",
            "prospective_trade_occurrence_training_ready": False,
            "news_feature_fields": ["time", "type", "text"],
            "news_source_metadata": "espn_events.jsonl",
            "timestamp_semantics": "ESPN provider wallclock is an occurrence proxy; per-event timing bases and explicit estimates are recorded in espn_events.jsonl",
            "complete_historical_espn_news_archive": False,
            "previous_datasets_modified": False}
        write_json(work / "manifest.json", manifest)
        if output.exists():
            raise ValueError("The output directory appeared during export; choose another --out")
        work.rename(output)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), **dict(counts)}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate per-actor trade and NO_TRADE rows. Example: python3 build_actor_dataset.py 1897059", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("market_id", help="Polymarket numeric market ID, condition ID, or binary-market slug")
    parser.add_argument("--out", type=Path, help="New output directory, never an existing dataset")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "market_actor_cache")
    parser.add_argument("--espn-event-id", help="Override automatic match lookup")
    parser.add_argument("--league", default="fifa.world", help="ESPN league slug, e.g. fifa.world or uefa.champions")
    parser.add_argument("--date", help="Fixture date YYYY-MM-DD, if absent from market metadata")
    parser.add_argument("--teams", nargs=2, help="Two ESPN team names, if absent from market metadata")
    parser.add_argument("--max-trades-per-actor", type=int, default=20, help="Inclusive maximum in this market; 0 keeps every actor")
    parser.add_argument("--key-events-only", action="store_true", help="Keep goal/card/substitution/penalty/VAR/injury/match-phase text")
    parser.add_argument("--include-core-plays", action="store_true", help="Also fetch paginated granular ESPN plays, including routine play")
    parser.add_argument("--gzip", action="store_true", help="Compress each actor's JSONL file")
    parser.add_argument("--start", help="Origin of the first interval (zoned ISO time or epoch seconds)")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--trades-file", type=Path, help="Existing market trades CSV/JSON/JSONL, optionally gzip")
    inputs.add_argument("--sqlite", type=Path, help="Existing repository trades SQLite, read-only")
    parser.add_argument("--trade-capture", type=Path, help="Directory of existing resumable v2 condition captures")
    parser.add_argument("--market-metadata", type=Path, help="Saved Gamma market object for offline lookup")
    parser.add_argument("--espn-file", type=Path, action="append", help="Saved ESPN summary/core-play JSON or normalized JSONL; repeatable")
    parser.add_argument("--time-map", type=Path, help="Explicit event UTC timestamps and/or same-period clock anchors")
    parser.add_argument("--allow-clock-estimates", action="store_true", help="Explicitly permit approximate same-period anchor timings")
    args = parser.parse_args()
    try:
        export(args)
    except (ValueError, OSError, KeyError, sqlite3.Error, InvalidOperation) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()

