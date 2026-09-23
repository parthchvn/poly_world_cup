"""Resumable collection of *observations* from Polymarket's v2 trade API.

This API does not expose a canonical log/order identifier, omits subthreshold
trades, and serves a bounded history. Exhausting it never certifies a complete
training label window. A row is an API observation, not a reconstructed order.
"""
from __future__ import annotations

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
