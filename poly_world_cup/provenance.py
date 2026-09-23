"""Read-only release checks binding normalized trade pages to raw captures.

This deliberately complements the normalized-only resume/audit verifier. It
checks local evidence, not the provider's completeness or historical truth.
Mutable request indexes are hints: an older immutable capture remains valid
after a refresh replaces its request index.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Callable, Iterable
import zlib

from .http import request_url
from .trades import (
    API_URL, HEX_32, _json_bytes, _validated_page, validate_collection,
)

SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceError(ValueError):
    """A committed page cannot be bound to intact local source evidence."""


def _safe_path(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ProvenanceError("Referenced file escapes its data directory")
    return resolved


def _capture_identity(metadata: dict) -> tuple:
    return (metadata.get("url"), metadata.get("body_sha256"), metadata.get("retrieved_at"))


class _CaptureStore:
    def __init__(self, root: Path):
        self.root = root
        self.captures: dict[tuple, list[Path]] | None = None

    def _checked_capture(self, path: Path, identity: tuple) -> dict:
        metadata = json.loads(_safe_path(self.root, path).read_text())
        if not isinstance(metadata, dict) or _capture_identity(metadata) != identity:
            raise ProvenanceError("Raw capture identity does not match committed page")
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        if path.name != f"{digest}.json":
            raise ProvenanceError("Immutable capture metadata checksum mismatch")
        if metadata.get("body_compression") not in (None, "gzip"):
            raise ProvenanceError("Unknown raw body compression")
        return metadata

    def find(self, identity: tuple) -> dict:
        url, body_hash, retrieved_at = identity
        if not isinstance(body_hash, str) or not SHA256.fullmatch(body_hash):
            raise ProvenanceError("Invalid raw body SHA-256")
        if not isinstance(retrieved_at, str) or not retrieved_at:
            raise ProvenanceError("Missing capture retrieval time")
        # Normal collection takes this direct path without scanning the cache.
        key = hashlib.sha256(url.encode()).hexdigest()
        index = self.root / "requests" / f"{key}.json"
        try:
            metadata = json.loads(_safe_path(self.root, index).read_text())
            if isinstance(metadata, dict) and _capture_identity(metadata) == identity:
                digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
                return self._checked_capture(self.root / "captures" / f"{digest}.json", identity)
        except (OSError, ValueError, TypeError):
            # A missing/replaced request index is not loss of the immutable
            # historical capture. Search it below and verify the record itself.
            pass
        if self.captures is None:
            self.captures = {}
            captures_dir = _safe_path(self.root, self.root / "captures")
            for path in captures_dir.glob("*.json"):
                try:
                    metadata = json.loads(_safe_path(self.root, path).read_text())
                    if isinstance(metadata, dict):
                        self.captures.setdefault(_capture_identity(metadata), []).append(path)
                except (OSError, ValueError, TypeError):
                    continue
        for path in self.captures.get(identity, []):
            try:
                return self._checked_capture(path, identity)
            except (OSError, ValueError, TypeError):
                continue
        raise ProvenanceError("No intact immutable capture matches the committed page")

    def body(self, metadata: dict) -> bytes:
        suffix = ".json.gz" if metadata.get("body_compression") == "gzip" else ".json"
        path = self.root / "bodies" / f"{metadata['body_sha256']}{suffix}"
        body = _safe_path(self.root, path).read_bytes()
        if metadata.get("body_compression") == "gzip":
            body = gzip.decompress(body)
        if hashlib.sha256(body).hexdigest() != metadata["body_sha256"]:
            raise ProvenanceError("Raw response body checksum mismatch")
        return body


def verify_raw_provenance(trades_root: Path, cache_root: Path, *,
                          condition_ids: Iterable[str] | None = None,
                          cache_layout: str = "per_condition",
                          on_progress: Callable[[dict], None] | None = None) -> dict:
    """Verify every committed page; return a compact, JSON-serializable report.

    ``per_condition`` matches collect-tournament's cache/<condition> layout;
    ``shared`` matches ingest's shared cache. Explicit condition_ids are
    recommended for releases: discovery alone cannot detect missing conditions.
    No network calls or writes occur, including on missing or corrupt evidence.
    A paused collection can have valid provenance, so traversal counts and
    completeness flags remain separate from the provenance pass/fail result.
    """
    if cache_layout not in {"per_condition", "shared"}:
        raise ValueError("cache_layout must be per_condition or shared")
    trades_root, cache_root = Path(trades_root).resolve(), Path(cache_root).resolve()
    discovered = condition_ids is None
    if condition_ids is None:
        condition_ids = [path.parent.name for path in trades_root.glob("*/manifest.json")]
    if isinstance(condition_ids, (str, bytes)):
        raise ValueError("condition_ids must be an iterable of condition IDs")
    supplied = list(condition_ids)
    if not supplied or any(not isinstance(c, str) or not HEX_32.fullmatch(c) for c in supplied):
        raise ValueError("At least one valid condition ID is required")
    conditions = sorted({condition.lower() for condition in supplied})
    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cache_layout": cache_layout,
        "condition_selection": "discovered_manifests" if discovered else "explicit",
        "requested_condition_count": len(conditions),
        "verified_condition_count": 0,
        "normalized_integrity_verified_conditions": 0,
        "api_exhausted_conditions": 0,
        "committed_page_count": 0,
        "verified_raw_page_count": 0,
        "verified_observation_count": 0,
        "errors": [],
        "raw_provenance_verified": False,
        "source_completeness_verified": False,
        "training_coverage_certified": False,
        "integrity_scope": "Committed normalized pages, matching immutable capture metadata, exact query/cursor, raw response hashes, source pagination, and schema-1 normalization replay. Unreferenced cache entries and upstream completeness are outside this check.",
    }
    stores: dict[Path, _CaptureStore] = {}
    for condition in conditions:
        error_count = len(report["errors"])
        try:
            directory = _safe_path(trades_root, trades_root / condition)
            _safe_path(trades_root, directory / "manifest.json")
            # Guard normalized paths before the standard verifier opens them.
            preview = json.loads((directory / "manifest.json").read_text())
            for page in preview["pages"]:
                _safe_path(directory, directory / page["file"])
            state = validate_collection(trades_root, condition_id=condition)
            report["normalized_integrity_verified_conditions"] += 1
            report["api_exhausted_conditions"] += state["api_traversal_status"] == "exhausted"
            report["committed_page_count"] += state["page_count"]
            cache_dir = (_safe_path(cache_root, cache_root / condition)
                         if cache_layout == "per_condition" else cache_root)
            store = stores.setdefault(cache_dir, _CaptureStore(cache_dir))
        except (OSError, ValueError, TypeError, KeyError, EOFError, zlib.error) as error:
            report["errors"].append({"condition_id": condition, "page_index": None,
                                      "error": str(error)[:1000]})
            if on_progress:
                on_progress(dict(report))
            continue
        seen_cursors = []
        for page in state["pages"]:
            try:
                parameters = dict(state["parameters"])
                if page["requested_cursor"] is not None:
                    parameters["cursor"] = page["requested_cursor"]
                expected_url = request_url(API_URL, parameters)
                if page["request_url"] != expected_url:
                    raise ProvenanceError("Committed request URL does not match query and cursor")
                identity = (expected_url, page["body_sha256"], page["retrieved_at"])
                capture = store.find(identity)
                body = store.body(capture)
                result = SimpleNamespace(
                    data=json.loads(body, parse_float=Decimal), url=expected_url,
                    body_sha256=page["body_sha256"], retrieved_at=page["retrieved_at"],
                )
                rows, next_cursor, has_more = _validated_page(
                    result, condition, page["requested_cursor"], seen_cursors,
                )
                if (next_cursor != page["next_cursor"] or has_more != page["has_more"]
                        or len(rows) != page["row_count"]):
                    raise ProvenanceError("Raw response pagination or row count disagrees with committed page")
                replay_hash = hashlib.sha256()
                for row in rows:
                    replay_hash.update(_json_bytes(row))
                if replay_hash.hexdigest() != page["normalized_sha256"]:
                    raise ProvenanceError("Normalized page differs from raw-source normalization replay")
                report["verified_raw_page_count"] += 1
                report["verified_observation_count"] += len(rows)
            except (OSError, ValueError, TypeError, KeyError, EOFError, zlib.error) as error:
                report["errors"].append({"condition_id": condition, "page_index": page["page_index"],
                                          "error": str(error)[:1000]})
            seen_cursors.append(page["requested_cursor"])
        if len(report["errors"]) == error_count:
            report["verified_condition_count"] += 1
        if on_progress:
            on_progress(dict(report))
    report["raw_provenance_verified"] = not report["errors"]
    return report
