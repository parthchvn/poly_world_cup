"""Concurrent, resumable tournament trade collection with durable progress.

API exhaustion means the filtered API traversal finished. It never establishes
complete historical market coverage. Progress counters for running/failed jobs
are journal observations; only successful returns count as validated here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Callable, Any, Iterable
import uuid

from .http import HttpClient
from .io import write_json
from .trades import HEX_32, ingest_condition


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RequestPacer:
    """Space logical HTTP call starts globally across this batch's workers.

    HttpClient performs bounded retries with its own backoff; those retries are
    internal to one logical call. Cache hits are conservatively paced as well.
    """
    def __init__(self, requests_per_second: float):
        if (isinstance(requests_per_second, bool)
                or not isinstance(requests_per_second, (int, float))
                or not math.isfinite(requests_per_second)
                or requests_per_second <= 0):
            raise ValueError("requests_per_second must be finite and positive")
        self.interval = 1.0 / requests_per_second
        self.next_start = 0.0
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)


class _PacedClient:
    def __init__(self, client: Any, pacer: RequestPacer):
        self.client = client
        self.pacer = pacer

    def get_json(self, url: str, params: dict | None = None):
        self.pacer.acquire()
        return self.client.get_json(url, params=params)


@contextmanager
def _batch_lock(output_dir: Path):
    import fcntl

    with (output_dir / ".batch.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another batch is writing this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _counts(state: dict) -> dict:
    """Small progress projection, deliberately omitting the growing page list."""
    return {key: state.get(key) for key in (
        "api_traversal_status", "page_count", "row_count",
        "earliest_block_timestamp", "latest_block_timestamp",
    )}


def run_batch(condition_ids: Iterable[str], *, output_dir: Path, cache_dir: Path,
              progress_path: Path | None = None, workers: int = 12,
              requests_per_second: float = 4.0, limit: int = 1000,
              max_pages_per_condition: int | None = None, compress: bool = True,
              minimum_size: str = "0.01",
              progress_interval: float = 5.0,
              client_factory: Callable[[Path], Any] | None = None,
              on_progress: Callable[[dict], None] | None = None) -> dict:
    """Collect every selected condition; one failure does not stop other jobs.

    ``None`` for max_pages_per_condition traverses every condition to API
    exhaustion. Existing manifests are always passed through ingest_condition's
    integrity verifier; old progress files never decide whether to skip a job.
    Each condition gets its own HttpClient cache. Injected client_factory takes
    that cache Path and must return an object supporting get_json.

    The atomic progress file is updated during long jobs and on every terminal
    job result. In-progress counters come from atomically committed manifests,
    without rehashing the entire collection on every heartbeat. After crashes,
    per-condition manifests/cursor journals (not this report) govern recovery.
    """
    if isinstance(condition_ids, (str, bytes)):
        raise ValueError("condition_ids must be an iterable of condition IDs")
    supplied = list(condition_ids)
    if not supplied or any(not isinstance(c, str) or not HEX_32.fullmatch(c) for c in supplied):
        raise ValueError("At least one valid 32-byte condition ID is required")
    conditions = sorted({c.lower() for c in supplied})
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("workers must be an integer between 1 and 64")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000")
    if max_pages_per_condition is not None and (type(max_pages_per_condition) is not int
                                               or max_pages_per_condition < 1):
        raise ValueError("max_pages_per_condition must be positive or None")
    if (isinstance(progress_interval, bool) or not isinstance(progress_interval, (float, int))
            or not math.isfinite(progress_interval) or progress_interval <= 0):
        raise ValueError("progress_interval must be finite and positive")
    pacer = RequestPacer(requests_per_second)
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    progress_path = Path(progress_path) if progress_path is not None else output_dir / "batch_progress.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = {c: {"condition_id": c, "status": "queued", "page_count": 0,
                    "row_count": 0, "integrity_validated": False} for c in conditions}
    start = _now()
    report = {
        "schema_version": 1, "run_id": uuid.uuid4().hex,
        "condition_set_sha256": hashlib.sha256("\n".join(conditions).encode()).hexdigest(),
        "started_at": start, "updated_at": start, "finished_at": None,
        "status": "running", "workers": workers,
        "logical_requests_per_second": requests_per_second,
        "page_limit": limit, "max_new_pages_per_condition": max_pages_per_condition,
        "requested_minimum_size_tokens": minimum_size,
        "compress": compress, "training_coverage_certified": False,
        "canonical_fill_identity_available": False,
        "count_semantics": "API observations, not unique canonical fills; running/failed job counters are unverified journal snapshots",
    }
    state_lock = threading.Lock()

    def snapshot(*, final: bool = False) -> dict:
        with state_lock:
            for condition, status in statuses.items():
                if status["status"] not in ("running", "failed"):
                    continue
                try:
                    state = json.loads((output_dir / condition / "manifest.json").read_text())
                    if state.get("condition_id") != condition:
                        continue
                    counts = _counts(state)
                    if any(type(counts[k]) is not int or counts[k] < 0 for k in ("page_count", "row_count")):
                        continue
                    status.update(counts)
                except (OSError, ValueError, TypeError, AttributeError):
                    # The final/resume integrity verifier decides correctness.
                    pass
            rows = [dict(statuses[c]) for c in conditions]
        result = dict(report)
        result.update({
            "updated_at": _now(), "conditions": rows,
            "condition_count": len(rows),
            "status_counts": {status: sum(r["status"] == status for r in rows)
                              for status in ("queued", "running", "exhausted", "paused", "failed")},
            "committed_observation_count": sum(r.get("row_count", 0) for r in rows),
            "committed_page_count": sum(r.get("page_count", 0) for r in rows),
            "validated_observation_count": sum(r.get("row_count", 0) for r in rows if r["integrity_validated"]),
        })
        if final:
            result["finished_at"] = result["updated_at"]
            result["status"] = ("failed_or_partial" if result["status_counts"]["failed"]
                                else "api_exhausted" if result["status_counts"]["exhausted"] == len(rows)
                                else "paused")
        write_json(progress_path, result)
        if on_progress is not None:
            on_progress(result)
        return result

    def collect(condition: str) -> dict:
        with state_lock:
            statuses[condition]["status"] = "running"
        client = (client_factory(cache_dir / condition) if client_factory else
                  HttpClient(cache_dir / condition, compress=compress))
        return ingest_condition(_PacedClient(client, pacer), condition_id=condition,
                                output_dir=output_dir, limit=limit,
                                max_pages=max_pages_per_condition, compress=compress,
                                minimum_size=minimum_size)

    with _batch_lock(output_dir):
        snapshot()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="worldcup-trades") as pool:
            futures = {pool.submit(collect, c): c for c in conditions}
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=progress_interval, return_when=FIRST_COMPLETED)
                for future in done:
                    condition = futures[future]
                    try:
                        state = future.result()
                        status = {"condition_id": condition, "status": state["api_traversal_status"],
                                  "integrity_validated": True, **_counts(state)}
                    except Exception as error:
                        status = {**statuses[condition], "status": "failed",
                                  "integrity_validated": False,
                                  "error_type": type(error).__name__, "error": str(error)[:1000]}
                    with state_lock:
                        statuses[condition] = status
                snapshot()
        return snapshot(final=True)
