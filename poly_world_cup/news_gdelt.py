"""Historical headline versions from timestamped GDELT GKG archive batches.

GKG PAGE_TITLE is the title captured by GDELT, not today's article text. Batch
timestamps identify the archive's update batch; publisher dates are not used to
backdate content. This is a sampled archive search unless every batch in the
requested range is actually visited. It cannot certify exhaustive news coverage.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import gzip
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import tempfile
import unicodedata
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile

from .attribution import is_2026_world_cup_headline
from .io import write_json, write_jsonl

BASE_URL = "https://data.gdeltproject.org/gdeltv2/"
MAX_COMPRESSED = 40_000_000
MAX_UNCOMPRESSED = 200_000_000
MAX_LINE = 5_000_000
_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
_SHA = re.compile(r"[0-9a-f]{64}")
CANDIDATE_POLICY_VERSION = 1


def batch_time(stamp: str) -> datetime:
    if not isinstance(stamp, str) or not re.fullmatch(r"\d{14}", stamp):
        raise ValueError("Batch timestamp must have fourteen digits")
    value = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    if value.minute % 15 or value.second:
        raise ValueError("GKG timestamp must lie on a fifteen-minute boundary")
    return value


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _normal(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


@lru_cache(maxsize=8)
def _team_phrases(teams: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(" " + _normal(team) + " " for team in teams)


def candidate_title(title: str, url: str, teams: list[str]) -> bool:
    """Broad retrieval filter only; fixture relevance needs a separate check."""
    text = _normal(title)
    if re.search(r"\b(?:world cup|fifa|soccer|football)\b", text):
        return True
    padded = " " + text + " "
    mentions = sum(team in padded for team in _team_phrases(tuple(teams)))
    return mentions >= 2 or (mentions >= 1 and bool(re.search(
        r"soccer|football|world.?cup|fifa|sports", url, re.I)))


def parse_record(raw_line: bytes, *, stamp: str, row_number: int,
                 archive_sha256: str, retrieved_at: str) -> dict | None:
    """Validate the archive identity before constructing a historical title."""
    batch = batch_time(stamp)
    if row_number < 1 or not _SHA.fullmatch(archive_sha256):
        raise ValueError("Invalid archive provenance")
    raw_line = raw_line.rstrip(b"\r\n")
    if len(raw_line) > MAX_LINE:
        raise ValueError("GKG row exceeds limit")
    fields = raw_line.decode("utf-8", errors="strict").split("\t")
    if len(fields) != 27:
        raise ValueError("Expected the 27 GKG 2.1 fields")
    if not re.fullmatch(re.escape(stamp) + r"-(?:T)?\d+", fields[0]):
        raise ValueError("GKG record ID disagrees with archive batch")
    if fields[1] != stamp:
        raise ValueError("GKG record DATE disagrees with archive batch")
    if fields[2] != "1":
        return None
    url = fields[4]
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        return None
    titles = list(dict.fromkeys(_TITLE.findall(fields[26])))
    if not titles:
        return None
    if len(titles) != 1:
        raise ValueError("Conflicting PAGE_TITLE values")
    title = " ".join(html.unescape(titles[0]).split())
    if (not title or len(title) > 1000 or "\ufffd" in title
            or re.search(r"access denied|just a moment|robot check|404 not found", title, re.I)):
        return None
    digest = hashlib.sha256(raw_line).hexdigest()
    # GKG's seen/processed timestamp has fifteen-minute resolution. The docs do
    # not identify a floor/ceiling convention, so use the end of that interval.
    # Keep the original batch label separate from this conservative bound.
    observed = max(batch, batch_time(fields[1]))
    available = _utc(observed + timedelta(minutes=15))
    archive_url = BASE_URL + stamp + ".gkg.csv.zip"
    evidence = {
        "kind": "archive_snapshot", "archive_provider": "GDELT GKG 2.1",
        "captured_at_utc": _utc(observed), "source_url": archive_url,
        "content_sha256": digest, "archive_zip_sha256": archive_sha256,
        "archive_row_number": row_number, "archive_record_id": fields[0],
        "archive_batch_utc": _utc(batch), "record_date_utc": _utc(observed),
        "capture_timestamp_precision": "15_minute_batch",
        "hash_scope": "exact_utf8_gkg_record_without_line_ending",
    }
    item_id = "gdelt:" + hashlib.sha256(url.encode()).hexdigest()[:24]
    news_id = "gdelt:" + hashlib.sha256((url + "\0" + title + "\0" + available).encode()).hexdigest()[:24]
    global_scope = is_2026_world_cup_headline(title)
    return {
        "news_id": news_id, "news_item_id": item_id, "version_rank": 0,
        "version_order_historically_verified": True,
        "source": "gdelt_gkg", "source_provider": fields[3],
        "source_url": url, "title": title, "content_type": "archived_headline",
        "historical_text_version_status": "verified_gdelt_archived_headline",
        "historical_availability_verified": True, "availability_upper_utc": available,
        "availability_precision": "15_minute_batch",
        "availability_rule": "end_of_gdelt_15_minute_observation_interval",
        "historical_content_sha256": digest, "availability_evidence": [evidence],
        "captured_at_utc": _utc(observed), "retrieved_at_utc": retrieved_at,
        "published_at_utc": None, "published_time_used_for_availability": False,
        "actor_exposure_verified": False, "causal_attribution": False,
        "full_article_text_collected_in_export": False,
        "evidence_scope": "archived_page_title_only", "fixture_ids": [], "fixture_links": [],
        "context_scope": "tournament" if global_scope else "candidate",
        "historical_tournament_scope_verified": global_scope,
        "tournament_scope_availability_upper_utc": available if global_scope else None,
        "tournament_scope_availability_evidence": [evidence] if global_scope else [],
    }


def parse_batch(raw: bytes, *, stamp: str, retrieved_at: str,
                teams: list[str]) -> tuple[list[dict], list[dict], dict]:
    """Return candidate news, exact retained evidence rows, and batch statistics."""
    batch_time(stamp)
    if len(raw) > MAX_COMPRESSED:
        raise ValueError("Compressed GKG archive exceeds limit")
    archive_sha = hashlib.sha256(raw).hexdigest()
    news, evidence_rows = [], []
    total = rejected = 0
    invalid_encoding_rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (len(members) != 1 or members[0].filename != stamp + ".gkg.csv"
                or members[0].file_size > MAX_UNCOMPRESSED):
            raise ValueError("Unexpected GKG ZIP layout or expanded size")
        expanded = 0
        with archive.open(members[0]) as stream:
            for total, raw_line in enumerate(stream, 1):
                expanded += len(raw_line)
                if expanded > MAX_UNCOMPRESSED or len(raw_line) > MAX_LINE:
                    raise ValueError("Expanded GKG archive exceeds limit")
                try:
                    record = parse_record(raw_line, stamp=stamp, row_number=total,
                                          archive_sha256=archive_sha, retrieved_at=retrieved_at)
                except UnicodeDecodeError:
                    # GKG occasionally contains an invalid byte in an unrelated
                    # record. Reject that entire record, never repair its title,
                    # while preserving the independently valid archive rows.
                    invalid_encoding_rows.append({
                        "row_number": total,
                        "record_sha256": hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest(),
                        "reason": "record_is_not_valid_utf8",
                    })
                    continue
                if record is None or not candidate_title(record["title"], record["source_url"], teams):
                    rejected += 1
                    continue
                news.append(record)
                evidence_rows.append({"row_number": total,
                                      "raw_record": raw_line.rstrip(b"\r\n").decode("utf-8"),
                                      "record_sha256": record["historical_content_sha256"]})
    return news, evidence_rows, {
        "stamp": stamp, "status": "ok", "archive_url": BASE_URL + stamp + ".gkg.csv.zip",
        "archive_zip_sha256": archive_sha, "compressed_bytes": len(raw),
        "retrieved_at_utc": retrieved_at, "record_count": total,
        "candidate_count": len(news), "noncandidate_count": rejected,
        "invalid_encoding_rows": invalid_encoding_rows,
    }


def _write_gzip(path: Path, rows: list[dict]) -> str:
    value = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    encoded = gzip.compress(value, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def _load_cached(directory: Path, stamp: str) -> tuple[list[dict], dict] | None:
    meta_path = directory / (stamp + ".json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("status") != "ok":
        return None
    evidence_path = directory / (stamp + ".rows.jsonl.gz")
    raw = evidence_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["retained_evidence_sha256"]:
        raise ValueError("Cached retained evidence checksum mismatch")
    news = []
    for line in gzip.decompress(raw).splitlines():
        row = json.loads(line)
        record_bytes = row["raw_record"].encode()
        if hashlib.sha256(record_bytes).hexdigest() != row["record_sha256"]:
            raise ValueError("Cached GKG row checksum mismatch")
        record = parse_record(record_bytes, stamp=stamp, row_number=row["row_number"],
                              archive_sha256=meta["archive_zip_sha256"], retrieved_at=meta["retrieved_at_utc"])
        if record is None:
            raise ValueError("Cached GKG record is no longer valid")
        news.append(record)
    if len(news) != meta["candidate_count"]:
        raise ValueError("Cached GKG retained count mismatch")
    return news, meta


def historical_versions(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep captured changes, including reversions; quarantine tied conflicts."""
    rows = sorted(rows, key=lambda n: (n["news_item_id"], n["availability_upper_utc"], n["news_id"]))
    by_instant: dict[tuple, list[dict]] = {}
    for row in rows:
        by_instant.setdefault((row["news_item_id"], row["availability_upper_utc"]), []).append(row)
    last_title, retained, ranks = {}, [], {}
    conflicts = 0
    for (item_id, _), simultaneous in by_instant.items():
        if len({row["title"] for row in simultaneous}) != 1:
            conflicts += len(simultaneous)
            # A later recapture of a previous title is informative after ambiguity.
            last_title.pop(item_id, None)
            continue
        row = dict(simultaneous[0])
        if last_title.get(item_id) == row["title"]:
            continue
        last_title[item_id] = row["title"]
        row["version_rank"] = ranks.get(item_id, 0)
        ranks[item_id] = row["version_rank"] + 1
        retained.append(row)
    retained.sort(key=lambda n: (n["availability_upper_utc"], n["news_id"]))
    return retained, conflicts


def retrieval_policy(teams: list[str]) -> dict:
    """Freeze candidate selection so resume cannot silently reuse narrower data."""
    names = sorted(set(teams))
    return {
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "team_names": names,
        "team_names_sha256": hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
    }


def collect_gdelt(*, output: Path, stamps: list[str], teams: list[str],
                  workers: int = 6, request_interval: float = 0.2,
                  progress=None) -> dict:
    """Resume immutable batch captures and export the union of requested batches."""
    if not 1 <= workers <= 12 or request_interval < 0:
        raise ValueError("Use one to twelve workers and nonnegative request interval")
    stamps = sorted(set(stamps))
    for stamp in stamps:
        batch_time(stamp)
    output = Path(output)
    directory = output / "batches"
    policy_path = output / "retrieval_policy.json"
    policy = retrieval_policy(teams)
    if policy_path.exists():
        if json.loads(policy_path.read_text()) != policy:
            raise ValueError("Retrieval policy changed; use a new output directory to recollect candidates")
    elif any(directory.glob("*.json")):
        raise ValueError("Existing batch cache lacks a recorded retrieval policy")
    directory.mkdir(parents=True, exist_ok=True)
    if not policy_path.exists():
        write_json(policy_path, policy)
    lock = threading.Lock()
    last_request = [0.0]

    def one(stamp):
        cached = _load_cached(directory, stamp)
        if cached is not None:
            return cached
        url = BASE_URL + stamp + ".gkg.csv.zip"
        try:
            with lock:
                delay = request_interval - (time.monotonic() - last_request[0])
                if delay > 0:
                    time.sleep(delay)
                last_request[0] = time.monotonic()
            request = Request(url, headers={"User-Agent": "poly-world-cup-research/1.0"})
            with urlopen(request, timeout=40) as response:
                if response.status != 200 or response.url != url:
                    raise ValueError("Archive request did not return exact successful URL")
                raw = response.read(MAX_COMPRESSED + 1)
            retrieved = _utc(datetime.now(timezone.utc))
            rows, evidence, meta = parse_batch(raw, stamp=stamp, retrieved_at=retrieved, teams=teams)
            meta["retained_evidence_sha256"] = _write_gzip(directory / (stamp + ".rows.jsonl.gz"), evidence)
            write_json(directory / (stamp + ".json"), meta)
            return rows, meta
        except Exception as error:
            meta = {"stamp": stamp, "archive_url": url, "status": "error",
                    "error": f"{type(error).__name__}: {error}"}
            write_json(directory / (stamp + ".error.json"), meta)
            return [], meta

    rows_by_id, batches = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, stamp): stamp for stamp in stamps}
        for future in as_completed(futures):
            rows, meta = future.result()
            batches.append(meta)
            for row in rows:
                rows_by_id.setdefault(row["news_id"], row)
            if progress:
                progress({"completed_batches": len(batches), "requested_batches": len(stamps),
                          "candidates": len(rows_by_id), "last_batch": meta["stamp"], "last_status": meta["status"]})
    retained, conflicts = historical_versions(list(rows_by_id.values()))
    write_jsonl(output / "news.jsonl", retained)
    report = {"source": "GDELT GKG 2.1 historical PAGE_TITLE archives", "requested_batches": len(stamps),
              "completed_batches": sum(b["status"] == "ok" for b in batches),
              "failed_batches": sum(b["status"] != "ok" for b in batches),
              "candidate_historical_title_versions": len(retained),
              "conflicting_same_instant_versions_quarantined": conflicts,
              "scanned_gkg_records": sum(b.get("record_count", 0) for b in batches),
              "invalid_utf8_records_rejected": sum(len(b.get("invalid_encoding_rows", [])) for b in batches),
              "downloaded_compressed_bytes": sum(b.get("compressed_bytes", 0) for b in batches),
              "historical_coverage_complete": False, "fixture_relevance_verified_by_collector": False,
              "capture_time_rule": "GKG archive batch timestamp plus fifteen minutes, accounting for fifteen-minute seen/processed resolution; batch required equal to record DATE and record-ID prefix",
              "retrieval_policy": policy,
              "publisher_publication_claim_used_as_availability": False,
              "batch_schedule_sha256": hashlib.sha256("\n".join(stamps).encode()).hexdigest(),
              "batches": sorted(batches, key=lambda b: b["stamp"])}
    write_json(output / "report.json", report)
    return report


def batch_schedule(start: str, end: str, step_minutes: int = 180) -> list[str]:
    if step_minutes < 15 or step_minutes % 15:
        raise ValueError("Step must be a positive multiple of fifteen minutes")
    first, last = batch_time(start), batch_time(end)
    if last < first:
        raise ValueError("End must not precede start")
    stamps = []
    while first <= last:
        stamps.append(first.strftime("%Y%m%d%H%M%S"))
        first += timedelta(minutes=step_minutes)
    return stamps


def package_gdelt_evidence(source: Path, output: Path, *, shard_bytes: int = 16_000_000) -> dict:
    """Package exact selected GKG records, batch provenance and title catalog.

    The title catalog is a candidate set, not a set of fixture-verified features.
    Raw GKG records are reproducibility evidence, never language-model prompts.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Evidence output already exists")
    if not 1 <= shard_bytes <= 16_000_000:
        raise ValueError("Use an uncompressed shard target at most sixteen million bytes")
    report = json.loads((source / "report.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    files, records_count = [], 0
    with tempfile.TemporaryDirectory(prefix=output.name + ".tmp-", dir=output.parent) as temp:
        target = Path(temp)

        def save_bytes(relative: str, value: bytes, rows: int | None = None):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value)
            meta = {"path": relative, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            if rows is not None:
                meta["rows"] = rows
            files.append(meta)

        def save_jsonl(relative: str, rows: list[dict]):
            raw = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
            save_bytes(relative, gzip.compress(raw, mtime=0), len(rows))

        pending, pending_bytes, part = [], 0, 0
        invalid = []
        for batch in report["batches"]:
            if batch["status"] != "ok":
                continue
            _, verified_meta = _load_cached(source / "batches", batch["stamp"])
            if verified_meta != batch:
                raise ValueError("Collection report disagrees with cached batch metadata")
            evidence_file = source / "batches" / (batch["stamp"] + ".rows.jsonl.gz")
            with gzip.open(evidence_file, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    packaged = {"archive_batch": batch["stamp"], "archive_url": batch["archive_url"],
                                "archive_zip_sha256": batch["archive_zip_sha256"], **row}
                    estimated = len(json.dumps(packaged, ensure_ascii=False).encode()) + 1
                    if pending and pending_bytes + estimated > shard_bytes:
                        part += 1
                        save_jsonl(f"records/part-{part:05}.jsonl.gz", pending)
                        pending, pending_bytes = [], 0
                    pending.append(packaged)
                    pending_bytes += estimated
                    records_count += 1
            for rejected in batch.get("invalid_encoding_rows", []):
                invalid.append({"archive_batch": batch["stamp"], "archive_url": batch["archive_url"],
                                "archive_zip_sha256": batch["archive_zip_sha256"], **rejected})
        if pending:
            part += 1
            save_jsonl(f"records/part-{part:05}.jsonl.gz", pending)
        save_jsonl("batches.jsonl.gz", report["batches"])
        save_jsonl("invalid_records.jsonl.gz", invalid)
        catalog = (source / "news.jsonl").read_bytes()
        save_bytes("candidate_news.jsonl.gz", gzip.compress(catalog, mtime=0), len(catalog.splitlines()))
        save_bytes("retrieval_policy.json", (source / "retrieval_policy.json").read_bytes())
        summary = {k: v for k, v in report.items() if k != "batches"}
        save_bytes("collection_report.json", (json.dumps(summary, sort_keys=True, indent=2) + "\n").encode())
        readme = """# Historical GDELT headline evidence

Source: The GDELT Project, https://www.gdeltproject.org/ .
These records are selected from GDELT's public GKG 2.1 archive. Credit and terms:
https://www.gdeltproject.org/about.html#termsofuse .

`candidate_news.jsonl.gz` preserves the collected headline candidates. Fixture
relevance is established separately. `records/` contains the exact selected GKG
metadata records, with source ZIP SHA256, one-based row number and record SHA256.
The record hash covers its exact UTF-8 bytes without the line ending. These raw
records are audit evidence and must not be used as SFT prompts. Article bodies
and full source ZIP files are not included. To verify against the upstream
archive, download the listed ZIP, check its checksum, and compare the indexed row.
`batches.jsonl.gz` records every requested batch and error. `invalid_records`
records rejected, invalid-UTF8 rows by archive identity and hash.

The search starts with three-hour sampling and adds denser windows around
fixtures with missing context. Its exact visited batch schedule is recorded.
This does not assert that all fifteen-minute batches or all relevant news were
collected. Existing headlines retain their archive version and never use today's
article text or a publisher's claimed publication date for historical timing.

Availability uses the GKG batch label plus fifteen minutes, a conservative
convention for GKG's documented fifteen-minute seen/processed resolution. The
raw batch label remains separate. This is not an exact page-capture timestamp.
Use only news whose availability upper bound is strictly before the query time.

Official format and timestamp sources:
- https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf
- https://blog.gdeltproject.org/gkg-2-0-now-includes-page-titles/
- https://blog.gdeltproject.org/announcing-our-first-api-gkg-geojson/
- https://blog.gdeltproject.org/a-behind-the-scenes-look-at-how-we-think-about-master-file-formats-and-timestamping/
"""
        save_bytes("README.md", readme.encode())
        if any(entry["bytes"] >= 20_000_000 for entry in files):
            raise ValueError("Evidence file exceeds twenty-million-byte publication bound")
        manifest = {"schema_version": 1, "source_credit": "The GDELT Project https://www.gdeltproject.org/",
                    "retained_archive_records": records_count, "candidate_title_versions": summary["candidate_historical_title_versions"],
                    "historical_coverage_complete": False, "raw_records_are_sft_features": False,
                    "files": files}
        (target / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        os.rename(target, output)
    return manifest
