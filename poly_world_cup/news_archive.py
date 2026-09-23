"""Recover headline-only historical news versions with Wayback evidence.

CDX rows prove URL capture, not the present headline. Only a verified replay can
produce a historical version. Raw HTML stays in the local cache, never exports.
"""
from __future__ import annotations

import base64
import concurrent.futures
import gzip
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .attribution import is_2026_world_cup_headline
from .http import HttpClient
from .io import atomic_write, write_json, write_jsonl

CDX_URL = "https://web.archive.org/cdx/search/cdx"
_FIELDS = ("timestamp", "original", "mimetype", "digest", "statuscode")


def utc_capture(timestamp: str) -> str:
    if not isinstance(timestamp, str) or not re.fullmatch(r"\d{14}", timestamp):
        raise ValueError("Archive timestamp must contain exactly 14 digits")
    return datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError("Expected a public HTTP(S) article URL")
    # HTTP/HTTPS are equivalent for article identity, but path/query are preserved.
    return urlunsplit(("https", parts.netloc.lower(), parts.path, parts.query, ""))


def parse_cdx(rows: list, source_url: str, cutoff: str) -> list[dict]:
    if not rows:
        return []
    if not isinstance(rows, list) or not isinstance(rows[0], list):
        raise ValueError("Unexpected CDX payload")
    fields = rows[0]
    if not set(_FIELDS).issubset(fields):
        raise ValueError("CDX fields missing")
    result = []
    for raw in rows[1:]:
        if len(raw) != len(fields):
            raise ValueError("Malformed CDX row")
        row = dict(zip(fields, raw))
        utc_capture(row["timestamp"])
        if row["statuscode"] != "200" or row["mimetype"] not in ("text/html", "application/xhtml+xml"):
            continue
        if row["timestamp"] > cutoff or canonical_url(row["original"]) != canonical_url(source_url):
            continue
        if not re.fullmatch(r"[A-Z2-7]{32}", row["digest"]):
            continue
        if row not in result:
            result.append(row)
    return sorted(result, key=lambda row: (row["timestamp"], row["original"], row["digest"]))


class HeadlineParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.og_titles = []
        self.titles = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("property", "").lower() == "og:title":
            self.og_titles.append(attrs.get("content", ""))
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.titles.append(data)

    def headline(self):
        choices = list(dict.fromkeys(" ".join(s.split()) for s in self.og_titles if s.strip()))
        if len(choices) > 1:
            raise ValueError("Conflicting archived og:title fields")
        value = choices[0] if choices else " ".join("".join(self.titles).split())
        if not value or len(value) > 1000:
            raise ValueError("No usable archived headline")
        if any(term in value.lower() for term in ("access denied", "just a moment", "robot check", "wayback machine")):
            raise ValueError("Replay contains an error/interstitial headline")
        return value


def verify_replay(capture: dict, final_url: str, headers: dict, raw: bytes) -> tuple[bytes, str]:
    """Require exact replay identity, Memento date and CDX payload digest."""
    headers = {key.lower(): value for key, value in headers.items()}
    expected = "https://web.archive.org/web/" + capture["timestamp"] + "id_/" + capture["original"]
    if final_url != expected:
        raise ValueError("Replay redirected away from requested exact capture")
    memento = parsedate_to_datetime(headers.get("memento-datetime", ""))
    if memento.tzinfo is None or memento.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S") != capture["timestamp"]:
        raise ValueError("Memento capture timestamp mismatch")
    original = re.search(r'<([^>]+)>;\s*rel="original"', headers.get("link", ""))
    if not original or canonical_url(original.group(1)) != canonical_url(capture["original"]):
        raise ValueError("Memento original URL mismatch")
    if "text/html" not in headers.get("content-type", "") and "application/xhtml+xml" not in headers.get("content-type", ""):
        raise ValueError("Replay is not HTML")
    if headers.get("content-encoding", "").lower() == "gzip":
        body = gzip.decompress(raw)
    elif headers.get("content-encoding", "").lower() in ("", "identity"):
        body = raw
    else:
        raise ValueError("Unsupported replay content encoding")
    if len(body) > 20_000_000:
        raise ValueError("Replay exceeds decoded size bound")
    digests = {base64.b32encode(hashlib.sha1(value).digest()).decode() for value in (raw, body)}
    if capture["digest"] not in digests:
        raise ValueError("Replay payload does not match CDX digest")
    parser = HeadlineParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    return body, parser.headline()


def archive_version(article: dict, capture: dict, replay: dict) -> dict:
    """Build a distinct archived headline version, never upgrade present text."""
    available = utc_capture(capture["timestamp"])
    evidence = {"kind": "archive_snapshot", "captured_at_utc": available,
                "source_url": replay["archive_url"], "content_sha256": replay["body_sha256"]}
    article_id = article.get("source_article_id") or article.get("article_id")
    # News IDs from the current collector may themselves contain version hashes.
    item_id = (f"espn:{article_id}" if article_id else article.get("news_item_id") or article.get("news_id", "unknown")) + ":archived-headline"
    news_id = "wayback:" + hashlib.sha256((capture["original"] + capture["timestamp"] + replay["body_sha256"]).encode()).hexdigest()[:24]
    fixture_links = []
    # Direct game report/preview URLs identify a fixture in the archived source.
    direct = re.search(r"/(?:report|preview)/_/gameId/(\d+)(?:$|[/?])", urlsplit(capture["original"]).path)
    direct_fixture = f"espn:{direct.group(1)}" if direct else None
    candidates = set(article.get("fixture_ids", []))
    # Registry membership comes from caller candidates; do not create an
    # unrelated fixture merely because an archived ESPN URL has a game ID.
    for fixture in sorted(candidates):
        verified = fixture == direct_fixture
        link = {"fixture_id": fixture, "historical_link_verified": verified,
                "relevance_basis": "archived_direct_game_url" if verified else "current_metadata_candidate_only"}
        if verified:
            link.update(link_availability_upper_utc=available, link_availability_evidence=[evidence])
        fixture_links.append(link)
    return {
        "news_id": news_id, "news_item_id": item_id, "version_rank": 0,
        "source": "wayback_espn", "source_article_id": article_id,
        "source_url": capture["original"], "title": replay["headline"],
        "description": None, "body": None,
        "fixture_ids": sorted(candidates), "fixture_links": fixture_links,
        "historical_availability_verified": True, "availability_upper_utc": available,
        "historical_content_sha256": replay["body_sha256"],
        "availability_evidence": [evidence], "evidence_scope": "archived_headline_only",
        "archive_capture_timestamp": capture["timestamp"], "archive_url": replay["archive_url"],
        "archive_cdx_digest": capture["digest"], "retrieved_at_utc": replay["retrieved_at"],
        "publisher_publication_time_claim": article.get("published_at_utc"),
        "current_news_id": article.get("news_id"),
        "raw_html_in_export": False,
        "context_scope": "tournament" if is_2026_world_cup_headline(replay["headline"]) else "candidate",
        "historical_tournament_scope_verified": bool(is_2026_world_cup_headline(replay["headline"])),
        "tournament_scope_availability_upper_utc": available if is_2026_world_cup_headline(replay["headline"]) else None,
        "tournament_scope_availability_evidence": [evidence] if is_2026_world_cup_headline(replay["headline"]) else [],
    }


class ArchiveCollector:
    """Bounded request rate; per-article checkpointing and immutable raw captures."""
    def __init__(self, output_dir: Path, *, cutoff="20260719235959", start="20260101", workers=2, interval=1.0):
        utc_capture(cutoff)
        if workers < 1 or workers > 16 or interval < 0:
            raise ValueError("Use 1-16 workers and a nonnegative request interval")
        self.output_dir = Path(output_dir)
        self.cutoff = cutoff
        self.start = start
        self.workers = workers
        self.interval = interval
        self.lock = threading.Lock()
        self.last_request = 0.0
        self.prefetched = {}
        self.prefix_statuses = []
        self.restore_lock = threading.Lock()
        self.restored_cdx = {}
        self.client = HttpClient(self.output_dir / "cdx_cache", timeout=45, retries=2, compress=True)

    def pace(self):
        with self.lock:
            remaining = self.interval - (time.monotonic() - self.last_request)
            if remaining > 0:
                time.sleep(remaining)
            self.last_request = time.monotonic()

    def prefetch_prefixes(self, articles: list[dict]) -> None:
        """Batch exact candidate URLs against modest ESPN index prefixes.

        A capped response cannot certify absence: unreturned candidates then fall
        back to exact lookups. CDX establishes candidates only, never content.
        """
        groups = {}
        for article in articles:
            source_url = article.get("source_url", "")
            parts = urlsplit(source_url)
            if parts.netloc.lower() != "www.espn.com":
                continue
            match = re.match(r"(/(?:soccer|football|espn)/story/_/id/\d{2})", parts.path)
            if not match:
                match = re.match(r"(/soccer/(?:report|preview)/_/gameId/\d{3})", parts.path)
            if match:
                prefix = parts.netloc.lower() + match.group(1)
                groups.setdefault(prefix, []).append(source_url)
        for prefix, urls in sorted(groups.items()):
            try:
                self.pace()
                fetched = self.client.get_json(CDX_URL, {"url": prefix, "matchType": "prefix", "output": "json", "filter": "statuscode:200", "from": self.start, "to": self.cutoff,
                    "fl": ",".join(_FIELDS), "collapse": "urlkey", "limit": 10000})
                rows = fetched.data
                if not isinstance(rows, list) or (rows and not set(_FIELDS).issubset(rows[0])):
                    raise ValueError("Unexpected prefix CDX fields")
                indexed = {}
                if rows:
                    for raw in rows[1:]:
                        if len(raw) != len(rows[0]):
                            raise ValueError("Malformed prefix CDX row")
                        row = dict(zip(rows[0], raw))
                        indexed.setdefault(canonical_url(row["original"]), []).append(raw)
                exhausted = len(rows) < 10001
                for url in urls:
                    matches = indexed.get(canonical_url(url), [])
                    if matches or exhausted:
                        captures = parse_cdx([rows[0], *matches] if rows else [], url, self.cutoff)
                        self.prefetched[url] = {"captures": captures, "body_sha256": fetched.body_sha256, "url": fetched.url}
                self.prefix_statuses.append({"prefix": prefix, "rows": max(0, len(rows) - 1), "response_limit_reached": not exhausted, "candidate_urls": len(urls)})
            except Exception as error:
                self.prefix_statuses.append({"prefix": prefix, "error": f"{type(error).__name__}: {error}"})
        write_json(self.output_dir / "prefix_queries.json", self.prefix_statuses)

    def replay(self, capture: dict):
        archive_url = "https://web.archive.org/web/" + capture["timestamp"] + "id_/" + capture["original"]
        key = hashlib.sha256(archive_url.encode()).hexdigest()
        metadata_path = self.output_dir / "replays" / f"{key}.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            body = gzip.decompress((self.output_dir / "html_cache" / f"{metadata['body_sha256']}.html.gz").read_bytes())
            if (hashlib.sha256(body).hexdigest() != metadata["body_sha256"] or metadata["archive_url"] != archive_url
                    or metadata.get("capture_timestamp") != capture["timestamp"] or metadata.get("cdx_digest") != capture["digest"]):
                raise ValueError("Archive cache integrity failure")
            payload_hash = metadata.get("payload_sha256")
            raw = None
            if payload_hash:
                raw = gzip.decompress((self.output_dir / "payload_cache" / f"{payload_hash}.bin.gz").read_bytes())
                if hashlib.sha256(raw).hexdigest() != payload_hash:
                    raise ValueError("Archive transport payload cache integrity failure")
            elif base64.b32encode(hashlib.sha1(body).digest()).decode() == capture["digest"]:
                # Legacy identity-encoded captures can be upgraded losslessly.
                raw = body
            if raw is not None:
                verified_body, headline = verify_replay(capture, archive_url, metadata["response_headers"], raw)
                if verified_body != body or headline != metadata["headline"]:
                    raise ValueError("Archived headline cache integrity failure")
                if not payload_hash:
                    payload_hash = hashlib.sha256(raw).hexdigest()
                    atomic_write(self.output_dir / "payload_cache" / f"{payload_hash}.bin.gz", gzip.compress(raw, mtime=0))
                    metadata["payload_sha256"] = payload_hash
                    write_json(metadata_path, metadata)
                return metadata
            # Earlier collector versions did not preserve encoded payload bytes.
            # Reacquire once; decoded text cannot prove a raw-only CDX digest.
        self.pace()
        with urlopen(Request(archive_url, headers={"User-Agent": "Mozilla/5.0 (compatible; PolyWorldCupResearch/0.1)", "Accept-Encoding": "identity"}), timeout=45) as response:
            raw = response.read(20_000_001)
            if len(raw) > 20_000_000:
                raise ValueError("Replay exceeds encoded size bound")
            headers = dict(response.headers)
            body, headline = verify_replay(capture, response.url, headers, raw)
        body_hash = hashlib.sha256(body).hexdigest()
        payload_hash = hashlib.sha256(raw).hexdigest()
        metadata = {"archive_url": archive_url, "body_sha256": body_hash, "payload_sha256": payload_hash, "headline": headline,
                    "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "capture_timestamp": capture["timestamp"], "cdx_digest": capture["digest"],
                    "response_headers": {k: v for k, v in headers.items() if k.lower() in ("memento-datetime", "link", "content-type", "content-encoding")}}
        atomic_write(self.output_dir / "html_cache" / f"{body_hash}.html.gz", gzip.compress(body, mtime=0))
        atomic_write(self.output_dir / "payload_cache" / f"{payload_hash}.bin.gz", gzip.compress(raw, mtime=0))
        write_json(metadata_path, metadata)
        return metadata

    def restored_captures(self, digest: str, source_url: str) -> list[dict]:
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid cached CDX digest")
        with self.restore_lock:
            if digest not in self.restored_cdx:
                indexed = gzip.decompress((self.output_dir / "cdx_cache" / "bodies" / f"{digest}.json.gz").read_bytes())
                if hashlib.sha256(indexed).hexdigest() != digest:
                    raise ValueError("CDX checkpoint cache integrity failure")
                rows = json.loads(indexed)
                grouped = {}
                if rows:
                    if not set(_FIELDS).issubset(rows[0]):
                        raise ValueError("Cached CDX fields missing")
                    for raw in rows[1:]:
                        if len(raw) != len(rows[0]):
                            raise ValueError("Malformed cached CDX row")
                        row = dict(zip(rows[0], raw))
                        grouped.setdefault(canonical_url(row["original"]), [rows[0]]).append(raw)
                self.restored_cdx[digest] = grouped
            rows = self.restored_cdx[digest].get(canonical_url(source_url), [])
        return parse_cdx(rows, source_url, self.cutoff)

    def collect_article(self, article: dict) -> dict:
        source_url = article.get("source_url")
        key = hashlib.sha256((str(source_url) + self.start + self.cutoff).encode()).hexdigest()
        checkpoint = self.output_dir / "articles" / f"{key}.json"
        if checkpoint.exists():
            previous = json.loads(checkpoint.read_text())
            if previous.get("source_url") != source_url:
                raise ValueError("Article checkpoint URL mismatch")
            if previous.get("status") not in ("error", "capture_found_replay_unverified"):
                captures = self.restored_captures(previous["cdx_body_sha256"], source_url)
                if previous.get("status") == "no_capture_found":
                    if captures or previous.get("versions"):
                        raise ValueError("No-capture checkpoint contradicts CDX evidence")
                elif previous.get("status") == "verified_headline":
                    if len(previous.get("versions", [])) != 1:
                        raise ValueError("Verified checkpoint must contain one headline version")
                    old = previous["versions"][0]
                    matches = [row for row in captures if row["timestamp"] == old.get("archive_capture_timestamp")
                               and row["digest"] == old.get("archive_cdx_digest")
                               and row["original"] == old["source_url"]]
                    if len(matches) != 1:
                        raise ValueError("Checkpoint capture has no matching cached CDX proof")
                    replay = self.replay(matches[0])
                    # Rebuild availability, evidence, scope and links from the
                    # verified capture. Never trust a checkpoint's timing flags.
                    previous["versions"] = [archive_version(article, matches[0], replay)]
                else:
                    raise ValueError("Unknown archive checkpoint status")
                write_json(checkpoint, previous)
                return previous
        result = {"source_url": source_url, "current_news_id": article.get("news_id"), "status": "error", "versions": []}
        try:
            canonical_url(source_url)
            if source_url in self.prefetched:
                prefetched = self.prefetched[source_url]
                captures = prefetched["captures"]
                result.update(cdx_body_sha256=prefetched["body_sha256"], capture_candidates=len(captures), cdx_query=prefetched["url"], discovery_mode="prefix_index")
            else:
                self.pace()
                fetched = self.client.get_json(CDX_URL, {"url": source_url, "output": "json", "filter": "statuscode:200", "from": self.start, "to": self.cutoff,
                    "fl": ",".join(_FIELDS), "limit": 3})
                captures = parse_cdx(fetched.data, source_url, self.cutoff)
                result.update(cdx_body_sha256=fetched.body_sha256, capture_candidates=len(captures), cdx_query=fetched.url, discovery_mode="exact_url")
            if not captures:
                result["status"] = "no_capture_found"
            else:
                errors = []
                for capture in captures:
                    try:
                        replay = self.replay(capture)
                        result["versions"] = [archive_version(article, capture, replay)]
                        result["status"] = "verified_headline"
                        break
                    except Exception as error:
                        errors.append(f"{type(error).__name__}: {error}")
                if errors:
                    result["replay_errors"] = errors
                if not result["versions"]:
                    result["status"] = "capture_found_replay_unverified"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        write_json(checkpoint, result)
        return result

    def collect(self, articles: list[dict], *, max_articles: int | None = None, prefetch: bool = True) -> dict:
        unique = {}
        for article in articles:
            if article.get("source_url"):
                unique.setdefault(article["source_url"], article)
        selected = list(unique.values())[:max_articles]
        write_json(self.output_dir / "progress.json", {"requested_articles": len(selected), "completed_articles": 0, "phase": "index_discovery", "statuses": {}})
        if prefetch:
            self.prefetch_prefixes(selected)
        statuses = {}
        versions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            pending = {executor.submit(self.collect_article, article): article for article in selected}
            for future in concurrent.futures.as_completed(pending):
                try:
                    result = future.result()
                except Exception as error:
                    # Corrupt cached evidence never enters the exported context.
                    result = {"source_url": pending[future].get("source_url"), "status": "error", "versions": [], "error": f"{type(error).__name__}: {error}"}
                    key = hashlib.sha256(str(result["source_url"]).encode()).hexdigest()
                    write_json(self.output_dir / "validation_errors" / f"{key}.json", result)
                statuses[result["status"]] = statuses.get(result["status"], 0) + 1
                versions.extend(result["versions"])
                write_json(self.output_dir / "progress.json", {"requested_articles": len(selected), "completed_articles": sum(statuses.values()), "statuses": statuses})
        versions.sort(key=lambda row: (row["availability_upper_utc"], row["news_id"]))
        write_jsonl(self.output_dir / "news_archive.jsonl", versions)
        report = {"unique_input_urls": len(unique), "requested_articles": len(selected), "completed_articles": sum(statuses.values()), "statuses": statuses,
                  "workers": self.workers, "global_request_interval_seconds": self.interval,
                  "index_prefix_queries": self.prefix_statuses,
                  "verified_headline_versions": len(versions),
                  "verified_global_tournament_headlines": sum(row.get("context_scope") == "tournament" for row in versions),
                  "fixture_links_verified": sum(sum(bool(link["historical_link_verified"]) for link in row["fixture_links"]) for row in versions),
                  "historical_coverage_complete": False, "content_scope": "first_verified_headline_among_bounded_capture_candidates",
                  "prefix_capture_collapse": "urlkey", "exact_lookup_candidate_limit": 3,
                  "lookup_start": self.start, "lookup_cutoff": self.cutoff, "raw_html_in_export": False}
        write_json(self.output_dir / "report.json", report)
        return report
