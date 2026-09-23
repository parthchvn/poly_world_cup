"""Public HTTP JSON retrieval with immutable response bodies and provenance.

A retrieval timestamp means 'we captured this now', never 'known then'. Cache
replays retain the original capture timestamp. Run one writer per cache directory.
"""

from __future__ import annotations

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

from .io import atomic_write, write_json


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
        write_json(self.cache_dir / "captures" / f"{capture_hash}.json", metadata)
        write_json(index_path, metadata)
        return FetchResult(data, full_url, retrieved_at, body_hash, False)
