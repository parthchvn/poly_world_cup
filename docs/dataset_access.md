# Open and query the collected dataset

The primary download is `world_cup_corpus.tar.gz`. Extract it into a new directory:

```bash
mkdir world_cup_corpus
tar -xzf world_cup_corpus.tar.gz -C world_cup_corpus
```

Open `world_cup_corpus/attribution.sqlite` using a SQLite database browser, the
`sqlite3` command-line program, or Python's built-in `sqlite3` module. The database
contains the normalized observations, not just a sample or links back to the
collector's machine. Reading it does not need the raw API files.

```python
import sqlite3

connection = sqlite3.connect("world_cup_corpus/attribution.sqlite")
for row in connection.execute("""
    SELECT trade_row_id, fixture_id, wallet, side, shares, price, block_timestamp
    FROM trades ORDER BY trade_row_id LIMIT 20
"""):
    print(row)
```

Prices, shares, and token IDs are stored as **text** to preserve decimal values
and identifiers. Use Python `decimal.Decimal` for exact arithmetic. SQL
`CAST(price AS REAL)` is suitable only for approximate exploration.

## Find a match and its trades

`registry.json` maps all tournament fixtures, market conditions, selections, and
outcome tokens. The SQLite trade table also includes fixture ID, contract
selection, and token outcome when the current registry maps them exactly.

```sql
SELECT fixture_id, COUNT(*) AS observations
FROM trades GROUP BY fixture_id ORDER BY observations DESC;

SELECT trade_row_id, wallet, selection, token_outcome, side, shares, price,
       block_timestamp, transaction_hash
FROM trades
WHERE fixture_id = 'espn:760415'
ORDER BY query_us, trade_row_id
LIMIT 100;
```

A BUY of a No token remains a BUY of that No token. The data does not silently
relabel it as a sale of Yes. `token_mapping_status` marks unresolved identities.
`query_us` is an execution block-time proxy in UTC microseconds, not a verified
order-submission or decision time.

## Inspect attributed context

From a checkout of this repository:

```bash
python -m poly_world_cup inspect-context \
  --database /absolute/path/to/world_cup_corpus/attribution.sqlite \
  --row 1
```

The response keeps historically verified fixture news, historically verified
broad World Cup news, retrospective news candidates, and earlier observed wallet
executions in distinct fields. A source being relevant does not establish that
the wallet read it or acted because of it. Broad tournament news does not prove
that a specific fixture was its subject. Earlier wallet executions cover only
the collected tournament contracts; holdings and external trading are unknown.

The bundle also contains `context_samples.jsonl`: up to three browsing examples
from each of twelve evenly spaced fixture IDs, selecting first, last, and the
first observation at or after the temporal midpoint. `sample_method.json`
describes the procedure. It is a small stratified inspection set, not a
statistically representative sample and not an SFT training set.

Useful SQL:

```sql
SELECT token_mapping_status, COUNT(*) FROM trades GROUP BY token_mapping_status;

SELECT SUM(eligible_context_event_count > 0) AS rows_with_verified_fixture_news,
       SUM(global_eligible_context_event_count > 0) AS rows_with_verified_global_news,
       COUNT(*) AS observations
FROM trades;

SELECT news_id, record_json FROM news LIMIT 5;

SELECT value_json FROM metadata WHERE key = 'report';
```

## Files and provenance

| File | Contents |
|---|---|
| `attribution.sqlite` | Every manifested normalized observation and queryable context index |
| `registry.json` | Fixture, condition, selection, and token mapping |
| `news/news.jsonl` | Retrospectively captured headline and attribution metadata |
| `news/archived_headlines_*.jsonl` | Separately recovered archived headline metadata and evidence |
| `trade_manifests/*.json` | Per-condition query filters, page hashes, retrieval times, and limits |
| `reports/*.json` | Collection, verification, reconciliation, and attribution reports supplied to the exporter |
| `schema.sql` | Exact SQLite schema |
| `MANIFEST.json` | Every other bundle member's size and SHA256 |

News exports contain bibliographic metadata and headlines, not full article text
or archived HTML. Publication claims and archive-capture bounds have different
meanings. Old publication dates on current captures are not enough to make those
captures historical features.

`source_pages.path` preserves the original local provenance path. It is not
required for querying a downloaded SQLite file. If raw provenance is distributed,
its pages are under `trades/CONDITION_ID/pages/` in the separate
`trade_provenance.tar.gz`. That optional archive contains only manifest-referenced
normalized trade pages, original Polymarket trade response bodies, and matching
capture metadata. It never includes a news cache. Uncompressed content hashes in
source manifests differ from hashes of compressed archive members.

The outer `release_manifest.json` contains each downloadable archive's SHA256 and
size. Verify those before extracting, then verify inner member hashes:

```python
import hashlib
import json
from pathlib import Path

root = Path("world_cup_corpus")
manifest = json.loads((root / "MANIFEST.json").read_text())
for member in manifest["members"]:
    path = root / member["path"]
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    assert path.stat().st_size == member["bytes"]
    assert sha.hexdigest() == member["sha256"], member["path"]
print("All members verified")
```

## Build a release from a completed collection

First complete collection, stop writers, build the full attribution database,
and obtain a passing raw-provenance verification report. The exporter checks
registry manifests against every SQLite source-page digest and row count, checks
news catalog equality with SQLite, and runs SQLite's integrity check. It preserves
all normalized rows, including economic duplicates; source observations are not
silently collapsed into supposedly canonical fills.

Estimate disk use without copying data:

```bash
python scripts/export_corpus.py \
  --database data/full/attribution.sqlite \
  --archive-news data/news_archive/news_archive.jsonl \
  --provenance-report data/full/raw_provenance_report.json \
  --report data/full/attribution_report.json \
  --report data/news/coverage.json \
  --report data/news_archive/report.json \
  --output data/releases/2026-world-cup \
  --immutable-database --dry-run
```

Remove `--dry-run` to create the bundle. Add `--raw-trades` to produce the separate
raw provenance archive; its default cache layout matches the tournament
collector's `data/full/cache/CONDITION_ID/` directories. Use a fresh output
directory each time; existing archives are never overwritten.

By default the exporter uses SQLite's backup API to make a consistent snapshot.
For a large, completed, immutable database, `--immutable-database` avoids that
extra database copy through a temporary hardlink. This requires the output and
database to share a filesystem, no remaining WAL contents, and **no concurrent
database writer**. File size and modification time are checked across export;
a detected change removes the invalid archive. Without this flag, account for
one extra uncompressed database copy plus compressed output. With it, the large
input database is streamed directly. Raw provenance is also streamed through
hardlinks where supported. Compression ratios are not assumed in free-space
estimates, and already compressed raw pages may compress little further.

API exhaustion describes the requested filtered traversal. It does not certify
complete exchange history, human decision timing, or an absence of trading in a
gap. The corpus is not yet SFT-ready: reconciliation, complete activity/holdings
state, label coverage, historical context selection, and baseline evaluation
remain separate requirements.
