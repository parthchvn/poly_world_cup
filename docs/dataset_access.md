# Open and query the collected dataset

A completed-traversal release is named `world_cup_corpus.tar.gz`. An explicitly
incomplete checkpoint is named **`world_cup_partial_corpus.tar.gz`**; use that
filename instead in the commands below. Its manifests state exactly which
contracts exhausted and which paused. A partial checkpoint includes every saved
observation, but uncollected pages remain missing.

Extract the downloaded archive into a new directory:

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
  --provenance-report data/full/provenance_report.json \
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

## Generate the final collection overview

After all collection, provenance, attribution, archive, and receipt-report inputs
are finalized, produce a compact overview without scanning the trade database:

```bash
python scripts/summarize_collection.py \
  --output reports/tournament_collection.json \
  --readme data/releases/2026-world-cup/DATASET_README.md
```

Optionally add `--code-revision FULL_GIT_SHA` for the exact code revision used.
The command checks the 104-fixture/312-contract universe, condition identities,
all exhausted/validated statuses, and matching observation/page totals across
batch, provenance, and attribution reports. It also compares the news catalogs
with their reports and includes the earlier diagnostic receipt probe, so a prior
mismatch is not lost behind a newer sample. Inputs are identified by SHA256.

The JSON includes exact observed date bounds, the requested trade-size filter,
one coverage row per fixture, source limitations, and explicit false flags for
upstream completeness and training readiness. No final output is written if a
required consistency check fails. The readable overview and compact JSON can be
published alongside the archives; they do not replace the per-file release hash
manifest.


## Explicitly incomplete checkpoints

Normal export still requires exhausted traversals for every registry contract.
If collection is interrupted, `--allow-partial` permits a clearly labeled
checkpoint only when all **104 fixtures and 312 contract manifests** are present,
every manifest is either `paused` or `exhausted`, and a passing provenance report
verifies every saved page. SQLite page hashes, exact observation counts, news
catalog equality, and body-exclusion checks remain mandatory.

```bash
python scripts/export_corpus.py \
  --database data/full/partial_attribution.sqlite \
  --archive-news data/news_archive/news_archive.jsonl \
  --provenance-report data/full/provenance_report.json \
  --output data/releases/2026-world-cup-partial \
  --immutable-database --allow-partial --dry-run
```

Remove `--dry-run` only after the inputs are finalized. The resulting archives
are `world_cup_partial_corpus.tar.gz` and, when requested with `--raw-trades`,
`trade_partial_provenance.tar.gz`. The outer release manifest and both inner
manifests have `partial=true` and exact paused/exhausted counts. Their READMEs
start with an **INCOMPLETE CHECKPOINT** notice. Partial export requires a passing
provenance report with an explicit registry condition universe; verifying saved
pages does not certify the uncollected part of the history. A missing manifest,
a failed/unknown state, or inconsistent counts still blocks export.

The September 23 interrupted collection recorded **293 exhausted contracts and
19 paused contracts**. Those statuses describe the captured checkpoint, not
completion of the user's full-history request. It remains unsuitable for SFT or
for inferring no-trade labels. The full-completion overview generator retains
its stricter gates and does not certify an incomplete checkpoint as complete.
