# Trade and news attribution

The attribution index links each observed wallet execution to a fixture and a
compact public-context timeline. These links describe **relevance and temporal
availability**. They do not establish that a wallet read an article, held a
particular belief, or acted because of it. API execution time is not a known
order-decision time.

## What is stored

`poly_world_cup.attribution.build_attribution_index` creates a SQLite snapshot.
It streams normalized JSONL or gzip JSONL pages, retaining one row per source
observation. It does not expand every trade against every news article.

| Table | Contents |
| --- | --- |
| `trades` | Wallet, exact instrument and side, decimal shares/price, source-page reference, fixture mapping, context-state reference |
| `source_pages` | Original normalized page path, uncompressed SHA-256, row count |
| `news` | One source-content version per ID, timestamps, evidence, original news metadata |
| `fixture_news` | News-to-fixture relationship, evidence, and earliest verified availability bound |
| `global_news` | One tournament-wide historical availability link per eligible content version |
| `context_states` | Stable prefix hashes of fixture and global verified context events |
| `metadata` | Coverage counts and explicit limitations |

State hashes change only when another historically verified event becomes
available. Appending later news does not change an earlier state. Equal-time
news is excluded: availability must be **strictly before** the execution proxy.
An event count is a number of eligible versions; the inspection helper selects
the highest established version rank for each article item at the cutoff.

Condition identity establishes the retrospective fixture association. An
observed token must exactly match a token in that contract's registry to receive
an outcome label. A migration-era or otherwise unknown token remains
`unresolved_token`; its row and known fixture association are retained.
`BUY No` stays `BUY No`, and does not become `SELL Yes`. Price and shares are
**provider-reported API amounts**, preserved exactly as decimal strings, and
have not been reconciled to canonical chain fills. Receipt inspection found
ambiguous fill associations and amount discrepancies; decimal preservation
alone does not establish economic accuracy. Index reports and inspection output
therefore state `trade_amounts_semantics="provider_reported_not_chain_reconciled"`
and `chain_reconciliation_verified=false`. Exact current-registry
token agreement is an identifier check, not proof that current rules or metadata
were available at an earlier time. Registry labels remain ineligible features.

Equal-looking rows are retained, including repeated observation IDs, which are
reported. These IDs identify API capture observations, not canonical executions.
Passing the identical input page path twice is an error rather than silent
reingestion. Economic deduplication requires later source reconciliation.

## News eligibility

An old publication timestamp on a page retrieved now does not establish the
historical headline or article version. Such records are retained as
`retrospective_news_candidates`; they are not historical model context.

To admit a news version into the verified public-context timeline, require:

1. `historical_availability_verified` is exactly `true`.
2. `availability_upper_utc` is a timezone-aware timestamp.
3. `historical_content_sha256` matches an `availability_evidence` item with
   `kind` equal to `archive_snapshot` or `contemporaneous_capture`, a source URL,
   and `captured_at_utc` no later than that upper bound.
4. The fixture link separately declares `historical_link_verified=true`,
   `link_availability_upper_utc`, and `link_availability_evidence` with the same
   required descriptor fields.
5. Both availability upper bounds precede the trade's execution timestamp.

The link gate matters especially for knockout matches: today's final team pair
cannot make earlier team news part of a then-unknown match context. News known
at time *t* is insufficient if its association with that fixture was established
only later.

Evidence descriptors are checked structurally. The function cannot authenticate
an archive or independently prove the upstream verification assertion. A
reviewed archive adapter must fetch, retain, and validate the identified content
bytes before declaring a record verified. A hash without retained source bytes
is insufficient research evidence.

The input news schema is:

```json
{
  "news_id": "source:article:content-version",
  "news_item_id": "source:article",
  "version_rank": 0,
  "fixture_ids": ["espn:fixture-id"],
  "published_at_utc": "2026-06-10T09:00:00Z",
  "captured_at_utc": "2026-09-23T10:00:00Z",
  "source_url": "https://example.org/article",
  "title": "Source headline",
  "historical_availability_verified": false,
  "availability_upper_utc": null,
  "availability_evidence": [],
  "fixture_links": [{
    "fixture_id": "espn:fixture-id",
    "relationship": "direct_match",
    "historical_link_verified": false
  }]
}
```

Use a unique `news_id` per content version. `news_item_id` defaults to `news_id`
and `version_rank` defaults to zero. A verified source revision order is needed
before assigning multiple ranks to one item; capture order does not establish
historical revision order. A nonzero rank also requires
`version_order_historically_verified=true` before the version can enter eligible
context. Duplicate item/rank identities fail rather than
silently replace content. Aggregate multiple fixture links into one news row.
Tournament-wide articles may have `fixture_ids=[]`; this preserves the source
without imposing a weak Cartesian association with every match.

## Separate global public context

Each mapped tournament observation also references `global_context_state_id`
and `global_eligible_context_event_count`. This independent timeline stores each
eligible global news version once. It does not copy links across 104 fixtures
or assert that the story directly concerns the target fixture.

The content must first pass the historical-version evidence gate, then have
`context_scope="tournament"`. Broad World Cup relevance is established either
by the **archived** headline explicitly saying `World Cup` with no explicit
conflicting edition, or by
`historical_tournament_scope_verified=true` with
`tournament_scope_availability_upper_utc` and
`tournament_scope_availability_evidence`. The latter evidence uses the same
capture descriptor requirements. Current article categories or today's title
cannot establish historical scope. The effective timestamp is the later of
content and scope availability bounds. An unmapped condition does not inherit
this tournament-specific timeline.

The shared `is_2026_world_cup_headline` predicate rejects headlines explicitly
naming other years, the Club or Women's World Cup, youth competitions, or other
sports. Known conflicting titles remain excluded even if older metadata claims
verified tournament scope. This conservative rule can reject useful historical
comparisons; manual source-backed review can support a future distinct
background-context policy. Absence of a conflicting title is a relevance
heuristic rather than proof of precise tournament identity.

This is broad public World Cup context: an eligible story can discuss another
team or a general issue. Global availability is neither
personal relevance nor evidence of actor attention. Per-fixture relevance keeps
its stricter independent historical-link gate.

`read_trade_context` includes `verified_global_public_context`. By default it
displays the 20 most recent eligible article versions after selecting each
item's highest eligible rank. `verified_global_public_context_item_count`
reports the total, and `global_public_context_display_truncated` makes display
limits explicit. The full version timeline remains in SQLite. Set
`global_context_limit` to change the inspection limit; this is not a final
training-context selection policy.

The count `retrospective_candidate_count` includes linked, historically
unverified records with a known publication timestamp strictly before the
execution proxy. It excludes unknown-date records and post-execution headlines.
All records remain in `news` for source inspection. This count is an audit
statistic, not an input feature or proof of what was then published.

## Python API

```python
from pathlib import Path
import json

from poly_world_cup.attribution import build_attribution_index, read_trade_context
from poly_world_cup.trades import validate_collection

registry = json.loads(Path("data/registry/registry.json").read_text())
trade_root = Path("data/full/trades")

# Validate before using the exact pages committed by each manifest.
def committed_pages():
    for contract in registry["contracts"]:
        condition = contract["condition_id"]
        manifest = validate_collection(trade_root, condition_id=condition)
        for page in manifest["pages"]:
            yield trade_root / condition / page["file"]

with Path("data/full/news/news.jsonl").open() as stream:
    report = build_attribution_index(
        registry=registry,
        trade_pages=committed_pages(),
        news_records=(json.loads(line) for line in stream),
        output_path=Path("data/full/attribution.sqlite"),
    )

print(json.dumps(report, indent=2))
print(json.dumps(read_trade_context(Path("data/full/attribution.sqlite"), 1), indent=2))
```

Collection should be paused while building a reproducible snapshot. The caller
must supply only audited committed page paths. Page hashing during indexing
records the observed bytes; a bare page iterable does not provide a manifest
against which to validate completeness. A failed rebuild leaves the previous
database unchanged. A successful rebuild atomically replaces it.

A SQL viewer can open the database directly. For example:

```sql
SELECT fixture_id, COUNT(*) AS observed_rows,
       SUM(token_mapping_status = 'unresolved_token') AS unresolved_tokens,
       SUM(eligible_context_event_count > 0) AS rows_with_verified_news
FROM trades
GROUP BY fixture_id;
```

`read_trade_context` returns one observation, eligible source versions, and
separate retrospective news and earlier tournament execution lists. Earlier
wallet execution lookup excludes equal timestamps and the entire target
transaction. That history remains `wallet_history_feature_eligible=false`:
block time does not establish public availability or the actor's order time.
History is `tournament_only`, not the wallet's global trading history. Missing
initial holdings, token transfers, splits, merges, and redemptions are not
silently converted to a position balance.

The database always records `source_completeness_certified=false` and
`training_ready=false`. It is a reproducible attribution and inspection layer;
coverage reconciliation, historical context verification, role identification,
and evaluation gates still precede SFT.

## Scaling and integrity boundaries

Trade rows stream directly into SQLite; Python does not accumulate the trade
corpus in memory. A bounded 8,192-entry timestamp cache reuses the same strict
parser for repeated block times; gzip input uses a 256 KiB binary buffer. News
event lists and their per-fixture availability indexes remain in memory. SQLite is configured with a 32 MiB page cache and disk-backed
temporary operations; process memory also includes Python, SQLite sort buffers,
and news metadata, so 32 MiB is not a process-wide cap. Wallet, fixture, and
observation indexes are built after row ingestion. Global context is indexed
once, rather than multiplied by the number of fixtures.

An atomic rebuild needs space for the old database, the new database, and
temporary indexes. Keep the collection files and raw evidence separately; the
SQLite index is a derived, reproducible inspection artifact. Check disk headroom
before a full rebuild and retain the previous complete snapshot until the new
one is committed. Source completeness and canonical fill reconciliation are
independent of index integrity.

A development integration run with 2,751 current metadata records, 146 verified
archive checkpoint versions, and 110,137 real observations produced a 99.64 MB
index in 9.88 seconds, with roughly 93 MiB peak process RSS. The same news
catalog with 200 observations occupied 15.37 MB. The incremental footprint was
about 767 bytes per observation, including indexes: approximately 7.7 GB for
10 million observations with a similar catalog. This is an extrapolation from
one contract, not a guarantee. Distinct wallets, ID lengths, source counts, and
additional news change storage; larger index sorts and concurrent collection
change runtime. Allow roughly 12 GB of additional free space for a first full
build and its temporary operations. An atomic rebuild also retains the old
index, making the combined old/new/temporary footprint roughly 20 GB at that
scale. The timing excludes upstream collection validation.
