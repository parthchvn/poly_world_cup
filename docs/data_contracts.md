# Data contracts and source evidence

## Layers

| Layer | Current output | Interpretation |
| --- | --- | --- |
| Source capture | `data/cache/bodies/<sha256>.json` | Exact bytes returned by a public endpoint |
| Capture provenance | `captures/<hash>.json` | Request URL, retrieval timestamp, selected response headers, body hash |
| Cache index | `requests/<url-hash>.json` | Latest locally captured response for this exact canonical URL |
| Fixture registry | `data/registry/fixtures.jsonl` | All 104 fixture identities, mappings, current schedule claims, uncertainty flags |
| Contract registry | `data/registry/contracts.jsonl` | Actual condition IDs, outcome-token pairing, exact rules text |
| Observed fills | `data/trades/<condition>/pages/*.jsonl` | API observations with provenance; not canonical executions |
| Ingestion state | `data/trades/<condition>/manifest.json` | Durable cursor chain, hashes, row counts, traversal state, limitations |
| Audit | `data/audit.json` | Registry structure, local ingestion progress, remaining SFT blockers |
| Historical features / labels | Not implemented | Require the evidence gates below |

`schema_version` is included in source-derived records. Raw body hashes are
provenance references; a hash alone does not make source bytes available.
Archive the ignored raw cache with each research dataset release. The initial
GitHub commit publishes compact audit snapshots, not the full source archive.

## Fixture and contract mapping

The current universe is 2026 only. ESPN's full-year scoreboard is fetched with
`dates=2026&limit=1000`. This is a public provider endpoint, not a guaranteed
archival service or an official FIFA fixture registry. A changed schema or a
count other than 104 prevents a successful complete-discovery report.

Gamma series `11433` (`soccer-fifwc`) is paginated with page size 100, ascending
event ID, and offsets advanced by actual returned row count until an empty
page. Duplicate event IDs or a finite page-limit breach fail discovery. This
is not a transactionally consistent snapshot of a changing database.

The currently audited primary event slug form is
`fifwc-<team>-<team>-2026-MM-DD`. Exact-score, halftime, and other suffix events
are excluded. Its date is not necessarily the UTC kickoff date. Matching uses
both canonical team names and source kickoff timestamps within three hours;
every nonzero time difference is warned. There is no fuzzy name matching.

The parser supports the **observed 2026 schema** of three binary moneyline
contracts: each team and draw. It checks the actual Yes/No outcomes and token
pairing. A different structure is flagged for review; the code never creates
fictional missing contracts. Read the rules separately for extra time,
postponement, and cancellation semantics. Current observed rules explicitly
refer to regulation time plus stoppage time.

`clobTokenIds`, `positionIds`, market IDs, condition IDs, and event IDs are
distinct identifiers. Do not interchange them. A late token mapping is not
proof of token identity throughout a migration. Historical fills require
independent reconciliation by their actual observed IDs and exchange version.

`accepting_orders_at`, `created_at`, and `start_date` are separate API claims.
An API creation timestamp does not establish when executable trading began.
Every fixture and contract record has `feature_eligible=false` and
`retrospective_metadata=true`. Final knockout opponents and actual schedules
must not enter a historical prompt until their then-available version exists.

## Wallet-side observation fields

| Field | Meaning |
| --- | --- |
| `observation_id` | Hash of request URL, raw body hash, and row index |
| `proxy_wallet` | Public wallet identifier returned by the provider; not a verified person |
| `condition_id`, `token_id` | The condition and token actually named in the row |
| `side` | Provider wallet-side `BUY` or `SELL` |
| `size`, `price` | Finite decimal strings; size > 0 and price in [0, 1] |
| `block_timestamp_seconds`, `block_timestamp` | Provider block time, original second precision and UTC string |
| `transaction_hash` | Transaction association, not a unique fill or decision ID |
| `maker_taker_role`, `order_id`, `log_index` | Unknown (`null`) for this API adapter |
| `publicly_available_at_upper_bound` | Unknown (`null`); capture now does not prove historical observability |
| `source` | Original URL, body hash, row index, capture timestamp |
| `quality_flags` | Explicit identity, timing, role, and observability limitations |

The same raw log can generate legitimate wallet-side rows. Identical API rows
can also represent different fills. This collector preserves row multiplicity;
it does not deduplicate by transaction hash or row contents. Observation IDs
cannot reconcile independent captures. Later chain reconciliation must retain
the distinction between raw-event and wallet-side identity.

The HTTP client parses fractional JSON numbers as `Decimal`; normalization
emits decimal strings. Exact original numeric spelling is retained in raw
bytes. Never pass monetary amounts through a binary float transformation.

## Trade API limits verified for this implementation

- V2 returns `{data, pagination}`, using opaque cursors.
- Cursor requests must repeat the condition, page limit, and all filters.
- `taker_only=false` requests both sides but does not add an explicit role field.
- The chosen `TOKENS` threshold is 0.01. The documented default is 0.01, and
  setting zero still selects the default. Smaller positive thresholds were
  not validated in this implementation.
- `start` and `end` apply to wallet queries, not condition/event queries. The
  condition history shape uses a fixed three-year window. The unfiltered
  global feed has a much shorter window and is not a substitute for a full
  historical wallet universe.
- `timestamp` is a block timestamp. The shown schema does not expose the
  canonical log/sequence identifier used internally for pagination.

Consequently `api_traversal_status=exhausted` describes only reaching the end
of this query. It does not certify all fills, zero activity, role completeness,
or eligibility for a training label. `training_coverage_certified` stays false.

Committed page files and metadata precede the manifest update. A subsequent
run verifies hashes and cursor continuity, recovers an interrupted page commit,
and rejects changed filters or inconsistent state. A failed request leaves
already committed pages intact. One POSIX writer is allowed per condition.

`validate_collection(output_dir, condition_id=...)` performs read-only local
verification with the same page and manifest checks used on resume. The audit
uses this verifier rather than trusting manifest totals. It validates the
supported query filters independently of the manifest's own assertions.
Missing pages, bad checksums, inconsistent summaries, and malformed manifests
are reported in `invalid_collections` and cause CLI exit 2. Only verified
collections contribute to observation counts; identical legitimate rows are
still preserved. Uncommitted recovery pages are not counted. Malformed
condition IDs are rejected before path lookup, and case-equivalent condition
IDs cannot cause a collection to be counted twice.

Local integrity and source coverage remain separate. The verifier checks
normalized files against their committed journal. It does not authenticate the
upstream source, verify raw HTTP bodies, prove absence of missing executions,
or promote a collection to `training_coverage_certified=true`.

## Prerequisites for a training row

For a row `(context B, observed target A)`, persist at least:

- Stable row, fixture, condition, token, wallet, query, and target identifiers.
- Task definition, target role, checkpoint horizon, label precision, and
  target execution identity evidence.
- Every feature's source/version, verified availability bound, missingness,
  and context serialization hash.
- Label-coverage evidence scoped to wallet, fixture, interval, and role.
- Historical universe procedure/source, eligibility, selection probability,
  and any sampling weight.
- Split assignment and fitted-transform version.

Never fabricate an explanation, belief label, missing balance, or probability
target. Numeric execution price is a realized outcome; it is not an order
limit price. A BUY of a No token remains that actual instrument and direction;
it must not silently become a SELL of Yes.

## Primary references

These are provider sources used for implementation, not guarantees of archive
completeness. Capture hashes and timestamps are in the live validation report.

- [Polymarket Data API v2 OpenAPI schema](https://data-api.polymarket.com/v2/openapi.json)
- [Polymarket Data API v2 documentation](https://data-api.polymarket.com/v2/docs)
- [Gamma World Cup event feed](https://gamma-api.polymarket.com/events?series_id=11433&limit=100&order=id&ascending=true)
- [ESPN 2026 World Cup scoreboard](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=2026&limit=1000)
- [Polymarket CTF Exchange v2 source](https://github.com/Polymarket/ctf-exchange-v2)

The exchange code is a starting point for the next reconciliation stage; this
commit does not implement its event decoding or certify a migration boundary.
