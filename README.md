# World Cup actor-decision dataset

Collection and validation foundations for studying public-wallet trading in
**all 104 matches of the 2026 FIFA World Cup** on Polymarket.

The research question is whether information observable before a query time
helps predict a wallet's subsequent **observed execution**. Executed fills are
not direct observations of private beliefs or order-submission decisions.

## Actor conversations, version 3

**The complete v3 export is available; final independent validation is pending.**
It covers all 104 fixtures and 312 binary contracts and contains **1,675,195
chronological actor conversations** built from **3,974,200 selected API
observations for 255,438 wallets**. These are export counts awaiting full
source, context, label and token reconciliation. See the
[release status](reports/actor_sequences_release_status.json).

Each JSONL row contains multiple context/answer turns. The model can attend to
earlier turns in that row; adjacent rows do not create persistent actor memory.
News and market context use compact updates, and each new conversation chunk
supplies bounded earlier history and numerical activity summaries.

The `conditional_trades` profile predicts observed execution details. The
`scheduled_windows` profile predicts captured observations over fixed future
15-minute windows, including `NO_TRADE`. That label means no captured observation
in the monitored contracts, not independently verified on-chain inactivity.
The filter retains actor–binary-contract pairs with **at most 20** observations
over the captured period. It is retrospective and differs from v2's
tournament-wide fewer-than-20 rule.

| Export measurement | Value, pending independent validation |
| --- | ---: |
| Conditional conversations | 750,420 |
| Scheduled conversations | 924,775 |
| Reference tokens across both task views | 8,283,490,246 |
| Largest conversation / conversations above the 8,192-token target | 10,301 / 872 |
| Fixtures with retained observations in each task profile | 104 |

[Dataset files](datasets/world_cup_2026_actor_sequences_v3/) ·
[Dataset format and inspection](docs/actor_sequence_dataset.md) ·
[Training and messages-only loading](docs/actor_sequence_training.md) ·
[Release manifest](datasets/world_cup_2026_actor_sequences_v3/manifest.json)

The training loader requires a matching passed full validation report, which is
not yet published. The export can be inspected now. Coverage of all 104 fixtures
does not certify that every historical execution was captured. No model has
been trained, and no predictive-accuracy result is claimed. Earlier dataset
releases remain unchanged.

## Previous prepared release, version 2

[All 104 matches](datasets/world_cup_2026_tournament_lt20_v2/),
with **961,023 targets per profile**. The filtered cohort contains
**208,296 wallets**. All **312 contract API histories are exhausted**, and all **104 fixtures have
exported examples with verified prior direct-match news**. The tournament-wide
filter retains wallets with 1–19 captured observations; original datasets and
version 1 are unchanged.

The two alternative conversational JSONL profiles contain verified historical
news and initial contract semantics, with or without earlier execution history.
They include fixed train/validation/test splits and a full per-observation audit.
**773,253 targets (80.46%)** have prior direct-match news; earlier targets
remain empty where no direct headline was yet verifiable. This is ready for
retrospective conditional-observation SFT within the documented capture scope.

[Inspect and load](docs/tournament_sft_v2.md) ·
[Readable examples](reports/sft_v2_preview.json) ·
[All-104 completion check](reports/tournament_sft_v2_completion.json) ·
[Full validation](reports/sft_v2_validation.json)

## What is implemented

- Discover the complete fixture universe from ESPN and map the corresponding
  Polymarket match-result contracts, with explicit missing/ambiguous statuses.
- Preserve actual Yes/No token mappings and contract rules. Include pregame and
  in-play observations; props, exact scores, halftime, and tournament outrights
  are outside this initial contract scope.
- Save immutable raw HTTP bodies, SHA-256 hashes, request parameters, and
  capture timestamps. Preserve decimal prices and sizes without a float round trip.
- Collect wallet-side trade observations using resumable Data API v2 cursors,
  identical filters on every page, atomic writes, and checked crash recovery.
- Traverse all mapped contracts concurrently with bounded request rates,
  compressed source captures, and restartable progress checkpoints.
- Verify historically captured ESPN and GDELT headlines, attribute direct fixture
  news across all 104 matches, and build a queryable SQLite context index.
- Recover all 312 initial contract questions and Yes/No token mappings from
  historical Polygon evidence, gated before each target and prior execution.
- Export two conversational SFT profiles with per-row context, label, provenance,
  split and activity-filter validation; preserve earlier releases unchanged.
- Check a bounded fixture-stratified sample against Polygon transaction receipts,
  retaining ambiguous logs and amount discrepancies.
- Guard historical availability, versions, wallet eligibility, coverage-aware
  labels, target-transaction exclusion, and global-time/fixture-disjoint splits.
- Report coverage and remaining limits in machine-readable completion and
  validation reports.

**Archived checkpoint — superseded by version 2:** 11,858,584 wallet-side observations
from 252,612 wallets across all 104 fixtures. Of 312 contract histories, 293
reached the API's end and 19 were paused after outbound source requests began
failing with proxy tunnel `403 Forbidden`. All 12,007 saved pages passed raw-source
normalization replay, and their hashes/counts match the SQLite index.

The news catalog contains 2,751 current metadata records and 1,891 separately
verified archived headlines. All captured observations have some verified prior
broad tournament context; only 9,175 have verified prior fixture-specific news.
These are distinct evidence levels, and neither establishes actor exposure.
Archive recovery was unfinished in that checkpoint. The archived version 1 SFT release used this partial
snapshot explicitly; it does not certify complete on-chain history or prospective
decision context.

See [checkpoint coverage](reports/collection_checkpoint.json),
[attribution counts](reports/attribution_checkpoint.json),
[raw provenance verification](reports/provenance_checkpoint.json), and
[archive coverage](reports/news_archive_checkpoint.json).
Open [one context example](reports/context_preview.json) to inspect the observation,
prior tournament wallet executions, and temporally gated news fields directly.
For a separate subset retaining wallet-market pairs with fewer than 20 saved
observations, use the [wallet activity filter](docs/wallet_activity_filter.md).
Prepared checkpoint archives are `world_cup_partial_corpus.tar.gz` and
`trade_partial_provenance.tar.gz`; [access and resume instructions](docs/dataset_access.md)
describe the formats. Downloads are split into five `world_cup_data.part*` files
and four `trade_sources.part*` files to meet the per-file download limit.
[Release checksums](reports/release_checkpoint.json) cover every part and both
reassembled archives. Large-file upload completion is unconfirmed after a
timeout; the prepared files are not yet a verified published download.
`--allow-partial` is required to export such a checkpoint;
the normal complete-release checks remain strict.

**Initial live validation:** 104/104 fixtures mapped; 312 binary result contracts;
624 tokens. Two kickoff discrepancies are flagged. A bounded trade smoke run
collected 200 observations across two pages for one condition. This validates
the collector, not the completeness of tournament trading history.

See [the live report](reports/live_validation.json),
[the full fixture/contract index](reports/fixture_index.json), and
[the scientific methodology](docs/methodology.md).

## Run

Python 3.11+ on Linux/macOS. Collection uses the standard library and requires
public internet access. Actor-sequence token measurement additionally uses the
pinned optional tokenizer dependencies in the [training guide](docs/actor_sequence_training.md).
Most unit tests are offline; real-tokenizer checks need the local reference
files. Trade writer locking currently uses POSIX `fcntl`.

```bash
git clone https://github.com/parthchvn/poly_world_cup.git
cd poly_world_cup
python -m unittest discover -s tests -v

# Discover every fixture and its match-result contracts.
python -m poly_world_cup discover

# Bounded smoke run: one of the discovered conditions, two pages.
python -m poly_world_cup ingest \
  --registry data/registry/registry.json \
  --max-conditions 1 --limit 100 --max-pages 2 \
  --output data/smoke/trades --cache data/smoke/cache

python -m poly_world_cup audit \
  --registry data/registry/registry.json \
  --trades-root data/smoke/trades
```

The discovery command writes `registry.json`, `fixtures.jsonl`,
`contracts.jsonl`, and `discovery_report.json` under `data/registry/`.
It returns exit code 2 for incomplete mapping while retaining the report.
Audit returns 2 for structural mapping errors. A structurally valid audit
still has `sft_ready: false`; check that field and its blockers.

Audit also verifies each collection's committed page hashes, counts,
timestamps, filters, and cursor chain. Missing or corrupt pages cause exit 2
and are excluded from observation totals. `conditions_with_manifests` counts
files found; `conditions_with_verified_manifests` counts collections that pass
these integrity checks. This does not certify the source's historical coverage.
See the [follow-up integrity review](reports/integrity_validation.json).

For API traversal over **every mapped condition**, use the batch collector:

```bash
python -m poly_world_cup collect-tournament \
  --workers 12 --requests-per-second 4 --minimum-size 0.000001
```

Repeat the same command to resume. Keep the same page limit and filters for a
run. `--max-pages` limits *additional* pages per condition in each invocation.
The progress file is `data/full/trades/batch_progress.json`. An exhausted
traversal stays exhausted; use new output **and cache** directories for an
independent later capture. The batch runner creates a cache per condition and
prevents concurrent writers to the same run.

Collect and verify news, then build the context index:

```bash
python -m poly_world_cup collect-news \
  --window-start 2026-01-01T00:00:00Z --window-end 2026-07-20T23:59:59Z
python -m poly_world_cup archive-news
python -m poly_world_cup reconcile-sample
python -m poly_world_cup verify-provenance
python -m poly_world_cup attribute \
  --archive-news data/news_archive/news_archive.jsonl
python -m poly_world_cup inspect-context --row 1
```

The attribution command audits every committed trade page and requires all
312 traversals to be exhausted. `--allow-partial` explicitly permits an
incomplete preview. The index joins observations to fixtures and wallet
prefixes while keeping verified historical news separate from retrospective
candidate links. See [attribution semantics](docs/attribution.md),
[news sources](docs/news_sources.md), [archive verification](docs/news_archive.md),
and [receipt checks](docs/reconciliation.md).
For downloaded data, follow [opening and querying the corpus](docs/dataset_access.md).

Raw responses, observations, and local caches live under ignored `data/`.
The prepared release under `datasets/` includes the requested filtered wallet
observations, news metadata, and chat examples. Original source captures remain
outside Git; reports preserve their coverage summaries, hashes, and timestamps.

## Limits of the original research target

| Issue | Implemented behavior / required next evidence |
| --- | --- |
| Current fixture/news metadata can leak future information | Registry is retrospective and `feature_eligible=false`; historical versions are required. |
| On-chain time differs from decision time | Preserve block timestamps; do not invent order placement or matching times. |
| Trade-only rows cannot learn whether someone trades | Separate conditional action prediction from checkpoint occurrence prediction. |
| Missing observations can look like inactivity | Scheduled `NO_TRADE` labels describe the declared captured scope only; uncertain or missing source coverage does not establish real inactivity. |
| API pagination is not archive certification | Every collection manifest has `training_coverage_certified=false`. |
| Final participants can contaminate historical cohorts | The prepared activity filter is retrospective; a prospective experiment needs eligibility fixed from historical information. |
| Fixture markets trade concurrently | Require both global time boundaries and disjoint fixture IDs. |
| Fills do not reveal beliefs | Source-grounded background first; inferred beliefs are an optional ablation. |

Data API collection explicitly uses `taker_only=false` and
`filter_type=TOKENS`. The initial smoke run requested `filter_amount=0.01`;
the tournament collector requests `0.000001`. Each manifest fixes the actual
threshold, and a resume rejects changed filters. A smaller accepted request
does not prove the provider has no hidden floor or coverage gaps. Consequently
these remain observations from a **filtered** API. Rows lack canonical log/order IDs and explicit
maker/taker roles. Distinct rows that look identical are retained. Grouped
targets collect observations by timestamp or forecast window; they do not
identify execution bundles or inventory balances. Scheduled empty windows
describe the captured observations and do not certify complete market inactivity.

## Next steps, in order

1. **Reconcile execution data:** extend bounded receipt checks to canonical
   event coverage across all mapped conditions, inspect the
   existing `trades.parquet` schema if available, and validate canonical
   event identity, wallet sides, exchange versions, amounts, and archive gaps.
2. **Reconstruct historical context:** global prior wallet activity,
   position-changing events, dated market state, and versioned news. Resolve
   kickoff discrepancies and unknown opening-time evidence.
3. **Use the declared prediction task:** the [actor conversations](docs/actor_sequence_dataset.md)
   separate conditional execution prediction from scheduled observation windows.
   Upgrade their evidence claims only as canonical execution and historical
   availability evidence improves.
4. **Establish baselines, then SFT:** evaluate history-only and market-only
   models, add sourced news, and test whether an LLM improves held-out results.
5. **Interpret the model:** perform controlled interventions only after
   predictive validity and leakage checks pass.

See [data contracts and API caveats](docs/data_contracts.md) for exact field
semantics. No training run or empirical forecasting claim is made by this
collection and attribution implementation.
