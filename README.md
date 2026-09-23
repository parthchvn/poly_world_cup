# World Cup actor-decision dataset

Collection and validation foundations for studying public-wallet trading in
**all 104 matches of the 2026 FIFA World Cup** on Polymarket.

The research question is whether information observable before a query time
helps predict a wallet's subsequent **observed execution**. Executed fills are
not direct observations of private beliefs or order-submission decisions.

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
- Guard historical availability, versions, wallet eligibility, coverage-aware
  labels, target-transaction exclusion, and global-time/fixture-disjoint splits.
- Report exactly what remains missing before SFT.

**Live validation:** 104/104 fixtures mapped; 312 binary result contracts;
624 tokens. Two kickoff discrepancies are flagged. A bounded trade smoke run
collected 200 observations across two pages for one condition. This validates
the collector, not the completeness of tournament trading history.

See [the live report](reports/live_validation.json),
[the full fixture/contract index](reports/fixture_index.json), and
[the scientific methodology](docs/methodology.md).

## Run

Python 3.11+ on Linux/macOS. Runtime code uses the standard library. Collection
requires public internet access; the unit tests are offline. Trade writer
locking currently uses POSIX `fcntl`.

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

For API traversal over **every mapped condition**, use a separate run directory:

```bash
python -m poly_world_cup ingest \
  --registry data/registry/registry.json --all-pages --limit 1000 \
  --output data/full/trades --cache data/full/cache
```

Repeat the same command to resume. Keep the same page limit and filters for a
run. `--max-pages` limits *additional* pages per condition in each invocation.
An exhausted traversal stays exhausted; use a new output **and cache** directory
for an independent later capture. Use one process per cache directory.
Full traversal was **not** performed in the initial commit.

Raw responses, observations, and local caches live under ignored `data/`.
The committed reports contain identifiers, coverage summaries, hashes, and
timestamps; wallet-level data and profile fields are not committed.

## Why this is not yet an SFT dataset

| Issue | Implemented behavior / required next evidence |
| --- | --- |
| Current fixture/news metadata can leak future information | Registry is retrospective and `feature_eligible=false`; historical versions are required. |
| On-chain time differs from decision time | Preserve block timestamps; do not invent order placement or matching times. |
| Trade-only rows cannot learn whether someone trades | Separate conditional action prediction from checkpoint occurrence prediction. |
| Missing observations can look like inactivity | Incomplete coverage produces `CENSORED`, never a negative label. |
| API pagination is not archive certification | Every collection manifest has `training_coverage_certified=false`. |
| Final participants can contaminate historical cohorts | Build wallet eligibility from a historical global or explicitly restricted World Cup prefix. |
| Fixture markets trade concurrently | Require both global time boundaries and disjoint fixture IDs. |
| Fills do not reveal beliefs | Source-grounded background first; inferred beliefs are an optional ablation. |

Data API collection explicitly uses `taker_only=false` and
`filter_type=TOKENS, filter_amount=0.01`. Consequently these are observations
from a **filtered** API. Rows lack canonical log/order IDs and explicit
maker/taker roles. Distinct rows that look identical are retained. No execution
bundles, inventory balances, or complete negative windows are inferred here.

## Next steps, in order

1. **Reconcile execution data:** collect all mapped conditions, inspect the
   existing `trades.parquet` schema if available, and validate canonical
   event identity, wallet sides, exchange versions, amounts, and archive gaps.
2. **Reconstruct historical context:** global prior wallet activity,
   position-changing events, dated market state, and versioned news. Resolve
   kickoff discrepancies and unknown opening-time evidence.
3. **Build examples:** first implement the explicitly defined execution target
   in [the study plan](docs/study_plan.md); apply temporal guards, preserve
   censoring/missingness, and freeze splits before model fitting.
4. **Establish baselines, then SFT:** evaluate history-only and market-only
   models, add sourced news, and test whether an LLM improves held-out results.
5. **Interpret the model:** perform controlled interventions only after
   predictive validity and leakage checks pass.

See [data contracts and API caveats](docs/data_contracts.md) for exact field
semantics. No training run or empirical forecasting claim is made by this
first-stage implementation.
