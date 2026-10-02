# Unrealized in-market P&L and interval supervision

Every new actor interval and execution row contains `unrealized_in_market_pnl`.
Every user turn produced by `build_actor_dataset.py prepare` or `sft` includes
the same feature. Basic, In-market and Global variants share this feature;
the latter two still add their separate actor-history summaries.

The field is a decimal string, or `null` when it cannot be computed. Its value
is the sum over YES and NO of:

```text
remaining shares * strictly earlier market price - remaining purchase cost
```

This is **unrealized** P&L before fees, using average purchase cost. Buying adds
shares and their purchase cost. Selling removes shares and their proportional
cost basis; sale proceeds and realized profits do not remain in this feature.
After a complete exit its value is zero. For example, buy 10 shares at 0.40,
then sell 4 at 0.90: with a prior market mark of 0.60 the remaining 6 shares
have unrealized P&L of 1.20, regardless of the realized sale profit.

The first query is zero under the explicit assumption of no holdings before
the supplied execution history. This reconstructs captured trade inventory,
not a verified wallet balance: transfers, splits, merges, redemptions and
uncaptured fills are not inferred. An unmatched sell makes that outcome's
inventory unknown. A missing/stale price for a held outcome also makes the
total `null`. A missing price for an outcome with no holdings does not.
No YES/NO complement or current execution price is substituted.

All same-timestamp executions receive one shared pre-execution feature. Their
updates happen only after the interval and execution records have been emitted.
Mixed buys and sells at one timestamp can make the remaining average cost
ambiguous; the feature stays `null` until the captured position is flat.
The collector retains pre-`--start` executions as first-pair audit history so
late-start exports do not reset inventory. Old exports that explicitly started
late without retaining that history produce `null` opening inventory.

Raw rows include `in_market_pnl_context` with the per-outcome holdings, remaining
cost, marks and missing reasons. Prepared prompts contain only the scalar plus
`in_market_pnl_missing_reasons` when needed. New exports/manifests declare
`in_market_pnl_version: 1`; preparation independently replays and validates the
raw accounting. Older raw exports are recomputed from their saved prior prices
and executions. Missing old price context remains missing.

Inventory traversal takes O(n) arithmetic operations per actor/market with
constant inventory state for two outcomes, excluding price-history lookup and
serialization. Fraction arithmetic avoids intermediate cost-basis rounding;
output is rounded to 40 significant decimal digits.

## Script coverage

| Path | Behavior |
|---|---|
| `scripts/build_actor_dataset.py` | Computes raw features and replays them during SFT preparation; preserves each NO_TRADE/TRADE pair. |
| `scripts/derive_actor_metrics.py` | Includes P&L in raw derived rows and carries it into enriched SFT. |
| `scripts/derive_global_actor_metrics.py` | Keeps current-market P&L alongside separate wallet-wide metrics. |
| `scripts/run_world_cup_experiments.py` | Preserves P&L and interval labels when deriving In-market from prepared Basic conversations; bundles the shared implementation. |
| `scripts/prepare_world_cup_evaluation.py` | Carries P&L and both labels into fresh-market evaluation, including when reference adapters were trained on legacy trade-only data. |
| `scripts/test_actor_activity.py` | Its shared context builder recomputes P&L at the independent prospective query, from all prior inventory even when displayed history is shortened. |
| `scripts/train_world_cup_multigpu.py` | Tokenizes the P&L input and supervises assistant labels only; checks balanced interval/execution counts and interval protocol structure. |
| Evaluation, comparison and loss-plot scripts | Consume the prepared inputs/results without rebuilding or removing model features or labels. |

## No-trade labels

Defaults retain one `{"action":"NO_TRADE"}` answer for the open interval before
each distinct execution timestamp, followed by a `TRADE` answer at that endpoint.
The initial interval is included. Tied executions form one TRADE answer. There
is no invented trailing interval after the last execution. Enrichment does not
advance inventory on NO_TRADE answers, so a pair has identical P&L.

The trainer rejects missing or unbalanced interval supervision by default and
validates pair structure rather than relying on overall counts alone. Explicit
legacy `--trade-only` / `--allow-trade-only` options remain available for old
experiment reproduction. Fresh-market evaluation always retains the gaps.
These are retrospective observed gaps, not evidence of deliberate abstention;
see [no-trade supervision](no_trade_supervision.md).

## Updating existing datasets

Pull the code, then prepare a **new** Basic dataset from the saved actor exports:

```bash
git pull --ff-only origin main
python3 scripts/build_actor_dataset.py prepare \
  --input-root /path/to/saved/actor_exports \
  --split-file /path/to/existing/split_plan.json \
  --out /path/to/new/basic_with_pnl
```

This preparation is offline. Derive In-market and Global again using the new
Basic dataset, or upload it to the automated runner for offline In-market
derivation. Reusing saved exports preserves actors and trade history. Old JSONL
files, token caches and trained adapters do not acquire the new feature merely
by pulling code; use the regenerated dataset for a new training run.
