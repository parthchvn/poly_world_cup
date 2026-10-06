# Overnight interval activity experiment

## Upload a whole-actor subset from Windows

`scripts/package_actor_subset.py` packages exactly 200,000 **raw actor history
rows**, including the original paired NO_TRADE rows. It keeps each selected
actor's entire captured history in a market, excludes actors with more than
20 captured executions in that market, and never truncates a history to fit.
This trade-count filter is a cohort rule, not proof that a wallet is human.
The selection uses a fixed seed and fails if the exact total is impossible.

In PowerShell, from this repository checkout:

```powershell
py -u scripts/package_actor_subset.py --input-root "$env:USERPROFILE\wc_collection_v1\exports" --out "$env:USERPROFILE\wc_collection_v1\wc_200k_by_actor_v2.zip" --target-rows 200000 --max-trades-per-actor 20 --seed 42
```

The ZIP contains `exports/market_*/` with complete selected actor files, rebuilt
actor indexes and subset manifest counts, plus shared ESPN news, official price
histories and their provenance. Original manifests are kept as
`source_manifest.json`; `selection.json` records the selected histories and
checksums. Existing archives are never overwritten. No API calls or third-party
Python packages are needed. Extract the ZIP on RunPod and use its `exports/`
folder as `--input-root` for the interval runner below.

**This raw-row budget is different from `run_interval_experiment.py --max-rows`:**
the latter caps newly constructed scheduled interval examples. Packaging 200,000
raw rows does not guarantee 200,000 five-minute examples. Supporting news/price
files and actor-index entries do not count toward the raw actor-row budget.

## Price and share-size tolerances

Use `--target-mode trade-details` to predict TRADE/NO_TRADE **plus BUY/SELL,
YES/NO, total shares, and weighted mean execution price** inside each interval.
The original binary `activity` mode remains available and is still the default.
Existing binary adapters do not gain numeric outputs by changing evaluation.
Prepare a new dataset and train a fresh adapter for the detailed task.

There are three independent choices: the prediction horizon, the absolute price
error tolerance, and the relative share-size error tolerance. The numeric values
below are **illustrative experimental settings**, not validated optima:

```bash
git pull --ff-only origin main

python3 scripts/run_interval_experiment.py \
  --input-root /workspace/world_cup_actor_data/data \
  --model /workspace/models/Qwen3.6-27B \
  --out /workspace/experiments/wc_trade_details_v1 \
  --target-mode trade-details \
  --window-seconds 300 \
  --price-delta 0.02 \
  --shares-relative-delta 0.20 \
  --max-rows 200000 \
  --gpus 2 --gpu-ids 0,1 \
  --install-xgb --background
```

The details mode requires explicit price and relative-share tolerances. There
are no hidden numeric defaults. The optional `--shares-absolute-delta` defaults
to zero and supplies a share-error floor for tiny trades. These choices are
saved in the dataset manifest, bound to the training run, and cannot be changed
by the evaluator. Tune settings on development data and freeze them before the
final experiment; do not widen tolerances after inspecting test results.

For each side/outcome combination, a prediction is accepted when both hold:

```text
abs(predicted_price - observed_price) <= price_delta
abs(predicted_shares - observed_shares)
    <= max(shares_absolute_delta, shares_relative_delta * observed_shares)
```

The endpoints are inclusive and decimal arithmetic is used. `price_delta=0.02`
means two cents per share, or two probability percentage points. It does not mean
two percent of the price. For an observed trade aggregate of 100 shares at 0.60,
the example settings accept a price estimate in [0.58,0.62] and a share estimate
in [80,120], provided side and outcome also match. The relative share tolerance
uses the **observed** size, not the prediction, as its denominator.

These are tolerance criteria around **point predictions**, not confidence bands
with a claimed coverage probability. SFT labels retain the observed aggregate
values; changing a tolerance does not change the labels or input features.
The model is not trained to output an arbitrarily wide interval. A predicted
price change relative to the current market price would be a different target.

Multiple fills are aggregated separately by `(side,outcome)` over the window.
For example, BUY YES 10 shares at 0.40 and BUY YES 30 shares at 0.60 yield:

```json
{"action":"TRADE","trades":[{"side":"BUY","outcome":"Yes","price":"0.55","shares":"40"}]}
```

Price is share-weighted mean execution price, and shares is the total executed
quantity. BUY/SELL are not netted against each other. There are at most four
groups. This prevents exchange fill fragmentation from changing the target;
group sums and weighted means are serialized deterministically to 40 significant
digits. NO_TRADE remains `{"action":"NO_TRADE"}`.

For detailed mode, SFT **keeps all derived features by default**. XGBoost still
ranks TRADE/NO_TRADE prediction, so its shortlist cannot establish which features
help with price or size. You can explicitly request `--sft-features selected`,
but that is an activity-based ablation. XGBoost price/size regression is not
implemented by this runner.

Tomorrow use the separate detailed evaluator:

```bash
python3 scripts/evaluate_interval_trade_details.py \
  --dataset-dir /workspace/experiments/wc_trade_details_v1/sft/selected \
  --run-dir /workspace/experiments/wc_trade_details_v1/runs/selected \
  --model /workspace/models/Qwen3.6-27B \
  --out /workspace/experiments/wc_trade_details_v1/evaluation/selected \
  --gpu 0
```

Decoding is deterministic. The full user context stays before the interval;
no true future side, outcome, price or size is passed to generation. The evaluator
checks the exact training prompt prefix, frozen manifest, test file and tolerance
scorer. Invalid or unterminated JSON fails the interval. `--check-only` validates
tokenization first; `--max-context` includes the generation reserve controlled by
`--max-new-tokens`. Increase the total context budget if valid training prompts
plus that reserve do not fit; input history is never silently truncated.

Reports include action accuracy, side/outcome-group precision/recall/F1 with both
numeric tolerances, whole-interval success, and success on **TRADE windows alone**.
An always-NO_TRADE baseline and actor/match summaries are included. Correct
negative windows cannot hide failure on every positive window. Predicted groups
are also aggregated before scoring: splitting a prediction into pieces does not
change its total, and duplicating quantity can make its size incorrect.

The following sections describe the original binary activity workflow and the
source preparation shared by both modes.

This pipeline trains **whether an actor executes at least one trade in a future
interval**, with targets `{"action":"TRADE"}` and `{"action":"NO_TRADE"}`. It is a
new task aligned with interval testing. It does not predict the future trade's
side, outcome, price, or size; those attributes remain in the earlier history.
The existing execution-attribute training workflow remains available separately.

## Tonight: one command

Use the existing working Qwen3.5/3.6 QLoRA environment, local model, and **raw
completed actor exports**. The input directory should contain market export
directories with `manifest.json`, `market.json`, `actors/`, `espn_events.jsonl`,
and `market_price_history.jsonl`. Prepared SFT-only files cannot reconstruct
independently scheduled negative windows. Reuse your saved captures; no API
requests are made by the new runner.

From your repository checkout, replace the raw export and model paths if needed:

```bash
git pull --ff-only origin main

python3 scripts/run_interval_experiment.py \
  --input-root /workspace/world_cup_actor_data/data \
  --model /workspace/models/Qwen3.6-27B \
  --out /workspace/experiments/wc_intervals_v1 \
  --max-rows 200000 \
  --window-seconds 300 \
  --gpus 2 --gpu-ids 0,1 \
  --install-xgb --background
```

`--install-xgb` installs missing CPU XGBoost/numpy dependencies without upgrading
working packages. SFT dependencies are checked before preparation. The optional
`--install-deps` uses the repository's existing pinned installer, which requires
the PyTorch 2.8.0 / CUDA 12.8 image. It is unnecessary in the working training
environment. GPU smoke validation runs after ten optimizer steps, then the same
run continues. Finite losses are checked; this is not a claim of model quality.

The command prints a background PID and log path. Inspect:

```bash
cat /workspace/experiments/wc_intervals_v1/status.json
tail -f /workspace/experiments/wc_intervals_v1/logs/train_selected.log
```

During earlier stages, follow the printed `logs/runner-*.log` or `logs/xgboost.log`.
Live SFT losses are saved to `runs/selected/losses.csv` and `metrics.jsonl`.

The runner performs preparation, XGBoost ranking, feature projection, and **one
SFT run**. It never launches test evaluation. Add `--compare-basic` to train a
second independent adapter using the common context without the numeric summary
block. Both adapters start from the same base model with matched settings.

Use `--prepare-only` for CPU-only data generation; `--rank-only` also fits XGBoost
and writes the final SFT input. Rerun the original command with `--resume` to reuse
verified stages and the latest complete training checkpoint. To continue a
prepare/rank-only run, remove that stop flag and add `--resume`. Keep the other
settings and checkout unchanged. Interrupted XGBoost fits restart safely. A run
without a complete training checkpoint is preserved before restarting training.
Changed source inventory, code, settings, or output checksums require a new run.

## What exactly is one row?

For a 17:00 kickoff and the default five-minute horizon, eligible windows are
`[17:00,17:05)`, `[17:05,17:10)`, and so on. At 17:05 the context uses only
observations strictly before 17:05. A captured execution at 17:05 belongs to
the second window; one at 17:10 belongs to the third. The same boundaries and
input cutoff apply during training and testing.

Defaults cover the fixed period from kickoff to kickoff + 150 minutes. Set
`--pre-match-minutes 60` to start one hour earlier, or change `--match-minutes`;
the total scheduled duration must be divisible by `--window-seconds`. Choose
these before training, and freeze them for testing. The schedule is independent
of trade times and the match's eventual duration.

An actor becomes eligible strictly after its first captured execution in that
binary market. This gives a deployable enrollment rule using existing history;
the task does not include first trades or never-observed actors. Negative
windows can continue after the actor's final trade, provided source coverage
extends through the scheduled window end. The existing at-most-20 captured
executions per actor/market cohort is retained by default.

`--max-rows 200000` caps the **total number of interval examples across all three
splits**, not the number of source trades or the training-only count. If there
are more eligible windows, deterministic hash sampling selects a uniform subset
without looking at labels. If fewer exist, all are used; examples are not
duplicated. Natural TRADE/NO_TRADE prevalence is retained. Training must contain
both classes; no label-dependent resampling is used to manufacture balance.

All markets for one match remain in one split. Default splits allocate roughly
80/10/10 of chronological kickoff batches, with simultaneous matches together.
Windows crossing the next split's earliest scheduled start are purged **before
sampling**, giving strictly ordered prediction periods. At least three distinct
kickoff batches with usable windows are required. An existing fixture split
mapping can be supplied with `--split-file`; chronology is still enforced.

## Features and XGBoost

The common SFT context includes a stable wallet pseudonym (`--no-include-actor-id`
omits it), market question, earlier sampled prices and ages, unrealized PnL,
recent news text, and earlier trades. No numeric ordering of wallet IDs is fed
to XGBoost. The raw identifier remains metadata for audits and actor evaluation.

By default the prompt shows the last 8 execution groups and up to 20 news events
from the last 20 minutes, each truncated to 300 characters. Change these with
`--history-groups`, `--max-news-items`, `--news-seconds`, and `--max-news-chars`.
Omitted history/news counts are explicit. Numeric summaries use the full eligible
prefix even when displayed history is shorter. The tokenizer fails on oversized
examples rather than silently truncating their labels.

The candidate table contains 81 numeric or unavailable values:

- Current remaining holdings, cost basis, unrealized PnL, and return on remaining cost.
- Prior trade counts, BUY/YES fractions, notional summaries, cash flow, time since
  last trade, earlier interexecution variability, and recent activity counts.
- Strictly earlier YES/NO price levels, freshness, changes, sampled price-change
  variability, and elapsed time since kickoff.
- Recent news counts and time since earlier goals, cards, shots, substitutions,
  fouls, corners, delays, and phase events.
- The existing completed-position and risk metrics when valid optional historical
  accounting inputs are supplied.

`--closed-positions PATH` supplies completed-position accounting.
`--returns-file PATH` supplies regular capital-flow-adjusted historical returns;
Sharpe/Sortino/return volatility require at least 30 eligible periods by default.
Both period end and availability time must precede the query. These values are
not inferred from current wallet snapshots, trade price returns, or cumulative
dollar PnL. Without the inputs they remain null and cannot get meaningful ranks.
See [the financial schemas](actor_metrics.md). PnL follows the existing
[average-cost remaining-inventory definition](in_market_pnl.md).

XGBoost trains on training intervals, stops on validation log loss, and ranks
features by repeated **validation** permutation log-loss increases. It writes
coverage, gain, grouped permutation diagnostics, calibration, average precision,
ROC AUC when defined, confusion metrics, and full/selected model results.
The default SFT shortlist contains up to 12 positive-importance features;
`--top-k` changes the cap. If none qualify, the summary block is empty and a
constant-prevalence selected XGBoost baseline is recorded. `--sft-features all`
instead exposes every candidate to SFT. Common news/history/prices/PnL remain
present regardless of the shortlist.

Selection is a development heuristic, not a significance test or causal finding.
Correlated features can substitute for each other; rankings need not transfer
to SFT. Use the optional matched basic adapter to test the added summaries.
The ranking command also offers `--drop-group-ablation` for extra CPU retraining.
Validation is reused for early stopping/ranking/threshold selection; test remains
unopened by the ranking code. No claim of generalization follows from validation
feature gains alone.

Important outputs:

| Path under the experiment | Contents |
|---|---|
| `dataset/` | Frozen SFT rows, numeric feature tables, split plan, source audit and checksums |
| `xgboost/feature_importance.csv` | Held-out validation ranking and coverage |
| `xgboost/report.json` | Full/selected XGBoost validation metrics and baselines |
| `xgboost/selected_features.json` | SFT summary shortlist |
| `sft/selected/` | Exact SFT dataset used by the selected adapter |
| `runs/selected/adapter/` | Final adapter and tokenizer |
| `runs/selected/training_metadata.json` | Training identity, validation losses and completion state |
| `status.json`, `logs/`, `stages/` | Progress, failures and resumability records |

## Tomorrow: explicit frozen testing

Test the SFT adapter after its training metadata says `completed`:

```bash
python3 scripts/evaluate_interval_decisions.py \
  --dataset-dir /workspace/experiments/wc_intervals_v1/sft/selected \
  --run-dir /workspace/experiments/wc_intervals_v1/runs/selected \
  --model /workspace/models/Qwen3.6-27B \
  --out /workspace/experiments/wc_intervals_v1/evaluation/selected \
  --gpu 0
```

The evaluator verifies the original dataset manifest and split checksums, then
scores both canonical JSON answers using exactly the trainer's assistant/EOS
token mask. Their sequence likelihoods are normalized into a probability over
the two allowed answers. This is conditional on that answer set, not an automatic
calibration guarantee. Test does not tune the 0.5 threshold. `--limit N` is an
explicit pilot; full testing uses all frozen test rows. `--resume` continues an
identical interrupted evaluation. `--check-only` checks tokenization without
loading GPU model weights.

`evaluation/selected/summary.json` contains log loss, Brier score, average
precision, ROC AUC when defined, calibration bins, precision/recall/F1, confusion
counts, and always-NO_TRADE/training-frequency baselines. It also separates actors
seen/unseen in training and reports actor/match metrics with equal-group averages.
Accuracy alone is insufficient with sparse positive intervals. These are point
estimates; no independent-row confidence interval is claimed for dependent
actor/match observations.

To evaluate XGBoost independently on its untouched feature-table test split:

```bash
python3 scripts/rank_interval_features.py evaluate \
  --dataset-dir /workspace/experiments/wc_intervals_v1/dataset \
  --model-dir /workspace/experiments/wc_intervals_v1/xgboost \
  --out /workspace/experiments/wc_intervals_v1/evaluation/xgboost.json
```

## Source requirements and limits

Use current official-price actor exports. If you need to collect/rebuild an
export, use `scripts/build_actor_dataset.py MARKET_ID --out NEW_EXPORT --cache
EXISTING_CACHE --skip-actor-snapshots --http-transport curl`, and repeat for your
World Cup markets. Reusing the same cache preserves completed trade captures.
At least three distinct match kickoff batches are needed; three matches are a
mechanical pilot, not a robust benchmark.

API captures must report exhaustion, and the capture must postdate the scheduled
window end. Missing price samples remain missing; raw interval news ending at a
future trade is never copied. Supplied local file/SQLite histories need an
explicit `--coverage-file` if the export has no exhausted API source. A certificate
asserts known source coverage, not a way to treat incomplete captures as complete:

```json
{"markets":{"1897059":{"start":"2026-06-14T00:00:00Z","end":"2026-06-15T00:00:00Z","complete":true,"reason":"Describe the independently verified capture coverage"}}}
```

Paused API captures are rejected even with a certificate. API exhaustion does
not prove every historical execution is visible. The cohort's full-market trade
cap is retrospective and does not establish that wallets are human. News times
are occurrence proxies; block times need not equal order-decision times. Fixed
150-minute schedules can include post-play inactivity. Base pretraining may
already include match outcomes. These limitations are carried in the manifest.

Offline checks include causal-boundary tests, future-data invariance, source
coverage/cap integrity, natural class proportions, chronological splits, selected
feature projection, actual CPU XGBoost fitting, trainer compatibility, and
likelihood scoring masks. Live CUDA training/evaluation must pass the RunPod
smoke gate; it is not exercised by CPU unit tests.
