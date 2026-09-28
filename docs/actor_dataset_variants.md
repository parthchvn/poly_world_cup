# Three datasets for a controlled feature comparison

Use three dataset generators and the existing shared trainer. The question is
whether summaries of an actor's history improve prediction of that actor's next
observed trade attributes.

| Variant | Script | Derived-history scope |
|---|---|---|
| Basic | `scripts/build_actor_dataset.py` | No derived actor metrics. Earlier trades in the current conversation remain. |
| Basic + in-market | `scripts/derive_actor_metrics.py` | One actor in the current binary market (`condition_id`). |
| Basic + global | `scripts/derive_global_actor_metrics.py` | One actor across all captured markets, including the current market. |

Both derived variants offer the same 18 metric names. “Global” changes the
history used to compute them; it does not add a second in-market metrics block.
Time since the previous trade and historical markout are excluded.

## What is and is not available automatically

Four metrics use earlier captured executions: average execution notional,
notional variability, execution frequency, and the fraction of notional bought.
The in-market script reads them from the actor exports. The global script
collects wallet executions across markets or reads your normalized wallet file.
A wallet with no eligible earlier observations has unavailable values, not a
made-up zero history.

The other 14 metrics require historical financial inputs:

- Ten completed-position metrics, including realized P&L, win rate, holding
  duration, and profit factor, need `--closed-positions`.
- Sharpe, Sortino, drawdown, and return volatility need `--returns-file`.

For in-market features, both files are scoped by actor and condition. For global
features, closed positions can come from every condition, while returns must be
whole-wallet equity returns. Use the same return-period duration across actors
and compared datasets, with compatible currency and accounting conventions,
so risk statistics have a common basis. Adding returns from different markets is not a
valid way to obtain the wallet's return. Today's open/closed positions and P&L
snapshots are not substituted for either historical input.

Without those financial files, both datasets still work. Unavailable metrics
stay `null` with reasons in raw audit files and are omitted from model inputs.
The shared instruction defines absent metrics as unavailable, not zero. Thus
the default experiment compares earlier captured
activity summaries, not a fully populated set of performance measures. See
[the formulas and financial input schemas](actor_metrics.md).

## Keep the same examples in all three datasets

Prepare the basic SFT dataset once. Derive the two augmented datasets from that
same directory. Preserve the base model, seed, optimizer settings, number of
epochs, and held-out examples across runs.

Global history requires an additional split check. If training and held-out
match windows overlap, an earlier held-out action could enter the global
summary for a later training query. The global SFT generator rejects this setup:
all training query times must precede all validation query times, and validation
must precede test. It does not silently mask other markets and call the result a
complete global history.

To retain the existing fixture assignments while obtaining strict chronological
query ranges, run the explicit `purge` command below **before** deriving either
variant. It keeps every training conversation. It removes entire validation
conversations starting at or before the latest training query, then removes
entire test conversations starting at or before the latest retained validation
query. It never cuts a conversation midway. The resulting basic cohort is the
common source for all three variants.

Purging can remove many examples, especially when pre-match trading windows
overlap. If an enabled validation or test split becomes empty, the command fails.
Collect later matches or supply a different fixture split plan and prepare again.
A disabled test split is accepted for development, but use a nonempty held-out
test split for the final comparison. Three matches are a pipeline check, not a
substantial performance benchmark.

## Commands from preparation to training

These RunPod paths match the existing project layout. Run from the repository
root; replace the raw export and model paths if yours differ. The output paths
must be new. If the repository is not present, first run:

```bash
cd /workspace
git clone https://github.com/parthchvn/poly_world_cup.git
cd poly_world_cup
```

If it is already present, `cd` into that checkout instead. Then:

```bash
(
set -e
git pull --ff-only origin main

actor_root=/workspace/world_cup_actor_data/data
model_path=/workspace/models/Qwen3.6-27B
experiment_root=/workspace/datasets/world_cup_variants_v1

python3 scripts/build_actor_dataset.py prepare \
  --input-root "$actor_root" \
  --out "$experiment_root/basic_unpurged" \
  --tokenizer "$model_path"

python3 tools/compare_actor_variants.py purge \
  --input "$experiment_root/basic_unpurged" \
  --out "$experiment_root/basic"

python3 scripts/derive_actor_metrics.py \
  --input-root "$actor_root" \
  --sft-dir "$experiment_root/basic" \
  --out "$experiment_root/inmarket"

python3 scripts/derive_global_actor_metrics.py \
  --input-root "$actor_root" \
  --sft-dir "$experiment_root/basic" \
  --cache /workspace/world_cup_actor_data/wallet_cache \
  --http-transport curl \
  --out "$experiment_root/global"

python3 tools/compare_actor_variants.py \
  --basic "$experiment_root/basic" \
  --inmarket "$experiment_root/inmarket/sft" \
  --global "$experiment_root/global/sft"
)
```

If actor exports have not been built yet, replace the first `prepare` command
with the combined collection-and-preparation command, selecting enough distinct
matches for your experiment:

```bash
python3 scripts/build_actor_dataset.py sft 1897035 1897038 1897059 \
  --data-root /workspace/world_cup_actor_data/data \
  --cache /workspace/world_cup_actor_data/data/market_actor_cache \
  --reuse-existing \
  --http-transport curl \
  --out /workspace/datasets/world_cup_variants_v1/basic_unpurged \
  --tokenizer /workspace/models/Qwen3.6-27B
```

The comparison tool checks identical actor/market conversations, query times,
ordering, target labels, base context, and system instructions, plus the expected
feature scopes and chronological split boundaries. It catches accidentally
mixing different export versions or applying different cohorts to the variants.
It does not evaluate a trained model.

Once comparison succeeds, train three independent adapters:

```bash
(
set -e
experiment_root=/workspace/datasets/world_cup_variants_v1
model_path=/workspace/models/Qwen3.6-27B
run_root="/workspace/runs/world_cup_variants_$(date +%Y%m%d_%H%M%S)"

for variant in basic inmarket global; do
  dataset_path="$experiment_root/$variant"
  if [ "$variant" != basic ]; then
    dataset_path="$dataset_path/sft"
  fi
  python3 scripts/train_world_cup_multigpu.py \
    --gpus 2 --gpu-ids 0,1 \
    --model "$model_path" \
    --dataset-dir "$dataset_path" \
    --seed 42 --epochs 1 \
    --out "$run_root/$variant" \
    --smoke-then-full
done
)
```

Each run starts from the same base model with a new adapter and optimizer. Do
not pass `--init-adapter` from another variant or resume one variant's checkpoint
in another. `--smoke-then-full` keeps weights loaded within each run; independent
runs load their own base model. Final adapters are under each run's `adapter/`.
The trainer checks the enriched token lengths; it fails on overlong conversations
instead of truncating history or targets.

The trainer uses validation during training and leaves test files untouched.
Use validation to select feature subsets and settings, then compare the chosen
models once on the same test examples. Report the predictive task and dataset
coverage alongside loss or action-attribute accuracy. This remains conditional
on an observed execution; it is not a test of when to trade or of strategy P&L.

## Feature subsets and token length

For a focused activity-only experiment, add the same option to both derived
commands:

```bash
--features average_execution_notional,execution_notional_cv,executions_per_day,buy_notional_share
```

The default includes all 18 features. Use identical feature selections when
comparing in-market versus global scope. Change one factor at a time when doing
further feature ablations, and write each experiment to new output directories.

All three SFT variants omit redundant source URLs, ESPN identifiers, wallet and
market IDs inside prompts, collection metadata, and duplicated probability
fields. The informative market description, news text and times, query time,
price and freshness, and previous trade targets stay readable. Identifiers,
source links, and detailed provenance remain outside model messages for audits.

Derived SFT prompts contain a short `scope` (`current_market` or `global_wallet`),
available selected feature values, and supporting sample counts. Only three
count families can enter prompts: `captured_executions`, `completed_positions`,
and `eligible_return_periods`. A count is included only when at least one metric
from its family is selected. Win/loss breakdowns remain in raw audits, so those
counts do not indirectly restore a win-rate feature excluded from an experiment.

Unavailable selected values are omitted rather than repeated as `null` in every
turn. An initial row can therefore have `values: {}`, with scope and supporting
counts still present. The shared system instruction explains that absent metrics
are unavailable, not zero, and risk ratios are not annualized. Raw metrics keep
all 18 values, including nulls, reasons, and the complete counts.

When risk metrics have an eligible regular return series, `return_period_seconds` identifies their time basis. Full
unavailable reasons and timing metadata stay in the raw metrics output. Derived
numeric values use 10 significant digits by default. Change that with
`--metric-significant-digits` (4–16). This only affects derived model inputs;
raw metrics keep their precision, and execution labels and market prices are
unchanged. No opaque abbreviated feature names or packed numeric arrays are
used to save tokens.

## Historical availability and remaining limits

At query time `t`, a captured fill must precede `t`. Supplemental positions must
both close and become known before `t`; returns must both finish and become known
before `t`. Equality is excluded, so no current or same-timestamp execution can
update its own input. Later changes to Sharpe do not rewrite earlier rows.

For global API collection, recorded execution timestamps are an availability
proxy. They do not establish exactly when the public feed published a fill or
when another trader could observe it. Offline records carry explicit `known_at`
timestamps, whose truthfulness remains the data producer's responsibility.
Current API snapshots, later settlements, and the current target are never used
to fill earlier gaps.

“Across all markets” describes the scope of captured wallet executions. It does
not certify every fill, transfer, split, merge, fee, or redemption is present.
A historical financial ledger needs those accounting facts when relevant.
The basic dataset also retains its documented execution-time and news-time
limitations. Its default actor filter, at most 20 captured executions in the
current market, is selected retrospectively from the full capture. The three
variants share that selected cohort; global enrichment does not change it.
These scripts prevent specified feature look-ahead; they do not certify the entire source dataset as a live
prospective benchmark.
