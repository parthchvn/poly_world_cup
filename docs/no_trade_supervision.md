# Correction: preserve interval labels in training

The raw actor export contains an open-interval NO_TRADE row followed by an
execution TRADE row. The previous SFT converter validated both raw rows but
created an assistant target only for the execution. Its system instruction
explicitly assumed an execution. This violated the requested interval supervision.
It affected Basic, In-market and Global SFT generated through that converter.

The correction changes the default `build_actor_dataset.py prepare` and `sft`
conversion. Both labels now reach assistant-token loss. A conversation with N
distinct execution timestamps has N NO_TRADE interval targets and N TRADE targets.
The first interval, from the recorded origin to the first execution, is retained.
Multiple executions at one timestamp remain one TRADE answer containing all fills.
Source numbers, split assignments and full actor histories are retained.

## Message sequence

1. User: open interval boundaries, query_time equal to its end, interval news,
   and official price samples strictly before that end. Market description is
   included on the first user turn.
2. Assistant: `{"action":"NO_TRADE"}`.
3. User: the endpoint query_time. Context from the preceding turn remains visible.
4. Assistant: the captured TRADE answer at that time.

Repeat for the next interval. News is not duplicated. Current execution values
appear only in the assistant answer. Future turns cannot be attended to by earlier
targets. In-market and Global metrics are computed strictly before the endpoint,
then attached to the pair without advancing execution history twice.

The manifest reports `no_trade_targets: true`, `prompt_schema_version: 3`, and
`target_protocol: observed_interval_and_execution_v1`. Each split reports total,
TRADE and NO_TRADE target counts. The trainer also reports actual `action_counts`
and rejects missing NO_TRADE supervision by default before loading model weights.
The automated runner rejects old trade-only uploads by default.

## Rebuild the existing Mac cohort without API calls

Use a checkout containing this correction. These paths refer to the existing
40k transfer bundle, whose selected exports preserve the original actor cohort.
Choose a new output directory; do not overwrite previous experiments.

```bash
cd "$HOME/poly_world_cup_fetch"

python3 scripts/build_actor_dataset.py prepare \
  --input-root "$HOME/world_cup_40k_transfer/common/prepared/selected_exports" \
  --split-file "$HOME/world_cup_40k_transfer/common/prepared/split_plan.json" \
  --out "$HOME/world_cup_40k_transfer/basic_with_no_trade"
```

Upload that new prepared Basic dataset. The automated runner can derive its
In-market variant offline. For Global, re-enrich against the existing compatible
raw-history metric index/cache; do not reuse the old trade-only SFT files as the
new mixed-label variant. No extra wallet traversal is intrinsically required to
attach the same strictly prior metrics to each interval/endpoint pair.

For the same saved cohort, target count doubles: the previous 40,001 train
execution-timestamp targets become 40,001 TRADE plus 40,001 NO_TRADE targets.
This is not twice as many collected trades. The collection planner's `--targets`
now counts both types and records a new request version. Rebuilding the saved
selected exports, rather than selecting a fresh budget, preserves the old cohort.
Token lengths are checked again; there is no truncation or automatic assumption
that the old maximum sequence length is sufficient.

Previously trained adapters do not acquire this supervision when code is pulled.
Use a new dataset and run directory. To compare training protocols, train new
adapters from the same original base with matched settings. Do not resume an old
optimizer checkpoint against changed training data. Existing completed results
still describe execution-attribute prediction, not trade/no-trade prediction.

Explicit legacy reproduction is available through conversion `--trade-only` and
training/runner `--allow-trade-only`. These are not defaults. Existing interrupted
runs must use the original pinned code, inputs and flags to resume.

## What the restored labels do not establish

These intervals are selected using their next observed execution endpoint. They
alternate with execution queries, and query scope/order reveals the action class.
Consequently, even perfect action accuracy on this reconstruction dataset is not
evidence of predicting future trades or deliberate abstention. NO_TRADE means no
captured execution in this actor-market open interval, not no submitted orders.

A prospective benchmark needs windows selected independently of future trades,
complete execution coverage, context available before the window starts, and
positive as well as negative training windows. That is a separate experiment;
this correction preserves the requested raw labels without claiming it implements
that forecasting task. Evaluation metadata and reports carry this distinction.
