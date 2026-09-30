# Unseen-wallet activity testing

`scripts/test_actor_activity.py` tests a separate question: **using observations
strictly before time t, will this wallet have any captured execution in the next
60 seconds?** Both models receive the same wallet, query time, market and target.
Only the In-market prompt gets its four earlier-execution summaries.

This is a zero-shot diagnostic for the existing adapters. Their earlier
trade-detail evaluation did not measure abstention, and even the corrected
interval-reconstruction training is a different task from future-window
prediction. This script does not change weights or retrain either adapter.

## Sampling and exclusions

- Keep wallets with **at most 20 captured executions in that binary market**, as
  requested. Count fills, including simultaneous fills, not timestamp groups.
  This is an activity filter, not proof of a wallet's market-maker status.
- Exclude the union of wallets in **both** variants' training **and validation**
  files. Match identities and dataset hashes are verified again against each
  completed adapter before evaluation. A wallet present in any training market
  is excluded from every test market, case-insensitively.
- Exclude training/validation matches and any match whose kickoff is not later
  than the latest training/validation query. Base-model pretraining exposure
  cannot be audited by these files.
- Build a fixed non-overlapping grid from scheduled kickoff through kickoff +
  120 minutes. The default window is `[t,t+60s)`: an execution exactly at t is a
  positive label, but is not available as input; one at t+60 belongs to the next
  window. The match duration is fixed before examining labels, not inferred from
  the last trade or match result.
- Enroll a wallet only after its first captured execution. Query times must be
  strictly later than that execution. This tests subsequent activity of known
  wallets, not whether an unknown wallet will place its first trade.
- Uniformly sample 2,000 eligible wallet/windows by a deterministic hash, seed 42.
  Selection does not use the labels and does not force a 50/50 class balance.
  Wallets with longer eligible exposure can contribute more windows. The seeded
  order is also the pilot order. Changing a later trade's size or timestamp
  cannot change earlier inputs or the grid; cohort membership remains subject
  to the explicitly requested whole-market trade-count filter.

The 20-trade cohort uses retrospective whole-market counts. That restriction is
deliberate and recorded in the manifest. The result applies to that cohort, not
to an unrestricted live wallet population. In particular, do not call this a
fully prospective enrollment study. If rebuilding without that restriction,
use unfiltered exports and `--max-trades-per-actor 0`; previously discarded
wallets cannot be recovered by changing this flag.

## Inputs and labels

Basic receives the market question and kickoff, query time and prediction window,
the latest strictly earlier saved YES/NO history sample (null if older than the
export's age limit), news from the previous 20 minutes, and up to the latest 20
earlier execution groups. The explicit history limits apply identically to both
variants; no target is removed for length. A prompt-length check occurs before
loading weights.

In-market additionally receives `average_execution_notional`,
`execution_notional_cv`, `executions_per_day`, and `buy_notional_share`, or the
subset configured in its training dataset. These use the earlier exported
history and the training dataset's metric settings. Other metrics are refused,
not substituted with snapshots or invented historical P&L.

News and prices are rebuilt at each query from the full saved timelines. The
next-trade interval rows are **not** used as input. Target actions and future
trade timestamps stay in the scoring data; there are no assistant labels in a
prompt. Current actor snapshots, resolution, future fills and later metrics are
never model inputs. Query time itself is visible: the hidden information is
whether an execution happens in its future window.

`NO_TRADE` means **no captured execution in that window**, not no intention,
submission or unfilled order. A non-exhausted API traversal and mismatched raw
export counts are rejected. Exhaustion still does not establish archive
completeness. ESPN event occurrence times and trade block times are not verified
publication/order-submission timestamps, so these are timestamp-controlled
retrospective observations, not reconstructed screenshots of the trader's UI.

## Prepare on the Mac, entirely offline

Use the **original SFT directories used to train the adapters**. Do not replace
them with newly rebuilt NO_TRADE datasets when evaluating old adapters. This
step only reads saved files; it makes no API calls or package installations.

```bash
(
set -e
cd "$HOME/poly_world_cup_fetch"
git fetch origin main
git checkout --detach origin/main

python3 scripts/test_actor_activity.py prepare \
  --basic-sft "$HOME/world_cup_40k_transfer/common/prepared/basic" \
  --inmarket-sft "$HOME/world_cup_40k_transfer/inmarket/sft" \
  --input-root "$HOME/world_cup_40k_transfer/common/exports" \
  --out "$HOME/world_cup_actor_activity" \
  --targets 2000 \
  --max-trades-per-actor 20

tar -czf "$HOME/world_cup_actor_activity.tar.gz" \
  -C "$HOME" world_cup_actor_activity
shasum -a 256 "$HOME/world_cup_actor_activity.tar.gz"
)
```

Preparation prints the retained wallet count, positive/negative window counts,
zero training/validation wallet overlap and eligible matches. It automatically
skips the training and validation markets in `common/exports`. It refuses to
overwrite an existing output. Reuse the completed bundle, or choose a new output
name for a deliberately different protocol.

If all eligible wallets were seen during training, or fewer than 2,000 eligible
windows remain, it stops before publishing; do not relax the wallet exclusion.
Add saved captures from additional later matches with `--exports PATH1 PATH2`,
or prespecify a smaller sample for a pilot. A small number of positive windows
means poor precision of recall estimates. Do not resample until a desired label
balance appears; use a larger predeclared test across more held-out matches.

Upload **only** `world_cup_actor_activity.tar.gz` to `/workspace` on the evaluation
pod. The full collection and wallet caches are not needed on that pod. If using
SCP, use the pod's current **direct TCP SSH** host and port from RunPod Connect:

```bash
scp -P YOUR_SSH_PORT "$HOME/world_cup_actor_activity.tar.gz" \
  root@YOUR_POD_HOST:/workspace/world_cup_actor_activity.tar.gz
```

## Run on the evaluation pod

Use the working Qwen training/evaluation package environment and an **idle GPU**.
Keep the Git checkout under `/root`; save results under `/workspace`. This command
checks both adapters first, evaluates them **sequentially**, then compares them.
It does not start training or compete with a training process intentionally. The
memory gate requires at least 55 GiB and 80% of the selected GPU's memory free.
It is a guard, not a guarantee against another process starting later.

```bash
(
set -e
repo=/root/poly_world_cup_activity
[ -d "$repo/.git" ] || git clone https://github.com/parthchvn/poly_world_cup.git "$repo"
cd "$repo"
git fetch origin main
git checkout --detach origin/main

gzip -t /workspace/world_cup_actor_activity.tar.gz
mkdir -p /root/eval_data
tar -xzf /workspace/world_cup_actor_activity.tar.gz -C /root/eval_data

python3 scripts/test_actor_activity.py run \
  --bundle /root/eval_data/world_cup_actor_activity \
  --basic-run-dir /workspace/base_migration/runs/basic \
  --inmarket-run-dir /workspace/inmarket_migration/runs/inmarket_16k \
  --model /workspace/models/Qwen3.6-27B \
  --out /workspace/evaluation/actor_activity \
  --gpu 0 \
  --resume
)
```

Add `--check-only` to the Python command for tokenizer/identity/length checks
without weights. Add `--limit 100` **and a different output directory** for a
technical pilot; pilot results are explicitly marked. A full run uses `--limit 0`
(the default). Before extraction you can additionally compare the archive's
`sha256sum` against the Mac's printed checksum. Internal bundle files are always
hash-verified by the evaluator.

If interrupted, rerun the same Python command with the same checkout/settings
and `--resume`; each prediction is flushed and fsynced. Do not update the checkout
mid-evaluation: code versions are part of the resume identity. A complete Basic
journal is reused while In-market resumes, without reloading Basic weights.
Each adapter loads weights once when it has unfinished work; this script does
not keep the base model resident between adapters.

To use different pods, use `evaluate --variant basic --run-dir BASIC_RUN` or
`evaluate --variant inmarket --run-dir INMARKET_RUN`, with the same bundle, model,
output/settings. Once results are on the same volume, run:

```bash
python3 scripts/test_actor_activity.py compare \
  --basic /workspace/evaluation/actor_activity/basic \
  --inmarket /workspace/evaluation/actor_activity/inmarket \
  --out /workspace/evaluation/actor_activity/comparison.json
```

## Scoring and output

The prompt maps `A=NO_TRADE` and `B=TRADE`. The saved tokenizer must encode each
as one token. **One forward pass per window** scores the next-token choice;
there is no multi-second trade JSON generation. The trade score is
`P(B) / (P(A)+P(B))`, with a fixed threshold of 0.5, never fitted on the test.
This is a forced-choice model preference, not an automatically calibrated
activity probability. The report also gives mean unconstrained probability
mass on A/B, useful for noticing a model that assigns little probability to
either requested answer. Letter/prompt sensitivity is a limitation of this
zero-shot test.

`basic/summary.json`, `inmarket/summary.json`, and `comparison.json` report:

- Actual class prevalence, TP/FP/TN/FN, trade precision, recall and F1.
- No-trade recall and balanced accuracy, so always predicting no trade is exposed.
- Average precision (ties handled as groups) and Brier score. The latter measures
  squared score error; it does not certify calibration.
- Always-NO_TRADE and a strictly prior repeat-execution-rate baseline. The latter
  counts earlier distinct execution groups after enrollment, divides by elapsed
  enrollment exposure, and maps the rate to the fixed horizon with a Poisson
  assumption. It is a simple baseline, not a fitted activity model.

Undefined metrics are null when there are no positives/negatives or no positive
predictions. Do not call a one-class pilot a successful activity test. Windows
from a wallet or match are correlated: this descriptive report does not give
independent-sample significance tests. Compare across multiple held-out matches
before making a model-selection claim. No local fixture test substitutes for
running the actual adapters on the pod.
