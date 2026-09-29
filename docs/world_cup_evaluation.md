# Compare Basic and In-market adapters on held-out matches

Here **Basic** means your fine-tuned Basic adapter, not the unadapted Qwen model.
Both models predict the same execution targets using their respective input
features. This is conditional execution prediction: the time is an observed
trade time. It does not test whether a trader chooses to trade, profitability,
winning outcomes, or a deployment policy.

## 1. Mac: freeze the evaluation data

Use the completed Basic and In-market SFT directories from the SAME cohort used
for training. No tokenizer, PyTorch, transformers, or GPU is required here.
Update the repository to the evaluation commit (or the latest main containing it).
Do not modify repositories that are currently running training on a pod.

```bash
cd "$HOME/poly_world_cup_fetch"
git fetch origin main
git checkout --detach origin/main

python3 scripts/prepare_world_cup_evaluation.py \
  --basic-sft "$HOME/world_cup_40k_transfer/common/prepared/basic" \
  --inmarket-sft "$HOME/world_cup_40k_transfer/inmarket/sft" \
  --out "$HOME/world_cup_eval_test"
```

This default is entirely offline. It packages the already frozen `test` split,
which training and smoke evaluation do not use. Your recorded cohort has 2,000
held-out targets in match `espn:760486` (market `2680256`). The script prints the
actual target and fixture counts rather than relying on these historical counts.

For a broader comparison, collect these three held-out matches instead:

```bash
python3 scripts/prepare_world_cup_evaluation.py \
  --basic-sft "$HOME/world_cup_40k_transfer/common/prepared/basic" \
  --inmarket-sft "$HOME/world_cup_40k_transfer/inmarket/sft" \
  --market-ids 2680256 2690981 2707625 \
  --capture-root "$HOME/world_cup_40k_transfer/common/exports" \
  --cache "$HOME/world_cup_40k_transfer/market_cache" \
  --http-transport curl \
  --out "$HOME/world_cup_eval_matches"
```

This reuses complete market exports, resumes existing capture caches, and fetches
only missing captures through the existing bounded collector. It computes the
four execution-history features with your training configuration. It does NOT
fetch global wallet history. It checks every requested match against the actual
training and validation references before any collection, including contracts
other than the trained contract in the same match. Query times must be later
than ALL train/validation query times. Whole actor conversations beginning before
that cutoff are excluded and counted, without consulting their target values.
All otherwise eligible conversations in those matches are kept, so this is not
restricted to the original 2,000-target test subset. The source collector's
20-execution actor filter is preserved. This filter uses completed-market
activity and therefore defines a retrospective cohort, not an online wallet filter.

The outputs contain only `basic.jsonl.gz`, `inmarket.jsonl.gz`, and a provenance
manifest. No training files, giant wallet caches, or local absolute source paths
are needed for inference. Existing output directories are never overwritten.

Package the chosen bundle (this example uses the three-match bundle):

```bash
COPYFILE_DISABLE=1 tar -czf "$HOME/world_cup_eval_matches.tar.gz" \
  -C "$HOME" world_cup_eval_matches
shasum -a 256 "$HOME/world_cup_eval_matches.tar.gz"
```

Upload with JupyterLab or SCP to `/workspace` on each evaluation pod. Compare
`sha256sum /workspace/world_cup_eval_matches.tar.gz` with the Mac hash, then:

```bash
mkdir -p /root/eval_data
tar -xzf /workspace/world_cup_eval_matches.tar.gz -C /root/eval_data
```

For the offline test bundle substitute `world_cup_eval_test` in these paths.

## 2. Pod or Jupyter: run inference after training completes

Use the same working training environment and the original frozen Qwen weights.
No installations or downloads occur in the evaluator. It checks completed full
training metadata, adapter files, model config, tokenizer, and training/validation
file hashes against the reference datasets used to prepare the bundle. A smoke
adapter, wrong variant, unfinished run, changed reference files or a seen match
is rejected. The model config checksum cannot prove identical base weight bytes;
use the same original model directory, not another fine-tuned or merged model.

To avoid changing a running trainer's checkout, clone the evaluation code into
`/root/poly_world_cup_eval` and pin the evaluation commit on both pods:

```bash
git clone https://github.com/parthchvn/poly_world_cup.git /root/poly_world_cup_eval
cd /root/poly_world_cup_eval
```

`notebooks/compare_world_cup_models.ipynb` is the Jupyter entry point. Its first
cell lets you choose `basic` or `inmarket`, bundle path, model path, and run path.
Run the notebook in the environment where the training packages are installed.
The notebook runs subprocesses with its own Python interpreter.

Alternatively run on the Basic pod:

```bash
python3 scripts/evaluate_world_cup.py \
  --bundle /root/eval_data/world_cup_eval_matches \
  --variant basic \
  --run-dir /workspace/base_migration/runs/basic \
  --model /workspace/models/Qwen3.6-27B \
  --out /workspace/evaluation/basic \
  --gpu 0
```

On the In-market pod:

```bash
python3 scripts/evaluate_world_cup.py \
  --bundle /root/eval_data/world_cup_eval_matches \
  --variant inmarket \
  --run-dir /workspace/inmarket_migration/runs/inmarket_16k \
  --model /workspace/models/Qwen3.6-27B \
  --out /workspace/evaluation/inmarket \
  --gpu 0
```

Each invocation loads one model once on one GPU, quantized to NF4 as in training.
This leaves the second GPU free; it is NOT two-GPU distributed inference. The two
variants can run simultaneously on separate pods. Use the same evaluation commit,
software versions, decoding flags and target selection on both.

Add `--check-only` first to verify identities and all prompt lengths without
loading weights. The default `--max-context 32768` checks prompt length plus the
fixed `--max-new-tokens 2048` generation budget. Nothing is truncated or silently
dropped. These are ceilings, not fixed padding. If memory is insufficient the
process fails rather than silently changing the evaluated cohort.

For a short mechanical pilot use `--limit 20 --out /workspace/evaluation/basic_pilot`
(or `inmarket_pilot`). A positive limit selects the same fixed prefix of target IDs
for both models. It does not establish performance on the full holdout. Decide
model settings using validation, not these test predictions. Run the full test
only for the frozen comparison; avoid repeatedly tuning based on its scores.

To continue an interrupted evaluation, repeat the identical command with `--resume`.
Predictions are written after every target and synced every ten. Previously saved
predictions are reused. Only an incomplete final JSONL line from interruption is
discarded. The model must reload after a process restart. Output directories are
locked against two concurrent evaluators. Do not use a pilot output directory for
a full run: target-selection identity is intentionally different.

No inference ETA is promised from training throughput: autoregressive generation
has different costs. The script reports measured seconds per target and a rough
remaining estimate after ten predictions. Long histories and multi-fill answers
make target times vary.

## 3. Compare outputs

Bring the completed `basic` and `inmarket` result directories together on a pod or
Mac. Only three files per result are required: `identity.json`, `predictions.jsonl`,
`summary.json`. They contain no model weights. Then:

```bash
python3 scripts/compare_world_cup_evaluations.py \
  --basic /workspace/evaluation/basic \
  --inmarket /workspace/evaluation/inmarket \
  --out /workspace/evaluation/comparison.json
```

Comparison rejects changed files, different targets, decoding settings, software,
evaluation code, tokenizers or bundles. It reports training-setting differences;
8192 versus 16384 training limits are separately shown and did not truncate data
in this pipeline. Other differences may confound attribution to the features.
The notebook also provides a comparison cell when both results are accessible.

Metrics (rates between 0 and 1 in JSON, percentages in the console):

| Metric | Meaning |
|---|---|
| `valid_json` | Strict answer schema, legal categories, finite positive shares, valid price; no markdown repair |
| `trade_count_correct` | Correct number of executions at the timestamp |
| `side_multiset_correct` | Correct BUY/SELL counts |
| `outcome_multiset_correct` | Correct Yes/No counts |
| `side_outcome_multiset_correct` | Correct counts of each (side, outcome) combination; primary categorical score |
| `exact_trade_multiset` | All four fields match numerically, including duplicate fills, ignoring within-timestamp order |
| `price_mae_conditional` | Mean absolute price error in dollars per share on category/count-matched targets |
| `shares_mae_conditional` | Mean absolute share-count error on those targets |
| `notional_mae_conditional` | Mean absolute error of shares times price on those targets |
| `numeric_target_coverage` | Fraction eligible for the conditional numeric errors |
| `generation_limit_hits` | Outputs stopped by token budget rather than the answer EOS |

For numeric matching, fills within each side/outcome category are sorted by shares
then price. This deterministic rule does not optimize matching against truth.
Price MAE times 100 gives cents/share. Numeric errors are also compared on the
**same jointly category-correct targets** in the paired report. Conditional errors
alone can look deceptively good when a model gets few categories right. Invalid
answers count as failures for every categorical score and are not dropped from
the denominator. Shares above 1e100 are invalid to keep error arithmetic bounded.
The output also preserves full prediction text, target labels, query times, IDs,
prompt hashes, token counts and token-budget hits for manual inspection.

Per-match scores and a paired match-cluster bootstrap (2,000 replicates, seed 42)
are included when at least two matches are present. With one match no interval is
reported. With very few matches intervals are exploratory, and the bootstrap does
not model wallet dependence across matches. A feature benefit on three matches is
not proof of tournament-wide generalization.

## Leakage and scope

Generation uses system context, earlier user/assistant turns, and the current user
query. The current answer and future turns are never supplied to `generate`.
At later timestamps the earlier TRUE observed actions are available, as they would
be in an observed-wallet setting. This is one-step observed-history evaluation,
not a simulation that feeds the model's mistakes back into the next prompt.
In-market metrics are updated after each timestamp's complete execution group.
The entire same-timestamp group is excluded from that group's feature calculation.
No current portfolio snapshots, settlement PnL or future history is added.

As in training, public execution timestamps and ESPN/history timestamps are
availability proxies; these checks cannot reconstruct a trader's exact live UI,
private orders or information arrival latency. Also, a fine-tuning holdout is not
proof that the original base model never saw the historical match in pretraining.

## Validation performed

CPU tests cover frozen-test pairing, fresh export conversion, chronological and
match-overlap rejection, unchanged targets, causal metrics, answer-prefix
construction, numeric scoring, strict JSON, duplicate fills, resume recovery,
training provenance checks and paired reporting. GPU model loading/generation
must still be verified in the user's Qwen RunPod environment; this development
environment has no torch installation or CUDA GPU.
