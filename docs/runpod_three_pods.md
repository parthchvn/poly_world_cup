# Three shared-volume RunPod jobs

Use one two-GPU pod per variant and the same shared `/workspace` volume. The
model is expected at `/workspace/models/Qwen3.6-27B`, using the same installed
training environment as the earlier successful QLoRA smoke test. The launcher
checks packages, GPUs, tokenizer and local weight files before collection. It
does not install or upgrade packages.

Run from the same pinned repository commit on all pods:

```bash
# Base pod
bash tools/runpod_basic.sh

# Base + in_market pod
bash tools/runpod_inmarket.sh

# Base + all_history pod
bash tools/runpod_global.sh
```

These are separate terminal commands, one per pod. The Base pod collects and
freezes the common dataset. The other two wait for its `common_ready.json`, check
the same code/configuration/source hashes and package versions, then enrich their own copy and train.
The Base pod can begin training while the other pods derive their features.

Defaults are about **40,000 training decisions**, **2,000 validation decisions**
and **2,000 test decisions**. A decision is one distinct actor/market trade
timestamp. It may contain multiple simultaneous fills. It is not a raw
interval row or a JSONL conversation line. Whole actor conversations are kept,
so each split can exceed its requested target count by up to 19 decisions.

The plan uses the bundled, audited 2026 fixture registry, one draw contract per
fixture, with disjoint candidate fixture ranges for train, validation and test.
It collects additional candidates until each budget is met. Actors retain the
existing maximum of 20 captured executions per binary market. Entire holdout
conversations starting before or at the previous split's latest selected query
are excluded. Insufficient eligible candidates stop the job with a clear error.
There is no automatic weakening of time boundaries or truncation of histories.

Current actor portfolio snapshots are skipped because they are audit-only data.
The original market exports and API caches remain resumable. Selected actor
files are copied unchanged into the common cohort. Only selected wallets are
sent to the global history collector.

The two augmented runs explicitly use the four automatically available metrics:

- `average_execution_notional`
- `execution_notional_cv`
- `executions_per_day`
- `buy_notional_share`

The in-market version uses earlier executions in that binary market. The global
version uses earlier API-served wallet executions across markets. These jobs do
not produce Sharpe or realized-PnL features because historical accounting and
capital-adjusted return ledgers have not been supplied. The general metrics
scripts support those inputs as described in [actor metrics](actor_metrics.md).

All runs start independently from the base model, with seed 42, one epoch,
effective batch 8, two GPUs, QLoRA rank 16, alpha 32 and learning rate 0.0001.
Length grouping is disabled so different feature lengths do not alter the
seeded conversation order. `--smoke-then-full` checks the first ten optimizer
steps and continues the same run without loading the weights again. Validation
is used during training; the held-out test split is not trained on. Conversations
over 8192 tokens fail rather than being silently truncated.

The shared experiment root is `/workspace/world_cup_40k_v1`:

| Location | Contents |
|---|---|
| `common/exports/` | Full collected market exports |
| `common/prepared/` | Frozen selected exports, basic SFT dataset and selection audit |
| `common_ready.json` | Common cohort hashes, counts and configuration |
| `features/inmarket/sft/` | In-market training dataset |
| `features/global/sft/` | Global training dataset |
| `runs/basic/adapter/` | Basic adapter |
| `runs/inmarket/adapter/` | In-market adapter |
| `runs/global/adapter/` | Global adapter |
| `receipts/` | Common source hashes and independent training commands |
| `token_cache/` | Separate tokenized cache per variant |

For a new experiment, pass the same fresh `--root` on all pods. Size overrides
are `--targets`, `--validation-targets` and `--test-targets`; they must also agree
on all pods. Do not point multiple variants at the same training output.

After an API interruption, rerun the same wrapper. Existing completed market
exports and wallet capture pages are reused. If training already created a
checkpoint, add `--resume`, for example:

```bash
bash tools/runpod_global.sh --resume
```

The launcher selects the latest complete two-GPU checkpoint and lets the trainer
validate the exact resume settings. Incomplete checkpoints are ignored. If there
is no complete checkpoint, use a new experiment root to restart. If Base
preparation fails, waiting pods stop with that failure instead of waiting
indefinitely. Fix/restart Base, then rerun the other wrappers.

Once all three datasets exist, verify the complete comparison:

```bash
python3 tools/compare_actor_variants.py \
  --basic /workspace/world_cup_40k_v1/common/prepared/basic \
  --inmarket /workspace/world_cup_40k_v1/features/inmarket/sft \
  --global /workspace/world_cup_40k_v1/features/global/sft
```

History retrieval is bounded by the API's available records. Execution time is
an availability proxy, and actor selection is retrospective. The task remains
predicting execution attributes given an observed execution time. See
[the comparison guide](actor_dataset_variants.md) for evaluation limits.
