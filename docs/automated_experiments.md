# Upload once, then train and evaluate automatically

`scripts/run_world_cup_experiments.py` controls the existing multi-GPU QLoRA
trainer, held-out evaluator, paired comparison and loss plotting. It makes **no
Polymarket or ESPN requests**. Basic means the fine-tuned Basic adapter, not an
unadapted Qwen baseline. This runner supports Basic and In-market; Global wallet
collection and Global evaluation are not part of this pipeline.

## What to upload

Upload a **prepared Basic SFT dataset** containing `manifest.json` and all three
splits: `train.jsonl`, `validation.jsonl`, `test.jsonl` (each may instead use
`.jsonl.gz`). These are the outputs of the existing cohort preparation, not raw
`actors/*.jsonl` exports. The splits must already be chronological and disjoint
by match. Whole actor conversations must be retained.

You can upload that directory, a `.tar.gz` containing it, or a directory containing
top-level `.tar.gz` uploads. The runner finds the dataset by its manifest, not a
required archive folder name. It validates gzip integrity, blocks links and path
traversal, and limits extraction to 10 GiB by default. A partial Jupyter upload
fails before training. Prefer transferring only the prepared files, not caches.

If only Basic is supplied, it creates In-market **offline** from earlier executions
in each Basic conversation. It uses the same four features as the 40k experiments:
average execution notional, execution-notional coefficient of variation,
executions per day, and BUY notional share. Every feature is computed before the
current timestamp's executions are added. Unavailable values are omitted, with
sample counts retained. It does not invent Sharpe ratios, portfolio returns,
positions, order-submission times or private wallet information.

If you already have a prepared In-market dataset, pass it with another `--data`
argument or include it in the same archive. It is used verbatim and checked
against Basic: same splits, actors, timestamps, answers and base contexts.

## One-time repository setup

Keep Git in the pod's local filesystem because some network volumes reject Git's
permission changes. Keep data, model weights, caches and results on persistent
storage. Both training and evaluation use local model files only.

```bash
git clone https://github.com/parthchvn/poly_world_cup.git /root/poly_world_cup_pipeline
cd /root/poly_world_cup_pipeline
```

Pin the commit you start with. **Use that same checkout to resume.** The pipeline
records code, data, model-config, package and setting fingerprints. Updating the
trainer during an existing experiment is deliberately rejected.

## One command after upload: two GPUs

For example, upload your prepared Basic archive as
`/workspace/world_cup_basic_sft.tar.gz`, then run:

```bash
python3 /root/poly_world_cup_pipeline/scripts/run_world_cup_experiments.py \
  --data /workspace/world_cup_basic_sft.tar.gz \
  --model /workspace/models/Qwen3.6-27B \
  --out /workspace/experiments/world_cup_v1 \
  --gpu-group 0,1 \
  --install-deps \
  --background
```

`--install-deps` is optional. On the previously successful PyTorch 2.8.0 / CUDA
12.8 / Triton 3.4.0 image, it installs missing packages at the known working
versions. It refuses conflicting existing versions and constrains torch/Triton
and the training packages; it does not replace the CUDA/PyTorch stack. Without
the flag, missing dependencies produce an actionable error. Use the same package
environment on both pods. Plotting also requires matplotlib.

The automatic sequence on one GPU group is:

1. Validate/stage data, derive In-market if needed, and freeze the paired test bundle.
2. Tokenize both train/validation datasets and check all test prompt lengths.
3. Train Basic, including an early smoke validation, with one training weights load.
4. Once its training process exits, load its final adapter for held-out evaluation.
5. Train and evaluate In-market the same way.
6. Save the paired comparison, numerical-error coverage, per-match results and plots.

The smoke validation continues into full training in the **same training process**.
Generation evaluation uses a separate process and loads weights once per evaluation
attempt. A restart must load weights again. The controller never loads weights.

No model is evaluated before its training completes. Training and generation never
share a GPU group concurrently. The default memory gate needs 60 GiB free per GPU,
targeting the idle 80 GB GPUs used in this project; this is a preflight gate, not a
guarantee against every possible OOM. Occupied GPUs cause a bounded wait and clear
failure, never termination of other jobs. Long examples are rejected, not truncated.

Defaults: one epoch, seed 42, effective batch 8, micro-batch 1, rank 16, alpha 32,
learning rate 0.0001, no length grouping, checkpoint every 100 steps, losses logged
every step. Both variants use a 16,384-token training ceiling. Evaluation retains
the same 32,768 context ceiling, 2,048 generated-token budget and greedy decoding.
Ceilings do not force padding to that length. Run `--help` for overrides; keep them
unchanged when resuming a campaign.

## Concurrent experiments

With four GPUs on **one pod**, add a second disjoint group:

```bash
python3 /root/poly_world_cup_pipeline/scripts/run_world_cup_experiments.py \
  --data /workspace/world_cup_basic_sft.tar.gz \
  --out /workspace/experiments/world_cup_parallel \
  --gpu-group 0,1 --gpu-group 2,3 --background
```

Each experiment trains and then evaluates on its own group. One adapter's
evaluation can run while the other group is still training. Groups must have the
same GPU count and may not overlap. `--global-batch` must be divisible by that
count. GPUs within a group are on one machine; this is not cross-pod distributed
training. Evaluation uses the first GPU of its group and reserves the group until
it exits, prioritizing safe memory isolation over maximal GPU packing.

For **two pods attached to the same network volume**, use the same repository
commit, package versions, model path, output path and training settings. Run the
same command with `--variant basic` on one pod and `--variant inmarket` on the
other, each using its local `--gpu-group 0,1`. Both need access to the uploaded
Basic data on that shared volume. Initialization is locked; staged data and caches
are reused. The last process to finish writes the comparison automatically.
The shared filesystem must support POSIX file locks and atomic renames.

## Status, interruption and resume

`--background` starts a detached process and prints its PID and controller-log
path. Closing a terminal does not stop it. It cannot survive stopping/terminating
the pod or a host failure; use persistent storage and rerun after restoring the
pod's environment. No script can repair a container whose binaries return I/O errors.

```bash
python3 /root/poly_world_cup_pipeline/scripts/run_world_cup_experiments.py \
  --out /workspace/experiments/world_cup_v1 --status
```

Each stage has a log under `OUT/logs/`; training also retains its normal `run.log`
and live `metrics.jsonl`/`losses.csv`. Controller progress is printed every minute.
Failures include the stage-log path and recent output. Inspect the log printed by
the background launch, especially for an initial dependency or upload failure.

**Rerun the identical launch command to resume.** Finished work is checked and
reused. The newest complete checkpoint with optimizer, scheduler, trainer state
and all per-rank RNG files is selected. A newer incomplete save is ignored. If no
complete checkpoint exists, the unfinished attempt is archived under
`unfinished_attempts/` and that variant restarts; it is never silently warm-started
from adapter-only weights. Evaluation resumes its prediction journal with unchanged
settings. Previously saved predictions are not regenerated. A checkpoint marker and
nonempty files cannot prove that every byte is healthy; the trainer still validates
and loads the checkpoint and reports corruption rather than silently discarding it.

Uploaded prepared files are copied under `OUT/data/`, so the same output can resume
without the original upload by omitting `--data`. Preserve the model and original
code/environment too. Do not change paths/settings/packages to force a resume;
choose a fresh `--out` for a genuinely different experiment. No automated unbounded
retry loop is used. A failed variant is recorded while the other variant may finish.

Use `--check-only` to perform dataset and token-length checks without loading weights.
This still requires tokenizer/datasets packages; it does not run CUDA kernels. Real
GPU import/BF16 checks happen in an isolated child before GPU work, and the trainer
and evaluator retain their existing kernel probes.

## Existing completed adapters

To evaluate completed runs without training them again, add:

```bash
  --basic-run /workspace/base_migration/runs/basic \
  --inmarket-run /workspace/inmarket_migration/runs/inmarket_16k
```

For existing adapters, upload **both exact original prepared variants**, so their
raw-file hashes match training. Automatically regenerated In-market data has its
own serialization/provenance and is intended for a new training experiment.
This mode verifies completion, training-data identity and held-out separation.
It does not resume an interrupted externally created training run, and does not
adopt evaluation journals from a different evaluator identity. Your older outputs
remain untouched. Reusing completed results from this runner needs no GPU.

## Outputs and interpretation

Everything is below `--out`:

| Path | Contents |
|---|---|
| `experiment.json` | Immutable code/data/environment/settings identity |
| `data/basic`, `data/inmarket` | Staged prepared SFT datasets |
| `bundle` | Paired held-out prompts and hidden answers |
| `runs/VARIANT/adapter` | Final LoRA adapter (base weights remain separate) |
| `runs/VARIANT/checkpoint-N` | Resume checkpoints |
| `evaluation/VARIANT` | Saved predictions, identity and scores |
| `plots/VARIANT_loss.png` | Per-model training/validation loss plot |
| `plots/loss_comparison.png`, `.pdf` | Overlaid training and validation panels |
| `comparison.json`, `.csv` | Paired metrics and audit details |
| `REPORT.md` | Readable comparison and loss-curve link |
| `completed.json` | Marker written after successful paired reporting |

The two models predict the same actor/market/timestamp targets; only the additional
In-market fields differ. Current/future answers are excluded, prior **observed**
actions are supplied, and generated answers are never rolled into later history.
Test scores do not choose checkpoints, hyperparameters or retry settings. The
early smoke check uses validation only. Checking test prompt length is a mechanical
compatibility check, not model fitting or checkpoint selection.

All labels are TRADE: these scores measure execution attributes conditional on an
observed trade, not timing, abstention, profit or prediction of match results.
Numerical errors are conditional on correct categories; `comparison.json` also
reports errors on the same jointly correct targets. One held-out match cannot
establish generalization across matches. Public timestamps and pretraining
contamination have the limitations documented in the existing evaluation guide.

Validation of this controller uses CPU fixtures and simulated GPU subprocesses for
scheduling/recovery, plus real data-contract/report/plot functions and process
cancellation tests. It does not substitute for a real GPU run in your environment.
