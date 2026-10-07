# Frozen 9B activity experiment

This experiment asks whether the completed SFT checkpoint contains useful information
for predicting a trade in the next five-minute window, even though its generated
answers always chose NO_TRADE. It does not update the base model or saved adapter.

Four independent workers extract the final decoder representation at the final input
token, using the saved SFT tokenizer and the exact original conditioning prefix.
The target answer is excluded from the model input. The LM vocabulary projection and
generation are skipped. Base and LoRA parameters are frozen. NF4/BF16 loading matches
the original evaluator. Physical GPUs 1–4 are used; GPU 0 is reserved.

Only train and validation are read. Both retain their natural class prevalence.
The CPU stage fits an L2 logistic classifier to frozen vectors, with preprocessing
fitted on train only. Three predefined regularization strengths (C=0.01, 0.1, 1)
are compared by validation average precision. No oversampling, class weighting,
trade-only continuation, or repeated LLM fine-tuning is performed.

Comparisons use exactly the same validation rows: training prevalence, a small
recent-activity logistic model, and the previously trained full XGBoost classifier.
Average precision, precision/recall, log loss, Brier score and calibration are
reported. A validation-selected F1 threshold is a development diagnostic; it is
not an independently tested operating point. The 0.5 threshold is also reported.
The report and precision–recall/calibration plot are under `comparison/`.

## RunPod

Upload `wc_frozen_probe_v1.zip` to `/workspace`, then:

```bash
(
set -euo pipefail
python3 -m zipfile -e /workspace/wc_frozen_probe_v1.zip /root
python3 -m pip install --no-deps 'scikit-learn==1.7.2' 'joblib==1.5.2' 'threadpoolctl==3.6.0'
OUT=/workspace/runs/wc_frozen9b_probe_v1
mkdir -p "$OUT/logs"
nohup python3 -u /root/poly_world_cup_probe_v1/scripts/run_frozen_activity_probe.py \
  --gpu-ids 1,2,3,4 --batch-size 4 --out "$OUT" \
  >> "$OUT/launcher.log" 2>&1 < /dev/null &
echo "Probe launcher PID: $!"
)
```

Watch progress:

```bash
tail -n 15 -F /workspace/runs/wc_frozen9b_probe_v1/launcher.log \
  /workspace/runs/wc_frozen9b_probe_v1/logs/gpu{1,2,3,4}.log
```

Ctrl+C exits the log viewer. The background experiment continues. Rerunning the
launcher with identical settings resumes verified extraction chunks. If a requested
GPU already has memory allocated, the launcher fails before starting workers.
It never stops unrelated jobs. Worker failures stop this experiment's remaining
workers and preserve complete chunks. Do not change the adapter or input files.

Paths default to this pod's completed `Qwen3.5-9B_retry1`, shared interval files,
and local base checkpoint. See `--help` for explicit alternatives. A saved identity
binds inputs, tokenizer, adapter, extraction settings and code. An initial padded
versus individual forward check guards batched extraction. If that check fails,
use `--batch-size 1` and a different `--out`; do not suppress the check.

The existing scientific Python, Qwen/PEFT and CUDA packages are reused. The install
above adds only the CPU classifier and its helpers without resolving or upgrading
the GPU stack. No NCCL collective or distributed training is used.

## Interpretation

The validation split has only 150 trade windows from one fixture. These results
can identify a promising ranking signal or a failed approach, but cannot establish
generalization across matches. Hyperparameter and threshold selection both use
validation; their reported scores are development results. The already inspected
test split is not opened by this experiment. Any final claim needs additional
unseen matches after the model and decision threshold are fixed.

Do not interpret a high accuracy or a nonzero recall alone as success. Compare
average precision and the entire precision–recall curve with XGBoost and the
simple activity baseline. Extra model complexity needs measurable benefit.
