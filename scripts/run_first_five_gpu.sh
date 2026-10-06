#!/usr/bin/env bash
# One GPU for Qwen3.5-0.8B; four for Qwen3.5-9B. No Git/authentication at runtime.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
RAW=/workspace/wc_actor_subset_v2
SHARED=/workspace/wc_shared_intervals_v2
RUNS=/workspace/runs/wc_first_5gpu_v1
ZIP=/workspace/wc_200k_by_actor_v2.zip
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export FLA_TILELANG=1 FLA_DISABLE_BACKEND_DISPATCH=0
export HF_HOME=/root/.cache/huggingface TRITON_CACHE_DIR=/root/.cache/triton
export TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
mkdir -p "$RUNS/logs" /root/.cache
exec 9>/root/.cache/wc_first_5gpu.lock
flock -n 9 || { echo 'Five-GPU launcher is already running.'; exit 1; }

echo 'STAGE 1: verify local models, packages, and five GPUs'
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 -u - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from run_world_cup_experiments import probe_environment, verify_weights
for name in ('Qwen3.5-0.8B', 'Qwen3.5-9B'):
    model = Path('/workspace/models') / name
    verify_weights(model)
    probe_environment(model, '0,1,2,3,4')
    output = Path('/workspace/runs/wc_first_5gpu_v1') / name
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'Existing training output: {output}. Resume its checkpoint; do not start over.')
PY

echo 'STAGE 2: unpack raw whole-actor subset'
if [ ! -f "$RAW/selection.json" ]; then
    test -f "$ZIP" || { echo "Upload $ZIP first."; exit 1; }
    mkdir -p "$RAW"
    python3 -m zipfile -e "$ZIP" "$RAW"
fi
test -d "$RAW/exports"

echo 'STAGE 3: prepare common interval examples and XGBoost ranking'
python3 -u scripts/run_interval_experiment.py \
  --input-root "$RAW/exports" --out "$SHARED" \
  --model /workspace/models/Qwen3.5-0.8B \
  --target-mode trade-details --window-seconds 300 \
  --price-delta 0.02 --shares-relative-delta 0.20 --shares-absolute-delta 0 \
  --max-rows 200000 --max-trades-per-actor 20 --history-groups 20 \
  --sft-features all --global-batch 24 --seed 42 --rank-only --resume
tar -czf /workspace/wc_shared_sft_v2.tar.gz -C "$SHARED" sft/selected

train_model() {
    local name="$1" ids="$2" count="$3"
    shift 3
    python3 -u scripts/train_world_cup_multigpu.py \
      --model "/workspace/models/$name" --dataset-dir "$SHARED/sft/selected" \
      --out "$RUNS/$name" --cache-dir "$RUNS/token_cache/$name" \
      --gpus "$count" --gpu-ids "$ids" --micro-batch 1 --global-batch 24 \
      --epochs 1 --learning-rate 0.0001 --rank 16 --alpha 32 \
      --max-length 8192 --seed 42 --logging-steps 1 --save-steps 100 \
      --eval-steps 277 "$@"
}

echo 'STAGE 4: tokenize 0.8B, then 9B (see logs/*.prepare.log)'
train_model Qwen3.5-0.8B 0 1 --prepare-only > "$RUNS/logs/0.8B.prepare.log" 2>&1
train_model Qwen3.5-9B 1,2,3,4 4 --prepare-only > "$RUNS/logs/9B.prepare.log" 2>&1

echo 'STAGE 5: train both models; loss plots refresh every 10 seconds'
plot_pids=()
cleanup_plots() {
    for pid in "${plot_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${plot_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup_plots EXIT
# Keep plot outputs outside training directories: the trainer requires fresh outputs.
mkdir -p "$RUNS/plots"
for name in Qwen3.5-0.8B Qwen3.5-9B; do
    python3 -u scripts/plot_training_losses.py --run-dir "$RUNS/$name" \
      --output "$RUNS/plots/$name.png" --watch 10 > "$RUNS/logs/$name.plot.log" 2>&1 &
    plot_pids+=("$!")
done
train_model Qwen3.5-0.8B 0 1 --smoke-then-full > "$RUNS/logs/0.8B.train.log" 2>&1 &
small_pid=$!
train_model Qwen3.5-9B 1,2,3,4 4 --smoke-then-full > "$RUNS/logs/9B.train.log" 2>&1 &
large_pid=$!
printf '0.8B\t%s\n9B\t%s\n' "$small_pid" "$large_pid" > "$RUNS/logs/job_pids.tsv"
failed=0
if wait "$small_pid"; then code=0; else code=$?; failed=1; fi
echo "$code" > "$RUNS/logs/0.8B.exit_code"
echo "Qwen3.5-0.8B exit code: $code"
if wait "$large_pid"; then code=0; else code=$?; failed=1; fi
echo "$code" > "$RUNS/logs/9B.exit_code"
echo "Qwen3.5-9B exit code: $code"
cleanup_plots
plot_pids=()
for name in Qwen3.5-0.8B Qwen3.5-9B; do
    python3 scripts/plot_training_losses.py --run-dir "$RUNS/$name" \
      --output "$RUNS/plots/$name.png" >> "$RUNS/logs/$name.plot.log" 2>&1 \
      || echo "Plot unavailable for $name; inspect its plot log."
done
if [ "$failed" -eq 0 ]; then echo 'SUCCESS: both training jobs completed. Test set not evaluated.'; fi
exit "$failed"
