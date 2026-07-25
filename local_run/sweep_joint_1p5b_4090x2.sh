#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.11}"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="$VENV/bin/python"
GPU_IDS="${GPU_IDS:-0,1}"
MODEL_PATH="${MODEL_PATH:-/rdata/models/Qwen2.5-1.5B-Instruct}"
ENCODER_PATH="${ENCODER_PATH:-/rdata/models/all-MiniLM-L6-v2}"
RESULTS_ROOT="${RESULTS_ROOT:-/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/joint_sweep}"
SFT_CACHE_ROOT="${SFT_CACHE_ROOT:-/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/sft_cache}"
SPEC="${SPEC:-$ROOT/configs/local/joint_sweep_1p5b_4090x2.yaml}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-15}"

mkdir -p "$RESULTS_ROOT/launcher_logs"
LAUNCH_TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
LAUNCH_LOG="$RESULTS_ROOT/launcher_logs/${LAUNCH_TIMESTAMP}-$$.log"
ln -sfn "$(basename "$LAUNCH_LOG")" "$RESULTS_ROOT/launcher_logs/current.log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

printf '[%s] launcher start pid=%s log=%s\n' \
  "$(date -u '+%FT%TZ')" "$$" "$LAUNCH_LOG"
printf '[config] repo=%s spec=%s gpus=%s results=%s progress=%ss\n' \
  "$ROOT" "$SPEC" "$GPU_IDS" "$RESULTS_ROOT" "$PROGRESS_INTERVAL_SECONDS"

if [[ ! -x "$PYTHON" ]]; then
  command -v "$PYTHON_BOOTSTRAP" >/dev/null
  "$PYTHON_BOOTSTRAP" -m venv "$VENV"
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
  "$PYTHON" -m pip install "torch==2.7.1"
  "$PYTHON" -m pip install -e ".[dev,campaign]"
fi

export HF_HOME="${HF_HOME:-/rdata/minsoo3.kim/hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON" - "$GPU_IDS" <<'PY'
import sys
import torch

gpu_ids = [int(value) for value in sys.argv[1].split(",")]
assert torch.cuda.is_available(), "CUDA is unavailable"
assert len(gpu_ids) == 2, "the 1.5B local runner requires exactly two GPU ids"
assert torch.cuda.device_count() > max(gpu_ids), (
    f"requested physical GPU ids {gpu_ids}, visible count={torch.cuda.device_count()}"
)
assert torch.version.cuda and torch.version.cuda.startswith("12."), torch.version.cuda
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": {gpu: torch.cuda.get_device_name(gpu) for gpu in gpu_ids},
})
PY

for required in "$MODEL_PATH" "$ENCODER_PATH"; do
  if [[ ! -e "$required" ]]; then
    printf 'required offline path is missing: %s\n' "$required" >&2
    exit 2
  fi
done

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  ACTIVE_COMPUTE="$(
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
  )"
  if [[ -n "$ACTIVE_COMPUTE" && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    printf 'GPU compute processes are already active; nothing was killed:\n%s\n' \
      "$ACTIVE_COMPUTE" >&2
    printf 'set ALLOW_BUSY_GPUS=1 only after checking that sharing is intentional\n' >&2
    exit 2
  fi
fi

exec "$PYTHON" -u experiments/paper/run_joint_dev_sweep.py \
  --spec "$SPEC" \
  --gpus "$GPU_IDS" \
  --model-source "$MODEL_PATH" \
  --sentence-encoder "$ENCODER_PATH" \
  --sft-cache-root "$SFT_CACHE_ROOT" \
  --output-root "$RESULTS_ROOT" \
  --progress-interval "$PROGRESS_INTERVAL_SECONDS" \
  "$@"
