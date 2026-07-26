#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset PYTHONHOME PYTHONPATH PIP_TARGET PIP_PREFIX
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0,1}"
JOINT_ROOT="${JOINT_ROOT:-/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/joint_sweep}"
FINAL_ROOT="${FINAL_ROOT:-/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/final}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-15}"

if [[ "${APPROVE_JOINT_BEST:-0}" != "1" ]]; then
  printf '[ERROR] review joint_sweep/BEST.json, then set APPROVE_JOINT_BEST=1\n' >&2
  exit 4
fi
if [[ ! -x "$PYTHON" ]]; then
  printf '[ERROR] Python environment is missing: %s\n' "$PYTHON" >&2
  exit 2
fi
if [[ "$PYTHON" == "$ROOT/.venv/bin/python" ]]; then
  bash local_run/ensure_4090_yaml.sh
fi

mkdir -p "$FINAL_ROOT/launcher_logs"
LAUNCH_TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
LAUNCH_LOG="$FINAL_ROOT/launcher_logs/${LAUNCH_TIMESTAMP}-$$.log"
ln -sfn "$(basename "$LAUNCH_LOG")" "$FINAL_ROOT/launcher_logs/current.log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

printf '[%s] joint-to-LaTeX start pid=%s log=%s\n' \
  "$(date -u '+%FT%TZ')" "$$" "$LAUNCH_LOG"
printf '[config] repo=%s python=%s gpus=%s joint=%s final=%s\n' \
  "$ROOT" "$PYTHON" "$GPU_IDS" "$JOINT_ROOT" "$FINAL_ROOT"

"$PYTHON" -c 'import datasets, torch, transformers, yaml; print(
    f"[deps] torch={torch.__version__} transformers={transformers.__version__} "
    f"datasets={datasets.__version__} pyyaml={yaml.__version__}"
)'

export HF_HOME="${HF_HOME:-/rdata/minsoo3.kim/hf_home}"
export RSUS_DATASETS_CACHE="${RSUS_DATASETS_CACHE:-$HF_HOME/rsus_datasets_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$PYTHON" -u experiments/paper/finalize_joint_sweep.py \
  --joint-root "$JOINT_ROOT" \
  --output-root "$FINAL_ROOT" \
  --table-out "$FINAL_ROOT/table1.tex" \
  --gpus "$GPU_IDS" \
  --python "$PYTHON" \
  --progress-interval "$PROGRESS_INTERVAL_SECONDS" \
  --approve-joint-best \
  "$@"
