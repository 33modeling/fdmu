#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset PYTHONHOME PYTHONPATH PIP_TARGET PIP_PREFIX
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0,1}"
RUN_ROOT="${RUN_ROOT:-/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b}"
RESULTS_ROOT="${RESULTS_ROOT:-$RUN_ROOT/joint_sweep}"
JOINT_ROOT="${JOINT_ROOT:-$RESULTS_ROOT}"
FINAL_ROOT="${FINAL_ROOT:-$RUN_ROOT/final}"
FIDELITY_SUMMARY="${FIDELITY_SUMMARY:-$RUN_ROOT/fidelity/fidelity_summary.json}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-15}"

if [[ ! -x "$PYTHON" ]]; then
  printf '[ERROR] Python environment is missing: %s\n' "$PYTHON" >&2
  exit 2
fi
if [[ ! -s "$FIDELITY_SUMMARY" ]]; then
  printf '[ERROR] declared fidelity summary is missing: %s\n' \
    "$FIDELITY_SUMMARY" >&2
  printf '[ERROR] run: bash local_run/run_tofu_1p5b_fidelity.sh\n' >&2
  exit 2
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
printf '[config] declared_fidelity=%s\n' "$FIDELITY_SUMMARY"

FINALIZE_COMMAND=(
  "$PYTHON" -u experiments/paper/finalize_joint_sweep.py
  --joint-root "$JOINT_ROOT"
  --output-root "$FINAL_ROOT"
  --table-out "$FINAL_ROOT/table1.tex"
  --gpus "$GPU_IDS"
  --python "$PYTHON"
  --progress-interval "$PROGRESS_INTERVAL_SECONDS"
  --fidelity-input "$FIDELITY_SUMMARY"
)

if "$PYTHON" -c 'import yaml' >/dev/null 2>&1 \
  && "${FINALIZE_COMMAND[@]}" --check-complete; then
  printf '[LATEX SKIPPED] validated Table 1 already exists; rerun=0\n'
  exit 0
fi

if [[ "$PYTHON" == "$ROOT/.venv/bin/python" \
  && "${FDMU_4090_BOOTSTRAPPED:-0}" != "1" ]]; then
  bash local_run/bootstrap_4090_env.sh
fi

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

exec "${FINALIZE_COMMAND[@]}" "$@"
