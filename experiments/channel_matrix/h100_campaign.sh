#!/usr/bin/env bash
set -Eeuo pipefail

# Safe H100 entry point for the sealed 7B/8B channel-matrix campaign.
#
# Examples:
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh preflight
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh fidelity
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh calibration
#   bash experiments/channel_matrix/h100_campaign.sh select-freeze
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh audit
#   bash experiments/channel_matrix/h100_campaign.sh aggregate
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh alpha-development
#   bash experiments/channel_matrix/h100_campaign.sh select-alpha-freeze
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh alpha-audit
#   bash experiments/channel_matrix/h100_campaign.sh legacy-alpha-diagnostic
# Two-GPU request sharding (use disjoint AUTHORS values):
#   GPU=0 AUTHORS=198 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh calibration
#   GPU=1 AUTHORS=199 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh calibration

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/group-volume/fdmu/.venv"
CONFIG="${CONFIG:-configs/channel_matrix/7b_tofu.yaml}"
MODEL_ID="${MODEL_ID:-qwen25_7b}"
GPU="${GPU:-0}"
AUTHORS="${AUTHORS:-}"
ACTION="${1:-}"

if [[ -z "${ACTION}" ]]; then
  echo "usage: GPU=<index> MODEL_ID=<alias|all> $0 {preflight|prefetch|dry-calibration|fidelity|calibration|select-freeze|audit|aggregate|dry-alpha-development|alpha-development|select-alpha-freeze|dry-alpha-audit|alpha-audit|legacy-alpha-diagnostic}" >&2
  exit 2
fi

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "missing official environment: ${VENV}/bin/activate" >&2
  echo "run: bash experiments/cluster/setup_group_volume.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
cd "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/experiments/cluster/cluster_env.sh"
export PYTHONUNBUFFERED=1
mkdir -p "$CLUSTER_RUNS_ROOT/logs"

CAMPAIGN_LOG_DIR="$CLUSTER_RUNS_ROOT/logs/channel_matrix"
mkdir -p "$CAMPAIGN_LOG_DIR"
CAMPAIGN_LOG="$CAMPAIGN_LOG_DIR/${MODEL_ID}_${ACTION}_$(hostname)_$(date -u '+%Y%m%dT%H%M%SZ')_$$.log"
ln -sfn "$(basename "$CAMPAIGN_LOG")" \
  "$CAMPAIGN_LOG_DIR/${MODEL_ID}_${ACTION}_$(hostname)_current.log"
exec > >(tee -a "$CAMPAIGN_LOG") 2>&1
echo "[campaign] action=$ACTION config=$CONFIG model=$MODEL_ID gpu=$GPU"
echo "[campaign] repo=$ROOT commit=$(git rev-parse --short HEAD) log=$CAMPAIGN_LOG"
echo "[storage] runs=$CLUSTER_RUNS_ROOT runtime=$CLUSTER_WORK_ROOT hf=$HF_HOME"

on_campaign_error() {
  local code=$?
  trap - ERR
  printf '[ERROR] campaign action=%s exit=%s line=%s command=%s\n' \
    "$ACTION" "$code" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}"
  printf '[ERROR] action log retained at %s\n' "$CAMPAIGN_LOG"
  exit "$code"
}
trap on_campaign_error ERR

model_args=()
if [[ "${MODEL_ID}" != "all" ]]; then
  model_args=(--model-id "${MODEL_ID}")
fi

author_args=()
if [[ -n "${AUTHORS}" ]]; then
  author_args=(--only-authors "${AUTHORS}")
fi

preflight() {
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "HF_HOME=${HF_HOME}"
  echo "CUDA_VISIBLE_DEVICES=${GPU}"
  echo "AUTHORS=${AUTHORS:-all-phase-authors}"
  git status --short
  nvidia-smi
  CUDA_VISIBLE_DEVICES="${GPU}" python - "${CONFIG}" "${MODEL_ID}" <<'PY'
from pathlib import Path
import sys

import datasets
import torch
import transformers
import yaml

config_path, selected = sys.argv[1:]
cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
models = [model for model in cfg["models"] if model.get("enabled", True)]
if selected != "all":
    models = [model for model in models if model["id"] == selected]
if not models:
    raise SystemExit(f"no enabled model matches MODEL_ID={selected!r}")
missing = [f"{model['id']}={model['path']}" for model in models if not Path(model["path"]).is_dir()]
if missing:
    raise SystemExit("missing model path(s): " + ", ".join(missing))
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
print(f"transformers={transformers.__version__} datasets={datasets.__version__}")
print("models=" + ",".join(model["id"] for model in models))
PY
}

prefetch() {
  # Audit itself is forced offline. Populate every non-model dependency before
  # the freeze and audit boundary.
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
  python - <<'PY'
from datasets import load_dataset

for subset in ("full", "forget10_perturbed"):
    split = load_dataset("locuslab/TOFU", subset)["train"]
    print(f"cached locuslab/TOFU/{subset}: {len(split)} rows")
PY
}

campaign_paths() {
  python - "${CONFIG}" "${MODEL_ID}" "${CLUSTER_RUNS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

import yaml

config_path, model_id, runs_root = sys.argv[1:]
cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
if not isinstance(cfg, dict):
    raise SystemExit(f"invalid campaign config: {config_path}")
enabled = {
    str(model["id"])
    for model in cfg.get("models", [])
    if isinstance(model, dict) and model.get("enabled", True)
}
if model_id == "all" or model_id not in enabled:
    raise SystemExit(
        f"aggregate requires one enabled MODEL_ID; found {model_id!r}, enabled={sorted(enabled)}"
    )
output_root = Path(str(cfg.get("output_root", "")))
if not output_root.parts:
    raise SystemExit(f"missing output_root in {config_path}")
if not output_root.is_absolute():
    if output_root.parts[0] == "runs":
        output_root = Path(runs_root, *output_root.parts[1:])
    else:
        output_root = Path.cwd() / output_root
tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)
print(output_root.resolve())
print(tag)
PY
}

run_phase() {
  local phase="$1"
  local phase_author_args=()
  if [[ "${phase}" != "fidelity" ]]; then
    phase_author_args=("${author_args[@]}")
  elif [[ -n "${AUTHORS}" ]]; then
    echo "AUTHORS is not applicable to the single frozen fidelity cell" >&2
    return 2
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python -u experiments/channel_matrix/run_campaign.py \
    --config "${CONFIG}" \
    --phase "${phase}" \
    --resume \
    "${phase_author_args[@]}" \
    "${model_args[@]}"
}

run_alpha_phase() {
  local phase="$1"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u experiments/channel_matrix/alpha_protection.py \
    --config "${CONFIG}" \
    --phase "${phase}" \
    --resume \
    "${author_args[@]}" \
    "${model_args[@]}"
}

case "${ACTION}" in
  preflight)
    preflight
    ;;
  prefetch)
    prefetch
    ;;
  dry-calibration)
    CUDA_VISIBLE_DEVICES="${GPU}" python experiments/channel_matrix/run_campaign.py \
      --config "${CONFIG}" \
      --phase calibration \
      --dry-run \
      "${author_args[@]}" \
      "${model_args[@]}"
    ;;
  fidelity)
    preflight
    run_phase fidelity
    ;;
  calibration)
    preflight
    run_phase calibration
    ;;
  select-freeze)
    python experiments/channel_matrix/select_freeze.py \
      --config "${CONFIG}" \
      --root "$CLUSTER_RUNS_ROOT/channel_matrix_7b/calibration" \
      --out "$CLUSTER_RUNS_ROOT/channel_matrix_7b/objective_freeze.recommended.yaml"
    echo "STOP: review the recommendation and commit a frozen objective_freeze.yaml before audit."
    ;;
  audit)
    if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
      echo "refusing audit: git worktree is dirty" >&2
      git status --short >&2
      exit 1
    fi
    preflight
    run_phase audit
    ;;
  aggregate)
    mapfile -t campaign_values < <(campaign_paths)
    if (( ${#campaign_values[@]} != 2 )); then
      echo "failed to resolve campaign output root and model tag" >&2
      exit 2
    fi
    CAMPAIGN_ROOT="${campaign_values[0]}"
    MODEL_TAG="${campaign_values[1]}"
    SCALE_LABEL="${SCALE_LABEL:-${MODEL_TAG#qwen25_}}"
    AGGREGATE_ROOT="$CAMPAIGN_ROOT/aggregate"
    echo "[aggregate] config=$CONFIG model=$MODEL_ID root=$CAMPAIGN_ROOT"
    python experiments/channel_matrix/aggregate.py \
      --root "$CAMPAIGN_ROOT/audit" \
      --out "$AGGREGATE_ROOT" \
      --config "$CONFIG" \
      --model-id "$MODEL_ID" \
      --n-boot 2000
    python experiments/channel_matrix/make_main_table.py \
      --report "$AGGREGATE_ROOT/pooled_channel_report.csv" \
      --summary "$AGGREGATE_ROOT/pooled_channel_report.json" \
      --out "$AGGREGATE_ROOT/table1_channel_matrix_${MODEL_TAG}.tex" \
      --stress-out "$AGGREGATE_ROOT/table1_stress_${MODEL_TAG}.tex" \
      --scale-label "$SCALE_LABEL" \
      --table-label "tab:channel-matrix-${MODEL_TAG}" \
      --stress-label "tab:channel-stress-${MODEL_TAG}"
    echo "[RESULT] aggregate=$AGGREGATE_ROOT"
    echo "[RESULT] latex=$AGGREGATE_ROOT/table1_channel_matrix_${MODEL_TAG}.tex"
    ;;
  dry-alpha-development)
    CUDA_VISIBLE_DEVICES="${GPU}" python experiments/channel_matrix/alpha_protection.py \
      --config "${CONFIG}" \
      --phase development \
      --dry-run \
      "${author_args[@]}" \
      "${model_args[@]}"
    ;;
  alpha-development)
    preflight
    run_alpha_phase development
    ;;
  select-alpha-freeze)
    python experiments/channel_matrix/select_alpha_freeze.py \
      --config "${CONFIG}" \
      --root "$CLUSTER_RUNS_ROOT/channel_matrix_7b/alpha_protection/development" \
      --out "$CLUSTER_RUNS_ROOT/channel_matrix_7b/alpha_protection_freeze.recommended.yaml"
    echo "STOP: review and commit configs/channel_matrix/alpha_protection_freeze.yaml before alpha audit."
    ;;
  dry-alpha-audit)
    CUDA_VISIBLE_DEVICES="${GPU}" python experiments/channel_matrix/alpha_protection.py \
      --config "${CONFIG}" \
      --phase audit \
      --dry-run \
      "${author_args[@]}" \
      "${model_args[@]}"
    ;;
  alpha-audit)
    if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
      echo "refusing alpha audit: git worktree is dirty" >&2
      git status --short >&2
      exit 1
    fi
    preflight
    run_alpha_phase audit
    ;;
  legacy-alpha-diagnostic)
    python experiments/channel_matrix/aggregate_alpha_protection.py \
      --legacy-diagnostic \
      --config "${CONFIG}" \
      --root "$CLUSTER_RUNS_ROOT/channel_matrix_7b/alpha_protection/audit" \
      --out "$CLUSTER_RUNS_ROOT/channel_matrix_7b/alpha_protection/aggregate" \
      --n-boot 2000
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    exit 2
    ;;
esac
