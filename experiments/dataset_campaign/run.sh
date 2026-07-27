#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATASET="${DATASET_CAMPAIGN:-}"
SCALE="${DATASET_SCALE:-}"
ACTION="${1:-}"
if (( $# != 1 )); then
  echo "usage: DATASET_CAMPAIGN=<dataset> DATASET_SCALE=<scale> $0 ACTION" >&2
  echo "actions: all | preflight | plan | calibration | freeze | audit | aggregate | render | status" >&2
  exit 2
fi
case "$DATASET" in
  wmdp_bio|muse_news|rwku|muse_books|pistol) ;;
  *)
    echo "invalid DATASET_CAMPAIGN=$DATASET" >&2
    exit 2
    ;;
esac
case "$SCALE" in
  1p5b|7b|14b) ;;
  *)
    echo "invalid DATASET_SCALE=$SCALE" >&2
    exit 2
    ;;
esac
case "$ACTION" in
  all|preflight|plan|calibration|freeze|audit|aggregate|render|status) ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac

MODEL_ID="qwen25_${SCALE}"
if [[ "$SCALE" == "1p5b" ]]; then
  MODEL_PATH="${MODEL_PATH:-/rdata/models/Qwen2.5-1.5B-Instruct}"
  RUN_ROOT="${RUN_ROOT:-/rdata/minsoo3.kim/results/dataset_campaigns/${DATASET}_${MODEL_ID}}"
  GPU_IDS="${GPU_IDS:-0,1}"
  export HF_HOME="${HF_HOME:-/rdata/minsoo3.kim/hf_home}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export FDMU_CAMPAIGN_RUNS_ROOT="${FDMU_CAMPAIGN_RUNS_ROOT:-/rdata/minsoo3.kim/results}"
  DEFAULT_PYTHON="$ROOT/.venv/bin/python"
else
  STORAGE_ROOT="${FDMU_STORAGE_ROOT:-/group-volume/fdmu}"
  if [[ "$ACTION" != "plan" && "$ACTION" != "status" ]]; then
    if [[ "${BOOTSTRAP_CLUSTER_ENV:-1}" == "1" ]]; then
      bash "$ROOT/experiments/cluster/setup_group_volume.sh"
    fi
    # shellcheck disable=SC1091
    source "$ROOT/experiments/cluster/cluster_env.sh"
  else
    export FDMU_CAMPAIGN_RUNS_ROOT="${FDMU_CAMPAIGN_RUNS_ROOT:-$STORAGE_ROOT/runs}"
  fi
  if [[ "$SCALE" == "7b" ]]; then
    MODEL_PATH="${MODEL_PATH:-/group-volume/models/Qwen2.5-7B-Instruct}"
  else
    MODEL_PATH="${MODEL_PATH:-/group-volume/models/Qwen2.5-14B-Instruct}"
  fi
  RUN_ROOT="${RUN_ROOT:-$FDMU_CAMPAIGN_RUNS_ROOT/dataset_campaigns/${DATASET}_${MODEL_ID}}"
  GPU_IDS="${GPU_IDS:-0,1,2,3}"
  DEFAULT_PYTHON="$STORAGE_ROOT/.venv/bin/python"
fi

PYTHON="${FDMU_PYTHON:-$DEFAULT_PYTHON}"
if [[ "$SCALE" == "1p5b" \
  && "$ACTION" != "plan" \
  && "$ACTION" != "status" \
  && "${BOOTSTRAP_4090_ENV:-1}" == "1" ]]; then
  bash "$ROOT/local_run/bootstrap_4090_env.sh"
fi
if [[ "$ACTION" == "plan" && ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] Python environment is missing: $PYTHON" >&2
  exit 2
fi

CONFIG="$RUN_ROOT/runtime/campaign.yaml"
RECOMMENDATION="$RUN_ROOT/freeze/objective_freeze.recommended.yaml"
FREEZE="$RUN_ROOT/freeze/objective_freeze.yaml"
AGGREGATE_ROOT="$RUN_ROOT/aggregate"
LATEX="$AGGREGATE_ROOT/table_channel_matrix.tex"
LOG_DIR="$RUN_ROOT/launcher_logs"
mkdir -p "$LOG_DIR" "$RUN_ROOT/freeze"
LOG="$LOG_DIR/${ACTION}_$(date -u '+%Y%m%dT%H%M%SZ')_$$.log"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/current.log"
exec > >(tee -a "$LOG") 2>&1

STAGE=initialization
on_error() {
  local code=$?
  trap - ERR
  printf '[ERROR] stage=%s exit=%s line=%s command=%q\n' \
    "$STAGE" "$code" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}" >&2
  printf '[CONTEXT] dataset=%s scale=%s action=%s run_root=%s config=%s log=%s\n' \
    "$DATASET" "$SCALE" "$ACTION" "$RUN_ROOT" "$CONFIG" "$LOG" >&2
  if [[ -f "$RUN_ROOT/launcher_logs/calibration/LAST_SUMMARY.json" ]]; then
    printf '%s\n' '----- calibration summary -----' >&2
    cat "$RUN_ROOT/launcher_logs/calibration/LAST_SUMMARY.json" >&2
  fi
  if [[ -f "$RUN_ROOT/launcher_logs/audit/LAST_SUMMARY.json" ]]; then
    printf '%s\n' '----- audit summary -----' >&2
    cat "$RUN_ROOT/launcher_logs/audit/LAST_SUMMARY.json" >&2
  fi
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader 2>&1 || true
  df -h "$RUN_ROOT" 2>&1 || true
  exit "$code"
}
trap on_error ERR

stage() {
  STAGE="$1"
  printf '\n========== [STAGE] %s time=%s ==========\n' \
    "$STAGE" "$(date -u '+%FT%TZ')"
}

materialize() {
  stage materialize-config
  "$PYTHON" -u experiments/dataset_campaign/materialize_config.py \
    --dataset "$DATASET" \
    --scale "$SCALE" \
    --model-path "$MODEL_PATH" \
    --output-root "$RUN_ROOT" \
    --out "$CONFIG"
}

preflight() {
  stage preflight
  "$PYTHON" -u experiments/dataset_campaign/preflight.py --config "$CONFIG"
  printf '[GPU] requested=%s\n' "$GPU_IDS"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
}

calibration() {
  stage calibration
  "$PYTHON" -u experiments/dataset_campaign/parallel_phase.py \
    --config "$CONFIG" \
    --phase calibration \
    --gpu-ids "$GPU_IDS"
}

freeze() {
  stage development-selection
  "$PYTHON" -u experiments/channel_matrix/select_freeze.py \
    --config "$CONFIG" \
    --root "$RUN_ROOT/calibration" \
    --out "$RECOMMENDATION"
  "$PYTHON" -u experiments/dataset_campaign/freeze_from_calibration.py \
    --config "$CONFIG" \
    --recommendation "$RECOMMENDATION" \
    --out "$FREEZE"
}

audit() {
  stage sealed-audit
  if [[ ! -s "$FREEZE" ]]; then
    echo "[ERROR] objective freeze is missing: $FREEZE" >&2
    echo "[NEXT] run the explicit freeze action after calibration" >&2
    return 2
  fi
  "$PYTHON" -u experiments/dataset_campaign/parallel_phase.py \
    --config "$CONFIG" \
    --phase audit \
    --gpu-ids "$GPU_IDS"
}

render_existing() {
  stage latex-render
  if [[ ! -s "$AGGREGATE_ROOT/pooled_channel_report.csv" ]]; then
    echo "[ERROR] aggregate CSV is missing: $AGGREGATE_ROOT/pooled_channel_report.csv" >&2
    return 2
  fi
  mapfile -t labels < <(
    "$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["dataset_label"])
print(cfg["model_label"])
PY
  )
  "$PYTHON" -u experiments/dataset_campaign/render_latex.py \
    --csv "$AGGREGATE_ROOT/pooled_channel_report.csv" \
    --dataset "${labels[0]}" \
    --model "${labels[1]}" \
    --out "$LATEX"
  printf '%s\n' '----- FINAL LATEX -----'
  cat "$LATEX"
  printf '%s\n' '----- END FINAL LATEX -----'
}

aggregate() {
  stage aggregate
  "$PYTHON" -u experiments/channel_matrix/aggregate.py \
    --root "$RUN_ROOT/audit" \
    --out "$AGGREGATE_ROOT" \
    --config "$CONFIG" \
    --model-id "$MODEL_ID" \
    --n-boot "${N_BOOT:-2000}"
  render_existing
}

status() {
  stage status
  printf '[STATUS] dataset=%s scale=%s run_root=%s\n' \
    "$DATASET" "$SCALE" "$RUN_ROOT"
  for path in \
    "$RUN_ROOT/launcher_logs/calibration/LAST_SUMMARY.json" \
    "$RUN_ROOT/launcher_logs/audit/LAST_SUMMARY.json" \
    "$FREEZE" \
    "$AGGREGATE_ROOT/pooled_channel_report.json" \
    "$LATEX"; do
    if [[ -s "$path" ]]; then
      printf '[FOUND] %s\n' "$path"
    else
      printf '[MISSING] %s\n' "$path"
    fi
  done
  if [[ -L "$LOG_DIR/current.log" || -f "$LOG_DIR/current.log" ]]; then
    printf '%s\n' '----- current launcher tail -----'
    tail -n 80 "$LOG_DIR/current.log" || true
  fi
}

materialize
case "$ACTION" in
  all)
    preflight
    calibration
    freeze
    audit
    aggregate
    ;;
  preflight)
    preflight
    ;;
  plan)
    stage dry-calibration-plan
    "$PYTHON" -u experiments/channel_matrix/run_campaign.py \
      --config "$CONFIG" --phase calibration --dry-run
    ;;
  calibration)
    preflight
    calibration
    ;;
  freeze)
    freeze
    ;;
  audit)
    preflight
    audit
    ;;
  aggregate)
    aggregate
    ;;
  render)
    render_existing
    ;;
  status)
    status
    ;;
esac

printf '\n[RESULT] action=%s complete dataset=%s scale=%s\n' \
  "$ACTION" "$DATASET" "$SCALE"
printf '[RESULT] run_root=%s\n' "$RUN_ROOT"
printf '[RESULT] aggregate_json=%s\n' "$AGGREGATE_ROOT/pooled_channel_report.json"
printf '[RESULT] latex=%s\n' "$LATEX"
printf '[RESULT] launcher_log=%s\n' "$LOG"
