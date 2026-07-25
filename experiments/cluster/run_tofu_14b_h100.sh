#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/cluster/cluster_env.sh"

QUEUE="$CLUSTER_RUNS_ROOT/cluster_queue/wave1_14b"
VENV="/group-volume/fdmu/.venv"
PYTHON="$VENV/bin/python"
MODEL_ID=qwen25_14b
LOG_DIR="$CLUSTER_RUNS_ROOT/logs/cluster"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/launcher_${MODEL_ID}_$(hostname)_$(date -u '+%Y%m%dT%H%M%SZ').out"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/launcher_${MODEL_ID}_$(hostname)_current.out"
exec > >(tee -a "$LOG") 2>&1

STAGE=bootstrap
on_error() {
  local code=$?
  trap - ERR
  printf '[ERROR] stage=%s exit=%s line=%s command=%s\n' \
    "$STAGE" "$code" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader 2>&1 || true
  "$PYTHON" experiments/cluster/workqueue.py status --brief --queue "$QUEUE" 2>&1 || true
  printf '[ERROR] full launcher log: %s\n' "$LOG"
  exit "$code"
}
trap on_error ERR

stage() {
  STAGE="$1"
  printf '[INFO] stage=%s start\n' "$STAGE"
}

stage fidelity
GPU="${FIDELITY_GPU:-0}" \
CONFIG=configs/channel_matrix/14b_tofu.yaml \
MODEL_ID="$MODEL_ID" \
  bash experiments/channel_matrix/h100_campaign.sh fidelity

stage fidelity-contract-validation
"$PYTHON" experiments/channel_matrix/run_campaign.py \
  --config configs/channel_matrix/14b_tofu.yaml --phase audit \
  --model-id "$MODEL_ID" --only-authors 181 \
  --dry-run --limit 1

stage failed-audit-recovery
"$PYTHON" experiments/cluster/workqueue.py retry-failed \
  --queue "$QUEUE" \
  --unit aud__qwen25_14b__a181 \
  --unit aud__qwen25_14b__a186 \
  --unit aud__qwen25_14b__a191

stage enqueue
bash experiments/cluster/enqueue_table12.sh audit-14b
stage worker-launch
bash experiments/cluster/launch_node.sh "$QUEUE"
stage queue-monitor
"$PYTHON" -u experiments/cluster/monitor_queue.py \
  --queue "$QUEUE" --match "$MODEL_ID" \
  --interval "${MONITOR_INTERVAL_SECONDS:-30}"
