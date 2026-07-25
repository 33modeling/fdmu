#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Keep the 7B fleet isolated from inherited 14B/local overrides. The cluster
# checkout itself is on group-volume, so this remains shared and writable
# without depending on the permission-conflicted legacy runs directory.
export CLUSTER_LOCAL_ENV=/dev/null
export CLUSTER_RUNS_ROOT="$ROOT/.cluster-runs/7b"
export CLUSTER_WORK_ROOT="$CLUSTER_RUNS_ROOT/_runtime"

# shellcheck disable=SC1091
source "$ROOT/experiments/cluster/cluster_env.sh"

QUEUE="$CLUSTER_RUNS_ROOT/cluster_queue/wave2"
VENV="${VENV:-/group-volume/jieuns.shin/venvs/exp}"
PYTHON="$VENV/bin/python"
MODEL_ID=qwen25_7b
LOG_DIR="$CLUSTER_RUNS_ROOT/logs/cluster"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/launcher_${MODEL_ID}_$(hostname)_$(date -u '+%Y%m%dT%H%M%SZ').out"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/launcher_${MODEL_ID}_$(hostname)_current.out"
exec > >(tee -a "$LOG") 2>&1

STAGE=bootstrap

print_context() {
  local workers
  workers="$(pgrep -af "experiments/cluster/worker.py --queue" || true)"
  printf '[CONTEXT] host=%s model=%s queue=%s commit=%s python=%s\n' \
    "$(hostname)" "$MODEL_ID" "$QUEUE" "$(git rev-parse --short HEAD)" "$PYTHON"
  if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    printf '[CONTEXT] worktree=dirty\n'
    git status --short --untracked-files=all
  else
    printf '[CONTEXT] worktree=clean\n'
  fi
  if [[ -n "$workers" ]]; then
    printf '[CONTEXT] local_workers:\n%s\n' "$workers"
  else
    printf '[CONTEXT] local_workers=none\n'
  fi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader 2>&1 || true
}

on_error() {
  local code=$?
  trap - ERR
  printf '[ERROR] stage=%s exit=%s line=%s command=%s\n' \
    "$STAGE" "$code" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}"
  case "$STAGE" in
    worker-launch)
      printf '[ANALYSIS] worker launch failed: inspect local_workers below; '\
'an active worker for another queue must finish or be stopped on this host.\n'
      ;;
    enqueue)
      printf '[ANALYSIS] enqueue failed: check worktree state and frozen audit configs below.\n'
      ;;
    fidelity*)
      printf '[ANALYSIS] fidelity failed: inspect the final error lines above and GPU state below.\n'
      ;;
    *)
      printf '[ANALYSIS] stage command failed; queue and host summaries follow.\n'
      ;;
  esac
  print_context
  "$PYTHON" experiments/cluster/workqueue.py status --brief --queue "$QUEUE" 2>&1 || true
  printf '[ERROR] launcher log retained at %s\n' "$LOG"
  exit "$code"
}
trap on_error ERR

stage() {
  STAGE="$1"
  printf '[INFO] time=%s stage=%s start\n' "$(date -u '+%FT%TZ')" "$STAGE"
}

print_context

stage fidelity
GPU="${FIDELITY_GPU:-0}" \
CONFIG=configs/channel_matrix/7b_tofu.yaml \
MODEL_ID="$MODEL_ID" \
  bash experiments/channel_matrix/h100_campaign.sh fidelity

stage fidelity-contract-validation
"$PYTHON" experiments/channel_matrix/run_campaign.py \
  --config configs/channel_matrix/7b_tofu.yaml --phase audit \
  --model-id "$MODEL_ID" --only-authors 181 \
  --dry-run --limit 1

stage failed-audit-recovery
"$PYTHON" experiments/cluster/workqueue.py retry-failed \
  --queue "$QUEUE" \
  --unit aud__qwen25_7b__a181 \
  --unit aud__qwen25_7b__a186 \
  --unit aud__qwen25_7b__a191

stage enqueue
bash experiments/cluster/enqueue_table12.sh audit-7b
stage worker-launch
bash experiments/cluster/launch_node.sh "$QUEUE"
printf '[RESULT] worker-launch complete; active local workers:\n'
pgrep -af "experiments/cluster/worker.py --queue" || true
"$PYTHON" experiments/cluster/workqueue.py status --brief --queue "$QUEUE"
stage queue-monitor
"$PYTHON" -u experiments/cluster/monitor_queue.py \
  --queue "$QUEUE" --match "$MODEL_ID" \
  --interval "${MONITOR_INTERVAL_SECONDS:-30}"
