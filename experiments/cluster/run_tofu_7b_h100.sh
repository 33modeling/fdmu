#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/experiments/cluster/cluster_env.sh"

QUEUE="$CLUSTER_RUNS_ROOT/cluster_queue/wave2"
VENV="/group-volume/fdmu/.venv"
PYTHON="$VENV/bin/python"
MODEL_ID=qwen25_7b
LOG_DIR="$CLUSTER_RUNS_ROOT/logs/cluster"
FIDELITY_CERT="$CLUSTER_RUNS_ROOT/channel_matrix_7b/fidelity/${MODEL_ID}.json"
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

stop_uncertified_local_audit() {
  local worker_pattern="experiments/cluster/worker.py --queue ${QUEUE}"
  local audit_pattern="run_campaign.py.*7b_tofu.yaml.*--phase audit"
  local attempt

  if [[ -s "$FIDELITY_CERT" ]]; then
    printf '[INFO] fidelity certificate present: %s\n' "$FIDELITY_CERT"
    return
  fi
  if ! pgrep -f "$worker_pattern|$audit_pattern" >/dev/null; then
    return
  fi

  printf '[RECOVERY] fidelity certificate missing; stopping local uncertified '\
'7B audit before fidelity: %s\n' "$FIDELITY_CERT"
  # Tell the worker to exit after its child, then terminate the invalid audit
  # child so it records a retryable failed unit and releases GPU 0.
  pkill -TERM -f "$worker_pattern" 2>/dev/null || true
  pkill -TERM -f "$audit_pattern" 2>/dev/null || true
  for attempt in {1..60}; do
    if ! pgrep -f "$worker_pattern|$audit_pattern" >/dev/null; then
      printf '[RECOVERY] local uncertified 7B audit stopped\n'
      return
    fi
    sleep 1
  done
  printf '[ERROR] local 7B audit did not stop within 60 seconds; '\
'refusing to double-book fidelity GPU\n' >&2
  pgrep -af "$worker_pattern|$audit_pattern" >&2 || true
  return 1
}

print_context

stage uncertified-audit-recovery
stop_uncertified_local_audit

stage fidelity
GPU="${FIDELITY_GPU:-0}" \
CONFIG=configs/channel_matrix/7b_tofu.yaml \
MODEL_ID="$MODEL_ID" \
  bash experiments/channel_matrix/h100_campaign.sh fidelity

stage failed-audit-partial-quarantine
"$PYTHON" experiments/cluster/quarantine_failed_audit.py \
  --queue "$QUEUE" \
  --config configs/channel_matrix/7b_tofu.yaml \
  --model-id "$MODEL_ID"

stage failed-audit-recovery
"$PYTHON" experiments/cluster/workqueue.py retry-failed \
  --queue "$QUEUE"

stage fidelity-contract-validation
"$PYTHON" experiments/channel_matrix/run_campaign.py \
  --config configs/channel_matrix/7b_tofu.yaml --phase audit \
  --model-id "$MODEL_ID" --only-authors 181 \
  --dry-run --limit 1

stage enqueue
bash experiments/cluster/enqueue_table12.sh audit-7b
stage worker-launch
bash experiments/cluster/launch_node.sh --dedicated-queue "$QUEUE"
printf '[RESULT] worker-launch complete; active local workers:\n'
pgrep -af "experiments/cluster/worker.py --queue" || true
"$PYTHON" experiments/cluster/workqueue.py status --brief --queue "$QUEUE"
stage queue-monitor
"$PYTHON" -u experiments/cluster/monitor_queue.py \
  --queue "$QUEUE" --match "$MODEL_ID" \
  --interval "${MONITOR_INTERVAL_SECONDS:-30}"
