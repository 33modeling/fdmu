#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset PYTHONHOME PYTHONPATH PIP_TARGET PIP_PREFIX
export VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"

RUN_ROOT="${RUN_ROOT:-/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b}"
export RUN_ROOT
export CALIBRATION_ROOT="${CALIBRATION_ROOT:-$RUN_ROOT/parent_calibration}"
export SFT_CACHE_ROOT="${SFT_CACHE_ROOT:-$RUN_ROOT/sft_cache}"
export RESULTS_ROOT="${RESULTS_ROOT:-$RUN_ROOT/joint_sweep}"
export JOINT_ROOT="${JOINT_ROOT:-$RESULTS_ROOT}"
export FINAL_ROOT="${FINAL_ROOT:-$RUN_ROOT/final}"
export FIDELITY_ROOT="${FIDELITY_ROOT:-$RUN_ROOT/fidelity}"
export PARENT_FREEZE="${PARENT_FREEZE:-$CALIBRATION_ROOT/freeze/tofu_parent_freeze_1p5b.yaml}"

timestamp() {
  date -u '+%FT%TZ'
}

mkdir -p "$RUN_ROOT/launcher_logs"
PIPELINE_TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
PIPELINE_LOG="$RUN_ROOT/launcher_logs/${PIPELINE_TIMESTAMP}-$$.log"
ln -sfn "$(basename "$PIPELINE_LOG")" "$RUN_ROOT/launcher_logs/current.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

STAGE=bootstrap
STAGE_INDEX=0
if [[ "${RUN_FINALIZE:-1}" == "1" ]]; then
  STAGE_TOTAL=6
else
  STAGE_TOTAL=4
fi
STAGE_STARTED_EPOCH="$(date +%s)"
STAGE_HEARTBEAT_PID=
ACTIVE_STAGE_PID=
ACTIVE_STAGE_PGID=
STATUS_FILE="$RUN_ROOT/CURRENT_STAGE.txt"
ERROR_FILE="$RUN_ROOT/LAST_ERROR.txt"
PIPELINE_HEARTBEAT_SECONDS="${PIPELINE_HEARTBEAT_SECONDS:-30}"
PIPELINE_CLEANUP_GRACE_SECONDS="${PIPELINE_CLEANUP_GRACE_SECONDS:-5}"

write_stage_status() {
  local state="$1"
  local exit_code="${2:-}"
  local now_epoch elapsed temporary
  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - STAGE_STARTED_EPOCH))
  temporary="${STATUS_FILE}.tmp.${BASHPID:-$$}"
  {
    printf 'state=%s\n' "$state"
    printf 'stage=%s\n' "$STAGE"
    printf 'stage_index=%s\n' "$STAGE_INDEX"
    printf 'stage_total=%s\n' "$STAGE_TOTAL"
    printf 'elapsed_seconds=%s\n' "$elapsed"
    printf 'pid=%s\n' "$$"
    printf 'updated_at_utc=%s\n' "$(timestamp)"
    printf 'log=%s\n' "$PIPELINE_LOG"
    [[ -z "$exit_code" ]] || printf 'exit_code=%s\n' "$exit_code"
  } > "$temporary"
  mv -f "$temporary" "$STATUS_FILE"
}

stop_stage_heartbeat() {
  if [[ -n "${STAGE_HEARTBEAT_PID:-}" ]]; then
    kill "$STAGE_HEARTBEAT_PID" 2>/dev/null || true
    wait "$STAGE_HEARTBEAT_PID" 2>/dev/null || true
    STAGE_HEARTBEAT_PID=
  fi
}

cleanup_active_stage() {
  local reason="${1:-pipeline-exit}"
  local elapsed=0
  if [[ -z "${ACTIVE_STAGE_PGID:-}" ]]; then
    return 0
  fi
  if kill -0 -- "-$ACTIVE_STAGE_PGID" 2>/dev/null; then
    printf '[CLEANUP] reason=%s stage=%s pgid=%s signal=TERM\n' \
      "$reason" "$STAGE" "$ACTIVE_STAGE_PGID" >&2
    kill -TERM -- "-$ACTIVE_STAGE_PGID" 2>/dev/null || true
    while kill -0 -- "-$ACTIVE_STAGE_PGID" 2>/dev/null \
      && (( elapsed < PIPELINE_CLEANUP_GRACE_SECONDS * 10 )); do
      sleep 0.1
      elapsed=$((elapsed + 1))
    done
    if kill -0 -- "-$ACTIVE_STAGE_PGID" 2>/dev/null; then
      printf '[CLEANUP] reason=%s stage=%s pgid=%s signal=KILL\n' \
        "$reason" "$STAGE" "$ACTIVE_STAGE_PGID" >&2
      kill -KILL -- "-$ACTIVE_STAGE_PGID" 2>/dev/null || true
    fi
  fi
  if [[ -n "${ACTIVE_STAGE_PID:-}" ]]; then
    wait "$ACTIVE_STAGE_PID" 2>/dev/null || true
  fi
  ACTIVE_STAGE_PID=
  ACTIVE_STAGE_PGID=
}

run_isolated_stage_command() {
  local status
  setsid -- "$@" &
  ACTIVE_STAGE_PID=$!
  ACTIVE_STAGE_PGID=$ACTIVE_STAGE_PID
  printf '[PROCESS] stage=%s pid=%s pgid=%s isolated=true\n' \
    "$STAGE" "$ACTIVE_STAGE_PID" "$ACTIVE_STAGE_PGID"
  if wait "$ACTIVE_STAGE_PID"; then
    status=0
  else
    status=$?
  fi
  if (( status == 0 )); then
    cleanup_active_stage stage-complete
  fi
  return "$status"
}

begin_stage() {
  STAGE_INDEX="$1"
  STAGE="$2"
  STAGE_STARTED_EPOCH="$(date +%s)"
  printf '\n========== [STAGE %s/%s] %s START time=%s ==========\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL" "$STAGE" "$(timestamp)"
  write_stage_status running
  (
    while sleep "$PIPELINE_HEARTBEAT_SECONDS"; do
      elapsed=$(( $(date +%s) - STAGE_STARTED_EPOCH ))
      write_stage_status running
      printf '[STAGE %s/%s] RUNNING name=%s elapsed_seconds=%s status=%s\n' \
        "$STAGE_INDEX" "$STAGE_TOTAL" "$STAGE" "$elapsed" "$STATUS_FILE"
    done
  ) &
  STAGE_HEARTBEAT_PID=$!
}

complete_stage() {
  stop_stage_heartbeat
  write_stage_status completed
  printf '========== [STAGE %s/%s] %s COMPLETE elapsed_seconds=%s ==========\n\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL" "$STAGE" \
    "$(( $(date +%s) - STAGE_STARTED_EPOCH ))"
}

on_error() {
  local code="${1:-$?}"
  local line="${BASH_LINENO[0]:-unknown}"
  local command="${BASH_COMMAND:-unknown}"
  local temporary="${ERROR_FILE}.tmp.${BASHPID:-$$}"
  trap - ERR
  stop_stage_heartbeat
  cleanup_active_stage "stage-failed-exit-$code"
  write_stage_status failed "$code"
  {
    printf 'state=failed\n'
    printf 'stage=%s\n' "$STAGE"
    printf 'stage_index=%s\n' "$STAGE_INDEX"
    printf 'stage_total=%s\n' "$STAGE_TOTAL"
    printf 'exit_code=%s\n' "$code"
    printf 'line=%s\n' "$line"
    printf 'command=%q\n' "$command"
    printf 'failed_at_utc=%s\n' "$(timestamp)"
    printf 'log=%s\n' "$PIPELINE_LOG"
    if [[ "$STAGE" == "joint-sweep" ]]; then
      if [[ -s "$JOINT_ROOT/LATEST_FAILURE.txt" ]]; then
        printf '\n'
        printf 'joint_failure_summary=%s\n' "$JOINT_ROOT/LATEST_FAILURE.txt"
        printf '\n'
        cat "$JOINT_ROOT/LATEST_FAILURE.txt"
      elif [[ -e "$JOINT_ROOT/launcher_logs/current.log" ]]; then
        printf '\n'
        printf 'joint_failure_summary=unavailable; joint launcher tail follows\n'
        printf '%s\n' '----- JOINT LAUNCHER LOG TAIL (last 160 lines) -----'
        tail -n 160 "$JOINT_ROOT/launcher_logs/current.log" 2>/dev/null || true
        printf '%s\n' '----- END JOINT LAUNCHER LOG TAIL -----'
      fi
    fi
  } > "$temporary"
  mv -f "$temporary" "$ERROR_FILE"
  printf '\n========== [STAGE %s/%s] %s FAILED ==========\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL" "$STAGE" >&2
  printf '[%s] [ERROR] exit=%s line=%s command=%q\n' \
    "$(timestamp)" "$code" "$line" "$command" >&2
  printf '%s\n' '----- COMPLETE FAILURE REPORT -----' >&2
  cat "$ERROR_FILE" >&2
  printf '%s\n' '----- END COMPLETE FAILURE REPORT -----' >&2
  printf '[CONTEXT] run_root=%s calibration=%s sweep=%s final=%s log=%s\n' \
    "$RUN_ROOT" "$CALIBRATION_ROOT" "$JOINT_ROOT" "$FINAL_ROOT" "$PIPELINE_LOG"
  df -h "$RUN_ROOT" 2>&1 || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
      --format=csv,noheader 2>&1 || true
  fi
  printf '[ERROR] unified pipeline log retained at %s\n' "$PIPELINE_LOG"
  printf '[ERROR] machine-readable failure summary: %s\n' "$ERROR_FILE"
  printf '[GUIDE] %s/docs/LLM_RUN_DIAGNOSTICS.md\n' "$ROOT"
  exit "$code"
}
trap on_error ERR
trap 'on_error 130' INT
trap 'on_error 143' TERM
trap 'code=$?; stop_stage_heartbeat; if (( code != 0 )); then cleanup_active_stage "pipeline-exit-$code"; fi' EXIT

run_stage() {
  local index="$1"
  local name="$2"
  shift 2
  begin_stage "$index" "$name"
  printf '[%s] command:' "$(timestamp)"
  printf ' %q' "$@"
  printf '\n'
  run_isolated_stage_command "$@"
  complete_stage
}

run_stage_accept() {
  local index="$1"
  local name="$2"
  local accepted="$3"
  local status
  shift 3
  begin_stage "$index" "$name"
  printf '[%s] command:' "$(timestamp)"
  printf ' %q' "$@"
  printf '\n'
  set +e
  run_isolated_stage_command "$@"
  status=$?
  set -e
  if [[ " $accepted " != *" $status "* ]]; then
    stop_stage_heartbeat
    write_stage_status failed "$status"
    printf '[%s] pipeline stage failed: %s exit=%s accepted=%s\n' \
      "$(timestamp)" "$name" "$status" "$accepted" >&2
    return "$status"
  fi
  cleanup_active_stage "stage-accepted-exit-$status"
  complete_stage
}

printf '[%s] TOFU 1.5B 4090x2 pipeline start pid=%s log=%s\n' \
  "$(timestamp)" "$$" "$PIPELINE_LOG"
printf '[RESUME] this is the only operator command; completed calibration, sweep, and final units are validated and reused\n'
printf '[RESUME] after a failure, rerun this same command without deleting RUN_ROOT or SFT cache\n'
printf '[CONFIG] repo=%s python=%s gpus=%s run_root=%s\n' \
  "$ROOT" "$PYTHON" "${GPU_IDS:-0,1}" "$RUN_ROOT"
printf '[STORAGE] calibration=%s sft_cache=%s sweep=%s final=%s\n' \
  "$CALIBRATION_ROOT" "$SFT_CACHE_ROOT" "$JOINT_ROOT" "$FINAL_ROOT"
printf '[STORAGE] parent_freeze=%s fidelity=%s\n' "$PARENT_FREEZE" "$FIDELITY_ROOT"
printf '[STATUS] current_stage=%s\n' "$STATUS_FILE"
printf '[STATUS] last_error=%s\n' "$ERROR_FILE"
rm -f "$ERROR_FILE"
if ! command -v setsid >/dev/null 2>&1; then
  printf '[ERROR] setsid is required to isolate and clean up stage processes\n' >&2
  exit 2
fi
df -h "$RUN_ROOT" 2>&1 || true
run_stage 1 environment-bootstrap bash local_run/bootstrap_4090_env.sh
export FDMU_4090_BOOTSTRAPPED=1
# Exit 4 is accepted for calibration artifacts produced by older checkouts.
# Current calibration exits 0 once its target-free proposal is resolved.
run_stage_accept 2 calibration "0 4" bash local_run/run_tofu_1p5b_calibration.sh
printf '[AUTO] calibration is resolved; creating or validating the parent freeze without operator input\n'
run_stage 3 automatic-parent-freeze \
  bash local_run/approve_tofu_1p5b_parent_freeze.sh --approve
run_stage 4 joint-sweep bash local_run/sweep_joint_1p5b_4090x2.sh "$@"
if [[ "${RUN_FINALIZE:-1}" == "1" ]]; then
  run_stage 5 declared-fidelity bash local_run/run_tofu_1p5b_fidelity.sh
  run_stage 6 target-evidence-latex \
    bash local_run/finalize_joint_sweep_to_latex.sh
else
  printf '[%s] finalize skipped: RUN_FINALIZE=%s\n' \
    "$(timestamp)" "${RUN_FINALIZE:-}"
  printf '[NEXT] resume validated finalization with:\n'
  printf '  GPU_IDS=%s RUN_ROOT=%q bash local_run/finalize_joint_sweep_to_latex.sh\n' \
    "${GPU_IDS:-0,1}" "$RUN_ROOT"
fi
STAGE=complete
STAGE_INDEX="$STAGE_TOTAL"
STAGE_STARTED_EPOCH="$(date +%s)"
write_stage_status completed
printf '[%s] TOFU 1.5B 4090x2 pipeline complete\n' "$(timestamp)"
printf '[RESULT] best=%s/BEST.json\n' "$JOINT_ROOT"
if [[ "${RUN_FINALIZE:-1}" == "1" ]]; then
  printf '[RESULT] latex=%s/table1.tex\n' "$FINAL_ROOT"
fi
printf '[RESULT] unified_log=%s\n' "$PIPELINE_LOG"
