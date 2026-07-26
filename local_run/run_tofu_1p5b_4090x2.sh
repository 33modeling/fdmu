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
# A one-command run must approve the winner only after it exists and is shown.
unset APPROVE_JOINT_BEST

timestamp() {
  date -u '+%FT%TZ'
}

mkdir -p "$RUN_ROOT/launcher_logs"
PIPELINE_TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
PIPELINE_LOG="$RUN_ROOT/launcher_logs/${PIPELINE_TIMESTAMP}-$$.log"
ln -sfn "$(basename "$PIPELINE_LOG")" "$RUN_ROOT/launcher_logs/current.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

STAGE=bootstrap

on_error() {
  local code=$?
  trap - ERR
  printf '[%s] [ERROR] stage=%s exit=%s line=%s command=%q\n' \
    "$(timestamp)" "$STAGE" "$code" "${BASH_LINENO[0]:-unknown}" \
    "${BASH_COMMAND:-unknown}"
  printf '[CONTEXT] run_root=%s calibration=%s sweep=%s final=%s log=%s\n' \
    "$RUN_ROOT" "$CALIBRATION_ROOT" "$JOINT_ROOT" "$FINAL_ROOT" "$PIPELINE_LOG"
  df -h "$RUN_ROOT" 2>&1 || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
      --format=csv,noheader 2>&1 || true
  fi
  printf '[ERROR] unified pipeline log retained at %s\n' "$PIPELINE_LOG"
  exit "$code"
}
trap on_error ERR

bash local_run/bootstrap_4090_env.sh

run_stage() {
  local name="$1"
  shift
  STAGE="$name"
  printf '[%s] pipeline stage start: %s\n' "$(timestamp)" "$name"
  printf '[%s] command:' "$(timestamp)"
  printf ' %q' "$@"
  printf '\n'
  "$@"
  printf '[%s] pipeline stage complete: %s\n' "$(timestamp)" "$name"
}

run_stage_accept() {
  local name="$1"
  local accepted="$2"
  local status
  shift 2
  STAGE="$name"
  printf '[%s] pipeline stage start: %s\n' "$(timestamp)" "$name"
  printf '[%s] command:' "$(timestamp)"
  printf ' %q' "$@"
  printf '\n'
  set +e
  "$@"
  status=$?
  set -e
  if [[ " $accepted " != *" $status "* ]]; then
    printf '[%s] pipeline stage failed: %s exit=%s accepted=%s\n' \
      "$(timestamp)" "$name" "$status" "$accepted" >&2
    return "$status"
  fi
  printf '[%s] pipeline stage complete: %s exit=%s\n' \
    "$(timestamp)" "$name" "$status"
}

require_file_approval() {
  local label="$1"
  local path="$2"
  local digest token response
  if [[ ! -f "$path" ]]; then
    printf '[ERROR] approval artifact is missing: %s\n' "$path" >&2
    return 2
  fi
  printf '[%s] review artifact for %s: %s\n' "$(timestamp)" "$label" "$path"
  printf '%s\n' '----- REVIEW ARTIFACT BEGIN -----'
  cat "$path"
  printf '%s\n' '----- REVIEW ARTIFACT END -----'
  if [[ ! -t 0 ]]; then
    printf '[ERROR] %s requires an interactive terminal; artifact was not approved\n' \
      "$label" >&2
    return 4
  fi
  digest="$("$PYTHON" - "$path" <<'PY'
import hashlib
import sys

print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"
  token="APPROVE ${label} ${digest:0:12}"
  printf '[APPROVAL] type exactly: %s\n> ' "$token"
  IFS= read -r response
  if [[ "$response" != "$token" ]]; then
    printf '[ERROR] %s approval rejected; expected exact typed token\n' "$label" >&2
    return 4
  fi
  printf '[%s] approval accepted: label=%s sha256=%s\n' \
    "$(timestamp)" "$label" "$digest"
}

printf '[%s] TOFU 1.5B 4090x2 pipeline start pid=%s log=%s\n' \
  "$(timestamp)" "$$" "$PIPELINE_LOG"
printf '[CONFIG] repo=%s python=%s gpus=%s run_root=%s\n' \
  "$ROOT" "$PYTHON" "${GPU_IDS:-0,1}" "$RUN_ROOT"
printf '[STORAGE] calibration=%s sft_cache=%s sweep=%s final=%s\n' \
  "$CALIBRATION_ROOT" "$SFT_CACHE_ROOT" "$JOINT_ROOT" "$FINAL_ROOT"
printf '[STORAGE] parent_freeze=%s fidelity=%s\n' "$PARENT_FREEZE" "$FIDELITY_ROOT"
df -h "$RUN_ROOT" 2>&1 || true
# Exit 4 is the successful protocol boundary: calibration is resolved and
# requires the explicit target-free approval performed by the next stage.
run_stage_accept calibration "0 4" bash local_run/run_tofu_1p5b_calibration.sh
run_stage parent-freeze-approval \
  bash local_run/approve_tofu_1p5b_parent_freeze.sh
run_stage joint-sweep bash local_run/sweep_joint_1p5b_4090x2.sh "$@"
if [[ "${RUN_FINALIZE:-1}" == "1" ]]; then
  run_stage declared-fidelity bash local_run/run_tofu_1p5b_fidelity.sh
  STAGE=joint-best-approval
  require_file_approval JOINT "$JOINT_ROOT/BEST.json"
  run_stage target-evidence-latex \
    env APPROVE_JOINT_BEST=1 bash local_run/finalize_joint_sweep_to_latex.sh
else
  printf '[%s] finalize skipped: RUN_FINALIZE=%s\n' \
    "$(timestamp)" "${RUN_FINALIZE:-}"
  printf '[NEXT] review %s/BEST.json, then run:\n' "$JOINT_ROOT"
  printf '  APPROVE_JOINT_BEST=1 GPU_IDS=%s RUN_ROOT=%q bash local_run/finalize_joint_sweep_to_latex.sh\n' \
    "${GPU_IDS:-0,1}" "$RUN_ROOT"
fi
printf '[%s] TOFU 1.5B 4090x2 pipeline complete\n' "$(timestamp)"
printf '[RESULT] best=%s/BEST.json\n' "$JOINT_ROOT"
if [[ "${RUN_FINALIZE:-1}" == "1" ]]; then
  printf '[RESULT] latex=%s/table1.tex\n' "$FINAL_ROOT"
fi
printf '[RESULT] unified_log=%s\n' "$PIPELINE_LOG"
