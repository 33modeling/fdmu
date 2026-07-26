#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset PYTHONHOME PYTHONPATH PIP_TARGET PIP_PREFIX
export VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"

bash local_run/ensure_4090_yaml.sh

timestamp() {
  date -u '+%FT%TZ'
}

run_stage() {
  local name="$1"
  shift
  printf '[%s] pipeline stage start: %s\n' "$(timestamp)" "$name"
  "$@"
  printf '[%s] pipeline stage complete: %s\n' "$(timestamp)" "$name"
}

run_stage_accept() {
  local name="$1"
  local accepted="$2"
  local status
  shift 2
  printf '[%s] pipeline stage start: %s\n' "$(timestamp)" "$name"
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

printf '[%s] TOFU 1.5B 4090x2 pipeline start\n' "$(timestamp)"
# Exit 4 is the successful protocol boundary: calibration is resolved and
# requires the explicit target-free approval performed by the next stage.
run_stage_accept calibration "0 4" bash local_run/run_tofu_1p5b_calibration.sh
run_stage parent-freeze-approval \
  bash local_run/approve_tofu_1p5b_parent_freeze.sh --approve
run_stage joint-sweep bash local_run/sweep_joint_1p5b_4090x2.sh "$@"
printf '[%s] TOFU 1.5B 4090x2 pipeline complete\n' "$(timestamp)"
