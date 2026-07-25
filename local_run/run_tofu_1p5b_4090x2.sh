#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

printf '[%s] TOFU 1.5B 4090x2 pipeline start\n' "$(timestamp)"
run_stage calibration bash local_run/run_tofu_1p5b_calibration.sh
run_stage parent-freeze-approval \
  bash local_run/approve_tofu_1p5b_parent_freeze.sh --approve
run_stage joint-sweep bash local_run/sweep_joint_1p5b_4090x2.sh "$@"
printf '[%s] TOFU 1.5B 4090x2 pipeline complete\n' "$(timestamp)"
