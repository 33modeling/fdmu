#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset PYTHONHOME PYTHONPATH PIP_TARGET PIP_PREFIX
export VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"

if [[ -x "$PYTHON" ]]; then
  if ! "$PYTHON" -c 'import yaml' >/dev/null 2>&1; then
    "$PYTHON" -m pip install --no-cache-dir --upgrade "PyYAML>=6.0"
  fi
  "$PYTHON" -c 'import site, sys, yaml; print(
      f"[deps] python={sys.executable} prefix={sys.prefix} "
      f"site={site.getsitepackages()} pyyaml={yaml.__version__} file={yaml.__file__}"
  )'
fi

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
