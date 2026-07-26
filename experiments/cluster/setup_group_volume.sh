#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STORAGE="/group-volume/fdmu"
VENV="$STORAGE/.venv"
BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.11}"
LOCK="$STORAGE/environment/setup.lock"
LOCK_OWNER="$STORAGE/environment/setup.lock.owner"
LOCK_TIMEOUT_SECONDS="${SETUP_LOCK_TIMEOUT_SECONDS:-1800}"
SETUP_LOG_DIR="$STORAGE/runs/logs/cluster/setup"
SETUP_LOG="$SETUP_LOG_DIR/setup_$(hostname)_$(date -u '+%Y%m%dT%H%M%SZ')_$$.out"
PINNED_PACKAGES=(
  "pip==25.1.1"
  "setuptools==80.9.0"
  "wheel==0.45.1"
  "torch==2.7.1"
  "transformers==4.53.2"
  "PyYAML==6.0.2"
  "pytest==8.3.5"
  "datasets==2.19.2"
)

export HOME="$STORAGE/bootstrap/home"
export TMPDIR="$STORAGE/bootstrap/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$STORAGE/bootstrap/xdg_cache"
export PIP_CACHE_DIR="$STORAGE/bootstrap/pip_cache"
export PYTHONPYCACHEPREFIX="$STORAGE/bootstrap/pycache"

mkdir -p \
  "$STORAGE/runs" \
  "$STORAGE/runtime" \
  "$STORAGE/environment" \
  "$HOME" \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX"

mkdir -p "$SETUP_LOG_DIR"
ln -sfn "$(basename "$SETUP_LOG")" \
  "$SETUP_LOG_DIR/setup_$(hostname)_current.out"
exec > >(tee -a "$SETUP_LOG") 2>&1

STAGE=initialization
LOCK_HELD=0
on_error() {
  local code=$?
  trap - ERR
  printf '[cluster-setup][ERROR] stage=%s exit=%s line=%s command=%s\n' \
    "$STAGE" "$code" "${BASH_LINENO[0]:-unknown}" \
    "${BASH_COMMAND:-unknown}" >&2
  printf '[cluster-setup][ERROR] host=%s pid=%s venv=%s log=%s\n' \
    "$(hostname)" "$$" "$VENV" "$SETUP_LOG" >&2
  df -h "$STORAGE" 2>&1 || true
  exit "$code"
}
trap on_error ERR

release_setup_lock() {
  if (( LOCK_HELD == 1 )); then
    rm -f "$LOCK_OWNER"
    flock -u 9 || true
    LOCK_HELD=0
    printf '[cluster-setup] released cross-host lock: %s\n' "$LOCK"
  fi
}
trap release_setup_lock EXIT

environment_is_ready() {
  [[ -x "$VENV/bin/python" ]] || return 1
  "$VENV/bin/python" - "${PINNED_PACKAGES[@]}" <<'PY'
import importlib
import importlib.metadata
import sys

expected = dict(item.rsplit("==", 1) for item in sys.argv[1:])
mismatches = []
for distribution, version in expected.items():
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        actual = "MISSING"
    if actual != version:
        mismatches.append(f"{distribution}: expected={version} actual={actual}")

for module in ("datasets", "torch", "transformers", "yaml"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        mismatches.append(f"import {module}: {type(exc).__name__}: {exc}")

if mismatches:
    print("[cluster-setup] environment is not ready:")
    for mismatch in mismatches:
        print(f"  {mismatch}")
    raise SystemExit(1)
PY
}

STAGE=fast-environment-check
if environment_is_ready; then
  printf '[cluster-setup] existing environment is ready; lock not required: %s\n' \
    "$VENV"
  printf '[cluster-setup] host=%s pid=%s log=%s\n' \
    "$(hostname)" "$$" "$SETUP_LOG"
  exit 0
fi

STAGE=lock-prerequisite
command -v flock >/dev/null 2>&1 \
  || { printf 'flock is required for shared environment setup\n' >&2; exit 2; }
exec 9>"$LOCK"
printf '[cluster-setup] waiting up to %ss for cross-host lock: %s\n' \
  "$LOCK_TIMEOUT_SECONDS" "$LOCK"
STAGE=lock-acquisition
if ! flock -x -w "$LOCK_TIMEOUT_SECONDS" 9; then
  printf '[cluster-setup][ERROR] lock timeout after %ss: %s\n' \
    "$LOCK_TIMEOUT_SECONDS" "$LOCK" >&2
  if [[ -f "$LOCK_OWNER" ]]; then
    printf '[cluster-setup][ERROR] current lock owner metadata:\n' >&2
    sed 's/^/  /' "$LOCK_OWNER" >&2 || true
  else
    printf '[cluster-setup][ERROR] no owner metadata; inspect active setup processes on all hosts\n' >&2
  fi
  exit 75
fi
LOCK_HELD=1
printf 'host=%s\npid=%s\nstarted_utc=%s\nlog=%s\n' \
  "$(hostname)" "$$" "$(date -u '+%FT%TZ')" "$SETUP_LOG" > "$LOCK_OWNER"
printf '[cluster-setup] acquired cross-host lock: %s\n' "$LOCK"
printf '[cluster-setup] note: the lock file remains by design; only flock state controls ownership\n'

STAGE=venv-bootstrap
if environment_is_ready; then
  printf '[cluster-setup] environment was prepared by another host while waiting: %s\n' \
    "$VENV"
  exit 0
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  command -v "$BOOTSTRAP" >/dev/null 2>&1 \
    || { printf 'Python bootstrap not found: %s\n' "$BOOTSTRAP" >&2; exit 2; }
  "$BOOTSTRAP" -m venv "$VENV"
fi

PYTHON="$VENV/bin/python"
STAGE=package-install
printf '[cluster-setup] installing deterministic repository environment\n'
"$PYTHON" -m pip install \
  --retries "${PIP_RETRIES:-5}" \
  --timeout "${PIP_DEFAULT_TIMEOUT:-120}" \
  --upgrade "${PINNED_PACKAGES[@]}"

STAGE=environment-import-validation
environment_is_ready
"$PYTHON" - <<'PY'
import datasets
import sys
import torch
import transformers
import yaml

print(f"python={sys.executable}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"datasets={datasets.__version__} pyyaml={yaml.__version__}")
PY

STAGE=environment-manifest
FREEZE="$STORAGE/environment/pip-freeze.txt"
FREEZE_TMP="$FREEZE.tmp.$$"
"$PYTHON" -m pip freeze > "$FREEZE_TMP"
mv -f "$FREEZE_TMP" "$FREEZE"
printf 'group-volume environment ready: %s\n' "$VENV"
df -h "$STORAGE"
