#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STORAGE="/group-volume/fdmu"
VENV="$STORAGE/.venv"
BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.11}"
LOCK="$STORAGE/environment/setup.lock"
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

command -v flock >/dev/null 2>&1 \
  || { printf 'flock is required for shared environment setup\n' >&2; exit 2; }
exec 9>"$LOCK"
printf '[cluster-setup] waiting for cross-host lock: %s\n' "$LOCK"
flock -x 9
printf '[cluster-setup] acquired cross-host lock: %s\n' "$LOCK"

release_setup_lock() {
  flock -u 9
  printf '[cluster-setup] released cross-host lock: %s\n' "$LOCK"
}
trap release_setup_lock EXIT

if [[ ! -x "$VENV/bin/python" ]]; then
  command -v "$BOOTSTRAP" >/dev/null 2>&1 \
    || { printf 'Python bootstrap not found: %s\n' "$BOOTSTRAP" >&2; exit 2; }
  "$BOOTSTRAP" -m venv "$VENV"
fi

PYTHON="$VENV/bin/python"
if ! "$PYTHON" - "${PINNED_PACKAGES[@]}" <<'PY'
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
if mismatches:
    print("[cluster-setup] environment mismatch:")
    for mismatch in mismatches:
        print(f"  {mismatch}")
    raise SystemExit(1)
PY
then
  printf '[cluster-setup] installing deterministic repository environment\n'
  "$PYTHON" -m pip install --upgrade "${PINNED_PACKAGES[@]}"
fi

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

FREEZE="$STORAGE/environment/pip-freeze.txt"
FREEZE_TMP="$FREEZE.tmp.$$"
"$PYTHON" -m pip freeze > "$FREEZE_TMP"
mv -f "$FREEZE_TMP" "$FREEZE"
printf 'group-volume environment ready: %s\n' "$VENV"
df -h "$STORAGE"
