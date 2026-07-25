#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STORAGE="/group-volume/fdmu"
VENV="$STORAGE/.venv"
BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.11}"

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

if [[ ! -x "$VENV/bin/python" ]]; then
  command -v "$BOOTSTRAP" >/dev/null 2>&1 \
    || { printf 'Python bootstrap not found: %s\n' "$BOOTSTRAP" >&2; exit 2; }
  "$BOOTSTRAP" -m venv "$VENV"
fi

PYTHON="$VENV/bin/python"
if ! "$PYTHON" -c 'import torch, transformers, datasets, yaml' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
  "$PYTHON" -m pip install \
    "torch>=2.1" \
    "transformers>=4.40" \
    "pyyaml>=6.0" \
    "pytest>=7.4" \
    "datasets>=2.19" \
    "sentence-transformers>=3.0"
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

"$PYTHON" -m pip freeze > "$STORAGE/environment/pip-freeze.txt"
printf 'group-volume environment ready: %s\n' "$VENV"
df -h "$STORAGE"
