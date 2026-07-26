#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset PYTHONHOME PYTHONPATH PIP_TARGET PIP_PREFIX
export PIP_NO_CACHE_DIR=1

VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.11}"
TORCH_VERSION="2.7.1"

diagnose() {
  printf '[4090-env-diag] venv=%s python=%s\n' "$VENV" "$PYTHON" >&2
  if [[ -x "$PYTHON" ]]; then
    "$PYTHON" - <<'PY' >&2 || true
import importlib.util
import site
import sys

print(f"[4090-env-diag] executable={sys.executable}")
print(f"[4090-env-diag] prefix={sys.prefix} base_prefix={sys.base_prefix}")
print(f"[4090-env-diag] site_packages={site.getsitepackages()}")
for name in ("torch", "transformers", "datasets", "yaml"):
    print(f"[4090-env-diag] {name}_spec={importlib.util.find_spec(name)}")
PY
    "$PYTHON" -m pip --version >&2 || true
    "$PYTHON" -m pip show torch transformers datasets PyYAML >&2 || true
  fi
}

if [[ ! -x "$PYTHON" ]]; then
  if ! command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1; then
    printf '[4090-env-error] Python bootstrap is missing: %s\n' \
      "$PYTHON_BOOTSTRAP" >&2
    exit 2
  fi
  printf '[4090-env] creating repository venv with %s: %s\n' \
    "$PYTHON_BOOTSTRAP" "$VENV"
  "$PYTHON_BOOTSTRAP" -m venv "$VENV"
fi

if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  printf '[4090-env] pip is missing; running ensurepip\n'
  "$PYTHON" -m ensurepip --upgrade
fi

needs_install=0
if ! "$PYTHON" - "$TORCH_VERSION" <<'PY' >/dev/null 2>&1
import sys
import datasets
import torch
import transformers
import yaml

if torch.__version__.split("+", 1)[0] != sys.argv[1]:
    raise SystemExit(1)
PY
then
  needs_install=1
fi

if (( needs_install )); then
  printf '[4090-env] installing repository environment with torch==%s\n' \
    "$TORCH_VERSION"
  diagnose
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
  "$PYTHON" -m pip install "torch==$TORCH_VERSION"
  "$PYTHON" -m pip install -e ".[dev]" "datasets>=2.19" "PyYAML>=6.0"
fi

if ! "$PYTHON" - "$TORCH_VERSION" <<'PY'
import site
import sys
import datasets
import torch
import transformers
import yaml

expected = sys.argv[1]
observed = torch.__version__.split("+", 1)[0]
if observed != expected:
    raise SystemExit(f"torch version mismatch: expected {expected}, found {torch.__version__}")
print(
    f"[4090-env-ok] python={sys.executable} prefix={sys.prefix} "
    f"site={site.getsitepackages()} torch={torch.__version__} "
    f"transformers={transformers.__version__} datasets={datasets.__version__} "
    f"pyyaml={yaml.__version__} yaml_file={yaml.__file__}"
)
PY
then
  printf '[4090-env-error] environment validation failed\n' >&2
  diagnose
  exit 2
fi
