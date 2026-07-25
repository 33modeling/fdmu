#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset PYTHONHOME PYTHONPATH PIP_TARGET PIP_PREFIX PIP_NO_CACHE_DIR
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.11}"

diagnose() {
  printf '[yaml-diag] venv=%s python=%s\n' "$VENV" "$PYTHON" >&2
  if [[ -x "$PYTHON" ]]; then
    "$PYTHON" - <<'PY' >&2
import importlib.util
import site
import sys

print(f"[yaml-diag] executable={sys.executable}")
print(f"[yaml-diag] prefix={sys.prefix} base_prefix={sys.base_prefix}")
print(f"[yaml-diag] site_packages={site.getsitepackages()}")
print(f"[yaml-diag] sys_path={sys.path}")
print(f"[yaml-diag] yaml_spec={importlib.util.find_spec('yaml')}")
PY
    "$PYTHON" -m pip --version >&2 || true
    "$PYTHON" -m pip show PyYAML >&2 || true
  fi
}

if [[ ! -x "$PYTHON" ]]; then
  if ! command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1; then
    printf '[yaml-error] Python bootstrap is missing: %s\n' "$PYTHON_BOOTSTRAP" >&2
    exit 2
  fi
  printf '[yaml-repair] creating repository venv with %s: %s\n' \
    "$PYTHON_BOOTSTRAP" "$VENV"
  "$PYTHON_BOOTSTRAP" -m venv "$VENV"
fi

if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  printf '[yaml-repair] pip is missing; running ensurepip\n'
  "$PYTHON" -m ensurepip --upgrade
fi

if ! "$PYTHON" -c 'import yaml' >/dev/null 2>&1; then
  printf '[yaml-repair] yaml import failed; force-reinstalling PyYAML\n'
  diagnose
  if ! "$PYTHON" -m pip install --force-reinstall "PyYAML>=6.0"; then
    printf '[yaml-error] PyYAML force reinstall failed\n' >&2
    diagnose
    exit 2
  fi
fi

if ! "$PYTHON" -c 'import yaml' >/dev/null 2>&1; then
  printf '[yaml-error] PyYAML installation completed but import yaml still fails\n' >&2
  diagnose
  exit 2
fi

"$PYTHON" - <<'PY'
import site
import sys
import yaml

print(
    f"[yaml-ok] python={sys.executable} prefix={sys.prefix} "
    f"site={site.getsitepackages()} version={yaml.__version__} file={yaml.__file__}"
)
PY
