#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-run}"
CONFIG_PATH="${2:-${PDF_V4_CONFIG:-configs/local/pdf_v4.local.yaml}}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ "${ACTION}" != "prepare-manifest" && "${ACTION}" != "inspect-model" && "${ACTION}" != "validate" && "${ACTION}" != "run" ]]; then
  CONFIG_PATH="${ACTION}"
  ACTION="run"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: Python not found. Create .venv and install -e '.[dev,campaign]'." >&2
    exit 2
  fi
fi

if [[ -f "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="$(cd -- "$(dirname -- "${CONFIG_PATH}")" && pwd)/$(basename -- "${CONFIG_PATH}")"
elif [[ -f "${REPO_ROOT}/${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
else
  echo "ERROR: config not found: ${CONFIG_PATH}" >&2
  echo "Copy configs/local/pdf_v4.example.yaml to configs/local/pdf_v4.local.yaml first." >&2
  exit 2
fi

if ! "${PYTHON_BIN}" -c 'import torch, transformers, yaml' >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} is missing torch/transformers/pyyaml." >&2
  echo "Install the local environment with: python -m pip install -e '.[dev,campaign]'" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" experiments/local_pdf_v4.py \
  --action "${ACTION}" \
  --config "${CONFIG_PATH}"
