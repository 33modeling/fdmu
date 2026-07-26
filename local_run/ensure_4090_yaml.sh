#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Compatibility entry point. YAML repair must never create a partial venv that
# bypasses the frozen torch version used by the 4090 experiment.
exec bash "$ROOT/local_run/bootstrap_4090_env.sh"
