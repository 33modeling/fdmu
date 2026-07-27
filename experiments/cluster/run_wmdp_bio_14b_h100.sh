#!/usr/bin/env bash
set -Eeuo pipefail
export DATASET_CAMPAIGN=wmdp_bio DATASET_SCALE=14b
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/experiments/dataset_campaign/run.sh" "$@"
