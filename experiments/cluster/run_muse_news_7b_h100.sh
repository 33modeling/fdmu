#!/usr/bin/env bash
set -Eeuo pipefail
export DATASET_CAMPAIGN=muse_news DATASET_SCALE=7b
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/experiments/dataset_campaign/run.sh" "$@"
