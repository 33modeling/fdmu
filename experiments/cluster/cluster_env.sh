#!/usr/bin/env bash
# Shared-volume-only cache and temporary paths for H100 cluster jobs.

GROUP_VOLUME_ROOT="${GROUP_VOLUME_ROOT:-/group-volume}"
CLUSTER_USER="${USER:-$(id -un)}"

export HF_HOME="${CLUSTER_HF_HOME:-$GROUP_VOLUME_ROOT/data/hf_home}"
export HF_DATASETS_CACHE="${CLUSTER_HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export RSUS_DATASETS_CACHE="${CLUSTER_RSUS_DATASETS_CACHE:-$HF_HOME/rsus_datasets_cache}"
export TORCH_HOME="${CLUSTER_TORCH_HOME:-$GROUP_VOLUME_ROOT/data/torch_home}"
export XDG_CACHE_HOME="${CLUSTER_XDG_CACHE_HOME:-$GROUP_VOLUME_ROOT/data/xdg_cache/$CLUSTER_USER}"
export TMPDIR="${CLUSTER_TMPDIR:-$GROUP_VOLUME_ROOT/tmp/$CLUSTER_USER}"
export CLUSTER_RUNS_ROOT="${CLUSTER_RUNS_ROOT:-$GROUP_VOLUME_ROOT/jieuns.shin/retain-susceptibility/runs}"

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$RSUS_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR" \
  "$CLUSTER_RUNS_ROOT"
