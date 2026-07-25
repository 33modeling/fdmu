#!/usr/bin/env bash
# Shared-volume-only cache and temporary paths for H100 cluster jobs.

GROUP_VOLUME_ROOT="${GROUP_VOLUME_ROOT:-/group-volume}"
CLUSTER_USER="${USER:-$(id -un)}"
export CLUSTER_RUNS_ROOT="${CLUSTER_RUNS_ROOT:-$GROUP_VOLUME_ROOT/jieuns.shin/retain-susceptibility/runs}"
CLUSTER_WORK_ROOT="${CLUSTER_WORK_ROOT:-$CLUSTER_RUNS_ROOT/_runtime}"

export HF_HOME="${CLUSTER_HF_HOME:-$GROUP_VOLUME_ROOT/data/hf_home}"
export HF_DATASETS_CACHE="${CLUSTER_HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export RSUS_DATASETS_CACHE="${CLUSTER_RSUS_DATASETS_CACHE:-$CLUSTER_WORK_ROOT/datasets_cache}"
export TORCH_HOME="${CLUSTER_TORCH_HOME:-$CLUSTER_WORK_ROOT/torch_home}"
export XDG_CACHE_HOME="${CLUSTER_XDG_CACHE_HOME:-$CLUSTER_WORK_ROOT/xdg_cache}"
export TMPDIR="${CLUSTER_TMPDIR:-$CLUSTER_WORK_ROOT/tmp}"

if [[ ! -d "$HF_HOME" ]]; then
  printf 'shared HF_HOME is missing or not mounted: %s\n' "$HF_HOME" >&2
  return 2 2>/dev/null || exit 2
fi

for writable_path in \
  "$RSUS_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR" \
  "$CLUSTER_RUNS_ROOT"; do
  if ! mkdir -p "$writable_path" || [[ ! -w "$writable_path" ]]; then
    printf 'cluster writable path is unavailable: %s\n' "$writable_path" >&2
    printf 'set CLUSTER_WORK_ROOT to a writable /group-volume directory\n' >&2
    return 2 2>/dev/null || exit 2
  fi
done

printf '[cluster-env] runs=%s writable_root=%s\n' \
  "$CLUSTER_RUNS_ROOT" "$CLUSTER_WORK_ROOT"
