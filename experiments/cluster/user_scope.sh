#!/usr/bin/env bash
# Resolve one stable queue/result namespace from the invoking Unix user.

if [[ -z "${CLUSTER_RUNS_ROOT:-}" ]]; then
  printf 'CLUSTER_RUNS_ROOT must be set before sourcing user_scope.sh\n' >&2
  return 2 2>/dev/null || exit 2
fi

_FDMU_RUN_USER_RAW="${FDMU_RUN_USER:-${USER:-}}"
if [[ -z "$_FDMU_RUN_USER_RAW" ]]; then
  _FDMU_RUN_USER_RAW="$(id -un)"
fi
FDMU_RUN_USER="$(
  printf '%s' "$_FDMU_RUN_USER_RAW" \
    | LC_ALL=C tr -c 'A-Za-z0-9_.-' '_' \
    | sed -E 's/^[._-]+//; s/[._-]+$//'
)"
if [[ -z "$FDMU_RUN_USER" ]]; then
  printf 'cannot derive a filesystem-safe run user from %q\n' \
    "$_FDMU_RUN_USER_RAW" >&2
  return 2 2>/dev/null || exit 2
fi

_FDMU_SCOPE_REPO_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
)"
_FDMU_LEGACY_OWNER_FILE="${FDMU_LEGACY_RUN_OWNER_FILE:-$_FDMU_SCOPE_REPO_ROOT/configs/cluster/legacy_run_owner.txt}"
_FDMU_LEGACY_OWNER="${FDMU_LEGACY_RUN_OWNER:-}"
if [[ -z "$_FDMU_LEGACY_OWNER" && -f "$_FDMU_LEGACY_OWNER_FILE" ]]; then
  IFS= read -r _FDMU_LEGACY_OWNER < "$_FDMU_LEGACY_OWNER_FILE"
fi

export FDMU_RUN_USER
export CLUSTER_RUN_USER="$FDMU_RUN_USER"
if [[ "${FDMU_SHARED_LEGACY_RUNS:-0}" == "1" \
  || ( "${FDMU_FORCE_USER_NAMESPACE:-0}" != "1" \
    && -n "$_FDMU_LEGACY_OWNER" \
    && "$CLUSTER_RUN_USER" == "$_FDMU_LEGACY_OWNER" ) ]]; then
  export FDMU_RUN_SCOPE=legacy
  export CLUSTER_USER_RUNS_ROOT="$CLUSTER_RUNS_ROOT"
  export CLUSTER_USER_QUEUE_ROOT="$CLUSTER_RUNS_ROOT/cluster_queue"
else
  export FDMU_RUN_SCOPE=user
  export CLUSTER_USER_RUNS_ROOT="$CLUSTER_RUNS_ROOT/users/$CLUSTER_RUN_USER"
  export CLUSTER_USER_QUEUE_ROOT="$CLUSTER_RUNS_ROOT/cluster_queue/users/$CLUSTER_RUN_USER"
fi
export FDMU_CAMPAIGN_RUNS_ROOT="$CLUSTER_USER_RUNS_ROOT"

unset \
  _FDMU_RUN_USER_RAW \
  _FDMU_SCOPE_REPO_ROOT \
  _FDMU_LEGACY_OWNER_FILE \
  _FDMU_LEGACY_OWNER
