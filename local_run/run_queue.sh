#!/usr/bin/env bash
# Run a list of gate jobs sequentially. Each entry:
#   DATASET NAME SUBDIR DTYPE GPU DMAP RWKU_AUTHOR RWKU_POOL
# DMAP "-" = none. RWKU_AUTHOR/POOL only used for rwku ("-" = default).
set -uo pipefail
REPO=/home/minsoo3.kim/dev/retain-susceptibility

run() {
  local ds="$1" name="$2" sub="$3" dt="$4" gpu="$5" dmap="$6" author="$7" pool="$8"
  rm -rf "$REPO/runs/gate_local_${ds}_${name}"
  local env=(DATASET="$ds" GPU="$gpu")
  [ "$dmap" != "-" ] && env+=(DMAP="$dmap")
  [ "$author" != "-" ] && env+=(RWKU_AUTHOR="$author")
  [ "$pool" != "-" ] && env+=(RWKU_POOL="$pool")
  echo "### QUEUE: $ds/$name ($sub $dt gpu=$gpu dmap=$dmap)"
  env "${env[@]}" bash "$REPO/local_run/run_one.sh" "$name" "$sub" "$dt"
}

# entries passed as args, newline-free: "ds name sub dt gpu dmap author pool"
for entry in "$@"; do
  run $entry
done
echo "### QUEUE DONE $(date '+%F %T')"
