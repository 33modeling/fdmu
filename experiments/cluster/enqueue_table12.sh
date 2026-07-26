#!/usr/bin/env bash
set -euo pipefail

# One-shot enqueue helper for the remaining Table 1/2 fleet waves.
# Run ON THE CLUSTER from the repository root (any node; enqueue happens once,
# the shared file queue does the rest).
#
#   bash experiments/cluster/enqueue_table12.sh [status]     per-queue overview (default)
#   bash experiments/cluster/enqueue_table12.sh audit-7b     7B TOFU audit (+alpha) -> wave2
#   bash experiments/cluster/enqueue_table12.sh audit-14b    14B TOFU audit (+alpha) -> wave1_14b
#   bash experiments/cluster/enqueue_table12.sh wmdp         WMDP fidelity+calibration -> wave_wmdp
#   bash experiments/cluster/enqueue_table12.sh wmdp-14b     WMDP-14B fidelity+calibration -> wave_wmdp14b
#   bash experiments/cluster/enqueue_table12.sh llama        Llama-8B fidelity+calibration -> wave_llama
#   bash experiments/cluster/enqueue_table12.sh rwku-audit   RWKU 7B audit -> wave_rwku
#
# Wave -> table mapping: Table 1 <- wave2 (7B audit + alpha waves, all in
# wave2); Table 2 rows <- wave1_14b, wave_rwku, wave_wmdp, wave_wmdp14b,
# wave_llama.
# New queues (wave_wmdp/wave_llama/wave_rwku) need a committed node assignment
# in configs/cluster/fleet.yaml before launch_node.sh will serve them.
#
# This script NEVER launches workers, NEVER git-pulls, and never touches queue
# state beyond `make_units.py --enqueue`.  Queues are append-only per unit id,
# so re-running a subcommand is safe: duplicates are refused by make_units and
# reported here as "already enqueued".

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# venv activation — same default as launch_node.sh.
VENV="/group-volume/fdmu/.venv"
PYTHON="$VENV/bin/python"
if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "[enqueue_table12] missing official environment: ${VENV}/bin/activate" >&2
  echo "[enqueue_table12] run: bash experiments/cluster/setup_group_volume.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
[[ -x "$PYTHON" ]] || {
  echo "[enqueue_table12] official Python is missing: $PYTHON" >&2
  exit 1
}
# shellcheck disable=SC1091
source "${ROOT}/experiments/cluster/cluster_env.sh"

CFG_DIR="configs/channel_matrix"
STATUS_QUEUES=(wave1 wave2 wave1_14b wave_wmdp wave_wmdp14b wave_llama wave_rwku)

log()  { echo "[enqueue_table12] $*"; }
die()  { echo "[enqueue_table12] ERROR: $*" >&2; exit 1; }

require_config() {  # $1 = config path
  if [[ ! -f "$1" ]]; then
    log "config not present: $1 — skipping (it may still be under construction)."
    exit 3
  fi
}

require_clean_tree() {
  # The sealed audit runner refuses a dirty worktree, and the runbook requires
  # committing before ANY enqueue (workers must not pick up drifting code).
  local tree_status
  if ! tree_status="$(git status --porcelain --untracked-files=all 2>&1)"; then
    die "git status failed in ${ROOT} — fix git before enqueueing: ${tree_status}"
  fi
  if [[ -n "${tree_status}" ]]; then
    {
      echo "[enqueue_table12] dirty checkout: ${ROOT}"
      echo "[enqueue_table12] commit: $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
      echo "[enqueue_table12] offending files (M=tracked edit, ??=untracked):"
      git status --short --untracked-files=all
    } >&2
    die "git worktree is dirty — THIS checkout must be clean. Tracked edits: commit+push from the workstation, then update this exact checkout (multiple checkouts exist on the cluster; a pull elsewhere does not fix this one). Untracked files: move them outside the checkout or gitignore them."
  fi
}

freeze_is_frozen() {  # $1 = freeze yaml path; mirrors run_campaign.py's gate
  grep -qE '^status:[[:space:]]*"?frozen"?[[:space:]]*(#.*)?$' "$1" \
    && grep -qE '^frozen_before_audit:[[:space:]]*true[[:space:]]*(#.*)?$' "$1"
}

require_frozen() {  # $1 = freeze yaml path, $2 = what for
  [[ -f "$1" ]] || die "freeze file missing: $1"
  freeze_is_frozen "$1" \
    || die "$2 requires $1 to have 'status: frozen' (currently draft). Drain calibration, run the selector, commit the freeze, then re-run."
}

freeze_path_of() {  # $1 = config path, $2 = yaml key (objective_freeze|alpha_freeze)
  local name
  name="$(grep -m1 -E "^[[:space:]]*$2:" "$1" | awk '{print $2}')"
  [[ -n "${name}" ]] || die "could not read $2 from $1"
  echo "${CFG_DIR}/${name}"
}

enqueue_phase() {  # $1 = description, $2 = queue dir, $3 = config, $4.. = phases
  local desc="$1" queue="$2" cfg="$3"; shift 3
  local args=() phase units_tmp
  for phase in "$@"; do args+=(--phase "${phase}"); done
  log "enqueue ${desc}: config=${cfg} phases=[$*] queue=${queue}"
  # Two-step enqueue: emit units, then skip-existing top-up. A one-shot
  # --enqueue aborts on the FIRST duplicate after already writing earlier
  # units, so a topped-up roster (e.g. an added audit author) would silently
  # lose the units after the duplicate.
  units_tmp="$(mktemp)"
  "$PYTHON" experiments/cluster/make_units.py \
    --config "${cfg}" "${args[@]}" --out "${units_tmp}" \
    || die "${desc}: make_units failed."
  "$PYTHON" experiments/cluster/workqueue.py enqueue \
    --queue "${queue}" --units "${units_tmp}" --skip-existing \
    || die "${desc}: workqueue enqueue failed."
  rm -f "${units_tmp}"
  log "${desc}: done (skipped ids above were already enqueued; for a"
  log "  deliberate re-run of a unit use make_units.py --unit-suffix <rN>)."
}

post_enqueue_notes() {  # $1 = queue dir
  cat <<EOF

[enqueue_table12] Units are queued in $1 — this script does NOT start workers.
  Reminders:
    - worker count and GPU selection belong to the caller-specific launcher.
      run_tofu_7b_h100.sh uses the fleet launcher; run_tofu_14b_h100.sh runs
      exactly one dedicated worker on GPU 0.
    - make_units-generated gate/audit units must KEEP the max_attempts that
      make_units set — do not hand-edit queue JSON to add retries (sealed
      runners refuse partial run dirs by design; see the runbook's triage).
    - progress: python experiments/cluster/workqueue.py status --brief --queue $1
EOF
}

cmd="${1:-status}"
case "${cmd}" in

  status)
    for q in "${STATUS_QUEUES[@]}"; do
      root="$CLUSTER_RUNS_ROOT/cluster_queue/${q}"
      echo "== ${root} =="
      if [[ -d "${root}" ]]; then
        "$PYTHON" experiments/cluster/workqueue.py status --brief --queue "${root}" \
          || log "status failed for ${root}"
      else
        echo "  (queue not initialized yet)"
      fi
    done
    echo
    log "fleet-wide view (hosts/GPUs/assignment drift): python experiments/cluster/fleet_status.py"
    ;;

  audit-7b)
    cfg="${CFG_DIR}/7b_tofu.yaml"
    require_config "${cfg}"
    require_clean_tree
    require_frozen "$(freeze_path_of "${cfg}" objective_freeze)" "7B audit"
    queue="$CLUSTER_RUNS_ROOT/cluster_queue/wave2"
    enqueue_phase "7B TOFU audit" "${queue}" "${cfg}" audit
    alpha_freeze="$(freeze_path_of "${cfg}" alpha_freeze)"
    if [[ -f "${alpha_freeze}" ]] && freeze_is_frozen "${alpha_freeze}"; then
      log "alpha freeze is frozen -> enqueueing alpha-audit."
      enqueue_phase "7B alpha-audit" "${queue}" "${cfg}" alpha-audit
    else
      log "alpha freeze still draft -> enqueueing alpha-development (freeze alpha before alpha-audit)."
      enqueue_phase "7B alpha-development" "${queue}" "${cfg}" alpha-development
    fi
    post_enqueue_notes "${queue}"
    ;;

  audit-14b)
    cfg="${CFG_DIR}/14b_tofu.yaml"
    require_config "${cfg}"
    require_clean_tree
    require_frozen "$(freeze_path_of "${cfg}" objective_freeze)" "14B audit"
    queue="$CLUSTER_RUNS_ROOT/cluster_queue/wave1_14b"
    enqueue_phase "14B TOFU audit" "${queue}" "${cfg}" audit
    alpha_freeze="$(freeze_path_of "${cfg}" alpha_freeze)"
    if [[ -f "${alpha_freeze}" ]] && freeze_is_frozen "${alpha_freeze}"; then
      log "alpha freeze is frozen -> enqueueing alpha-audit."
      enqueue_phase "14B alpha-audit" "${queue}" "${cfg}" alpha-audit
    else
      log "alpha freeze still draft -> enqueueing alpha-development (freeze alpha before alpha-audit)."
      enqueue_phase "14B alpha-development" "${queue}" "${cfg}" alpha-development
    fi
    post_enqueue_notes "${queue}"
    ;;

  wmdp)
    cfg="${CFG_DIR}/wmdp_7b.yaml"
    require_config "${cfg}"
    require_clean_tree
    queue="$CLUSTER_RUNS_ROOT/cluster_queue/wave_wmdp"
    enqueue_phase "WMDP fidelity" "${queue}" "${cfg}" fidelity
    enqueue_phase "WMDP calibration" "${queue}" "${cfg}" calibration
    post_enqueue_notes "${queue}"
    ;;

  wmdp-14b)
    cfg="${CFG_DIR}/wmdp_14b.yaml"
    require_config "${cfg}"
    require_clean_tree
    queue="$CLUSTER_RUNS_ROOT/cluster_queue/wave_wmdp14b"
    enqueue_phase "WMDP-14B fidelity" "${queue}" "${cfg}" fidelity
    enqueue_phase "WMDP-14B calibration" "${queue}" "${cfg}" calibration
    post_enqueue_notes "${queue}"
    ;;

  llama)
    cfg="${CFG_DIR}/llama8b_tofu.yaml"
    require_config "${cfg}"
    model_path="$(grep -m1 -E '^[[:space:]]*path:' "${cfg}" | awk '{print $2}')"
    if [[ ! -f "${model_path}/config.json" ]]; then
      die "model not provisioned at ${model_path} — run: bash experiments/cluster/provision_llama.sh"
    fi
    require_clean_tree
    queue="$CLUSTER_RUNS_ROOT/cluster_queue/wave_llama"
    enqueue_phase "Llama-8B fidelity" "${queue}" "${cfg}" fidelity
    enqueue_phase "Llama-8B calibration" "${queue}" "${cfg}" calibration
    post_enqueue_notes "${queue}"
    ;;

  rwku-audit)
    cfg="${CFG_DIR}/rwku_7b.yaml"
    require_config "${cfg}"
    # RWKU audit is blocked on a real (non-TOFU-wired) fidelity certificate.
    if ! ls "$CLUSTER_RUNS_ROOT"/channel_matrix_rwku7b/fidelity/*.json >/dev/null 2>&1; then
      die "no fidelity certificate under $CLUSTER_RUNS_ROOT/channel_matrix_rwku7b/fidelity/ — run experiments/diag/fd_fidelity.py --dataset rwku first."
    fi
    require_clean_tree
    require_frozen "$(freeze_path_of "${cfg}" objective_freeze)" "RWKU audit"
    queue="$CLUSTER_RUNS_ROOT/cluster_queue/wave_rwku"
    enqueue_phase "RWKU 7B audit" "${queue}" "${cfg}" audit
    post_enqueue_notes "${queue}"
    ;;

  *)
    die "unknown subcommand '${cmd}' (expected: status | audit-7b | audit-14b | wmdp | wmdp-14b | llama | rwku-audit)"
    ;;
esac
