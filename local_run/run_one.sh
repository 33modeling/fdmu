#!/usr/bin/env bash
# Run the gate experiment for ONE model on ONE dataset on RTX 4090, then summarize.
# Usage: run_one.sh NAME SUBDIR DTYPE
#   NAME    result label (e.g. 1p5b)
#   SUBDIR  model dir under /rdata/models
#   DTYPE   float32 | bfloat16
# Env:
#   DATASET     tofu (default) | rwku      -> results are split per dataset
#   RWKU_AUTHOR forget_target index (rwku, default 0)
#   RWKU_POOL   frozen remote-target pool range (rwku, default "100-119")
#   GPU         CUDA_VISIBLE_DEVICES (default 0); DMAP/MAXMEM for multi-GPU sharding
set -uo pipefail

NAME="$1"; SUB="$2"; DTYPE="$3"
DATASET="${DATASET:-tofu}"
REPO=/home/minsoo3.kim/dev/retain-susceptibility
MODELS=/rdata/models                          # shared team model zoo
RESULTS="/rdata/minsoo3.kim/results/$DATASET"  # split per dataset
ENC="$MODELS/all-MiniLM-L6-v2"
GPU="${GPU:-0}"

cd "$REPO"
source .venv/bin/activate
export HF_HOME=/rdata/minsoo3.kim/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RESULTS"

model="$MODELS/$SUB"
out="$REPO/runs/gate_local_${DATASET}_$NAME"
log="$RESULTS/gate_local_$NAME.out"

DMAP_ARGS=()
if [ -n "${DMAP:-}" ]; then DMAP_ARGS=(--device-map "$DMAP"); fi
if [ -n "${MAXMEM:-}" ]; then DMAP_ARGS+=(--max-memory-gib "$MAXMEM"); fi
if [ -n "${S2ETA2:-}" ]; then DMAP_ARGS+=(--s2-eta2 "$S2ETA2"); fi   # tuned Stage2 repair step (3e-5)
if [ -n "${BETA:-}" ]; then DMAP_ARGS+=(--beta "$BETA"); fi           # NPO-family temperature (0.1; 1.0 kills ascent)
if [ -n "${GENLR:-}" ]; then DMAP_ARGS+=(--gen-lr "$GENLR"); fi        # forget lr; raise so NPO reaches recall<0.10
if [ -n "${S1GATE:-}" ]; then DMAP_ARGS+=(--s1-recall-gate "$S1GATE"); fi  # Stage1 forgets to recall gate (ours/s2s reach)

# dataset-specific request selection
DS_ARGS=(--dataset "$DATASET")
case "$DATASET" in
  rwku)  DS_ARGS+=(--author "${RWKU_AUTHOR:-0}" --candidate-authors "${RWKU_POOL:-100-119}") ;;
  muse_news|muse_books) : ;;   # single corpus-level request; no roster args
  *)     DS_ARGS+=(--universe-authors 20) ;;   # tofu
esac

echo "=== [$(date '+%F %T')] START $DATASET/$NAME ($SUB, $DTYPE) on GPU$GPU ==="
python experiments/gate_1p5b/gate.py \
  --model "$model" --model-id "$NAME" \
  --device cuda --dtype "$DTYPE" "${DMAP_ARGS[@]}" \
  --trainable-scope "${SCOPE:-probe_block}" \
  "${DS_ARGS[@]}" \
  --pool-size 16 --batch-size 4 --seed 2025 \
  --predictors "fd,fd_norm,knn_feature,knn_embed,knn_lexical,grad_norm,random_rank" \
  --sentence-encoder "$ENC" \
  --t2-roster "${T2ROSTER:-npo,npo_transplant}" \
  --out-dir "$out" \
  > "$log" 2>&1
rc=$?
echo "=== [$(date '+%F %T')] END $DATASET/$NAME rc=$rc ==="
if [ $rc -eq 0 ]; then
  python local_run/summarize.py "$NAME" "$out" "$RESULTS" | tee -a "$RESULTS/CAMPAIGN_REPORT.md"
else
  echo "[FAIL] $DATASET/$NAME rc=$rc — tail of $log:"; tail -25 "$log"
fi
exit $rc
