#!/usr/bin/env bash
# Sweep ONLY the `ours` two-stage repair knobs on TOFU 1.5B (2-GPU), reusing a
# shared SFT cache. Regime is fixed from the verification run (gen 60 / t2 240 /
# alpha select). Reports ours reach + collateral (mean/CVaR dNLL) + min_forget
# per config so the best stage1/stage2 setting can be frozen.
set -uo pipefail
cd /home/minsoo3.kim/dev/retain-susceptibility
source .venv/bin/activate
export HF_HOME=/rdata/minsoo3.kim/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SFTC=/tmp/sweep_ours_sft.pt
SUM=/home/minsoo3.kim/dev/retain-susceptibility/local_run/OURS_SWEEP.md
echo "# ours sweep (TOFU 1.5B, gen60/t2240/alpha-select) — $(date '+%F %T')" > "$SUM"
echo "| s1-lr | s1-max | s2-steps | s2-eta2 | reach | mean dNLL | CVaR | min_forget |" >> "$SUM"
echo "|---|---|---|---|---|---|---|---|" >> "$SUM"

# wait for GPUs free
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; do sleep 30; done

# grid: s1lr:s1max:s2steps:s2eta2
for cfg in "1e-5:600:320:1e-4" "1e-5:600:320:3e-4" "1e-5:600:400:1e-4" "1e-5:600:400:3e-5"; do
  IFS=":" read -r s1lr s1max s2s s2e <<< "$cfg"
  tag="s1_${s1lr}_${s1max}_s2_${s2s}_${s2e}"
  out="runs/gate_ours_$tag"; rm -rf "$out"
  echo "### [$(date +%H:%M)] ours sweep $cfg"
  python experiments/gate_1p5b/gate.py \
    --model /rdata/models/Qwen2.5-1.5B-Instruct --model-id "$tag" \
    --device cuda --dtype float32 --device-map balanced \
    --trainable-scope full --dataset tofu --universe-authors 8 --pool-size 8 --batch-size 4 --seed 2025 \
    --sft-steps 250 --sft-cache "$SFTC" \
    --predictors fd,fd_norm,knn_embed,grad_norm --sentence-encoder /rdata/models/all-MiniLM-L6-v2 \
    --generators npo,graddiff,rmu --gen-steps 60 \
    --gen-lr-per "npo=1.6e-5,graddiff=8e-7,rmu=3.2e-5" --gen-beta-per "npo=0.1" --beta 0.1 \
    --t2-roster ours --t2-steps 240 \
    --s1-lr "$s1lr" --s1-max-steps "$s1max" --s2-eta2 "$s2e" --s2-steps "$s2s" --s1-recall-gate 0.10 \
    --partition-predictor fd --partition-proximity knn_embed --partition-alpha select \
    --out-dir "$out" > "/tmp/ours_$tag.out" 2>&1
  line=$(grep -A2 'protection method: ours' "$out/gate.log" 2>/dev/null | tail -2)
  reach=$(echo "$line" | grep -oE 'reach=[A-Za-z]+' | cut -d= -f2)
  mean=$(echo "$line" | grep -oE 'mean dNLL=[0-9.eE+-]+' | cut -d= -f2)
  cvar=$(echo "$line" | grep -oE 'CVaR=[0-9.eE+-]+' | cut -d= -f2)
  mf=$(echo "$line" | grep -oE 'min_forget=[0-9./]+' | cut -d= -f2)
  echo "| $s1lr | $s1max | $s2s | $s2e | ${reach:-?} | ${mean:-?} | ${cvar:-?} | ${mf:-?} |" >> "$SUM"
  echo "  -> reach=$reach mean=$mean cvar=$cvar min_forget=$mf"
done
echo "### ours sweep done $(date '+%F %T')"
cat "$SUM"
