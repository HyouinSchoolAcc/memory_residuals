#!/usr/bin/env bash
# Refresh the paper once v28d-seed4 (1.7B, F3-off) finishes on GH200.
# Steps:
#   1. rsync the v28d checkpoint from GH200 to ./runs/
#   2. eval it on LME (callback-aware, evpos)
#   3. eval it cross-corpus (synthd4/d4v2/d5 + LoCoMo + MSC)
#   4. rebuild paper (auto picks up n=4 at 1.7B)
set -euo pipefail
cd "$(dirname "$0")/.."

GH200="ubuntu@192.222.50.225"
RUN="chain_v28d_no_probe_seed4_1p7b_frozen_gh200"

# ---- 1. probe GH200 status ----
echo "[refresh_v28d] probing GH200..."
state=$(ssh -o BatchMode=yes "$GH200" "cd ~/memory_residuals && \
    if [ -d output/$RUN/final ]; then echo done; \
    elif pgrep -f \"run_name $RUN\" >/dev/null; then echo running; \
    elif [ -d output/$RUN ]; then echo crashed; \
    else echo missing; fi" 2>/dev/null || echo "ssh-error")
echo "[refresh_v28d] GH200 state: $state"
if [ "$state" != "done" ]; then
    echo "[refresh_v28d] v28d not finished yet — exiting." >&2
    exit 1
fi

# ---- 2. rsync checkpoint to local ----
mkdir -p runs
rsync -avz --partial \
    "$GH200:~/memory_residuals/output/$RUN/final/" \
    "runs/$RUN/final/"
rsync -avz --partial \
    "$GH200:~/memory_residuals/output/$RUN/best/" \
    "runs/$RUN/best/" 2>/dev/null || true

# ---- 3. eval on LME (callback-aware, evpos) ----
echo "[refresh_v28d] LME eval..."
CUDA_VISIBLE_DEVICES=0 python tools/eval_callback.py \
    --model_path "runs/$RUN/final" \
    --corpora paper_artifacts/chains/lme_val_s512_evpos.pt \
    --names lme_val \
    --output "results/eval_v25_seed_pack_evpos/v28d_no_probe_seed4_final_lme_val_evpos.json"

# ---- 4. cross-corpus eval ----
echo "[refresh_v28d] cross-corpus eval..."
bash scripts/eval_cross_corpus.sh \
    "runs/$RUN/final" v28d_seed4 0

# ---- 5. rebuild paper ----
echo "[refresh_v28d] rebuilding paper..."
cd paper && bash build.sh
echo "[refresh_v28d] DONE — main.pdf now at n=4 at 1.7B."
