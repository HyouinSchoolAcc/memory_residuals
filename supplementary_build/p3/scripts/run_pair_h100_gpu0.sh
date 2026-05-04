#!/usr/bin/env bash
# Sequentially run v14i then v14j on local H100 GPU 0.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0

echo "[pair-gpu0 $(date '+%F %T')] starting v14i_d4v2_norm_lowlr_local"
bash Scripts/train_v14i_d4v2_norm_lowlr_local.sh
RC1=$?
echo "[pair-gpu0 $(date '+%F %T')] v14i exited rc=$RC1"

# Brief pause so we can observe the GPU draining before the next cell.
sleep 30

echo "[pair-gpu0 $(date '+%F %T')] starting v14j_d4v2_norm_softbias_local"
bash Scripts/train_v14j_d4v2_norm_softbias_local.sh
RC2=$?
echo "[pair-gpu0 $(date '+%F %T')] v14j exited rc=$RC2"
echo "[pair-gpu0 $(date '+%F %T')] pair done (rc1=$RC1 rc2=$RC2)"
