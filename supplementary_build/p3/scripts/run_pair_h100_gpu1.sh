#!/usr/bin/env bash
# Sequentially run v14k then v14l on local H100 GPU 1.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

export CUDA_VISIBLE_DEVICES=1

echo "[pair-gpu1 $(date '+%F %T')] starting v14k_d4v2_norm_no_warmup_local"
bash Scripts/train_v14k_d4v2_norm_no_warmup_local.sh
RC1=$?
echo "[pair-gpu1 $(date '+%F %T')] v14k exited rc=$RC1"

sleep 30

echo "[pair-gpu1 $(date '+%F %T')] starting v14l_d4v2_norm_full_writer_local"
bash Scripts/train_v14l_d4v2_norm_full_writer_local.sh
RC2=$?
echo "[pair-gpu1 $(date '+%F %T')] v14l exited rc=$RC2"
echo "[pair-gpu1 $(date '+%F %T')] pair done (rc1=$RC1 rc2=$RC2)"
