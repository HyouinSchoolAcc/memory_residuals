#!/usr/bin/env bash
# Orchestrate GPU 1 work for the v15 wave (after v14k completes):
#   v15b (0.6B joint train, ~60min) -> v15f-fallback or 1.7B repeat
#
# This script WAITS for any existing python train_chain.py process on
# GPU 1 to finish before starting (so it's safe to launch even while
# v14k is mid-run).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
export CUDA_VISIBLE_DEVICES=1

wait_for_gpu1() {
    while pgrep -f 'CUDA_VISIBLE_DEVICES=1.*train_chain' >/dev/null 2>&1 \
        || nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits \
            | awk '{ if ($1+0 > 4096) exit 0; else exit 1 }'; do
        sleep 60
    done
    echo "[orch-gpu1 $(date '+%F %T')] gpu 1 is now free"
}

run_cell() {
    local script="$1"
    local name="$2"
    echo "[orch-gpu1 $(date '+%F %T')] starting $name"
    bash "Scripts/$script"
    rc=$?
    echo "[orch-gpu1 $(date '+%F %T')] $name exited rc=$rc"
    sleep 30
}

echo "[orch-gpu1 $(date '+%F %T')] waiting for GPU 1 to free up"
wait_for_gpu1

run_cell train_v15b_d4v2_norm_jointtrain_local.sh v15b_jointtrain

echo "[orch-gpu1 $(date '+%F %T')] gpu1 v15 wave A complete"
