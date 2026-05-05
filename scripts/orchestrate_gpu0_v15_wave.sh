#!/usr/bin/env bash
# Orchestrate GPU 0 work for the v15 wave:
#   v15a (0.6B replicate, ~30min) -> v15e (1.7B no-warmup, ~50min) ->
#   v15f (1.7B joint train, ~75min)
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0

run_cell() {
    local script="$1"
    local name="$2"
    echo "[orch-gpu0 $(date '+%F %T')] starting $name"
    bash "Scripts/$script"
    rc=$?
    echo "[orch-gpu0 $(date '+%F %T')] $name exited rc=$rc"
    sleep 30
}

run_cell train_v15a_d4v2_norm_replicate_local.sh v15a_replicate
run_cell train_v15e_d4v2_1p7b_norm_local.sh v15e_1p7b_norm
run_cell train_v15f_d4v2_1p7b_jointtrain_local.sh v15f_1p7b_jointtrain

echo "[orch-gpu0 $(date '+%F %T')] v15 wave complete"
