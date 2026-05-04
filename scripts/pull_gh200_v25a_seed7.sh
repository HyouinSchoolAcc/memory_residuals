#!/usr/bin/env bash
# Polls GH200 for the v25a-seed7 run to finish, then rsyncs the
# `best/` and `final/` checkpoints down to local
# `output/chain_v25a_v21c_lme_seed7_1p7b_frozen_gh200/` so the watcher
# can evaluate them with the patched LME corpus.
#
# Run after v25a-seed7 trainer is launched. Idempotent.
set -uo pipefail
cd "$(dirname "$0")/.."

REMOTE_DIR=ubuntu@192.222.50.225:/home/ubuntu/memory_residuals/output/chain_v25a_v21c_lme_seed7_1p7b_frozen_gh200
LOCAL_DIR=output/chain_v25a_v21c_lme_seed7_1p7b_frozen_gh200
LOG=logs/pull_gh200_v25a_seed7.log
mkdir -p "$LOCAL_DIR"
> "$LOG"

while true; do
    # Look for the trainer's `final/` ckpt remotely.
    has_final=$(ssh ubuntu@192.222.50.225 \
        "test -d /home/ubuntu/memory_residuals/output/chain_v25a_v21c_lme_seed7_1p7b_frozen_gh200/final && echo yes || echo no" \
        2>/dev/null)
    if [ "$has_final" = "yes" ]; then
        echo "[pull] $(date +%H:%M:%S) final/ exists on GH200; rsyncing best+final" | tee -a "$LOG"
        rsync -az --info=progress2 "$REMOTE_DIR/best"  "$LOCAL_DIR/" 2>&1 | tail -3 >> "$LOG"
        rsync -az --info=progress2 "$REMOTE_DIR/final" "$LOCAL_DIR/" 2>&1 | tail -3 >> "$LOG"
        echo "[pull] $(date +%H:%M:%S) rsync done" | tee -a "$LOG"
        break
    fi
    echo "[pull] $(date +%H:%M:%S) waiting on GH200 v25a-seed7 final/ checkpoint" >> "$LOG"
    sleep 120
done
