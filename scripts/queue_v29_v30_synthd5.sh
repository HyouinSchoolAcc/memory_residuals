#!/usr/bin/env bash
# Sequential local queue runner for v29a synthd5 seed 3 (GPU 0 or 1)
# and the GH200 queue for v30a synthd5 seed 2 (after seed 1 finishes).
#
# Local part: polls every 60 s for a free H100 (mem.used < 5 GB),
# launches v29a-seed3 there once one is free.
#
# Remote part: polls the GH200 every 2 min for v30a-seed1's `final/`
# checkpoint to land, then starts v30a-seed2 in the same tmux session.
#
# Usage:
#   nohup bash scripts/queue_v29_v30_synthd5.sh > logs/queue_v29_v30_synthd5.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/queue_v29_v30_synthd5.log
mkdir -p logs
echo "[queue] $(date -Iseconds) start" | tee -a "$LOG"

REMOTE=ubuntu@192.222.50.225
REMOTE_BASE=/home/ubuntu/memory_residuals

# ----- local: v29a seed 3 -----
v29a3_done=false
while ! $v29a3_done; do
    if [ -d runs/chain_v29a_synthd5_seed3_0p6b_frozen_local/final ]; then
        echo "[queue] $(date -Iseconds) v29a-seed3 final already present; skipping" | tee -a "$LOG"
        v29a3_done=true
        break
    fi
    if pgrep -f chain_v29a_synthd5_seed3 >/dev/null; then
        echo "[queue] $(date -Iseconds) v29a-seed3 already running; waiting" | tee -a "$LOG"
        sleep 120
        continue
    fi
    # find a GPU with < 5 GB used
    free_gpu=""
    while read -r idx used; do
        if [ "$idx" -le 1 ] && [ "$used" -lt 5000 ]; then
            free_gpu=$idx
            break
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ',')
    if [ -n "$free_gpu" ]; then
        echo "[queue] $(date -Iseconds) launching v29a-seed3 on GPU $free_gpu" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES=$free_gpu bash scripts/train_v29a_synthd5_seed3_0p6b_frozen_local.sh
        v29a3_done=true
    else
        echo "[queue] $(date -Iseconds) no free GPU yet (gpu0 used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' MiB')MB, gpu1 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' MiB')MB); sleeping" | tee -a "$LOG"
        sleep 60
    fi
done

# ----- remote: v30a seed 2 -----
echo "[queue] $(date -Iseconds) waiting for GH200 v30a-seed1 to finish" | tee -a "$LOG"
while true; do
    has_final=$(ssh "$REMOTE" "test -d $REMOTE_BASE/output/chain_v30a_synthd5_seed1_1p7b_frozen_gh200/final && echo yes || echo no" 2>/dev/null)
    if [ "$has_final" = "yes" ]; then
        echo "[queue] $(date -Iseconds) v30a-seed1 final detected; pulling and launching seed 2" | tee -a "$LOG"
        # rsync seed 1 ckpt down
        mkdir -p runs/chain_v30a_synthd5_seed1_1p7b_frozen_gh200
        rsync -az "$REMOTE:$REMOTE_BASE/output/chain_v30a_synthd5_seed1_1p7b_frozen_gh200/best"  runs/chain_v30a_synthd5_seed1_1p7b_frozen_gh200/ 2>&1 | tail -2 >> "$LOG" || true
        rsync -az "$REMOTE:$REMOTE_BASE/output/chain_v30a_synthd5_seed1_1p7b_frozen_gh200/final" runs/chain_v30a_synthd5_seed1_1p7b_frozen_gh200/ 2>&1 | tail -2 >> "$LOG" || true
        # launch seed 2 in the same tmux session
        ssh "$REMOTE" "tmux send-keys -t v30a 'CUDA_VISIBLE_DEVICES=0 bash Scripts/train_v30a_synthd5_seed2_1p7b_frozen_gh200.sh' Enter" 2>&1 | tee -a "$LOG"
        break
    fi
    sleep 120
done

echo "[queue] $(date -Iseconds) v30a-seed2 launched on GH200; queue done" | tee -a "$LOG"
