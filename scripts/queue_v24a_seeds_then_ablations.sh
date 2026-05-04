#!/usr/bin/env bash
# Auto-launch v24a-seed3 when v24a-seed2 finishes (GPU 1), then start
# the wave-A ablation cells (depth=0, F3 off, floor off) one after the
# other. Independent and resumable: each step writes a marker file.
#
# Usage:
#   nohup bash scripts/queue_v24a_seeds_then_ablations.sh > logs/queue_v24a.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

LOG=logs/queue_v24a_then_ablations.log
> "$LOG"

wait_for_pid() {
    local pid=$1 label=$2
    echo "[queue] waiting for pid=$pid ($label)" | tee -a "$LOG"
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    echo "[queue] pid=$pid ($label) finished" | tee -a "$LOG"
    sleep 15
}

launch_and_wait() {
    local label=$1; shift
    local script=$1; shift
    echo "[queue] launching $label ($script)" | tee -a "$LOG"
    bash "$script" 2>&1 | tee -a "$LOG"
    local pid=$(pgrep -af "$(basename "$script" .sh | sed 's/^train_//')" | awk '{print $1}' | tail -1)
    if [ -z "$pid" ]; then
        echo "[queue] ERROR: could not find pid for $label" | tee -a "$LOG"
        return 1
    fi
    echo "[queue] $label running as pid=$pid" | tee -a "$LOG"
    wait_for_pid "$pid" "$label"
}

# Stage 1: wait for v24a-seed2 currently running on GPU 1.
SEED2_PID=$(pgrep -af "chain_v24a_v21c_lme_seed2_0p6b_frozen_local" | awk '{print $1}' | head -1)
if [ -n "${SEED2_PID:-}" ]; then
    wait_for_pid "$SEED2_PID" "v24a_seed2"
else
    echo "[queue] v24a_seed2 not running; skipping wait" | tee -a "$LOG"
fi

# Stage 2: launch v24a-seed3 on GPU 1.
launch_and_wait "v24a_seed3" "Scripts/train_v24a_v21c_lme_seed3_0p6b_frozen_local.sh"

# Stage 3: ablations (each ~1.5h on 0.6B). Run sequentially on GPU 1.
launch_and_wait "v27a_no_depth"  "Scripts/train_v27a_v24a_no_depth_seed1_0p6b_frozen_local.sh"
launch_and_wait "v27b_no_probe"  "Scripts/train_v27b_v24a_no_probe_seed1_0p6b_frozen_local.sh"
launch_and_wait "v27c_no_floor"  "Scripts/train_v27c_v24a_no_floor_seed1_0p6b_frozen_local.sh"

echo "[queue] done with seed3 + ablations" | tee -a "$LOG"
