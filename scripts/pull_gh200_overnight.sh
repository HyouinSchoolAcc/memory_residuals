#!/usr/bin/env bash
# Polls GH200 every 2 min for the overnight cells to finish, then
# rsyncs best/ + final/ ckpts down to local output/ so the watcher can
# evaluate them. Idempotent. Exits when all 4 cells have been pulled.
#
# Cells:
#   chain_v27b_v24a_no_probe_seed3_0p6b_frozen_gh200
#   chain_v27b_v24a_no_probe_seed4_0p6b_frozen_gh200
#   chain_v28a_no_probe_seed1_1p7b_frozen_gh200
#   chain_v28b_no_probe_seed2_1p7b_frozen_gh200
#
# Usage:
#   nohup bash scripts/pull_gh200_overnight.sh > logs/pull_gh200_overnight.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

REMOTE=ubuntu@192.222.50.225
REMOTE_BASE=/home/ubuntu/memory_residuals/output
LOG=logs/pull_gh200_overnight.log
mkdir -p logs
echo "[pull] $(date -Iseconds) start" | tee -a "$LOG"

declare -a CELLS=(
    "chain_v27b_v24a_no_probe_seed3_0p6b_frozen_gh200"
    "chain_v27b_v24a_no_probe_seed4_0p6b_frozen_gh200"
    "chain_v28a_no_probe_seed1_1p7b_frozen_gh200"
    "chain_v28b_no_probe_seed2_1p7b_frozen_gh200"
)

declare -A DONE=()

while true; do
    for cell in "${CELLS[@]}"; do
        if [ "${DONE[$cell]:-}" = "yes" ]; then continue; fi
        has_final=$(ssh "$REMOTE" "test -d $REMOTE_BASE/$cell/final && echo yes || echo no" 2>/dev/null)
        if [ "$has_final" = "yes" ]; then
            mkdir -p "output/$cell"
            echo "[pull] $(date -Iseconds) rsyncing $cell" | tee -a "$LOG"
            rsync -az "$REMOTE:$REMOTE_BASE/$cell/best"  "output/$cell/" 2>&1 | tail -2 >> "$LOG" || true
            rsync -az "$REMOTE:$REMOTE_BASE/$cell/final" "output/$cell/" 2>&1 | tail -2 >> "$LOG" || true
            echo "[pull] $(date -Iseconds) rsync done for $cell" | tee -a "$LOG"
            DONE[$cell]=yes
        fi
    done

    # All done?
    all=true
    for cell in "${CELLS[@]}"; do
        if [ "${DONE[$cell]:-}" != "yes" ]; then all=false; break; fi
    done
    if $all; then
        echo "[pull] $(date -Iseconds) all cells pulled; exit" | tee -a "$LOG"
        break
    fi
    sleep 120
done
