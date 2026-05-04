#!/usr/bin/env bash
# Waits for v25a-seed7 to finish on the GH200, then kicks off
# scripts/gh200_overnight_queue.sh in the same shell. Idempotent: a
# second invocation while the queue is running becomes a no-op via the
# .lock file.
#
# Run on GH200 (after sourcing venv):
#     nohup bash scripts/gh200_overnight_launcher.sh > logs/gh200_launcher.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=logs/gh200_overnight_launcher.log
LOCK=.gh200_overnight_queue.lock
mkdir -p logs
echo "[launcher] $(date -Iseconds) start" | tee -a "$LOG"

if [ -e "$LOCK" ]; then
    echo "[launcher] $LOCK exists; queue already running. exit." | tee -a "$LOG"
    exit 0
fi

while true; do
    n=$(pgrep -af train_chain | grep -c -E 'chain_v25a_v21c_lme_seed7' || true)
    if [ "$n" -eq 0 ]; then
        echo "[launcher] $(date -Iseconds) v25a-seed7 not running; starting queue" | tee -a "$LOG"
        break
    fi
    echo "[launcher] $(date -Iseconds) v25a-seed7 still running (n=$n); sleep 60" >> "$LOG"
    sleep 60
done

echo "$$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
bash scripts/gh200_overnight_queue.sh 2>&1 | tee -a logs/gh200_overnight.log
echo "[launcher] $(date -Iseconds) queue finished" | tee -a "$LOG"
