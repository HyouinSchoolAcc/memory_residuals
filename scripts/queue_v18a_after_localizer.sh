#!/usr/bin/env bash
# Wait for the v18 read-side localizer (GPU 0) to finish, then
# launch v18a on GPU 0.  Detects completion by the marker line
# "GPU0 read-side localizer DONE" written at the end of
# scripts/run_ttt_mc_readout_gpu0.sh.
set -euo pipefail
cd /home/exx/Desktop/fine-tune/memory_residuals

LOG=logs/ttt_mc_v18pre/gpu0_extended_ttt.log
MARKER="GPU0 read-side localizer DONE"
TIMEOUT_S=2400  # 40 min safety cap
SLEEP_S=15
elapsed=0

echo "[queue_v18a] waiting for localizer marker in $LOG"
while ! grep -q "$MARKER" "$LOG" 2>/dev/null; do
    if [ $elapsed -ge $TIMEOUT_S ]; then
        echo "[queue_v18a] TIMEOUT waiting for localizer; aborting"
        exit 1
    fi
    sleep $SLEEP_S
    elapsed=$((elapsed + SLEEP_S))
    if (( elapsed % 60 == 0 )); then
        echo "[queue_v18a] still waiting ($elapsed s elapsed)"
    fi
done

echo "[queue_v18a] localizer done; launching v18a on GPU 0"
sleep 5  # let the GPU drain
bash Scripts/train_v18a_f3_codes_0p6b_frozen_local.sh
echo "[queue_v18a] v18a launched"
