#!/usr/bin/env bash
# Universal pull+eval watcher. Polls every ~2 minutes for any of the 8
# expected cells (6 local v27a/v27c, 2 remote v28c/v28d) to land its
# final/ checkpoint, then runs tools/eval_callback.py against the
# locked validation corpus and writes the result JSON to
# results/eval_v25_seed_pack_evpos/.
#
# Local cells: just check local final/.
# GH200 cells: SSH check remote final/ -> rsync -> eval.
#
# Designed to coexist with running training queues:
#   * eval is pinned to a CUDA device chosen via $EVAL_CUDA_DEV
#     (defaults to 1, i.e. share with the v27c queue; the eval is ~30s
#     for 0.6B / ~1.5min for 1.7B, so the contention is small).
#   * skips any cell whose result JSON already exists.
#
# Run inside `tmux` so it survives SSH/desktop disconnects.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
WLOG="$LOG_DIR/watcher_universal_eval.log"

REMOTE="ubuntu@192.222.50.225"
REMOTE_BASE="/home/ubuntu/memory_residuals"
LOCAL_BASE="$(pwd)"
EVAL_DIR="$LOCAL_BASE/results/eval_v25_seed_pack_evpos"
mkdir -p "$EVAL_DIR"

EVAL_CUDA_DEV="${EVAL_CUDA_DEV:-1}"
EVAL_CORPUS="paper_artifacts/chains/lme_val_s512_evpos.pt"

# Format: "cell_name:host"  (host = local | gh200)
CELLS=(
  "chain_v27a_v24a_no_depth_seed2_0p6b_frozen_local:local"
  "chain_v27a_v24a_no_depth_seed3_0p6b_frozen_local:local"
  "chain_v27a_v24a_no_depth_seed4_0p6b_frozen_local:local"
  "chain_v27c_v24a_no_floor_seed2_0p6b_frozen_local:local"
  "chain_v27c_v24a_no_floor_seed3_0p6b_frozen_local:local"
  "chain_v27c_v24a_no_floor_seed4_0p6b_frozen_local:local"
  "chain_v28c_no_probe_seed3_1p7b_frozen_gh200:gh200"
  "chain_v28d_no_probe_seed4_1p7b_frozen_gh200:gh200"
)

POLL_INTERVAL="${POLL_INTERVAL:-120}"
MAX_HOURS="${MAX_HOURS:-30}"

echo "[$(date -Iseconds)] watcher start; cells=${#CELLS[@]}; eval_cuda=$EVAL_CUDA_DEV; corpus=$EVAL_CORPUS" >> "$WLOG"

t0=$(date +%s)
deadline=$(( t0 + MAX_HOURS * 3600 ))

while :; do
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    echo "[$(date -Iseconds)] hit MAX_HOURS=$MAX_HOURS; exiting" >> "$WLOG"
    exit 0
  fi

  for ENTRY in "${CELLS[@]}"; do
    CELL="${ENTRY%:*}"
    HOST="${ENTRY#*:}"
    LOCAL_OUT_DIR="$LOCAL_BASE/output/$CELL"
    EVAL_TAG="${CELL#chain_}_lme_val_evpos"
    EVAL_JSON="$EVAL_DIR/${EVAL_TAG}.json"

    if [ -s "$EVAL_JSON" ]; then continue; fi

    # 1) Make sure local final/ exists. For gh200 cells we need to rsync
    #    first.
    if [ "$HOST" = "gh200" ]; then
      if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE" \
           "test -f $REMOTE_BASE/output/$CELL/final/config.json" 2>/dev/null; then
        continue
      fi
      if [ ! -f "$LOCAL_OUT_DIR/final/config.json" ]; then
        echo "[$(date -Iseconds)] $CELL: remote final/ ready, syncing..." >> "$WLOG"
        mkdir -p "$LOCAL_OUT_DIR"
        if ! rsync -aq --partial \
             "$REMOTE:$REMOTE_BASE/output/$CELL/final/" \
             "$LOCAL_OUT_DIR/final/" >> "$WLOG" 2>&1; then
          echo "[$(date -Iseconds)] $CELL: rsync FAILED rc=$?" >> "$WLOG"
          continue
        fi
      fi
    else
      if [ ! -f "$LOCAL_OUT_DIR/final/config.json" ]; then
        continue
      fi
    fi

    # 2) Run eval.
    echo "[$(date -Iseconds)] $CELL: running eval_callback.py on cuda=$EVAL_CUDA_DEV" >> "$WLOG"
    CUDA_VISIBLE_DEVICES="$EVAL_CUDA_DEV" python -u tools/eval_callback.py \
      --model_path "$LOCAL_OUT_DIR/final" \
      --corpora "$EVAL_CORPUS" \
      --names lme_val \
      --output "$EVAL_JSON" \
      >> "$WLOG" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -s "$EVAL_JSON" ]; then
      # Tiny human-readable summary line.
      python -c "
import json
d = json.load(open('$EVAL_JSON'))
v = next(iter(d.values()))
dnm = v.get('pa_cb_dnm'); dsh = v.get('pa_cb_dsh'); el = v.get('pa_cb_evidence_lift')
n = v.get('n_chains_scored', '?')
print(f'$CELL  Δ_dnm={dnm:+.4f}  Δ_dsh={dsh:+.4f}  ev_lift={el:+.4f}  n_chains={n}')
" >> "$WLOG" 2>&1
      echo "[$(date -Iseconds)] $CELL: eval ok -> $EVAL_JSON" >> "$WLOG"
    else
      echo "[$(date -Iseconds)] $CELL: eval FAILED rc=$rc" >> "$WLOG"
    fi
  done

  all_done=1
  for ENTRY in "${CELLS[@]}"; do
    CELL="${ENTRY%:*}"
    EVAL_TAG="${CELL#chain_}_lme_val_evpos"
    if [ ! -s "$EVAL_DIR/${EVAL_TAG}.json" ]; then all_done=0; break; fi
  done
  if [ "$all_done" -eq 1 ]; then
    echo "[$(date -Iseconds)] watcher: ALL 8 cells synced + evaluated. Exiting." >> "$WLOG"
    exit 0
  fi

  sleep "$POLL_INTERVAL"
done
