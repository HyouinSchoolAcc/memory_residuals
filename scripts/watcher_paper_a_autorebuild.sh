#!/usr/bin/env bash
# Paper-A auto-rebuild watcher. Polls every 2 min; whenever a new
# memres eval JSON appears in results/eval_v25_seed_pack_evpos/,
# regenerates numbers + figures + paper PDF.
#
# Run inside tmux so it survives SSH disconnects.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG="logs/paper_a_autorebuild.log"
EVAL_DIR="results/eval_v25_seed_pack_evpos"
PAPER_DIR="paper_a"

POLL_INTERVAL="${POLL_INTERVAL:-120}"
MAX_HOURS="${MAX_HOURS:-24}"

# Track which JSONs we have already seen so we only rebuild when the
# input changes.
last_signature=""

echo "[$(date -Iseconds)] paper_a auto-rebuild watcher start" >> "$LOG"
t0=$(date +%s)
deadline=$(( t0 + MAX_HOURS * 3600 ))

# Initial build
echo "[$(date -Iseconds)] initial build" >> "$LOG"
bash "$PAPER_DIR/build.sh" >> "$LOG" 2>&1 || \
    echo "[$(date -Iseconds)] initial build FAILED" >> "$LOG"

while :; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[$(date -Iseconds)] hit MAX_HOURS=$MAX_HOURS; exiting" >> "$LOG"
    exit 0
  fi

  # Build a stable signature over all relevant inputs (any new ckpt eval
  # → signature changes → rebuild)
  signature=$(ls -la "$EVAL_DIR"/v27{a,b,c}*.json "$EVAL_DIR"/v28{a,b,c,d}*.json 2>/dev/null \
             | awk '{print $5,$6,$7,$8,$9}' | sha256sum)
  if [ "$signature" != "$last_signature" ]; then
    if [ -n "$last_signature" ]; then  # skip the very first cycle (already built)
      echo "[$(date -Iseconds)] new eval JSON detected; rebuilding" >> "$LOG"
      bash "$PAPER_DIR/build.sh" >> "$LOG" 2>&1 \
        && echo "[$(date -Iseconds)] rebuild ok" >> "$LOG" \
        || echo "[$(date -Iseconds)] rebuild FAILED" >> "$LOG"
    fi
    last_signature="$signature"
  fi

  sleep "$POLL_INTERVAL"
done
