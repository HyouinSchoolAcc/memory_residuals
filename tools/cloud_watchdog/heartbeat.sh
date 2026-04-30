#!/usr/bin/env bash
# cloud_watchdog/heartbeat.sh
# -----------------------------------------------------------------------------
# Periodically push a single phone notification summarizing the watchdog state:
# how many jobs queued, running, done, failed; GPU util/temp; disk free.
#
# Run as:    nohup tools/cloud_watchdog/heartbeat.sh <topic> 1800 \
#                  > logs/heartbeat.log 2>&1 &
#            (push every 30 minutes)
# -----------------------------------------------------------------------------
set -u

topic="${1:?usage: heartbeat.sh <topic> [interval_sec]}"
interval="${2:-1800}"

WD="$(cd "$(dirname "$0")" && pwd)"
NOTIFY="$WD/notify.sh"

while :; do
  q=$(ls -1 "$WD/queue" 2>/dev/null | grep -c '\.json$' || echo 0)
  r=$(ls -1 "$WD/running" 2>/dev/null | grep -c '\.json$' || echo 0)
  d=$(ls -1 "$WD/done" 2>/dev/null | grep -c '\.json$' || echo 0)
  f=$(ls -1 "$WD/failed" 2>/dev/null | grep -c '\.json$' || echo 0)
  gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu \
              --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'no-gpu')"
  disk="$(df -h / 2>/dev/null | awk 'NR==2 {print $4 " free of " $2}')"
  msg="q=$q r=$r d=$d f=$f | gpu=$gpu_line | disk=$disk"
  "$NOTIFY" "$topic" "heartbeat $(date '+%H:%M')" /dev/null 0 || true
  curl --silent --max-time 8 \
       -H "Title: heartbeat $(date '+%H:%M')" \
       -d "$msg" \
       "https://ntfy.sh/$topic" >/dev/null 2>&1 || true
  sleep "$interval"
done
