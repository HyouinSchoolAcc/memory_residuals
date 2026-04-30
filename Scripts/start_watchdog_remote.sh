#!/usr/bin/env bash
# One-shot remote-only launcher: start the cloud_watchdog daemon and
# heartbeat in fully detached sessions so they survive the parent
# ssh disconnect. Idempotent: kills any stale instances first.
set -eu
cd "$(dirname "$0")/.."

LOG=tools/cloud_watchdog/logs
mkdir -p "$LOG"

TOPIC=$(cat tools/cloud_watchdog/.ntfy_topic)
echo "ntfy topic: $TOPIC"

pkill -f 'cloud_watchdog/watchdog\.sh'  2>/dev/null || true
pkill -f 'cloud_watchdog/heartbeat\.sh' 2>/dev/null || true
sleep 1

setsid nohup tools/cloud_watchdog/watchdog.sh \
  > "$LOG/watchdog.log" 2>&1 < /dev/null &
disown
echo "watchdog launched"

setsid nohup tools/cloud_watchdog/heartbeat.sh "$TOPIC" 1800 \
  > "$LOG/heartbeat.log" 2>&1 < /dev/null &
disown
echo "heartbeat launched"

sleep 2
echo "--- live cloud_watchdog procs ---"
pgrep -af cloud_watchdog | grep -v "$$" | grep -v "start_watchdog_remote" || echo "(no procs)"
