#!/usr/bin/env bash
# cloud_watchdog/enqueue.sh
# -----------------------------------------------------------------------------
# Convenience wrapper to drop a job spec into the watchdog queue.
#
# Usage:  enqueue.sh <name> <cmd> [gpu] [ntfy_topic]
#   name        short identifier, used as tmux session and log file name
#   cmd         full shell command line, single-quoted if it contains spaces
#   gpu         CUDA_VISIBLE_DEVICES value (default: "0")
#   ntfy_topic  optional ntfy topic for phone notifications
# -----------------------------------------------------------------------------
set -eu

WD="$(cd "$(dirname "$0")" && pwd)"
QUEUE="$WD/queue"
mkdir -p "$QUEUE"

name="${1:?usage: enqueue.sh <name> <cmd> [gpu] [ntfy_topic]}"
cmd="${2:?usage: enqueue.sh <name> <cmd> [gpu] [ntfy_topic]}"
gpu="${3:-0}"
topic="${4:-}"

ts="$(date +%s)"
spec="$QUEUE/${ts}_${name}.json"

if [ -n "$topic" ]; then
  cat > "$spec" <<EOF
{
  "name": "$name",
  "cmd":  $(printf '%s' "$cmd" | python3 -c "import json,sys;print(json.dumps(sys.stdin.read()))"),
  "gpu":  "$gpu",
  "ntfy_topic": "$topic",
  "enqueued_at": "$(date -Iseconds)"
}
EOF
else
  cat > "$spec" <<EOF
{
  "name": "$name",
  "cmd":  $(printf '%s' "$cmd" | python3 -c "import json,sys;print(json.dumps(sys.stdin.read()))"),
  "gpu":  "$gpu",
  "enqueued_at": "$(date -Iseconds)"
}
EOF
fi

echo "queued: $spec"
