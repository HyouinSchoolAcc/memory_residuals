#!/usr/bin/env bash
# cloud_watchdog/notify.sh
# -----------------------------------------------------------------------------
# Push a notification to ntfy.sh.  Optionally tail the last N lines of a log
# file into the message body so you see context on your phone.
#
# Usage:  notify.sh <topic> <title> [log_path] [tail_lines]
# -----------------------------------------------------------------------------
set -u

topic="${1:-}"
title="${2:-cloud-watchdog}"
log_path="${3:-}"
tail_n="${4:-12}"

if [ -z "$topic" ]; then
  echo "notify.sh: missing topic" >&2
  exit 2
fi

body=""
if [ -n "$log_path" ] && [ -r "$log_path" ]; then
  body="$(tail -n "$tail_n" "$log_path" 2>/dev/null || true)"
fi

# Fall back to the title if there's no log body to show.
[ -z "$body" ] && body="$title"

curl --silent --show-error --max-time 8 \
     -H "Title: $title" \
     -H "Tags: brain,robot" \
     -d "$body" \
     "https://ntfy.sh/$topic" >/dev/null || {
  echo "notify.sh: ntfy push failed (topic=$topic)" >&2
  exit 1
}
