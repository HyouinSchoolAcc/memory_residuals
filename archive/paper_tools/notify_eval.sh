#!/usr/bin/env bash
#
# Watch one or more trainer logs and push every EVAL line to ntfy.sh as a
# phone notification.  Free, no account, no API key -- just a topic name
# only you and the watcher know.
#
# Setup (one-time):
#   1. pick a hard-to-guess topic, e.g. memres-exx-$(openssl rand -hex 4)
#   2. install the ntfy app on your phone, subscribe to that topic
#   3. run this script in the background:
#         nohup paper_tools/notify_eval.sh memres-exx-3a9f \
#             logs/chain_v2_phaseA_softparity_b4.log \
#             logs/chain_v2_abl_residual_mode.log \
#             > logs/notify.log 2>&1 &
#
# Usage:  notify_eval.sh <ntfy_topic> <log1> [<log2> ...]
#
# The script runs `tail -F` on each log and posts every line containing
# "EVAL @ step" or "Saved checkpoint" to ntfy.sh.  Reconnect-on-rotate is
# handled by tail -F.  Lines are deduped per-process (each tail's PID
# is its own context, no global state).

set -u
topic="${1:-}"
shift || true
if [[ -z "$topic" || $# -eq 0 ]]; then
    echo "usage: $0 <ntfy_topic> <log1> [<log2> ...]" >&2
    exit 2
fi

push() {
    local title="$1"
    local body="$2"
    # ntfy.sh accepts plain text body; Title and Priority are HTTP headers.
    curl --silent --max-time 5 \
        -H "Title: $title" \
        -H "Priority: default" \
        -H "Tags: chart_with_upwards_trend" \
        -d "$body" \
        "https://ntfy.sh/${topic}" > /dev/null \
        || echo "[notify] push failed for: $title" >&2
}

push "memres watcher started" "tracking: $*"

for log in "$@"; do
    (
        tag="$(basename "$log" .log)"
        tail -F -n 0 "$log" 2>/dev/null | while IFS= read -r line; do
            case "$line" in
                *"EVAL @ step"*)
                    # Strip leading whitespace, send the metric body verbatim.
                    push "$tag eval" "${line#"${line%%[![:space:]]*}"}"
                    ;;
                *"Saved checkpoint"*)
                    push "$tag ckpt" "${line#"${line%%[![:space:]]*}"}"
                    ;;
                *"Traceback"*|*"OutOfMemoryError"*|*"NaN"*)
                    push "$tag ERROR" "$line"
                    ;;
            esac
        done
    ) &
done

# Reap children when this script is killed.
trap 'kill 0' SIGINT SIGTERM
wait
