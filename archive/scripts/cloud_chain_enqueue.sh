#!/usr/bin/env bash
# Cloud-side sentinel: poll the watchdog's done/ + failed/ until the
# named "wait_for" job lands there, then enqueue "next_name" from
# "next_script".
#
# Usage:
#   cloud_chain_enqueue.sh <wait_for_name> <next_name> <next_script>
#
# Example (run on the cloud GPU):
#   nohup scripts/cloud_chain_enqueue.sh \
#       chain_v4_hidden14_msc \
#       chain_v4_hidden14_msc_contrast \
#       scripts/train_headline_plus_contrast.sh \
#       > paper_tools/cloud_watchdog/logs/chain_enqueue.log 2>&1 &
#
# Heartbeat every 5 minutes; final log line is the enqueue result.
set -eu

REPO="${REPO:-/home/ubuntu/memory_residuals}"
WD="$REPO/paper_tools/cloud_watchdog"
DONE="$WD/done"
FAIL="$WD/failed"
QUEUE="$WD/queue"
RUNNING="$WD/running"

wait_for="${1:?usage: cloud_chain_enqueue.sh <wait_for_name> <next_name> <next_script>}"
next_name="${2:?usage: cloud_chain_enqueue.sh <wait_for_name> <next_name> <next_script>}"
next_script="${3:?usage: cloud_chain_enqueue.sh <wait_for_name> <next_name> <next_script>}"

cd "$REPO"

log() { printf '[chain-enqueue %s] %s\n' "$(date '+%F %T')" "$*"; }

# A job spec exists in {queue,running,done,failed}/*_<name>.json (the
# unix timestamp prefix and .json suffix are added by enqueue.sh).
# We treat the most recently enqueued (latest timestamp) version as
# the canonical one; older done/failed entries are ignored so this
# sentinel survives kill-and-restart cycles where the same name has
# multiple specs floating around.
latest_in() {
    local dir="$1"
    ls -1tr "$dir"/*"_${wait_for}".json 2>/dev/null | tail -1
}
ts_of_spec() {
    local p="$1"
    [ -z "$p" ] && { echo 0; return; }
    basename "$p" | sed 's/_.*//'
}

log "waiting for job '$wait_for' to terminate (latest spec only)"

ts_start=$(date +%s)
loop_count=0
verdict=""
while :; do
    # Pick the latest spec for `wait_for` across all four buckets;
    # the bucket it sits in is the verdict.
    cur_q="$(latest_in "$QUEUE")"
    cur_r="$(latest_in "$RUNNING")"
    cur_d="$(latest_in "$DONE")"
    cur_f="$(latest_in "$FAIL")"
    ts_q="$(ts_of_spec "$cur_q")"
    ts_r="$(ts_of_spec "$cur_r")"
    ts_d="$(ts_of_spec "$cur_d")"
    ts_f="$(ts_of_spec "$cur_f")"

    # Find the bucket holding the highest-timestamp spec.
    max_ts=$ts_q;     winner="queue"
    if [ "$ts_r" -gt "$max_ts" ]; then max_ts=$ts_r; winner="running"; fi
    if [ "$ts_d" -gt "$max_ts" ]; then max_ts=$ts_d; winner="done"; fi
    if [ "$ts_f" -gt "$max_ts" ]; then max_ts=$ts_f; winner="failed"; fi

    case "$winner" in
        done)   verdict="done";   break ;;
        failed) verdict="failed"; break ;;
        queue|running) ;;  # still pending or in-flight
    esac

    if [ "$max_ts" -eq 0 ]; then
        log "no spec named '$wait_for' found in any bucket yet"
    fi
    loop_count=$((loop_count + 1))
    if [ $((loop_count % 10)) -eq 0 ]; then
        elapsed=$(( ($(date +%s) - ts_start) / 60 ))
        log "still waiting; latest spec in '$winner' (ts=$max_ts), elapsed ${elapsed} min"
    fi
    sleep 30
done

elapsed=$(( ($(date +%s) - ts_start) / 60 ))
log "'$wait_for' is $verdict after ${elapsed} min"

if [ "$verdict" = "failed" ]; then
    log "REFUSING to enqueue '$next_name' because '$wait_for' failed."
    log "If this is intentional, manually run: $next_script"
    exit 1
fi

# Enqueue the next job.  Source the ntfy topic if present.
TOPIC=$(cat "$WD/.ntfy_topic" 2>/dev/null || echo "")
log "enqueuing '$next_name' from '$next_script' (topic='$TOPIC')"
"$WD/enqueue.sh" "$next_name" "bash $next_script" 0 "$TOPIC"
log "enqueue complete; watchdog should pick it up within 30 s"
