#!/usr/bin/env bash
# cloud_watchdog/watchdog.sh
# -----------------------------------------------------------------------------
# Polls queue/ for new job spec files and runs them inside detached tmux
# sessions, so SSH drops + laptop power-offs never kill in-flight work.
#
# A job spec is one JSON file in queue/ with at least:
#   {
#     "name": "niah_v3_softparity",
#     "cmd":  "python paper_tools/niah_eval.py --model output/.../best ...",
#     "cwd":  "/home/ubuntu/memory_residuals",       (optional, default: $REPO)
#     "venv": "/home/ubuntu/venv",                   (optional, default: $VENV)
#     "gpu":  "0",                                   (optional, default: "0")
#     "ntfy_topic": "memres-..."                     (optional)
#   }
#
# Lifecycle:  queue/ -> running/ -> done/ | failed/
# Each transition writes a status file with timestamps, exit code, log paths.
#
# Run as:    nohup paper_tools/cloud_watchdog/watchdog.sh > logs/watchdog.log 2>&1 &
# -----------------------------------------------------------------------------
set -u

REPO="${REPO:-$HOME/memory_residuals}"
VENV="${VENV:-$HOME/venv}"
WD="$REPO/paper_tools/cloud_watchdog"
QUEUE="$WD/queue"
RUN="$WD/running"
DONE="$WD/done"
FAIL="$WD/failed"
LOGS="$WD/logs"
NOTIFY="$WD/notify.sh"
POLL_SEC="${POLL_SEC:-30}"

mkdir -p "$QUEUE" "$RUN" "$DONE" "$FAIL" "$LOGS"

log() { printf '[watchdog %s] %s\n' "$(date '+%F %T')" "$*"; }

require() {
  for cmd in "$@"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      log "FATAL: missing required command: $cmd"
      exit 1
    fi
  done
}
require tmux jq

trap 'log "watchdog received SIGTERM/SIGINT, exiting"; exit 0' INT TERM

log "starting watchdog: poll=${POLL_SEC}s, queue=$QUEUE"

while :; do
  # Reap finished tmux sessions (move running/<name> -> done/ or failed/)
  for spec in "$RUN"/*.json; do
    [ -e "$spec" ] || continue
    name="$(jq -r .name "$spec" 2>/dev/null || true)"
    [ -z "$name" ] || [ "$name" = "null" ] && continue
    sess="cwd-$name"
    if ! tmux has-session -t "$sess" 2>/dev/null; then
      # Session ended; pick verdict from the per-job exit-code file.
      ec_file="$LOGS/$name.exit_code"
      if [ -f "$ec_file" ]; then
        ec="$(cat "$ec_file")"
      else
        ec="?"
      fi
      ts="$(date -Iseconds)"
      finished_dir="$DONE"
      verdict="done"
      if [ "$ec" != "0" ]; then
        finished_dir="$FAIL"
        verdict="failed"
      fi
      mv "$spec" "$finished_dir/"
      jq --arg ts "$ts" --arg ec "$ec" --arg verdict "$verdict" \
         '. + {finished_at: $ts, exit_code: ($ec|tonumber? // $ec), status: $verdict}' \
         "$finished_dir/$(basename "$spec")" > "$finished_dir/$(basename "$spec").tmp" \
         && mv "$finished_dir/$(basename "$spec").tmp" "$finished_dir/$(basename "$spec")"
      log "$name -> $verdict (exit=$ec)"
      topic="$(jq -r '.ntfy_topic // empty' "$finished_dir/$(basename "$spec")" 2>/dev/null)"
      if [ -n "$topic" ] && [ -x "$NOTIFY" ]; then
        "$NOTIFY" "$topic" "[$verdict] $name (exit=$ec)" "$LOGS/$name.log" || true
      fi
    fi
  done

  # Pick up new jobs from queue/ in oldest-first order.
  for spec in $(ls -1tr "$QUEUE" 2>/dev/null | grep -E '\.json$'); do
    spec_path="$QUEUE/$spec"
    name="$(jq -r .name "$spec_path" 2>/dev/null || true)"
    cmd="$(jq -r .cmd "$spec_path" 2>/dev/null || true)"
    cwd="$(jq -r '.cwd // empty' "$spec_path" 2>/dev/null)"
    venv="$(jq -r '.venv // empty' "$spec_path" 2>/dev/null)"
    gpu="$(jq -r '.gpu // "0"' "$spec_path" 2>/dev/null)"
    topic="$(jq -r '.ntfy_topic // empty' "$spec_path" 2>/dev/null)"
    if [ -z "$name" ] || [ -z "$cmd" ] || [ "$name" = "null" ] || [ "$cmd" = "null" ]; then
      log "skipping malformed spec: $spec"
      mv "$spec_path" "$FAIL/"
      continue
    fi

    sess="cwd-$name"
    if tmux has-session -t "$sess" 2>/dev/null; then
      log "session $sess already running; skipping enqueue of $spec"
      mv "$spec_path" "$RUN/"
      continue
    fi

    cwd="${cwd:-$REPO}"
    venv="${venv:-$VENV}"
    log_file="$LOGS/$name.log"
    ec_file="$LOGS/$name.exit_code"
    : > "$log_file"
    rm -f "$ec_file"

    # Build the command we run inside the tmux pane:
    #   1. activate venv (if exists), 2. cd into cwd, 3. set CUDA_VISIBLE_DEVICES,
    #   4. run the user command, 5. write exit code so we can pick a verdict later.
    inner=$(cat <<EOF
cd "$cwd" || exit 99
[ -f "$venv/bin/activate" ] && . "$venv/bin/activate"
export CUDA_VISIBLE_DEVICES="$gpu"
echo "[\$(date -Iseconds)] starting: $cmd" >> "$log_file"
( $cmd ) >> "$log_file" 2>&1
ec=\$?
echo "\$ec" > "$ec_file"
echo "[\$(date -Iseconds)] exit=\$ec" >> "$log_file"
EOF
    )

    # Move spec to running/ first so a crash mid-launch is debuggable.
    mv "$spec_path" "$RUN/"
    ts="$(date -Iseconds)"
    jq --arg ts "$ts" '. + {started_at: $ts, status: "running"}' \
       "$RUN/$spec" > "$RUN/$spec.tmp" && mv "$RUN/$spec.tmp" "$RUN/$spec"

    tmux new-session -d -s "$sess" "bash -lc '$inner'" || {
      log "tmux launch failed for $name; moving back to queue"
      mv "$RUN/$spec" "$FAIL/"
      continue
    }
    log "started $name in tmux session $sess (gpu=$gpu, log=$log_file)"
    if [ -n "$topic" ] && [ -x "$NOTIFY" ]; then
      "$NOTIFY" "$topic" "[start] $name on gpu $gpu" || true
    fi
  done

  sleep "$POLL_SEC"
done
