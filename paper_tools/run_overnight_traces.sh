#!/usr/bin/env bash
# run_overnight_traces.sh
# -----------------------------------------------------------------------------
# Run routing_trace + counterfactual_eval on a fixed grid of (ckpt, corpus)
# pairs, serially.  Designed to be enqueued as a *single* watchdog job so the
# evals don't stomp each other on the GH200 while v4 training is running.
#
# Each call writes its output JSON to paper_artifacts/eval/, appends a one-
# line summary to paper_artifacts/eval/overnight_traces_summary.txt, and
# never aborts the wrapping script -- a per-call failure is logged but the
# remaining calls still execute.
#
# At the end we curl an ntfy push with the full summary file as the body so
# the operator gets every headline number on their phone in one go.
# -----------------------------------------------------------------------------
set -u

REPO="${REPO:-$HOME/memory_residuals}"
OUT_DIR="$REPO/paper_artifacts/eval"
SUMMARY="$OUT_DIR/overnight_traces_summary.txt"
NTFY_TOPIC="${NTFY_TOPIC:-memres-e6ebdc70}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUT_DIR"
{
  echo "=== overnight traces ==="
  echo "started:   $(date -Iseconds)"
  echo "host:      $(hostname)"
  echo "repo:      $REPO"
  echo "ntfy:      $NTFY_TOPIC"
  echo "---"
} | tee "$SUMMARY"

run_one() {
  local label="$1"; shift
  local out_json="$1"; shift
  local cmd_log="$OUT_DIR/${label}.cmdlog"
  echo ""                                                 | tee -a "$SUMMARY"
  echo "[start] $label"                                   | tee -a "$SUMMARY"
  echo "  out:  $out_json"                                | tee -a "$SUMMARY"
  echo "  cmd:  $*"                                       | tee -a "$SUMMARY"
  local t0; t0="$(date +%s)"
  if "$@" > "$cmd_log" 2>&1; then
    local t1; t1="$(date +%s)"
    local dur=$((t1 - t0))
    echo "[ok]    $label  (${dur}s)"                      | tee -a "$SUMMARY"
    if [ -f "$out_json" ]; then
      echo "--- summary ---"                              | tee -a "$SUMMARY"
      tail -n 40 "$cmd_log" | grep -E '^( |\}|\{|".*":)' | tee -a "$SUMMARY"
      echo "---"                                          | tee -a "$SUMMARY"
    fi
  else
    local rc=$?
    local t1; t1="$(date +%s)"
    local dur=$((t1 - t0))
    echo "[FAIL]  $label  (rc=$rc, ${dur}s)"              | tee -a "$SUMMARY"
    echo "  log:  $cmd_log"                               | tee -a "$SUMMARY"
    echo "--- last 20 lines of $cmd_log ---"              | tee -a "$SUMMARY"
    tail -n 20 "$cmd_log"                                 | tee -a "$SUMMARY"
    echo "---"                                            | tee -a "$SUMMARY"
  fi
}

# --- ckpt list ---
declare -A CKPTS=(
  [v3sp]="$REPO/output/chain_v3_softparity_full/best"
  [v3ab]="$REPO/output/chain_v3_attentionbase_full/best"
  [v2sp]="$REPO/output/chain_v2_phaseA_softparity_b4/best"
)

# --- corpus list ---
declare -A CORPORA=(
  [val]="$REPO/paper_artifacts/chains/stage1_validation_s512.pt"
  [locomo]="$REPO/paper_artifacts/chains/locomo_s512.pt"
)

# --- run grid ---
for ck_key in v3sp v3ab v2sp; do
  ckpt="${CKPTS[$ck_key]}"
  if [ ! -d "$ckpt" ] || [ ! -f "$ckpt/model.safetensors" ]; then
    echo "[SKIP]  ckpt missing: $ckpt"                    | tee -a "$SUMMARY"
    continue
  fi
  for co_key in val locomo; do
    corpus="${CORPORA[$co_key]}"
    if [ ! -f "$corpus" ]; then
      echo "[SKIP]  corpus missing: $corpus"              | tee -a "$SUMMARY"
      continue
    fi
    label="routing_${ck_key}_${co_key}"
    out="$OUT_DIR/${label}.json"
    run_one "$label" "$out" \
      $PYTHON "$REPO/paper_tools/routing_trace.py" \
      --model_path "$ckpt" \
      --corpus "$corpus" \
      --score_window 4 \
      --max_chains 40 \
      --output "$out"

    label="cf_${ck_key}_${co_key}"
    out="$OUT_DIR/${label}.json"
    run_one "$label" "$out" \
      $PYTHON "$REPO/paper_tools/counterfactual_eval.py" \
      --model_path "$ckpt" \
      --corpus "$corpus" \
      --depths 1 2 4 8 \
      --max_chains 30 \
      --output "$out"
  done
done

echo ""                                                   | tee -a "$SUMMARY"
echo "finished: $(date -Iseconds)"                        | tee -a "$SUMMARY"
echo "=== end overnight traces ==="                       | tee -a "$SUMMARY"

# --- final ntfy push with full summary as the message body ---
if [ -n "$NTFY_TOPIC" ]; then
  body="$(tail -n 200 "$SUMMARY")"
  curl -fsS -m 30 \
    -H "Title: overnight traces complete" \
    -H "Priority: high" \
    -H "Tags: tada,brain" \
    -d "$body" \
    "https://ntfy.sh/$NTFY_TOPIC" > /dev/null || true
fi
