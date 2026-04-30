#!/usr/bin/env bash
# cloud_handoff.sh
# -----------------------------------------------------------------------------
# Hand new local checkpoints off to the cloud GH200 watchdog so that
# evaluation continues even if the laptop powers off after the training
# runs finish.
#
# What it does:
#   1. rsync each $RUN/best/ from local -> cloud
#   2. enqueue NIAH + bootstrap-CI jobs for each checkpoint via the cloud
#      watchdog's enqueue.sh
#
# Run as:
#   paper_tools/cloud_handoff.sh chain_v3_softparity_full chain_v3_attentionbase_full
# -----------------------------------------------------------------------------
set -u

CLOUD_USER=ubuntu
CLOUD_HOST=192.222.50.225
CLOUD_REPO=/home/ubuntu/memory_residuals
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "usage: cloud_handoff.sh <run_name> [<run_name> ...]" >&2
  exit 2
fi

# Pull the ntfy topic from the cloud once so we can include it in queued specs.
TOPIC="$(ssh -o BatchMode=yes "$CLOUD_USER@$CLOUD_HOST" \
            "cat $CLOUD_REPO/paper_tools/cloud_watchdog/.ntfy_topic" 2>/dev/null \
        || echo '')"

for run in "$@"; do
  src="$LOCAL_REPO/output/$run/best/"
  if [ ! -d "$src" ]; then
    echo "[handoff] WARNING: $src does not exist; skipping $run"
    continue
  fi

  echo "[handoff] rsyncing $run/best to cloud..."
  rsync -avz --partial --mkpath "$src" \
    "$CLOUD_USER@$CLOUD_HOST:$CLOUD_REPO/output/$run/best/" 2>&1 | tail -3

  echo "[handoff] enqueueing eval jobs for $run..."
  niah_cmd="python paper_tools/niah_eval.py \
    --model_path output/$run/best \
    --filler_corpus paper_artifacts/chains/stage1_validation_s512.pt \
    --depths 1,5,10,20,30 --positions 0.1,0.5,0.9 --n_seeds 6 \
    --output paper_artifacts/eval/niah_${run}.json"

  eval_cmd="python paper_tools/eval_chain.py \
    --model_path output/$run/best \
    --corpora paper_artifacts/chains/stage1_validation_s512.pt \
              paper_artifacts/chains/stage1_test_s512.pt \
              paper_artifacts/chains/locomo_s512.pt \
    --names pg19_val pg19_test locomo \
    --score_window 4 --oracle_window 4 \
    --output paper_artifacts/eval/${run}_cloud_eval.json"

  bootstrap_cmd="python paper_tools/bootstrap_ci.py \
    --input paper_artifacts/eval/${run}_cloud_eval.json \
    --output paper_artifacts/eval/${run}_cloud_ci.json \
    --n_resamples 1000"

  ssh "$CLOUD_USER@$CLOUD_HOST" \
    "$CLOUD_REPO/paper_tools/cloud_watchdog/enqueue.sh \
       niah_${run} '$(echo "$niah_cmd" | tr -s ' ')' 0 '$TOPIC'"

  ssh "$CLOUD_USER@$CLOUD_HOST" \
    "$CLOUD_REPO/paper_tools/cloud_watchdog/enqueue.sh \
       eval_${run} '$(echo "$eval_cmd" | tr -s ' ')' 0 '$TOPIC'"

  ssh "$CLOUD_USER@$CLOUD_HOST" \
    "$CLOUD_REPO/paper_tools/cloud_watchdog/enqueue.sh \
       bootstrap_${run} '$(echo "$bootstrap_cmd" | tr -s ' ')' 0 '$TOPIC'"
done

echo "[handoff] done. Cloud queue:"
ssh "$CLOUD_USER@$CLOUD_HOST" "ls $CLOUD_REPO/paper_tools/cloud_watchdog/queue/"
