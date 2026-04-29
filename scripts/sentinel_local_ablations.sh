#!/usr/bin/env bash
# Local-side sentinel.  Polls the live v3 trainer logs every 60 s; when
# a v3 run reports its terminal "Saved" message (training loop exit),
# launches the corresponding v4 ablation on that same GPU.
#
#   v3_softparity_full   on GPU 0  ->  ablation B (chain_v4_embed_msc)
#   v3_attentionbase_full on GPU 1  ->  ablation C (chain_v4_hidden14_pgtv)
#
# The sentinel is one bash process; safe to nohup or run inside tmux.
# It exits cleanly once both ablations are launched.
set -u

REPO="${REPO:-/home/exx/Desktop/fine-tune/memory_residuals}"
cd "$REPO"
mkdir -p logs

V3_RUNS=(
    "0:chain_v3_softparity_full:scripts/train_ablation_b_embed_msc.sh:logs/chain_v4_embed_msc.log"
    "1:chain_v3_attentionbase_full:scripts/train_ablation_c_hidden_pgtv.sh:logs/chain_v4_hidden14_pgtv.log"
)

# Determine completion of a v3 run.  Heuristic: once the trainer is
# done, no python process is still loading its run name AND the log
# file's last 'step' line shows step >= max_steps - log_every (i.e.
# the trainer has finished its final logging).  We use the simpler
# "no live python -u process matching the run name" check.
is_done() {
    local run_name="$1"
    if pgrep -f "train_chain.py.+--run_name $run_name" >/dev/null 2>&1; then
        return 1
    fi
    if pgrep -f "train_chain.py.+$run_name" >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

launch_ablation() {
    local gpu="$1"
    local script="$2"
    local logfile="$3"
    local run_name="$(basename "$script" .sh | sed 's/^train_ablation_._//')"
    echo "[sentinel $(date '+%T')] launching $script on GPU $gpu (log -> $logfile)"
    CUDA_VISIBLE_DEVICES="$gpu" nohup bash "$script" > "$logfile" 2>&1 &
    echo "[sentinel $(date '+%T')]   pid=$!"
}

declare -A LAUNCHED

while :; do
    all_launched=1
    for entry in "${V3_RUNS[@]}"; do
        IFS=':' read -r gpu v3_name script logfile <<< "$entry"
        key="$gpu:$v3_name"
        if [ "${LAUNCHED[$key]:-0}" = "1" ]; then
            continue
        fi
        if is_done "$v3_name"; then
            launch_ablation "$gpu" "$script" "$logfile"
            LAUNCHED["$key"]=1
        else
            all_launched=0
        fi
    done
    if [ "$all_launched" = "1" ]; then
        echo "[sentinel $(date '+%T')] all ablations launched, exiting"
        break
    fi
    sleep 60
done
