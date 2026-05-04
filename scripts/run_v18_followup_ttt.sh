#!/usr/bin/env bash
# Wait for v18a/best and v18b/best checkpoints to land, then re-run
# §5 (TTT-on-M_c, frozen read-side) on each.  Decisive empirical
# answer to "did the read-side probe loss restore read-side capacity?"
#
# Decision (mirrors v17 §5 schema):
#   ttt_lift_vs_floor > +0.30  -> readout architecture has capacity;
#                                 v18 read-side intervention WORKED.
#   in (0, +0.30]              -> partial; multi-layer readout depth
#                                 (v19 candidate) likely also needed.
#   <= 0                       -> v18 intervention insufficient;
#                                 multi-layer readout depth becomes
#                                 the next architectural change.
#
# Evidence_lift trajectory (read straight from training log) is the
# fast in-flight signal; this §5 re-run is the gold-standard
# cross-check that survives any best/-checkpoint quirks.
set -euo pipefail
cd /home/exx/Desktop/fine-tune/memory_residuals

CORPUS=paper_artifacts/chains/synthd5_random_codes_val_s512.pt
N=32
STEPS=50
LR=1e-2
OUT=results/ttt_mc_v18post
LOG=logs/ttt_mc_v18post/v18_followup.log
mkdir -p "$OUT" "$(dirname "$LOG")"

# Wait for both v18 best/ checkpoints to land.
for tag in v18a_f3 v18b_f2f3; do
    case $tag in
        v18a_f3)   ckpt_dir=output/chain_v18a_f3_codes_0p6b_frozen_local/best ;;
        v18b_f2f3) ckpt_dir=output/chain_v18b_f2f3_codes_0p6b_frozen_local/best ;;
    esac
    echo "[followup] waiting for $ckpt_dir/model.safetensors" | tee -a "$LOG"
    elapsed=0
    while [ ! -f "$ckpt_dir/model.safetensors" ]; do
        if [ $elapsed -ge 18000 ]; then
            echo "[followup] TIMEOUT waiting for $ckpt_dir; skipping" | tee -a "$LOG"
            continue 2
        fi
        sleep 60
        elapsed=$((elapsed + 60))
    done
    echo "[followup] $ckpt_dir landed" | tee -a "$LOG"

    # Two §5 cells per ckpt: writer-init + iid-init.
    for init in writer iid; do
        echo "==========================================" | tee -a "$LOG"
        echo "RUN: $tag init=$init" | tee -a "$LOG"
        echo "==========================================" | tee -a "$LOG"
        # Use whichever GPU is currently idle.  Try GPU 1 first
        # (frees up after v18b finishes); fall back to GPU 0.
        for gpu in 1 0; do
            mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu | tr -d ' ')
            if [ "$mem" -lt 5000 ]; then
                CUDA_VISIBLE_DEVICES=$gpu python tools/eval_ttt_mc.py \
                    --ckpt "$ckpt_dir" \
                    --eval_corpus "$CORPUS" \
                    --n_chains "$N" \
                    --ttt_steps "$STEPS" \
                    --ttt_lr "$LR" \
                    --init_mode "$init" \
                    --seed 0 \
                    --out "${OUT}/${tag}__${init}.json" 2>&1 | tee -a "$LOG"
                break
            fi
        done
    done
done
echo "[followup] DONE" | tee -a "$LOG"
