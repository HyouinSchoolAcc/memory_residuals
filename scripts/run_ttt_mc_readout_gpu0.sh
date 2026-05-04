#!/usr/bin/env bash
# v18 read-side localizer: extends §5 by adding MemoryReadout's
# Q/K/V to the TTT-able set.  Decision rule:
#   ttt_lift_vs_floor  > +0.30 -> read-side probe loss is the right
#                                 v18 intervention (architecture has
#                                 capacity; joint training collapsed it)
#   in (0, +0.30]              -> probe loss + multi-layer readout depth
#   <= 0                       -> multi-layer readout depth is also
#                                 load-bearing (architectural pivot)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /home/exx/Desktop/fine-tune/memory_residuals

OUT=results/ttt_mc_v18pre
LOG=logs/ttt_mc_v18pre/gpu0_extended_ttt.log
mkdir -p "$OUT" "$(dirname "$LOG")"

# Mirror §5's three baseline cells (v14k/v15a on D4v2; later v15e on D4v2)
# but add MemoryReadout Q/K/V to the TTT-able set so we can separate
# "joint-trained readout collapse" from "readout architecture defect".

declare -a CELLS=(
  "v15a|chain_v15a_d4v2_norm_replicate_local|synthd4v2_persona_callback_val_s512"
  "v17a|chain_v17a_f2_codes_0p6b_frozen_local|synthd5_random_codes_val_s512"
)

for cell in "${CELLS[@]}"; do
  IFS='|' read -r tag ckpt corpus <<<"$cell"
  CKPT=Runs/${ckpt}/best
  CORPUS=paper_artifacts/chains/${corpus}.pt
  for mode in qkv qkv_reset; do
    for init in writer iid; do
      OUT_JSON="$OUT/${tag}__${mode}__${init}.json"
      echo "==========================================" | tee -a "$LOG"
      echo "RUN: ${tag} mode=${mode} init=${init}" | tee -a "$LOG"
      echo "==========================================" | tee -a "$LOG"
      python tools/eval_ttt_mc_readout.py \
        --ckpt "$CKPT" \
        --eval_corpus "$CORPUS" \
        --n_chains 16 \
        --ttt_steps 80 \
        --ttt_lr_mc 1e-2 \
        --ttt_lr_readout 1e-3 \
        --readout_unfreeze "$mode" \
        --init_mode "$init" \
        --seed 0 \
        --out "$OUT_JSON" 2>&1 | tee -a "$LOG"
    done
  done
done
echo "GPU0 read-side localizer DONE" | tee -a "$LOG"
