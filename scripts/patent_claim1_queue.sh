#!/usr/bin/env bash
# Claim-alignment evidence queue (2x H100).
#
# Motivated by external review of the patent draft:
#  1. Claim 1's stage 3 is the single-round 2K-softmax judge
#     (writer_kind=original, update_mode=competitive), but every headline
#     LME checkpoint used the iterative slot-attention judge. Train the
#     exact claim-1 embodiment with 4 seeds.
#  2. Closest-prior-art stage-3 baselines, single-variable vs claim-1:
#       gated_nojudge : simple gated replacement, no competition
#       selfattn      : RMC-style self-attention over [old||new], no
#                       learned judge query
# All cells use the identical v27b headline recipe otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/patent_experiments

SIZE_FLAGS="--pretrained Qwen/Qwen3-0.6B --memres_num_vectors 128 \
  --memres_extraction_depth 4 --memres_num_blocks 8"

cell() {  # cell <gpu> <name> <seed> <writer_kind> <update_mode>
  CUDA_VISIBLE_DEVICES="$1" bash scripts/patent_train_cell.sh "$2" "$3" \
    $SIZE_FLAGS --memres_writer_kind "$4" --memres_update_mode "$5"
}

eval_ckpt() {  # eval_ckpt <gpu> <run> <tag>
  CUDA_VISIBLE_DEVICES="$1" python tools/eval_callback.py \
    --model_path "runs/$2/final" \
    --corpora paper_artifacts/chains/lme_val_s512_evpos.pt \
    --names lme_val \
    --output "results/patent_experiments/eval_trained_$3.json"
}

echo "=== round 1: claim1 seeds 1+2 ==="
cell 0 chain_patent_claim1judge_seed1_0p6b 1 original competitive &
P0=$!
cell 1 chain_patent_claim1judge_seed2_0p6b 2 original competitive &
P1=$!
wait $P0; wait $P1

echo "=== round 2: claim1 seed 3 + selfattn ==="
cell 0 chain_patent_claim1judge_seed3_0p6b 3 original competitive &
P0=$!
cell 1 chain_patent_selfattn_seed1_0p6b 1 original selfattn &
P1=$!
wait $P0; wait $P1

echo "=== round 3: gated_nojudge + claim1 seed 4 ==="
cell 0 chain_patent_gatednojudge_seed1_0p6b 1 original gated_nojudge &
P0=$!
cell 1 chain_patent_claim1judge_seed4_0p6b 4 original competitive &
P1=$!
wait $P0; wait $P1

echo "=== evals ==="
eval_ckpt 0 chain_patent_claim1judge_seed1_0p6b claim1judge_seed1 &
P0=$!
eval_ckpt 1 chain_patent_claim1judge_seed2_0p6b claim1judge_seed2 &
P1=$!
wait $P0; wait $P1
eval_ckpt 0 chain_patent_claim1judge_seed3_0p6b claim1judge_seed3 &
P0=$!
eval_ckpt 1 chain_patent_claim1judge_seed4_0p6b claim1judge_seed4 &
P1=$!
wait $P0; wait $P1
eval_ckpt 0 chain_patent_selfattn_seed1_0p6b selfattn_seed1 &
P0=$!
eval_ckpt 1 chain_patent_gatednojudge_seed1_0p6b gatednojudge_seed1 &
P1=$!
wait $P0; wait $P1

echo CLAIM1_QUEUE_DONE
