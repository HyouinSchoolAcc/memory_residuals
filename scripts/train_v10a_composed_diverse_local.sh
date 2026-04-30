#!/usr/bin/env bash
# v10a COMPOSED CURRICULUM + long context + carry_state (LOCAL H100 GPU 0)
#
# Goal: close the train/eval distribution mismatch that left standard
# Δ_sh-m ≈ 0 through v8/v9 while keeping the v9 / v9c phase-aligned
# callback win.
#
# Diff vs v9c (the current leading cell):
#   - window_k 3 -> 8            : match the eval distribution (40-sess
#                                  sequential M_c) much more closely.
#   - carry_state OFF -> ON      : persist M_c across TBPTT windows so
#                                  the readout sees deep-update M_c
#                                  during training, not only fresh-3-
#                                  sess M_c.
#   - curriculum_competition 1.0 -> 0.6 : still dominantly competition
#                                  (so the judge keeps learning keep-
#                                  vs-write), but 30% evidence-callback
#                                  windows give the writer + readout
#                                  exposure to longer credit-assignment
#                                  chains and the remainder samples
#                                  contiguous windows.
#   - curriculum_evidence   0.0 -> 0.3
#   - callback_loss_weight  3.0 -> 2.5 : slightly softer; v9/v9c showed
#                                  peak-then-decay consistent with
#                                  callback-token over-specialisation.
#   - steps 4000 -> 8000         : double budget; give composed recipe
#                                  room to converge on the richer
#                                  distribution without rushing.
#
# Corpus: v6_lme_msc (same as v9c). Source weights preserved.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
export CUDA_VISIBLE_DEVICES=0
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode simple_gate \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/v6_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 8 \
    --carry_state \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 8000 \
    --warmup 300 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 2.5 \
    --callback_window_bias 0.2 \
    --curriculum_evidence_bias 0.3 \
    --curriculum_competition_bias 0.6 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 48 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v10a_composed_diverse_local \
    --out_dir output/chain_v10a_composed_diverse_local \
    2>&1 | tee logs/chain_v10a_composed_diverse_local.log
