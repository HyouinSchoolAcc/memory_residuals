#!/usr/bin/env bash
# v8c DIVERSE-CORPUS SIMPLE_GATE + readout RMSNorm + multi-axis regularisation
#
# Hypothesis under test (motivated by the v8a step-1000 catastrophic
# overfitting result):
#
#   Curriculum mixing (v8b) plus architecture norm fix (v8a/b) is
#   necessary but not sufficient when training on 450 LME chains
#   alone -- the readout finds a degenerate solution that maximises
#   callback-token specificity (good) by producing extreme,
#   miscalibrated injections (bad) that on aggregate destroy
#   non-callback prediction quality.
#
#   The fundamental fix is data diversity: train on a corpus where
#   most chains DON'T have callback supervision, so generic LM
#   gradient pressure penalises the readout for over-injecting on
#   tokens where memory shouldn't help. The 7% LME slice still
#   provides callback-specific gradient.
#
# Diff vs v8b (the previous "mixed-bias only" cell):
#
#   1. corpus  : v6_lme_msc_train_s512.pt (6378 chains, 14x more)
#                vs lme_train_s512.pt (450 chains, callback-only)
#   2. corpus weights : LME=4 MSC=3 pg19=1 tv=4 realtalk=1
#                       (effective ~13% LME callback signal,
#                        87% generic LM regularisation)
#   3. callback_loss_weight : 3.0 (was 10.0)
#                  Less concentrated gradient pressure -- the
#                  readout can't win the loss by maximising one
#                  small token mask
#   4. lr (memres) : 1e-4 (was 2e-4)
#                  Slower memres learning lets the backbone catch
#                  up; readout is the hardest module to train
#   5. memory_dropout : 0.1 (was 0.0)
#                  Randomly drops the m^t injection per training
#                  window; classical regularisation that prevents
#                  the readout from learning a single-injection
#                  shortcut
#   6. context_dropout : 0.05 (was 0.0)
#                  Drops C_t (extracted source) randomly so the
#                  writer can't rely on every token being available
#
# Decision triggers (sharp):
#   step 500: pa_cb_dsh > +0.005 AND standard delta_nm_minus_mem > -0.005
#   step 1000: pa_cb_dsh > +0.020 AND standard delta_shuffle_minus_mem > 0
#   step 2000: standard delta_shuffle_minus_mem > +0.010 -> ship as v9 baseline
#   anywhere: train loss < 0.6 -> kill (overfitting detected)
#
# Memory budget: window_k=8 with simple_gate, bs=2 grad_accum=4 fits
# in ~50 GiB on H100 NVL with gradient checkpointing.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode simple_gate \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/v6_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 8 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 1e-4 \
    --lr_backbone 2e-5 \
    --steps 4000 \
    --warmup 200 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.5 \
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
    --run_name chain_v8c_diverse_simplegate_rmsnorm \
    --out_dir output/chain_v8c_diverse_simplegate_rmsnorm \
    2>&1 | tee logs/chain_v8c_diverse_simplegate_rmsnorm.log
