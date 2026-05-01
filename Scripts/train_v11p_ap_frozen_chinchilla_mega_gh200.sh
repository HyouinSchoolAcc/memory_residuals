#!/usr/bin/env bash
# v11p (GH200, 0.6B) -- attention_parity +4/0
#                     * FROZEN BACKBONE
#                     * CHINCHILLA-BUDGETED
#                     * MEGA CORPUS
# v11 HEADLINE replacement for the killed 4B mega cell.
#
# User directives 2026-04-30 ~22:30 UTC:
#   "remove the 4B model"
#   "If data is issue, make more data"
#   "Think of a way we can apply the chincilla budget to our system"
#   "I know from a fact that attention parity works better"
#
# Why this replaces v11_4b_mega
# -----------------------------
# v11_4b_mega died at startup with CUDA OOM in the watchdog launch
# race; even after the gpu_is_busy patch the 4B preset is unstable
# at 52 GB peak on a 94 GB GH200 once activations + grad checkpoints
# + AdamW state for the 4B backbone all land at once. Backing off to
# 4B was itself a backoff from 8B for the same reason.
#
# The 4B cell's stated thesis -- deep extraction (L_E=10) + block
# AttnRes parity + 100x data diversity -- assumed full-backbone
# training. The user-directed correction (frozen backbone,
# attention_parity decided) reframes the headline question: does the
# from-scratch memres subsystem learn to plateau when given (a) a
# clean, non-co-evolving backbone target, (b) a Chinchilla-scale
# token budget, and (c) the largest evidence-aware diverse corpus we
# have on disk?
#
# Configuration
# -------------
# Backbone Qwen3-0.6B (28 layers), pretrained, FROZEN.
# Memres ~9.7 M params, attention_parity routing, +4/0 bias.
# Corpus v11_mega_train_s512.pt (67 745 chains, 450 with evidence
# annotations from LongMemEval). 10x larger than v11_lme_msc.
# Token budget 25 000 steps x bs 4 x ga 2 x window_k 4 x 512 tok
#              = 410 M tokens (gradient-flowing through memres)
#              = 2.1x Chinchilla for 9.7 M params.
# LR bumped to 1e-4 for memres (v11g uses 5e-5). With backbone frozen
# there is no co-evolution penalty for moving memres faster; still
# 2x more conservative than the phase1 default of 2e-4.
# Warmup 1500 steps (6% of training).  Tests the larger-warmup
# hypothesis as a side benefit of the Chinchilla scaling.
# carry_state ON; burn_in_max 12 + burn_in_resample close the
# train/eval recurrent-depth gap (P5 in the v10 audit).
#
# Decision triggers
# -----------------
#   step 1500  : alpha_mem_max > 0 AND ||m^t||/||embed|| in [0.3, 50]
#   step 5000  : alpha_mem_max > 1e-2 AND pa_cb_dnm > +0.05
#                                       (matches v9c step 1400)
#   step 12500 : pa_cb_dsh > +0.020 AND alpha_mem_max NOT decaying
#   step 25000 : standard Delta_sh-m > +0.010
#   KILL: step 5000 with alpha_mem_max < 1e-3 -- frozen backbone +
#         Chinchilla budget + mega corpus all together cannot open
#         the channel; falsifies the v11 architectural recipe and
#         requires a routing intervention before further scaling.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --freeze_backbone \
    --train_chains paper_artifacts/chains/v11_mega_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0, "ultrachat": 2.0, "pippa": 2.0, "soda": 1.5, "lmsys": 2.0, "synthdlg": 1.5}' \
    --window_k 4 \
    --carry_state \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 1e-4 \
    --lr_backbone 0 \
    --steps 25000 \
    --warmup 1500 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.0 \
    --curriculum_competition_bias 1.0 \
    --burn_in_max 12 \
    --burn_in_resample \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 500 \
    --save_every 1500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v11p_ap_frozen_chinchilla_mega_gh200 \
    --out_dir output/chain_v11p_ap_frozen_chinchilla_mega_gh200
