#!/usr/bin/env bash
# v12d-headline-FROZEN (GH200, 0.6B) -- slot_attention writer
#                                     * FROZEN BACKBONE
#                                     * CHINCHILLA-BUDGETED
#                                     * MEGA CORPUS
# Frozen arm of the v12d frozen-vs-trained comparison study.  Pairs
# with train_v12d_headline_trained_chinchilla_mega_gh200.sh (the only
# axis difference is --freeze_backbone / --lr_backbone).
#
# v12 HEADLINE replacement for v11p.  Combines:
#
#   (1) the v11l/v11p "frozen backbone removes co-evolution" finding
#       (no backbone gradient flow to keep block summaries stationary
#       in the attention_parity depth softmax)
#   (2) the v11p Chinchilla scaling on token budget
#   (3) the v11p mega corpus (67.7k chains, 450 with evidence
#       annotations, 10x v11_lme_msc)
#   (4) the v12a/b architectural finding: writer subsystem replaced by
#       Slot-Attention (softmax over slots, GRU keep/write per slot).
#       --memres_writer_kind defaults to slot_attention; pass
#       slot_attention_full as $1 to use both stages.
#
# LR choice: 5e-5 memres / 0 backbone (v11g-matched memres lr; v11p
# bumped to 1e-4 because frozen-backbone removed the co-evolution
# penalty -- but the *trained* arm of this study would not tolerate
# 1e-4, so we hold the memres lr equal across arms for a clean
# 1-axis comparison.  Costs ~2x more steps to peak vs aggressive lr,
# but Chinchilla budget covers it.)
#
# Why this is the headline
# ------------------------
# The v11 audit closed with three orthogonal findings:
#   * structural -- writer is decision-less (D2, eff_rank 1.02)
#   * gradient  -- writer signal swamped by backbone co-evolution
#                  (v11l alpha_mem_max trajectory)
#   * data      -- LME alone is too sparse / chain-confounded
# v12d composes the architectural fix (slot attention) with the
# gradient and data fixes (frozen backbone, Chinchilla budget, mega
# corpus).  If this cell still does not move pa_cb_dsh significantly
# above v11g's +0.0024 ceiling, the depth-wise residual stream itself
# is the bottleneck and the v13 plan must consider out-of-spec routing
# (separate memory cross-attention sublayer, m^t out of the depth
# softmax).
#
# Configuration -- copies of v11p with one architectural delta
# ------------------------------------------------------------
# Backbone Qwen3-0.6B (28 layers), pretrained, FROZEN.
# Memres ~9.8M params (slot_attention) or ~12M (slot_attention_full),
# attention_parity routing, +4/0 bias.
# Corpus v11_mega_train_s512.pt (67 745 chains, 450 with evidence
# annotations from LongMemEval).
# Token budget 25 000 steps x bs 4 x ga 2 x window_k 4 x 512 tok
#              = 410 M tokens (gradient-flowing through memres)
#              = 2.1x Chinchilla for ~10M memres params.
# LR 1e-4 (v11p baseline; backbone frozen so co-evolution penalty
#          is gone, can move memres faster).
# Warmup 1500 steps (6%).  carry_state ON; burn_in_max 12 +
# burn_in_resample close the train/eval recurrent-depth gap.
#
# Decision triggers (matches v11p for direct comparability)
# ---------------------------------------------------------
#   step 1500  : alpha_mem_max > 0 AND ||m^t||/||embed|| in [0.3, 50]
#   step 5000  : alpha_mem_max > 1e-2 AND pa_cb_dnm > +0.05
#                                       (matches v9c step 1400)
#   step 12500 : pa_cb_dsh > +0.020 AND alpha_mem_max NOT decaying
#   step 25000 : standard Delta_sh-m > +0.010 (vs v11p step ?)
#   KILL: step 5000 with pa_cb_dsh < +0.005 -- architectural fix +
#         frozen backbone + Chinchilla budget + mega corpus all
#         together cannot move the headline metric; v13 plan must
#         consider routing intervention (m^t out of depth softmax).
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
WRITER_KIND="${1:-slot_attention}"
case "$WRITER_KIND" in
    slot_attention|slot_attention_full) ;;
    *) echo "writer_kind must be slot_attention or slot_attention_full" >&2; exit 2 ;;
esac
# GH200 layout is flat (train_chain.py at project root; no src/ dir).
# Local layout has src/. Detect.
if [ -f train_chain.py ]; then
    TRAIN=train_chain.py
elif [ -f src/train_chain.py ]; then
    TRAIN=src/train_chain.py
else
    echo "train_chain.py not found in . or src/" >&2; exit 2
fi
exec python -u "$TRAIN" \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind "$WRITER_KIND" \
    --memres_slot_attention_iters 3 \
    --freeze_backbone \
    --train_chains paper_artifacts/chains/v11_mega_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0, "ultrachat": 2.0, "pippa": 2.0, "soda": 1.5, "lmsys": 2.0, "synthdlg": 1.5}' \
    --window_k 4 \
    --carry_state \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
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
    --run_name "chain_v12d_${WRITER_KIND}_frozen_chinchilla_mega_gh200" \
    --out_dir "output/chain_v12d_${WRITER_KIND}_frozen_chinchilla_mega_gh200"
