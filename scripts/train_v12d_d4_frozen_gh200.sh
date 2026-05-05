#!/usr/bin/env bash
# v12d-D4-FROZEN (GH200) -- slot_attention writer * FROZEN BACKBONE
#                                                 * D4 SYNTH CORPUS
#                                                 * MODEST BUDGET
#
# Frozen arm of the v12d frozen-vs-trained comparison study, retargeted
# from the (cancelled) chinchilla-mega cell to the D4 synthetic
# persona-callback corpus.
#
# Why D4 instead of mega + chinchilla
# -----------------------------------
# v12a's step-800 D2 reading (runs.md "v12a grow-then-decay reproduces
# the v11g failure mode" §) showed the slot-attention writer settles
# into the same symmetric uniform fixed point as the original judge
# even on a clean synthetic corpus -- row_entropy_norm 0.999, eff_rank
# 1.01, keep_mean 0.500.  Throwing 25k Chinchilla-budgeted steps on
# a 10x-bigger corpus at a writer that hits this fixed point in 800
# steps on a 5000-chain synthetic corpus is ~50x compute waste.
#
# D4 is the diagnostic gold standard.  Each 9-session chain has:
#   session 0:  "User: My favorite tool in the world is scissors.
#                Assistant: Got it, your favorite tool is scissors.
#                I'll remember that."
#   sessions 1-7: 7 unrelated distractor turns
#   session 8:  "User: Quick question, what was my favorite tool again?
#                Assistant: Your favorite tool is scissors."
#
# 256-item closed-set callback (tool_scissors, instrument_trombone, ...);
# log(256) = 5.55 nats irreducible callback CE under no-memory baseline.
# If a writer cannot move pa_cb_ce_mem toward the floor on D4, no
# amount of mega + chinchilla scaling will recover it -- the writer is
# architecturally broken.
#
# Single-axis diff vs v12d-D4-trained:
#   frozen:  --freeze_backbone     --lr_backbone 0
#   trained: (no freeze flag)      --lr_backbone 2e-5
# Everything else identical (memres lr 5e-5, 4000 steps, k=3, no
# carry_state, no burn_in -- matches v12a/v11g recipe exactly).
#
# Decision triggers (sharp, mirror trained arm exactly)
# -----------------------------------------------------
#   step  200 : pa_cb_dnm > +0.5  AND  D2 row_entropy_norm < 0.95
#   step 1000 : pa_cb_dnm > +2.0  AND  pa_cb_evidence_lift > +1.0
#                                AND  D2 keep_mean != 0.500 +/- 0.05
#                                AND  D2 eff_rank > 4
#   step 2000 : pa_cb_ce_mem < 2.0 nats (~35% of log(256)=5.55)
#   step 4000 : pa_cb_ce_mem < 1.0 nats (~18% of log(256)) -- writer
#                                works in principle on the cleanest
#                                possible corpus
#   KILL @ step 1000 with D2 row_entropy_norm > 0.98:
#         slot attention also stays uniform with frozen backbone too;
#         confirms (with the trained arm) that the symmetric uniform
#         fixed point is not breakable by the slot-attention writer
#         alone; v13 must add per-slot untied GRU or slot-bias.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
WRITER_KIND="${1:-slot_attention}"
case "$WRITER_KIND" in
    slot_attention|slot_attention_full) ;;
    *) echo "writer_kind must be slot_attention or slot_attention_full" >&2; exit 2 ;;
esac
# GH200 layout is flat (train_chain.py at root); local has src/.
if   [ -f train_chain.py ];     then TRAIN=train_chain.py
elif [ -f src/train_chain.py ]; then TRAIN=src/train_chain.py
else echo "train_chain.py not found in . or src/" >&2; exit 2
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
    --train_chains paper_artifacts/chains/synthd4_persona_callback_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd4_persona_callback_val_s512.pt \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 0 \
    --steps 4000 \
    --warmup 200 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.0 \
    --curriculum_competition_bias 1.0 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups \
    --diagnose_memory_dynamics \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name "chain_v12d_${WRITER_KIND}_d4_frozen_gh200" \
    --out_dir "output/chain_v12d_${WRITER_KIND}_d4_frozen_gh200"
