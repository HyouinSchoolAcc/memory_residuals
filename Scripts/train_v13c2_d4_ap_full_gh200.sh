#!/usr/bin/env bash
# v13c2-d4-ap-full-gh200 (GH200 GPU 0) -- AP + FULL v13 stack on D4.
#
# Why this run exists
# -------------------
# The 2026-04-26 pair-recipe manuscript (exp1_pair_recipe/manuscript.tex
# Table 2) showed attention_parity (soft +-4 bias init) beating
# simple_gate by 1.6x-3.8x on Δ_sh-m at every step of the matched-seed
# head-to-head on PG-19 pairs, with AP @ step 2000 (+0.0272) already
# surpassing SG @ step 5200 (+0.0249).  That is the strongest
# routing-side prior we have from the paper line-of-evidence.
#
# On the chain trainer the same AP routing has consistently collapsed
# (v5 softparity, v7 softerbias + v3bias, v8/9/11 every AP cell): the
# router closes early (alpha_mem ~ 4e-4), writer gradient attenuates
# through alpha_mem, writer stays random, router sees noisy m^t, router
# closes harder.  COMPREHENSIVE.md §v7 diagnosed this as "causally
# equivalent to no memory regardless of M_c."
#
# v13c was supposed to be the first test of the full v13 stack against
# that collapse cycle.  But a _build_model bug (fixed 2026-05-01 ~22:45
# UTC-5) silently dropped CLI overrides for memres_writer_kind,
# memres_slot_positional, memres_num_vectors, etc.  v13c actually ran as
# "AP + writer_warmup + orthogonal init only" (no slot_attention, no
# slot_positional, L_E=0 instead of 4) -- NOT the v13 stack we thought
# was being tested.  Result: +0.22 peak at step 600 during anneal, then
# standard collapse to +0.004 by step 4000.
#
# v13c2 re-runs v13c with the SAME intent (AP + full v13 stack) now
# that the fix is in place, so every lever is actually active:
#   R (routing):    attention_parity (Eq. 9 depth softmax, spec-strict)
#   O (objective):  writer_warmup (500 steps phase 1, 200 anneal)
#                   -- force-opens mem_bias to 4.0 so the router sees
#                   alpha_mem ~ 0.5 during phase 1, giving the writer
#                   full LM gradient on a stable frozen backbone
#   S (symmetry):   memres_queries_init=orthogonal + slot_positional
#                   -- breaks the permutation-invariant fixed point
#                   at t=0 so the judge softmax can't stay uniform
#   W (writer):     slot_attention (iter=3) -- softmax-over-slots
#                   forces slot specialisation regardless of router
#   (unfreeze):     backbone unfreezes at step 500; no freeze-keep
#
# Headline hypothesis: if the pair-recipe AP > SG result transfers to
# the chain trainer once the collapse-cycle is broken by the O+S+W
# interventions, then v13c2 should HOLD (not collapse) the warmup
# peak past step 700.  Decision triggers below quantify "hold."
#
# Decision triggers (4000-step schedule):
#   step  500 (warmup end):  alpha_mem_max > 0.1 AND judge
#                            row_entropy / log(2K) visibly below 1.0
#                            (NOT = 0.999 as in v13c buggy)
#   step  700 (anneal end):  pa_cb_evidence_lift > +0.1 (vs v13c's
#                            +0.0037 at this point post-anneal)
#   step 1500:               pa_cb_dsh > +0.010 AND evidence_lift > +0.2
#   step 3000:               pa_cb_dsh > +0.030 AND evidence_lift > +0.5
#   step 4000 (final):       pa_cb_dsh > +0.050 AND evidence_lift > +0.8
#   KILL @ 1000 if:          alpha_mem_max < 0.01 (router re-closed)
#                            OR evidence_lift < 0 (chain hashing)
#
# If v13c2 CLEARS these triggers we promote AP to the headline paper
# run (v13p) -- currently specced as simple_gate -- and re-run v13b
# ablations in AP mode.  If v13c2 COLLAPSES like v13c, then the
# pair-recipe AP advantage is pair-specific and SG is the correct
# chain-trainer choice.
set -eu
cd "$(dirname "$0")"/..
mkdir -p output logs
[ -f "$HOME/venv/bin/activate" ] && . "$HOME/venv/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention \
    --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal \
    --memres_slot_positional \
    --writer_warmup_steps 500 \
    --writer_warmup_router_bias 4.0 \
    --writer_warmup_anneal_steps 200 \
    --train_chains paper_artifacts/chains/synthd4_persona_callback_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd4_persona_callback_val_s512.pt \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
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
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v13c2_d4_ap_full_gh200 \
    --out_dir output/chain_v13c2_d4_ap_full_gh200
