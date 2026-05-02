#!/usr/bin/env bash
# v13r-ap-mega-longwarmup-gh200 (GH200 GPU 0) -- HEADLINE candidate.
# AP + full v13 stack + 3000-step writer_warmup + 1000-step anneal +
# mega corpus + 16000 total steps.
#
# Why this run, in one sentence
# -----------------------------
# v13c2 showed the full v13 AP stack produces +1.4085 evidence_lift at
# step 400 of a 500-step warmup (6.4x any prior project measurement);
# then the anneal+unfreeze at step 500 produced a gradient shock
# (grad_norm 6.5e8 pre-clip, evidence_lift collapsed to -0.56 at
# step 600).  v13r scales the warmup 6x (3000 steps) so the writer
# reaches a stable content-specific basin BEFORE the backbone
# unfreezes, and scales the anneal 5x (1000 steps) to smooth the
# phase transition.  On the mega corpus (899 MB, 67k chains, 5
# sources) so warmup isn't on one-source D4.
#
# Step budget breakdown
#   phase 1 (0-3000):        writer_warmup, backbone FROZEN, mem_bias
#                            force-held at 4.0.  Writer sees 49M
#                            tokens (~1.7 tok/writer-param) on a
#                            stable backbone.  Target: evidence_lift
#                            > +1.5 by step 3000 on mega eval.
#   phase 2a (3000-4000):    anneal mem_bias 4.0 -> 0.0 over 1000
#                            steps, backbone unfreezes.  Target:
#                            grad_norm (pre-clip) stays < 10^5,
#                            evidence_lift drops at most 50% during
#                            transition (vs v13c2's -140% collapse).
#   phase 2b (4000-16000):   joint training on full memres stack +
#                            backbone.  Target: evidence_lift recovers
#                            to >= phase 1 peak by step 8000, climbs
#                            past it by step 16000.
#
# Lever stack (ALL v13 levers active; the _build_model bugfix of
# 2026-05-01 makes this the first AP run that actually gets them):
#   R (routing):    attention_parity (Eq. 9 depth softmax, spec-strict;
#                   strongest routing-side prior per exp1 manuscript
#                   Table 2 where AP beat SG by 1.6-3.8x on pair)
#   O (objective):  writer_warmup 3000 + anneal 1000 = 4000 steps of
#                   gradient-amplified extraction training
#   S (symmetry):   memres_queries_init=orthogonal + slot_positional
#   W (writer):     slot_attention (iter=3)
#   (joint):        backbone unfreezes at step 3000; lr_backbone=2e-5
#
# Decision triggers (16000-step schedule, mega corpus):
#   step 3000 (warmup end):    evidence_lift > +1.5, alpha_mem_max >
#                              0.2, judge row_entropy / log(2K) < 0.95
#   step 4000 (anneal end):    grad_norm (clipped) < 2, alpha_mem_max
#                              > 0.1, evidence_lift > +0.5
#   step 6000:                 evidence_lift > +0.8 (recovered past
#                              phase-2 shock)
#   step 10000:                pa_cb_dsh > +0.050 (the paper headline
#                              threshold)
#   step 16000 (final):        pa_cb_dsh > +0.100, evidence_lift >
#                              +1.0 sustained over last 2k steps
#   KILL @ 4000 if evidence_lift < 0 for 3 consecutive evals (phase
#     transition destroyed the content-specific memory permanently).
#
# If v13r clears step-10000 trigger, v13p-mega-SG becomes the
# SG-ablation and v13r is promoted to the paper headline.
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
    --writer_warmup_steps 3000 \
    --writer_warmup_router_bias 4.0 \
    --writer_warmup_anneal_steps 1000 \
    --train_chains paper_artifacts/chains/v11_mega_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 4 \
    --carry_state \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 16000 \
    --warmup 500 \
    --max_norm 1.0 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 \
    --curriculum_competition_bias 1.0 \
    --burn_in_max 12 \
    --burn_in_resample \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 500 \
    --save_every 1000 \
    --eval_n_chains 48 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups \
    --diagnose_memory_dynamics \
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v13r_ap_mega_longwarmup_gh200 \
    --out_dir output/chain_v13r_ap_mega_longwarmup_gh200
