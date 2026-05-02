#!/usr/bin/env bash
# v13q-ap-frozen-curriculum-mega-gh200 (GH200 GPU 0) -- the four-lever
# ablation requested by the user (2026-05-01 ~19:28 UTC-5):
#   "make sure to include an ablation with a frozen backbone, an AP
#    style routing, a curriculum style training, on a large corpus
#    for 6000 steps."
#
# This fills an important gap in the v13 campaign.  Earlier cells:
#   v13c (RUNNING on GH200): AP + warmup + orth+pos + slot  but on D4.
#   v13p (HELD):             SG + warmup + orth+pos + slot + FROZEN +
#                            curriculum + MEGA corpus + 16k steps.
# Neither composes AP + FROZEN + curriculum + MEGA at 6k.  v13q tests
# whether the spec's Eq. 9 depth softmax (kept intact in AP) can
# actually produce a real memory signal when combined with:
#   - frozen backbone (forces all learning into the memres subsystem,
#     same regime where v12d_d4_frozen produced the only positive
#     evidence_lift in the project)
#   - curriculum (evidence_bias=1.0 + competition_bias=1.0 -- the same
#     two knobs that v13p uses; they reweight sessions by how much
#     chain-specific content they contain, so the writer gets a richer
#     gradient signal per step)
#   - v11_mega corpus (899 M tokens, LME + MSC + PG19 + TV + RealTalk,
#     the most diverse training mix available)
#   - 6000 steps (2x the D4 budget, ~3/8 of v13p's 16k) -- enough to
#     clear writer_warmup + burn-in + a reasonable post-warmup training
#     window, but cheap enough to inform whether v13p should run at
#     all in AP mode.
#
# Lever stack (all six v13 levers, with R = AP instead of SG):
#   O: writer_warmup (600 steps, bias=4) -- matches v13p/v13c ratio
#   S: memres_queries_init=orthogonal + slot_positional
#   R: attention_parity (m^t stays in the Eq. 9 depth softmax; spec-
#      strict -- this is the ablation vs v13p's simple_gate)
#   W: slot_attention writer (iter=3)
#   F: --freeze_backbone + --lr_backbone 0 +
#      --writer_warmup_keep_backbone_frozen
#   C: --curriculum_evidence_bias 1.0 --curriculum_competition_bias 1.0
#      --burn_in_max 12 --burn_in_resample
#
# Decision triggers (scaled for 6000 steps, AP+frozen regime):
#   step  600 (end-warmup): alpha_mem_max > 0.01 (router opened), and
#                            judge row-entropy / log(2K) dropping from
#                            1.0 toward 0.8 (D2 fix).
#   step 1000 (post-anneal): pa_cb_evidence_lift > +0.1 AND
#                            alpha_mem_max stays > 0.05.
#   step 2000:               pa_cb_dsh > +0.010 AND evidence_lift > +0.3.
#   step 4000:               pa_cb_dsh > +0.030 AND evidence_lift > +0.8.
#   step 6000 (final):       pa_cb_dsh > +0.050 AND evidence_lift > +1.0.
#
#   KILL:
#     step 1000 with alpha_mem_max < 0.005  (router re-closed despite
#                                            frozen backbone -- then
#                                            AP itself is unsalvageable).
#     step 2000 with evidence_lift < 0      (memory is chain-identity
#                                            hashing, not content, so
#                                            the stack doesn't buy us
#                                            the spec back).
#
# LR: 1e-4 (same as v13p, the "frozen-backbone writer boost" -- the
# memres subsystem is the only thing training so it can absorb a 2x
# learning rate vs the joint regime).
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
    --writer_warmup_steps 600 \
    --writer_warmup_router_bias 4.0 \
    --writer_warmup_anneal_steps 300 \
    --writer_warmup_keep_backbone_frozen \
    --freeze_backbone \
    --train_chains paper_artifacts/chains/v11_mega_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 4 \
    --carry_state \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 1e-4 \
    --lr_backbone 0 \
    --steps 6000 \
    --warmup 300 \
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
    --eval_every 400 \
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
    --run_name chain_v13q_ap_frozen_curriculum_mega_gh200 \
    --out_dir output/chain_v13q_ap_frozen_curriculum_mega_gh200
