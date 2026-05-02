#!/usr/bin/env bash
# v14a-ap-mega-fullfix-gh200 (GH200 GPU 0) -- v14 HEADLINE candidate.
# AP + full v13 stack + v14 fixes (judge QK-LN, alpha_mem floor aux
# loss, InfoNCE contrastive, AP router_mem_bias anneal) + mega corpus.
#
# v14 thesis (the one-sentence version)
# -------------------------------------
# v13 proved the WRITER can specialise -- v13c2 hit evidence_lift
# +1.41 at step 400, D3-MC pair/self stayed at 0.004 all run.  But
# the ROUTER learned to reject memory (alpha_mem_mean ~ 0.001 on
# v13r @ step 10000) and the JUDGE re-collapsed to uniform (D2
# row_entropy / log(2K) = 0.988).  v14 attacks those two orthogonal
# failures directly while preserving every v13 win.
#
# The four v14 fixes, with citations
# ----------------------------------
#  (1) alpha_mem floor aux loss  -- MoE-style load-balance penalty
#      (Fedus et al. 2021 Switch Transformer; Wang et al. 2024
#      auxiliary-loss-free balancing).  Forces the depth router to
#      keep sampling memory so the writer / readout keep receiving
#      downstream gradient.  Target=0.05, weight=0.01.
#  (2) InfoNCE contrastive loss -- already implemented in
#      train_chain.py but unused in every v13 run.  Gives the writer
#      a DIRECT discriminative signal: for each batch element, its
#      own M_c should beat (B-1) negative M_c's on predicting its
#      own callback tokens.  weight=0.5 on callback-only, same
#      protocol as AutoCompressors (Chevalier et al. EMNLP 2023).
#  (3) Judge QK-LayerNorm        -- post-projection RMSNorm on Q and
#      K of MemoryBlock.judging.  Decouples attention-logit
#      magnitude from W_Q/W_K spectral norm (Zhai et al. 2023
#      "Preventing Attention Entropy Collapse"), letting the judge
#      softmax find sharper distributions.  Cheap, standard (used
#      in Qwen, Gemma, DeepSeek-V3).
#  (4) AP router_mem_bias anneal -- router.mem_bias is force-held at
#      +4 during the 500-step writer warmup then annealed to 0 over
#      200 steps.  Parallel to simple_gate's memory_gate force-open
#      that fixed SG gradient starvation; required for AP because
#      otherwise router.mem_bias sits at 0 the entire warmup and
#      the writer never sees LM gradient at realistic alpha_mem.
#
# Plus all v13 wins preserved
# ---------------------------
#    R (routing):    attention_parity (strongest routing-side prior;
#                    exp1 manuscript Table 2: AP beats SG by 1.6-
#                    3.8x on Delta_sh-m at matched compute)
#    S (symmetry):   memres_queries_init=orthogonal +
#                    memres_slot_positional  (v13 D3-MC: slots stay
#                    orthogonal pair/self=0.004 throughout training)
#    W (writer):     slot_attention (iter=3) with QK-LN on judge
#    O (objective):  writer_warmup 500 + anneal 200 + InfoNCE + floor
#    corpus:         mega (67k chains, 5 sources, curriculum-weighted)
#    update mode:    gated (per-slot sigmoid keep-gate, init g~0.27;
#                    v13r D3-MC 0.028 delta_step showed competitive
#                    mode under-wrote on natural prose)
#
# Step budget: 8000
#   phase 1 (0-500):     writer warmup, backbone FROZEN, mem_bias=4,
#                        InfoNCE on, alpha_floor on
#   phase 2 (500-700):   anneal mem_bias 4->0, backbone unfreezes,
#                        all four aux losses active throughout
#   phase 3 (700-8000):  joint training on mega with curriculum,
#                        alpha_floor keeps router recruiting memory,
#                        InfoNCE keeps writer content-specific,
#                        judge QK-LN keeps the softmax decisive
#
# Decision triggers (8000-step schedule, mega corpus)
#   step 700 (anneal end):  grad_norm (clipped) < 2, alpha_mem_mean
#                           > 0.04 (above floor), evidence_lift > 0
#   step 2000:              alpha_mem_mean > 0.05 (sustained recruit),
#                           judge row_entropy/log(2K) < 0.92 (QK-LN
#                           working), InfoNCE gap > +0.05
#   step 4000:              pa_cb_dsh > +0.030, evidence_lift > +0.10
#   step 8000 (final):      pa_cb_dsh > +0.080 (paper-publishable),
#                           alpha_mem_mean > 0.05 sustained last 2k
#                           steps, D2-JUDGE norm < 0.92 sustained
#   KILL @ 4000 if pa_cb_dsh < 0.010 for 3 consecutive evals (the
#     four fixes together can't rescue the mega joint regime; v14b
#     should then test frozen-backbone + v14 fixes as fallback).
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
    --memres_judge_qk_layernorm \
    --writer_warmup_steps 500 \
    --writer_warmup_router_bias 4.0 \
    --writer_warmup_anneal_steps 200 \
    --alpha_mem_floor_aux_weight 0.01 \
    --alpha_mem_floor_target 0.05 \
    --contrastive_infonce_weight 0.5 \
    --contrastive_infonce_temperature 1.0 \
    --contrastive_infonce_callback_only \
    --contrastive_infonce_initial_weight 0.0 \
    --contrastive_infonce_warmup_steps 500 \
    --train_chains paper_artifacts/chains/v11_mega_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 4 \
    --carry_state \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 8000 \
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
    --run_name chain_v14a_ap_mega_fullfix_gh200 \
    --out_dir output/chain_v14a_ap_mega_fullfix_gh200
