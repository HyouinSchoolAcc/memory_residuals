#!/usr/bin/env bash
# v18a_f3_codes_0p6b_frozen_local (LOCAL GPU 0)
# F3 / ReadoutProbeHead read-side extractive supervision.
#
# Diagnosis recap (from v17 §5 + v18 read-side localizer):
#   * §5 (tools/eval_ttt_mc.py): 6/6 NEG with read-side frozen and
#     only M_c TTT-able.  v14k/v15a/v15e at 0.6B and 1.7B, D4v2 and
#     D5.  Falsifies "trained read-side decodes any informative M_c".
#   * §5b (tools/eval_ttt_mc_readout.py, this run, 2026-05-03): with
#     MemoryReadout's W_Q/W_K/W_V also TTT-able, ev_loss drops 0.5
#     -> 0.15 (huge over-fit on evidence) but callback CE worsens
#     ~0.9 nats vs floor.  TTT cannot generalise across the
#     evidence -> callback gap on a single chain.
#   * v17a/b/e (F2 / WriterProbeHead): KILLED at launch -- the writer
#     probe gradient bypasses MemoryReadout entirely (own Q/K/V), so
#     it cannot fix a content-blind read-side.
#
# F3 fix (this cell):
#   --readout_probe_enabled with --readout_probe_loss_weight 1.0
#   adds a separate gradient channel that supervises m_t -- the
#   ACTUAL MemoryReadout output at the callback session's first
#   answer-token position -- to be probe-decodable to the chain's
#   first answer-token id.  Because m_t = MemoryReadout(embed(X),
#   M_c), the probe-loss gradient flows through MemoryReadout's
#   W_Q/W_K/W_V (and onward to M_c, judge, extract).  This is the
#   missing read-side gradient channel that v17 / F2 cannot install
#   from its writer-side bypass.
#
# UPDATED 2026-05-03 16:14 UTC-5 -- v18b first eval revealed the next
# bottleneck.  At step 200 of v18b (F2+F3): both wprobe and rprobe
# successfully drop from ~12 -> ~6 (writer puts chain content into
# M_c, readout puts chain content into m_t), but
# evidence_lift = -0.003 with alpha_mem_mean = 0.0087.  The depth
# router is keeping the memory channel almost closed -- chain-
# specific content in m_t never reaches the LM head.  Classic
# chicken-and-egg: router won't open until LM benefits; LM can't
# benefit when router is closed.
#
# v17a's prescription killed alpha_mem_floor because under a
# content-blind writer the floor was gameable (router spread alpha
# thinly to satisfy the constraint).  With v18's read-side probe
# loss now actively pressuring m_t to be chain-specific, the floor
# becomes a *supporting* mechanism that lets the chain-specific
# content actually reach the LM head -- not a gameable one.
#
# v18a now tests F3 + alpha_mem_floor.  v18b stays as the
# "probe-only, no router pressure" comparison cell.  A clean A/B
# on whether the router-opening intervention is necessary.
#
# Single-variable test rationale (read-side intervention isolated):
#   v18a runs F3 (read-side probe) + alpha_mem_floor (open router).
#   v18b runs F2 + F3 (no alpha floor) -- the "probe-only" comparison.
#   Comparing v18a vs v18b at step 1500 isolates whether the alpha
#   floor's router-opening is the load-bearing piece on top of
#   the probe gradient channel.
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v18a_f3_codes_0p6b_frozen_local"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 nohup python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 --router_mem_bias_init 0 \
    --memres_update_mode gated --memres_extract_source hidden_14 \
    --memres_extract_input_norm \
    --memres_gate_init 0.0 --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal --memres_slot_positional \
    --memres_judge_qk_layernorm \
    --writer_warmup_steps 0 --writer_warmup_router_bias 0.0 --writer_warmup_anneal_steps 0 \
    --freeze_backbone \
    --readout_probe_enabled \
    --readout_probe_loss_weight 1.0 \
    --readout_probe_warmup_steps 200 \
    --alpha_mem_floor_aux_weight 0.01 \
    --alpha_mem_floor_target 0.05 \
    --train_chains paper_artifacts/chains/synthd5_random_codes_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --window_k 3 --batch_size 4 --grad_accum 2 \
    --lr 1e-4 --lr_backbone 0 --steps 1500 --warmup 200 --max_norm 1.0 \
    --memory_dropout 0.10 --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 5.0 --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 --curriculum_competition_bias 0.0 \
    --burn_in_max 0 --mask_padding_loss --score_tail_frac 1.0 \
    --mask_evidence_session_loss \
    --kill_on_memory_collapse --kill_on_memory_collapse_min_step 200 \
    --eval_every 100 --save_every 500 \
    --eval_n_chains 24 --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups --diagnose_memory_dynamics \
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric evidence_lift \
    --run_name chain_v18a_f3_codes_0p6b_frozen_local \
    --out_dir "$OUT_DIR" \
    > "$LOG_DIR/chain_v18a_f3_codes_0p6b_frozen_local.log" 2>&1 &
echo "v18a launched pid=$!"
