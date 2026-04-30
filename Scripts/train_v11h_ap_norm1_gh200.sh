#!/usr/bin/env bash
# v11h (GH200, 0.6B) -- attention_parity +4/0 with readout_norm_init=1.0
#
# A/B against v11g_ap_baseline. The ONLY knob change is dropping the P2
# magnitude fix (out_norm_init returns to 1.0 -> ||m^t||/||embed|| ~ 70-100
# at step 0). Tests whether attention_parity's depth softmax handles
# magnitude on its own (it normalizes scores, but the values still get
# weighted-summed at full magnitude). If alpha_mem opens but ||m^t||
# is still 70+, the readout will dominate the residual stream and the
# backbone will route around it (alpha_mem -> 0). If, on the other hand,
# the softmax does effectively self-regulate and v11h matches v11g,
# we learn that P2 is only needed for simple_gate routing.
#
# Decision triggers:
#   step 200  : ||m^t||/||embed|| reported (just observe magnitude)
#   step 500  : if alpha_mem_max < 5e-4 -> KILL (readout dominance
#               killed routing, P2 is necessary even for AP)
#   step 1000 : compare to v11g pa_cb_dnm to quantify the gap
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 1.0 \
    --train_chains paper_artifacts/chains/v11_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
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
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v11h_ap_norm1_gh200 \
    --out_dir output/chain_v11h_ap_norm1_gh200
