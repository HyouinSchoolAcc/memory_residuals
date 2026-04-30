#!/usr/bin/env bash
# Re-run the phase-aligned eval on v10b best (step 200) at n=256 chains
# x 5 seeds, to test whether the reported PA CB Δ_sh-m=+0.011 is real
# signal or within eval noise. See README open-problem #2.
set -eu
cd "$(dirname "$0")/.."
mkdir -p paper_artifacts/eval
export CUDA_VISIBLE_DEVICES=0   # co-locate with v10a; 0.6B fits trivially
CKPT=${CKPT:-output/chain_v10b_attnparity_pm4_diverse_local/best}
OUT=${OUT:-paper_artifacts/eval/v10b_best_pa_n256_s5.json}

exec python -u train_chain.py \
    --pretrained "$CKPT" \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init -4 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/v6_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 1 \
    --eval_n_chains 0 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 256 \
    --diag_routing_n_chains 16 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_only \
    --eval_seeds 5 \
    --eval_out "$OUT" \
    --run_name re_eval_v10b_best_n256 \
    --out_dir output/re_eval_v10b_best_n256 \
    2>&1 | tee logs/re_eval_v10b_best_n256.log
