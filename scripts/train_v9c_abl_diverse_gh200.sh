#!/usr/bin/env bash
# v9c ABLATION — competition curriculum on diverse corpus (LME+MSC+PG-19+TV)
#
# Question: does the v9 win survive when the model also has to do generic
# language modelling on diverse data, or was LME-only the key?
# v8c failed on diverse corpus + heavy regularization (couldn't tell whether
# diversity itself or the regularization stack was the killer); v9c isolates
# diversity by keeping every other v9 knob fixed. Source weights tilt the
# mix toward the callback-bearing corpora (longmemeval, MSC) so we keep
# enough competition-curriculum signal even though chain windows from
# pg19/tv contain no callback structure.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode simple_gate \
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
    --phase_aligned_eval_n_chains 48 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v9c_abl_diverse_gh200 \
    --out_dir output/chain_v9c_abl_diverse_gh200 \
    2>&1 | tee logs/chain_v9c_abl_diverse_gh200.log
