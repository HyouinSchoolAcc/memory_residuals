#!/usr/bin/env bash
# After overwrite trainings finish: eval them, train K-sweep cells,
# run LoRA sweep, synthd5 stage ladder, then rebuild figures+docx.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/patent_experiments

wait_for_final() {
  local run="$1"
  echo "[wait] $run"
  while [ ! -f "runs/$run/final/config.json" ]; do
    sleep 60
  done
  echo "[ready] $run"
}

eval_ckpt() {
  local path="$1" out="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python tools/eval_callback.py \
    --model_path "$path" \
    --corpora paper_artifacts/chains/lme_val_s512_evpos.pt \
    --names lme_val \
    --output "$out"
}

echo "=== waiting for overwrite trains ==="
wait_for_final chain_patent_overwrite_seed1_0p6b
wait_for_final chain_patent_overwrite_seed2_0p6b

echo "=== eval overwrite seeds ==="
eval_ckpt runs/chain_patent_overwrite_seed1_0p6b/final \
  results/patent_experiments/eval_trained_overwrite_seed1.json 0
eval_ckpt runs/chain_patent_overwrite_seed2_0p6b/final \
  results/patent_experiments/eval_trained_overwrite_seed2.json 1

echo "=== train K=32 (GPU0) and K=512 (GPU1) in parallel ==="
CUDA_VISIBLE_DEVICES=0 bash scripts/patent_train_cell.sh \
  chain_patent_k32_seed1_0p6b 1 \
  --pretrained Qwen/Qwen3-0.6B --memres_num_vectors 32 \
  --memres_extraction_depth 4 --memres_num_blocks 8 \
  --memres_update_mode gated &
PID_K32=$!
CUDA_VISIBLE_DEVICES=1 bash scripts/patent_train_cell.sh \
  chain_patent_k512_seed1_0p6b 1 \
  --pretrained Qwen/Qwen3-0.6B --memres_num_vectors 512 \
  --memres_extraction_depth 4 --memres_num_blocks 8 \
  --memres_update_mode gated &
PID_K512=$!
wait $PID_K32
wait $PID_K512

echo "=== eval K cells ==="
eval_ckpt runs/chain_patent_k32_seed1_0p6b/final \
  results/patent_experiments/eval_trained_k32_seed1.json 0
eval_ckpt runs/chain_patent_k512_seed1_0p6b/final \
  results/patent_experiments/eval_trained_k512_seed1.json 1

# headline K=128 already evaluated; copy reference
cp results/eval_v25_seed_pack_evpos/v27b_no_probe_seed3_final_lme_val_evpos.json \
  results/patent_experiments/eval_trained_k128_seed3.json

echo "=== LoRA joint/seq sweep (GPU0) + synthd5 stage ladder (GPU1) ==="
CUDA_VISIBLE_DEVICES=0 bash -c '
  set -e
  python tools/patent_lora_forgetting.py \
    --n_chains 8 --steps_per_session 8 --mode joint --lr 1e-4 \
    --general_source fineweb \
    --output results/patent_experiments/lora_joint_lr1e-4_0p6b.json
  python tools/patent_lora_forgetting.py \
    --n_chains 8 --steps_per_session 8 --mode joint --lr 2e-5 \
    --general_source fineweb \
    --output results/patent_experiments/lora_joint_lr2e-5_0p6b.json
  python tools/patent_lora_forgetting.py \
    --n_chains 8 --steps_per_session 8 --mode sequential --lr 2e-5 \
    --general_source fineweb \
    --output results/patent_experiments/lora_seq_lr2e-5_0p6b.json
' &
PID_LORA=$!
CUDA_VISIBLE_DEVICES=1 python tools/patent_stage_ablation.py \
  --model_path runs/chain_v32a_sparse_synthd5_seed1_0p6b_frozen_local/final \
  --corpus paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
  --n_chains_max 100 \
  --output results/patent_experiments/stage_ablation_synthd5_v32a.json &
PID_SYNTH=$!
wait $PID_LORA
wait $PID_SYNTH

echo "=== rebuild figures + patent doc ==="
python tools/patent_make_figures.py
python tools/patent_write_docx.py \
  --src "/tmp/patent_doc/一种基于注意力读出与竞争式更新的语言模型外置记忆方法（发明，初稿）1 (1).docx" \
  --out "一种基于注意力读出与竞争式更新的语言模型外置记忆方法（发明，含实验验证）.docx"

echo POST_TRAIN_QUEUE_DONE
