#!/usr/bin/env bash
# Watchdog: wait for chain_v3_{softparity,attentionbase}_full to finish,
# run the post-train pipeline, then schedule K=64 and L_E=0 ablations.
#
# Run as:
#   nohup paper_tools/watchdog.sh > logs/watchdog.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

LOG_DIR=logs
WATCH_DIR=output

main_done() {
  # We treat a run as "done" when its step-6000 dir exists or its log
  # contains a "Saved checkpoint -> output/<run>/final" line.
  local run="$1"
  [[ -d "$WATCH_DIR/$run/step-6000" ]] && return 0
  grep -q "Saved checkpoint -> $WATCH_DIR/$run/final" "$LOG_DIR/$run.log" 2>/dev/null
}

# 1. Wait for the two main training runs to finish.
while true; do
  s1=0; s2=0
  main_done chain_v3_softparity_full && s1=1
  main_done chain_v3_attentionbase_full && s2=1
  echo "[watchdog] $(date) main done? softparity=$s1 attentionbase=$s2"
  if (( s1 == 1 && s2 == 1 )); then
    break
  fi
  sleep 300
done
echo "[watchdog] both main runs finished -- starting post-train pipeline"

# 2. Run the post-train pipeline (eval + callback + horizon + figures + paper).
bash paper_tools/post_train_pipeline.sh 2>&1 | tee logs/post_train_pipeline.log

# 3. Schedule additional ablations: K=64 on GPU 0, L_E=0 on GPU 1.
echo "[watchdog] $(date) starting K=64 ablation on GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup python -u train_chain.py \
  --pretrained Qwen/Qwen3-0.6B \
  --memres_mode attention_parity \
  --memres_num_vectors 64 --memres_extraction_depth 4 --memres_num_blocks 8 \
  --router_mem_bias_init -4.0 --router_recent_bias_init 4.0 \
  --window_k 8 --session_len 512 \
  --steps 6000 --batch_size 4 --grad_accum 4 \
  --warmup 200 --lr 3e-4 --lr_backbone 3e-5 \
  --memory_dropout 0.0 --context_dropout 0.0 \
  --neg_chain_weight 0.0 --burn_in_max 0 \
  --train_chains paper_artifacts/chains/stage1_train_s512_full.pt \
  --eval_chains paper_artifacts/chains/stage1_validation_s512.pt \
  --gradient_checkpointing --eval_every 200 --save_every 1000 --log_every 25 \
  --eval_n_chains 24 --eval_window 4 --save_best_metric composite \
  --out_dir output/chain_v3_K64 --seed 42 \
  > $LOG_DIR/chain_v3_K64.log 2>&1 &
PID_K64=$!
echo "[watchdog] K=64 PID=$PID_K64"

echo "[watchdog] $(date) starting L_E=0 ablation on GPU 1"
CUDA_VISIBLE_DEVICES=1 nohup python -u train_chain.py \
  --pretrained Qwen/Qwen3-0.6B \
  --memres_mode attention_parity \
  --memres_num_vectors 128 --memres_extraction_depth 0 --memres_num_blocks 8 \
  --router_mem_bias_init -4.0 --router_recent_bias_init 4.0 \
  --window_k 8 --session_len 512 \
  --steps 6000 --batch_size 4 --grad_accum 4 \
  --warmup 200 --lr 3e-4 --lr_backbone 3e-5 \
  --memory_dropout 0.0 --context_dropout 0.0 \
  --neg_chain_weight 0.0 --burn_in_max 0 \
  --train_chains paper_artifacts/chains/stage1_train_s512_full.pt \
  --eval_chains paper_artifacts/chains/stage1_validation_s512.pt \
  --gradient_checkpointing --eval_every 200 --save_every 1000 --log_every 25 \
  --eval_n_chains 24 --eval_window 4 --save_best_metric composite \
  --out_dir output/chain_v3_LE0 --seed 42 \
  > $LOG_DIR/chain_v3_LE0.log 2>&1 &
PID_LE0=$!
echo "[watchdog] L_E=0 PID=$PID_LE0"

# 4. Wait for the ablation runs to finish.
wait $PID_K64
wait $PID_LE0

echo "[watchdog] $(date) ablation runs finished -- final pipeline pass"

# 5. Final eval + paper rebuild including the ablation rows.
run_eval() {
  local model_dir="$1"
  local out_basename="$2"
  local cuda_id="$3"
  local out="paper_artifacts/eval/${out_basename}_eval.json"
  if [[ -f "$out" ]]; then return; fi
  CUDA_VISIBLE_DEVICES="$cuda_id" python paper_tools/eval_chain.py \
    --model_path "$model_dir" \
    --corpora paper_artifacts/chains/stage1_validation_s512.pt \
              paper_artifacts/chains/stage1_test_s512.pt \
              paper_artifacts/chains/locomo_s512.pt \
    --names pg19_val pg19_test locomo \
    --score_window 4 --oracle_window 4 \
    --output "$out" 2>&1 | tail -3
}
run_eval output/chain_v3_K64/best chain_v3_K64 0
run_eval output/chain_v3_LE0/best chain_v3_LE0 1

python paper_tools/horizon_analysis.py \
  --inputs paper_artifacts/eval/chain_v3_K64_eval.json \
           paper_artifacts/eval/chain_v3_LE0_eval.json \
  --out_dir paper_artifacts/eval/

python paper_tools/build_tables.py
cd paper
for _ in 1 2 3; do
  pdflatex -interaction=nonstopmode memory_residuals_empirical.tex > /dev/null 2>&1
done
echo "[watchdog] $(date) all done -- paper at paper/memory_residuals_empirical.pdf"
