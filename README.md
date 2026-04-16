# Memory Residuals

## Train from scratch

```bash
torchrun --nproc_per_node=1 train_memres.py \
    --data_path data/friends_scripts.jsonl \
    --hidden_size 128 --num_layers 4 --num_heads 4 --num_kv_heads 2 \
    --intermediate_size 256 --head_dim 32 \
    --memres_num_memory_vectors 32 --memres_num_memory_heads 4 \
    --history_len 256 --current_len 128 \
    --steps 200 --batch_size 2 --out_dir output/test_memres
```

## Attach MemRes on top of a pretrained Qwen3

```bash
torchrun --nproc_per_node=1 train_memres.py \
    --pretrained Qwen/Qwen3-0.6B \
    --data_path data/friends_scripts.jsonl \
    --memres_num_memory_vectors 128 --memres_num_memory_heads 8 \
    --history_len 512 --current_len 256 \
    --steps 1000 --batch_size 1 --grad_accum 4 \
    --lr 2e-5 --warmup 100 \
    --out_dir output/qwen3_0p6b_memres
```

## Quick test

```bash
python eval_memres.py --model_path output/test_memres/final \
    --data_path data/friends_scripts.jsonl \
    --history_len 256 --current_len 128 --num_eval 16
```
