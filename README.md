# Memory Residuals

## Train

```bash
torchrun --nproc_per_node=1 train_memres.py \
    --data_path data/friends_scripts.jsonl \
    --hidden_size 128 --num_layers 4 --num_heads 4 --num_kv_heads 2 \
    --intermediate_size 256 --head_dim 32 \
    --memres_num_memory_vectors 32 --memres_num_memory_heads 4 \
    --history_len 256 --current_len 128 \
    --steps 200 --batch_size 2 --out_dir output/test_memres
```

## Quick test

```bash
python eval_memres.py --model_path output/test_memres/final \
    --data_path data/friends_scripts.jsonl \
    --history_len 256 --current_len 128 --num_eval 16
```
