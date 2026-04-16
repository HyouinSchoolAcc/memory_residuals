# Memory Residuals

A Qwen3 variant that keeps a fixed-size compressed memory of past sessions and
lets every layer softmax-route attention over `[local hidden states || memory]`.
When the next token is generic filler, the routing collapses to local tokens.
When it's a callback to history, mass flows to the memory block.

Four scripts:
- `train_memres.py` — train (from scratch or attached to pretrained Qwen3)
- `eval_memres.py` — measure loss gain from having memory vs not
- `probe_memres.py` — measure whether routing is semantic (callback > filler)
- `visualize_memres.py` — plot per-layer routing mass

---

## Train from scratch

Small sanity run (~20M params, single GPU):

```bash
torchrun --nproc_per_node=1 train_memres.py \
    --data_path data/friends_scripts.jsonl \
    --hidden_size 128 --num_layers 4 --num_heads 4 --num_kv_heads 2 \
    --intermediate_size 256 --head_dim 32 \
    --memres_num_memory_vectors 32 --memres_num_memory_heads 4 \
    --history_len 256 --current_len 128 \
    --steps 200 --batch_size 2 --out_dir output/test_memres
```

Larger from-scratch run (~80M):

```bash
torchrun --nproc_per_node=1 train_memres.py \
    --data_path data/friends_scripts.jsonl \
    --hidden_size 512 --num_layers 12 --num_heads 8 --num_kv_heads 4 \
    --intermediate_size 1536 \
    --memres_num_memory_vectors 128 --memres_num_memory_heads 8 \
    --history_len 512 --current_len 256 \
    --steps 3000 --batch_size 2 --grad_accum 2 \
    --out_dir output/bigger_memres
```

## Attach MemRes on top of a pretrained Qwen3

The base weights load from the HF checkpoint; MemRes projections and the
memory block stay randomly initialized and learn from scratch. The gate
(below) keeps the random MemRes from corrupting the pretrained residual
stream at step 0.

```bash
torchrun --nproc_per_node=1 train_memres.py \
    --pretrained Qwen/Qwen3-0.6B \
    --data_path data/friends_scripts.jsonl \
    --memres_num_memory_vectors 128 --memres_num_memory_heads 8 \
    --memres_use_gate --memres_gate_init -2.0 \
    --history_len 512 --current_len 256 \
    --steps 2000 --batch_size 1 --grad_accum 4 \
    --lr 2e-5 --warmup 100 \
    --out_dir output/qwen3_0p6b_memres_gated
```

---

## Arguments

### Data

- `--data_path` — JSONL file with `{"history": "...", "current": "..."}` per
  line. `history` is compressed into `M_c`; `current` is the target sequence
  the model is scored on.
- `--simulate_sessions` — alternative to `--data_path`. Streams raw text and
  splits each document in half (first half = history, second half = current).
- `--dataset`, `--dataset_name` — HuggingFace dataset id when using
  `--simulate_sessions` (default `HuggingFaceFW/fineweb-edu`).

### Model size (from-scratch only, ignored when `--pretrained` is set)

- `--hidden_size` — model dim `d`.
- `--num_layers` — transformer blocks.
- `--num_heads` — attention heads.
- `--num_kv_heads` — grouped-query KV heads (must divide `num_heads`).
- `--intermediate_size` — MLP hidden dim.
- `--head_dim` — overrides `hidden_size / num_heads` if set.

### Pretrained base

- `--pretrained Qwen/Qwen3-0.6B` — load base weights from HF. Size flags above
  are ignored. Only MemRes params (memory block + per-layer projections) are
  randomly initialized.

### Memory Residual knobs

- `--memres_num_memory_vectors K` — number of latent memory slots. Fixed,
  independent of history length. Common values: 32 (small), 128 (default),
  256 (big).
- `--memres_num_memory_heads` — multi-head count inside the memory block's
  cross-attention. Must divide `hidden_size`.
- `--memres_apply_at {attn, mlp, both}` — where to insert the MemRes routing
  site in each decoder layer:
  - `attn` — once, before self-attention.
  - `mlp` — once, before the MLP.
  - `both` — two sites per layer (default; more routing capacity).
- `--memres_use_gate` — add a learned per-token sigmoid gate that mixes the
  memory-routed output with the untouched hidden states:
  `h = (1 - σ(W·h)) · h + σ(W·h) · h_tilde`. Needed when attaching MemRes on
  top of a pretrained base so random-init projections don't corrupt the
  pretrained residual stream at step 0.
- `--memres_gate_init` — bias initialization for the gate (default `-2.0`).
  `σ(-2) ≈ 0.12`, so at step 0 the layer is ~88% original, ~12% memory-routed.
  The gate is free to open to 1.0 or close to 0.0 through training.

### Sequence lengths

- `--history_len` — token length of the past session compressed into memory.
- `--current_len` — token length of the future session scored for loss.

### Optimization

- `--steps` — optimizer steps (after gradient accumulation).
- `--batch_size` — per-step micro-batch.
- `--grad_accum` — micro-batches accumulated per optimizer step. Effective
  batch = `batch_size * grad_accum * world_size`.
- `--lr`, `--lr_min` — cosine schedule endpoints. Use `6e-4 → 6e-5` from
  scratch, `2e-5 → 2e-6` when attaching to pretrained weights.
- `--warmup` — linear warmup steps.
- `--max_norm` — gradient clipping.
- `--save_every`, `--log_every` — checkpoint and log cadence.
- `--out_dir` — where to save.
- `--wandb_project`, `--wandb_entity`, `--run_name` — optional W&B logging.
- `--seed` — RNG seed.

### Tokenizer

The Qwen3 tokenizer is always used (`Qwen/Qwen3-0.6B`), regardless of model
size, so from-scratch and pretrained runs are interchangeable on the same
data.

---

## Evaluate

Scores each held-out `(history, current)` pair twice: once with memory, once
without. Reports mean cross-entropy and the delta.

```bash
python eval_memres.py \
    --model_path output/qwen3_0p6b_memres_gated/final \
    --data_path data/friends_scripts.jsonl \
    --history_len 512 --current_len 256 --num_eval 32
```

Args:
- `--model_path` — checkpoint dir (contains `config.json`, weights).
- `--data_path` — same JSONL as training.
- `--tokenizer` — default `Qwen/Qwen3-0.6B`.
- `--history_len`, `--current_len` — must match what was used at train time
  (sequence length the model saw).
- `--num_eval` — number of held-out samples.
- `--eval_start` — index where the held-out split starts. Training uses
  the first `eval_start` lines (default 200).

Output: `delta = loss_without_memory - loss_with_memory`. Positive means
memory reduces loss.

---

## Probe (is routing actually semantic?)

Compresses a history once, then feeds two paired continuations:
- a **callback** ("Remember what we just talked about? ...")
- a **filler** ("The quick brown fox ...")

Measures α mass routed to `M_c` in each case. Positive `callback - filler`
Δ means the model opens memory more on callbacks than filler.

```bash
python probe_memres.py \
    --model_path output/qwen3_0p6b_memres_gated/final \
    --data_path data/friends_scripts.jsonl \
    --num_samples 16
```

Args:
- `--model_path`, `--data_path`, `--tokenizer`, `--eval_start`, `--device`
  — same as eval.
- `--history_len` — how much history to compress (default 512).
- `--probe_len` — token length of the suffix continuation to score routing
  on (default 32).
- `--num_samples` — number of held-out histories to probe.

Output per MemRes site: mean α on memory for callback vs filler, and Δ.

---

## Visualize

Same measurement as the probe, plotted as (a) per-site bar chart and
(b) per-sample × per-site Δα heatmap.

```bash
python visualize_memres.py \
    --model_path output/qwen3_0p6b_memres_gated/final \
    --data_path data/friends_scripts.jsonl \
    --num_samples 16 \
    --output memres_routing.png
```

Args: same as the probe, plus `--output` for the PNG path.
