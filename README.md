# Memory Residuals

A Qwen3 variant implementing the Memory Residuals architecture from the paper:
end-to-end differentiable lifelong memory via two key advances.

**Advance 1 — Two-Stage QKV Competition (Section 2.1):**
A recurrent memory block that compresses each session into a candidate via
learnable extraction queries, then forces old memory and new candidate into
zero-sum softmax competition via judging queries.  Mundane filler sessions
pass old memory through untouched; critical state-changes overwrite.

**Advance 2 — Depth-Wise Residual Stream Injection (Section 2.2):**
The memory matrix is compressed into a single vector `m^t` via a learned
readout query, then registered as `v_0` in a depth-wise attention pool.
Each layer's input is a softmax-weighted mix of all preceding layer outputs
plus this memory source, using `phi(q, k) = exp(q^T RMSNorm(k))` with a
learned per-layer pseudo-query `w_l`.  During filler turns, local layer
outputs dominate the softmax; during callbacks, mass shifts onto `v_0`.

> **Paper:** see [`memory_residuals.pdf`](memory_residuals.pdf) in this repo for
> the full theoretical framework and the call for collaboration.

Four scripts:
- `train_memres.py` — train (from scratch or attached to pretrained Qwen3)
- `eval_memres.py` — measure loss gain from having memory vs not
- `probe_memres.py` — measure whether routing is semantic (callback > filler)
- `visualize_memres.py` — plot per-layer routing mass

The canonical implementation lives in `modeling_memres.py` (all four scripts
import from it). `modeling_memory_residuals.py` is an earlier variant kept for
reference and is not imported by the current pipeline.

---

## Paper-to-Code Map

| Paper Section / Equation | Module in `modeling_memres.py` |
|---|---|
| Section 2.1, Eq. 1 — Extraction | `MemoryBlock.extract()` via `CrossAttention` |
| Section 2.1, Eq. 2 — Judging | `MemoryBlock.judge()` via `CrossAttention` |
| Section 2.1.1, Eq. 3-5 — Multi-layer judging | `MemoryBlock.judging_layers` + `readout` |
| M_in (extraction queries) | `MemoryBlock.M_in` |
| M_judge (judging queries) | `MemoryBlock.M_judge` |
| Section 2.2, Eq. 6 — Readout | `MemoryReadout.forward()` |
| Section 2.2, Eq. 7 — v_0 := m^t | `Qwen3MemResModel.forward()` (v_0 broadcast) |
| Section 2.2, Eq. 8 — Depth-wise routing | `DepthWiseRouter.route()` |
| phi(q,k) = exp(q^T RMSNorm(k)) | `DepthWiseRouter.route()` (scores computation) |
| w_l (per-layer pseudo-query) | `DepthWiseRouter.w` |
| r (readout query) | `MemoryReadout.r` |

---

## Train from scratch

Small sanity run (~20M params, single GPU):

```bash
torchrun --nproc_per_node=1 train_memres.py \
    --data_path data/friends_scripts.jsonl \
    --hidden_size 128 --num_layers 4 --num_heads 4 --num_kv_heads 2 \
    --intermediate_size 256 --head_dim 32 \
    --memres_num_vectors 32 \
    --history_len 256 --current_len 128 \
    --steps 200 --batch_size 2 --out_dir output/test_memres
```

Larger from-scratch run (~80M):

```bash
torchrun --nproc_per_node=1 train_memres.py \
    --data_path data/friends_scripts.jsonl \
    --hidden_size 512 --num_layers 12 --num_heads 8 --num_kv_heads 4 \
    --intermediate_size 1536 \
    --memres_num_vectors 128 --memres_judging_depth 1 \
    --history_len 512 --current_len 256 \
    --steps 3000 --batch_size 2 --grad_accum 2 \
    --out_dir output/bigger_memres
```

## Attach MemRes on top of a pretrained Qwen3

The base weights load from the HF checkpoint; MemRes params (memory block,
readout, depth-wise router) stay randomly initialized and learn from scratch.

```bash
torchrun --nproc_per_node=1 train_memres.py \
    --pretrained Qwen/Qwen3-0.6B \
    --data_path data/friends_scripts.jsonl \
    --memres_num_vectors 128 --memres_judging_depth 1 \
    --history_len 512 --current_len 256 \
    --steps 2000 --batch_size 1 --grad_accum 4 \
    --lr 2e-5 --warmup 100 \
    --out_dir output/qwen3_0p6b_memres
```

---

## Arguments

### Data

- `--data_path` — JSONL file with `{"history": "...", "current": "..."}` per
  line. `history` is compressed into memory; `current` is the target sequence.
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

### Memory Residual knobs

- `--memres_num_vectors K` — number of latent memory slots (K in the paper).
  Fixed, independent of history length. Common values: 32 (small), 128
  (default), 256 (big).
- `--memres_judging_depth L_J` — number of judging refinement layers
  (Section 2.1.1). Default 1 (single-pass judging). Higher values (2-8) add
  non-linear refinement capacity for complex multi-hop callbacks.

### Sequence lengths

- `--history_len` — token length of the past session compressed into memory.
- `--current_len` — token length of the future session scored for loss.

### Optimization

- `--steps` — optimizer steps (after gradient accumulation).
- `--batch_size` — per-step micro-batch.
- `--grad_accum` — micro-batches per optimizer step.
- `--lr`, `--lr_min` — cosine schedule endpoints.
- `--warmup` — linear warmup steps.
- `--max_norm` — gradient clipping.
- `--save_every`, `--log_every` — checkpoint and log cadence.
- `--out_dir` — where to save.
- `--wandb_project`, `--wandb_entity`, `--run_name` — optional W&B logging.
- `--seed` — RNG seed.

---

## Evaluate

Scores each held-out `(history, current)` pair twice: once with memory, once
without. Reports mean cross-entropy and the delta.

```bash
python eval_memres.py \
    --model_path output/qwen3_0p6b_memres/final \
    --data_path data/friends_scripts.jsonl \
    --history_len 512 --current_len 256 --num_eval 32
```

---

## Probe (is routing actually semantic?)

Compresses a history once, then feeds two paired continuations:
- a **callback** ("Remember what we just talked about? ...")
- a **filler** ("The quick brown fox ...")

Measures `alpha_{0->l}` (attention mass routed to `v_0 = m^t`) at each
depth-wise routing layer. Positive `callback - filler` delta means the model
opens memory more on callbacks than filler.

```bash
python probe_memres.py \
    --model_path output/qwen3_0p6b_memres/final \
    --data_path data/friends_scripts.jsonl \
    --num_samples 16
```

---

## Visualize

Same measurement as the probe, plotted as (a) per-layer bar chart and
(b) per-sample x per-layer delta-alpha heatmap.

```bash
python visualize_memres.py \
    --model_path output/qwen3_0p6b_memres/final \
    --data_path data/friends_scripts.jsonl \
    --num_samples 16 \
    --output memres_routing.png
```
