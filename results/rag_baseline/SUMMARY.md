# Simple RAG Baseline vs MemRes — comparison ledger

**Date:** 2026-05-04
**Eval corpus:** `paper_artifacts/chains/lme_val_s512_evpos.pt` (50 LongMemEval-S
validation chains, evidence-positions patched).
**Metric:** callback cross-entropy in nats, averaged over chain-callback
positions (the `n_callback_tok` answer span of the callback session).
**Eval scripts:**
- MemRes: `tools/eval_callback.py` (existing).
- RAG: `tools/eval_rag_baseline.py` (added in this study).

For both methods the metric of interest is
`Δ = ce_baseline − ce_method` (positive ⇒ method reduces callback CE).
For RAG, `ce_baseline = ce_nomem` (same model, no retrieval).
For MemRes, `ce_baseline = ce_nomem` (same memres-architecture model with
`M_c=None`).

---

## Headline comparison

| size | method | params added | ce_baseline | ce_method | Δ_callback (nats) | Δ_shuffle (nats) | per-chain positive |
|---|---|---|---|---|---|---|---|
| 0.6B | MemRes v27b (n=4 mean, F3-OFF, headline) | +41.5 M | 3.726 | 2.403 | **+1.323 ± 0.530** | +0.000 ± 0.010 | 49/50 (best seed) |
| 0.6B | MemRes v27b seed3 (best seed) | +41.5 M | 4.091 | 2.258 | +1.833 | +0.000 | 49/50 |
| 0.6B | RAG, BM25 top-1 | encoder=∅ | 4.318 | 4.284 | **+0.033** | +0.506 | 20/50 |
| 0.6B | RAG, BM25 top-3 | encoder=∅ | 4.318 | 4.330 | −0.012 | +0.462 | 19/50 |
| 0.6B | RAG, dense top-3 (MiniLM-L6, +22.7 M) | +22.7 M | 4.318 | 4.269 | +0.048 | +0.394 | 20/50 |
| 0.6B | RAG, ORACLE top-3 (gold evidence) | n/a | 4.318 | 4.206 | +0.111 | n/a | 25/50 |
| 1.7B | MemRes v28 (n=2 mean, F3-OFF) | +164.8 M | 2.942 | 2.015 | **+0.926** | −0.005 | reproduced both seeds |
| 1.7B | RAG, BM25 top-3 | encoder=∅ | 4.703 | 4.450 | +0.253 | +0.326 | 29/50 |
| 1.7B | RAG, dense top-3 (MiniLM-L6) | +22.7 M | 4.703 | 4.668 | +0.035 | +0.229 | 32/50 |
| 1.7B | RAG, ORACLE top-3 (gold evidence) | n/a | 4.703 | 4.315 | +0.388 | n/a | 31/50 |

**ce_baseline numbers differ between MemRes and RAG rows because they use
different no-memory references** — the MemRes column scores the trained
memres model with `M_c=None` (so the trained extraction / readout / depth
extra-blocks are still in the forward path), while the RAG column scores
the bare pretrained `Qwen/Qwen3-{0.6B,1.7B}` backbone. Reading the **Δ
columns** is the apples-to-apples comparison: each row's gain over its own
no-context reference.

## Interpretation

1. **Magnitude (Δ over no-context).**
   - At 0.6B, MemRes (+1.32 nats average across n=4 seeds) is **12× larger
     than the strongest realistic RAG cell** (BM25 top-1, +0.033 nats) and
     **12× larger than even the oracle RAG that gets handed the ground-truth
     evidence-session indices** (+0.111 nats).
   - At 1.7B, MemRes (+0.93 nats, n=2 seeds) is **3.7× larger** than the
     strongest realistic RAG cell (BM25 top-3, +0.25 nats) and **2.4× larger**
     than oracle RAG with gold evidence at top-3 (+0.39 nats).
   - At 1.7B, RAG starts becoming usable on this metric: the larger LM is
     better at consuming the appended retrieved sessions. At 0.6B the LM
     simply cannot exploit the appended context — even oracle retrieval
     barely lifts CE.

2. **Chain-specificity profile (Δ over chain-shuffle).**
   - MemRes: `Δ_dsh ≈ 0`. Splicing in a *different* chain's M_c gives the
     same callback CE as the own chain's M_c. The MemRes gain shows up over
     the no-memory floor but not over the random-memory floor — see
     `NEURIPS_NUMBERS.md` "What 'Δ_dsh ≈ 0' means" for the project's
     interpretation.
   - RAG: `Δ_dsh ∈ [+0.23, +0.51]`. Retrieved context *from a different
     chain* is meaningfully worse than retrieved context from the right
     chain. RAG retrieval is provably operating on chain-specific
     information.

3. **Per-chain robustness.**
   - MemRes v27b best seed: 49 of 50 chains improved, median per-chain
     Δ = +0.91 nats — broad and roughly uniform improvement.
   - RAG, every cell: ~20–32 / 50 chains improved. The chain-level
     distribution is bimodal — the LM either uses the retrieved context
     well (large positive Δ) or is confused by it (negative Δ).

4. **Compute and storage.**
   - MemRes: trains 41.5 M params (164.8 M at 1.7B) for ~1.5 h on a single
     H100 (~6 h at 1.7B); inference cost is `O(K=128 slots × d)` per
     callback token, no retrieval index, no encoder model, no extra context
     length consumed.
   - RAG: needs an encoder model (or BM25 inverted index), top-k retrieval
     per query, and **adds 1.5–3 k tokens of context** per callback (top-3
     × ~512 tokens stripped of pad). On a quadratic-attention LM that's a
     direct latency cost; MemRes does not pay it.

## Caveats / what this does and does not show

- This is a **simple** RAG: per-session embedding, top-k by cosine /
  BM25, raw concat into the prompt, no chunking, no rerank, no system
  prompt. A more elaborate pipeline (chunked + reranked + chat-template
  with an "answer using the context" instruction + an instruction-tuned
  LM) would close some of the gap. The point here is the *like-for-like*
  comparison: same backbone family, same eval, no extra fine-tuning, no
  extra training data — just retrieval over the chain's own prior
  sessions.
- The MemRes "chain-shuffle ≈ 0" property is what gives it its
  chain-specificity headline in the paper, but mechanistically it means
  that on this metric **a different chain's M_c is just as good as the
  own chain's M_c** — most of the +1.32 nats gain is shared with the
  shuffled M_c. The RAG comparison is *strictly larger* on the shuffle
  axis: RAG retrieval is doing distinguishable work per chain, even if
  its absolute Δ is smaller.
- All RAG results use the bare pretrained backbone with no fine-tuning.
  Fine-tuning the backbone on LongMemEval (matching MemRes's training
  budget) is the obvious next baseline if this gap is contested.

## Repro

```bash
# MemRes numbers (already produced; here for reference)
python tools/eval_callback.py \
  --model_path Runs/chain_v27b_v24a_no_probe_seed1_0p6b_frozen_local/final \
  --corpora paper_artifacts/chains/lme_val_s512_evpos.pt \
  --output results/eval_v25_seed_pack_evpos/v27b_no_probe_final_lme_val_evpos.json

# Simple RAG baseline (this study)
python tools/eval_rag_baseline.py \
  --base_model Qwen/Qwen3-0.6B \
  --corpus paper_artifacts/chains/lme_val_s512_evpos.pt \
  --rag_method bm25 --top_k 1 \
  --output results/rag_baseline/qwen3_0p6b_bm25_top1.json

python tools/eval_rag_baseline.py \
  --base_model Qwen/Qwen3-0.6B \
  --corpus paper_artifacts/chains/lme_val_s512_evpos.pt \
  --rag_method dense --top_k 3 \
  --encoder sentence-transformers/all-MiniLM-L6-v2 \
  --output results/rag_baseline/qwen3_0p6b_dense_top3.json

python tools/eval_rag_baseline.py \
  --base_model Qwen/Qwen3-0.6B \
  --corpus paper_artifacts/chains/lme_val_s512_evpos.pt \
  --rag_method oracle --top_k 3 \
  --output results/rag_baseline/qwen3_0p6b_oracle_top3.json
```

All RAG JSONs are in `results/rag_baseline/`.
