# Paper 2 — A training recipe for long-horizon dialogue recall on Memory Residuals

## What this paper claims

The architectural primitive defined in
[paper 1](../paper1_drop_in_primitive/README.md) is **necessary but not
sufficient** for long-horizon recall on dialogue. Naively training the
primitive with vanilla NLL on PG-19-style book chains produces large
in-trainer Δ_sh-m on books but *collapses on dialogue* (LoCoMo
Δ_sh-m within bootstrap CI of zero). The architecture *can* support
long-horizon dialogue recall — the question is what training recipe
gets it there.

This paper defends a four-piece recipe:

1. **Contextual extract source** (`hidden_<L>` instead of token
   embeddings). The compressor needs syntax / anaphora / entity
   binding, not a bag of token identities.
2. **Mixed conversational corpus** with explicit per-source weighting
   (`{pg19:1, tv:4, msc:8}`). Train on the kind of data we want to
   evaluate on; include enough non-dialogue to anchor the LM head.
3. **Negative-chain contrastive loss** ramped over the first 1000
   steps. Gives the routing pool a reason to recruit memory beyond
   what plain NLL provides on dialogue.
4. **Init-parity-preserving router init** (`mem_bias=-32,
   recent_bias=+32` rather than the soft `-4 / +4`). Bit-exact at
   step 0 and lets the contrastive gradient drive memory recruitment
   from a clean baseline.

## What this paper has to demonstrate

A successful paper 2 must clear:

- **LoCoMo Δ_sh-m bootstrap CI excludes zero** (not just in-trainer).
- **Beats no-memory baseline on dialogue callback EM**.
- **Matches or beats compute-matched RAG baseline** (BM25, MiniLM,
  Contriever, fine-tuned MiniLM) on dialogue NLL and callback EM.
- **Source-weighting ablation:** removing MSC from the training mix,
  or removing the contextual extract source, or removing the
  contrastive loss, each measurably hurts the headline.

The "why papers 1 and 2 are separate papers" boundary: paper 1 owns
*"the primitive functions and is sample-efficient"*; paper 2 owns
*"here's the recipe that makes it useful on dialogue."* Each
contribution stands on its own.

## What changed between v3 (legacy) and v4 (this paper)

After the first round of empirical work we hit two reproducible
failure modes:

1. **Bag-of-token-embeddings extraction.** The compressor's input was
   `C_t = embed_tokens(input_ids)`. Without contextual information
   the extractor learned to attend to a weighted average of *token
   identities*, which gives strong in-trainer Δ_sh-m on PG-19 books
   (style/lexical memory) but collapses on dialogue (where two MSC
   sessions about the same person share <30% of token identities).
   **Fix:** `--memres_extract_source hidden_<L>` runs a no_grad
   bare-backbone partial forward and feeds the layer-L hidden state
   to the extractor. Init-parity-preserving (verified bit-exactly).
2. **Out-of-distribution training corpus.** The first paper draft
   trained on PG-19 + TV (96% books by tokens) but evaluated headline
   numbers on LoCoMo (real-life persona dialogues). **Fix:** add MSC
   (multi-session chat) to the training mix with explicit per-source
   weighting; train on the same kind of conversational corpus we want
   to evaluate on.

## Headline experiment matrix (factorial 2 × 2)

Qwen3-0.6B + `attention_parity` routing, K=128, L_E=4, N=8 blocks,
6000 steps, `--mask_padding_loss` (essential for the MSC short-session
+ EOS-padding combo), `--burn_in_resample`, `--neg_chain_weight`
ramped 0.05 → 0.5 over 1000 steps.

|             | extract = `embed` (legacy)            | extract = `hidden_14` (new) |
|-------------|---------------------------------------|----------------------------|
| **PG-19 + TV** (legacy corpus) | **D** = `chain_v3_softparity_full` (DONE; gates ≈ 0) | **C** = `chain_v4_hidden14_pgtv` |
| **PG-19 + TV + MSC** (new corpus) | **B** = `chain_v4_embed_msc` | **A** = `chain_v4_hidden14_msc` (HEADLINE) |

The 2 × 2 factorial cleanly attributes any gain to:

- **Extract source** (column contrast at fixed corpus) → A vs B, or C vs D.
- **Corpus** (row contrast at fixed extract) → A vs C, or B vs D.
- **Joint** (the diagonal contrast) → A vs D, the headline number.

**Compute placement.**

- A (headline) lives **uninterrupted on the GH200** under the cloud
  watchdog (`cwd-chain_v4_hidden14_msc` tmux session). 6000 steps
  × ~24 k tok/step at ~2.8 k tok/s → ~14 h wall-clock.
- B and C are queued behind the local v3 trainers via
  `scripts/sentinel_local_ablations.sh`: when the v3 process on
  GPU N exits, the sentinel launches the corresponding v4 ablation on
  that GPU (B → GPU 0, C → GPU 1).
- D (legacy `embed` + PG-19+TV) is the existing `v3_softparity_full`
  run, used as-is.

## Eval matrix (per checkpoint)

1. **Init parity** (sanity, 30 s/run).
2. **Standalone `eval_chain.py`** on PG-19 val + MSC val + LoCoMo
   reporting Δ_nm-m and Δ_sh-m with bootstrap 95% CIs over chains.
3. **Callback probe** on MSC val (per-callback NLL on persona facts
   that appeared in earlier sessions).
4. **Horizon analysis**: Δ as a function of position-in-chain.
5. **Routing diagnostic**: per-sublayer gate / α_mem trace, plotted
   vs training step.
6. **RAG baselines** (BM25, MiniLM, Contriever, MiniLM-FT) at matched
   FLOPs.
7. Optional: NIAH / RULER-short for cross-arch context.

## What's explicitly *not* in this paper

- 8B run. Compute does not fit.
- 32k+ context window claims. Honest scope is k=8 sessions of 512
  tokens each (4096 tokens of effective in-stream context, augmented
  by $M_c$).
- Long-document QA sweeps (LongBench / RULER-long). We have a single
  training-recipe fix to defend.
- The architectural primitive itself (paper 1's contribution).

## Decision triggers

- **Day +2 sentinel.** If B's standalone Δ_sh-m on MSC val is within
  bootstrap CI of zero, the contextual-extract fix did not transfer
  and we shift the paper from *"contextual extract makes MemRes work
  on dialogue"* to *"why a parity-preserving recurrent memory does
  not transfer to dialogue at this scale"* (a workshop paper, not
  main-track).
- **Day +4 sentinel.** If routing gate norm stays at zero in B (same
  failure as v3), we drop `attention_parity` for the headline and
  rerun the headline with `--router_mem_bias_init -4
  --router_recent_bias_init 4` (loses bit-exact parity but gives the
  router gradient signal from step 0).

## Reproducing

### Build the mixed-corpus chain artifact

```bash
# 1. Convert MSC parquet to per-chain JSONLs
python paper_tools/build_msc_chains.py \
    --in_parquet ../memory_residuals_data/hf_corpora/msc/data/train-*.parquet \
    --out_dir    ../memory_residuals_data/stage1/msc/train \
    --min_sessions 3
python paper_tools/build_msc_chains.py \
    --in_parquet ../memory_residuals_data/hf_corpora/msc/data/validation-*.parquet \
    --out_dir    ../memory_residuals_data/stage1/msc/val \
    --min_sessions 3

# 2. Pretokenise MSC, then merge with the existing PG-19+TV pretokenised corpus
python paper_tools/pretokenize_chains.py \
    --in_dir ../memory_residuals_data/stage1/msc/train \
    --out_path paper_artifacts/chains/msc_train_s512.pt \
    --session_len 512 --min_tokens 32 --min_sessions_per_chain 3
python paper_tools/merge_chain_corpora.py \
    --in paper_artifacts/chains/stage1_train_s512.pt \
    --in paper_artifacts/chains/msc_train_s512.pt \
    --out paper_artifacts/chains/stage1_msc_train_s512.pt
# (likewise for validation)
```

### Train the headline run (A)

```bash
python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_mem_bias_init -32 \
    --router_recent_bias_init 32 \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/stage1_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/stage1_msc_val_s512.pt \
    --source_weights '{"pg19":1.0,"tv":4.0,"msc":8.0}' \
    --window_k 8 --batch_size 4 --grad_accum 4 \
    --steps 6000 --eval_every 250 --save_every 500 \
    --neg_chain_weight 0.5 --neg_chain_initial_weight 0.05 --neg_chain_warmup_steps 1000 \
    --neg_chain_margin 0.05 \
    --mask_padding_loss --burn_in_resample \
    --run_name chain_v4_hidden14_msc \
    --out_dir output/chain_v4_hidden14_msc
```

### Standalone eval

```bash
bash paper_tools/post_train_pipeline.sh chain_v4_hidden14_msc
```

This runs `eval_chain.py` on PG-19, MSC, and LoCoMo; runs the callback
probe on MSC; runs horizon analysis; rebuilds figures.

## Status (live)

- **Headline A (`chain_v4_hidden14_msc`)**: TRAINING on GH200 in tmux
  `cwd-chain_v4_hidden14_msc`. (See cloud watchdog status for
  current step / loss / ETA.)
- **Mixed corpus** `stage1_msc_train_s512.pt`: 5928 chains
  (1899 PG-19 + 29 TV + 4000 MSC), 102 664 sessions. All eligible
  at `--window_k 3`. Source weights `{pg19:1, tv:4, msc:8}`.
- **Init parity** verified bit-exactly for both `embed` and
  `hidden_14` extract sources, both with and without memory injected.
- **Local v3 (legacy embed baselines)**: STOPPED Apr 28 21:23 at
  step 4425 / 6000; both `best/` ckpts persisted at step 4400. See
  [`paper_artifacts/eval/chain_v3_training_summary.md`](../../paper_artifacts/eval/chain_v3_training_summary.md).
  These are the failure-mode baseline that motivates this paper's
  recipe — not the headline.
