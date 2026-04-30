# Experiment 2 — A training recipe for long-horizon dialogue recall on Memory Residuals

## What this paper claims

The architectural primitive defined in
[experiment 1](../exp1_drop_in_primitive/README.md) is **necessary but not
sufficient** for long-horizon recall on dialogue. Naively training the
primitive with vanilla NLL on PG-19-style book chains produces large
in-trainer Δ_sh-m on books but *collapses on dialogue* (LoCoMo
Δ_sh-m within bootstrap CI of zero). The architecture *can* support
long-horizon dialogue recall — the question is what training recipe
gets it there.

The recipe (v6 — what we ship after the v5 dead end). Five pieces.
Live training results in [`runs.md`](runs.md):

1. **Conversational long-chain corpus with explicit recall-callback
   supervision.** Train on LongMemEval-S (450 chains × ~50 sessions of
   conversational history × a memory question whose answer is in some
   prior session). Each chain ships a `session_callback_mask` marking
   the answer-span tokens; those tokens get an extra
   `--callback_loss_weight 10.0` in the per-session NLL. PG-19 books
   and MSC alone gave no gradient signal that says "look at M_c
   instead of just predicting from local context"; the callback span
   is the gradient signal that fixes that.
2. **Callback-aligned window sampling**
   (`--callback_window_bias 0.7`). 70% of sampled chain windows are
   aligned so the callback session is the LAST position in the
   window. The remaining 30% sample uniformly. Without this, only a
   small fraction of the supervision tokens are reachable per epoch
   and the strong callback weight is wasted.
3. **Non-replacing gated memory update** (`--memres_update_mode
   gated`). The legacy update was a competitive softmax over
   `[M_c^{t-1} || M_new]` (zero-sum: writing to a slot fully
   overwrites it). v6 adds a per-slot sigmoid write gate
   `g = σ(W [M_c^{t-1} || M_new] + b_g)` initialised to bias `-1.0`
   (so `g ≈ 0.27` at init: modest writes, dominant carry). The new
   slot becomes `M_c^t = (1 - g) * M_c^{t-1} + g * judge(...)`,
   matching the well-known GRU/LSTM "convex combination" update.
   This stops M_c from being clobbered every step on long chains.
4. **Soft parity router init** (`--router_mem_bias_init -4
   --router_recent_bias_init 4`) on `attention_parity` routing.
   The initially-attractive hard `±32` init looks parity-preserving
   but the depth-router softmax mass on memory at that init is
   `exp(-64) ≈ 1.6e-28` — ~20 orders of magnitude below bf16
   representability, so the optimizer step on `mem_bias` is a literal
   numerical zero and no amount of supervision can move it. At soft
   `±4` the init mass is `~1.7e-4`, well above bf16, and the gate
   is recruitable. **`attention_parity` is locked in here**:
   ablations against ReZero-style routing (`attention_base` with
   zero-init pseudo-queries) and against `simple_gate` showed both
   are strictly worse at the same `±4` bias level on v3 trajectories.
5. **Backbone fine-tune at `--lr_backbone 2e-5`** (~7× the
   conservative freeze-warm rate) with `--carry_state` so M_c
   persists detached across minibatches. The LM head needs to adapt
   to actually consume memory rather than ignore it; carrying state
   means the model practices reading from already-warm M_c rather
   than always from zero.

What v5 had that v6 dropped:

- Per-session memory + context dropouts. v5 cell B confirmed they
  cause posterior collapse on long chains under the new gated update.
- Negative-chain contrastive loss. v5 cells confirmed the model can
  game the margin by making both `mem` and `shuffle` forwards bad,
  and the loss isn't doing the architectural recruitment work the
  callback supervision now does directly.
- `--memres_extract_source hidden_14` (a v5-recipe piece). v6 keeps
  it on the GH200 cell but the local cells both run with the
  default; if the local cells succeed without it, we drop it from
  the headline recipe.

## What this paper has to demonstrate

A successful experiment 2 must clear:

- **LoCoMo Δ_sh-m bootstrap CI excludes zero** (not just in-trainer).
- **Beats no-memory baseline on dialogue callback EM**.
- **Matches or beats compute-matched RAG baseline** (BM25, MiniLM,
  Contriever, fine-tuned MiniLM) on dialogue NLL and callback EM.
- **Source-weighting ablation:** removing MSC from the training mix,
  or removing the contextual extract source, or removing the
  contrastive loss, each measurably hurts the headline.

Boundary with experiment 1: exp 1 (now treated as background /
baseline) owns *"the primitive functions and is sample-efficient on
books"*; exp 2 owns *"here's the recipe that makes it useful on
dialogue, and here's the bf16-saturation failure mode that almost
killed it."*

## What was wrong before (v3 → v4 → v5 → v6 dead-end ladder)

Four reproducible failure modes, each fixed in turn:

1. **Bag-of-token-embeddings extraction (v3 era).** The compressor's
   input was `C_t = embed_tokens(input_ids)`. Without contextual
   information the extractor learned to attend to a weighted average
   of *token identities*, which gives strong in-trainer Δ_sh-m on
   PG-19 books (style/lexical memory) but collapses on dialogue
   (where two MSC sessions about the same person share <30% of token
   identities). **Fix:** `--memres_extract_source hidden_<L>` runs a
   no_grad bare-backbone partial forward and feeds the layer-L
   hidden state to the extractor. Init-parity-preserving (verified
   bit-exactly).
2. **Wrong domain training corpus (v3-v5 era).** The first paper
   draft trained on PG-19 + TV (96% books by tokens) but evaluated
   headline numbers on LoCoMo (real-life persona dialogues).
   v5 added MSC (5-session chats with persona facts) to the mix,
   which still wasn't long enough or supervised enough to recruit
   memory. **Fix:** train directly on LongMemEval-S — 50-session
   chains with explicit memory-question annotations; build a
   `session_callback_mask` aligned to the answer span.
3. **bf16-saturating router init (v4 era).** The hard
   `mem_bias=-32, recent_bias=+32` parity init looks scientifically
   correct (bit-exact at step 0) but produces a depth-router softmax
   mass on memory of `exp(-64) ≈ 1.6e-28` — ~20 orders of magnitude
   below bf16's representable smallest positive. The optimizer's
   gradient on `mem_bias` is therefore a literal numerical zero. We
   trained `chain_v4_hidden14_msc` for 5400 steps before the EVAL
   trace (`mem == nomem == shuffle` to 4 decimals at every check)
   made the diagnosis unambiguous. **Fix:** soft `±4` init — same
   parity shape, but mass `~1.7e-4` at init, well above bf16.
4. **Memory channel never recruits under uniform NLL (v5 era).** All
   three v5 cells (different corpora, different extract sources,
   with and without regularisers) converged on the same `gate_max=0`
   signature: train loss drops fine but the depth router's α_mem
   never opens, Δ_sh-m saturates at ~+0.0006 (statistical noise). The
   `Δ_or-m ≈ -0.23 nats` oracle gap proved memory COULD help by 0.23
   nats per token if it were ever consulted. The bottleneck is
   structural: under uniform NLL on conversational chat, predicting
   the next token from local context is always good enough for the
   loss; nothing in the loss says "you'd be 0.23 nats better if you
   read memory." **Fix:** weighted NLL with `--callback_loss_weight
   10.0` on tokens from the answer span of an explicit memory
   question. Combined with `--callback_window_bias 0.7` so the
   sampler reliably visits the supervised tokens. This is recipe
   pieces 1+2 above.

A fifth issue, **memory clobbering on long chains under the
competitive softmax update**, is hypothesised but not yet
empirically isolated. The competitive update writes to slots in a
zero-sum way each step, which on a 50-session chain potentially
overwrites useful state from session 5 by session 30. Recipe piece 3
(gated update) tests whether allowing partial writes helps; the v6
COMPETITIVE cell is the matched control.

## Headline experiment matrix (v6, three cells)

Qwen3-0.6B + `attention_parity` routing at soft `±4`, K=128, L_E=4,
N=8 blocks, 8000 steps, callback supervision on. All three cells
share every other knob.

| cell | machine | update mode | window_k | extract source |
|---|---|---|---:|---|
| **GATED** = `chain_v6_lme_gated_callback` (HEADLINE) | local GPU 0 | gated | 8 | embed |
| **COMPETITIVE** = `chain_v6_lme_competitive_callback` (arch A/B) | local GPU 1 | competitive | 8 | embed |
| **GATED-DEEP** = `chain_v6_lme_gated_callback_w12` (depth ablation) | GH200 | gated | 12 | hidden_14 |

Single-knob ablation map:

- **Update mode** (GATED vs COMPETITIVE at fixed window_k=8 and
  extract): isolates the gated update's contribution.
- **TBPTT depth** (GATED vs GATED-DEEP at fixed mode=gated):
  isolates whether deeper recurrence through the judge helps.
- **Extract source** is partly confounded with depth (only GATED-DEEP
  uses `hidden_14`); a dedicated extract ablation will be queued
  after we see whether the headline is unblocked at all.

For per-cell launch commands, machine placement, current step / loss
state, ETA, and historical (v3/v4/v5) cells, see [`runs.md`](runs.md).

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
- The architectural primitive itself (experiment 1's contribution).

## Decision triggers (v6, per cell)

Evaluated on the in-trainer EVAL line on `lme_val`:

- **Step ~500: `gate_max > 0`.** v3-v5 cells were stuck at 0
  indefinitely. If still 0 at step 500, the architecture isn't
  engaging memory at all and the supervision pivot didn't fix it
  either; we ablate further (try `--callback_loss_weight 30.0`,
  try a learnable per-slot init).
- **Step ~1000: `Δ_sh-m > +0.005`.** This is the v3 envelope on
  PG-19; matching it on LME means the recipe transferred to dialogue.
- **Step ~2500: `Δ_sh-m > +0.020` AND `α_mem > 5%`** on a routing
  trace. This is where we declare the recipe works and start
  finalising eval / writeup.

Cross-cell rules:

- If GATED catches up to COMPETITIVE on Δ_sh-m by step 1000: gated
  update isn't earning its keep → drop from recipe in v7.
- If GATED beats COMPETITIVE by ≥+0.005 Δ_sh-m at step 2500: gated
  is the recipe and we ship it.
- If GATED-DEEP beats GATED on Δ_sh-m by step 1500: window_k=8 is
  too shallow → bump in v7.

`Δ_sh-m = +0.0000` to 4 decimals (the v4 hard-init signature, and
the v5 `gate_max=0` signature) is the hard fail.

## Reproducing

### Build the LongMemEval callback corpus

```bash
mkdir -p ../memory_residuals_data/longmemeval
huggingface-cli download xiaowu0162/longmemeval longmemeval_s.json \
    --local-dir ../memory_residuals_data/longmemeval --repo-type dataset

python paper_tools/build_conversational_callback_chains.py \
    --source longmemeval \
    --in_path ../memory_residuals_data/longmemeval/longmemeval_s.json \
    --out_train paper_artifacts/chains/lme_train_s512.pt \
    --out_val   paper_artifacts/chains/lme_val_s512.pt \
    --val_n 50 --seed 123 \
    --tokenizer_preset qwen3-0.6b-large --session_len 512
```

Optional add-ons (used in some ablations, not in headline):

```bash
python paper_tools/build_conversational_callback_chains.py \
    --source msc --append_persona_callback \
    --in_path ../memory_residuals_data/hf_corpora/msc/raw_train.jsonl \
    --out_train paper_artifacts/chains/msc_callback_train_s512.pt

python paper_tools/build_conversational_callback_chains.py \
    --source realtalk \
    --in_dir ../memory_residuals_data/realtalk/data \
    --out_eval paper_artifacts/chains/realtalk_eval_s512.pt
```

### Train

The canonical recipe lives in [`Scripts/`](../../Scripts/) — the
shell scripts are the source of truth for the *exact* config used to
produce numbers in this paper. The README above lists the *intent*
of the recipe; the scripts list the *flags*. If they ever drift,
trust the scripts.

> **Note (2026-04-30):** the three v6 launchers that produced the
> numbers in this paper (`train_v6_lme_gated_callback.sh`,
> `train_v6_ablation_a_competitive.sh`,
> `train_v6_lme_gated_w12_gh200.sh`) were pruned from `Scripts/` in
> the repo cleanup. Only one v6 variant survives on disk:
> [`archive/scripts/train_v6_lme_msc_gated_gh200.sh`](../../archive/scripts/train_v6_lme_msc_gated_gh200.sh),
> the GH200 flavour of the GATED headline cell. The deleted
> launchers differed only in corpus and `window_k`; the full flag
> lists are preserved in git history and in this README's recipe
> section above. The *active* launchers in `Scripts/` are v11 (see
> `runs.md` and the v11 "Stop everything" banner in the top-level
> README).

Each script's header documents its purpose and what it diverges from
the others on.

### Standalone eval

The v3-era one-shot wrapper (`paper_tools/post_train_pipeline.sh`)
was hardcoded to v3 ckpt names and lives in
`archive/paper_tools/`. For v6, run the steps individually:

```bash
# eval on lme_val (the matched train/val split)
python paper_tools/eval_chain.py \
  --model_path Runs/<run_name>/best \
  --corpora paper_artifacts/chains/lme_val_s512.pt \
            paper_artifacts/chains/realtalk_eval_s512.pt \
  --names lme_val realtalk \
  --score_window 4 --oracle_window 4 \
  --output paper_artifacts/eval/<run_name>_eval.json

# bootstrap CI on Δ_sh-m
python paper_tools/bootstrap_ci.py \
  --input paper_artifacts/eval/<run_name>_eval.json \
  --output paper_artifacts/eval/<run_name>_ci.json

# horizon decomposition
python paper_tools/horizon_analysis.py \
  --inputs paper_artifacts/eval/<run_name>_eval.json \
  --out_dir paper_artifacts/eval/

# routing trace + counterfactual on the best ckpt
python paper_tools/routing_trace.py --model_path Runs/<run_name>/best \
  --output paper_artifacts/eval/routing_<run_name>.json
python paper_tools/counterfactual_eval.py --model_path Runs/<run_name>/best \
  --output paper_artifacts/eval/cf_<run_name>.json
```

## Live state and run ledger

Per-run status (training/done/failed, last step, last `Δ_sh-m`,
machine, log path, exact launch command) lives in
[`runs.md`](runs.md). That file is the single source of truth for
which checkpoint produced which number — keep it updated as runs
finish, fail, or get re-launched.

## Static facts

- **v6 train corpus** `lme_train_s512.pt`: 450 LongMemEval-S chains,
  21 869 sessions, 5 367 callback positions, ~2.7M tokens after
  Qwen3 BPE tokenisation. All eligible at `--window_k ≤ 12`.
- **v6 val corpus** `lme_val_s512.pt`: 50 held-out LongMemEval-S
  chains, 2 498 sessions, 666 callback positions.
- **Routing config** locked at `attention_parity` with
  `--router_mem_bias_init -4 --router_recent_bias_init 4` for all
  v6 cells. Earlier ablations established that ReZero-style routing
  (zero-init pseudo-queries in `attention_base` mode) is strictly
  worse at the same bias level, and `simple_gate` is strictly worse
  than both. Hard ±32 saturates bf16 (recipe piece 4 above).
- **Init parity** verified bit-exactly for both `embed` and
  `hidden_14` extract sources, both with and without memory injected
  (`paper_tools/init_parity_test.py`). Holds at hard `±32` and at
  soft `±4`; the saturation problem is a gradient issue, not a
  forward-pass issue. Init parity for the gated update is
  trivially preserved at step 0 because `M_c^0 = 0` so
  `M_c^1 = g * judge(0, ...) = 0` as well.
