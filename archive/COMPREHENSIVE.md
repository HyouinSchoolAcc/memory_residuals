# Memory Residuals — Comprehensive Notes

This file consolidates everything that was previously split across
`MEMORY_RESIDUALS_BRIEFING.md` and `SUMMARY.md`. Read the [README](README.md)
first; come here only if you need full background, every prior failure mode,
or the per-run history.

The original position paper is in `memory_residuals.pdf` /
`memory_residuals.txt`. The Block Attention Residuals reference is in
`atn_residuals.pdf`. The training data is documented in the sibling
`memory_residuals_data/` folder (untouched here on purpose).

> **Path note (2026-04-30 repo cleanup).** Historical path references
> below — `Runs/chain_v*/best`, `Scripts/train_v{3..10}*.sh` — point
> to run directories and launcher scripts that were pruned from disk
> in the 2026-04-30 cleanup. `output/` is now a compat symlink to
> `Runs/` (preserved for the active v11 process); `scripts/` is a
> symlink to `Scripts/`. The surviving run directories are listed in
> the top-level `README.md` File map. This document is the full
> historical ledger; the configs remain reproducible via the flag
> lists written inline.

---

# Part I — Status briefing (was MEMORY_RESIDUALS_BRIEFING.md)


*Comprehensive briefing for someone picking the project up cold. Audience: a
collaborator deciding whether to invest GPU time, and on which recipe.*

---

## 1. What Memory Residuals is

### 1.1 The problem it tries to solve

A deployed conversational agent that must remember a user across days or
weeks faces three bad options today:

1. **Long-context concatenation.** Cost scales as $O(N^2)$ in the
   conversation length, and current models exhibit "lost-in-the-middle"
   degradation past ~8K tokens.
2. **Retrieval-augmented generation (RAG).** A vector store of past
   utterances is queried at inference time. Retrieval is non-differentiable,
   brittle when callbacks depend on implicit state rather than lexical
   overlap, and weak on the *abstractive* recall regime where the relevant
   history cannot be served as a verbatim snippet.
3. **Learned recurrent memory.** A fixed-size internal state crosses
   session boundaries; differentiable and bounded in cost, but every prior
   incarnation (Transformer-XL, Compressive Transformer, RMT,
   Block-Recurrent, Memorizing Transformers, LRMT) has reported some flavor
   of channel collapse, training instability, or shortcut learning where
   the recurrent channel encodes only style cues rather than episodic
   content.

Memory Residuals is an attempt at the third option done correctly: a
mathematically-elegant, end-to-end-differentiable recurrent memory that
plugs into a pretrained Transformer backbone without disrupting it.

### 1.2 The two core architectural advances

Both live in `memory_residuals/modeling_memres.py` and are referenced
back to specific equations of `memory_residuals/memory_residuals.txt` /
`memory_residuals.pdf`.

**Advance 1 — Two-Stage QKV Competition (Section 2.1 of the paper).**
The recurrent state is a fixed matrix $M_c \in \mathbb{R}^{K \times d}$
(default $K{=}128$, $d{=}$ backbone hidden size). It is updated once per
session boundary by two separate cross-attentions:

- **Stage 1, Extraction (Eq. 1; Eqs. 3–4 with $L_E{>}0$).** A learnable
  query bank $M_{\text{in}} \in \mathbb{R}^{K\times d}$ cross-attends over
  the just-completed session $C_t \in \mathbb{R}^{N_t\times d}$ to
  distill it into a $K$-slot candidate $M_{\text{new}}$. With
  $L_E > 0$ a Perceiver-style refinement stack lets the latent state
  re-query $C_t$ for $L_E$ extra rounds with a residual connection.
- **Stage 2, Judging (Eq. 2).** A *separate* cross-attention with
  learnable judging queries $M_{\text{judge}} \in \mathbb{R}^{K\times d}$
  attends over the concatenated pool $[\,M_c^{t-1}\,\Vert\,M_{\text{new}}\,]
  \in \mathbb{R}^{2K\times d}$. Because the softmax is computed *across the
  $2K$-row dimension*, old and new content compete for each output slot's
  attention mass — mundane sessions let old memory pass through; salient
  sessions overwrite. This is the **zero-sum forgetting defense**, and
  it is the load-bearing reason the project does not need RMT-style
  warm-up tricks.

**Advance 2 — Off-sequence depth-wise injection (Section 2.2).**
At inference into the *next* session, every position cross-attends $M_c$
once via Eq. 6:

\[
m^t = \mathrm{Softmax}\!\left(\frac{X W_Q^{\text{read}} (M_c W_K^{\text{read}})^\top}{\sqrt{d}}\right) M_c W_V^{\text{read}} \in \mathbb{R}^{S\times d}.
\]

The readout $m^t$ has the same shape as any attention layer's output,
which is what lets it slot into the residual stream cleanly. The paper
proposes either:

- **Block AttnRes routing pool** (Eqs. 7–10): $m^t$ is registered as a
  parallel foundational source $b_{-1}$ alongside the embedding source
  $b_0 = h_1$ and the prior block summaries $b_1, \dots, b_{n-1}$.
  Per-sublayer pseudo-queries $w_{n,i}$ govern a depth-wise softmax
  $\alpha_{v\to(n,i)}$ over the pool (the paper's preferred formulation).
- **ReZero-style gated injection** (the codebase's `simple_gate` mode
  — a strict simplification of the routing pool used as the toy /
  baseline variant):
  $h^{\text{pre}}_\ell = h^{\text{post}}_{\ell-1} + g_\ell \cdot m^t$.
  All per-sublayer scalar gates $g_\ell$ are zero-initialized so the
  augmented model is *exactly* equivalent to the bare backbone at step 0,
  while $W_V^{\text{read}}$ retains its default normal init so $m^t$ has
  non-zero magnitude and gate gradients flow non-trivially from step 1.

The `simple_gate` mode was introduced specifically because the
*delta-pool* Block AttnRes init (zero pseudo-queries, single negative
bias on the memory source) leaves all $N$ non-memory sources with
weight $\approx 1/N$ in the softmax. That uniform average over the
embedding, every prior block output, and the running intra-block
partial is **not** the same forward pass as the bare backbone —
which at sublayer $(n,i)$ takes the single accumulated state
$h_{n,i-1}$ as input — so the pretrained residual-stream conditioning
is disturbed at step 0 and several thousand steps of warm-up are
needed to recover it.

With the ReZero/`simple_gate` gate at zero, attaching MemRes to a
Qwen3 checkpoint is *exactly* equivalent to the bare backbone at step
0 (both forward and backward through the trunk), and the gradient
into the memory module itself is identically zero at step 0 because
$g_\ell = 0$ multiplies $\partial L / \partial h_\ell$ before it
reaches $W^{\text{read}}_{V}$. This is the "provably non-disruptive
at $t{=}0$" property, verified by
`paper_artifacts/eval/init_parity_test.json` to within $10^{-5}$ on
logits.

**A nuance worth flagging.** The Block AttnRes pool *can* in
principle be init-parity'd, but it requires two coupled changes the
brief originally elided: (i) the value pool must store *cumulative
hidden-state checkpoints* ($b_0=h_0$, $b_k=h_k$ at end of block $k$,
running partial $h_{n,i-1}$) instead of the original AttnRes
delta-only sources; under softmax (weights summing to $1$) the
delta-source pool literally cannot reconstruct the residual stream
$h = b_0 + \sum b_k + b_n^{i-1}$ from any one-hot, while the
cumulative-source pool already contains the residual stream as the
most-recent slot; and (ii) the router needs a strong positive bias
on the most-recent source so the softmax is one-hot at init.

Both options are implemented as separate canonical modes and tested
side-by-side:

| mode | $\max\lvert\Delta_{\text{logit}}\rvert$ vs bare Qwen3-0.6B (bf16) | verdict |
| --- | --- | --- |
| `simple_gate` (ReZero gate $g_\ell{=}0$), no memory | $0.000$ | bit-exact parity |
| `simple_gate` (ReZero gate $g_\ell{=}0$), memory attached | $0.000$ | bit-exact parity |
| `attention_base` (uniform softmax over delta sources), no memory | $34.5$ | massively perturbed |
| `attention_base` (uniform softmax over delta sources), memory | $34.4$ | massively perturbed |
| `attention_parity` (cumulative pool + `recent_bias = +32`), no memory | $0.000$ | bit-exact parity |
| `attention_parity` (cumulative pool + `recent_bias = +32`), memory | $0.000$ | bit-exact parity |

(Source: `paper_artifacts/eval/init_parity_test.json`, reproducible
via `python paper_tools/init_parity_test.py`.) The
`+32`/`-32` bias magnitudes are needed because per-step softmax
leakage compounds across all $2L = 56$ routed sublayers; at
`recent_bias` $=+16$ each off-source still carries $\sim e^{-16}/N
\approx 3\!\times\!10^{-7}$ of mass, which feeds back into the next
sublayer's input and accumulates to $\sim\!0.31$ in the final
logits. At $+32$ the off-source mass drops to $\sim e^{-32}\approx
1.3\!\times\!10^{-14}$, well below bf16 precision, and parity is
exact.

The trade-off is that `attention_parity` puts the model in a
saturated-softmax regime: the per-source pseudo-queries $w_{n,i}$
get effectively zero gradient at step 0, so the router can only
learn by first relaxing the bias. The `simple_gate` mode avoids the
warm-up problem entirely with one learnable scalar per sublayer, but
it sacrifices the depth-wise pool semantics that motivate Block
AttnRes in the first place — it is the **toy / baseline simplification**
we run to isolate the contribution of the routing pool from the
contribution of the memory module.  The `attention_parity` mode is
the **full implementation** we ultimately want to ship, and the
ablation table will compare them head-to-head.

(Naming back-compat: the old strings `residual` and `block_attnres`
are still accepted everywhere; `residual` translates to `simple_gate`,
`block_attnres` translates to `attention_parity` by default and to
`attention_base` when paired with `--no-block_attnres_parity_init`.)

### 1.3 Five non-negotiable design choices

From `memory_residuals_data/DESIGN_RATIONALE.md`. Each one defends against a
specific named failure mode in the prior literature; "simplifying" any of
them reproduces the named failure.

| Choice | Defends against |
| --- | --- |
| Separate extract + judge parameters (not one-stage write) | RMT narrow-channel collapse |
| Softmax across the $2K$ pool (not a learned scalar gate) | Block-Recurrent stability nightmare |
| Off-sequence routing via $v_0/b_{-1}$ (not prepend-to-sequence) | Transformer-XL / RMT cost explosion + lost-in-the-middle |
| Fixed $K$ at training and inference (not adaptive) | LRMT horizon overfit |
| Session-level write, token-level read (not per-token write) | Mamba/SSM commit-before-knowing |

### 1.4 The data plan

Two corpora, two stages — see
`memory_residuals_data/STAGE_PLAN.md` and `data_audit.md`:

- **Stage 1 (memory pretraining, ~485M tokens).** PG-19 (4,995 books cut
  at chapter boundaries, 218,588 sessions, ~470M tokens) plus 30
  high-continuity TV shows (2,189 episode sessions, ~16M tokens). Flat
  text, one session per JSONL row, sessions of the same document share a
  `book_id` / `show_id` and the trainer treats each id as one rolling
  memory chain.
- **Stage 2 (conversational SFT, ~15M tokens).** TV-only dialogue,
  same 30 shows, one episode per row with `turns: [{speaker, text}]`.
- **Eval-only:** MSC test split, LoCoMo-MC10 (10 long synthetic chats up
  to 35 sessions each), held-out TV. Synthetic chat (Persona-Chat,
  LoCoMo) is **eval-only** by user mandate to keep the training
  distribution clean.

### 1.5 Codebase layout

```
memory_residuals/
├── memory_residuals.{txt,pdf}     position paper (the source spec)
├── modeling_memres.py             MemoryBlock, MemoryReadout, MemoryGate,
│                                   BlockAttnResRouter, Qwen3MemRes{Model,ForCausalLM}
├── presets.py                     {qwen3-0.6b, 8b} × {small=L_E0, large=L_E4}
├── train_phase1.py                pair-based "warm-up" trainer (single-step compress)
├── train_chain.py                 recurrent chain TBPTT trainer (the real one)
├── paper_tools/
│   ├── pretokenize_chains.py / pretokenize_pairs.py
│   ├── locomo_to_chains.py
│   ├── eval_chain.py / eval_suite.py
│   ├── rag_baseline.py / rag_baseline_finetuned.py
│   ├── callback_probe.py
│   └── aggregate_results.py
├── paper_artifacts/
│   ├── chains/                    pre-tokenized chain corpora (.pt)
│   ├── eval/                      per-checkpoint eval JSONs
│   └── *.png                      gate-profile and probe plots
├── Runs/                        trained checkpoints, see §2 below
└── paper/memory_residuals_empirical.{tex,pdf}    in-progress write-up
```

---

## 2. What's been tried, in order, and where each attempt fails

The sandbox has reached the point where the **pair-based** Phase-0
warm-up trainer produces a publishable-looking checkpoint (`run3`), but
every attempt at the **chain-based recurrent training** that the
architecture actually requires has failed at least one of the load-bearing
diagnostics. This section is the honest accounting.

### 2.1 Diagnostics, in priority order

A run is judged by four numbers, in this order of importance:

| Symbol | What it measures | Pass criterion |
| --- | --- | --- |
| $\Delta_{\text{nm-m}} = \text{CE}_{\text{nomem}} - \text{CE}_{\text{mem}}$ | Aggregate "does memory help at all?" | $> 0.01$ nats on multi-session eval |
| $\Delta_{\text{sh-m}} = \text{CE}_{\text{shuffle}} - \text{CE}_{\text{mem}}$ | History specificity ("this chain's" vs "any chain's" memory) | $> 0$, monotonic in training |
| **Callback help ratio** | Memory help on callback tokens / on filler tokens | $\geq 1.5\times$ |
| Capture ratio = $\Delta_{\text{nm-m}}\,/\,\Delta_{\text{or-m}}$ | How much of the oracle (raw concat) headroom we captured | $\geq 0.3$ is interesting |

**Why $\Delta_{\text{sh-m}}$ is load-bearing.** Aggregate NLL improvements
are easy to fake by encoding genre/style ("this is PG-19 prose", "this is
a Breaking Bad transcript"). The shuffle test directly measures whether
memory has learned *episodic* content. This is the single test that has
killed every prior memory-augmented Transformer.

### 2.2 The runs

| Run dir | Trainer | Init | Data | Steps | Standalone eval result | Verdict |
| --- | --- | --- | --- | ---: | --- | --- |
| `Runs/run3_qwen3-0.6b-large/` | `train_phase1.py` (pair) | bare Qwen3-0.6B | PG-19+TV pairs (h=1024, c=512) | 8000 | $\Delta_{\text{nm-m}}{=}{+}0.026$, $\Delta_{\text{sh-m}}{=}{+}0.029$, callback ratio **1.77×** on **pair** eval ($n{=}256$); but on long-horizon **chain** eval explodes to $\text{CE}_{\text{mem}}{=}8.7$ vs $\text{CE}_{\text{nomem}}{=}2.5$ | Looks great on its own training distribution, **catastrophic OOD** to the actual chain regime |
| `Runs/chain2_qwen3-0.6b-large/` (deleted; numbers in `chain2_eval_pg19_locomo.json`) | `train_chain.py` warm-started from `run3`, judge RMSNorm added | warm | PG-19+TV chains, $k{=}4$ | 3000 | $\Delta_{\text{sh-m}}{=}{-}0.014$ PG-19, ${-}0.012$ LoCoMo | Stable but **PITFALLS §3 shortcut-learning failure** — memory became style-only |
| `Runs/chain_fresh1/` | fresh, no warm start | bare | PG-19+TV chains, $k{=}8$ | 5000 | $\Delta_{\text{nm-m}}{=}{+}0.008$ PG-19, ${-}0.021$ LoCoMo; $\Delta_{\text{sh-m}}{=}{-}0.036$ PG-19, ${-}0.016$ LoCoMo | Stable at 30+ session unrolls but **shuffle goes negative** — same shortcut-learning failure |
| `Runs/chain_tv1/`, `chain_tv2/` | TV-only | fresh | TV chains only, $k{=}8$ | 6000 | LoCoMo $\Delta_{\text{sh-m}}{=}{+}0.011, {+}0.002$ at step 500, then **overfits** | Domain-shift ablation only |
| `Runs/chain_neg1/` (latest) | `train_chain.py` + **negative-chain contrastive** ($\lambda{=}0.5$, margin $0.05$) | fresh | PG-19+TV chains, $k{=}8$ | 1000 | In-trainer (n=60): $\Delta_{\text{nm-m}}{\approx}0$, $\Delta_{\text{sh-m}}{=}{+}0.014$ — channel-collapse on aggregate, marginal shuffle gap | **Most promising recipe so far**, but unfinished; see open question #1 below |

### 2.3 The qualitative failure mode and what the data is telling us

There are two distinct failure surfaces and the runs are exhibiting both:

**Failure surface A — channel collapse (PITFALLS §1).**
After training, $\Delta_{\text{nm-m}} \approx 0$ on a rigorous standalone
eval. The depth-wise gates have effectively learned $g_\ell \approx 0$
everywhere; the model has fallen back to the bare backbone. This is
visible in `chain_neg1/best/eval_metrics.json` where
$\Delta_{\text{nm-m}}{=}{-}0.0003$.

**Failure surface B — shortcut learning (PITFALLS §3).**
Memory does help aggregate NLL by a small amount, but it helps *because*
it has learned style cues that lower NLL on every token uniformly. The
shuffle test then comes back negative or zero, because injecting another
PG-19 book's memory still encodes "this is PG-19 prose". This is what
killed `chain2` ($\Delta_{\text{sh-m}}{=}{-}0.014$) and `chain_fresh1`
($\Delta_{\text{sh-m}}{=}{-}0.036$).

**Why the pair trainer ("run3") didn't show either failure but the
chain trainer does.** The pair trainer concatenates four sessions into
a single `history` field and compresses it once per step — there is no
recurrent judge stack to optimize, no cross-window gradient signal, and
no horizon overfit. It is teaching the *readout* to be useful, not the
*recurrent competition*. The reason `run3` looks good on the pair eval
but explodes on the chain eval is that its $M_c$ has never been carried
across more than one judge step, so $\|M_c\|_F$ drifts unboundedly when
we do carry it; the addition of `judge_norm` in `chain2` fixes the
explosion but does not fix the upstream design problem of which the
explosion was a symptom.

### 2.4 Concrete metrics summary

PG-19 validation, 47 chains, 188 score positions, full prefix unroll:

| Run | $\text{CE}_{\text{mem}}$ | $\text{CE}_{\text{nomem}}$ | $\text{CE}_{\text{shuffle}}$ | $\text{CE}_{\text{oracle}}$ | $\Delta_{\text{nm-m}}$ | $\Delta_{\text{sh-m}}$ | Capture ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chain2` | 3.199 | 3.262 | 3.186 | 3.109 | $+0.063$ | $-0.014$ | 0.41 |
| `chain_fresh1` | 3.101 | 3.109 | 3.065 | 3.009 | $+0.008$ | $-0.036$ | 0.08 |
| `chain_tv1` | (PG-19 not run) | — | — | — | — | — | — |
| `chain_neg1` (in-trainer only) | 3.143 | 3.143 | 3.157 | 2.934 | $-0.0003$ | $+0.014$ | $\approx 0$ |

LoCoMo 10 chains, 40 score positions:

| Run | $\Delta_{\text{nm-m}}$ | $\Delta_{\text{sh-m}}$ | Notes |
| --- | ---: | ---: | --- |
| `chain2` | $+0.063$ | $-0.012$ | aggregate up, shuffle wrong sign |
| `chain_fresh1` | $-0.021$ | $-0.016$ | both wrong sign |
| `chain_tv1` | $+0.005$ | $+0.011$ | brief promising window at step 500 |
| `chain_tv2` | $+0.006$ | $+0.002$ | overfit by step 6000 |

### 2.5 What we *do* have that works (the pair-trained `run3` story)

To be clear about scope: the pair-trained Qwen3-0.6B-large (`run3`) is a
real, reproducible result on the pair task. From the empirical paper
draft (`paper/memory_residuals_empirical.tex`):

- $\Delta_{\text{nm-m}} = +0.026$ nats on $n{=}256$ PG-19 validation pairs.
- $\Delta_{\text{sh-m}} = +0.029$ nats — shuffled history is strictly
  worse than no memory; the memory channel encodes history-specific
  episodic information.
- **Callback help ratio 1.77×** (from `run3_callback_probe.json`):
  memory helps callback tokens 1.77× more than filler tokens, comfortably
  above the 1.5× threshold the diagnostic playbook calls out as the
  success criterion for "actually using memory for memory."
- Beats compute-matched RAG (top-1, 128-token prefix) by 0.279 nats and
  top-3-1024-token RAG over the bare backbone by 0.204 nats.
- Per-sublayer gate profile shows the model has learned non-uniform
  depth-wise routing (deepest sublayers $g_\ell \approx 0.12$, mid
  sublayers slightly negative).

This is a real demonstration that the architecture *can* learn semantic
memory under the right training protocol — it just hasn't yet been
shown to do so under the harder, more realistic chain-recurrent
protocol, which is what makes the architecture worth publishing.

---

## 3. Diagnosis: why the chain trainer keeps failing

Triangulating across the runs, the codebase, and the PITFALLS playbook:

1. **Gradient signal vs. cost asymmetry.** The pair trainer gets one
   judge step per backward pass; the chain trainer gets $k$ (window
   size). Each judge step is a softmax over a $2K$ pool, which
   attenuates gradient magnitude at every recurrence depth. With $k{=}8$
   and bf16 + gradient checkpointing, the gradient that reaches
   $M_{\text{in}}$ from the loss on session 8 is several orders of
   magnitude smaller than the gradient that reaches it from the loss on
   session 8 itself. The optimizer therefore learns "what to encode in
   $M_c$ so that *the same session's* readout is useful" before it
   learns "what to encode so that the *next* session's readout is
   useful" — and the former is exactly the style-only shortcut.

2. **The burn-in problem.** Eval chains are 20+ sessions long; training
   TBPTT only sees $k{=}8$. With `--burn_in_max=12` the trainer runs
   the prefix in `no_grad`, but those $M_c$ states are *out of
   distribution* for the readout (they were produced under a different
   gradient regime), and they cause loss spikes around step 5 of the
   gradient-tracked window. The default has been reverted to
   `burn_in_max=0` to dodge the spikes, which means the model only ever
   sees $M_c$ states from the first few judge steps. **This is exactly
   the LRMT horizon-overfit failure mode (PITFALLS §5).**

3. **Regularizer interactions.** Memory dropout ($p_{\text{mem}}{=}0.10$)
   and context dropout ($p_{\text{ctx}}{=}0.30$) are both on by
   default. With the small effective batch size we use ($B{\times}\text{grad\_accum}{=}4$),
   on any given step there is a ~38% chance that the only signal the
   model gets is from a degraded forward pass. This makes it harder, not
   easier, to learn the recurrent competition. The pair trainer doesn't
   have this problem because it gets one clean forward per step.

4. **The negative-chain auxiliary loss is doing the right thing on the
   wrong target.** `chain_neg1` is the first attempt to directly
   penalize shortcut learning by adding a contrastive loss against
   `loss_match - loss_shuffle + margin`. It moves $\Delta_{\text{sh-m}}$
   in the right direction (positive) — but it does so by *suppressing
   $\Delta_{\text{nm-m}}$* (driving aggregate memory help toward zero).
   The model finds it easier to "forget how to use memory at all" than
   to "use memory in a chain-specific way." Reaching both diagnostics
   positive simultaneously is going to require either (a) a much larger
   effective batch so the contrastive gradient isn't fighting the
   per-step gradient noise, or (b) a curriculum where Phase A learns
   `Δ_nm-m > 0` first (no contrastive) and Phase B adds the
   contrastive only after `Δ_nm-m` is robust.

5. **Backbone LR is probably too low.** `lr_backbone=3e-6` keeps the
   pretrained Qwen3 weights mostly frozen. This was intentional to
   preserve the bare-backbone parity at $t{=}0$, but it also means the
   *backbone* never adapts to actually consume the memory readout — only
   the gates and the readout/extract/judge parameters do. The memory
   contribution at any depth is bottlenecked by what the unmodified
   backbone happens to do with $h + g_\ell \cdot m^t$ when
   $g_\ell \neq 0$. The right move is a longer warmup of just the
   memres params, then a careful unfreezing of the late backbone layers.

6. **Data is fine; it's the recipe that isn't.** PG-19 + TV at 485M
   tokens is the same scale that has trained successful recurrent memory
   models in the past. We are not data-bound. We are bound by how many
   *reliable* gradient signals we get into the recurrent parameters.

---

## 4. Compute-targeted plans forward

Two concrete recipes, sized to the two compute envelopes you specified.
Both presume the same starting point: the existing PG-19 + TV chain
corpus (already pre-tokenized in `paper_artifacts/chains/`), the
`Qwen3MemRes*` modeling code unchanged, and `train_chain.py` as the
foundation (with the modifications listed below).

### 4.1 The 2× H100 plan (single-node, ~2–7 days wall-clock)

This is the realistic plan for *finishing the existing paper draft*, not
for adding the 8B-class scale-up. Goal: land
**$\Delta_{\text{nm-m}} > 0.01$ AND $\Delta_{\text{sh-m}} > 0$ AND
callback ratio $\geq 1.5\times$ on rigorous standalone eval, on PG-19 and
LoCoMo, simultaneously.**

**Hardware budget.** $2 \times $ H100 80GB. Effective per-step memory at
bf16 + gradient checkpointing on Qwen3-0.6B: ~30 GB peak with
`window_k=8`, batch=4 — fits comfortably with headroom for $K{=}256$
ablation if needed.

**Recipe.**

1. **Two-phase curriculum, ~36 hours total.**
   - **Phase A (12 h, 6000 steps).** Vanilla chain TBPTT with no
     contrastive loss. Goal: drive $\Delta_{\text{nm-m}}$ positive on
     standalone eval. Settings:
     - `--preset qwen3-0.6b-large`, `--window_k 8`, `--session_len 512`
     - `--batch_size 8 --grad_accum 4` (eff. batch 32 across 2 GPUs)
     - `--lr 3e-4 --lr_backbone 3e-5` (10× higher than current default
       to let the backbone *learn to consume the readout*)
     - `--memory_dropout 0.0 --context_dropout 0.0` for the first 1500
       steps (no regularizer churn while the gates lift off zero), then
       linearly ramp to `0.10 / 0.30` by step 4000.
     - `--burn_in_max 0` (avoid the OOD burn-in problem — see §3.2;
       fix it later via the long-horizon protocol below)
     - `--gradient_checkpointing`
     - Save best by `Δ_nm-m` (not `ce_mem`) — current trainer saves by
       `ce_mem` which can be minimized by overfitting; needs a one-line
       change in `Trainer._save / fit`.

   - **Phase B (24 h, 12000 steps).** Add the negative-chain contrastive
     loss, *but* with a curriculum on the contrastive weight:
     - Warm-start Phase A's best checkpoint via `--init_from`.
     - `--neg_chain_weight 0.05 --neg_chain_margin 0.02` for steps
       0–2000 (very gentle), then `0.2 / 0.05` for 2000–6000, then
       `0.5 / 0.10` for 6000–12000.
     - Same LR / batch as Phase A. Save best by
       `Δ_nm-m + 2 * Δ_sh-m` (a composite that penalizes the shortcut
       failure even when aggregate memory help is high).

2. **Long-horizon training protocol (PITFALLS §5).** Once Phase B is
   stable, add a second outer loop: run the chain in `no_grad` forward
   for the first $20$ sessions, then enable gradient tracking only on
   the last $k{=}8$. This makes the model see $M_c$ states from
   realistic recurrence depths *with the gradient-tracked* judge step
   downstream. Implementation: change `train_chain.py:_train_step` so
   `burn_in_max` is *resampled per chain* uniformly in
   $\{0, 4, 8, 12, 16, 20\}$, with the no-grad burn-in run at the
   *current* parameters (not detached early). Cost: roughly $1.5\times$
   the per-step time, so plan ~12 h of additional training after
   Phase B.

3. **Eval budget (~4 h).** Run the full standalone eval suite at every
   500 steps:
   - `paper_tools/eval_chain.py` on
     `paper_artifacts/chains/stage1_validation_s512.pt` and
     `paper_artifacts/chains/locomo_s512.pt` with `--score_window 4
     --oracle_window 4`.
   - `paper_tools/callback_probe.py` once at end of Phase A, end of
     Phase B, and end of long-horizon protocol.
   - `paper_tools/aggregate_results.py` to roll all eval JSONs into the
     paper's tables.

4. **Diagnostic gates (kill-the-run criteria).**
   - End of Phase A, step 6000: if $\Delta_{\text{nm-m}} < 0.005$ on
     PG-19 validation, the channel has collapsed during regularizer
     ramp-up. Roll back to step 1500, halve the regularizer ramp rate.
   - End of Phase B, step 6000 (= Phase B step 2000 with $\lambda{=}0.05$):
     if $\Delta_{\text{sh-m}}$ has not moved positive, the contrastive
     gradient is being washed out by per-step noise. Increase
     `--batch_size` to 16 (uses more memory; consider K=64 fallback)
     and rerun from Phase A best.
   - End of long-horizon protocol: if PG-19 and LoCoMo $\Delta_{\text{sh-m}}$
     have *both* gone positive but callback ratio is still under 1.5×,
     add aux loss (C) from PITFALLS §"auxiliary loss spectrum"
     (memory-only forward pass on a 64-token slice with $\lambda{=}0.1$).
     Cost: another ~6 h.

5. **Outputs delivered.** A best checkpoint at
   `Runs/chain_2h100_final/best/`, complete eval JSONs in
   `paper_artifacts/eval/` (replace the current `chain_*` ones), updated
   tables in `paper/memory_residuals_empirical.tex`, and the gate-profile
   PNG. Total wall-clock: 2–7 days depending on how many diagnostic
   rollbacks are needed; budget 4 days.

**What this plan deliberately does *not* do.** It does not attempt the
8B-class scale-up. It does not attempt Phase 2 conversational SFT. It
does not attempt MQAR / phonebook benchmarks. The 2× H100 envelope is
not large enough to do all of these well; it is large enough to land
the headline chain result.

### 4.2 The 20× A100 plan (multi-node, 168 GPU-hours = 24×7)

This is the plan if you have a substantially larger compute envelope and
want to *both* finish the chain result *and* scale to 8B *and* run the
ablation matrix. 20× A100 80GB = 1600 GB of aggregate HBM, easily enough
for FSDP-sharded Qwen3-8B at sequence length 512 with $K{=}128$ memory.

**Topology.** 20 GPUs across $\geq 3$ nodes (assuming 8-GPU servers,
this is a 3-node 4+8+8 layout; if 4-GPU servers, this is 5 nodes).
NCCL with InfiniBand if available; otherwise 100GbE Ethernet is
sufficient at the 8B scale because intra-node NVLink dominates the
collective cost.

**Time budget.** 168 GPU-hours $\times$ 20 GPUs = 3360 GPU-hours,
treating wall-clock as a hard cap.

**Recipe — three parallel workstreams.**

**Workstream 1 — Finish the 0.6B chain result (5 GPUs × 24 h = 120
GPU-hours).** Same as the 2× H100 plan above, but on 4 GPUs of the 5
allocated to this stream (one GPU dedicated to running standalone eval
asynchronously every 500 steps so eval doesn't block training). Runs
to completion in ~24 wall-clock hours instead of ~4 days because the
larger effective batch (now 64) gives cleaner gradients and converges
faster.

**Workstream 2 — Ablation matrix on 0.6B (4 GPUs × 24 h × 7 = 672
GPU-hours, parallelized as 8 sequential 24-hour 4-GPU runs over 7
calendar days... or 4-way parallel into 28 GPU-hours/run if scheduling
allows).** From `TRAINING_PLAYBOOK.md` "ablation matrix to run before
submission":

1. $K \in \{32, 64, 128, 256\}$ memory-budget sweep (4 runs).
2. Context dropout on/off (1 run).
3. Memory dropout on/off (1 run).
4. Zero-init $W_V^{\text{read}}$ on/off (1 run).
5. $L_E \in \{0, 2, 4, 8\}$ extraction-depth sweep (4 runs; 0 and 4
   already baked into the presets).
6. TBPTT window $k \in \{1, 2, 4, 8, 16\}$ (5 runs).
7. `memres_mode` $\in \{$`simple_gate`, `attention_base`, `attention_parity`$\}$ comparison (3 runs).
8. Aux loss spectrum: A only / A+B / A+B+C / A+B+C+D (4 runs).

That's $4+1+1+1+4+5+1+4 = 21$ ablation runs at the 0.6B scale. At ~5
GPU-hours per run on 4 A100s (since the 0.6B fits comfortably in
data-parallel across 4 cards), all 21 ablations finish in ~24 wall-clock
hours of each-runs-on-4-GPUs scheduling. Use the remaining capacity
to run each ablation twice with different seeds so we can report
uncertainty.

**Workstream 3 — 8B-class scale-up (11 GPUs × 168 h = 1848
GPU-hours).** This is the headline experiment we don't have yet.

1. **Phase 0 (24 h).** Pair-trainer warm-up on `qwen3-8b-large` using
   `train_phase1.py --shard_strategy fsdp_full --gradient_checkpointing`,
   8000 steps. Replicates the `run3_qwen3-0.6b-large` recipe at scale.
   Expected: $\Delta_{\text{nm-m}} \in [0.02, 0.04]$ on pairs, callback
   ratio $\geq 1.5\times$. The 8B backbone has more capacity to be
   useful for memory readout, so the gate profile should be sharper.
2. **Phase 1 (96 h).** Chain TBPTT recurrent training using
   `train_chain.py --shard_strategy fsdp_full --gradient_checkpointing
   --preset qwen3-8b-large`, 12000 steps. Apply the two-phase
   curriculum from §4.1. Expected wall-clock per step at the 8B scale
   on FSDP across 11 A100s: ~25 s, so 12000 steps ≈ 84 h. Eval
   asynchronously on 1 GPU.
3. **Phase 2 (24 h).** Conversational SFT on `stage2/` TV dialogue
   only, with 5–10% Stage-1 replay batches mixed in (PITFALLS §4 fix
   for catastrophic forgetting). Lower LR ($5 \times 10^{-5}$ peak),
   freeze $M_{\text{in}}$ and $M_{\text{judge}}$ entirely (per
   PITFALLS §4). 2000 steps, ~24 h.
4. **Eval (24 h).** Full standalone eval on PG-19 validation, LoCoMo,
   MSC-test, held-out TV. Compare against:
   - the 0.6B Workstream-1 final;
   - bare Qwen3-8B + RAG (top-3, 1024-token prefix);
   - bare Qwen3-8B + concat-oracle (4-session prefix);
   - the 0.6B + MemRes from Workstream 1.
   Plus run MQAR / phonebook (PLAYBOOK eval #3) and report associative-recall
   parity vs. attention-class baselines.

**What this delivers.** A complete empirical paper: 0.6B chain
result + 8B chain result + Phase-2 SFT + the full ablation matrix.
The paper currently in `paper/memory_residuals_empirical.tex` describes
roughly the 0.6B *pair* result; this plan upgrades it to the *chain*
result at *two scales* with a full ablation suite, which is the bar
NeurIPS / ICLR will actually want.

**Critical risks at this scale.**

- **NCCL hangs at 20-GPU scale.** Mitigate by setting
  `NCCL_SOCKET_IFNAME` explicitly per-node, pinning
  `NCCL_IB_DISABLE=0` if InfiniBand is present, and confirming
  `NCCL_DEBUG=INFO` startup banners agree on world size before
  proceeding past step 0.
- **FSDP + gradient checkpointing + Memory Residuals.** The custom
  forward in `Qwen3MemResModel.forward` calls
  `_gradient_checkpointing_func` on `attention_delta` and `mlp_delta`
  separately; FSDP wraps `Qwen3MemResDecoderLayer` automatically via
  `transformer_auto_wrap_policy`. This combination has been *tested in
  `train_phase1.py`* but **not** in `train_chain.py` because chain TBPTT
  introduces a `static_graph=True` requirement on DDP that conflicts
  with the default FSDP autograd hooks. Expect to spend ~8 GPU-hours
  early in Workstream 3 just getting the FSDP+TBPTT combination to step
  cleanly. If it doesn't, fall back to ZeRO-3 via `deepspeed` (needs
  ~1 day of porting; the modeling code is HF-compatible so the lift is
  small).
- **Long-horizon evaluation memory.** Eval at chain length $\geq 30$
  sessions on the 8B model needs ~60 GB of activations even with
  no_grad and recompute; one A100 is enough but only barely. Don't try
  to eval on the same GPU that's training.

### 4.3 What to drop in either plan if time runs short

In rough priority order (last to keep, first to drop):

1. **Keep:** the chain $\Delta_{\text{sh-m}}$ result at 0.6B on PG-19
   and LoCoMo. This is the headline; everything else is supporting.
2. **Keep:** the callback probe at 0.6B. Cheap (5 minutes), and the
   single most rhetorically-powerful number in the paper.
3. **Keep:** the 8B-scale Phase 1 result. Without scaling to ≥8B the
   paper is "small-model demo"; with it, it's "method that scales."
4. **Drop first:** Phase 2 SFT. Useful for product but not for the
   paper's central claim. Can be reported as "future work."
5. **Drop second:** the full ablation sweep. Pick the 5 most-likely-to-be-asked
   ones (zero-init $W_V^{\text{read}}$ on/off, context dropout
   on/off, memory dropout on/off, $L_E \in \{0, 4\}$, $K \in \{64, 128\}$)
   and skip the rest.
6. **Drop third:** MQAR / phonebook. Comparing to SSMs is a side bet;
   the central claim does not require it.
7. **Never drop:** `paper_artifacts/eval/init_parity_test.json`. The
   bare-backbone parity at $t{=}0$ is what makes the architecture
   distinctive; if we lose verification of that, reviewers will ask.

---

## 5. Open questions, in priority order for the paper

These are unresolved. None of them require new compute to *answer* — they
just require analysis of existing or near-future runs.

1. **Is `chain_neg1` actually salvageable, or is the contrastive loss
   permanently suppressing the aggregate channel?** Running it past 1000
   steps with the curriculum from §4.1 Phase B should answer this in
   ~18 wall-clock hours.
2. **Is the in-trainer-vs-standalone $\Delta_{\text{sh-m}}$ discrepancy
   methodology or substance?** The in-trainer eval uses 16 chains with
   a possibly-shorter shuffle prefix; the standalone eval uses 47
   chains with the full prefix. The standalone protocol is the one that
   should be reported, but the gap suggests the 16-chain in-trainer
   shuffle is artificially easy. Worth standardizing on the standalone
   protocol for all in-trainer eval too — needs a small refactor of
   `Trainer.evaluate` to load the held-out 47-chain set.
3. **What is the effective ceiling?** $\text{CE}_{\text{oracle}}$ is
   only ~0.20 nats below $\text{CE}_{\text{nomem}}$ on PG-19, so any
   compressed-memory scheme has at most that much to win. Reporting
   *capture ratio* ($\Delta_{\text{nm-m}} / \Delta_{\text{or-m}}$) is
   probably more honest than reporting raw $\Delta$s. PG-19 is also
   inherently low-callback (long prose, characters mentioned by name
   often); LoCoMo and MSC have higher oracle ceilings.
4. **Should we mix MSC v2 / Persona-Chat into training** to get more
   dialogue exposure during Phase 1? The user's standing rule has been
   "no synthetic" — but MSC is human-crowdsourced. Worth re-asking
   before a final 8B run; it would give more consistent
   $\Delta_{\text{sh-m}}$ on the LoCoMo / MSC eval surface.
5. **Long-horizon decay vs. degradation.** PITFALLS §5 flags that
   memory benefit may go from "helps" to "hurts" past some horizon.
   Eval at chain length 30 (LoCoMo conv-41 has 32) and report whether
   memory benefit decays smoothly or hits a cliff. This is the single
   architecturally-most-serious failure mode and we don't yet have data
   for it.

---

## 6. TL;DR for a collaborator deciding whether to invest

**What's working.** A pair-trained `qwen3-0.6b-large` checkpoint
(`Runs/run3_qwen3-0.6b-large/best/`) that satisfies all four
diagnostics on the pair eval, including the 1.77× callback ratio.
Architecture, data pipeline, eval scripts all exist and are reproducible.
The empirical paper draft (`paper/memory_residuals_empirical.tex`) is
~80% written around this checkpoint.

**What's not working.** The chain-recurrent trainer
(`train_chain.py`) has not yet produced a checkpoint that passes both
$\Delta_{\text{nm-m}} > 0$ and $\Delta_{\text{sh-m}} > 0$ on rigorous
standalone eval. The latest attempt (`chain_neg1`) trades aggregate
memory help for shuffle gap — the wrong tradeoff. The chain regime is
the regime the architecture is *designed* for; the paper without a
chain result is a Phase 0 demo at best.

**What it would take to fix.** §4.1 above: ~4 wall-clock days on 2×
H100, with a two-phase curriculum (Phase A for $\Delta_{\text{nm-m}}$,
Phase B for $\Delta_{\text{sh-m}}$) and an explicit long-horizon training
protocol to defuse the LRMT failure mode. ~80% confidence this lands
the chain result; ~20% confidence it reveals a deeper architectural
issue and we need to revisit one of the five non-negotiable choices —
most likely the per-token read step's interaction with the off-sequence
$v_0$ at long horizons.

**What it would take to ship the full paper.** §4.2: 168 GPU-hours on
20× A100, three parallel workstreams (0.6B chain, 0.6B ablations, 8B
scale-up + Phase 2 SFT). This is the version that is publishable at a
top venue.

The architecture is not obviously broken; the diagnostics are
well-instrumented; the data pipeline is in place. The thing standing
between the current state and a publishable result is a focused 1–2
weeks of training-loop iteration with the fixes from §3 and the
curriculum from §4.

---

# Part II — Project summary (was SUMMARY.md)


A faithful implementation, training pipeline, and empirical study of the
*Memory Residuals* architecture from `memory_residuals.pdf`: a Transformer
modification that maintains a fixed-size recurrent memory matrix
$M_c \in \mathbb{R}^{K \times d}$ and injects a per-position memory readout
$m^t$ into every sublayer through a learned ReZero-style gate, without
breaking the pretrained backbone's residual stream.

This is the working repository.  See `paper/memory_residuals_empirical.tex`
for the in-progress write-up.

## File map (post-cleanup)

```
memory_residuals/
├── SUMMARY.md                       <- this file
├── README.md                        <- original architecture-only README
├── memory_residuals.pdf / .txt      <- the source paper
├── modeling_memres.py               <- model: MemoryBlock, MemoryReadout,
│                                       MemoryGate, Qwen3MemRes{Model,ForCausalLM}
├── presets.py                       <- {qwen3-0.6b-small, qwen3-0.6b-large,
│                                       qwen3-8b-small, qwen3-8b-large}
├── train_phase1.py                  <- pair-based "warm-up" trainer
│                                       (single-step compress + score)
├── train_chain.py                   <- recurrent chain TBPTT trainer
│                                       (the real one — exercises M_c
│                                       across consecutive sessions)
├── paper_tools/
│   ├── pretokenize_chains.py        <- chain JSONL -> packed (N, S) tensor
│   ├── pretokenize_pairs.py         <- pair JSONL -> packed history/current
│   ├── locomo_to_chains.py          <- LoCoMo JSON -> chain JSONL
│   ├── eval_chain.py                <- chain-aware eval: mem/nomem/shuffle/oracle
│   ├── eval_suite.py                <- pair eval (history/current pairs)
│   ├── rag_baseline.py              <- bare-backbone + dense retrieval
│   ├── rag_baseline_finetuned.py    <- finetuned-backbone + dense retrieval
│   ├── callback_probe.py            <- per-callback NLL probe (PITFALLS §3)
│   ├── audit_data.py                <- corpus token audit
│   ├── prepare_pairs.py             <- chain -> pair window converter
│   └── aggregate_results.py         <- collect eval JSONs into a CSV/MD table
├── paper/
│   └── memory_residuals_empirical.{tex,pdf}
├── paper_artifacts/
│   ├── chains/                      <- pre-tokenised chain corpora
│   │   ├── stage1_train_s512.pt     <- PG-19 + 30 TV shows, ~89K sessions
│   │   ├── stage1_validation_s512.pt
│   │   ├── stage1_test_s512.pt
│   │   ├── tv_train_s512.pt         <- TV-only (30 chains)
│   │   └── locomo_s512.pt           <- LoCoMo (10 chains, eval-only)
│   ├── eval/                        <- per-checkpoint eval JSONs
│   ├── locomo_chains/               <- LoCoMo session-format JSONLs
│   └── *.png                        <- gate-profile and probe plots
├── Runs/
│   ├── run3_qwen3-0.6b-large/       <- pair "warm-up" baseline (8000 steps)
│   ├── chain_fresh1/                <- fresh chain TBPTT (5000 steps,
│   │                                   no contrastive)
│   ├── chain_tv1/, chain_tv2/       <- TV-only chain training (overfit
│   │                                   then plateau, kept for ablation)
│   └── chain_neg1/                  <- chain TBPTT + negative-chain
│                                       contrastive loss (currently the
│                                       most promising recipe)
└── legacy/                          <- earlier reference implementations,
                                       not used by the current pipeline
```

## What's been tried, in order

| Run | Trainer | Data | Steps | Memory eval (rigorous) | Notes |
|---|---|---|---:|---|---|
| `run3_qwen3-0.6b-large` | `train_phase1.py` (pair) | PG-19+TV pairs | 8000 | $\Delta_{nm-m}=+0.026$, $\Delta_{sh-m}=+0.029$ on **pair** eval (n=256); explodes to mem CE=8.7 on long-horizon **chain** eval | Pair-trained warm-up.  Two architectural fixes landed here: zero-init $g_\ell$ (gate) and `memres_mode=simple_gate` (legacy `residual`) to keep the backbone residual stream intact |
| `chain2_qwen3-0.6b-large` | `train_chain.py` warm-started from `run3` | PG-19+TV chains, k=4 | 3000 | $\Delta_{sh-m}=-0.014$ PG-19, $-0.012$ LoCoMo | First chain run with `judge_norm` RMSNorm.  Stable but PITFALLS §3 shortcut-learning failure — memory had become style-only |
| `chain_fresh1` | fresh init, no warm-start | PG-19+TV chains, k=8 | 5000 | $\Delta_{sh-m}=-0.036$ PG-19, $-0.016$ LoCoMo (standalone); $+0.020$ PG-19 in-trainer eval | Stable at 30+ session unrolls.  In-trainer Δ_sh-m drifted positive but the rigorous standalone eval revealed the in-trainer protocol was overstating specificity |
| `chain_tv1`, `chain_tv2` | TV-only training | TV chains, k=8 | 6000 | LoCoMo eval positive at step 500 ($\Delta_{sh-m}=+0.011$, $+0.002$), then overfit | Useful as a domain-shift ablation; not a final result |
| `chain_neg1` (currently training) | chain TBPTT + **negative-chain contrastive** loss ($\lambda=0.5$, margin 0.05) | PG-19+TV chains, k=8 | 5000 (in progress) | step 1000 in-trainer $\Delta_{sh-m}=+0.0114$ | Implements the PITFALLS §3 prescription: explicit gradient against shortcut learning |

## Architecture invariants (verified in code)

1. **Init = bare backbone.**  All `MemoryGate.gate[i]=0` at construction
   and after `_init_weights`, so $h^{\mathrm{pre}}_\ell = h^{\mathrm{post}}_{\ell-1}$
   and the augmented model produces bit-identical logits to a vanilla
   Qwen3-0.6B forward pass.  Verified: `paper_artifacts/eval/init_parity_test.json`.
2. **Two-stage write.**  `MemoryBlock.extract` (Eq. 1, 3-4) and
   `MemoryBlock.judge` (Eq. 2) have separate parameters, separate
   queries ($M_{\mathrm{in}}$, $M_{\mathrm{judge}}$), and separate
   cross-attention heads.
3. **Zero-sum forgetting.**  The judge softmax is across the $2K$-row
   pool $[M_c^{t-1}; M_{\mathrm{new}}]$, not within slot.
4. **Stable recurrence at long horizons.**  `MemoryBlock.judge_norm` is
   an RMSNorm applied after the judge cross-attention; without it
   $\|M_c\|_F$ drifts after ~10 unrolls and eval CE explodes.
5. **Off-sequence routing.**  $M_c$ never enters the token sequence.
   Cost is $O(SK)$ for the readout + $O(L)$ for the depth-wise gates per
   token, vs $O((S+K)^2)$ for a sequence-prepended baseline.

## Reproducing the headline experiment

On a 2× H100 node with the existing data:

```bash
# 1. Pre-tokenise (already done; outputs land in paper_artifacts/chains/)
python paper_tools/pretokenize_chains.py \
  --in_dir ../memory_residuals_data/stage1/pg19/train \
  --in_dir ../memory_residuals_data/stage1/tv \
  --out_path paper_artifacts/chains/stage1_train_s512.pt \
  --session_len 512 --min_tokens 64 --min_sessions_per_chain 4 \
  --max_chains 2000 --workers 32

python paper_tools/pretokenize_chains.py \
  --in_dir ../memory_residuals_data/stage1/pg19/validation \
  --out_path paper_artifacts/chains/stage1_validation_s512.pt

python paper_tools/locomo_to_chains.py
python paper_tools/pretokenize_chains.py \
  --in_dir paper_artifacts/locomo_chains \
  --out_path paper_artifacts/chains/locomo_s512.pt

# 2. Train (single H100 is enough for Qwen3-0.6B-large)
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=0 \
python -u train_chain.py \
  --preset qwen3-0.6b-large \
  --window_k 8 --session_len 512 \
  --steps 5000 --batch_size 2 --grad_accum 2 \
  --warmup 200 --lr 5e-4 --lr_backbone 1e-5 \
  --memory_dropout 0.10 --context_dropout 0.30 \
  --neg_chain_weight 0.5 --neg_chain_margin 0.05 \
  --gradient_checkpointing \
  --out_dir Runs/chain_neg_repro

# 3. Rigorous eval on PG-19 validation + LoCoMo
python paper_tools/eval_chain.py \
  --model_path Runs/chain_neg_repro/best \
  --corpora paper_artifacts/chains/stage1_validation_s512.pt \
            paper_artifacts/chains/locomo_s512.pt \
  --names pg19_validation locomo \
  --score_window 4 --oracle_window 4 \
  --output paper_artifacts/eval/chain_neg_repro_eval.json
```

A 5000-step single-GPU run takes ~2 hours, eval takes ~1 minute.

## Open questions / future steps

In rough priority order for the paper:

1. **Land positive $\Delta_{sh-m}$ on rigorous standalone eval** for
   PG-19 *and* LoCoMo.  `chain_neg1` is the active attempt; in-trainer
   numbers are positive but standalone numbers are pending.

2. **Run the callback probe on the chain-trained checkpoints.**  We
   have probe results for `run3` (pair-trained) only; need them on
   `chain_neg1/best` once it converges.  Target: callback help ratio
   $> 1.5\times$ on PG-19 *and* on LoCoMo.

3. **8B-class run.**  `train_chain.py` accepts `--shard_strategy
   fsdp_full --gradient_checkpointing` and `--preset qwen3-8b-large`.
   Untried; expect ~1 GPU-day on 2× H100 with FSDP.

4. **Burn-in stability fix.**  The current `--burn_in_max=0`
   default skips burn-in because the gradient-tracked $k=8$ window
   already hits memory budgets in bf16 + gradient checkpointing.  The
   no-grad burn-in path is implemented but it produces $M_c$ states
   that are out-of-distribution for downstream readout (causes loss
   spikes at step 5).  Either keep burn-in gradient-tracked (i.e.
   bigger $k$) or warm-start from a chain-trained checkpoint.

5. **Long-horizon test.**  Eval at chain length $\geq 30$ sessions (LoCoMo
   conv-41 has 32) and report whether memory benefit decays vs
   degrades.  This is the LRMT failure mode — flagged in PITFALLS §5.

6. **Mix in MSC v2 / Persona-Chat** for training only if user re-permits
   crowd-sourced human-written multi-session chat.  Currently training
   uses **only** PG-19 (human-written prose) + 30 TV-show transcripts
   (human-written dialogue); LoCoMo and pair-MSC are eval-only.

## Things I'd ask reviewers about before submission

- The in-trainer-vs-standalone $\Delta_{sh-m}$ discrepancy (16-chain
  subset with possibly-shorter shuffle prefix vs 47-chain rigorous
  full-length) is a methodology subtlety that needs to be transparent
  in the paper.  We standardise on the standalone protocol everywhere
  in published numbers.
- The fundamental ceiling: oracle is only ~0.20 nats below no-memory on
  PG-19, so any compressed-memory scheme has at most that much to win.
  The interesting metric is the *capture ratio* (memory $\Delta$ as a
  fraction of oracle $\Delta$), not the raw $\Delta$.

## Removed / archived from the working tree

Anything not immediately needed for the active v6 recipe lives under
`archive/`. Reference papers (`memory_residuals.{pdf,txt}`,
`atn_residuals.pdf`) deliberately stay at the top level: agents
should be able to step back to the architectural spec without
spelunking. Inventory of the archive:

- `archive/Scripts/` — every KILLED v3/v4/v5/v6-attempt training
  script (`train_ablation_*`, `train_headline*`, `train_v6_lme_msc_gated_gh200.sh`),
  plus `sentinel_local_ablations.sh` and `cloud_chain_enqueue.sh`.
- `archive/paper_tools/` — superseded or one-shot tools:
  `rag_baseline*.py`, `audit_data.py`, `eval_suite.py`,
  `prepare_pairs.py`, `build_msc_chains.py`,
  `build_callback_pairs.py`, `build_longhorizon_chains.py`,
  `post_train_pipeline.sh` (hardcoded to v3 paths),
  `watchdog.sh`, `build_tables.py`, `eval_matrix.sh`,
  `notify_eval.sh`, `pull_overnight.sh`,
  `run_overnight_traces.sh`, `cloud_handoff.sh`.
- `archive/datasets/` — pre-v6 chain corpora (PG-19+TV `stage1_*`,
  MSC `msc_*` and `stage1_msc_*`, `tv_train_s512.pt`,
  `locomo_s512.pt`, the synthetic-passkey `archive_pg19_passkey/`)
  and the legacy pair JSONLs (`pairs/`, `pairs_eval/`,
  `locomo_chains/`).
- `archive/eval/` — eval JSONs from KILLED runs and pre-v3 small
  pilots (`run1_*`, `run2_*`, `run3_*`, `chain2_*`,
  `chain_fresh1_*`, `chain_tv*`, `chain_v2_abl_residual_mode_*`,
  `chain_v2_phaseA_trajectories.*`, `qwen3-0.6b-*.json`,
  `qwen3-8b-*.json`, `rag_qwen3-0.6b.json`).
- `archive/figures/` — pre-v6 plots (`callback_probe_chart.png`,
  `run2_*.png`, `run3_*.png`).
- `archive/agent_sessions/` — v5-era observer/writer/monitor notes
  (`agent_session_20260429_1011/`).
- `archive/docs/` — archived data audit (`data_audit.{json,md}`).

What's gitignored entirely: `Runs/`, `logs/`, `wandb/`,
`__pycache__/`, large .pt files (see `.gitignore`).

---

# Part III — Stage-1 6-day calendar (was docs/paper1_calendar.md)

This was the original day-by-day plan for paper 1 when it was still
being aimed at multi-session dialogue. It is preserved here as a
historical record of how the scope landed where it did. The active
paper-2 plan is in [`experiments/exp2_long_horizon_recipe/runs.md`](experiments/exp2_long_horizon_recipe/runs.md);
the active paper-1 status is summarised in
[`experiments/exp1_drop_in_primitive/README.md`](experiments/exp1_drop_in_primitive/README.md).

## Original Stage-1 claim (since narrowed)

> *On multi-session dialogue benchmarks (LoCoMo, MSC test, NIAH,
> RULER), at matched compute, Memory Residuals with the soft-parity
> Block AttnRes routing pool reach lower NLL and higher callback-EM
> than (a) the no-memory baseline, (b) BM25 retrieval, (c) dense
> MiniLM retrieval, and (d) a fine-tuned dense retriever.*

That claim has been narrowed to "sample-efficient on books" for
paper 1; the dialogue half is paper 2.

## Resources (2× H100 NVL local + 1× GH200 cloud)

| asset | location | hours / day | notes |
|---|---|---:|---|
| local GPU 0 | `cuda:0` (H100 NVL 96 GB) | 16 | residential power-down 8 h |
| local GPU 1 | `cuda:1` (H100 NVL 96 GB) | 16 | residential power-down 8 h |
| cloud GPU   | `192.222.50.225` (GH200 480 GB) | 24 | $2.49/h × 96 h ≈ $240, $260 reserve |

## Day-by-day skeleton

- **Day 0.** v3 softparity_full + v3 attentionbase_full extending v2
  to step 6000; cloud watchdog up; first eval JSONs landing in
  `paper_artifacts/eval/`. Decision triggers: adopt v3 softparity if
  Δ_sh-m > +0.020 on PG-19 standalone; keep attentionbase row of
  ablation table if Δ_sh-m > +0.010.
- **Day 1.** Mixed-corpus softparity (PG-19 40% + TV 20% + MSC 40%)
  on local GPU 0; mixed-corpus simple_gate baseline on local GPU 1;
  cloud runs NIAH on v3 best ckpts and pre-fetches RAG baseline
  models. Hard blocker: build `paper_artifacts/chains/mixed_train_s512.pt`
  via `paper_tools/build_msc_chains.py` first.
- **Day 2.** Standalone eval on the day-1 ckpts (PG-19 val/test,
  LoCoMo, MSC test, TV held-out); cloud runs RAG baselines (BM25,
  MiniLM, Contriever, MiniLM-FT) at matched FLOPs and downloads the
  RULER S=4k subset. Decision: if E1 LoCoMo Δ_sh-m ≥ E2 Δ_sh-m + 1 SE,
  the headline holds; otherwise reframe as paper 2.
- **Day 3.** Local GPU 0 runs the counterfactual sensitivity probe
  (alter session t-k, measure ΔNLL on session t for k ∈ {1, 5, 10})
  + per-depth routing trace; local GPU 1 runs E3 mixed-corpus
  `attention_base` to fill the ablation table; cloud runs RULER and
  begins LongBench-en.
- **Day 4.** Local GPU 0 runs E4, the hidden-state extraction
  ablation (consume `hidden_states[14]` instead of
  `embed_tokens(input_ids)` in `compress_session`); cloud finishes
  the RAG / RULER / LongBench cells and runs callback EM on
  generation outputs.
- **Day 5.** Standalone eval on E4; ablation table fill; cloud
  released. Save the remaining $260 of cloud budget.

The calendar was overtaken by the v4→v5→v6 dead-end ladder
(documented in [`experiments/exp2_long_horizon_recipe/README.md`](experiments/exp2_long_horizon_recipe/README.md)
"What was wrong before"). The current state is: paper 1 is dormant
as a self-contained primitive paper; paper 2 owns the active
research thread.

---

# Part IV — Historical run ledger (v2 → v6, was runs.md)

The `runs.md` ledger now contains only the active v7 cells (the
v6 entries were superseded by v7 on 2026-04-29 ~20:33 UTC). The
v2/v3/v4/v5/v6 entries — all KILLED or superseded — are preserved
here so the v5 → v6 → v7 narrative is auditable.

## 2026-04-29: v6 → v7 transition and the simple_gate finding

After two failed v5 → v6 pivots that attacked only the writer side
(LME corpus, callback supervision, gated update, callback-window
bias), the failure analysis on the v3 standalone routing trace
(`paper_artifacts/eval/routing_v3sp_*.json`) located the binding
constraint on the *router* side: α_mem ≈ 4.7e-4 averaged across 55
sublayers AND α_mem on `mem` runs essentially identical to α_mem on
`shuffle` runs at every sublayer. The router was content-blind and
saturated against memory.

The v7 P0 cells launched 2026-04-29 ~20:40 UTC pivoted the recipe
on **two new axes simultaneously**:

1. **Compression curriculum P0**: every training window is
   `[evidence_session, callback_session]` (window_k=2, non-contiguous
   sampling via new `--curriculum_evidence_bias` flag). Shortest
   possible credit-assignment path between an early-session fact and
   the callback that depends on it. Implemented in
   `train_chain.py: ChainCorpus.chain_curriculum_window` +
   `ChainSampler.sample_window` curriculum branch.
2. **Routing-mode A/B**: two `attention_parity` cells with different
   bias settings (mem=−2 / mem=−4) plus one `simple_gate` cell on the
   GH200 testing whether the depth-wise softmax pool is the
   structural problem.

**Result**: simple_gate produced non-zero `gate_max` within 20 steps
(0.0004 → 0.0074 by step 400, monotonically growing) where every
`attention_parity` configuration in the v3-v7 lineage has stayed
pinned at 0.0000 indefinitely. This is the empirical confirmation of
the routing-side diagnosis. The paper's "block AttnRes routing pool"
section needs to be rewritten — `attention_parity` is the structural
problem, not the feature, on the chain trainer.

**Caveat**: pure-P0 curriculum (`curriculum_evidence_bias=1.0`)
introduced a separate train/eval distribution mismatch that produced
strongly negative Δ_nm-m on the standard eval (Δ_nm-m=−0.147 on
V3BIAS at step 1200). Memory is *harmful* on the eval distribution
because the readout learned to use M_c built from one fresh evidence
session and the eval M_c is built sequentially through 40+ sessions.
Curriculum mix needs to be redesigned for v8 (mixed-bias 0.5,
phase-aligned eval). See `experiments/exp2_long_horizon_recipe/runs.md`
"v7 P0 results so far" for the full diagnosis and v8 plan.

**Watchdog patch landed alongside v7**: `paper_tools/cloud_watchdog/watchdog.sh`
now has a `gpu_is_busy()` precheck (defers spec launch when another
CUDA process holds > `GPU_BUSY_MIB` MiB on the target GPU; default
2048). Fixes the silent first-step OOM that killed PURIST on
2026-04-29 18:30 UTC when the watchdog co-launched it onto a busy
GH200.

## v6 long-horizon runs (KILLED 2026-04-29 ~20:33 UTC and ~21:03 UTC)

Three v6 cells launched 2026-04-29 17:55-18:08 UTC, all KILLED in
favour of the v7 pivot.

### A — `chain_v6_lme_gated_callback`

- **Status:** KILLED 2026-04-29 ~20:33 UTC at step ~1410 (PID 37321
  SIGTERM). Δ_sh-m regressed to negative at steps 1200/1400 after
  flatlining around +0.0005 / step 1000.
- **Machine:** local H100 NVL, GPU 0.
- **Final eval trajectory:** Δ_sh-m peaked at +0.0005 (step 1000)
  then went −0.0004 (step 1200) and −0.0001 (step 1400). gate_max
  stayed 0.0000 throughout. Detailed table in `runs.md` (v6
  KILLED section).
- **Verdict:** four-axis writer pivot (corpus, gated update,
  callback loss, callback-window bias) did not open the router.
- **Superseded by:** v7 P0 cells (router-side pivot via simple_gate
  ablation + compression curriculum).

### B — `chain_v6_lme_competitive_callback` (architecture A/B for the writer)

- **Status:** KILLED 2026-04-29 ~20:33 UTC at step ~1400 (PID 38226
  SIGTERM). Same gate_max=0 / Δ_sh-m ≈ 0 pattern as cell A.
- **Machine:** local H100 NVL, GPU 1.
- **Single-knob diff vs v6 A:** `--memres_update_mode competitive`
  vs `gated`.
- **Verdict:** A/B against gated is uninterpretable when both cells
  fail at the router. Killed for information yield.

### C — `chain_v6_lme_gated_callback_w12` (window-depth ablation, GH200)

- **Status:** KILLED 2026-04-29 21:03 UTC at step ~420 of 8000.
  Same gate_max=0 / Δ_sh-m ≈ 0 / Δ_nm-m ≈ 0 pattern as the local
  cells, just at slower step rate (2.3k tok/s vs 5.3k locally
  because TBPTT is 1.5× deeper).
- **Machine:** GH200, GPU 0.
- **Single-knob diff vs v6 A:** `--window_k 12` (vs 8);
  `--burn_in_max 32` (vs 24); `--eval_window 12` (vs 8).
- **Verdict:** confirmed the v3/v6 router-collapse pattern is
  depth-axis-independent — deeper TBPTT alone does not rescue v6.
- **Replaced by:** `chain_v7_p0_simplegate` (architectural ablation
  on the same GH200 GPU; much higher information yield).
- **Watchdog spec moved to `failed/`.**

### D — `chain_v6_lme_gated_purist` (GH200, FAILED to launch)

- **Status:** FAILED 2026-04-29 18:30:49 UTC at step 0 with
  `torch.OutOfMemoryError`. Spec in
  `paper_tools/cloud_watchdog/failed/1777487405_chain_v6_lme_gated_purist.json`.
- **Cause:** watchdog co-launched PURIST (`"gpu": "0"`) onto the
  same GH200 GPU as the still-running w12 cell (also `"gpu": "0"`,
  holding ~58 GiB). PURIST loaded its 35 GiB process and OOM-killed
  at the first forward pass. Fixed in watchdog by `gpu_is_busy()`
  precheck (2026-04-29 ~21:03 UTC).
- **Cell (planned, never executed):** v6 GATED minus carry_state,
  burn_in, with v3-equivalent learning rates (3e-4 / 3e-5).
- **Subsumed by:** v7 P0 V3BIAS (which tests v3's bias on LME with
  curriculum). Not separately re-queued.

## v5 soft-init runs (KILLED — superseded by v6)

All v5 runs share: soft `±4` parity init, `attention_parity` routing,
K=128, L_E=4, N=8, window_k=3, mem_drop=0.10, ctx_drop=0.30,
carry_state, neg_chain ramp 0.05 → 0.5 over 1000 steps,
`burn_in_max=8` with resample, `mask_padding_loss`,
`save_best_metric=composite`, gradient checkpointing. 6000 steps total.

### A (HEADLINE) — `chain_v5_softhidden14_msc`

- **Status:** KILLED 2026-04-29 18:01 UTC at step ~1080 (v6 pivot).
- **Reason:** `gate_max` stuck at 0.000 through 1080 steps — memory
  channel never opened despite 2.5h of GH200 wallclock. Same failure
  mode as local cells C and B prime: regulariser-removal alone isn't
  enough; the corpus + supervision + architecture pivot is needed.
- **Final EVAL (step ~600 last reported):** Δ_sh-m ≈ 0 (memory
  contributing virtually nothing).
- **Machine:** GH200 (192.222.50.225), tmux killed; watchdog spec
  moved to `failed/`.
- **Script:** `archive/Scripts/train_headline_softinit.sh`
- **Superseded by:** `chain_v6_lme_gated_callback_w12`.
- **Log preserved:** remote
  `paper_tools/cloud_watchdog/logs/chain_v5_softhidden14_msc_KILLED_v6_pivot_step1080.log`.

### B — `chain_v5_softembed_msc`

- **Status:** KILLED 2026-04-29 16:53 UTC at step ~1460.
- **Reason:** catastrophic divergence in eval forward pass starting
  step ~1200. `ce_mem` ran from 3.10 → 3.62 in 400 steps; `ce_nomem`
  stable at ~3.14. Memory channel actively poisoning prediction.
  Train loss was fine (~2.7), so the breakage was eval-only.
  Diagnosis: the regulariser stack (`mem_drop=0.10 + ctx_drop=0.30 +
  neg_chain ramp 0.05→0.5`) drove early-stage posterior collapse on
  memory; once the model learned to ignore memory, the contrastive
  ramp peaking at step 1000 satisfied its margin loss adversarially
  by making both `mem` and `shuffle` forwards bad (with `shuffle`
  slightly worse). EVAL trajectory:

  | step | ce_mem | ce_nomem | ce_shuffle | Δ_sh-m |
  |---:|---:|---:|---:|---:|
  | 200 | 3.1183 | 3.1159 | 3.1188 | +0.0005 |
  | 400 | 3.0863 | 3.0891 | 3.0870 | +0.0006 |
  | 600 | 3.0987 | 3.0994 | 3.0997 | +0.0011 |
  | 800 | 3.1003 | 3.0981 | 3.0984 | -0.0019 |
  | 1000 | 3.0982 | 3.0962 | 3.0943 | -0.0039 |
  | 1200 | 3.1364 | 3.1134 | 3.1849 | +0.0485 (fake — both bad) |
  | 1400 | **3.6253** | 3.1453 | 3.5211 | -0.1041 |

- **Script:** `archive/Scripts/train_ablation_b_softembed_msc.sh`
- **Superseded by:** `chain_v5_softembed_msc_noreg` (cell B prime).

### B' — `chain_v5_softembed_msc_noreg` (cell B prime)

- **Status:** KILLED 2026-04-29 17:53 UTC at step ~1540 (v6 pivot).
- **Reason:** confirmed regulariser-removal is NOT enough. Loss
  dropped fine (3.13 → 2.65) but `gate_max` stuck at 0.000
  throughout training. This is the same "memory channel never opens"
  failure as cell A and cell C; the bottleneck is corpus +
  supervision + architecture, not regularisers.
- **Script:** `archive/Scripts/train_ablation_b_prime_no_reg.sh`
- **Superseded by:** `chain_v6_lme_competitive_callback`.

### C — `chain_v5_softhidden14_pgtv`

- **Status:** KILLED 2026-04-29 17:53 UTC at step ~1620 (v6 pivot).
- **Reason:** same `gate_max=0` failure as A and B'. Final EVAL
  @ step 1600: Δ_sh-m = +0.0006 (memory contributing virtually
  nothing), Δ_or-m = -0.2268 (oracle is 0.23 nats better, so memory
  COULD help if recruited).
- **Script:** `archive/Scripts/train_ablation_c_softhidden14_pgtv.sh`
- **Superseded by:** `chain_v6_lme_gated_callback` (local H100 GPU 0)
  and `chain_v6_lme_competitive_callback` (local H100 GPU 1).

### D — `chain_v5_softembed_pgtv` (CANCELLED)

- **Status:** NEVER STARTED. v5 factorial 2x2 was abandoned in favour
  of v6 pivot — all v5 cells confirmed `gate_max=0` regardless of
  corpus/extract source/regulariser combination.

## v4 hard-init runs (negative-result baselines)

All v4 runs share v5's knobs except `--router_mem_bias_init -32
--router_recent_bias_init 32`. Across every logged EVAL line, all v4
runs show `mem == nomem == shuffle` to 4 decimals (`Δ_sh-m =
+0.0000`). This is the bf16-saturation failure mode that motivates
the recipe paper's central methodological contribution. Diagnosis in
`archive/agent_sessions/agent_session_20260429_1011/writer/findings_alpha_mem.md`.

### A_v4 — `chain_v4_hidden14_msc`

- **Status:** KILLED 2026-04-29 15:28 UTC at step ~5500/6000.
- **Reason:** bf16 softmax saturation diagnosis; replaced by v5 cell A.
- **Machine:** GH200; watchdog spec moved to
  `paper_tools/cloud_watchdog/failed/1777425221_chain_v4_hidden14_msc.json`.
- **Script:** `archive/Scripts/train_headline.sh` (legacy).
- **Final state:** `Δ_sh-m = +0.0000` at every EVAL from step 200 → 5400.

### B_v4 — `chain_v4_embed_msc`

- **Status:** KILLED 2026-04-28 23:00 UTC at step 1500. Local GPU was
  needed for another job; never completed.
- **Final state:** same `Δ_sh-m = +0.0000` signature as A_v4.

### C_v4 — `chain_v4_hidden14_pgtv`

- **Status:** KILLED 2026-04-28 22:58 UTC at step 500.

## v3 legacy runs (proximate baselines, kept on disk)

These predate the v5 2x2 design and use a different knob set
(`window_k=8`, no dropouts, no `carry_state`, no `neg_chain` loss).
They serve as the proximate prior baseline that the recipe paper
compares against to motivate the recipe's pieces. Best ckpts and
logs are still in `Runs/` and `logs/`.

### `chain_v3_softparity_full`

- **Status:** STOPPED 2026-04-28 21:23 UTC at step 4425 / 6000.
- **Cell-ish:** `embed` × PG-19+TV, soft `±4`. Closest to (but not
  matching) cell D of the v5 factorial.
- **Result:** in-trainer Δ_sh-m = +0.0379 at best step 4400; α_mem
  on PG-19 val = 4.7e-4 in the overnight routing trace.
- **Eval:** `paper_artifacts/eval/routing_v3sp_*.json` (overnight
  sweep), summary in
  `paper_artifacts/eval/overnight_traces_writeup.md`.

### `chain_v3_attentionbase_full`

- **Status:** STOPPED 2026-04-28 21:23 UTC at step 4425 / 6000.
- **Cell-ish:** `embed` × PG-19+TV, `attention_base` (uniform softmax
  over deltas, no parity init).
- **Result:** in-trainer Δ_sh-m = +0.0149 at step 4400; α_mem ≈ 1.5e-4.

### `chain_v2_phaseA_softparity_b4` (oldest kept)

- **Status:** STOPPED 2026-04-28 02:31 UTC at step 4400.
- **Result:** PG-19 val Δ_sh-m = +0.0529 [+0.0246, +0.0915]; PG-19
  test Δ_sh-m = +0.0279 [+0.0221, +0.0338] (bootstrap CI).
- **Use:** the original "Δ_sh-m CI excludes zero on books" point;
  carried over from exp 1 / paper-1 era.

## v3-vs-v5-cell-C comparison (motivation for the v6 PURIST cell)

Asked late on 2026-04-29: "is v5 cell C just v3 with different
corpus, and if so why did it fail so badly?" Answer: it isn't just
corpus — seven knobs differ, and several of them appear to have
actively broken what v3 had working. Both runs use PG-19+TV,
attention_parity at -4/+4.

| knob | v3 sp | v5 cell C |
|---|---:|---:|
| **window_k** | **8** | **3** |
| extract source | embed | hidden_14 |
| memory_dropout | 0.0 | 0.10 |
| context_dropout | 0.0 | 0.30 |
| carry_state | False | True |
| neg_chain_weight | 0.0 (no contrastive) | 0.5 with 0.05→0.5 ramp |
| burn_in_max | 0 (default) | 8 + resample |
| lr (memres / backbone) | 3e-4 / 3e-5 | 2e-4 / 2e-5 |

EVAL trajectories:

```
        v3 sp                     v5 cell C
step    Δ_sh-m   Δ_nm-m            Δ_sh-m   Δ_nm-m
 200    -0.0001  +0.0019            +0.0008  -0.0011
 400    +0.0042  +0.0025            +0.0003  -0.0105
 600    +0.0089  +0.0040            +0.0005  -0.0129
 800    +0.0107  +0.0054            -0.0003  -0.0104
1000    +0.0177  +0.0068
1600                                +0.0006  (last EVAL before kill)
4400    +0.0379  +0.0131
```

v3 climbed monotonically with mem CE *decreasing*; v5 cell C stayed
flat with Δ_nm-m **negative and deepening** (memory actively hurting)
and mem CE *rising* (3.14 → 3.16 by step 600). Hypothesis ranking
for the regression:

1. `window_k 3` vs `8` cuts recurrent-judge gradient depth by 2.7×;
   credit assignment for "compress prior session into M_c so it pays
   off later" gets crushed.
2. `neg_chain_weight 0.5` with ramp gives the model an incentive
   solution-set that includes "make memory weak so mem ≈ shuffle"
   (lazy) and "make both mem and shuffle bad with shuffle slightly
   worse" (adversarial — what killed cell B).
3. `memory_dropout 0.10` + `context_dropout 0.30` add noise before
   the channel has any signal to lose.
4. `carry_state True` × `window_k 3` carries M_c forward without
   enough TBPTT depth to clean it; M_c becomes a noise reservoir.
5. `hidden_14` vs `embed` extract is plausibly neutral or positive
   per writer's thesis but never empirically validated.
6. `lr 2e-4` vs `3e-4` is a small uniform slowdown.

v6 already addresses items 1-3 and partly 4 (window_k=8, no
dropouts, no contrastive, but carry_state still True). v6 PURIST
tests the remaining items (carry_state, burn_in, lr) by running
v3's training knobs on the LME callback corpus.

---

# Part V — Historical run ledger (v7 → v10, was runs.md)

The `runs.md` ledger now contains only the active v11+ cells. The
v7 / v8 / v9 / v10 entries — all KILLED, SUPERSEDED, or audited —
are preserved here so the v3 → v6 → v7 → v8 → v9 → v10 → v11
narrative stays auditable end-to-end. Each subsection mirrors the
original `runs.md` entry with the planning prose dropped and the
verdicts, result tables, and mechanism statements retained. Date
order is oldest → newest within Part V.

## v7 P0 runs (2026-04-29) — compression curriculum + bias relaxation

User directive 2026-04-29 ~14:30 CDT after the v6 architectural
diagnosis: the depth router is the real bottleneck, not the writer.
v3 standalone routing trace
(`paper_artifacts/eval/routing_v3sp_*.json`) showed α_mem ≈ 4.7e-4
averaged across 55 sublayers AND α_mem essentially identical
between true-chain and shuffled-chain memory at every sublayer —
i.e. the router was content-blind. v6 GATED/COMPETITIVE reproduced
the v3 failure mode with a more severe collapse.

### v7 pivot summary (vs v6)

| axis | v6 | v7 P0 |
|---|---|---|
| credit-assignment path | full chain TBPTT through ~8 sessions of mostly-irrelevant filler | **compression curriculum P0**: every training window is `[evidence_session, callback_session]` (window_k=2); fresh M_c |
| router bias | `recent=+4, mem=−4` (v3 default; ≈ 3e-5 mass at init) | A/B: SOFTERBIAS uses `mem=−2` (~70× lift); V3BIAS keeps `−4` |
| burn-in | `burn_in_max=24` with resample | **0** |
| carry_state | True | **False** |
| callback alignment | `callback_window_bias=0.7` | superseded by `curriculum_evidence_bias=1.0` |

### v7 cells & verdicts

| cell | machine | routing | bias | first non-zero gate_max | gate_max @ 400 | killed |
|---|---|---|---|---|---:|---|
| `chain_v7_p0_softerbias` | local H100 GPU 0 | attention_parity | mem=−2 | never (0.0000 through step 1400) | 0.0000 | step ~2000, `‖m^t‖≈1.7e-10` |
| `chain_v7_p0_v3bias` | local H100 GPU 1 | attention_parity | mem=−4 | never | 0.0000 | step ~2000; std Δ_nm-m=−0.147 by step 1200 |
| `chain_v7_p0_simplegate` | GH200 | **simple_gate** | n/a | **step 20 (0.0004)** | **0.0074** | once v8a passed step 1000 trigger |

Eval trajectory of `chain_v7_p0_softerbias` (n=256, eval_window=8):

| step | mem CE | nomem CE | shuffle CE | Δ_nm-m | Δ_sh-m | gate_max |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 1.5322 | 1.5269 | 1.5321 | −0.0052 | −0.0001 | 0.0000 |
| 400 | 1.4866 | 1.4805 | 1.4866 | −0.0060 | +0.0001 | 0.0000 |
| 600 | 1.4727 | 1.4649 | 1.4727 | −0.0078 | +0.0000 | 0.0000 |
| 800 | 1.4390 | 1.4365 | 1.4390 | −0.0025 | −0.0000 | 0.0000 |
| 1000 | 1.4254 | 1.4237 | 1.4254 | −0.0018 | +0.0000 | 0.0000 |
| 1200 | 1.4178 | 1.4159 | 1.4178 | −0.0019 | +0.0000 | 0.0000 |

Eval trajectory of `chain_v7_p0_v3bias` (worst of the trio):

| step | mem CE | nomem CE | shuffle CE | Δ_nm-m | Δ_sh-m | gate_max |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 1.5435 | 1.5393 | 1.5436 | −0.0042 | +0.0001 | 0.0000 |
| 400 | 1.5066 | 1.4807 | 1.5059 | −0.0259 | −0.0007 | 0.0000 |
| 600 | 1.5135 | 1.4609 | 1.5135 | −0.0526 | −0.0000 | 0.0000 |
| 800 | 1.5444 | 1.4271 | 1.5438 | −0.1173 | −0.0006 | 0.0000 |
| 1000 | 1.5086 | 1.4140 | 1.5086 | −0.0945 | +0.0000 | 0.0000 |
| 1200 | 1.5526 | 1.4057 | 1.5524 | −0.1470 | −0.0003 | 0.0000 |

`chain_v7_p0_simplegate` per-step gate (every 20 steps):

| step | gate_mean | gate_max | loss |
|---:|---:|---:|---:|
| 20 | −0.0000 | 0.0004 | 2.74 |
| 100 | −0.0023 | 0.0055 | 1.74 |
| 200 | −0.0019 | 0.0051 | 1.56 |
| 300 | −0.0023 | 0.0065 | 1.16 |
| 400 | −0.0024 | 0.0074 | 1.07 |

Step-200 eval: `Δ_nm-m=−0.044`, `Δ_sh-m=−0.0001`, `gate_max=0.0051`.
The gate opens but the readout produces noise on the eval
distribution (curriculum-mismatch failure inherited from the other
cells).

### Two findings the v7 trio nailed down

1. **simple_gate opens the memory channel where attention_parity
   cannot.** Same corpus, writer, curriculum, extract source. The
   `+4` recent-source bias in `attention_parity` saturates the
   depth softmax against memory and the gradient signal needed to
   relax it is overwhelmed by gradient that wants to keep recent.
   simple_gate has a direct gradient path to its scalar gate (no
   softmax competition) and gradient finds it within 20 steps.
   This was the empirical refutation of the paper's "block AttnRes
   routing pool over delta sources" choice as the canonical
   primitive on the chain trainer.
2. **Pure-P0 curriculum (`curriculum_evidence_bias=1.0`) is a
   train/eval distribution mismatch trap.** All three cells learn
   the train task aggressively (loss 3 → <1 in 800 steps) but show
   negative Δ_nm-m on the standard eval (eval_window=8, sequential
   M_c through ~40 prior sessions). v3-default V3BIAS is the worst
   (−0.147 by step 1200). Mechanism: training builds M_c from one
   fresh evidence session; eval builds it sequentially through 40+
   sessions of compounding judge updates — completely different
   statistics.

### v7 sampler implementation (kept)

`--curriculum_evidence_bias <float>` (default 0.0). Picks a random
evidence position in `[0, cb_pos)`, then `window_k − 2` random
intermediate positions strictly between evidence and callback,
stacks `[evidence, ...intermediates, callback]` chronologically.
Falls back to `callback_window_bias` when no callback annotation
exists; falls all the way through to contiguous sampling for
non-callback corpora (PG-19 / TV / REALTALK). Phase implied by
window_k: P0=2, P1=3, P2=5, P3=8. Code lives in
[`src/train_chain.py`](../src/train_chain.py)
(`ChainCorpus.chain_curriculum_window` +
`ChainSampler.sample_window` curriculum branch).

### Watchdog patch landed alongside v7

`paper_tools/cloud_watchdog/watchdog.sh` got a `gpu_is_busy()`
precheck (defers spec launch if another CUDA process holds
> `GPU_BUSY_MIB` MiB on the target GPU; default 2048). Fixes the
silent first-step OOM that killed v6 PURIST on 2026-04-29 18:30 UTC.

## v8 (2026-04-29) — diagnostic lens-change + readout-norm fix

User directive 2026-04-29 ~22:00 UTC: "I somehow think there's a
problem with the way we are evaluating; the system is learning a
hard task; it needs to first learn to summarize and use the
memory… we have to be more wary of other guidance metrics
though." This single observation triggered an evaluation-methodology
redesign and downstream of it an architectural diagnosis the v7 P0
cells could not have surfaced.

### v8 evaluation-methodology rewrite

Four diagnostics added to `Trainer.evaluate()` and exposed as a
standalone tool
([`paper_tools/diagnose_recruitment.py`](../paper_tools/diagnose_recruitment.py)):

1. **Phase-aligned eval (`pa_*`).** For each LME chain with
   `cb_pos ≥ 1`, pick a random evidence position
   `e ∈ [0, cb_pos)`, build M_c from session `e` alone (single
   judge step — exactly what P0 trains on), then score the callback
   session under three M_c regimes: `mem`, `nomem`, `shuffle`.
   Reduce two ways: whole-session (`pa_ws_*`) and callback-only
   (`pa_cb_*`) using `session_callback_mask`.
2. **Per-sublayer routing recruitment (`rec_*`).** Replaces coarse
   `gate_max`. For `simple_gate` it snapshots top-3 sublayer gates
   with sign and the fraction with `|gate| > 1e-3`. For
   `attention_parity` / `attention_base` it runs an eval batch
   with `collect_alpha_trace=True` and reports per-sublayer α_mem
   (the actual realised mass; `mem_bias` alone is misleading).
3. **Readout magnitude (`mt_norm_ratio_*`).** Pulse check:
   `mean(‖m^t‖) / mean(‖embed‖)` on the phase-aligned setup.
   ≈0 means V projection collapsed; ≫1 means m^t dominates.
4. **`save_best` switched to `phase_aligned`** with composite
   `−(pa_cb_dsh + 0.5 · pa_ws_dsh)`. Standard eval becomes a
   sanity check, not a checkpoint-selection signal.

### Lens-change finding 1 — `attention_parity` is *causally* collapsed

`paper_tools/diagnose_recruitment.py` against
`chain_v7_p0_softerbias/step-2000`
([`paper_artifacts/eval/diag_v7_softerbias_step2000.json`](../paper_artifacts/eval/diag_v7_softerbias_step2000.json)):

| metric | value | reading |
|---|---:|---|
| α_mem_max | 0.00092 | depth router uniformly content-blind |
| α_mem_mean | 0.00037 | below uniform-prior floor 1/(N+2) |
| frac_open (α > 0.05) | 0.00 | zero sublayers above threshold |
| **‖m^t‖/‖embed‖** | **1.66 × 10⁻¹⁰** | **readout output is fp-zero** |
| pa_cb_dsh | **0.0 exactly** | mem and shuffle produce identical logits |
| pa_cb_dnm | −0.057 | memory mildly hurts callback CE |

Mechanism. attention_parity with `mem=−2/recent=+4` initialises α_mem
at ~exp(−2)/exp(+4) ≈ 4×10⁻⁴ per sublayer; gradient on
`MemoryReadout.W_V` flows through `α_mem · (...)` so it is ~1000×
weaker than backbone gradient; AdamW with `weight_decay=0.1` consumes
this weak gradient each step and steadily drives `‖W_V^read‖` toward
zero. Once `‖W_V^read‖ ≈ 0`, `‖m^t‖ ≈ 0`, and the architecture is
*causally* equivalent to "no memory" regardless of M_c.

This explains every `attention_parity` cell in the v3-v7 lineage —
they were not failing to learn, they were learning to *delete* the
memory channel.

### Lens-change finding 2 — `simple_gate` is *exploded*

Same diagnostic on `chain_v7_p0_simplegate/step-500`
([`paper_artifacts/eval/diag_v7_simplegate_step500.json`](../paper_artifacts/eval/diag_v7_simplegate_step500.json)):

| metric | value | reading |
|---|---:|---|
| gate_max_abs / frac_open | 0.0085 / **0.86** | 86% of sublayers active |
| **‖m^t‖/‖embed‖** | **164.5** | readout swamps residual stream |
| pa_cb_dsh | −0.0059 | shuffle slightly better than mem |
| pa_cb_dnm | **−0.66** | memory destroys callback CE |

Opposite failure mode: scalar gate gives `W_V^read` direct LM
gradient, so it grows without bound. After 500 steps `gate · m^t`
per sublayer adds ~1.3× ‖embed‖ to the residual; across 56 sublayers
the perturbation swamps the LM signal. Gate IS opening, the readout
it gates is unbounded.

### v8 architecture fix — RMSNorm on `MemoryReadout` output

Single line change in [`src/modeling_memres.py`](../src/modeling_memres.py):
`MemoryReadout.__init__` adds `self.out_norm = Qwen3RMSNorm(...)`
and `forward` returns `self.out_norm(attn @ V)`. Init parity
preserved across all six cases of
[`paper_tools/init_parity_test.py`](../paper_tools/init_parity_test.py)
(both simple_gate and attention_parity-with-mem hit 0.000e+00).

### v8 cells & outcomes

| cell | corpus | curriculum | result | killed |
|---|---|---|---|---|
| `chain_v8a_p0_simplegate_rmsnorm` | LME-only | pure P0 | step 200 PA CB Δ_nm-m = +0.029, then collapsed monotonically to −1.30 by step 1000 | step ~880 (catastrophic overfit on narrow distribution) |
| `chain_v8b_mixed_simplegate_rmsnorm` | LME-only | mixed-bias 0.5, k=8 | step 1200 PA CB Δ_sh-m = +0.0067 (peak), then collapsed to −0.032 by step 2400 | step ~2460 (delayed v8a trap by ~1000 steps but didn't escape it) |
| `chain_v8c_diverse_simplegate_rmsnorm` | v6_lme_msc (diverse) + heavy regularisation | mixed-bias | flat ±0.007 for 2000 steps; gate stuck at ~0.004 / frac_open 0.30 | step ~2000 (over-regularisation prevents both learning and overfitting) |

`chain_v8a` full eval trajectory (the canonical "good then collapse"):

| step | std Δ_nm-m | std Δ_sh-m | PA WS Δ_nm-m | PA CB Δ_nm-m | PA CB Δ_sh-m | gate_max | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | −0.0162 | −0.0001 | +0.0668 | **+0.0288** | **+0.0035** | 0.0069 | 1.27 |
| 400 | −0.0576 | −0.0003 | −0.1248 | −0.6512 | +0.0136 | 0.0085 | 1.05 |
| 600 | −0.0709 | +0.0002 | −0.1001 | −0.6350 | +0.0197 | 0.0107 | 0.85 |
| 800 | −0.1164 | −0.0003 | −0.4202 | −1.0071 | +0.0056 | 0.0119 | 0.78 |
| 1000 | −0.1756 | −0.0002 | −0.5958 | −1.2978 | −0.0041 | 0.0134 | 0.75 |

`chain_v8b` step-200 sanity check (the qualitatively-different
"sparse late-layer" pattern that hinted at the v9 fix):

| lens | reading |
|---|---|
| Standard Δ_nm-m | +0.0020 (memory helps long-chain) |
| PA-EVAL CB Δ_nm-m | +0.0076 |
| PA-EVAL CB Δ_sh-m | +0.0033 |
| gate_max / frac_open | 0.0050 / **0.46** (sparser than v8a) |
| top sublayers | 52, 55, 53 (positive) |
| ‖m^t‖/‖embed‖ | 73.5 (RMSNorm holding) |

`chain_v8b` later trajectory through collapse:

| step | std Δ_nm-m | PA CB Δ_nm-m | PA CB Δ_sh-m | gate_max | frac_open |
|---:|---:|---:|---:|---:|---:|
| 1000 | −0.0004 | +0.0046 | −0.0002 | 0.0054 | 0.39 |
| 1200 | +0.0010 | +0.0073 | **+0.0067** | 0.0054 | 0.54 |
| 1400 | −0.0003 | +0.0118 | +0.0038 | 0.0074 | 0.52 |
| 1600 | −0.0025 | +0.0052 | −0.0027 | 0.0087 | 0.62 |
| 2200 | −0.0005 | **−0.0418** | **−0.0133** | 0.0088 | 0.59 |
| 2400 | −0.0002 | −0.0340 | **−0.0320** | 0.0085 | 0.68 |

### Why the standard eval was misleading us for 2k steps

Both v3-v7 failure modes (attention_parity collapse, simple_gate
explosion) produce `Δ_sh-m ≈ 0` on the standard eval. That standard
`Δ_sh-m ≈ 0` was being read as "memory channel is not yet
content-aware, keep training" — both reads miss that the architecture
has reached a *terminal* state. The phase-aligned callback diagnostic
discriminates cleanly:

- Collapsed (attention_parity): `pa_cb_dsh = 0.0` exactly, `‖m^t‖ ≈ 0`.
- Exploded (simple_gate w/o RMSNorm): `pa_cb_dsh ≈ 0` but
  `pa_cb_dnm ≪ 0` and `‖m^t‖ ≫ ‖embed‖`.
- Healthy (v8a step 200): `pa_cb_dsh > 0`, `pa_cb_dnm > 0`,
  `‖m^t‖` of the same order as `‖embed‖`, gate growing.

## v9 (2026-04-30) — judge-competition curriculum (the breakthrough)

User directive 2026-04-30 ~00:00 UTC after v8a/b/c oscillated
around the same noisy minimum on PA CB Δ_sh-m: "we separately learn
the summarizer and the competition; the individual problems then
can generalize."

### Problem v9 isolated

The v8 mixed-bias curriculum trains writer + readout but **never
explicitly trains the judge layer's keep-vs-write decision**: the
P0 sub-window `[evidence, callback]` has `M_c_prev = 0` at the
judge step, so `compress_session(extract(evidence), 0)` degenerates
to a no-competition aggregation. Every v8 cell oscillated around
`pa_cb_dsh ≈ 0` ±0.005 with no monotone improvement on judge
behaviour.

### Curriculum design (new)

`--curriculum_competition_bias <float>` builds 3-session windows
that ISOLATE the judge subproblem. Two paired structures sampled
50/50, both scoring the same callback:

| sample | window | judge step at session 1 | correct gate | label |
|---|---|---|---:|---|
| **A: KEEP-PREV** | `[evidence, distractor, callback]` | prev_M = M_after(evidence) (relevant), C_t = extract(distractor) | small | "keep" |
| **B: WRITE-NEW** | `[noise, evidence, callback]` | prev_M = M_after(noise) (irrelevant), C_t = extract(evidence) | large | "write" |

Both samples score the same callback ⇒ gradient from callback CE
directly trains `write_gate` / judge weights to be content-aware.
Implementation: new `ChainSampler` Branch 0 in
[`src/train_chain.py`](../src/train_chain.py) ahead of the existing
curriculum + alignment branches.

**Label noise is intentional.** LME does not annotate which earlier
session contains the actual referenced fact. The "evidence" in
Sample A and "noise" in Sample B are uniform random picks from
`[0, cb_pos)`. This mirrors the deployment regime — at inference
the model also has no oracle evidence signal — so the writer + judge
are forced to compact and decide using only content cues.

### v9 baseline result — `chain_v9_compete_lme_gh200`

The strongest signal of the entire experiment-2 trajectory in its
first 1400 steps:

| metric (PA-aligned) | v8b PEAK (step 1200) | v9 step 1400 | factor |
|---|---:|---:|---:|
| CB Δ_nm-m | +0.0073 | **+0.178** | **24.4×** |
| CB Δ_sh-m | +0.0067 | **+0.0152** | 2.3× |
| WS Δ_nm-m | ~0 | +0.094 | n/a |
| frac_open | 0.54 | **0.89** | 1.6× |
| train loss (LME) | ~1.27 | **~1.0** | −0.27 |

v9 trajectory (PA, 48 chains):

| step | train loss | CB Δ_nm-m | CB Δ_sh-m | WS Δ_nm-m | gate_max | frac_open | top sublayers |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 1.15 | −0.045 | +0.016 | −0.056 | 0.0045 | 0.88 | l54+, l55+, l49+ |
| 1200 | 0.99 | +0.023 | −0.004 | +0.017 | 0.0049 | 0.86 | l55+, l54+, l49+ |
| 1400 | 1.13 | **+0.178** | **+0.015** | **+0.094** | 0.0052 | 0.89 | l55+, l41+, l54+ |

What's structurally different vs v8b/v8c: `frac_open ≈ 0.89` (almost
every sublayer using memory; v8b at 0.54). Top sublayers shifted
from l48/l49 (v8b) to l54/l55 + l41 (v9) — different layers light
up when training tells the gate "memory really does matter."
Standard `Δ_nm-m` is negative on the 8-session contiguous eval as
expected (7/8 sessions have no callback; aggressive callback writing
hurts non-callback CE) — this is the deployment-faithful trade-off.

`chain_v9_compete_lme_gh200/best` (saved at step 1400) was the
leading checkpoint of the program at v9 time.

### v9 ablation queue (4 cells, GH200 watchdog)

| order | cell | knob varied vs v9 | what it answered |
|---:|---|---|---|
| 1 | `chain_v9a_abl_cbw1_gh200` | `callback_loss_weight` 3.0 → **1.0** | is v9's win from competition curriculum or from upweighted callback supervision? |
| 2 | `chain_v9b_abl_mixed_gh200` | `competition_bias` 1.0 → 0.5 + `evidence_bias` 0.0 → 0.5 + `window_k` 3 → 8 | is *pure* competition necessary, or is half competition + half mixed-bias enough? |
| 3 | `chain_v9c_abl_diverse_gh200` | corpus LME-only → **v6_lme_msc_train** (LME+MSC+PG-19+TV) | does v9 survive data diversity? |
| 4 | `chain_v9d_abl_attnparity_gh200` | `memres_mode` simple_gate → **attention_parity** (with RMSNorm fix retained) | now that readout magnitude is bounded, is paper-spec attention_parity viable under the v9 curriculum? |

Outcome that fed the v10 directive: `v9c` matched `v9` baseline peak
performance on the diverse corpus. `v9d` stayed architecturally
collapsed (α_mem=0) on LME-only even with the competition curriculum
and RMSNorm fix — but that signal was confounded with the LME-only
data axis, motivating v10b as the cleaner test.

### v8b/v8c verdict (in the v9 light)

| recipe | curriculum | cb-token benefit | content discr. | gate openness | verdict |
|---|---|---:|---:|---:|---|
| v8a | pure-P0 | overfit @ step 800 | overfit @ 800 | 0.4 (negative-signed) | killed |
| v8b | mixed-bias | +0.012 then collapse @ 2200 | +0.007 peak then collapse | 0.6 | killed |
| v8c | mixed + diverse + heavy reg | flat ~0 | flat ~0 | 0.30 stuck | killed |
| **v9** | **pure-competition** | **+0.178** @ 1400 | **+0.015** @ 1400 | **0.89** | **HEADLINE** |

The v8 cohort comprehensively bracketed the failure modes —
overfitting at one extreme (v8a/v8b), stalling at the other (v8c).
Both extremes were about training writer + readout + judge *together*
through generic LM loss on a callback-token mix; the competition
curriculum (v9) is the structural fix that decomposes the problem
into trainable subparts.

## v10 (2026-04-30) — data diversity is the load-bearing axis

User directive 2026-04-30 ~19:00 UTC after reviewing the v9 results:
> "something tells me it has a lot to do with the LME dataset being
> bad; notice how out of all the v9's, only the v9c diverse dataset
> had a really good survivability. … This tells me that constructing
> a SUPER diverse dataset with MANY MANY memory-using data is useful."

Hypothesis: the v3–v9 failure modes (router collapse, readout drift,
peak-then-decay) were not primarily architectural — they were
manifestations of training on a narrow, callback-only distribution.
v9c matched the v9 peak on the diverse corpus despite the same
recipe; the v10 campaign reframes diversity of memory-requiring
training distributions as the load-bearing axis.

### Campaign design

| cell | machine | backbone | L_E | routing | corpus |
|---|---|---|---:|---|---|
| **v10a composed_diverse** | local H100 GPU 0 | Qwen3-0.6B-large | 4 | simple_gate | v6_lme_msc (6378 chains) |
| **v10b attnparity_pm4_diverse** | local H100 GPU 1 | Qwen3-0.6B-large | 4 | attention_parity (+4/−4) | v6_lme_msc |
| **v10 4b_mega_attnparity** | GH200 | **Qwen3-4B-xlarge (L_E=10)** | **10** | **attention_parity (+4/−4)** | **mega (~150-300k chains)** |

Routing rationale: v10b (0.6B + attention_parity + diverse) was the
cheap proxy that told us in ~6 h whether attention_parity could work
*at all* on diverse data; v10 4b_mega pursued the headline recipe
independently on GH200. 4B (not 8B) because qwen3-8b-xlarge peaks at
~106 GB HBM under full AdamW (>96 GB GH200) and the user directive
was "if it can't fit, make the model smaller" (no frozen-backbone
hack, no bitsandbytes dependency).

### Mega corpus (target ~200-300k chains)

Build pipeline: `Scripts/build_mega_corpus_gh200.sh` plus
`paper_tools/build_synthetic_dialogue_chains.py`. Sources:

- v6_lme_msc_train (existing 6378 chains; preserved base)
- ultrachat (HuggingFaceH4/ultrachat_200k, ~25k chains capped)
- pippa (PygmalionAI/PIPPA, persona chats)
- soda (allenai/soda, ~25k synthetic social dialogues)
- oasst1 (OpenAssistant/oasst1, tree-flattened assistant chats)
- no_robots (HuggingFaceH4/no_robots, multi-turn)
- narrativeqa (deepmind/narrativeqa, doc-grounded Q/A)
- writingprompts (euclaise/writingprompts, long narrative
  continuation — PG-19-like)

Optional sources gated behind `EXTRA_SOURCES=hh_rlhf lmsys`. Source-
bucket weights:
`{"longmemeval":4.0, "msc":3.0, "ultrachat":2.0, "pippa":2.5,
"soda":1.5, "synthdlg":1.5, "pg19":1.0, "tv":3.0, "realtalk":2.0,
"lmsys":1.5}` — preserves the v9c LME-heavy mix while giving
dialogue / narrative non-trivial weight. New synthetic-dialogue
chains are emitted without `session_callback_mask` so the
competition / evidence curriculum branches still only fire on the
450 LongMemEval chains in the base corpus; the rest contributes
contiguous-window LM gradient that regularises the readout against
over-injection on callback-less distributions.

### v10 4B MEGA architecture & schedule

`qwen3-4b-xlarge` preset: Qwen3-4B (d=2560, 36 layers) backbone +
**L_E=10** eleven-layer Perceiver extraction + K=128 slots + N=8
AttnRes blocks. ~4.3 B total params, ~52 GB peak HBM under full
AdamW. Routing: `attention_parity` `+4/−4`. Extract source:
`hidden_18` (middle of 36-layer backbone). Schedule: window_k=4,
carry_state=True, bs=2 ga=4 (effective 8), lr_memres=3e-5,
lr_backbone=5e-6, steps=30000, warmup=500, cosine decay. Curriculum:
`competition=0.5, evidence=0.3, callback_window=0.3, callback_w=3.0`
(composed). `save_best=phase_aligned`. ntfy: `memres-e6ebdc70`.

### v10 outcome (the trigger for v11)

The v10 campaign was killed mid-training and audited (see
[`README.md`](../README.md#stop-everything) "Stop everything and
read this first" / post-v10 audit). The audit revealed five
compounding causally-independent failures, each sufficient on its
own:

- **P0 (data, ~100× leverage).** The corpus builder threw away the
  `answer_session_ids` annotations LongMemEval-S ships with, so 96 %
  of training windows had M_c built from sessions that demonstrably
  did *not* contain the answer. The LM-loss-optimal policy under
  that distribution is "ignore memory."
- **P1 (chicken-and-egg).** Gate, readout, and writer all multiply
  each other in the forward path (`h += g · m^t`). With `g = 0` and
  `W_V^read = randn(d⁻¹ᐟ²)` at init, no parameter sees gradient at
  step 0 and they all stay at zero forever.
- **P2 (magnitude).** The readout RMSNorm pinned `‖m^t‖/‖embed‖ ≈ 73`,
  so the useful gate range was `[0, ~0.014]` — too narrow for AdamW's
  natural step size to land in stably.
- **P3 (PA-eval misalignment).** The phase-aligned eval picked the
  "evidence" session uniformly, matching the broken curriculum and
  making honest measurement impossible.
- **P5 (recurrent-depth mismatch).** Standard-eval ran at recurrent
  depths the model had never seen during training.

All three v10 cell launchers and the mega corpus pipeline survive in
the repo; the v10 4B mega launcher itself was deleted as part of the
2026-04-30 cleanup (replaced by v11p in spirit). The v11 campaign in
[`results/exp2_chain_recipe/runs.md`](../results/exp2_chain_recipe/runs.md)
addresses P0/P2/P3 in code and P5 in two of its ablation cells.

---

# Part VI — Historical run ledger (v11 → v14, was runs.md)

The active `runs.md` ledger now contains only v15 (the live campaign).
The v11 / v12 / v13 / v14 entries — every cell of which is now KILLED,
SUPERSEDED, or whose mechanistic conclusion has been promoted to the
README's "Architectural priors" block — are preserved here so the
v3 → v6 → v7 → v8 → v9 → v10 → v11 → v12 → v13 → v14 → v15 narrative
remains end-to-end auditable. Each subsection mirrors the original
`runs.md` entry with the planning prose dropped and the verdicts,
result tables, and mechanism statements retained. Date order is
oldest → newest within Part VI (v11 → v14).

## v11 — first wave (g/h/i/j/k/l + 4b_mega; 2026-04-30 → 2026-05-01)

User directive 2026-04-30 ~21:00 UTC after the v10 audit:
> "create 4-5 ablation studies on the GH200 for 0.6B and then the
> 'most make sense initializers' for 4B model to train. be sure to
> use a bias for the memory at +4 0. I know that the system from v3
> definitively is better."

Hypothesis: with the data fix (P0), magnitude fix (P2), and the user-
directed softer bias (`+4 / 0`, ~50× more initial α_mem mass than v3's
`+4 / -4`), the memory channel will (a) open (`α_mem_max > 1e-2`), (b)
learn content-specific writes (`pa_cb_evidence_lift > 0`), and (c)
start closing the deployment-distribution gap.

### Code/data fixes shipped before launch

* `paper_tools/build_conversational_callback_chains.py` — LongMemEval
  loader now reads `answer_session_ids` and emits
  `chain_evidence_positions`; `has_answer` turns flip
  `session_callback_mask`.
* `paper_tools/merge_chain_corpora.py` + `Scripts/build_v11_corpora_remote.sh`
  — preserves `chain_evidence_positions`; built
  `v11_lme_msc_train_s512.pt` (6378 chains / 450 with evidence) and
  `v11_mega_train_s512.pt` (67745 chains / 450 with evidence).
* `train_chain.py` — `ChainSampler.sample_window` uses
  `chain_evidence_positions` for the competition-curriculum evidence
  slot when present; `_phase_aligned_eval` picks evidence from the
  same labels and emits `pa_cb_evidence_lift = pa_cb_dnm(actual evidence) − pa_cb_dnm(random haystack)`;
  added `--memres_gate_init`, `--memres_readout_norm_init`,
  `--curriculum_competition_bias 1.0`.
* `modeling_memres.py` — `Qwen3MemResConfig` accepts the two new init
  knobs; `_init_memres_params` applies `out_norm.weight.fill_(scale)`.

### Cells & verdicts (final-step PA-eval)

| cell | wall-clock (UTC) | core change vs v11g | final α_mem_max | final PA CB Δ_nm-m | final PA CB Δ_sh-m | final `evidence_lift` | verdict |
|---|---|---|---:|---:|---:|---:|---|
| **v11g** baseline | Apr 30 21:46 → May 1 00:26 | reference (P0+P2, AP +4/0, gated, hidden_14, k=3) | 0.0093 | −0.116 | −0.018 | n/a | peak step 600 (PA CB Δ_nm-m=+0.030, Δ_sh-m=+0.016); decayed monotonically through step 4000 |
| **v11h** drop P2 | May 1 00:26 → 03:08 | `--memres_readout_norm_init 1.0` (norm=1.0) | 0.0086 | −0.002 | −0.002 | −0.013 | `‖m^t‖/‖embed‖ = 72`; AP softmax self-regulates magnitude; finishes at PA CB ≈ 0 |
| **v11i** mem_bias=−4 | May 1 03:08 → 05:47 | `--router_mem_bias_init -4` | **0.0001** | +0.006 | +0.005 | +0.002 | router stays fully collapsed step 0 → step 4000 |
| **v11j** depth (k=4 + carry + burn=12) | May 1 05:47 → 09:13 | tests P5 alone | 0.0053 | −0.052 | +0.001 | +0.007 | adding depth *reduced* α_mem opening vs v11g; P5 in isolation is a no-op |
| **v11k** P0 reverted | May 1 09:13 → 12:07 | counterfactual (no `chain_evidence_positions`) | 0.0078 | −0.265 | +0.027 | n/a | confirms P0 helps quantitatively (~2.3× better Δ_nm-m) but doesn't change the failure shape |
| **v11l** frozen backbone | May 1 15:48 (3 sec) | `--freeze_backbone` | — | — | — | — | FAILED at startup — script invoked `python src/train_chain.py` but file was at `train_chain.py` in CWD. Relaunched locally as v11l-fix |
| **v11l-fix** local relaunch | step 600 readout | `--freeze_backbone`; otherwise IDENTICAL to v11g | 0.0005 | −0.0023 | −0.0022 | — | with backbone frozen (zero co-evolution by construction) α_mem_max collapses to 5e-4. **Mechanism (b) backbone-co-evolution REJECTED.** The writer subsystem itself is the bottleneck on LME with the original architecture. |
| **v11r** readout warmup + InfoNCE | May 1 12:07 → 15:48 | `--readout_warmup_steps 500` + `--readout_warmup_router_bias 4.0` + `--contrastive_infonce_weight 0.5` | 0.0101 | +4.91 | **−1.14** | **−1.12** | broken regime: memory is *worse* than shuffle by 1.14 nats; oracle-evidence M_c is *worse* than random-haystack M_c by 1.12 nats |
| **v11q** InfoNCE alone | May 1 15:49 → killed step 2600 | `--contrastive_infonce_weight 0.5` | 0.019 | +13.04 | **−0.96** | **−0.57** | NCE diag/off gap +6 to +14 (head learning chain-identity discrimination), but `pa_cb_ce_mem = 6.3 nats` (vs ~1.5 nats for normal LM regime) — the readout is destroying generic next-token prediction without helping the answer span |
| **v11_4b_mega** | Apr 30 21:46 (5 sec) | qwen3-4b-xlarge, L_E=10, k=4 | — | — | — | — | FAILED at startup — CUDA OOM during `model.to(device)` (peak ~106 GB on a 96 GB card). Same OOM that killed the v10 4B mega. |
| **v11m_chinchilla** | — | 16k steps, k=4, carry, burn=12 (1.35× Chinchilla) | — | — | — | — | CANCELLED 2026-05-01 — token-starvation hypothesis decisively rejected by D2 on v11g/best (uniform fixed point) + v11l-fix (frozen backbone still collapses) |
| **v11p_frozen_chinchilla_mega** | — | mega corpus, frozen backbone, 25k steps, lr=1e-4 (2.1× Chinchilla) | — | — | — | — | CANCELLED 2026-05-01 — superseded by v12d_frozen which composes the same compute budget with the slot-attention writer |

### What the cross-cell evidence says about P0–P5

* **P0 (data: missing evidence labels) — confirmed real, magnitude
  modest.** v11g vs v11k differ in *only* the corpus. PA CB Δ_nm-m
  moves from −0.265 → −0.116 (factor of 2.3 in the right direction).
  Necessary but not sufficient; the failure shape is unchanged.
* **P1 (router saturation) — causally confirmed.** v11i with
  `mem_bias = −4` parks at α_mem_max = 1e-4 from step 0 to step 4000;
  v11g/h/j/k/r with `mem_bias = 0` open α_mem to 5e-3 to 1.9e-2.
  Cleanest single-knob result of the v11 campaign.
* **P2 (readout magnitude) — irrelevant for AP, only matters for
  `simple_gate`.** v11g (norm=0.05, ‖m^t‖=3.6) and v11h (norm=1.0,
  ‖m^t‖=72) finish at the same PA CB metrics. The depth softmax in
  `attention_parity` handles magnitude on its own. **Recommend
  dropping P2 from the v12+ design constraints.**
* **P3 (PA-eval misalignment) — fixed and now informative.**
* **P4 (gradient dilution) — partially addressed by `cb_loss_w=3` +
  InfoNCE.** v11q's NCE diag/off gap of +6 to +14 nats proves the
  gradient signal is now strong enough to shape M_c — but in the
  wrong direction (chain-identity instead of content).
* **P5 (recurrence depth mismatch) — no-op alone.** v11j with
  `window_k=4 + carry_state + burn_in_max=12` finished with *lower*
  α_mem opening (0.0053 vs v11g's 0.0093).

### `evidence_lift` is the smoking gun

`evidence_lift = pa_cb_ce_mem(actual evidence) − pa_cb_ce_mem(random haystack)`.
A working memory should make this strongly negative. v11h/i/j sit at
~0; v11r/q go strongly negative under InfoNCE. The v11r/v11q pattern
is unambiguous: **aggressive contrastive losses on M_c teach the
writer to encode chain-identity rather than chain-content.** The
InfoNCE objective rewards "M_c[i] uniquely predicts chain i's
callback" but does not specify *which* feature of chain i must drive
the prediction — and the easiest discriminator is a chain-identity
hash, not the answer text.

### D5 audit on `chain_v11g_ap_baseline_gh200/best` (2026-05-01)

```text
ROUTE: mode=attention_parity  alpha_or_gate_max=0.0092  frac_open=0.00
READOUT: ||m^t||/||embed|| mean=77.23
D2-JUDGE: row_entropy=5.541 (uniform=5.545; norm=0.999) keep_mean=0.500
          keep_var=0.0000 eff_rank=1.02
D3-MC   : Δ_step mean=1.347 max=1.383 self||M||=1.000 pair=0.022
pa_cb_dnm = +0.373  pa_cb_dsh = +0.0023  pa_cb_evidence_lift = +0.016 (synth D4)
```

D5 with 300 readout-only steps (lr 1e-3, batch 4):

```text
baseline callback_ce: 8.1989
final    callback_ce: 4.2629
Δ                   : -3.9359  (-48.0%)
VERDICT: LIKELY R: writer encoded the information; readout was the bottleneck.
```

### v11 headline conclusion (mechanism, not symptoms)

After eliminating P0/P2 (data + magnitude), confirming P1 (router
gating), and disconfirming P5 (depth alone), the failure mode that
survives is:

> **The writer is content-blind.** With the LM-only objective
> (v11g/h/i/j/k) it learns to compress *something* about each session
> but `evidence_lift ≈ 0`. With dense contrastive supervision (v11r/q)
> it learns chain-identity and `evidence_lift` becomes strongly
> negative. Either way the writer never learns "extract the answer
> from this session" because no objective in the v11 pipeline tells
> it to.

D5 confirms M_c does carry chain-discriminative content (just not
*answer-specific* content). The most informative single next cell is
not another curriculum tweak but a writer-only warmup against an
extractive objective. That is the v13 design.

### Diagnostic toolkit (D1–D5; 2026-05-01)

Five mechanism-level audits, all integrated into `src/train_chain.py`
(`--diagnose_grad_groups`, `--diagnose_memory_dynamics`) plus
standalone scripts under `tools/`:

| ID  | What it measures | How |
|-----|------------------|-----|
| D1  | Per-module gradient L2 norms (M_in / extract / M_judge / judge / readout / router / write_gate / backbone) at each `--log_every` step. Surfaces gradient starvation in the writer subsystem. | `--diagnose_grad_groups` flag; logs `grad/<group>` and prints `|g|/|g_bb|` ratios per step. |
| D2  | Judge attention decisiveness: row-entropy / `log(2K)`, mean keep-vs-write mass, variance over rows, effective rank of the average judge attention pattern. | `--diagnose_memory_dynamics`; uses `MemoryBlock.judge_attention(...)`. |
| D3  | M_c stability per session step (`||M_c^t - M_c^{t-1}||_F / ||M_c^{t-1}||_F`); chain-distinguishability via pairwise normalised Frobenius distance between distinct-chain M_c^Ts. Detects content-blind writer. | Same `_memory_dynamics_eval` pass. |
| D4  | Synthetic gold-standard task: 5000-chain persona-callback corpus, 256-item closed set, 9 sessions/chain. Hard ground truth: `callback_ce → 0` only if memory works. | `tools/build_synthetic_persona_callback.py`; corpora at `paper_artifacts/chains/synthd4_persona_callback_{train,val}_s512.pt`. |
| D5  | TTT-on-readout disambiguator: freeze writer + router + LM head, train ONLY the readout for 300 steps. If callback CE drops, writer encoded the info; readout was the bottleneck. | `tools/d5_ttt_readout.py`. |

## v12 — slot-attention writer (2026-05-01 ~19:00 UTC)

User directive: replace the original judge with slot-attention
(softmax over slots, not over inputs, so slots are forced to
specialise). Spec-strict; m^t stays as foundational source `b_{-1}`
in the Block-AttnRes pool (Eq. 9). Only judge layer internals fair
game.

### Diagnosis: the original writer is decision-less by construction

D2 on `v11g/best` reported `row_entropy / log(2K) = 0.999` (uniform),
`keep_mean = 0.500`, `eff_rank = 1.02` (all 128 slots collapsed to ~1
direction). The original judge is a single softmax over the inputs
axis; at init, all M_judge rows i.i.d. random, all P rows i.i.d.
random ⇒ `attn[b,k,j] ≈ 1/(2K)`. The **symmetric uniform fixed point
is also a gradient fixed point** because permuting any two slots
leaves both forward and loss unchanged.

### Architectural change

`SlotAttentionWriter` (`src/modeling_memres.py` lines 460–620)
implements Locatello et al. 2020:

```
slots^(0) = M_judge.broadcast(B)
P         = [M_c^{t-1} || M_new]
for t in 1..T:
    q          = W_Q(slot_norm(slots))
    k, v       = W_K/W_V(input_norm(P))
    attn       = softmax(q kᵀ / √d, dim=-2)   # SOFTMAX OVER SLOTS
    attn       = attn / attn.sum(dim=-1)
    updates    = attn @ v
    slots      = GRUCell(updates, slots)
return slots
```

CLI: `--memres_writer_kind {original,slot_attention,slot_attention_full}`
(default `original`); `--memres_slot_attention_iters` (default 3).
Init parity preserved (`results/eval/init_parity_test_v12.json`: 4 new
slot-attention parity cases pass at `max|Δ| = 0.000e+00`).

### v12a-slot-judge-D4 trajectory through step 800

| step | α_mem_max | PA CB Δ_nm-m | PA CB Δ_sh-m | EVID Δ_nm-m_floor | evidence_lift | D2 row_ent_norm | D2 keep_mean | D2 eff_rank |
| ---- | --------- | ------------ | ------------ | ----------------- | ------------- | --------------- | ------------ | ----------- |
| 200  | 0.020     | **+0.386**   | **+0.021**   | +0.388            | -0.003        | 0.998           | 0.500        | 1.02        |
| 400  | 0.029     | +0.023       | +0.003       | +0.033            | -0.010        | 0.999           | 0.500        | 1.01        |
| 600  | 0.033     | -0.046       | -0.008       | -0.041            | -0.004        | 0.999           | 0.500        | 1.01        |
| 800  | 0.045     | -0.003       | -0.000       | +0.005            | -0.008        | 0.999           | 0.500        | 1.01        |

**The slot-attention writer also collapses to the symmetric uniform
fixed point.** PA CB Δ_nm-m=+0.386 at step 200 was a real but
transient signal: the slot-axis softmax provides symmetry breaking
*at init* but the GRUCell uses shared weights across slots, so as
training progresses slot states drift toward each other and the
softmax returns to ~1/K. D3-MC says M_c IS chain-specific
(pair=0.015) and dynamic (Δ_step=1.55), so the writer is producing
*different* outputs per chain, but those outputs are not in
content-relevant directions — same chain-identity-hash failure.

### v12d retargeted onto D4

After v12a's collapse, v11p (chinchilla mega frozen) was cancelled and
v12d was redesigned as a frozen-vs-trained backbone single-knob study
on the cleanest possible corpus (D4 synthetic persona-callback,
log(256)=5.55-nat floor). v12d_frozen produced the only positive
v12-era signal: `evidence_lift +0.03` (the baseline v13 needed to
beat).

## v13 — writer warmup + symmetry break (2026-05-01 ~19:20 UTC-5)

### CRITICAL BUGFIX 2026-05-01 ~22:45 UTC-5 — config override silently dropped

`_build_model` in `train_chain.py` was detecting "is this a memres
checkpoint?" with try/except around `Qwen3MemResConfig.from_pretrained`.
Because `Qwen3MemResConfig` subclasses `Qwen3Config`, it loads fine
from a plain Qwen3 config.json — the try never throws. The subsequent
`overridable` merge then DROPPED CLI overrides for every architecture-
shape flag not in the explicit allow-list:

```
memres_mode, memres_writer_kind, memres_slot_positional,
memres_update_mode, memres_extract_source, memres_num_vectors,
memres_extraction_depth, memres_num_blocks, memres_slot_attention_iters
```

Net effect on every v13 cell pre-fix:
- `--memres_mode simple_gate` → silently ran `attention_parity`
- `--memres_writer_kind slot_attention` → silently ran `original`
- `--memres_slot_positional` → silently ignored
- `--memres_extraction_depth 4` → silently ran L_E=0

Fix: detect `from_memres_ckpt` by `base_cfg.model_type == "qwen3_memres"`
(from the raw config.json), not by from_pretrained success.

Bonus side fix in `_set_mem_bias`: in `simple_gate` mode the router's
`mem_bias` doesn't control the forward path; force-opening it had no
effect. Now also sets `memory_gate.gate = 0.5·tanh(bias/2) ≈ 0.48` for
simple_gate.

Effect on post-fix v13a restart vs the buggy run:

|                       | v13a (buggy)           | v13a (fixed)         |
|-----------------------|-----------------------:|---------------------:|
| `mode` (actual)       | attention_parity       | **simple_gate**      |
| `writer_kind`         | original               | **slot_attention**   |
| `slot_positional`     | False                  | **True**             |
| L_E                   | 0                      | **4**                |
| trainable memres      | 9.76M                  | **28.91M**           |
| loss @ step 80        | 13.25                  | **3.72**             |
| `gate_mean` @ phase 1 | 0.0000 (bug)           | **+0.4824**          |

### v13 design — three orthogonal interventions on the OSR triad

The persistent symmetric-uniform-attractor collapse has three causes
(O = Objective starvation, S = Symmetry, R = Routing). v13 attacks all
three at once:

- **O.** New `writer_warmup` phase (500 steps): freezes backbone +
  embed + LM head, forces `mem_bias = 4` AND
  `memory_gate.gate ≈ 0.48` (simple_gate fix), trains the entire
  memres subsystem directly against the LM objective. At step 500 the
  bias anneals to its configured init over 200 steps, then phase 2 is
  joint.
- **S.** `memres_queries_init=orthogonal` (`nn.init.orthogonal_` on
  M_in/M_judge — BF16 upcast to FP32 for the QR op, then cast back).
  `memres_slot_positional=True` adds a deterministic Fourier pattern
  as a learnable per-slot positional offset added to `q_seed` before
  expansion, giving every slot a unique identity that the optimiser
  cannot permute away.
- **R.** Initially `--memres_mode simple_gate` to take m^t out of the
  depth softmax entirely; later reverted to `attention_parity` after
  the user's "v3 routing is definitively better" reminder + Exp 1
  pair-recipe Table 2 cited.

Init parity preserved (`max|Δlogits| = 0` for both modes with orth+pos
init).

### v13 cells & verdicts

| cell | wall-clock | stack | step | result | verdict |
|---|---|---|---:|---|---|
| **v13a (buggy)** | killed step 980 | "AP + warmup + orth", silently dropped slot/pos/L_E | 980 | partial v13 stack | KILLED post-bugfix |
| **v13a (fixed)** | killed step 500 | SG + full v13 stack | 500 | mode=simple_gate, gate_mean=+0.48 verified; train_loss=2.49, eval_loss=5.54 (mem hurts by 4.4 nats) | classical chain-hash overfit on 28.9M writer × 5000-chain × 500 warmup |
| **v13c (buggy)** | finished 4000 | "AP + warmup + orth", silently dropped slot/pos/L_E | 4000 | peak `evidence_lift +0.22` at step 600 (warmup anneal); collapsed to +0.0037 at step 4000; D2 entropy = 0.999 | NOT a valid test of the headline v13 stack |
| **v13c2** | finished 4000 | full v13 AP stack on D4, trained backbone | 4000 | mid-warmup (step 400) **`evidence_lift = +1.4085`** (6.4× any prior measurement). Phase-2 transition at step 500 produced `grad_norm = 6.5e8` pre-clip; `evidence_lift` collapsed to −0.56 at step 600 | symmetry break works during warmup; phase-2 unfreeze destroys it |
| **v13r** | killed step 10500/16000 | full v13 AP stack on mega; warmup 3000 + anneal 1000 | 10500 | `evidence_lift +0.006` sustained, PA-EVAL CB Δ_sh-m +0.021 ceiling; D2 row_entropy / log(2K) = 0.988 (re-collapsed); D3-MC `Δ_step mean = 0.028` (44× smaller than v13c2's 1.244 — the judge converges to "always preserve old memory" on natural prose) | **long-warmup hypothesis falsified** |
| **v13q** | superseded | AP + FROZEN backbone + curriculum + mega + 6000 steps | — | — | superseded by v14 frozen-backbone experiments |
| **v13b/d/p** | dequeued | SG-side ablations | — | — | redundant once v14 tests all 4 interventions on AP |

### v13 headline finding

The writer CAN specialise per chain (v13c2 `evidence_lift +1.4`,
D3-MC `pair/self = 0.004` sustained through v13r's 10500 steps —
orthogonal init + slot_positional is a permanent symmetry break).
But two new failures appeared:

1. The **router actively rejects memory during joint training** (v13r
   step 10000 `α_mem_mean = 0.0011` — textbook MoE expert collapse;
   nothing in the architecture obligates the router to recruit).
2. The **judge re-collapses to uniform when the backbone unfreezes**
   (D2 row_entropy/log(2K) = 0.988 on v13r vs 0.890 on v13c2's
   frozen-backbone D4 regime — attention-entropy collapse direction
   from Zhai et al. 2023).

Both motivate v14.

## v14 — router recruits, judge decides, writer discriminates (2026-05-02 ~15:15 UTC-5)

Four orthogonal interventions, each with prior-art backing:

1. **`alpha_mem` floor auxiliary loss** (weight=0.01, target=0.05) —
   MoE load-balance penalty. Router obligated to keep sampling memory
   so downstream LM gradient keeps reaching the writer/readout. Cite:
   Fedus et al. 2021 Switch Transformer; Wang et al. 2024.
2. **InfoNCE contrastive loss** (weight=0.5, callback-only) — dense
   discriminative signal; B-way over in-batch negatives. Cite:
   AutoCompressors (Chevalier et al. EMNLP 2023), TRIME (Zhong et
   al. EMNLP 2022).
3. **Judge QK-LayerNorm** — post-projection RMSNorm on Q/K of
   `MemoryBlock.judging`. Decouples attention-logit magnitude from
   W_Q/W_K spectral norm. Cite: Zhai et al. ICML 2023 σReparam;
   Qwen / Gemma / DeepSeek-V3 attention convention.
4. **AP `router_mem_bias` warmup anneal** — force-held at +4 during
   the 500-step writer warmup, annealed to 0 over 200 steps. Parallel
   fix to simple_gate's memory_gate force-open from v13.

Code shipped in `src/modeling_memres.py` and `src/train_chain.py`:
`--memres_judge_qk_layernorm`, `--alpha_mem_floor_aux_weight`,
`--alpha_mem_floor_target` flags; `CrossAttention` /
`SlotAttentionWriter` accept `qk_layernorm`; `MemoryBlock.__init__`
threads `judge_qk_layernorm` into `self.judging`. Backwards-compat
Identity when flag off.

### v14abl_a / v14abl_b — D4 frozen-backbone ablation pair

Both KILLED 2026-05-02 ~19:45 UTC-5 at step ~3400/4000. Single-knob
isolation of `--memres_judge_qk_layernorm`.

Last full eval block @ step 3200:

|                              | **v14abl_a** (judge_qk_ln ON) | **v14abl_b** (judge_qk_ln OFF) |
|------------------------------|------------------------------:|--------------------------------:|
| EVAL `mem`                   | 1.1052                        | 1.2174                          |
| EVAL `nomem`                 | 1.0268                        | 0.7777                          |
| EVAL `Δ_nm-m`                | **−0.0784**                   | **−0.4397**                     |
| PA-EVAL CB `Δ_nm-m`          | +0.3857                       | −0.0869                         |
| PA-EVAL CB `Δ_sh-m`          | +0.0000                       | −0.0015                         |
| EVID-EVAL `evidence_lift`    | +0.0000                       | +0.0007                         |
| ROUTE `α_mem_max`            | 0.0213                        | 0.0684                          |
| READOUT `‖m^t‖/‖embed‖` mean | **0.000**                     | **3.868**                       |
| D2-JUDGE row_entropy/log(2K) | **0.997** (uniform)           | **0.887** (decisive)            |
| D2-JUDGE keep_mean / var     | 0.500 / 4e-4                  | 0.755 / 0.155                   |
| D2-JUDGE eff_rank            | 1.23                          | 2.57                            |
| D3-MC `self ‖M‖`             | **0.000**                     | **1.000**                       |
| D3-MC `pair/self`            | nan (writer = 0)              | **0.005**                       |
| D3-MC `Δ_step` mean / max    | 0.000 / 0.000                 | 0.258 / 1.141                   |

**Findings**

1. **`--memres_judge_qk_layernorm` is anti-causal on the D4
   frozen-backbone regime.** v14abl_a's writer never lifted off zero
   (`self ‖M‖ = 0`, `‖m^t‖/‖embed‖ = 0`), the judge stayed at the
   uniform fixed point (`row_entropy/log(2K) = 0.997`), and PA-EVAL
   CB `Δ_sh-m` was identically zero. The post-Q/K RMSNorm in
   `MemoryBlock.judging` appears to interact with the slot-attention
   writer's GRU update such that `write_gate` stays inert. **Ship
   default OFF until investigated.**
2. **Without judge QK-LN (v14abl_b) the writer DOES specialise per
   chain.** D3-MC `pair/self = 0.005` (chains separable by `M_c`
   Frobenius distance), `Δ_step mean = 0.258` (writer moving M_c
   non-trivially), readout magnitude alive (`‖m^t‖/‖embed‖ = 3.87`),
   judge decisive (`keep_var = 0.155`, `eff_rank = 2.57`). The
   slot-attention writer + orth/positional + InfoNCE stack works;
   QK-LN was the single-knob blocker.
3. **But the standard EVAL gets *worse* with memory in v14abl_b**
   (`Δ_nm-m = −0.44`). The writer's specialised content is not
   LM-useful on the standard eval; PA-EVAL CB `Δ_sh-m` oscillates
   around 0. The InfoNCE objective is satisfying itself with
   chain-distinguishable M_c that doesn't translate to LM benefit.

### v14a — GH200 mega + trained backbone

Killed at step ~1760/8000. Same `‖m^t‖/‖embed‖ = 0`, `gate_mean = 0`,
`D3 mean = 0`, `evidence_lift = 0` collapse signature as v14abl_a
(QK-LN on); router open per the alpha_floor aux but writer dead.

### v14g..l — D4v2 second wave (2026-05-02 evening)

Six cells exploring warmup × norm × strong-warmup × no-warmup ×
slot-vs-cross writer × InfoNCE-α-floor combinations.

| cell | preset | knob highlight | result |
|---|---|---|---|
| **v14g** | qwen3-0.6b FROZEN | norm ON, warmup 200 | mid Δ_sh-m, ~0 evidence_lift |
| **v14h** | qwen3-0.6b FROZEN | norm OFF, warmup 200 | mid Δ_sh-m, ~0 evidence_lift, slightly worse |
| **v14i** | GH200 FROZEN | warmup_router_bias 8.0 | recent-bias lock-in, neg lift |
| **v14j** | qwen3-0.6b FROZEN | warmup 0, norm ON, slot writer | **mid lift (+0.04)** |
| **v14k** | qwen3-0.6b FROZEN | warmup 0, norm ON, slot writer, alpha-floor + InfoNCE | **`evidence_lift +0.071`** ✅ |
| **v14l** | GH200 FROZEN | as v14k but writer=cross_attention (no slot) | similar lift, slightly noisier |

v14k is the first reproducibly positive evidence_lift checkpoint of
the project. The standalone-eval discrepancy uncovered while debugging
v14k → built `tools/eval_callback.py` (mirrors in-trainer
`pa_cb_*` metric — only callback-token positions, evidence-redacted
floor). On v14k_best it reports `pa_cb_dnm = +1.44`,
`evidence_lift = +0.071`, **38× higher than `eval_chain.py`** which
averages CE over the entire `score_tail_frac=1.0` window (4 sessions).
The localised callback-token effect was being diluted into noise.

`tools/eval_callback.py` is now the canonical post-train eval for
D4-style corpora.

---

# Part VII — runs.md folding convention

When the active `runs.md` ledger gets crowded (rule of thumb: > ~300
lines, or > 2 finished campaigns ahead of the live one), fold the
oldest superseded campaign into a new Part of this file. Each Part
mirrors the original `runs.md` entry per cell with:

- Planning prose **dropped** (hypotheses, decision triggers, launch
  ETAs, queue-management commentary).
- Result tables **kept** (final-step PA-eval, EVID-eval, ROUTE,
  READOUT, D1-D5 readings).
- Mechanism statements **kept** (the one-line "what we learned").
- Inline launch flags **kept** (so the config remains reproducible
  from this file alone).

The active `runs.md` should fit in a single screen of section
headings.
