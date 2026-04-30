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
