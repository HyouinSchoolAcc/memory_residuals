# Memory Residuals — Status, Failures, and a Compute-Targeted Plan Forward

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
- **ReZero-style gated injection** (the variant the codebase actually
  defaults to via `memres_mode="residual"`):
  $h^{\text{pre}}_\ell = h^{\text{post}}_{\ell-1} + g_\ell \cdot m^t$.
  All per-sublayer scalar gates $g_\ell$ are zero-initialized so the
  augmented model is *exactly* equivalent to the bare backbone at step 0,
  while $W_V^{\text{read}}$ retains its default normal init so $m^t$ has
  non-zero magnitude and gate gradients flow non-trivially from step 1.

The "residual" mode was introduced specifically because the *default*
Block AttnRes init (zero pseudo-queries, single negative bias on the
memory source) leaves all $N$ non-memory sources with weight $\approx
1/N$ in the softmax. That uniform average over the embedding, every
prior block output, and the running intra-block partial is **not** the
same forward pass as the bare backbone — which at sublayer $(n,i)$
takes the single accumulated state $h_{n,i-1}$ as input — so the
pretrained residual-stream conditioning is disturbed at step 0 and
several thousand steps of warm-up are needed to recover it.

With the ReZero/`residual` gate at zero, attaching MemRes to a Qwen3
checkpoint is *exactly* equivalent to the bare backbone at step 0
(both forward and backward through the trunk), and the gradient into
the memory module itself is identically zero at step 0 because $g_\ell
= 0$ multiplies $\partial L / \partial h_\ell$ before it reaches
$W^{\text{read}}_{V}$. This is the "provably non-disruptive at
$t{=}0$" property, verified by
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

Both options are now implemented and tested side-by-side:

| mode | $\max\lvert\Delta_{\text{logit}}\rvert$ vs bare Qwen3-0.6B (bf16) | verdict |
| --- | --- | --- |
| `residual` (ReZero gate $g_\ell{=}0$), no memory | $0.000$ | bit-exact parity |
| `residual` (ReZero gate $g_\ell{=}0$), memory attached | $0.000$ | bit-exact parity |
| `block_attnres` default (uniform softmax over deltas), no memory | $34.5$ | massively perturbed |
| `block_attnres` default (uniform softmax over deltas), memory | $34.4$ | massively perturbed |
| `block_attnres_parity_init` (cumulative pool + recent_bias = +32), no memory | $0.000$ | bit-exact parity |
| `block_attnres_parity_init` (cumulative pool + recent_bias = +32), memory | $0.000$ | bit-exact parity |

(Source: `paper_artifacts/eval/init_parity_test.json`, reproducible
via `python paper_tools/init_parity_test.py`.) The
`+32`/`-32` bias magnitudes are needed because per-step softmax
leakage compounds across all $2L = 56$ routed sublayers; at
recent_bias $=+16$ each off-source still carries $\sim e^{-16}/N
\approx 3\!\times\!10^{-7}$ of mass, which feeds back into the next
sublayer's input and accumulates to $\sim\!0.31$ in the final
logits. At $+32$ the off-source mass drops to $\sim e^{-32}\approx
1.3\!\times\!10^{-14}$, well below bf16 precision, and parity is
exact.

The trade-off is that `block_attnres_parity_init` puts the model in
a saturated-softmax regime: the per-source pseudo-queries $w_{n,i}$
get effectively zero gradient at step 0, so the router can only
learn by first relaxing the bias, which makes it an objectively
clunkier place to start training a *recurrent* memory module. This
is why the `residual` mode remains the recommended default for
attaching MemRes to pretrained checkpoints — it avoids the warm-up
problem entirely with one learnable scalar per sublayer. The
`block_attnres_parity_init` mode is included for direct comparison
and as the natural starting point for from-scratch ablations of the
full Block AttnRes architecture.

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
├── output/                        trained checkpoints, see §2 below
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
| `output/run3_qwen3-0.6b-large/` | `train_phase1.py` (pair) | bare Qwen3-0.6B | PG-19+TV pairs (h=1024, c=512) | 8000 | $\Delta_{\text{nm-m}}{=}{+}0.026$, $\Delta_{\text{sh-m}}{=}{+}0.029$, callback ratio **1.77×** on **pair** eval ($n{=}256$); but on long-horizon **chain** eval explodes to $\text{CE}_{\text{mem}}{=}8.7$ vs $\text{CE}_{\text{nomem}}{=}2.5$ | Looks great on its own training distribution, **catastrophic OOD** to the actual chain regime |
| `output/chain2_qwen3-0.6b-large/` (deleted; numbers in `chain2_eval_pg19_locomo.json`) | `train_chain.py` warm-started from `run3`, judge RMSNorm added | warm | PG-19+TV chains, $k{=}4$ | 3000 | $\Delta_{\text{sh-m}}{=}{-}0.014$ PG-19, ${-}0.012$ LoCoMo | Stable but **PITFALLS §3 shortcut-learning failure** — memory became style-only |
| `output/chain_fresh1/` | fresh, no warm start | bare | PG-19+TV chains, $k{=}8$ | 5000 | $\Delta_{\text{nm-m}}{=}{+}0.008$ PG-19, ${-}0.021$ LoCoMo; $\Delta_{\text{sh-m}}{=}{-}0.036$ PG-19, ${-}0.016$ LoCoMo | Stable at 30+ session unrolls but **shuffle goes negative** — same shortcut-learning failure |
| `output/chain_tv1/`, `chain_tv2/` | TV-only | fresh | TV chains only, $k{=}8$ | 6000 | LoCoMo $\Delta_{\text{sh-m}}{=}{+}0.011, {+}0.002$ at step 500, then **overfits** | Domain-shift ablation only |
| `output/chain_neg1/` (latest) | `train_chain.py` + **negative-chain contrastive** ($\lambda{=}0.5$, margin $0.05$) | fresh | PG-19+TV chains, $k{=}8$ | 1000 | In-trainer (n=60): $\Delta_{\text{nm-m}}{\approx}0$, $\Delta_{\text{sh-m}}{=}{+}0.014$ — channel-collapse on aggregate, marginal shuffle gap | **Most promising recipe so far**, but unfinished; see open question #1 below |

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
   `output/chain_2h100_final/best/`, complete eval JSONs in
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
7. `memres_mode` $\in \{$residual, block_attnres$\}$ comparison (1 run).
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
(`output/run3_qwen3-0.6b-large/best/`) that satisfies all four
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
