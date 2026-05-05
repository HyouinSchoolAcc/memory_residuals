# Paper B — Attention Parity Beats ReZero: Init-Preserving Depth-Wise Routing for Recurrent Memory in Pretrained LLMs

**Status:** plan (2026-05-04). Targets NeurIPS 2026 **Main Track** as
`General` or `Concept & Feasibility` contribution. Best workshop fit
(if main-track flips to a workshop pivot): **NeurIPS 2026 Workshop on
Long-Context and Memory in Foundation Models** or any "Foundation
Models / Architecture" workshop. The most defensible reframe for a
workshop is `Methodological research`: the architectural primitive is
the contribution, the head-to-head is the evidence.

---

## TL;DR

> Three concrete instantiations of a depth-wise attention-residual
> memory router on a frozen Qwen3-0.6B backbone, trained recurrently
> on ~113 M PG-19 + TV-dialogue tokens: a softly-initialised
> attention-residual router (`attention_parity`) learns
> history-specific memory **~2× more sample-efficiently** than a
> per-sublayer ReZero gate (`simple_gate`), while a delta-source
> attention router (`attention_base`) without parity-preserving init
> never recovers from its 34-nat init perturbation at matched compute
> and fails to surpass the bare backbone. Held-out probes confirm
> that parity-init opens **3× more routing mass** on the memory
> source, produces a positive memory-vs-shuffle gap localised to mid-
> network sublayers (top sublayer α_mem = 0.0043), and a held-out
> next-token NLL that is **2.5× more causally sensitive** to memory
> perturbations than the no-parity baseline.

## One-line headline claim

`At step-0 bit-exact backbone parity, a softmax-pooled depth-wise
router with parity-preserving init learns chain-specific recurrent
memory ~2× more sample-efficiently than a ReZero scalar gate, and
the load-bearing ingredient is the parity-preserving init itself —
without it the same router never opens the channel.`

---

## Why this is its own paper

`Paper A` and the central memres paper `P1` both argue *whether*
memres works (yes: +1.32 nats on LME-S, beats RAG). **Paper B asks
the orthogonal architectural question: *which routing primitive*
makes a depth-wise attention residual learn memory at all under a
realistic compute budget?**

The contribution is *not* a new architecture. It is a head-to-head
comparison of three published-style routing primitives — a per-
sublayer ReZero gate (`simple_gate`), a vanilla Block-Attention-
Residual router with no parity init (`attention_base`), and the
same router with parity-preserving init (`attention_parity`) — under
strictly matched optimiser settings, seed, training data, and step
budget. The result is the soft `attention_parity` variant is the
only one that empirically opens a chain-specific memory channel
within ~2000–4400 steps of plain TBPTT next-token NLL on PG-19+TV.

This paper inherits the manuscript already drafted at
`results/exp1_pair_recipe/manuscript.tex` (12 pp). The plan below
re-skins it for NeurIPS 2026 with a 9-page main + appendix split,
adds the held-out routing-mass and counterfactual-sensitivity probes
(already implemented in `paper_tools/routing_trace.py` and
`paper_tools/counterfactual_eval.py`), and threads the "init parity"
finding as the load-bearing claim.

---

## Story / outline (9 main pages)

### §1 Intro (1 page)

A pretrained Transformer's residual stream is a finely-balanced
construction. Augmenting it with an off-sequence memory readout `m_t
∈ R^{S×d}` requires choosing **how `m_t` enters the residual stream
at every depth**. Three routing primitives are concretely available
to the architecture-design literature:

- **`simple_gate`** — a per-sublayer ReZero scalar gate, `h ← h + g_ℓ
  · m_t`, `g_ℓ` initialised at zero. Bit-exact step-0 backbone
  parity at all sublayers.
- **`attention_base`** — the Block-Attention-Residual router (Du et
  al. 2025) with `m_t` registered as the foundational source `b_{-1}`
  alongside per-block delta sources `b_0, …, b_{n-1}`. Uniform-softmax
  init perturbs final-layer logits by ~34 nats.
- **`attention_parity`** — the same router with two coupled changes:
  (i) the value pool stores cumulative hidden-state checkpoints (so
  the residual stream is recoverable as a one-hot select on the
  most-recent slot), and (ii) two scalar bias parameters
  `mem_bias`, `recent_bias` initialise the softmax one-hot on the
  most-recent source. Bit-exact step-0 parity at strict ±32 bias;
  the **soft** variant uses ±4 bias, sacrificing exact parity for
  non-saturated softmax gradients from step 1.

**Empirical question:** under matched compute (~113 M PG-19 + TV
tokens, ~6000 TBPTT steps, single H100), which router learns
chain-specific recurrent memory most sample-efficiently?

**Answer:** soft `attention_parity` reaches `simple_gate`'s
asymptotic shuffle-gap plateau in ~2/5 the steps and continues
climbing; `attention_base` spends most of its budget reabsorbing the
init perturbation and at matched compute does not surpass the bare
backbone. Held-out probes confirm the difference is mechanistic, not
just numerical.

**Contributions.**

1. A clean architectural decomposition of the three routing
   primitives, showing where each fits in the existing literature
   (ReZero, AttnRes, RIMs).
2. A bit-exact init-parity verification table (Table 1):
   `simple_gate` and strict `attention_parity` both achieve
   `Δ_logit < 10^{-6}`; `attention_base` perturbs by 34 nats.
3. A matched-compute head-to-head on Qwen3-0.6B trained recurrently
   on PG-19 + TV: `attention_parity` (soft) is 1.6×–3.8× ahead of
   `simple_gate` on every in-trainer eval point, asymptotically
   surpassing `simple_gate`'s plateau.
4. **Two held-out mechanistic probes**: a routing-mass trace
   (where on the network does the router put weight on the memory
   source under true vs shuffled memory?) and a counterfactual
   sensitivity (does perturbing the prior session change the
   held-out next-token NLL?). Parity-init opens 3× more mass and
   2.5× more causal sensitivity than no-parity. Without parity init
   no such structure forms.
5. A clean **negative result** for `attention_base`: the same
   architecture without parity init does **not** learn a memory
   channel under realistic compute budgets. The architectural
   primitive *needs* the init.

### §2 Background and related work (0.75 page)

Pulled from `manuscript.tex` §2, trimmed to fit:

- **Recurrent memory in Transformers** — Transformer-XL, Compressive
  Transformer, RMT, Block-Recurrent, Memorising Transformers, LRMT
- **Block Attention Residuals** — Du et al. 2025
- **Building blocks** — Perceiver-style fixed-bandwidth latents,
  ReZero scalar gates, Mamba SSMs (single-token recurrent state).
- **Soft routing across multiple sources** — Mixture-of-Experts /
  routing softmax literature; the parity-init trick is to seed the
  router with a strongly-biased init that lets the first gradient
  step move it off the all-mass-on-recent baseline without
  saturating its softmax.

### §3 Memory Residuals architecture (1 page)

Self-contained spec for a reader who has not seen `P1` or `Paper A`:

- Stage-1 extract (Eq. 1): K-slot Perceiver cross-attention into
  current session, optionally L_E rounds of refinement. K = 128.
- Stage-2 judge (Eq. 2): 2K-pool cross-attention with separate
  judge queries. Softmax across the row dimension implements
  "zero-sum forgetting".
- Off-sequence read (Eq. 3): token-level cross-attention into M_c,
  output `m_t ∈ R^{S × d}`.

Then the **three routing primitives** in detail:

- `simple_gate`: `h^{pre}_ℓ = h^{post}_{ℓ-1} + g_ℓ · m_t`, `g_ℓ`
  initialised at zero. Memory pathway = a single learnable scalar
  per sublayer.
- `attention_base`: `m_t` registered as `b_{-1}` alongside
  `b_0, …, b_{n-1}` per-block delta sources. The next pre-residual
  hidden is a softmax-weighted mix over the pool. Vanilla
  uniform-softmax init.
- `attention_parity`: same router as `attention_base`, but
  - **value pool stores cumulative hidden-state checkpoints**:
    `b_0 = h_0, b_k = h_k`, plus the running intra-block partial
    `h_{n,i-1}`. The residual stream `h = b_0 + Σ b_k + h_{n,i-1}`
    is then expressible as a one-hot selection on the most-recent
    slot.
  - two scalar bias parameters `mem_bias`, `recent_bias` at strict
    ±32 (bit-exact init parity) or soft ±4 (non-saturated softmax).

### §4 Init parity verification (0.5 page)

Table 1 (from `manuscript.tex`):

| mode | no memory | with memory |
|---|---|---|
| `simple_gate` (g_ℓ = 0) | 0.000 | 0.000 |
| `attention_base` (uniform pool) | 34.5 | 34.4 |
| `attention_parity` (strict ±32) | 0.000 | 0.000 |
| `attention_parity` (soft ±4) | ≈ 10^{-5} | ≈ 10^{-5} |

Bit-exact parity is the architectural requirement for "additive on
top of the bare backbone" to hold at step 0. The 34-nat perturbation
is what `attention_base` has to undo before it can do useful work.

### §5 Experimental setup (0.5 page)

- **Backbone:** Qwen3-0.6B, frozen.
- **Memory hyperparameters:** K = 128, L_E = 4, judge RMSNorm on,
  N = 8 attention-residual blocks for `attention_*`.
- **Optimiser:** AdamW (β₁, β₂ = 0.9, 0.95), wd = 0.1, cosine LR
  schedule, 200 warmup, 5 % min ratio. Memory params η_m = 3e-4,
  backbone η_b = 3e-5.
- **Data:** 4995 PG-19 books (chapter-cut, ~470 M Qwen tokens) +
  30 high-continuity TV transcripts (~16 M tokens), packed into
  one chain per book/show at S = 512.
- **Training:** TBPTT k = 8, B = 4 chains, gradient accumulation 4
  (effective batch 16), 6000 steps, single H100 80 GB at bf16
  with gradient checkpointing. ~10.8k tok/s, ~10 wall-clock hours per
  run. **All three runs share seed 42 and data shuffling**, so any
  trajectory difference is attributable to the routing primitive
  alone.
- **Eval:** PG-19 val (47 chains), PG-19 test (95 chains), LoCoMo
  (9 chains, MC10 split). Four CE configs per scored session:
  `CE_mem`, `CE_nomem`, `CE_shuffle`, `CE_oracle`. Headline metrics:
  `Δ_nm = CE_nomem - CE_mem` (aggregate help) and `Δ_sh =
  CE_shuffle - CE_mem` (history-specificity).

### §6 Routing-primitive head-to-head (1.5 pages, headline section)

Trajectory plot (Fig. 1, `figures/trajectory.pdf`): in-trainer eval
of `Δ_nm` and `Δ_sh` over training steps for the three primitives
on Qwen3-0.6B.

Table 2 — in-trainer eval `Δ_sh` trajectory:

| step | `simple_gate` | `attn_parity` (soft) | ratio |
|---|---|---|---|
| 200  | +0.0002 | −0.0001 | (init noise) |
| 400  | +0.0011 | +0.0042 | 3.8× |
| 600  | +0.0048 | +0.0089 | 1.9× |
| 800  | +0.0068 | +0.0107 | 1.6× |
| 1000 | +0.0104 | +0.0177 | 1.7× |
| 1200 | +0.0103 | +0.0164 | 1.6× |
| 1400 | +0.0103 | +0.0191 | 1.9× |
| 1600 | +0.0123 | +0.0264 | 2.1× |
| 1800 | +0.0118 | +0.0240 | 2.0× |
| 2000 | +0.0138 | **+0.0272** | 2.0× |

Key observation: at step 2000, `attention_parity` (soft) has
already exceeded `simple_gate`'s asymptotic plateau (Δ_sh = +0.0249
at step 5200, separate run). Roughly 2/5 the compute, still
climbing.

`attention_base` has not yet reached zero on Δ_nm by step 2000 — it
is still recovering from its 34-nat init perturbation.

### §7 Standalone eval (0.5 page)

Table 3 — rigorous standalone eval after training (from
`manuscript.tex` Table 4):

| ckpt | corpus | Δ_nm | Δ_sh | Δ_or | capture |
|---|---|---|---|---|---|
| soft `attn_parity` (step 2000) | PG-19 val | +0.0113 | −0.0329 | −0.0633 | 0.15 |
| soft `attn_parity` (step 2000) | PG-19 test | +0.0116 | +0.0061 | −0.0693 | 0.14 |
| soft `attn_parity` (step 2000) | LoCoMo | +0.0013 | −0.0161 | −0.1809 | 0.01 |
| `simple_gate` (step 5200) | PG-19 val | +0.0098 | −0.0348 | −0.0629 | 0.13 |
| `simple_gate` (step 5200) | PG-19 test | +0.0100 | +0.0033 | −0.0677 | 0.13 |
| `simple_gate` (step 5200) | LoCoMo | −0.0020 | −0.0137 | −0.1931 | −0.01 |

Three readings:

1. On PG-19 test, parity beats gate by ~1.85× on Δ_sh despite
   ~2.6× less training compute.
2. On PG-19 val (in-domain) both routes have negative Δ_sh — known
   pattern consistent with partial style-encoding shortcut.
3. LoCoMo (n = 40 score positions, OOD) shows neither route has
   robustly positive Δ_sh at this checkpoint. We acknowledge OOD as
   an explicit limit (§9).

### §8 Held-out mechanistic probes (1.5 pages — the new claim)

Two probes that target *behaviour* rather than output (from
`manuscript.tex` §6.3):

**Routing-mass trace (Table 4).** For every score position, forward
with `collect_alpha_trace=True` and record the softmax mass
`α_mem` on `b_{-1}` at every routing sublayer, under both true and
shuffled memory, aggregated over n = 156 score positions across 40
held-out chains.

| ckpt (step 4400) | mean α_mem | mem-vs-shuffle gap | × init floor |
|---|---|---|---|
| soft `attn_parity` | 4.7e-4 | +4.79 % | 7.6× |
| `attn_base` | 1.5e-4 | −0.40 % | 2.3× |

Three observations:

1. Parity-init has moved 7.6× off the analytic init floor of
   `exp(−8)/N ≈ 6.2e-5`; `attn_base` only 2.3× despite matched
   compute. The router does open up under parity init.
2. Parity-init has a **+4.79 %** mem-vs-shuffle gap; `attn_base`
   has −0.40 %, indistinguishable from zero. The parity router
   differentiates true from foreign memory; the no-parity router
   does not.
3. The gap localises to **mid-network sublayers** {32, 9, 20, 6, 14}
   (top-five gaps). Largest single sublayer's α_mem^true = 0.0043.
   `attn_base`'s largest gap across all sublayers is +0.00004.

**Counterfactual sensitivity (Table 5).** For each chain c with L ≥
2 + d, rebuild M_c from the true prefix vs a perturbed prefix where
slot L−1−d is replaced with a uniformly-sampled session from a
different chain. Δ(d) = NLL_perturbed - NLL_normal.

| ckpt (step 4400) | d=1 | d=2 | d=4 | d=8 | d=ALL |
|---|---|---|---|---|---|
| soft `attn_parity` | +0.0094 ± 0.014 | +0.0008 | +0.0003 | +0.0000 | **+0.0127 ± 0.010** |
| `attn_base`        | +0.0031 ± 0.006 | +0.0002 | +0.0001 | −0.0001 | +0.0050 ± 0.006 |

Parity-init held-out loss is **2.5× more causally sensitive** to
its own memory than the no-parity baseline. Both probes agree.

### §9 Discussion & limitations (1 page)

Discussion: parity-init is not a numerical safety device. It biases
optimisation toward a memory-using minimum. Without it, 4400 steps
of plain next-token NLL on PG-19 do **not** open the channel.

Limitations:

- 0.6B-class result. 1.7B / 8B-class verification is ongoing in the
  companion paper (P1 / Paper A) but uses a different training
  cell (LME-S not PG-19) so the routing-primitive comparison is
  not directly transferable. We acknowledge this and mark it as
  future work.
- Magnitude of held-out causal effect saturates at Δ(ALL) ≈ +0.013
  nat under plain LM-NLL on PG-19; the **strength** of the channel
  (separately from whether the channel opens at all) is bounded by
  the training signal. Auxiliary contrastive losses or chain-corpora
  with explicit callbacks (LME-S, the headline corpus of paper P1
  and Paper A) lift this ceiling. Out of scope here.
- The shuffle test substitutes "next chain index mod N", a uniform
  but non-adversarial reshuffle. Stronger test = adversarially-mined
  similar-style chain; left to future work.

### §10 Conclusion (0.5 page)

We have given a clean architectural decomposition of three concrete
routing primitives for off-sequence memory readout in a frozen
pretrained Transformer; verified that parity-preserving init is
load-bearing both numerically (no init perturbation) and
mechanistically (router opens the memory channel; held-out causal
sensitivity is 2.5× the no-parity baseline); and shown that under
matched compute, the soft `attention_parity` variant is roughly 2×
more sample-efficient than the ReZero gate baseline at learning
chain-specific memory. The strict negative result for `attention_base`
is the empirical case for parity-init being an architectural
choice, not just a safety device.

---

## Numbers ledger

All numbers come from `results/exp1_pair_recipe/manuscript.tex` (the
existing 12-page draft). The eval JSONs are in
`paper_artifacts/eval/`; the routing-trace and counterfactual
evaluators are at `paper_tools/routing_trace.py` and
`paper_tools/counterfactual_eval.py`.

If we re-run any experiment between now and the May 7 PDF deadline,
update **manuscript.tex** and the markdown ledger together.

---

## What still has to land before May 7 PDF deadline

1. **Trim manuscript.tex from 12 pp to 9 pp** main text. Push to
   appendix:
   - Horizon-bucketed analysis (Fig. 2 + Table from Appendix A)
   - Per-sublayer gate-profile figure (Fig. 4)
   - Standalone-eval extra rows (val + LoCoMo)
   - Routing-trace and counterfactual full tables (keep teaser tables
     in main, push n=156 / per-chain decomposition to appendix)
2. **Re-anonymise** — manuscript already says `Anonymous Authors` but
   double-check there are no S;G studio mentions, no GitHub-URL
   leaks, and the `<anonymised-for-review>` placeholder is still in
   place.
3. **NeurIPS 2026 paper checklist** (mandatory).
4. **Cross-reference cleanup.** If P1 / Paper A also goes in,
   add a 2–3-sentence concurrent-submission note in §3 distinguishing
   contributions: P1 / Paper A use a different training cell
   (LongMemEval-S 450 chains) and a different headline metric
   (callback-token CE). Paper B's contribution is the routing
   primitive comparison and the parity-init load-bearingness.
5. **Rebuild supplementary zip** — `pair_recipe_supplementary.zip`
   already exists at the repo root (sha256 `aaef5d671c0c`); verify
   the manifest matches the trimmed paper's appendix references.

## Estimated write-time

- 12 pp → 9 pp trim: 4 h
- Anonymisation + cross-reference: 1 h
- Checklist: 1 h
- Final polish + bib check: 2 h

Total: ~8 h. Achievable May 5–7.

## Submission metadata (OpenReview form fields)

- **Title:** `Attention Parity Beats ReZero: Init-Preserving Depth-Wise Routing for Recurrent Memory in Pretrained LLMs`
- **Track:** Main Track
- **Contribution Type:** `General` (or `Concept & Feasibility` if the
  primitive is read as a high-risk-high-reward design choice; the
  empirical case for parity-init is preliminary at 0.6B scale).
- **Primary Area:** `Deep Learning` → `Architectures`
- **Secondary Area:** `Foundation or Frontier Models` → `Long-Context / Memory`
- **TL;DR:** see top of this file
- **Abstract:** ~250 words, paste from `manuscript.tex` line 46-74,
  trim if portal hard-limits at 1750 chars
- **Keywords:** recurrent memory, attention residual, ReZero, parity
  initialisation, depth-wise routing, history-specificity, frozen
  backbone, Qwen3, PG-19

## Risk register

| risk | mitigation |
|---|---|
| Reviewer reads the n = 156 routing-trace probes as "small sample" | Bootstrap CIs on the routing-mass trace (n = 156 score positions, n = 40 chains). Existing `paper_tools/routing_trace.py` already supports a `--n_bootstrap` flag. Run before May 7 and add CI columns to Table 4. |
| Reviewer demands LoCoMo / MSC head-to-head with Δ_sh > 0 | Already conceded explicitly in §9 — LoCoMo is an OOD eval and the 0.6B run at 4400 steps does not produce Δ_sh > 0 on it. The routing-trace and counterfactual probes are reported on **PG-19 val** where they are clean. Frame as a positive trans-corpus diagnostic, not as a SoTA QA result. |
| `attention_base` looks like a strawman | The clean negative result IS the contribution. We acknowledge `attention_base` is the natural application of the AttnRes paper as published, and the parity-init lesson is exactly that the published primitive needs an init lemma to be useful for off-sequence sources. Cite Du et al. 2025 carefully — this paper extends, not contradicts. |
| Concurrent submission with P1 / Paper A flags as dual-submission | The contribution is orthogonal: P1 / Paper A study a different training cell (LongMemEval-S, frozen-backbone chain training) and a different metric (callback-token CE). Paper B's routing-primitive comparison runs on PG-19 + TV with TBPTT next-token NLL — a different empirical regime. Add an explicit cross-reference paragraph in §3 noting this. |
| 0.6B-only result is too thin for main track | We can re-pitch this as `Concept & Feasibility` — the bar there is preliminary results on a high-risk idea, which fits perfectly. Workshop fallback also remains: the architecture-primitive workshops at NeurIPS 2026 (final list announced 2026-07-11) are the natural venue if main flips. |
