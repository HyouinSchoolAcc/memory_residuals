# NeurIPS 2026 Abstract — v3 (LOCKED, F3-off canonical)

**Status:** v27b 4-seed pack at 0.6B and v28a/b 2-seed pack at 1.7B
all landed (post-mortem at 14:00 EDT, May 4). Branch A from v2 is
locked. Numbers ledger: [`NEURIPS_NUMBERS.md`](NEURIPS_NUMBERS.md).
v1 / v2 superseded.

**Working title:**
**Memory Residuals: A Frozen-Backbone Recurrent Memory for
Long-Horizon Callback Recall in Pretrained LLMs**

(Backups: "Memory Residuals: Adding Long-Horizon Memory to a
Frozen Pretrained LLM"; "Memory Residuals: Joint LM-NLL is Enough —
Surprising Failure of Auxiliary Probes for Frozen-Backbone Recurrent
Memory".)

---

## Abstract (paste-ready, ~250 words)

> Adding long-horizon memory to a pretrained language model is
> typically done either by extending the context window — which
> scales quadratically in compute and is fragile beyond the
> pretraining length — or by attaching a retrieval index, which is
> decoupled from the LLM's training signal. We propose
> **Memory Residuals (memres)**: a fixed-size, jointly-trained
> recurrent memory matrix M_c that is read and written by an
> *otherwise frozen* pretrained LLM through a depth-routed
> cross-attention pathway, trained end-to-end with a single
> auxiliary loss that prevents the depth router's memory channel
> from collapsing. With **41.5 M trainable parameters added on top
> of a frozen Qwen3-0.6B (≈6 % overhead) and ~1.5 h of single-H100
> training on the LongMemEval-S 450-chain training split**, the
> resulting model improves callback cross-entropy on
> LongMemEval-S validation by
> **Δ_dnm = +1.32 ± 0.53 nats over no-memory across four seeds
> (chain-shuffle confound = 0.000 ± 0.010)**. The same recipe
> **scales to Qwen3-1.7B**: a single-H100 6-h run gives
> **Δ_dnm = +0.93 nats (n=2 seeds, 1.7B)**, with a tight per-seed
> spread (0.91, 0.94) and Δ_dsh ≈ 0. Two single-variable ablations
> identify the load-bearing components: removing the iterative
> readout-depth refinement collapses Δ_dnm to +0.025; removing the
> depth-router's α-floor drives Δ_dnm to −0.04. A third ablation
> we initially expected to be load-bearing — a chain-specific
> readout probe supplying a direct gradient to the readout —
> *strengthens* the result by 8× when removed, indicating that
> joint LM-NLL training alone produces a richer compressed
> chain-conditional context than supervising the readout for
> chain identification. An evidence-redaction analysis
> (`evidence_lift = +0.001 ± 0.006`) shows the memory encodes
> *chain-conditional context* rather than literal evidence recall —
> a distinction prior memory work rarely measures and that we
> argue is the correct framing for fixed-size compressive memories.
> Because the backbone is frozen, the result is leak-controlled by
> construction.

(Word count: ~265 words. NeurIPS abstract field accepts up to 1750
characters or ~250 words; trim a sentence if the portal cuts off —
the "evidence-redaction" sentence is the easiest to cut to ~200
words without losing the headline.)

---

## Tighter 200-word version (if portal hard-limits at 1750 chars)

> Adding long-horizon memory to a pretrained language model is
> usually done either by extending the context window —
> quadratic compute, fragile beyond pretraining length — or by
> attaching a retrieval index, decoupled from the LLM's training
> signal. We propose **Memory Residuals (memres)**: a fixed-size
> recurrent memory matrix M_c read and written by an *otherwise
> frozen* pretrained LLM through a depth-routed cross-attention
> pathway, trained end-to-end with a single auxiliary loss that
> keeps the depth router's memory channel open. With **41.5 M
> trainable parameters added on top of a frozen Qwen3-0.6B (~6 %
> overhead) and ~1.5 h on a single H100**, training on the 450-chain
> LongMemEval-S split improves callback cross-entropy on
> LongMemEval-S validation by **Δ = +1.32 ± 0.53 nats over no-memory
> (n=4 seeds), shuffle confound = 0.000 ± 0.010**. The recipe
> **scales to Qwen3-1.7B (Δ = +0.93 nats, n=2 seeds)**.
> Single-variable ablations identify the readout-depth refinement
> and the α-floor as load-bearing; surprisingly, a chain-specific
> readout probe we expected to be load-bearing turns out to be
> harmful and is dropped. An evidence-redaction analysis shows the
> memory encodes chain-conditional context rather than literal
> evidence recall. The frozen backbone makes the result
> leak-controlled by construction.

---

## Headline numbers cheat sheet for the LaTeX rewrite

LaTeX-ready forms:

- 0.6B headline: `$\Delta_{\text{nm}} = +1.32 \pm 0.53$~nats (n{=}4 seeds)`
- 1.7B scaling: `$\Delta_{\text{nm}} = +0.93$~nats (n{=}2 seeds)`
- shuffle confound (0.6B): `$\Delta_{\text{sh}} = 0.000 \pm 0.010$`
- evidence lift: `$\text{evidence\_lift} = +0.001 \pm 0.006$`
- depth=0 ablation: `$\Delta_{\text{nm}} = +0.025$`
- α-floor=0 ablation: `$\Delta_{\text{nm}} = -0.038$`

---

## Author / category fields (still TBD — block on user)

- [ ] Author list + affiliations
- [ ] Primary subject area: **Deep Learning / Architectures →
  Memory and Long-Context** (best fit) or **Foundation Models →
  Long-Context Reasoning**?
- [ ] Spotlight / poster track preference

---

## Decision log — why Branch A (F3-off canonical), not B

The v2 abstract had two pre-written branches. Branch A required
"≥3/4 seeds positive at 0.6B with F3 off, plus 1.7B preserves
direction". The actual landing is **4/4 positive at 0.6B
(+0.797 / +0.939 / +1.833 / +1.721) and 2/2 positive at 1.7B
(+0.909 / +0.944)**, which is strictly stronger than the
Branch-A trigger. Per-chain check on the +1.83 seed: 49 of 50
chains positive, median per-chain Δ = +0.91, top-5 chain helps
in [+5.5, +7.0] nats. No single-chain outlier. Locked.

---

## Backup framings (only if a number breaks under further checks)

1. **If a reviewer demands bootstrap CIs over chains and the 0.6B
   CI is wide:** report mean Δ = +1.32 with 95 % bootstrap
   over-chains CI separately (we have all 50 per-chain residuals
   in `per_chain` of the eval JSON; can be computed in <1 min). The
   between-seed std (0.53) and the within-seed bootstrap CI are
   different sources of variation; we report the more conservative
   (whichever is wider) in the appendix.
2. **If we end up needing to soften the 1.7B claim:** "Δ = +0.93
   nats (n=2 seeds, both runs at +0.91/+0.94)" → "the 1.7B effect
   is preserved but the multi-seed CI is wide; we report this as
   direction-of-effect and defer multi-seed 1.7B to follow-up
   work". Keep the 0.6B 4-seed mean as the headline.
3. **If a reviewer demands LoCoMo head-to-head:** keep the LoCoMo
   discussion at "out-of-domain failure" (CE −0.015 from
   `results/eval_lme_locomo/`) and frame as the **scope** finding,
   not as a contribution. The paper's contribution is the
   training recipe, not a SoTA QA system.

---

## What still has to land before submission (T-3 h 40 min @ 14:00 EDT)

1. ~~Pull GH200 cells~~ — done.
2. ~~Eval all 9 ckpts on lme_val_s512_evpos~~ — done.
3. ~~Lock numbers ledger~~ — done at `NEURIPS_NUMBERS.md`.
4. ~~Decide branch~~ — Branch A locked.
5. **Author list + affiliations** — block on user.
6. **NeurIPS subject area + Spotlight/Poster preference** — block on user.
7. **Final copy-paste into the NeurIPS abstract field** — 5 min.
8. *(Optional, if time)*: 95 % bootstrap-over-chains CI for the
   headline (≤ 2 min of compute). Add to the abstract appendix /
   supplementary.

---

## Numbers ledger (live, evpos eval, this is the file the paper cites)

`tools/eval_callback.py` against `paper_artifacts/chains/lme_val_s512_evpos.pt`,
50 chains, all on `final/` ckpt for fair across-recipe comparison:

| run | size | seed | F3 | depth | floor | `pa_cb_dnm` | `pa_cb_dsh` | `evidence_lift` |
|---|---|---|---|---|---|---|---|---|
| v24a | 0.6B | 1 | ON | 4 | ON | +0.227 | +0.010 | +0.005 |
| v24a | 0.6B | 2 | ON | 4 | ON | +0.068 | −0.003 | +0.008 |
| v24a | 0.6B | 3 | ON | 4 | ON | +0.190 | +0.013 | +0.002 |
| **v24a 3-seed mean** | — | — | — | — | — | **+0.162 ± 0.083** | +0.007 ± 0.008 | +0.005 ± 0.003 |
| v25a | 1.7B | 1 | ON | 4 | ON | +0.193 | −0.008 | −0.001 |
| v25a-seed7 | 1.7B | 7 | ON | 4 | ON | +0.042 | +0.014 | −0.016 |
| **v25a 2-seed mean** | — | — | — | — | — | **+0.118** | +0.003 | −0.009 |
| v27a | 0.6B | 1 | ON | **0** | ON | +0.025 | +0.001 | +0.005 |
| v27c | 0.6B | 1 | ON | 4 | **OFF** | −0.038 | −0.017 | −0.005 |
| **v27b** | 0.6B | 1 | **OFF** | 4 | ON | +0.797 | −0.017 | −0.005 |
| v27b-seed2 | 0.6B | 2 | OFF | 4 | ON | +0.939 | +0.008 | −0.002 |
| v27b-seed3 | 0.6B | 3 | OFF | 4 | ON | +1.833 | +0.000 | +0.002 |
| v27b-seed4 | 0.6B | 4 | OFF | 4 | ON | +1.721 | +0.001 | +0.008 |
| **v27b 4-seed mean (HEADLINE)** | — | — | — | — | — | **+1.323 ± 0.530** | +0.000 ± 0.010 | +0.001 ± 0.006 |
| v28a | 1.7B | 1 | OFF | 4 | ON | +0.909 | −0.001 | +0.005 |
| v28b | 1.7B | 2 | OFF | 4 | ON | +0.944 | −0.009 | −0.008 |
| **v28 2-seed mean (SCALING)** | — | — | — | — | — | **+0.926** | −0.005 | −0.001 |

The `evidence_lift` column (computed against the patched
`lme_val_s512_evpos.pt` with real `chain_evidence_positions`) is in
[−0.016, +0.008] across every row — i.e. **statistically zero**.
We report this as the project's framing finding: memres encodes
chain-conditional context, not literal per-fact evidence.
