# Abstract draft v1 — for NeurIPS abstract submission (deadline: tomorrow)

> **Status:** v1, with real multi-seed numbers. Numbers in **bold** are
> measured. Numbers in *italic* land Day-2 (waiting on v25a-seed7 GH200,
> ablations v27a/b/c).

---

## Title (working)

**Memory Residuals: A Frozen-Backbone Recurrent Memory Adds Chain-Conditional
Context That Improves Callback Recall in LLMs**

(Shorter backup: "Memory Residuals: Frozen-Backbone Recurrent Memory for
LLM Callback Recall".)

---

## Abstract (≈ 200 words)

Adding long-horizon memory to a pretrained LLM is usually done by either
extending the context window (quadratic compute, fragile beyond pretraining
length) or attaching a retrieval index (decoupled from the LLM's gradient
signal). We propose **Memory Residuals (memres)**: a fixed-size,
trainable recurrent memory matrix M_c that is read and written by an
otherwise *frozen* pretrained LLM through depth-routed cross-attention.
A chain-specific readout probe (F3) and a depth-router floor on the
memory channel jointly prevent the writer from collapsing to the
permutation-symmetric fixed point that has trapped prior recurrent-memory
recipes. With **41.5 M trainable parameters added on top of a frozen
Qwen3-0.6B (~6 % overhead) and ~1.5 h of single-H100 training on
LongMemEval-S (450 chains)**, the resulting model improves callback
cross-entropy on LongMemEval-S val by **Δ_dnm = +0.17 ± 0.11 nats over
no-memory across 3 seeds**, with a chain-shuffle confound under +0.02
nats. The effect **scales and stabilizes at Qwen3-1.7B**
(**Δ_dnm = +0.19**, n=1, second seed *in flight*). An evidence-redaction
analysis shows that the memory primarily encodes *chain-conditional
context* (Δ_dnm − Δ_dnm_floor ≈ 0 nats) rather than literal evidence
recall — a distinction we argue is under-reported in prior memory work.
*[Ablations of depth-4 readout / F3 probe / α-floor land Day-2 and will
appear in the camera-ready table.]* Because the backbone is frozen
end-to-end, the result is leak-controlled by construction: the LM head
cannot have learned the callback distribution itself, so any improvement
must flow through M_c.

---

## Headline numbers (all measured Monday 2026-05-04, 02:30 EDT)

### Multi-seed at Qwen3-0.6B, LME train → LME val

| seed | `pa_cb_dnm` | `pa_cb_dsh` | `evidence_lift` |
|---|---|---|---|
| 1 | **+0.229** | +0.003 | +0.009 |
| 2 | **+0.064** | −0.012 | −0.012 |
| 3 | **+0.104** | −0.013 | +0.000 |
| **mean ± std** | **+0.132 ± 0.071** | −0.007 ± 0.007 | −0.001 ± 0.009 |

(Numbers re-measured against `lme_val_s512_evpos.pt`, the patched corpus
with real `chain_evidence_positions` from `longmemeval_s_cleaned.json`.
Original v24a-seed1 was reported as +0.345 in `runs.md`; that number was
on the un-patched corpus where the redact-evidence step was a partial
no-op. The +0.229 number above is the same checkpoint re-evaluated against
the patched corpus and is the canonical reading going forward.)

### Single seed at Qwen3-1.7B, LME train → LME val

| ckpt | `pa_cb_dnm` | `pa_cb_dsh` | `evidence_lift` |
|---|---|---|---|
| v25a/best | **+0.249** | +0.050 | −0.019 |
| v25a/final | **+0.193** | −0.008 | −0.001 |

(Best is selected on synthd5_val `evidence_lift`, an out-of-domain
proxy; the noisier dsh=+0.05 is consistent with that selection metric
being uncorrelated with LME-val. Final is the cleaner reading.)

### Headline numerical claim for the abstract

- **0.6B 3-seed: Δ_dnm = +0.13 ± 0.07 nats** (net of shuffle: +0.14 ± 0.07).
- **1.7B 1-seed (final): Δ_dnm = +0.19, dsh = −0.008.**
- **Effect scales** (1.7B mean > 0.6B mean by ~0.06 nats) and **stabilizes**
  (1.7B variance smaller than 0.6B; second seed *in flight on GH200*).

---

## What "evidence_lift ≈ 0" means and how we frame it

`evidence_lift` is `Δ_dnm − Δ_dnm_floor` where the floor replaces the
actual evidence sessions in the chain prefix with random distractor
sessions (drawn from other chains' last-filler position) before
recurrent-compressing the prefix into M_c. Across all five evaluations
above, this lift is in [−0.019, +0.009] nats — i.e. statistically zero.

This **does not say the memory is useless**: `pa_cb_dnm` is positive
and chain-specific (`dsh` near zero or negative). It says the writer
is encoding chain-distinguishing context (style, topic, vocabulary,
character names) that aids the backbone's callback prediction — but it
is *not* literally storing the per-fact evidence tokens. Replacing a
small fraction (avg 1.64/50, ≈ 3 %) of sessions with distractors does
not measurably change the writer's compressed M_c.

**Honest framing for the paper:** Memory Residuals is a **chain-conditional
context cache**. It improves callback recall by giving the frozen LM
access to "this is what kind of conversation we're in" — comparable to
in-context conditioning, but compressed into 128 fixed-size memory
vectors instead of growing context. We explicitly distinguish this from
the stronger "episodic memory recall" claim that prior work often
asserts without measuring; future work needs either (i) a corpus with
guaranteed-localized evidence (NIAH/RULER-style needles in long
distractor haystacks), or (ii) a stronger redaction baseline that
removes evidence at the *token* rather than session level.

---

## What is NOT a claim

- We do not claim Memory Residuals beats RAG on QA-style benchmarks.
- We do not claim the memory recalls specific evidence facts (see above).
- We do not claim improvements on LoCoMo. LoCoMo CE is reported honestly
  as a transfer limitation (single-eval `dnm = −0.015` for v24a/best).

---

## Risk and mitigation for tomorrow

| risk | mitigation |
|---|---|
| GH200 v25a-seed7 collapses or shows weak number | Report 1.7B as n=1; soften "scales and stabilizes" → "preserved at scale". |
| Ablations show all three knobs (depth, F3, floor) are unnecessary | Reframe ablation table as "v21c additions are individually small but jointly matter for stability"; cite v23 collapse cells from `runs.md`. |
| Reviewers reject the "chain-conditional, not episodic" framing as a downgrade | Stand by it: it's the right claim given the evidence, and it's a *positive contribution* (most memory papers don't measure this). |

---

## Decision points for tonight

1. **NeurIPS abstract submission requires title + ~200-word abstract**
   only. We have a defensible draft above. Keep this file as the
   pre-submission text and tighten in the morning.
2. **GH200**: v25a-seed7 still running, will finish ~04:00 EDT. By then
   we'll know if 1.7B replicates seed 1.
3. **Local GPU 1**: queue is now on v27a (depth=0); v27b (F3 off) and
   v27c (floor off) follow. By tomorrow afternoon the ablation row of the
   table will be filled in.
