# Memory Residuals

A fixed-size recurrent memory matrix $M_c \in \mathbb{R}^{K \times d}$
that gets updated end-to-end through the language-modelling loss — no
retrieval index, no separate memory controller, no hand-engineered
gating heuristic. A learned compression of past sessions that reads
into the depth-wise residual stream of a pretrained transformer.

The architectural spec is in
[`memory_residuals.pdf`](memory_residuals.pdf) /
[`memory_residuals.txt`](memory_residuals.txt). What follows is the
honest story of ~45 training cells across ten version epochs — what
broke, why it broke, and what the surviving recipe actually is.

---

## ⚠ Stop everything and read this first — what is actually broken (2026-04-30, post-v10 audit)

> All v10 cells (`chain_v10a_composed_diverse_local`,
> `chain_v10b_attnparity_pm4_diverse_local`,
> `chain_v10_4b_mega_attnparity_gh200`) were killed mid-training on
> 2026-04-30 ~20:50 UTC. No v10 checkpoint should be promoted. The
> sections after this one (TL;DR, "Walls 1-6", surviving recipe) describe
> the PRIOR mental model that drove v9 → v10 — which was wrong about
> *what* was broken. They are kept for the historical record and
> because the architectural-primitive observations are still correct,
> but the "data diversity is the load-bearing axis" headline of v10 is
> a partial truth that masked the real root cause described below. The
> v10b 0.6B proxy at step 200 (CB Δ_sh-m = +0.011, α_mem = 4e-4)
> looked like a win for ~6 hours; by step 2000 it had decayed to
> noise. v10a (simple_gate, composed) never opened gate_max above
> 0.0042 and CB Δ_nm-m went *negative* (memory hurts callback
> prediction) by step 400. Neither cell escaped the fundamental
> failure mode below.

The right framing is: **the system is converging to "ignore memory" as the
LM-loss-optimal policy, because the training distribution actively
*rewards* ignoring it.** Six interacting failures, ranked by leverage
(highest first). Each is independently sufficient to kill training
even if all the others were fixed.

### P0. The corpus builder throws away the only ground-truth evidence labels we have, and the curriculum sampler then picks "evidence" sessions uniformly at random from the haystack

**Verified empirically on `lme_train_s512.pt`** (`paper_tools/build_conversational_callback_chains.py`):

- LongMemEval-S `longmemeval_s_cleaned.json` ships per-turn `has_answer:
  bool` annotations *and* a chain-level `answer_session_ids` list
  identifying which sessions actually contain the answer the callback
  asks about.
- Mean haystack length per chain: **47.7 sessions**.
- Mean fraction of haystack sessions that contain the answer: **3.6%**
  (median ≈ 1 evidence session per ~50-session chain).
- `build_conversational_callback_chains.load_longmemeval` reads each
  turn's `content` and `role` but **never reads `has_answer`** and
  **never reads `answer_session_ids`**. The pretokenized blob's
  `chain_meta[i]` keys are `{question, answer, question_type}` —
  `answer_session_ids` is *gone by construction*.
- Confirmed empirically: of the 450 LongMemEval chains in
  `lme_train_s512.pt`, exactly **0** have any `session_callback_mask` bit
  set on a non-callback session. Only the synthesised callback session
  (the one we appended) has any cb-marked tokens, and those are the
  *answer text*, not pointers back to where the answer was *learned*
  during the haystack.

The downstream consequence in `train_chain.ChainSampler.sample_window`
(curriculum_competition branch, the *primary* training distribution
for v9 / v9c / v10b / v10):

```python
# Sample A (KEEP-PREV):  [evidence, distractor, callback]
evidence_pos   = self.rng.randint(0, cb_pos - 2)        # uniform!
distractor_pos = self.rng.randint(evidence_pos + 1, cb_pos - 1)
# Sample B (WRITE-NEW):  [noise, evidence, callback]
evidence_pos = self.rng.randint(1, cb_pos - 1)          # uniform!
noise_pos    = self.rng.randint(0, evidence_pos - 1)
```

With ~3.6% prior on any given session being the actual evidence:

- Sample A: **P(evidence-slot session actually contains the answer) ≈ 3.6%**.
  The other 96.4% of the time the writer is given an irrelevant session
  and the judge is asked to defend it as "keep" through the distractor —
  i.e. the gradient says *protect noise*.
- Sample B: same prior on the "evidence" slot. The "noise" slot is
  almost always also irrelevant (no evidence labels exist anywhere
  earlier in the chain to bias it). The judge is asked to "write" a
  random session over a different random session. The gradient says
  *replace noise with noise*.
- The callback is then scored under an M_c built from those two
  random sessions. **For 96%+ of training samples, M_c demonstrably
  cannot contain the answer** because neither of the source sessions
  did.

The trainer comment block actively defends this:

> "We do NOT special-case the 'evidence is the actual referent'
> subset because doing so would teach the judge to rely on an oracle
> signal (ground-truth evidence labels) that won't exist at inference."
> *(`train_chain.py` line ≈700-730)*

This argument is **wrong in two ways**:

1. At inference, M_c is built sequentially over **all** ~50 haystack
   sessions, so it contains the answer with probability ≈ 1. At
   training, M_c is built over **2** random sessions, so it contains
   the answer with probability ≈ 0.072 (1 - 0.964²). These are not
   the same distribution — the inference distribution is *easier*.
   We are training on a strict superset of "harder than deployment"
   samples and rewarding the model for marginalising memory away.
2. Using `answer_session_ids` at training time is *not* an oracle
   leak — at inference you would use the M_c that has actually
   accumulated all the haystack content; the training-time signal
   should match. The right curriculum is "build M_c from a window
   that *actually contains* the answer, then ask the callback".

The user's hypothesis (2026-04-30 19:51 UTC) — *"the system is
simply not able to initialize the entire memory compression branch"* —
is half right: the architecture cannot bootstrap because, on the
distribution it is being shown, the LM-loss-optimal answer **is** to
ignore memory. The branch is not failing to initialise; it is being
correctly trained to its target, and the target is "produce nothing
useful". The 96% irrelevant-evidence supervision pushes
$g \to 0$ / $\alpha_{\text{mem}} \to 0$ at every step; the 4% relevant
supervision is averaged away. The prior PG-19-on-v2/v3 "always had
some signal" observation reads the same way: PG-19 has no evidence
labels at all, so the curriculum_competition branch never fires on
PG-19 chains — they fall through to *contiguous* uniform windows,
where local stylistic continuity weakly rewards memory and the
gradient at least points the right way. That's why every "diverse"
campaign with a substantial PG-19 share (v2/v3/v9c/v10) shows mild
positive signals, while every LME-only campaign collapses.

### P1. The bootstrap chicken-and-egg: gate, readout, and writer must all become useful in lockstep, and our training signal will not let any of them go first

The four learned MemRes components form a directed cycle of dependencies:

- `MemoryGate` (`g`) and `BlockAttnResRouter.mem_bias` (the per-sublayer
  routing mass on b_{-1}) only receive non-trivial gradient signal
  when `m^t` is actually doing something useful downstream — i.e.
  only after the readout has learned to project `M_c` into something
  the LM head can consume.
- `MemoryReadout` (`W_Q^read`, `W_K^read`, `W_V^read`, RMSNorm scale)
  only receives gradient signal when `g * m^t` (or
  `α_mem * m^t`) measurably moves logits — i.e. only after the gate /
  router has opened.
- `MemoryBlock` (`M_in`, `M_judge`, the L_E extraction stack, the
  judge attention) only receives gradient signal when `M_c` is
  consumed by the readout in a way that affects loss — i.e. only
  after both of the above have opened.

At init: $g \equiv 0$, $\alpha_{\text{mem}} \approx 3 \cdot 10^{-4}$
(parity case at -4/+4) or $\approx 1/N$ (base case), and the readout's
$W_V$ is freshly random. The augmented model is *behaviourally* the
bare backbone, but the LM gradient cannot see the memory branch at
all because every gradient component flowing to a MemRes parameter is
multiplied by either $g$ or $\alpha_{\text{mem}}$, both of which are
~0.

Concretely, in `attention_parity` mode at $-4/+4$:

$$\frac{\partial L}{\partial \texttt{mem\_bias}} \propto
  \alpha_{\text{mem}}(1 - \alpha_{\text{mem}}) \cdot
  \frac{\partial L}{\partial h_{n,i}} \cdot m^t \approx
  3 \cdot 10^{-4} \cdot (\text{normal-magnitude term}).$$

The gradient on `mem_bias` is attenuated by ~3000× at every step.
With `weight_decay = 0.1` on top, the bias drifts toward 0 (which
*would* open the router) but at a rate of $\text{lr} \cdot \text{wd}
\cdot 4 = 5 \cdot 10^{-5} \cdot 0.1 \cdot 4 = 2 \cdot 10^{-5}$ per step
— so 2000 steps to move it from -4 to -3.96. Empirically (`v10b`
log): `α_mem_max` started at 0.0004 at step 200 and was 0.0001 at
step 2200 — *decreasing*, because the (rare) useful gradient pushed it
the wrong way more often than the right way under the misaligned
P0 distribution.

The `simple_gate` route is supposed to side-step this by putting the
gate directly on the gradient path (not behind a softmax). But the
v8 readout RMSNorm (added to defend against v7's
$\|\text{W}_V^{\text{read}}\| \to \infty$ explosion) overshoots in
the other direction: the readout output magnitude is now pinned at
$\|m^t\|/\|\text{embed}\| \approx 73$ (verified across every v8/v9/v10
log). So the per-sublayer gate cannot operate in a normal `[0, 1]`
range — useful gates are in $[0, \approx 0.014]$. v10a gate_max moved
from 0 to **+0.0042 by step 540 then back down to +0.0034 by step
880** — exactly the bouncing-off-noise pattern P1 + P0 predict.

There is no warm-up phase that lets the writer + readout learn
to produce useful `m^t` *before* the gate sees gradient. Pair training
(`train_phase1.py`) was supposed to do this, but the only positive
chain transfer (v2 phaseA on PG-19+TV) needed both the warm-start
*and* a corpus where the contiguous-window LM signal alone provides
weakly correct memory gradient. That doesn't exist for callback-style
LongMemEval, which is precisely the corpus we need for the recipe
paper.

### P2. The readout RMSNorm is mis-calibrated: it solved v7's explosion but created a 73× injection that cannot be gated cleanly

In `MemoryReadout.forward`:

```python
return self.out_norm(attn @ V)   # Qwen3RMSNorm, weight init 1.0
```

For $d=1024$, `out_norm` produces token vectors with $\|m^t\| \approx
\sqrt{d} \cdot \|\text{weight}\| \approx 32$. Embedding norms are
$\|\text{embed}\| \approx 0.5$. The reported ratio $\|m^t\| /
\|\text{embed}\| \approx 73$ holds across every v8/v9/v10 cell *with no
training-induced drift* (max-min over 2000 steps in v10b is
73.45–73.55). RMSNorm is doing exactly what RMSNorm does — it pins
the magnitude.

That magnitude is wrong by ~30-70× for the residual stream the gate /
router is supposed to schedule. Concretely:

- A scalar gate `g` sees `h <- h + g * m^t`. With $\|h\| \sim 1$
  and $\|m^t\| \sim 30$, the "useful" range of `g` is $[0, \approx
  0.03]$. The gradient $\partial L/\partial g \propto m^t \cdot
  \partial L / \partial h$ has magnitude ~30 per token, but the SCALAR
  reduction sums over all tokens in the batch. Mixed signs across
  tokens cancel, leaving very small absolute gradients on `g` — and
  those gradients live in a parameter space where the *useful answer
  is in $[0, 0.014]$*. The optimizer's natural step size is too coarse
  to find that window stably.
- For the depth-pool router, the scale is mediated by the softmax, so
  the magnitude is not the direct problem; but the same "memory
  source dwarfs every other source" property means the router learns
  early that the memory source is "loud" and attenuates it via
  `mem_bias` — exactly opposite to what we want.

**RMSNorm-on-readout is a defence against v7-style explosion that
overshot.** A more principled fix is initial parity at the readout
output (re-introduce $W_V^{\text{read}} = 0$ at init, then *later*
add a small no-op-at-init regulariser if explosion comes back), so
that $m^t \approx 0$ at step 0 and $\|m^t\|$ ramps as the
writer / readout learn — keeping injection magnitude commensurate
with the gradient signal that's learning it.

### P3. The phase-aligned eval inherits P0: it picks evidence uniformly at random and reports the noise floor

`Trainer._phase_aligned_eval`:

```python
e = rng.randint(0, cb_pos - 1)        # uniform — same bug as training
evidence = ev.chain_session_at(ci, e)
M_c      = compress_session(extract_source(evidence), None)
# ...score callback under M_c, M_c=None, M_c=shuffled
```

So the eval is asking: "if I build M_c from one random session of the
haystack (3.6% chance it's the actual evidence), how well does it
help predict the callback?" The answer in expectation is "barely",
because 96% of the time M_c does not contain the answer. The 4%
that do produce a real signal are buried in the noise. Per
`COMPREHENSIVE.md`, CB Δ_sh-m std at n=48 is ≈ 0.014 nats and the
peak-then-decay we kept treating as "the model overfitting and
collapsing" is mostly max-of-K-noise with a slow drift, not a
mechanistic signal.

The standard sequential eval (`Trainer.evaluate`, `eval_window=8` at
the tail of each ~50-session chain) is the *correct* eval for what
this architecture is supposed to do at inference, and it has been
sitting at $\Delta_{\text{sh-m}} \approx 0.000$ across every cell of
v9 / v10. We were dismissing this as "the train-eval distribution
gap" and chasing the phase-aligned metric instead. The phase-aligned
metric is measuring the same broken distribution as the training
loss, so of course it correlates with progress; but progress on what
the trainer can see is not the same as progress on the metric we want
to claim.

### P4. The composite "competition + diverse + cbw=3" recipe trains < 8% of gradient on memory-relevant tokens

For a typical v10b step:

- v6_lme_msc corpus weights (from log):
  pg19 ≈ 380k weighted units, msc ≈ 72k, **lme ≈ 36k**, tv ≈ 6k.
  → P(LME chain sampled) ≈ **7.3 %**.
- For the 92.7% non-LME share, `cb_pos = -1` so the
  curriculum_competition branch can't fire and the sampler falls
  through to a contiguous uniform window. The LM loss is plain
  next-token prediction with no callback mask, no memory supervision.
  Gradient on memory params from these samples points at "do not
  inject noise into the residual stream" — i.e., **deflate the gate**.
- For the 7.3% LME share, even within a 3-session
  competition window where 1 session is the synthesised callback (~30
  content tokens, ~20 callback-marked):
  - Total weighted token count ≈ 3 × 200 + 20 × 3 (cbw bonus) = 660
  - Callback-marked share ≈ 80 / 660 ≈ 12 %
  - → callback-token gradient share of total ≈ 7.3% × 12% ≈ **0.9 %**
  - Of which ~3.6% (P0) is on a *correctly* aligned evidence-callback
    pair → useful-signal share ≈ **0.03 %**.

The remaining 99.97% of gradient is "ignore memory because either it
isn't supervised, or the supervision is misaligned, or the supervision
is on tokens that don't actually need memory". The model converges
to its training distribution. The training distribution is not the
benchmark.

### P5. We never trained M_c at the recurrent depths the eval uses

`Trainer.evaluate` builds M_c sequentially across **all preceding
sessions** of each held-out chain — typically 40+ judge updates.
Training never sees this depth:

- `--burn_in_max 0` in v9 / v10b: zero burn-in, M_c is *literally*
  reset to zeros every step, recurrent depth ≤ window_k - 1 = 2.
- `--burn_in_max 12` in earlier cells: M_c sees up to 12 detached
  judge updates, but gradient does not flow through them.
- `--carry_state` in v10a: M_c carries between TBPTT windows but
  detached, so the optimizer cannot update the writer / judge to make
  M_c well-conditioned at depth.

The judge's `judge_norm` (Qwen3RMSNorm in `MemoryBlock.judge`) was
added in v8 specifically as a band-aid for unbounded $\|M_c\|_F$
growth across long chains. With it, $\|M_c\|$ does stay bounded —
but the *direction* of $M_c$ in slot space at depth 40 is in a
region the readout has never seen during training. That's the
"standard Δ_sh-m ≈ 0 even at v9 peak" we keep observing.

### Cross-cutting: the symptoms we kept treating as architectural failures were data + signal failures

| Symptom we observed | Diagnosis we wrote | What it actually was |
|---|---|---|
| α_mem ≡ 0 in attention_parity (v3-v9d) | router collapse, paper-spec routing is dead | gradient on `mem_bias` is α_mem-multiplied; with α_mem ≈ 3e-4 and 96% misaligned data, no escape velocity |
| `‖m^t‖ → 0` under attention_parity + WD (v7) | readout decay, need RMSNorm | same — α_mem ≈ 3e-4 attenuates W_V grad, WD then dominates |
| `‖m^t‖ → 165` under simple_gate (v7) | scale explosion, need RMSNorm | gate got a small useful gradient on a few PG-19 samples and overshot, then the LM head couldn't recover |
| Standard Δ_sh-m ≈ 0 at all v9 peaks | "eval-distribution gap, need carry_state + window_k=8" | the *training* signal contains ~0.03% useful-memory gradient; the model converges to ignore memory; the standard eval reports this faithfully |
| Phase-aligned CB Δ_sh-m peaks then decays | "noise + max-selection bias" | partly true (eval noise dominates) AND the training distribution is not actually pushing the model toward content-discriminative memory |
| v9c (diverse) outperforms v9 (LME-only) | "data diversity is load-bearing" | partly true: PG-19's contiguous-window LM signal weakly rewards style-prior memory, and that's *better than* the actively-misaligned LME competition supervision. v9c is "less wrong", not "right" |
| v10b α_mem opens to 4e-4 in 200 steps | "diverse data validates attention_parity" | bare-init artifact: pretrained-backbone activations + zero pseudo-queries gives a tiny non-zero score that decays to noise within 1k steps |

### What this implies for the next campaign (v11 design constraints)

These are *constraints*, not a recipe. The recipe needs to be
designed against them.

1. **Use `answer_session_ids` and per-turn `has_answer`**.
   Re-tokenise the corpus to (a) preserve the evidence-session
   indices in chain_meta, and (b) extend `session_callback_mask` to
   *also* mark the answer-bearing tokens *inside* the haystack
   evidence sessions (not just the appended callback span). The
   curriculum sampler must build windows that *actually* contain
   the answer in M_c by the time the callback is scored — every
   single training sample, no exceptions.

2. **Build a real warm-up phase that does not rely on the LM gradient
   to bootstrap the memory branch**. Candidates:
   (a) A dedicated retrieval-objective phase: ask the model to
       reconstruct answer tokens from M_c alone (no prior context),
       gradient flows into writer + readout directly with no gate
       in the way.
   (b) A masked-callback regime: the callback's answer span is
       *masked from in-context attention* but visible to the
       readout via M_c, so the model is forced to use the memory
       channel or fail.
   (c) A targeted gate-warmup schedule: start with `g` initialised
       to a small *positive* value (e.g. 0.01 in simple_gate) so
       the readout has nonzero forward influence and the LM
       gradient can actually flow back; anneal toward "useful"
       rather than starting from zero.

3. **Re-calibrate readout magnitude to the gate's operating range**.
   Either drop the readout RMSNorm and zero-init `W_V^read` (so
   $m^t \approx 0$ at step 0 and grows commensurately), or keep
   the RMSNorm but scale its weight init way down (e.g. 0.01) so
   $\|m^t\| \sim 0.3$ rather than 30. Without this, the gate
   cannot cleanly schedule injection.

4. **Make the eval honest**. The phase-aligned eval should *also*
   pick evidence from `answer_session_ids` and report the
   "evidence available" upper-bound separately from the
   "evidence absent" floor. The standard sequential eval should
   become the primary metric again — the phase-aligned variant has
   too much noise and bakes in the same data bug.

5. **Guarantee gradient through deep M_c**. Either a longer
   undetached TBPTT (window_k ≥ 16 with gradient checkpointing,
   8B-class memory budget) or an explicit "synthetic deep M_c"
   loss where we periodically backprop a callback-NLL through a
   30-step recurrent build.

6. **Stop combining recipes that haven't each individually been
   shown to move the standard eval**. v10 layered competition +
   evidence + callback_window + carry_state + diverse corpus + 4B
   backbone simultaneously. None of those axes individually closed
   the standard-eval gap; combining them did not either. Single-axis
   ablations on a 0.6B proxy with the P0 fix in place are the next
   step.

A `train_phase1.py` Phase-0 warm-up *with* the P0 evidence labels
preserved is the minimum-viable next experiment. Until P0 is fixed,
no architectural change can be properly tested.

---

## What we learned (v1 → v10)

### TL;DR

1. **Data diversity is the load-bearing axis.** Every positive result
   we have has always come from a diverse corpus (PG-19 + TV, or
   LME + MSC + PG-19 + TV). Every prolonged failure used a narrow,
   single-source distribution (LongMemEval alone). We spent three
   months debugging the architecture when the limit was the data.
2. **The phase-aligned callback-token diagnostic pair
   (Δ_nm-m, Δ_sh-m) is the single highest-value eval we added.**
   Adopting it in v8 exposed architectural failure modes that the
   standard eval read as `Δ_sh-m ≈ 0 "noise"`. The signal to watch
   is **CB Δ_nm-m**, which on v9c grows *monotonically* from −0.03
   to +0.16 nats across training — memory learning to help callback
   tokens more and more. PA CB Δ_sh-m (shuffle-discrimination) is
   weaker and largely drowned in eval noise at our default n=48
   chains; it needs n≥256 to be resolvable.
3. **Routing mode is not the binding constraint — distribution is.**
   On LME-only corpora, `attention_parity +4/-4` collapsed (v9d,
   α_mem ≡ 0). On the *same* architecture + *same* curriculum but
   a diverse corpus, `attention_parity +4/-4` opens α_mem to 4e-4
   and produces PA CB Δ_sh-m = +0.011 inside 200 steps (v10b).
4. **Three classes of "fix" turned out to be treating symptoms:**
   bigger TBPTT windows (v6 w12), heavier regularisers (v5 / v8c),
   and architectural routers (simple_gate vs attention_parity). None
   of them moved the number until the data changed.
5. **The eval-distribution gap is still open.** Standard `Δ_sh-m`
   (eval with 40+ sequential judge updates on held-out chains) is
   at ≈0 even on our best checkpoints. v10 is the first campaign
   with a realistic shot at closing it (~68k chain, ~365k session
   mega corpus + 4B backbone + L_E=10 extraction stack).

### The walls we hit, in order

#### Wall 1 — Training-distribution mismatch (run3 → chain2, early 2026-01)

The pair trainer (`train_phase1.py`, run `run3_qwen3-0.6b-large`)
produces a genuinely positive-result checkpoint on the pair task —
$\Delta_{sh\text{-}m} = +0.029$, callback help ratio 1.77×, beats a
compute-matched RAG baseline. But the chain trainer warm-started from
that checkpoint (`chain2`) explodes on the long-horizon eval:
$\text{CE}_{\text{mem}} = 8.7$ vs $\text{CE}_{\text{nomem}} = 2.5$.
The pair distribution (concatenate 4 sessions, compress once) teaches
the *readout* to be useful; the chain distribution (carry $M_c$
through many judge updates) teaches the *recurrent competition* —
and they are genuinely different problems. **Takeaway: publishable
numbers on the training distribution are not transferable if the
deployment distribution has a different recurrent depth. The pair
paper ships as Experiment 1; the chain recipe is a separate
empirical claim.**

#### Wall 2 — Shortcut learning / style-only memory (chain2 / chain_fresh1, 2026-01)

Once the chain trainer stabilises on PG-19 + TV, $\Delta_{nm\text{-}m}$
turns modestly positive (memory helps aggregate NLL) but
$\Delta_{sh\text{-}m}$ goes *negative*: $-0.014$ on PG-19 val,
$-0.036$ on `chain_fresh1`. Injecting *another* PG-19 book's memory
performs better than injecting nothing at all. The memory channel
has learned a style prior ("this is prose") not episodic content.
This is PITFALLS.md §3 in the flesh. **Takeaway: $\Delta_{sh\text{-}m}$
is load-bearing precisely because aggregate improvements are easy to
fake with genre / style cues. Any recipe that ships must have
$\Delta_{sh\text{-}m} > 0$ with bootstrap CI.**

#### Wall 3 — PG-19 + TV: the "v2 phaseA" positive result (2026-02)

`chain_v2_phaseA_softparity_b4` produces the first real positive chain
result: **$\Delta_{sh\text{-}m} = +0.0529\;[+0.0246,+0.0915]$** on
PG-19 val (bootstrap 95% CI excludes zero), $+0.0279$ on PG-19 test.
This ships as the architectural-primitive result backing Experiment 1.
The recipe: attention_parity with *softened* $\pm 4$ router biases
(vs the original $\pm 32$ bit-exact-parity init), two-stage QKV
competition with soft-init parity, negative-chain contrastive loss at
ramp $\lambda:0.05\to0.5$, PG-19 + TV corpus, `burn_in_max=24` with
resample. **Takeaway: on diverse long-narrative data with the PITFALLS
anti-shortcut regularisers, the architecture works. This result held
up under held-out routing traces and counterfactual horizon sweeps.**

#### Wall 4 — Router collapse on callback-only corpora (v3 → v8, 2026-02 → 2026-04-29)

When we moved to LongMemEval-S (the obvious "corpus with annotated
callbacks" for the recipe paper), the picture collapsed. v3 → v6
cells on LME or LME+MSC all showed the same pattern: $\text{gate}_{\max}
\equiv 0.0000$ across 1.4–4 k steps, $\Delta_{sh\text{-}m} \approx 0$
at every eval. Six consecutive campaigns failed at the same number:

| campaign | single-axis change | result | diagnosed cause |
|---|---|---|---|
| v3 | attention_parity + hard init | $\Delta_{sh\text{-}m} \approx 0$ | router saturated, LME-only |
| v4 | `hidden_14` extract + MSC corpus | $\Delta_{sh\text{-}m} \approx 0$ | router saturated, corpus mix dilutes callback |
| v5 | soft $\pm 4$ init + neg_chain ramp | $\Delta_{sh\text{-}m} \approx 0$ | gate_max stuck at 0, 5 + cells KILLED |
| v6 | LME + gated update + callback loss + callback window bias | gate_max $\equiv 0$ | four-axis writer pivot did not open the router |
| v7 (simple_gate) | routing ablation: scalar gate per sublayer | **gate_max opens** (0.0004 → 0.0074 in 400 steps), but **catastrophic overfit** on P0 curriculum ($\Delta_{nm\text{-}m} = -0.044$) | readout is *exploding* (`‖m^t‖/‖embed‖ = 165`); no scale control |
| v7 (attention_parity) | curriculum + softer bias | $\alpha_{\text{mem}} \equiv 0$, **`‖m^t‖` → 0** under weight decay | readout is *collapsing*; router + optimizer symbiosis |
| v8 (RMSNorm readout) | scale fix on top of simple_gate | gate opens, phase-aligned CB $\Delta_{sh\text{-}m} > 0$ at step 200, but **overfits within 800 steps** on P0-only curriculum | train / eval distribution mismatch (P0 windows ≠ sequential M_c) |
| v8c (+ diverse corpus) | v8 + LME + MSC + PG-19 + TV + over-regularisation | flat everywhere, gate suppressed | regulariser stack (cbw=3, drop=.10, lr=1e-4) prevented *both* overfitting and learning |

The v7 diagnostic-lens change revealed the real dynamics. Both
routing modes had failed, for *opposite* reasons — attention_parity
under weight decay decayed `W_V^{read}` to zero (bit-exact "no-memory"
despite gate firing, read as $\Delta_{sh\text{-}m}=0.0$ *exactly*);
simple_gate got direct gradient on its scalar gate and inflated
`‖W_V^{read}‖` unboundedly until injection overwhelmed the residual
stream (`‖m^t‖/‖embed‖ = 165`). The standard eval read both states
as "noise" and we read "noise" as "keep training." **Takeaway: the
paper's spec assumes depth-pool sources are commensurate in scale;
the code never enforced that. RMSNorm on the readout output
(`MemoryReadout.out_norm`, one line in `modeling_memres.py`) is a
prerequisite for any further training. It's also not sufficient.**

#### Wall 5 — The phase-aligned / standard-eval gap (v8 → v9)

Once v8 added the phase-aligned callback-token $\Delta_{sh\text{-}m}$
evaluation (matches the training distribution: 1 evidence session,
1 judge step, score the callback span), v9 produced the first clean
positive signal: PA CB $\Delta_{sh\text{-}m}$ peak **+0.0266** on v9
baseline, **+0.0310** on v9c diverse. But the standard eval
(40-session sequential $M_c$ on held-out LongMemEval) stayed at
+0.0003. Memory is content-discriminative on the distribution it was
trained on; on the deployment-like distribution the $M_c$ statistics
have drifted to a regime the readout has never seen.

v9 ablation matrix (4 cells, GH200 over 16 h):

| cell | change vs v9 | routing | peak PA CB Δ_sh-m | verdict |
|---|---|---|---:|---|
| v9  | (reference) | simple_gate | +0.0266 | alive |
| v9a | `callback_loss_weight` 3.0 → **1.0** | simple_gate | +0.000 (readout decayed to ‖m^t‖ ≈ 0) | cbw=3.0 is load-bearing; without it `W_V^{read}` decays under weight decay |
| v9b | competition 1.0 → 0.5 + evidence 0.5 + k=8 | simple_gate | +0.000 (readout decayed) | **pure** competition is required; mixing dilutes the judge gradient below the WD floor |
| v9c | LME-only → **diverse v6_lme_msc** | simple_gate | **+0.0310** | diversity is strictly non-harmful on the PA metric and slightly better-calibrated |
| v9d | simple_gate → attention_parity | attention_parity +4/-4 | +0.000 (α_mem ≡ 0) | attention_parity stays architecturally collapsed on LME-only |

**Takeaway: competition curriculum + RMSNorm readout + cbw=3.0 +
simple_gate = first reproducible PA CB signal. Peak value is similar
or slightly better under data diversity. Neither recipe closes the
standard-eval gap.**

#### Wall 6 — Data diversity is not a nice-to-have (v10, 2026-04-30)

The v9 ablation signal and a reread of the historical record
(PG-19-only cells in v2/v3 always had *some* signal regardless of
setup; v9c was the only surviving v9 ablation) led to the v10
hypothesis: **LongMemEval alone is pathologically narrow and the
routing collapses we blamed on architecture are actually
distribution-collapse artefacts.** v10 tests this directly by
constructing a ~68 k-chain / ~365 k-session mega corpus
(LME + MSC + PG-19 + TV + UltraChat + PIPPA + SODA +
OpenAssistant + no_robots + NarrativeQA + WritingPrompts) and
retrying the exact `attention_parity +4/-4` routing that v9d
declared dead.

v10b step-200 eval (0.6 B cheap proxy, same router config as v9d,
only knob different is the corpus):

```
PA-EVAL @ step 200: WS Δnm-m=+0.0040  Δsh-m=+0.0029
                    CB Δnm-m=+0.0113  Δsh-m=+0.0110
ROUTE @ step 200  : mode=attention_parity
                    α_mem_max=0.0004  α_mem_mean=0.0002
READOUT @ step 200: ‖m^t‖/‖embed‖ = 73.53
```

Non-zero α_mem, positive CB $\Delta_{sh\text{-}m}$, readout scale
bounded — all three in the *first eval cycle*. Compare v9d, same
architecture, 4000 steps on LME-only: α_mem exactly zero throughout.
**The routing-mode "failure" that motivated the v7 simple_gate pivot
was a data-distribution failure wearing an architecture mask.**

### What the surviving recipe actually is (as of v10-day-0)

**Architecture.**

- Two-stage QKV competition writer, `--memres_update_mode gated`
  (sigmoid write gate, init bias −1.0 so $g \approx 0.27$).
- Extraction source: `hidden_14` (0.6B, 28-layer backbone) or
  `hidden_18` (4B, 36-layer backbone) — mid-stack contextualised
  token representation, detached from backbone gradient.
- Readout RMSNorm (`MemoryReadout.out_norm`, v8 fix) — prerequisite
  for stability under either routing mode.
- Routing: either `simple_gate` (one scalar gate per sublayer,
  zero-init) or `attention_parity` (block AttnRes depth pool,
  softened $\pm 4$ biases — *works on diverse data, fails on
  LME-only*).

**Training.**

- Diverse corpus, full stop. Minimum viable is v6_lme_msc
  (6 378 chains across LME, MSC, PG-19, TV, RealTalk); the v10
  mega corpus scales this ~10× with UltraChat, PIPPA, SODA, OASST1,
  NoRobots, NarrativeQA, WritingPrompts.
- `curriculum_competition_bias = 1.0` (v9) or 0.5 with 0.3
  evidence-callback (v10 composed). *Pure competition is required
  at 0.6 B; at 4 B we test whether composed curriculum closes the
  eval-distribution gap.*
- `callback_loss_weight = 3.0`. Load-bearing (v9a proved <3.0 lets
  the readout decay under weight decay).
- `window_k = 3` (minimum sessions to enable competition) or 4
  (v10 4B). Larger `window_k` with `carry_state=True` exposes the
  readout to deeper $M_c$ distributions and is the primary lever
  against the standard-eval gap.
- Optimizer: AdamW, lr memres = 3–5e-5, lr backbone = 2–5e-6,
  weight_decay = 0.1, cosine decay with warmup = 200–500.

**Evaluation** (all five at every save cycle):

1. **Phase-aligned callback-token $\Delta_{nm\text{-}m}$** (`pa_cb_dnm`).
   *The primary progress metric.* Measures how much memory helps on
   callback tokens vs bare backbone, on a single-evidence + single-
   judge window that matches training. Grows monotonically when the
   recipe is working (−0.03 → +0.16 on v9c over 4 000 steps).
2. **Phase-aligned callback-token $\Delta_{sh\text{-}m}$** (`pa_cb_dsh`).
   Tests whether the channel is *episodic* vs generic. Must eventually
   be > 0 with a tight CI, but at our default `n=48` chains the per-
   eval std is ~0.014 nats — so individual readings are noisy and
   selecting on this metric alone biases toward noise peaks. Use
   n≥256 or bootstrap CI for gating decisions.
3. **Standard $\Delta_{sh\text{-}m}$ on a 40-sequential-session
   held-out chain.** The deployment-distribution metric; the
   eval-distribution gap is quantified here. Currently ≈ 0 at all
   v9 peaks — the open "Wall 5" problem.
4. **Per-sublayer routing recruitment** (`α_mem_max`, `α_mem_mean`,
   `frac_open`, top sublayers). Mechanistic evidence that *some*
   sublayer × position actually uses memory on held-out data.
   `paper_tools/routing_trace.py`.
5. **Readout magnitude** (`‖m^t‖/‖embed‖`). Sanity pulse — values
   near 0 mean `W_V^{read}` has collapsed, values ≫ 1 mean it's
   exploded. Healthy range is ~50–100 with RMSNorm in place.

Plus causal evidence per promoted checkpoint: counterfactual eval
(perturb session $t-k$, measure CE on $t$ callback) for a horizon
curve (`paper_tools/counterfactual_eval.py`), and bootstrap 95% CI
on the primary $\Delta_{sh\text{-}m}$ against a 1000+-chain sample
(`paper_tools/bootstrap_ci.py`).

### Open problems

1. **Close the eval-distribution gap.** Standard $\Delta_{sh\text{-}m}$
   is still ≈0 on our best checkpoints. v10's 4B mega run is the
   current attempt (window_k=4, carry_state, composed curriculum,
   67 k chains); success threshold is `standard Δ_sh-m > +0.005` at
   step ~20 000.
2. **PA CB Δ_sh-m eval variance is larger than the effect we're
   trying to measure.** The "peak-then-decay" pattern we initially
   read as a mechanistic failure (v9: +0.0266 → −0.0146 in 400
   steps; v9c: +0.0310 → −0.0315 in 1200 steps) is *mostly* eval
   noise + max-selection bias, not a coherent decay. On the v9c
   trajectory the load-bearing diagnostics tell a completely
   different story: `gate_max` saturates at 0.0044 by step 1400 and
   stays *exactly* there for the next 2 200 steps; `frac_open`
   parks at ~0.64; readout magnitude never drifts outside
   73.48 ± 0.02; **CB Δ_nm-m grows *monotonically* from −0.03 to
   +0.16** — memory is doing *more* useful work as training
   progresses, not less. What actually oscillates is only
   Δ_sh-m, and eval-noise math (CB Δ_sh-m std ≈ 0.014 nats at
   n=48 chains × ~20 callback tokens) says the max-of-18-evals
   under zero-mean noise is about +0.032 — which is almost
   exactly the "peak" we were treating as signal. The real
   problem here is (i) we need ~500 chains on the PA eval to
   resolve "Δ_sh-m > 0 in expectation" at p < 0.05, and (ii) the
   `save_best` on phase-aligned PA CB Δ_sh-m probably latches
   onto noise. Workarounds: bump `--phase_aligned_eval_n_chains`
   to 256+, add bootstrap CI to the selection metric, consider
   switching `--save_best_metric` to a composite that also weighs
   CB Δ_nm-m (which is the metric with real monotonic signal).
3. **8B scaling.** `qwen3-8b-xlarge` (L_E=10, ~8.8 B total) exceeds
   a single 96 GB GH200 under full AdamW (~106 GB peak). The v10 run
   is on 4B (fits at ~52 GB peak with gradient checkpointing). 8B
   needs either bitsandbytes AdamW8bit (flag `--use_adam8bit` is in
   the trainer, ready when the wheel is installed) or multi-GPU ZeRO.
4. **Evidence-annotation-free curriculum.** The competition
   curriculum currently picks a random "evidence" session from
   `[0, cb_pos)` — label noise is intentional (matches deployment)
   but we don't know whether cleaner evidence annotations would
   sharpen the judge's gradient further. Needs a synthesis run on
   NarrativeQA-style corpora with ground-truth provenance.
5. **Backbone-free evaluation.** Every eval uses the trained
   backbone. A frozen-backbone ablation (`--freeze_backbone`, added
   in v10) would separate "memory learned episodic recall" from
   "backbone co-adapted to memory injection." Not yet run.

---

## File map

```
memory_residuals.pdf / .txt  position paper (the architectural spec)
atn_residuals.pdf            Block Attention Residuals reference
COMPREHENSIVE.md             full historical ledger, every prior run
                             (long; reference, not reading order)

modeling_memres.py           architecture (config, model, init)
presets.py                   named (backbone, K, L_E, N) tuples
train_chain.py               recurrent chain TBPTT trainer (the active one)
train_phase1.py              pair-based warm-up trainer (Paper 1 recipe)

scripts/                     launchers per cell (v10 cells are active)
paper_tools/                 eval / probes / corpus builders
paper_tools/cloud_watchdog/  remote-survivable job queue + ntfy daemon

paper_artifacts/chains/      pretokenised corpora (lme_*, v6_lme_msc_*,
                             v10 mega_train_s512.pt, realtalk_*)
paper_artifacts/eval/        eval JSONs + bootstrap CIs that paper text cites

experiments/exp1_drop_in_primitive/    Paper 1 (pair recipe, run3) manuscript
experiments/exp2_long_horizon_recipe/  Paper 2 (chain recipe) draft, runs.md

output/                      training checkpoints (gitignored)
logs/                        training & sync logs (gitignored)

archive/                     superseded scripts / datasets / agent notes
```

## Compute

- **Local.** 2 × H100 NVL (94 GB) at the lab box. Currently running
  v10a (simple_gate composed, GPU 0) + v10b (attention_parity +4/-4,
  GPU 1). ~16 h/day usable (residential power-down overnight).
- **Cloud.** 1 × NVIDIA GH200 480 GB at `192.222.50.225`. Currently
  running `chain_v10_4b_mega_attnparity_gh200` (Qwen3-4B + L_E=10
  + mega corpus, 30 000 steps, ~3-day budget). Jobs run in detached
  `tmux` under the cloud watchdog, so they survive SSH drops and
  lab-box power-offs.

## Stop everything

```bash
pkill -f train_chain.py
ssh ubuntu@192.222.50.225 'tmux kill-session -t cwd-daemon; pkill -f heartbeat.sh'
```

## Reading order

1. **This README** — what we learned, what's active, where things
   live.
2. [`memory_residuals.pdf`](memory_residuals.pdf) — the position
   paper. The formal architectural spec (Eqs. 1–10, two-stage QKV
   competition, off-sequence depth-wise injection).
3. [`atn_residuals.pdf`](atn_residuals.pdf) — the Block Attention
   Residuals reference. Read when the routing trace looks weird or
   you're tuning router init biases.
4. [`experiments/exp2_long_horizon_recipe/runs.md`](experiments/exp2_long_horizon_recipe/runs.md)
   — the active run ledger (which cell is doing what right now,
   decision triggers, kill criteria).
5. [`COMPREHENSIVE.md`](COMPREHENSIVE.md) — every prior run, every
   architectural decision, every failure mode in detail (long;
   reference material, not a reading order).
