# Audit A — Corpus-prior leak in D4v2 (synthd4v2_persona_callback)

**Audit date**: 2026-05-03 ~02:50 UTC-5
**Auditor**: parent agent (this session)
**Trigger**: user concern (~02:30 UTC-5) — "the backbone shouldn't be
able to learn how to uncover the MEMORY specific token; the memory-
specific encoding should be the ONLY place in the system where that
find is remotely possible."

## TL;DR

The D4v2 corpus has a **first-order prior leak**. The callback session
text reveals the answer's category twice:

```
User: Quick question, what was my favorite COLOR again?
Assistant: Your favorite COLOR is RED.
```

Any model that learns the empirical conditional distribution
P(item | category) from the closed-set training corpus achieves
**CE = 2.91 nats per callback-mask token** (per-token reduction over
log(32) = 3.47 due to multi-token items). The jointly-trained 0.6B
backbone (v15b) reaches `pa_cb_ce_mem ~= 2.97` — within 0.06 nats of
the oracle prior. Memory's marginal contribution is ~0 because no
per-chain-discriminative signal is left to capture.

**This is a corpus design bug, not a memory architecture bug.**

It explains:

1. Why the 0.6B joint-train (v15b) collapsed `evidence_lift` to ~0.
2. Why the 1.7B frozen (v15e) showed `Δnm-m_floor = +2.5` but
   `evidence_lift = -0.18` — the larger writer has enough capacity to
   memorise the prior into M_c during evidence and non-evidence
   sessions alike, so the *evidence-redacted* M_c carries the same
   prior and "helps" equally.
3. Why even the frozen 0.6B (v15a / v14k) shows only +0.07
   evidence_lift: the readout cannot reach the prior floor without
   backbone updates, so memory does some real work, but the headroom
   is tiny because the task's information-theoretic ceiling
   (log(32) = 3.47 nats) is small.

## A.1 — Empirical category-conditional prior CE (oracle predictor)

Computed by `tools/audit_category_prior.py`:

```
$ python tools/audit_category_prior.py
       fruit: 30 items, 644 chains, H=3.3803
  instrument: 32 items, 652 chains, H=3.4403
      object: 32 items, 651 chains, H=3.4493
       color: 32 items, 605 chains, H=3.4474
        tool: 32 items, 600 chains, H=3.4249
       sport: 32 items, 605 chains, H=3.4288
      animal: 32 items, 593 chains, H=3.4315
       hobby: 32 items, 650 chains, H=3.4268

per-CHAIN mean NLL = 3.4982 nats (n=500)
per-TOKEN mean NLL = 2.9107 nats
avg toks/chain     = 1.38
log(256) = 5.5452; log(32) = 3.4657
```

The achievable floor for a model that only sees the callback session
is **2.91 nats per callback-mask token** (the per-token reduction
below log(32) is structural — multi-token BPE items have predictable
continuations).

## A.2 — Base-model CE (no FT, no memory)

Computed by `tools/audit_base_prior.py` over 64 val chains:

| backbone        | CE on callback tokens |
|-----------------|----------------------:|
| Qwen3-0.6B base | 7.7950 nats           |
| Qwen3-1.7B base | 8.9034 nats           |

Both are above uniform-256 = 5.55 nats: Qwen3 was not pretrained on
"Your favorite color is __" → {32-color closed set}, so it disperses
mass across many possible continuations. **5–6 nats of headroom for
finetuning to "discover" the prior**, which is exactly what we
observe in joint-train.

## A.3 — Trained checkpoints — observed CEs

From in-trainer EVID-EVAL logs:

| run                                                | step | backbone | pa_cb_ce_mem | pa_cb_ce_nomem (inferred via floor) | gap to 2.91 |
|----------------------------------------------------|-----:|----------|-------------:|-------------------------------------:|------------:|
| **v14k_best** (0.6B frozen, no warmup, norm ON)    | 1500 | frozen   | 4.78         | ~6.22                                | +1.87 nats above prior |
| **v15a_best** (replicate of v14k)                  | 2000 | frozen   | 6.01         | ~7.07                                | +3.10 |
| **v15e_best** (1.7B frozen, norm ON)               | 2000 | frozen   | 5.46         | ~5.93                                | +2.55 |
| **v15b_best** (0.6B JOINT, lr_b=2e-5)              | 4000 | trained  | **2.98**     | **2.97**                             | **−0.07 (≈ prior)** |

`v15b_best` is the smoking gun. The jointly-trained 0.6B reaches
exactly the empirical-prior CE without any memory contribution
(`pa_cb_ce_mem ~= pa_cb_ce_nomem`).

## Mechanism — why this happens

1. The trainer's `--callback_loss_weight 3.0` upweights LM loss on
   callback-mask tokens by 4x (1 + 3.0).
2. With the unfrozen backbone, the LM gradient on those tokens flows
   directly into the embedding/decoder/LM head.
3. The callback session text contains the CATEGORY word twice (in
   the question and again immediately before the answer span), so
   the model gets a strong, attendable cue for the answer's
   category.
4. With ~5000 unique chains x ~4 epochs of training and only 8
   categories, the model has plenty of supervision to learn
   `P(item | category)` to within the empirical noise floor.
5. Since all the predictability of callback tokens is now in the
   backbone, memory has nothing to add — the writer's gradient
   signal on the LM loss vanishes and writer parameters drift
   towards convenience features (warmup-induced recent-bias,
   alpha-floor-induced uniform routing, InfoNCE-induced per-chain
   discrimination that is nonetheless useless to the LM).

## Why this is NOT just a "frozen vs joint" finding

The user's concern is correct: if memory were the *only* pathway for
chain-specific information, then unfreezing the backbone should never
remove memory's value. Joint training would simply make the writer +
backbone *jointly* better at the task. The fact that joint training
*kills* the memory channel proves that the backbone has an
independent, non-memory pathway to the answer. We have now identified
that pathway: the category cue in the callback text.

## Fix — the v16 corpus design

The corpus must satisfy:

(i) `P(item | callback_session_text)` is essentially uniform over a
    large set when no memory is provided.
(ii) The only signal that disambiguates is in the evidence sessions.
(iii) The evidence sessions are NEVER inside the LM-attended window
      of the callback session (already true: each session is its own
      forward pass).

Concrete proposals (in order of preference):

### v16-a — Random unique IDs (cleanest)

Each chain draws a random 5-digit number (or 4-character random
string) as its "secret." The callback session reads:

```
User: Earlier I told you a secret code. What was it?
Assistant: Your code is 47821.
```

- 100,000 possible IDs ⇒ uniform prior CE = ~11.5 nats per chain.
- No category cue, no closed set the backbone can learn.
- Tokenization: each 5-digit number is 1-3 BPE tokens; the per-token
  CE under uniform is also 4-11 nats. Well above the achievable
  memory-mediated CE (which can be ~0 if memory works).

Implementation: trivial — replace `render_persona` and
`render_callback` with the secret-code variants. Keep
`n_evidence_sessions=2` (two secrets per chain, callback asks about
one of them).

### v16-b — Random rare words

Use words rare in the Qwen3 pretraining distribution (e.g. chemistry
compounds, technical jargon, made-up nonsense words). The backbone
has weak priors over these so it cannot learn a useful marginal even
with category cues. Harder to design (need to source the vocabulary)
and easier to get wrong. **v16-a is preferred.**

### v16-c — Drop the category cue

Keep the closed-set items but strip the category from the callback:

```
User: Quick question, what was the THING I told you about earlier?
Assistant: Your THING was RED.
```

Reduces the prior from 32-way to 256-way (3.47 → 5.55 nats). Helps
but doesn't eliminate per-item-frequency learning. Inferior to v16-a.

### v16-d — Per-chain item shuffle

Each chain randomly permutes the closed-set items into 8 fake
categories defined ONLY in the chain's evidence sessions. The
callback's category cue then provides no information unless memory
captured the shuffle. Cute but adds template complexity; unnecessary
if v16-a works.

## Recommendation

Build **v16-a** immediately. It's a one-screen change to
`tools/build_synthetic_persona_callback.py`. Re-run the v15a recipe
(0.6B frozen, no warmup, alpha-floor + InfoNCE, norm ON) on v16-a.
The in-trainer `pa_cb_ce_nomem` should now hover at 11+ nats
(uniform-100k); if `pa_cb_ce_mem` drops well below that, memory is
doing real work.

Also re-run v15b recipe (0.6B joint train) on v16-a. Expected
behaviour now: the backbone CANNOT learn `P(item | callback)`
because it is uniform — so the joint-train CE should track the
frozen-backbone CE, and any reduction below the uniform floor must
come from memory. This is a clean test of "does memory help when it
has to."

## Open questions for v16

* What's the right item-vocabulary size? 1k, 10k, 100k? Larger ⇒
  stricter test but slower convergence.
* Should the secret be a single token (e.g. a rare 5-digit number
  that tokenises to one token) or multi-token? Multi-token gives
  more per-callback signal and fits the existing
  `session_callback_mask` machinery, but introduces
  predictability-via-BPE artifacts. Worth measuring on a held-out
  base model.
* Should we randomise the callback template too? E.g. uniformly draw
  from 5 different question phrasings to prevent the model from
  learning "is " → trigger answer prediction.

These can be ablated within v16; the core v16-a fix is enough to
restore validity.

## Status

* Corpus generator audited
  (`tools/build_synthetic_persona_callback.py`): category leak
  confirmed at `render_callback` (line 247-259).
* v15b joint-train results audited: 2.97 nats (matches prior).
* Other auditors running: see `audit_b_literature.md` (literature)
  and `audit_c_code_leak.md` (code-side leakage).
* v16-a corpus build: design lock pending.
