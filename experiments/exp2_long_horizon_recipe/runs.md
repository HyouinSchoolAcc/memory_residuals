# Runs ledger — experiment 2 (long-horizon recipe)

Single source of truth for which checkpoint produced which number in
the recipe paper. Keep it updated as runs finish, fail, or get
re-launched. Newest at the top.

Each entry: status, run name, machine, exact launch command (or
script reference), key knobs, last logged step / `Δ_sh-m`, log path,
notes.

For the historical v2/v3/v4/v5/v6 ledger (all KILLED or superseded)
and the v6 → v7 transition narrative, see
[`COMPREHENSIVE.md`](../../COMPREHENSIVE.md) Part IV.

---

## v10 campaign — data diversity is the load-bearing axis (2026-04-30 ~19:00 UTC)

User directive 2026-04-30 ~19:00 UTC after reviewing the v9 results:
> "something tells me it has a lot to do with the LME dataset being
> bad; notice how out of all the v9's, only the v9c diverse dataset
> had a really good survivability. We saw this earlier with the PG-19
> dataset where no matter what setup we had, it always had some
> signals of using the memory. This tells me that constructing a
> SUPER diverse dataset with MANY MANY memory-using data is useful."

### Hypothesis under test

The v3–v9 failure modes (router collapse, readout drift, peak-then-
decay) were not primarily architectural — they were manifestations of
training on a narrow, callback-only distribution (LongMemEval-S alone,
or LME + MSC = 93% callback-free fillers that dilute the memory
gradient). v9c matched v9 baseline peak performance on a diverse
corpus despite having the same competition-curriculum recipe; earlier
PG-19-only cells also had consistent memory signal regardless of
setup. The v10 campaign reframes: *diversity of memory-requiring
training distributions is the axis that makes the recipe robust*.

### Campaign design (three cells across all three devices)

| cell | machine | backbone | L_E | routing | corpus | test |
|---|---|---|---:|---|---|---|
| **v10a composed_diverse** | local H100 GPU 0 | Qwen3-0.6B-large | 4 | simple_gate | v6_lme_msc (6378 chains) | control: composed curriculum (comp=0.6, evid=0.3, k=8, carry_state) + simple_gate + diverse corpus closes standard-eval gap |
| **v10b attnparity_pm4_diverse** | local H100 GPU 1 | Qwen3-0.6B-large | 4 | attention_parity (+4/-4) | v6_lme_msc | 0.6B proxy for the 8B routing choice: does attention_parity with softer biases survive on diverse data where simple_gate already did? |
| **v10 4b_mega_attnparity** | GH200 | **Qwen3-4B-xlarge (L_E=10)** | **10** | **attention_parity (+4/-4)** | **mega (v6_lme_msc + ultrachat + pippa + soda + oasst1 + no_robots + narrativeqa + writingprompts, ~150-300k chains)** | **THE HEADLINE**: deep extraction + block AttnRes parity + 100x data diversity, 3-day budget, 30k steps |

**Why 4B and not 8B.** qwen3-8b-xlarge (8.8B total w/ L_E=10 memres)
peaks ~106 GB HBM under full AdamW, exceeding the 96 GB GH200. User
directive was "if it can't fit, make the model smaller" (no frozen-
backbone hack, no bitsandbytes dependency). qwen3-4b-xlarge (~4.3B
total, ~52 GB peak) fits with ample headroom for activations and
gradient checkpointing; preserves full-training semantics (both
backbone and memres update) that produced every v9 signal.

**Why attention_parity on the 4B run.** User directive: "I
definitively want the 8B training run to have blocked attention
parity at +4 -4". attention_parity is the paper's preferred routing
(block AttnRes depth pool); v9d showed it stays architecturally
collapsed (α_mem=0) on LME-only even with the competition curriculum
and RMSNorm-on-readout fix, but the v9d signal is confounded with
the LME-only data axis. v10b (0.6B, attention_parity +4/-4, diverse
corpus) is the cheaper 0.6B proxy that tells us in ~6 h whether
attention_parity can work at all on diverse data; v10 4b_mega pursues
the full headline recipe independently on GH200.

**Mega corpus** (target ~200-300k chains). Build pipeline:
[`scripts/build_mega_corpus_gh200.sh`](../../scripts/build_mega_corpus_gh200.sh)
and
[`paper_tools/build_synthetic_dialogue_chains.py`](../../paper_tools/build_synthetic_dialogue_chains.py).
Sources:

- **v6_lme_msc_train** (existing, 6378 chains) — preserved base
- **ultrachat** (HuggingFaceH4/ultrachat_200k, ~25k chains capped)
- **pippa** (PygmalionAI/PIPPA, CharacterAI-style persona chats)
- **soda** (allenai/soda, ~25k synthetic social dialogues)
- **oasst1** (OpenAssistant/oasst1, tree-flattened assistant chats)
- **no_robots** (HuggingFaceH4/no_robots, high-quality multi-turn)
- **narrativeqa** (deepmind/narrativeqa, doc-grounded Q/A —
  memory-critical because later Qs reference earlier doc content)
- **writingprompts** (euclaise/writingprompts, long narrative
  continuation — PG-19-like memory-requiring data)

Optional sources gated behind `EXTRA_SOURCES=hh_rlhf lmsys` env var
(lmsys requires HF auth token; hh_rlhf is sometimes noisy).

All new synthetic-dialogue chains are tokenised with the Qwen3
tokenizer (identical across 0.6B/4B/8B; verified) and emitted without
`session_callback_mask` — the competition/evidence curriculum
branches still only fire on the 450 LongMemEval chains in the base
corpus, so the added chains contribute contiguous-window LM gradient
that regularises the readout against over-injection on callback-less
distributions. Source-bucket weights
(`--source_weights '{"longmemeval":4.0, "msc":3.0, "ultrachat":2.0,
"pippa":2.5, "soda":1.5, "synthdlg":1.5, "pg19":1.0, "tv":3.0,
"realtalk":2.0, "lmsys":1.5}'`) preserve the v9c LME-heavy mix while
giving dialogue / narrative sources non-trivial weight.

### v10a — `chain_v10a_composed_diverse_local` (simple_gate, composed curriculum, local GPU 0)

- **Status:** NOT STARTED → launching in tmux `local-v10a` on local H100 GPU 0.
- **Script:** [`scripts/train_v10a_composed_diverse_local.sh`](../../scripts/train_v10a_composed_diverse_local.sh)
- **Diff vs v9c:**
  - `window_k` 3 → 8 (match eval distribution)
  - `carry_state` OFF → ON (persist M_c across windows)
  - `curriculum_competition_bias` 1.0 → 0.6
  - `curriculum_evidence_bias` 0.0 → 0.3 (composed)
  - `callback_window_bias` 0.0 → 0.2
  - `callback_loss_weight` 3.0 → 2.5 (softer; v9 peak-then-decay)
  - `steps` 4000 → 8000, `batch_size` 4 → 2 `grad_accum` 2 → 4
- **Decision triggers:**
  - step 500:  `pa_cb_dsh > +0.005` AND `std_dsh > -0.003` (alive)
  - step 1500: `pa_cb_dsh > +0.015` AND `std_dsh > 0`
  - step 4000: `std_dsh > +0.005` — closes the eval-distribution gap
- **Log:** `logs/chain_v10a_composed_diverse_local.log`

### v10b — `chain_v10b_attnparity_pm4_diverse_local` (attention_parity +4/-4, local GPU 1)

- **Status:** NOT STARTED → launching in tmux `local-v10b` on local H100 GPU 1.
- **Script:** [`scripts/train_v10b_attnparity_pm4_diverse_local.sh`](../../scripts/train_v10b_attnparity_pm4_diverse_local.sh)
- **Diff vs v9c:** single-knob on routing — `memres_mode` simple_gate
  → attention_parity, `router_recent_bias_init 4`,
  `router_mem_bias_init -4`. Every other axis (corpus, curriculum,
  callback weight, window_k, grad schedule) held fixed.
- **What it tests:** with diverse data and the v9c curriculum, does
  attention_parity open α_mem > 0 within 500 steps? If yes, the 8B
  routing choice is validated and we expect the 4B GH200 cell to
  follow a similar trajectory. If no, attention_parity is
  architecturally dead independent of data and the 4B run will also
  need simple_gate — at which point we kill the 4B run early and
  relaunch.
- **Decision triggers:**
  - step 200:  `α_mem_max > 0` (any non-zero)
  - step 500:  `α_mem_max > 5e-3` AND `pa_cb_dsh > 0`
  - step 1500: `pa_cb_dsh > +0.010` — attention_parity is viable
  - **kill trigger:** step 1000 with `α_mem_max < 1e-3` — collapse
- **Log:** `logs/chain_v10b_attnparity_pm4_diverse_local.log`

### v10 4B MEGA — `chain_v10_4b_mega_attnparity_gh200` (headline, GH200, 3 days)

- **Status:** ENQUEUED on GH200 cloud_watchdog → auto-starts after
  `scripts/build_mega_corpus_gh200.sh` produces
  `paper_artifacts/chains/mega_train_s512.pt` (first invocation
  builds + tokenises + merges; ~2 h preflight).
- **Machine:** GH200 (192.222.50.225), GPU 0.
- **Script:** [`scripts/train_v10_4b_mega_attnparity_gh200.sh`](../../scripts/train_v10_4b_mega_attnparity_gh200.sh)
- **Architecture:** `qwen3-4b-xlarge` preset —
  Qwen3-4B (d=2560, 36 layers) backbone + **L_E=10**
  eleven-layer Perceiver extraction stack + K=128 slots + N=8
  AttnRes blocks. ~4.3B total params, ~52 GB peak HBM under full
  AdamW (bf16 params/grads, fp32 m/v).
- **Routing:** `memres_mode attention_parity` with
  `--router_recent_bias_init 4 --router_mem_bias_init -4`. This is
  the v3-default bias (700× softmax advantage to recent at init, but
  relaxed enough for gradient to flow to memory). Paper-spec
  depth-wise routing pool over [m^t, h_1, b_1, …, b_{N-1}].
- **Extract source:** `hidden_18` (middle of 36-layer backbone,
  consistent with hidden_14 for the 28-layer 0.6B).
- **Training schedule:** window_k=4, carry_state=True, bs=2 ga=4
  (effective 8), lr_memres=3e-5, lr_backbone=5e-6, steps=30000,
  warmup=500, cosine decay. 3-day budget: ~25-40 k steps depending
  on 4B+memres throughput (~1.5-2.5k tok/s expected on GH200 with
  gradient checkpointing).
- **Curriculum:** `curriculum_competition_bias=0.5,
  curriculum_evidence_bias=0.3, callback_window_bias=0.3,
  callback_loss_weight=3.0`. Composed: pure-competition still gets
  50% of windows (keeps judge learning), evidence-callback 30%, and
  callback-aligned contiguous 20% — with 80% of chains lacking any
  callback annotation the remaining windows are contiguous uniform
  samples from the mega corpus, supplying generic-LM regularisation.
- **Save_best:** `phase_aligned` (same as v9).
- **Decision triggers:**
  - step 500 (≈2 h): `α_mem_max > 0` and training loss ∈ [0.8, 2.5]
  - step 2000 (~8 h): `pa_cb_dsh > +0.010` AND `α_mem_max > 5e-3` AND
    `standard Δ_sh-m > -0.002` — means attention_parity is opening
    under diversity pressure; continue unconditionally.
  - step 8000 (~24 h, end of day 1): `pa_cb_dsh > +0.020` — matches
    or beats v9c peak; on track.
  - step 20000 (~48-60 h, end of day 2): `standard Δ_sh-m > +0.005`
    — the eval-distribution generalisation gap is finally closing.
  - **kill trigger:** step 2000 with `α_mem_max < 1e-3` or
    `||m^t||/||embed|| > 200` (collapse or explosion; fall back to
    simple_gate variant).
- **Log (remote):** `logs/chain_v10_4b_mega_attnparity_gh200.log`.
  Watchdog-mirror at
  `paper_tools/cloud_watchdog/logs/chain_v10_4b_mega_attnparity_gh200.log`.
- **Notifications:** ntfy topic `memres-e6ebdc70` (phone push on
  watchdog events).

---

## v9 first results + v9 ablation queue + v8b/v8c verdict (2026-04-30 ~01:35 UTC)

### v9 first results — competition curriculum is the breakthrough

`chain_v9_compete_lme_gh200` produced the strongest signal of the
entire experiment 2 trajectory in its first 1400 steps. Direct
comparison vs the leading v8b cell at the same metric class:

| metric (PA-aligned) | v8b PEAK (step 1200) | v9 step 1400 | factor |
|---|---:|---:|---:|
| CB Δ_nm-m (memory benefit on cb tokens) | +0.0073 | **+0.178** | **24.4×** |
| CB Δ_sh-m (content discrimination on cb tokens) | +0.0067 | **+0.0152** | 2.3× |
| WS Δ_nm-m | ~0 | +0.094 | n/a |
| frac_open (sublayers actively recruited) | 0.54 | **0.89** | 1.6× |
| train loss (LME) | ~1.27 | **~1.0** | −0.27 |

Full v9 trajectory (PA = phase-aligned eval, 48 chains):

| step | train loss | CB Δ_nm-m | CB Δ_sh-m | WS Δ_nm-m | gate_max | frac_open | top sublayers |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 1.15 | −0.045 | +0.016 | −0.056 | 0.0045 | 0.88 | l54+, l55+, l49+ |
| 1200 | 0.99 | +0.023 | −0.004 | +0.017 | 0.0049 | 0.86 | l55+, l54+, l49+ |
| 1400 | 1.13 | **+0.178** | **+0.015** | **+0.094** | 0.0052 | 0.89 | l55+, l41+, l54+ |

What's structurally different vs v8b/v8c:

- **frac_open ≈ 0.89** — almost every sublayer is using memory; in
  v8b only ~50% recruited. The competition curriculum makes the
  channel actually *required* for the model to lower callback CE.
- **Train loss 25% lower** at the same data corpus, same
  architecture, only the curriculum knob differs. The competition
  curriculum gives the writer + judge a sharp, definite gradient
  per sample (Sample A "keep prev" vs Sample B "write new"),
  whereas mixed-bias spreads the same supervision across diffuse
  long chains.
- **Top sublayers shifted from l48/l49 (v8b) to l54/l55 + l41 (v9)** —
  *different layers* light up when training tells the gate "memory
  really does matter." This is the recruitment pattern v6/v7/v8
  could not produce on this corpus.
- **Standard `Δ_nm-m` is NEGATIVE (−0.017)** on the 8-session
  contiguous eval, *as expected and accepted*: 7/8 of those sessions
  have no callback, and a model trained to *aggressively* write
  callback-relevant content will hurt non-callback CE. This is the
  deployment-faithful trade-off — we optimise the *callback* metric,
  not the average over corpora the model was never trained to
  predict.
- Some instability: grad_norm spiked to 30.7 at step 1300; manageable
  but worth watching. The judge-supervised regime is sharper but
  noisier than continuous LM gradient.

**Verdict:** the curriculum-decomposition hypothesis (your "Curriculum
2: Competing") is validated. Decomposing the recipe into
[judge keep-vs-write] vs [writer compaction] gives 24× more callback-
token CE benefit than mixing both axes through generic LM loss.
v9 has been promoted to the headline cell; v9_compete_lme_gh200/best
(saved at step 1400) is currently the leading checkpoint of the
program.

### v9 ablation queue — four GH200 cells, all enqueued via the watchdog

To validate that the v9 win is from the *curriculum decomposition*
specifically (and not from the callback-weight, the LME-only data,
or the simple_gate routing), four ablations were rsync'd to the
GH200 and dropped into the watchdog queue. They will auto-launch
sequentially on GH200 GPU 0 once the v9 baseline finishes
(`gpu_is_busy()` precheck defers the queue head until v9 vacates).

| order | cell | knob varied vs v9 | what it answers | script |
|---:|---|---|---|---|
| 1 | `chain_v9a_abl_cbw1_gh200` | `callback_loss_weight` 3.0 → **1.0** | is v9's win from competition curriculum or from upweighted callback supervision? if v9a still wins, curriculum is doing the heavy lifting | [`train_v9a_abl_cbw1_gh200.sh`](../../scripts/train_v9a_abl_cbw1_gh200.sh) |
| 2 | `chain_v9b_abl_mixed_gh200` | `competition_bias` 1.0 → **0.5** + `evidence_bias` 0.0 → **0.5** + `window_k` 3 → **8** | is *pure* competition necessary, or is half competition / half mixed-bias enough? | [`train_v9b_abl_mixed_gh200.sh`](../../scripts/train_v9b_abl_mixed_gh200.sh) |
| 3 | `chain_v9c_abl_diverse_gh200` | corpus LME-only → **v6_lme_msc_train** (LME+MSC+PG-19+TV) | does v9 survive data diversity, or did LME-only matter? distinguishes v8c failure (regularisation) from data axis itself | [`train_v9c_abl_diverse_gh200.sh`](../../scripts/train_v9c_abl_diverse_gh200.sh) |
| 4 | `chain_v9d_abl_attnparity_gh200` | `memres_mode` simple_gate → **attention_parity** (with RMSNorm fix retained) | now that readout magnitude is bounded, does the original paper-spec attention_parity routing become viable under the v9 curriculum? | [`train_v9d_abl_attnparity_gh200.sh`](../../scripts/train_v9d_abl_attnparity_gh200.sh) |

Each cell: 4000 steps, ~3.3 h on GH200 → total queue ≈ 13 h after v9
finishes (~03:00 UTC tomorrow). Watchdog ntfy topic
`memres-e6ebdc70` — start/finish notifications go to the phone
subscription.

Decision matrix once each ablation finishes:

- **v9a (cbw=1.0)** wins → callback-weight is a tunable secondary
  knob; primary win is competition curriculum.
- **v9a regresses to v8b-tier** → competition + cbw=3.0 are jointly
  necessary; document both as required knobs in the recipe.
- **v9b (half mixed)** matches v9 → pure competition isn't required;
  the recipe gets a regularisation bonus from contiguous windows.
- **v9b underperforms** → curriculum has to dominate; mixing dilutes
  the judge's signal.
- **v9c (diverse)** matches or beats v9 → diversity + competition is
  the headline recipe (best of both worlds; v8c's failure was the
  regularisation stack, not the data).
- **v9c collapses** → data diversity + competition curriculum
  interact negatively; LME-only is a real constraint of the recipe.
- **v9d (attention_parity)** works → original paper-spec routing is
  viable, simple_gate was just the architectural workaround for the
  v7 failure mode; ship attention_parity as the canonical primitive.
- **v9d collapses** → simple_gate is permanently part of the recipe;
  attention_parity is a paper-spec curiosity.

### v8b — KILLED (overfitting trap re-emerged at step ~2200)

- **Status:** KILLED 2026-04-30 ~01:30 UTC at step ~2460 of 4000.
- **Why killed:** the mixed-bias curriculum *delayed* the v8a
  overfitting trap by ~1000 steps but did not escape it. Between
  step 1400 and 2400, PA CB Δ_nm-m fell from +0.012 to **−0.034**
  and CB Δ_sh-m collapsed to **−0.032**. The gate stayed open
  (frac_open 0.68, gate_max 0.0085 stable) and memory was being
  used, but the readout had learned spurious callback-specific
  writes that hurt prediction. Train loss flat in [1.13, 1.34] —
  the overfitting was invisible on standard CE; only the PA-CB
  diagnostic exposed it. v9 is the curriculum-axis fix.
- **Best ckpt (preserved, was best in cohort until v9):**
  `output/chain_v8b_mixed_simplegate_rmsnorm/best` — step 1200,
  PA CB Δ_sh-m = +0.0067.
- **Full eval trajectory:**

  | step | std Δ_nm-m | PA CB Δ_nm-m | PA CB Δ_sh-m | gate_max | frac_open |
  |---:|---:|---:|---:|---:|---:|
  | 1000 | −0.0004 | +0.0046 | −0.0002 | 0.0054 | 0.39 |
  | 1200 | +0.0010 | +0.0073 | **+0.0067** | 0.0054 | 0.54 |
  | 1400 | −0.0003 | +0.0118 | +0.0038 | 0.0074 | 0.52 |
  | 1600 | −0.0025 | +0.0052 | −0.0027 | 0.0087 | 0.62 |
  | 2000 | (no eval shown — flat trajectory) |
  | 2200 | −0.0005 | **−0.0418** | **−0.0133** | 0.0088 | 0.59 |
  | 2400 | −0.0002 | −0.0340 | **−0.0320** | 0.0085 | 0.68 |

### v8c — KILLED (frozen trajectory, gate suppressed by over-regularisation)

- **Status:** KILLED 2026-04-30 ~01:30 UTC at step ~2000 of 4000.
- **Why killed:** v8c is the opposite failure mode of v8a/v8b — 2000
  steps of *no recruitment progress*. Gate magnitude has been ~0.004
  the whole time, frac_open stuck at 0.30, top sublayers signed
  *negative* (the gate is being mildly suppressed). PA-CB metrics
  oscillate in a tight ±0.007 noise band around zero. The diverse
  corpus + multi-axis regularization stack (memres_LR=1e-4, dropout,
  cbw=3.0) prevents overfitting by also preventing learning. v9c
  separates the data-diversity effect (kept) from the regularisation
  stack (dropped) to test whether diversity itself is the culprit.
- **Best ckpt:** none worth promoting; not banked in cohort.
- **Last eval (step 2000):** PA CB Δ_nm-m = +0.0034, PA CB Δ_sh-m =
  −0.0056, gate_max = 0.0036, frac_open = 0.36.

### Summary of v8/v9 cohort failure modes

| recipe | curriculum | cb-token benefit | content discr. | gate openness | verdict |
|---|---|---:|---:|---:|---|
| v8a | pure-P0 | overfit @ step 800 | overfit @ 800 | 0.4 (negative-signed) | killed |
| v8b | mixed-bias | +0.012 then collapse @ 2200 | +0.007 peak then collapse | 0.6 | **killed** |
| v8c | mixed + diverse + heavy reg | flat ~0 | flat ~0 | 0.30 stuck | **killed** |
| **v9** | **pure-competition** | **+0.178** @ 1400 | **+0.015** @ 1400 | **0.89** | **HEADLINE** |

The v8 cohort comprehensively bracketed the failure modes:
overfitting at one extreme (v8a/v8b), stalling at the other (v8c).
Both extremes are about training the writer + readout + judge
*together* through generic LM loss on a callback-token mix; the
competition curriculum (v9) is the structural fix that decomposes
the problem into trainable subparts.

---

## v9 — judge-competition curriculum (2026-04-30 00:00 UTC)

User directive 2026-04-30 ~00:00 UTC after the v8a/b/c results showed
oscillation around the same noisy minimum on PA CB Δ_sh-m: "we
separately learn the summarizer and the competition; the individual
problems then can generalize."

### Problem v9 isolates

The v8 mixed-bias curriculum (`--curriculum_evidence_bias`) trains
the writer + readout but **never explicitly trains the judge layer's
keep-vs-write decision**. Concretely, the P0 sub-window
`[evidence, callback]` has `M_c_prev = 0` at the judge step, so
`compress_session(extract(evidence), 0)` degenerates to a no-
competition aggregation (judge has nothing to compete against).
The contiguous sub-window does fire the judge with non-zero prev,
but credit assignment is diffuse across 6+ sequential judge steps.

Result: every v8 cell oscillates around `pa_cb_dsh ≈ 0` ±0.005,
with no monotone improvement on the judge's discriminative behavior.

### Curriculum design (new)

`--curriculum_competition_bias <float>` builds 3-session windows
that ISOLATE the judge subproblem. Two paired structures sampled
50/50, both scoring the same callback:

| sample | window | judge step at session 1 | correct gate | label |
|---|---|---|---:|---|
| **A: KEEP-PREV** | `[evidence, distractor, callback]` | prev_M = M_after(evidence) (relevant), C_t = extract(distractor) (irrelevant) | small | "keep" |
| **B: WRITE-NEW** | `[noise, evidence, callback]` | prev_M = M_after(noise) (irrelevant), C_t = extract(evidence) (relevant) | large | "write" |

Both samples score the same callback session ⇒ the gradient signal
from callback CE directly trains `write_gate` / judge weights to be
content-aware. The pairing forces a content-discriminative decision
rather than a degenerate "always keep" or "always write" solution.

Code change in
[`train_chain.py`](../../train_chain.py): new `ChainSampler` Branch 0
(competition pair) ahead of the existing curriculum + alignment
branches; new CLI flag `--curriculum_competition_bias`. Smoke-tested
to produce 200/200 valid 3-session windows with callback tokens
ONLY on session 2 and `first_pos < cb_pos` always.

**Label noise is intentional, not a shortcoming.** LME does not
annotate which earlier session contains the actual referenced fact
(only that `cb_pos` contains the callback question). The "evidence"
in Sample A and "noise" in Sample B are uniform random picks from
`[0, cb_pos)`. This is the *correct* training regime, because at
deployment we will not have ground-truth evidence annotations either.
The model's job is to compact arbitrary incoming sessions into M_c
such that whatever might be callback-relevant survives, and to make
keep-vs-write decisions from observed content alignment with
downstream demand alone. Across many windows the writer + judge are
forced to compact and decide using only content cues — which is
exactly the inference-time regime, and exactly the generalisable
inductive bias we want. Special-casing "evidence is the true
referent" would teach the judge to rely on an oracle signal that
won't exist at use time. The curriculum design therefore mirrors
the deployment distribution faithfully.

### v9 PURE-COMPETITION LME — `chain_v9_compete_lme_gh200`

- **Status:** TRAINING (started 2026-04-30 00:03 UTC, GH200 tmux `gh200-v9`).
- **Machine:** GH200 (192.222.50.225), GPU 0 (replaces killed v7 simplegate).
- **Script:** [`scripts/train_v9_compete_lme_gh200.sh`](../../scripts/train_v9_compete_lme_gh200.sh)
- **Cell:** simple_gate + RMSNorm + LME-only + window_k=3 + competition_bias=1.0.
  LME-only (NOT v6 mixed corpus) so the comparison vs v8b (also LME-
  only, mixed-bias 0.5) cleanly isolates the curriculum-design axis
  from the data-diversity axis (v8c separately tests data diversity).
- **Knob diff vs v8b:**

  | knob | v8b | v9 |
  |---|---:|---:|
  | window_k | 8 | **3** |
  | curriculum_evidence_bias | 0.5 | **0.0** |
  | curriculum_competition_bias | 0.0 | **1.0** (NEW) |
  | callback_loss_weight | 10.0 | **3.0** |
  | memres LR | 2e-4 | **5e-5** (1/4 of v8b) |
  | memory_dropout | 0.0 | **0.10** |
  | context_dropout | 0.0 | **0.05** |
  | batch_size / grad_accum | 2 / 4 | 4 / 2 (same effective) |

- **What it tests:** can the judge layer learn keep-vs-write in
  isolation? If yes, `pa_cb_dsh > +0.020` by step 1000 with the
  judge specifically becoming content-discriminative (visible in
  per-sublayer α_mem / write_gate trace). If no even in isolation,
  that's a sharp diagnostic of architectural insufficiency in the
  judge layer itself.
- **Decision triggers:**
  - step 200: `pa_cb_dsh > 0` AND `||m^t||/||embed||` stable
  - step 500: `pa_cb_dsh > +0.005`
  - step 1000: `pa_cb_dsh > +0.020` → judge subproblem is solvable
  - step 2000: standard `Δ_sh-m > +0.005` → ship as warm-start for
    v10 (composed curriculum: P0 + competition + contiguous)
  - **kill trigger:** train loss < 0.6 → overfitting
- **Log (remote):** `~/memory_residuals/logs/chain_v9_compete_lme_gh200.log`

---

## v8 — diagnostic lens-change + readout-norm architecture fix (2026-04-29 ~22:00 UTC)

User directive 2026-04-29 ~22:00 UTC: "I somehow think there's a problem
with the way we are evaluating; the system is learning a hard task; it
needs to first learn to summarize and use the memory, which is really
hard. I think we need to be more forgiving for the model and not look
at immediate CE, we do have to be more wary of other guidance metrics
though." This single observation triggered both an evaluation-methodology
redesign AND, downstream of that redesign, an architectural diagnosis
that the v7 P0 cells could not have surfaced.

### v8 evaluation methodology (`Trainer.evaluate` rewrite)

Four diagnostics added to `Trainer.evaluate()` and exposed both in the
training log and as a standalone tool
([`paper_tools/diagnose_recruitment.py`](../../paper_tools/diagnose_recruitment.py))
that re-evaluates any saved checkpoint:

1. **Phase-aligned eval (`pa_*`)** — matches the curriculum training
   distribution. For each LME chain with `cb_pos ≥ 1`, pick a random
   evidence position `e ∈ [0, cb_pos)`, build M_c from session `e`
   alone (fresh start, single judge step — exactly what P0 trains on),
   then score the callback session under three M_c regimes: `mem`
   (this chain's evidence), `nomem`, `shuffle` (a different chain's
   evidence). Reduce per-position NLL two ways:
   - **Whole-session (`pa_ws_*`)** over non-padding tokens (legacy
     reduction; dominated by filler).
   - **Callback-only (`pa_cb_*`)** using `session_callback_mask` —
     scores only the answer span. The strongest single signal that
     the readout is content-discriminative on the tokens that
     actually require memory to predict.

2. **Per-sublayer routing recruitment (`rec_*`)** — replaces the
   coarse `gate_max`. For `simple_gate` mode, snapshots the per-
   sublayer scalar gate (top-3 indices and signed values, fraction
   of sublayers with `|gate| > 1e-3`). For `attention_parity` /
   `attention_base`, runs a small eval batch with
   `collect_alpha_trace=True` and reports per-sublayer α_mem (the
   actual mass the depth router places on the memory source — the
   `mem_bias` parameter alone does NOT report recruitment because
   the realised α depends on the pseudo-query × normalised value
   alignment).

3. **Readout magnitude (`mt_norm_ratio_*`)** — pulse check.
   Computes `mean(||m^t||) / mean(||embed||)` on the phase-aligned
   eval setup. A ratio of ≈0 means the V projection has collapsed
   and any gate / α opening downstream is moot; a ratio ≫1 means
   m^t is dominating the residual stream.

4. **`save_best` switched to `phase_aligned`** (default in v8+).
   Composite score = `−(pa_cb_dsh + 0.5 · pa_ws_dsh)` (lower-is-
   better convention preserved). The legacy standard-eval composite
   is now a sanity check, not a checkpoint-selection signal —
   save_best should track the metric that measures *what we're
   actually training for*, not a metric on a distribution the model
   has never seen during training.

The legacy standard-eval block (eval_window=8 with sequential M_c
through 40+ sessions) is still emitted on every eval — it remains
the held-out long-chain sanity check, especially important once we
move past P0 — but it is now ONE of four diagnostics, not the only
one.

### v7 lens-change finding — `attention_parity` is *causally* collapsed

Standalone diagnostic
([`paper_tools/diagnose_recruitment.py`](../../paper_tools/diagnose_recruitment.py))
applied retroactively to `chain_v7_p0_softerbias/step-2000`
([`paper_artifacts/eval/diag_v7_softerbias_step2000.json`](../../paper_artifacts/eval/diag_v7_softerbias_step2000.json)):

| metric | value | interpretation |
|---|---:|---|
| `α_mem_max` | 0.00092 | depth router uniformly content-blind |
| `α_mem_mean` | 0.00037 | well below uniform-prior floor of 1/(N+2) |
| `frac_open` (α > 0.05) | 0.00 | zero sublayers above threshold |
| **`||m^t|| / ||embed||`** | **1.66 × 10⁻¹⁰** | **readout output is literal floating-point zero** |
| `pa_cb_dsh` (callback Δ_sh-m) | **0.0 exactly** | mem and shuffle produce identical logits to fp precision |
| `pa_cb_dnm` | −0.057 | memory mildly hurts callback CE |

The `||m^t|| / ||embed|| = 1.66e-10` and `pa_cb_dsh = 0.0` (exact)
together prove a structural failure that the standard eval was
*hiding*: the architecture has collapsed to bit-exact "no-memory"
behavior, and the standard eval reads this as `Δ_sh-m = +0.0001`
noise. The collapse mechanism:

1. `attention_parity` with `mem=−2/recent=+4` initialises α_mem at
   ~ exp(−2)/exp(+4) per non-recent source ≈ 4 × 10⁻⁴ per sublayer.
2. Gradient on `MemoryReadout.W_V` flows through `α_mem · (...)`,
   so it is ~1000× weaker than gradient on backbone params.
3. AdamW with `weight_decay=0.1` consumes this weak gradient each
   step and steadily drives `||W_V^read||` toward zero.
4. Once `||W_V^read|| ≈ 0`, `||m^t|| ≈ 0`, and the architecture is
   *causally* equivalent to "no memory" regardless of M_c.
5. Standard eval shows `Δ_sh-m ≈ 0` not because nothing is happening
   but because nothing CAN happen — both regimes contribute 0 · (...)
   to the residual stream.

This explains every `attention_parity` cell in the v3-v7 lineage. They
were not failing to learn; they were learning to *delete* the memory
channel because the saturated softmax made keeping the channel
gradient-cheaper than using it.

### v7 lens-change finding — `simple_gate` is *exploded*

Same diagnostic on `chain_v7_p0_simplegate/step-500` (GH200 cell, pulled
back via rsync into `output/chain_v7_p0_simplegate_remote/`)
([`paper_artifacts/eval/diag_v7_simplegate_step500.json`](../../paper_artifacts/eval/diag_v7_simplegate_step500.json)):

| metric | value | interpretation |
|---|---:|---|
| `gate_max_abs` / `frac_open` | 0.0085 / **0.86** | 86% of sublayers actively recruit memory |
| **`||m^t|| / ||embed||`** | **164.5** | **readout output dominates the residual stream** |
| `pa_cb_dsh` | −0.0059 | shuffle slightly *better* than mem on callback |
| `pa_cb_dnm` | **−0.66** | memory destroys callback CE — m^t is poison |

simple_gate has the *opposite* failure mode of attention_parity.
The scalar gate gives `W_V^read` direct LM gradient, so `||W_V^read||`
grows without bound. After 500 steps the readout output is 165× the
embedding norm — gate · m^t per sublayer adds ~1.3× ||embed|| to the
residual stream, and across 56 sublayers the cumulative perturbation
swamps the LM signal. The gate IS opening (the routing fix works),
but the readout it gates is unbounded.

### Root cause shared across both modes

The paper's spec assumes the foundational sources of the depth pool
(b_{−1}=m^t, b_0=h_1, b_k=block summaries) are *commensurate in
scale* — they all live in R^{S × d}. The code never enforced this for
m^t directly: there's RMSNorm on M_c (the writer's output) and
RMSNorm in `BlockAttnResRouter` for *score* computation, but no
RMSNorm on m^t itself. Consequently the scale of m^t = (attn @ V)
where V = W_V·M_c drifts to whatever ||W_V^read|| the optimizer
happens to produce — collapsing under attention_parity, exploding
under simple_gate.

### v8 architecture fix — RMSNorm on `MemoryReadout` output

Single line change in [`modeling_memres.py`](../../modeling_memres.py):
`MemoryReadout.__init__` adds `self.out_norm = Qwen3RMSNorm(...)` and
`forward` returns `self.out_norm(attn @ V)`. RMSNorm rescales m^t to
`||m^t|| ≈ √d · weight` per position, with a learnable scalar weight
initialised to 1.0. This makes m^t commensurate with embedding
magnitude regardless of `||W_V^read||`.

**Init parity preserved** across all six cases of
[`paper_tools/init_parity_test.py`](../../paper_tools/init_parity_test.py)
(re-run after the change, all six PARITY/PERTURBED expectations met
at 0.000e+00):
- `simple_gate_with_mem`: parity (gate=0 at init, so 0·RMSNorm(m^t) = 0)
- `attention_parity_with_mem`: parity (α_mem ≈ 0 at init kills the
  contribution to h regardless of m^t scale)

### v8a P0 SIMPLE_GATE + readout-RMSNorm — `chain_v8a_p0_simplegate_rmsnorm`

- **Status:** TRAINING (started 2026-04-29 22:07 UTC, local tmux `local-v8a`).
- **Machine:** local H100 NVL, GPU 0.
- **Script:** [`scripts/train_v8a_p0_simplegate_rmsnorm.sh`](../../scripts/train_v8a_p0_simplegate_rmsnorm.sh)
- **Cell:** identical to v7 SIMPLE_GATE except `MemoryReadout` now
  has `out_norm` (RMSNorm on output). Single-knob architectural diff.
- **What it tests:** with the readout magnitude bounded, does
  simple_gate's gate grow into a content-discriminative routing
  pattern on callback tokens?
- **Step-200 eval (first cycle, brand-new architecture):**

  | lens | reading |
  |---|---|
  | Standard `Δ_nm-m` (legacy) | **−0.0162** ("memory hurts") |
  | Standard `Δ_sh-m` (legacy) | −0.0001 (noise) |
  | **PA-EVAL `WS Δ_nm-m`** | **+0.0668** (memory helps whole session) |
  | **PA-EVAL `WS Δ_sh-m`** | +0.0013 (chain-specific) |
  | **PA-EVAL `CB Δ_nm-m`** | **+0.0288** (memory helps callback CE by 2.9 nats) |
  | **PA-EVAL `CB Δ_sh-m`** | **+0.0035** (callback-token content discrimination — POSITIVE) |
  | `gate_max` / `frac_open` | 0.0069 / **0.93** (top sublayers 28-30) |
  | `||m^t|| / ||embed||` | 73.5 (vs v7 simplegate 165 — RMSNorm working) |

  Under the legacy lens this checkpoint looks like memory is mildly
  harmful. Under the matched-distribution lens it's *already*
  content-discriminating on the answer span at step 200. save_best
  fired correctly on the new phase-aligned composite metric and saved
  the checkpoint at `output/chain_v8a_p0_simplegate_rmsnorm/best`.

- **Decision triggers (sharp):**
  - step 200: `pa_cb_dsh > +0.001` AND `||m^t|| / ||embed|| ∈ [0.3, 100]`. **MET** at +0.0035 and 73.5.
  - step 500: `pa_cb_dsh > +0.005` AND `gate_max > 1e-3`. *Pending.*
  - step 1000: `pa_cb_dsh > +0.020` AND `pa_ws_dsh > +0.005` →
    architecture fix is conclusive; promote v8b to headline cell on
    next checkpoint.

- **Log:** [`logs/chain_v8a_p0_simplegate_rmsnorm.log`](../../logs/chain_v8a_p0_simplegate_rmsnorm.log)

### v8c DIVERSE-CORPUS SIMPLE_GATE + readout-RMSNorm + multi-axis regularization — `chain_v8c_diverse_simplegate_rmsnorm`

- **Status:** TRAINING (started 2026-04-29 22:45 UTC, local tmux `local-v8c`).
- **Machine:** local H100 NVL, GPU 0 (replaces killed v8a).
- **Script:** [`scripts/train_v8c_diverse_simplegate_rmsnorm.sh`](../../scripts/train_v8c_diverse_simplegate_rmsnorm.sh)
- **Cell:** v8b's recipe + data-axis fix. Multi-axis attack on the v8a
  catastrophic-overfitting failure mode.

  | knob | v8b | v8c | rationale |
  |---|---:|---:|---|
  | corpus | `lme_train` (450 chains, all callback) | `v6_lme_msc_train` (6378 chains) | data diversity — only 7% of chains have callback supervision; remaining 93% provide generic LM gradient that penalises miscalibrated memory injection |
  | source weights | n/a | LME=4, MSC=3, pg19=1, tv=4, realtalk=1 | preserves ~13% effective LME callback signal in the training mix |
  | callback_loss_weight | 10.0 | **3.0** | less concentrated gradient pressure on callback tokens; readout can't win the loss by maximising one small token mask |
  | memres LR | 2e-4 | **1e-4** | slows readout learning so backbone catches up — readout is the hardest module, should learn slower not faster |
  | memory_dropout | 0.0 | **0.10** | classical regularisation — randomly drops the m^t injection per training window |
  | context_dropout | 0.0 | **0.05** | drops C_t (extracted source) randomly so the writer can't rely on every token being available |

- **Hypothesis under test:** the v8a overfitting failure was a degenerate
  solution to a narrow training distribution (450 chains × ~38 evidence
  positions = ~17k pairs, all callback-supervised). Curriculum mixing
  (v8b) is a partial fix; the deeper fix is exposing the readout to a
  data distribution where injecting memory is *usually* harmful — only
  the LME 13% rewards selective injection, the rest punishes
  over-injection. This adds a calibration constraint v8a / v8b
  fundamentally lack.
- **Decision triggers (sharp):**
  - step 500: `pa_cb_dsh > +0.005` AND standard `Δ_nm-m > -0.005`
  - step 1000: `pa_cb_dsh > +0.020` AND standard `Δ_sh-m > 0`
  - step 2000: standard `Δ_sh-m > +0.010` → ship as v9 baseline
  - **kill trigger anywhere:** train loss < 0.6 → overfitting detected
- **First-step telemetry:** loss step 20 = 3.36 (vs v8a/b at step 20
  ≈ 1.7-2.7) — much higher because the model now sees PG-19 fiction,
  MSC chats, and TV scripts, not just LME conversational format. This
  initial loss differential is *expected and good*: it means the
  diverse corpus is providing real distribution variety the model has
  to fit, not just trivial memorisation pressure.
- **Log:** [`logs/chain_v8c_diverse_simplegate_rmsnorm.log`](../../logs/chain_v8c_diverse_simplegate_rmsnorm.log)

### v8a P0 SIMPLE_GATE + readout-RMSNorm — `chain_v8a_p0_simplegate_rmsnorm` (KILLED at step ~880)

- **Status:** KILLED 2026-04-29 22:40 UTC at step ~880 of 4000.
  Catastrophic overfitting to the pure-P0 distribution; phase-aligned
  `CB Δ_nm-m` collapsed monotonically from +0.029 (step 200) →
  −1.298 (step 1000), confirming the v7 train/eval distribution
  mismatch trap survives even with the RMSNorm architectural fix.
- **What it taught us:** RMSNorm prevents the readout magnitude from
  exploding (`||m^t||/||embed||` stayed at 73.5 throughout), but
  doesn't prevent the readout from learning a degenerate solution
  that produces extreme, miscalibrated injections to maximise
  callback-token specificity. The narrow training distribution and
  high callback gradient pressure are independent failure axes that
  needed independent fixes (delivered in v8b for curriculum, v8c
  for data + regularisation).
- **Best checkpoint (saved correctly by phase-aligned save_best):**
  `output/chain_v8a_p0_simplegate_rmsnorm/best` — step 200, the
  phase-aligned callback-token Δ_sh-m peak before drift.
- **Eval trajectory (full):**

  | step | std Δ_nm-m | std Δ_sh-m | PA WS Δ_nm-m | PA CB Δ_nm-m | PA CB Δ_sh-m | gate_max | loss |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  |  200 | −0.0162 | −0.0001 | +0.0668 | **+0.0288** | **+0.0035** | 0.0069 | 1.27 |
  |  400 | −0.0576 | −0.0003 | −0.1248 | −0.6512 | +0.0136 | 0.0085 | 1.05 |
  |  600 | −0.0709 | +0.0002 | −0.1001 | −0.6350 | +0.0197 | 0.0107 | 0.85 |
  |  800 | −0.1164 | −0.0003 | −0.4202 | −1.0071 | +0.0056 | 0.0119 | 0.78 |
  | 1000 | −0.1756 | −0.0002 | −0.5958 | −1.2978 | −0.0041 | 0.0134 | 0.75 |

  Pattern: callback specificity (PA CB Δ_sh-m) grew monotonically
  through step 600 and then collapsed; PA CB Δ_nm-m collapsed
  immediately after the step-200 peak. The model learned to be MORE
  chain-discriminative on callback tokens (Δ_sh-m up) but in a way
  that produced miscalibrated injection that hurt overall predictions
  (Δ_nm-m down). Train loss kept dropping = textbook overfitting on a
  degenerate solution.
- **Log:** [`logs/chain_v8a_p0_simplegate_rmsnorm.log`](../../logs/chain_v8a_p0_simplegate_rmsnorm.log)

### v8b MIXED SIMPLE_GATE + readout-RMSNorm — `chain_v8b_mixed_simplegate_rmsnorm`

- **Status:** TRAINING (started 2026-04-29 22:07 UTC, local tmux `local-v8b`). At step ~250 of 4000 as of 22:25 UTC.
- **Machine:** local H100 NVL, GPU 1.
- **Script:** [`scripts/train_v8b_mixed_simplegate_rmsnorm.sh`](../../scripts/train_v8b_mixed_simplegate_rmsnorm.sh)
- **Cell:** v8a + `--curriculum_evidence_bias 0.5` + `--window_k 8`.
  Half the windows are P0 evidence+callback (compressed credit-
  assignment), half are full contiguous window_k=8 chains.
  When the curriculum branch fires the window structure is
  `[evidence, intermediate_1, ..., intermediate_6, callback]` —
  1 fact + 6 distractors + 1 recall. The writer must DEFEND the
  evidence-bearing M_c slot through 6 intervening judge updates
  before callback supervision pulls on it. The contiguous branch
  exposes the readout to the eval-shaped M_c regime regularly.
- **What it tests:** does the mixed-bias curriculum prevent the
  v7-P0 train/eval distribution mismatch trap WITHOUT losing the
  callback-token specificity gain?
- **Step-200 eval (FIRST CYCLE — headline result):**

  | lens | reading | v8a step 200 (compare) |
  |---|---|---|
  | Standard `Δ_nm-m` | **+0.0020** (memory helps long-chain!) | −0.0162 |
  | Standard `Δ_sh-m` | −0.0004 (noise) | −0.0001 |
  | Standard `Δ_or-m` | +0.0767 (mem better than oracle) | +0.0162 |
  | PA-EVAL `WS Δ_nm-m` | −0.0014 (neutral) | +0.0668 |
  | PA-EVAL `WS Δ_sh-m` | −0.0008 (noise) | +0.0013 |
  | **PA-EVAL `CB Δ_nm-m`** | **+0.0076** (positive on cb tokens) | +0.0288 |
  | **PA-EVAL `CB Δ_sh-m`** | **+0.0033** (matched callback specificity) | +0.0035 |
  | `gate_max` / `frac_open` | **0.0050 / 0.46** (sparser) | 0.0069 / 0.93 |
  | Gate top sublayers | **52, 55, 53 (positive)** | 28, 30, 51 (negative) |
  | `||m^t||/||embed||` | 73.5 (RMSNorm holding) | 73.5 |
  | training loss | 1.55 → 1.47 | 1.57 → 1.27 (overfitting) |
  | gradient norm (typ.) | **2.0–2.5** | 13–22 |

  The mixed-bias regularisation is producing a qualitatively
  different (and much more sane) memory-usage pattern:
  - Memory injection is sparser (46% of sublayers vs 93%) and
    concentrated in the LATE layers (52, 53, 55 — the last block of
    the model). This is the architecturally natural place for a
    "remember this fact" head: late enough that the LM has set up
    its prediction state, early enough that the modification can
    propagate to logits.
  - Gate values are POSITIVE in the active sublayers (vs v8a's
    negatives in mid-sublayers), meaning the model has learned to
    *augment* the residual stream with memory rather than *suppress*
    parts of it.
  - Standard-eval `Δ_nm-m` is positive — the cell that v7 P0 could
    never make non-negative is *already* positive at step 200 of
    v8b, on the same eval, with the only change being:
    `(routing simple_gate) + (RMSNorm on readout) + (mixed-bias 0.5)
    + (window_k=8)`. This is the recipe we've been hunting for.

- **Decision triggers:**
  - step 200: `pa_cb_dsh > +0.001` AND standard `Δ_nm-m > -0.005`. **MET** at +0.0033 and +0.0020.
  - step 500: `pa_cb_dsh > +0.005` (curriculum survives 6-distractor depth)
  - step 1000: `pa_cb_dsh > +0.020` AND standard `Δ_sh-m > 0`
  - step 2000: standard `Δ_sh-m > +0.010` → ship as v9 baseline
- **Log:** [`logs/chain_v8b_mixed_simplegate_rmsnorm.log`](../../logs/chain_v8b_mixed_simplegate_rmsnorm.log)

### Why the standard eval was misleading us for 2k steps

A retrospective on what the lens change reveals: every cell in the
v3-v7 lineage was being judged on a metric whose two failure modes
(attention_parity collapse, simple_gate explosion) both produce
`Δ_sh-m ≈ 0` on the standard eval. That standard `Δ_sh-m ≈ 0` was
read as "memory channel is not yet content-aware, keep training" or
"distribution mismatch noise." Neither read is wrong, but both miss
that the underlying architecture has reached a *terminal* state from
which more training cannot recover. The phase-aligned callback-token
diagnostic discriminates between these states cleanly:

- Collapsed (attention_parity): `pa_cb_dsh = 0.0` exactly,
  `||m^t|| ≈ 0`, RMSNorm or no-weight-decay are necessary fixes.
- Exploded (simple_gate without RMSNorm): `pa_cb_dsh ≈ 0` but
  `pa_cb_dnm ≪ 0` and `||m^t||` ≫ `||embed||`, RMSNorm fixes it.
- Healthy (v8a step 200): `pa_cb_dsh > 0`, `pa_cb_dnm > 0`,
  `||m^t||` of the same order as `||embed||`, gate growing.

---

## v7 P0 runs — KILLED 2026-04-29 22:00 UTC after diagnostic lens-change

All three v7 cells killed when the new diagnostics revealed the
shared root cause (no scale control on m^t):

- `chain_v7_p0_softerbias` — KILLED at step ~2000. `||m^t|| ≈ 1.7e-10`,
  readout collapsed under weight decay. attention_parity terminal state.
- `chain_v7_p0_v3bias` — KILLED at step ~2000. Same collapse mode as
  softerbias, plus the standard-eval Δ_nm-m had drifted to −0.23
  (runaway divergence on the eval distribution).
- `chain_v7_p0_simplegate` (GH200) — *still running* as of v8 launch
  for additional no-RMSNorm baseline data; will be killed once v8a
  passes step-1000 decision trigger or v8b confirms the mixed-bias
  hypothesis.

Original v7 P0 ledger entries below are preserved for the paper trail.

## v7 P0 runs (active) — compression curriculum + bias relaxation

User directive 2026-04-29 ~14:30 CDT after the v6 architectural
diagnosis: the depth router is the real bottleneck, not the writer.
v3 standalone routing trace (`paper_artifacts/eval/routing_v3sp_*.json`)
shows α_mem ≈ 4.7e-4 averaged across 55 sublayers AND α_mem essentially
identical between true-chain and shuffled-chain memory at every
sublayer — i.e. the router is content-blind. v6 GATED/COMPETITIVE
reproduced the v3 failure mode (gate_max=0.0000 through step 1400,
both cells *regressed* on Δ_sh-m at step 1200-1400) with a more severe
collapse than v3, plausibly because the gated update with g_init≈0.27
weakens the LM's gradient pressure on the router to use M_c. v6
COMPETITIVE A/B is uninterpretable when both cells fail at the
router; v6 GATED itself is just retracing v3. Both KILLED in favour
of the v7 pivot below.

### v7 pivot summary (vs v6)

| axis | v6 | v7 P0 |
|---|---|---|
| credit-assignment path | full chain TBPTT through ~8 sessions of mostly-irrelevant filler before the callback | **compression curriculum P0**: every training window is `[evidence_session, callback_session]` (window_k=2). Sampler builds non-contiguous windows; M_c starts fresh, compresses ONE evidence session, next session contains the callback. Shortest possible path from "fact written into M_c" to "callback supervision pulls on it." |
| router bias | `recent=+4, mem=-4` (v3 default). Memory mass per sublayer at init ≈ e^{-8}/(e^4 + 8) ≈ 3e-5 — saturated softmax against memory. | A/B: cell SOFTERBIAS uses `recent=+4, mem=-2` (~70× lift on initial memory mass to ~2e-3); cell V3BIAS keeps `recent=+4, mem=-4` to isolate the curriculum's contribution. |
| burn-in | `burn_in_max=24` with resample | **0** — clean credit-assignment chain, no no-grad prefix |
| carry_state | True (M_c persists across windows) | **False** — fresh M_c per window so the recurrent depth being trained is exactly 1 (P0) |
| callback alignment | `callback_window_bias=0.7` (70% of windows include callback) | superseded by `curriculum_evidence_bias=1.0` (100% of windows are curriculum windows) |
| writer | `--memres_update_mode gated` (init g≈0.27) | **same** — orthogonal to the router question; we'll ablate later if P0 succeeds |

### v7 P0 SOFTERBIAS — `chain_v7_p0_softerbias`

- **Status:** TRAINING (started 2026-04-29 20:40 UTC, local tmux `local-v7-p0-softerbias`). At step ~1400 of 4000 as of 21:24 UTC.
- **Machine:** local H100 NVL, GPU 0.
- **Script:** [`scripts/train_v7_p0_softerbias.sh`](../../scripts/train_v7_p0_softerbias.sh)
- **Cell:** P0 curriculum × `recent=+4, mem=-2` (relaxed) × gated × LME × callback loss × window_k=2.
- **What it tests:** can compression curriculum P0 + a 70× memory-bias lift open the depth router that v3-v6 could not? Concrete first signal: gate_max > 1e-4 by step 200.
- **Decision triggers (sharp, falsifiable):**
  - step 200: `gate_max > 1e-4`. **MISSED** — gate_max=0.0000 throughout (see eval table).
  - step 500: `Δ_sh-m on lme_val > +0.005`. **MISSED** — Δ_sh-m sits at +0.0001 / +0.0000 across all evals.
  - step 1000: `Δ_sh-m > +0.01` AND `gate_max > 1e-3`. **MISSED** — both still zero.
- **Eval trajectory (n=256, eval_window=8 sequential M_c):**

  | step | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m | Δ_sh-m | gate_max | loss |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  |  200 | 1.5322 | 1.5269 | 1.5321 | 1.5698 | **−0.0052** | −0.0001 | 0.0000 | ~1.5 |
  |  400 | 1.4866 | 1.4805 | 1.4866 | 1.5397 | **−0.0060** | +0.0001 | 0.0000 | ~1.0 |
  |  600 | 1.4727 | 1.4649 | 1.4727 | 1.5344 | **−0.0078** | +0.0000 | 0.0000 | ~1.0 |
  |  800 | 1.4390 | 1.4365 | 1.4390 | 1.4999 | **−0.0025** | −0.0000 | 0.0000 | ~0.9 |
  | 1000 | 1.4254 | 1.4237 | 1.4254 | 1.4956 | **−0.0018** | +0.0000 | 0.0000 | ~0.85 |
  | 1200 | 1.4178 | 1.4159 | 1.4178 | 1.4841 | **−0.0019** | +0.0000 | 0.0000 | ~0.8 |

- **Verdict so far:** memory is mildly *harmful* on the eval distribution and gate_max never moves off zero. Train loss drops cleanly from 4.17 → 0.78 because the model fits the (evidence + callback) curriculum task — but the readout learned on that train distribution produces actively counterproductive output when applied to the eval distribution (M_c built sequentially through 40+ sessions). This is the curriculum/eval mismatch failure mode (see "v7 P0 results so far" below). The bias relaxation alone did not break attention_parity's collapse.
- **Memory footprint:** 16.5 GiB / 95.8 GiB on H100 (vs v6 GATED's 50.7 GiB) — TBPTT depth is 2 instead of 8.
- **Log:** [`logs/chain_v7_p0_softerbias.log`](../../logs/chain_v7_p0_softerbias.log)

### v7 P0 SIMPLE_GATE — `chain_v7_p0_simplegate` (architectural ablation, GH200)

- **Status:** TRAINING (started 2026-04-29 21:03 UTC via cloud_watchdog spec `1777496603_chain_v7_p0_simplegate.json`). At step ~400 of 4000 as of 21:24 UTC.
- **Machine:** GH200 (192.222.50.225), GPU 0.
- **Script:** [`scripts/train_v7_p0_simplegate_gh200.sh`](../../scripts/train_v7_p0_simplegate_gh200.sh)
- **Cell:** P0 curriculum × **`--memres_mode simple_gate`** (vs `attention_parity`) × gated × LME × callback loss × window_k=2.
- **Single-knob diff vs v7 P0 SOFTERBIAS:** `--memres_mode simple_gate` (vs `attention_parity`). Router bias args removed (unused in simple_gate mode — there is no depth-wise softmax router; `memory_gate.gate` is one learnable scalar per sublayer with bit-exact init parity at g=0).
- **What it tests:** does simple_gate's direct gradient path open the memory channel where attention_parity could not?
- **Decision triggers:**
  - step 20: gate_max > 1e-4. **MET** at 0.0004.
  - step 200: gate_max > 1e-3. **MET** at 0.0051.
  - step 500: Δ_sh-m on lme_val > +0.005. *Pending* (~next 2 evals, step 400/600).
  - step 1000: gate_max > 0.01 AND Δ_sh-m > +0.01 → simple_gate becomes the new architectural baseline.
- **gate_max trajectory (per-step telemetry, every 20 steps):**

  | step | gate_mean | gate_max | loss |
  |---:|---:|---:|---:|
  |  20 | −0.0000 | **0.0004** | 2.74 |
  |  40 | −0.0004 | **0.0015** | 2.28 |
  |  60 | −0.0015 | **0.0032** | 2.24 |
  |  80 | −0.0023 | **0.0049** | 1.69 |
  | 100 | −0.0023 | **0.0055** | 1.74 |
  | 200 | −0.0019 | **0.0051** | 1.56 |
  | 300 | −0.0023 | **0.0065** | 1.16 |
  | 400 | −0.0024 | **0.0074** | 1.07 |

  Compare attention_parity at any step in v3/v6/v7: gate_max stays at 0.0000 indefinitely. simple_gate's gate is monotonically growing (0.0004 → 0.0074 across step 20-400), with gate_mean firmly negative — a learned per-sublayer routing pattern (some sublayers up, most sublayers slightly down). **This is the only configuration in the v3-v7 lineage on the chain trainer that has produced a non-zero gate_max.**

- **Eval trajectory (n=256, eval_window=8):**

  | step | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m | Δ_sh-m | gate_max |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 200 | 1.5656 | 1.5216 | 1.5655 | 1.5553 | **−0.0440** | −0.0001 | 0.0051 |

  - **Δ_sh-m is in the noise** (−0.0001) at step 200 — the gate is open but the readout is not yet producing chain-discriminative output.
  - **Δ_nm-m is strongly negative** (−0.044) — same curriculum/eval-mismatch failure mode as the local cells, despite the architectural change. Confirms the failure is *upstream* of routing mode: it's a train/eval distribution mismatch on M_c, not a routing-side problem.

- **Replaces:** killed `chain_v6_lme_gated_callback_w12` at 2026-04-29 21:03 UTC step ~420 (GH200 freed). w12 was the deepest TBPTT cell in the v6 family (window_k=12, eval_window=12) and was showing the same gate_max=0 pattern at step 400 (Δ_sh-m=−0.0001, Δ_nm-m=−0.0004) — confirming the v3/v6 router-collapse pattern is depth-axis-independent. Spec now in `paper_tools/cloud_watchdog/failed/`. Decision rationale: simple_gate cell is much higher information per GH200-hour than letting w12 run to completion.
- **Watchdog patch landed:** `paper_tools/cloud_watchdog/watchdog.sh` now has a `gpu_is_busy()` precheck (defers spec launch if another CUDA process is using > `GPU_BUSY_MIB` MiB on the target GPU; default 2048). Watchdog process was restarted to pick up the patch — confirmed in log `[watchdog 2026-04-29 21:03:07] starting watchdog: poll=30s, queue=..., gpu_busy_mib=2048`. Prevents the 2026-04-29 PURIST-style co-launch OOM from recurring.
- **Log (remote):** `paper_tools/cloud_watchdog/logs/chain_v7_p0_simplegate.log`

### v7 P0 V3BIAS — `chain_v7_p0_v3bias` (single-knob A/B for the bias)

- **Status:** TRAINING but **escalation candidate** (kill-eligible per the v7 results section). At step ~1400 of 4000 as of 21:24 UTC. Started 2026-04-29 20:40 UTC, local tmux `local-v7-p0-v3bias`.
- **Machine:** local H100 NVL, GPU 1.
- **Script:** [`scripts/train_v7_p0_v3bias.sh`](../../scripts/train_v7_p0_v3bias.sh)
- **Cell:** identical to SOFTERBIAS except `--router_mem_bias_init -4` (v3 default) instead of `-2`.
- **Single-knob diff vs SOFTERBIAS:** `--router_mem_bias_init -4` (was `-2`).
- **What it tests:** does the curriculum alone open the channel, independent of bias relaxation? (Now confirmed answered: NO. Both V3BIAS and SOFTERBIAS show gate_max=0.0000 throughout — the bias relaxation was insufficient and the curriculum alone is also insufficient against attention_parity collapse.)
- **Eval trajectory (n=256, eval_window=8):**

  | step | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m | Δ_sh-m | gate_max |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  |  200 | 1.5435 | 1.5393 | 1.5436 | 1.5763 | **−0.0042** | +0.0001 | 0.0000 |
  |  400 | 1.5066 | 1.4807 | 1.5059 | 1.5416 | **−0.0259** | −0.0007 | 0.0000 |
  |  600 | 1.5135 | 1.4609 | 1.5135 | 1.5195 | **−0.0526** | −0.0000 | 0.0000 |
  |  800 | 1.5444 | 1.4271 | 1.5438 | 1.4964 | **−0.1173** | −0.0006 | 0.0000 |
  | 1000 | 1.5086 | 1.4140 | 1.5086 | 1.4889 | **−0.0945** | +0.0000 | 0.0000 |
  | 1200 | 1.5526 | 1.4057 | 1.5524 | 1.4794 | **−0.1470** | −0.0003 | 0.0000 |

- **Verdict:** worst trajectory of the three v7 cells. Δ_nm-m is monotonically deteriorating: memory makes the model 14.7% worse than no memory at step 1200. Δ_or-m has gone *negative* at step 800/1200 — the no-memory baseline now beats the oracle (raw concat of last-4 sessions), which is a sign the model has overfit hard to the curriculum's 2-session distribution and is producing degenerate eval-time logits when the M_c looks unfamiliar. Recommend kill at next eval (~step 1400) and free GPU 1 for a v8 cell.
- **Log:** [`logs/chain_v7_p0_v3bias.log`](../../logs/chain_v7_p0_v3bias.log)

### v7 P0 results so far (snapshot 2026-04-29 21:24 UTC, ~step 200-1400)

Two distinct findings landed inside the first 1k steps of the v7 P0
trio. Both are publishable, both are uncomfortable.

**Finding 1 — `simple_gate` opens the memory channel where
`attention_parity` cannot.** This is the empirical confirmation of the
routing-side failure diagnosis from the v3 standalone routing trace.
Same corpus, same writer (gated), same curriculum (P0 evidence+callback,
window_k=2), same hidden_14 extract — the only knob that differs across
the three cells is the routing mode. Routing-side telemetry:

| cell | routing | step at first non-zero gate_max | gate_max @ step 400 |
|---|---|---|---|
| v7 P0 SOFTERBIAS | `attention_parity`, mem=−2 | never (0.0000 through step 1400) | 0.0000 |
| v7 P0 V3BIAS    | `attention_parity`, mem=−4 | never (0.0000 through step 1400) | 0.0000 |
| v7 P0 SIMPLE_GATE | `simple_gate`           | **step 20** (gate_max=0.0004)    | **0.0074** |

simple_gate's gate_max grew monotonically from 0.0004 (step 20) →
0.0074 (step 400), with gate_mean firmly negative (~−0.002) — i.e. a
*learned* per-sublayer routing pattern (a few sublayers up, most slightly
down). The two `attention_parity` cells stayed pinned at 0.0000 across
1400+ steps despite the curriculum and bias relaxation. The `+4`
recent-source bias in `attention_parity` saturates the depth softmax
against memory and the gradient signal needed to relax that bias is
overwhelmed by the gradient signal that wants to keep recent. simple_gate
has a direct gradient path to its scalar gate (no softmax competition),
and gradient finds it within 20 steps.

**This rules out `attention_parity` as the routing mode that ships in
the recipe.** The paper's "block AttnRes routing pool over delta sources"
section was the architectural rejection of simple_gate; the empirical
result is the opposite. Implications for writing in §6 below.

**Finding 2 — pure-P0 curriculum (`curriculum_evidence_bias=1.0`) is a
train/eval distribution mismatch trap.** All three cells learn the
training task aggressively (loss drops from ~3 to <1 in 800 steps) but
all three show **negative Δ_nm-m on the standard eval** (eval_window=8,
sequential M_c through ~40 prior sessions). V3BIAS is the worst —
Δ_nm-m=−0.147 at step 1200, i.e. memory makes the model 14.7% worse than
no memory. Even SIMPLE_GATE, which fixes the routing-mode failure,
shows Δ_nm-m=−0.044 at step 200.

Mechanism: training windows are `[evidence, callback]` with M_c starting
fresh at evidence (no carry_state, no burn-in). The readout learns
"attend to M_c built from one fresh evidence session." At eval, M_c is
built sequentially through 40+ sessions of compounding judge updates;
its statistics are completely different. The readout produces
chain-discriminative-but-noisy output on this distribution, which the
LM head then compounds into a CE that's worse than turning memory off.

Δ_sh-m on all three cells stays in the noise (|Δ_sh-m| ≤ 0.001) — the
readout *is* discriminating chain content (otherwise mem ≈ shuffle)
but the discrimination signal is tiny compared to the noise the
distribution mismatch introduces.

**Why this matters strategically.** simple_gate's gate-opening result
is real and important, but on its own it does *not* solve the chain-
trainer failure — it solves one of two stacked failures. We need a
curriculum redesign that doesn't trade routing collapse for distribution
collapse. Three options on the table for v8:

1. **Mixed-bias curriculum.** `curriculum_evidence_bias=0.5` — half the
   windows are P0 evidence+callback, half are full contiguous
   window_k=8. Model sees both regimes during training; the readout
   learns to handle both M_c distributions. Cheapest fix, most likely
   to "just work" without further surgery.
2. **Match eval to train AND eval-on-promotion.** Add a phase-aligned
   eval (eval_window=2) alongside the standard eval_window=8 so we can
   *see* P0 progress without distribution drift; only graduate to P1
   when phase-aligned eval Δ_sh-m crosses +0.005. Track both curves.
3. **In-run phase annealing.** Anneal `curriculum_evidence_bias`
   linearly from 1.0 → 0.0 over the first 1k steps so the model
   bootstraps on P0, then is forced to handle the full distribution
   without re-launching. Rolls 1+2 into one run but harder to debug.

Likely v8 plan: combine (1) and (2) — mixed-bias 0.5 with phase-aligned
eval — running on simple_gate routing (since that's the only one that
passes the routing test). One v8 cell on each axis combination, pending
go-ahead.

**What the running cells will tell us in the next ~3 hours.** The
GH200 SIMPLE_GATE cell hits step 800/1000/1200 evals over the next
several hours. If its Δ_nm-m trajectory mirrors V3BIAS (deteriorating
toward −0.1) the curriculum failure dominates routing-mode benefit and
we should kill it too. If Δ_nm-m stabilises or recovers (i.e.
simple_gate's content-aware routing partially compensates for the
distribution mismatch) the cell continues being informative. The local
SOFTERBIAS cell is borderline — Δ_nm-m has *recovered* slightly from
−0.0078 at step 600 to −0.0019 at step 1200, possibly because the
softer bias gives the readout a smaller "lever" to break things with.
Worth one more eval (~step 1400-1600) before deciding.

### Curriculum sampler implementation

New CLI knob: `--curriculum_evidence_bias <float>` (default 0.0; both v7
cells set 1.0 = pure P0). When active for a chain that has an annotated
callback position with enough room (cb_pos ≥ window_k − 1), the
sampler:

1. Picks a random evidence position in `[0, cb_pos)`.
2. Picks `window_k − 2` random intermediate positions strictly
   between evidence and callback.
3. Stacks `[evidence, ...intermediates, callback]` in chronological
   order via `ChainCorpus.chain_curriculum_window(positions)`.
4. Returns the precomputed callback mask aligned to the
   non-contiguous slice (the trainer falls back to slicing the
   contiguous mask only when this is None).

For chains with insufficient callback gap (rare on LME — all 450/450
LME chains have cb_pos ≥ 38) the sampler falls through to the existing
`callback_window_bias` path. For non-callback corpora (PG-19 / TV /
REALTALK) the sampler always falls through to contiguous sampling
because `chain_callback_position` is the -1 sentinel.

The phase is implied by `window_k`: P0 = window_k 2, P1 = 3,
P2 = 5, P3 = 8 (= no curriculum).

---

## v6 long-horizon runs (folded into COMPREHENSIVE.md Part IV)

All four v6 cells (`chain_v6_lme_gated_callback`,
`chain_v6_lme_competitive_callback`, `chain_v6_lme_gated_purist`,
`chain_v6_lme_gated_callback_w12`) and the v5/v6 pivot summary, v6
corpus build provenance, and v6 code changes have been folded into
[`COMPREHENSIVE.md`](../../COMPREHENSIVE.md) Part IV per the
"fold superseded entries when the ledger gets crowded" convention
below. The v6 → v7 transition narrative lives there too.

Last-eval summary (full tables in Part IV):

| cell | machine | killed at | Δ_sh-m peak | Δ_sh-m last | gate_max |
|---|---|---:|---:|---:|---:|
| v6 GATED | local H100 GPU 0 | step ~1410 | +0.0005 (step 1000) | −0.0001 | 0.0000 |
| v6 COMPETITIVE | local H100 GPU 1 | step ~1400 | +0.0000 | −0.0004 | 0.0000 |
| v6 GATED-DEEP w12 | GH200 GPU 0 | step ~420 | +0.0003 (step 200) | +0.0003 | 0.0000 |
| v6 PURIST | GH200 GPU 0 | step 0 | n/a (OOM) | n/a | n/a |

v6 PURIST OOM was caused by the watchdog co-launching it onto a busy
GPU; fixed by the `gpu_is_busy()` precheck in
`paper_tools/cloud_watchdog/watchdog.sh` (2026-04-29 ~21:03 UTC).

---

## How to add a new run

1. Create / pick a `scripts/train_*.sh` for the run. The script's
   header MUST document what it varies vs the closest existing run.
2. Pre-allocate the `chain_<version>_<descriptor>` run name in this
   ledger as `Status: NOT STARTED` with the script path filled in.
3. Launch (locally in tmux or via the cloud watchdog
   `enqueue.sh`).
4. Update Status to TRAINING with start timestamp + machine.
5. On finish / kill / failure, update Status with end timestamp,
   reason, last step, last `Δ_sh-m`, location of the `best/` ckpt,
   and link to the log.
6. When the post-train pipeline runs, link the eval JSONs.

Do NOT delete entries when a run is superseded. Mark them KILLED /
SUPERSEDED-BY and keep the paper trail. When the active ledger gets
crowded, fold superseded entries into `COMPREHENSIVE.md` Part IV.

## Conventions

- Run names: `chain_v<N>_<descriptor>` where `<N>` increments on a
  major recipe / architectural change.
- Tmux naming: cloud watchdog uses `cwd-<run_name>`; local launches
  use `local-<run_name>`.
- Log paths: cloud → `~/memory_residuals/paper_tools/cloud_watchdog/logs/<run_name>.log`;
  local → `logs/<run_name>.log`.
- Output dirs: always `output/<run_name>/{step-N, best/}`.
