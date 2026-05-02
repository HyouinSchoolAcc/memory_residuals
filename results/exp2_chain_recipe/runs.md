# Runs ledger — experiment 2 (long-horizon recipe)

Single source of truth for which checkpoint produced which number in
the recipe paper. Keep it updated as runs finish, fail, or get
re-launched. Newest at the top.

## v14 campaign — router recruits, judge decides, writer discriminates (2026-05-02 ~15:15 UTC-5)

### Why v14 exists (one paragraph)

v13c2 + v13r settled two things and exposed two new ones.  Settled: the
WRITER can specialise (evidence_lift +1.41 on v13c2 @ step 400; D3-MC
pair/self = 0.004 held throughout v13r's 10500 steps — orthogonal init
+ slot-positional is a permanent symmetry break).  Exposed: (A) the
ROUTER actively rejects memory during joint training (v13r @ step
10000 alpha_mem_mean = 0.0011 — textbook MoE expert collapse, nothing
in the architecture obligates the router to recruit); (B) the JUDGE
re-collapses to uniform when the backbone unfreezes (D2 row_entropy /
log(2K) = 0.988 on v13r vs 0.890 on v13c2's frozen-backbone D4 regime
— the attention-entropy-collapse direction flagged in Zhai et al.
2023).  A third sign we missed until the deep-rereading pass: v13r's
D3-MC `Δ_step mean = 0.028` is **44× smaller** than v13c2's (1.244) —
the judge on natural prose converges to "always preserve old memory,"
so M_c barely updates in joint training (Paper Eq. 2's "forgetting
defense" became a learning defense).

### v14 fixes and the prior art each draws from

Four orthogonal interventions, each with decades of literature behind it:

1. **`alpha_mem` floor auxiliary loss** (weight=0.01, target=0.05) —
   MoE load-balance penalty.  Router is obligated to keep sampling
   memory so the downstream LM gradient keeps reaching the writer /
   readout.  Cite: Fedus et al. 2021 Switch Transformer; Wang et al.
   2024 auxiliary-loss-free balancing.
2. **InfoNCE contrastive loss** (weight=0.5, callback-only) — already
   implemented in `train_chain.py` (contrastive_infonce_*) but
   unused on every v13 run; gives the writer direct discriminative
   signal ("my M_c beats B-1 negatives on predicting MY callback
   tokens").  Cite: AutoCompressors (Chevalier et al. EMNLP 2023)
   and TRIME (Zhong et al. EMNLP 2022).
3. **Judge QK-LayerNorm** — post-projection RMSNorm on Q/K of
   `MemoryBlock.judging`.  Decouples attention-logit magnitude from
   W_Q/W_K spectral norm, letting the judge softmax find sharper
   distributions.  Applied only to the judge (D1/D3 show extract
   side is fine).  Cite: Zhai et al. ICML 2023 σReparam; Qwen /
   Gemma / DeepSeek-V3 attention convention.
4. **AP `router_mem_bias` warmup anneal** — force-held at +4 during
   the 500-step writer warmup, annealed to 0 over 200 steps.
   Parallel fix to simple_gate's memory_gate force-open from v13
   (which already shipped); required for AP because otherwise
   router.mem_bias sits at 0 the entire warmup and writer gradient
   vanishes.

### v14 code landed (2026-05-02 ~15:15 UTC-5)

- `src/modeling_memres.py`:
  * New `Qwen3MemResConfig.memres_judge_qk_layernorm: bool`
  * `CrossAttention.__init__(qk_layernorm)` + `SlotAttentionWriter.__init__(qk_layernorm)`
    with post-projection RMSNorm on Q and K; Identity when flag off
    (bit-exact backwards compat for v13 runs)
  * `MemoryBlock.__init__(judge_qk_layernorm)` threads the flag into
    `self.judging`
- `src/train_chain.py`:
  * New CLI `--memres_judge_qk_layernorm`, `--alpha_mem_floor_aux_weight`,
    `--alpha_mem_floor_target`
  * Main training forwards pass `collect_alpha_trace=True` iff aux
    weight > 0 (kept off on memory-dropout forwards where the floor
    isn't well-defined)
  * alpha_floor aux computed as `mean_l relu(target - mean(alpha_mem_l))`
    and added to `total_loss`
  * Log line + wandb payload expose `a_floor` and `a_mean` when
    aux weight > 0

Unit-tested locally + GH200: build-forward-backward with all flags
on produces non-NaN grads on M_in, M_judge, judge.q_norm/.k_norm,
depth_router.mem_bias, memory_readout.W_Q (run: `tools/init_parity_test.py`-
shaped sanity — see conversation thread).

### v14 queue and kept/dequeued v13 cells

Per the 2026-05-02 user directive ("kill the current run and save
the frozen backbone run"):
- **Killed**: v13r (step 10500/16000; evidence_lift +0.006 sustained,
  PA-EVAL CB Δsh-m +0.021 but ceiling; long-warmup hypothesis
  falsified)
- **Dequeued**: v13b (SG D4 frozen), v13d (D4 no-warmup) — both were
  SG-side ablations, redundant once v14 tests all four interventions
  at once on AP
- **Kept**: v13q (AP + FROZEN BACKBONE + curriculum + mega + 6000
  steps) as the single CORE v13 test — directly tests whether
  *permanent* freeze prevents the joint-training washout we
  observed on v13r

| # | Run | State | Routing | Backbone | Corpus | v14 fixes? | Steps | Tests |
|---|---|---|---|---|---|---|---|---|
| 0 | **v13q** | **RUNNING** (step ~140 as of 15:17 UTC-5) | attention_parity | **FROZEN** | mega | no (v13 stack) | 6000 | Does permanent freeze alone prevent phase-2 washout?  If YES, paper headline = frozen-backbone config; no v14 needed for this question |
| 1 | **v14a** | **QUEUED NEXT** | attention_parity | trained (phase 2) | mega | **YES (all 4)** | 8000 | Full v14 stack HEADLINE candidate.  Tests whether the four orthogonal interventions close the gap v13r left open on joint training |

#### v13q — core frozen-backbone test, running

Script: `scripts/train_v13q_ap_frozen_curriculum_mega_gh200.sh`
(unchanged from 2026-05-01; backbone stays frozen for full 6000
steps via `--writer_warmup_keep_backbone_frozen`).

Rationale for keeping: v13r's failure mode (phase-2 unfreeze destroys
writer specialisation) strongly predicts the frozen regime is the
safer headline if we want a clean "Memory Residuals are a drop-in
augmentation over a pretrained LM" paper story.  v13q is the
cleanest empirical test of that hypothesis at scale (AP + mega +
6000 steps), so it must finish regardless of v14's outcome.

Decision triggers (6000-step schedule):
- step 500: evidence_lift > +0.5 (warmup should produce specialisation)
- step 2000: alpha_mem_mean > 0.02 (frozen backbone can't re-collapse
  the router), PA-EVAL CB Δsh-m > +0.02
- step 6000 (final): PA-EVAL CB Δsh-m > +0.05, evidence_lift > +0.3
  sustained over last 1k steps
- KILL @ 3000 if alpha_mem_mean < 0.005 for 3 consecutive evals
  (frozen backbone still doesn't rescue routing → v14 is the only
  path forward).

#### v14a — v14 headline candidate (queued, launches after v13q)

Script: `scripts/train_v14a_ap_mega_fullfix_gh200.sh`

Launch ETA: ~02:00 UTC-5 (2026-05-03), after v13q completes.
Runtime estimate: ~6.5 h (8000 steps × ~2.9 s/step on mega).

Stack summary:
- R: attention_parity (exp1 prior unchanged)
- O: writer_warmup 500 + anneal 200 + **InfoNCE 0.5 callback-only**
  + **alpha_floor 0.01 @ target 0.05**
- S: orthogonal queries + slot_positional (v13 wins preserved)
- W: slot_attention (iter=3) + **judge QK-LayerNorm**
- update: gated (per-slot sigmoid keep-gate; v13r D3-MC 0.028
  Δ_step indicated competitive mode under-wrote on natural prose)
- backbone: unfrozen after warmup; lr_backbone=2e-5

Decision triggers (8000-step schedule):
- step 700 (anneal end): grad_norm (clipped) < 2, alpha_mem_mean
  > 0.04 (above floor — if floor isn't holding, aux weight too
  small), evidence_lift > 0
- step 2000: alpha_mem_mean > 0.05 sustained, judge
  row_entropy/log(2K) < 0.92 (QK-LN working), InfoNCE gap > +0.05
- step 4000: PA-EVAL CB Δsh-m > +0.030, evidence_lift > +0.10
- step 8000 (final): **PA-EVAL CB Δsh-m > +0.080** (paper-publishable),
  alpha_mem_mean > 0.05 sustained over last 2k, D2-JUDGE norm <
  0.92 sustained
- KILL @ 4000 if PA-EVAL CB Δsh-m < 0.010 for 3 consecutive evals
  (the four fixes together can't rescue the mega joint regime;
  v14b = v14 fixes on top of a frozen-backbone baseline becomes
  the fallback)

---



Each entry: status, run name, machine, exact launch command (or
script reference), key knobs, last logged step / `Δ_sh-m`, log path,
notes.

For the historical v2/v3/v4/v5/v6 ledger (all KILLED or superseded)
and the v6 → v7 transition narrative, see
[`COMPREHENSIVE.md`](../../COMPREHENSIVE.md) Part IV.

> **Path note (2026-04-30 repo cleanup).** `Runs/` was renamed to
> `Runs/` and `Scripts/` to `Scripts/`. Only three runs are still on
> disk: `Runs/chain_v11_evidence_aware_local/` (active),
> `Runs/chain_v2_phaseA_softparity_b4/` (Exp-1 cited), and
> `Runs/run3_qwen3-0.6b-large/` (Exp-1 pair trainer). All v3-v10 run
> directories and v6-v10 launcher scripts were pruned; references
> below that resolve to now-missing paths are preserved as the
> historical audit trail. The entries' configs remain reproducible
> from the flag lists written inline in each section and from git
> history.

---

## v13 campaign — writer warmup + symmetry break (2026-05-01 ~19:20 UTC-5)

### CRITICAL BUGFIX 2026-05-01 ~22:45 UTC-5 — config override silently dropped

While inspecting the first v13 cells (v13c finished, v13a in-flight) we
noticed `ROUTE @ step N: mode=attention_parity` in the v13a log despite
the launch script passing `--memres_mode simple_gate`.  Root cause:
`_build_model` in `train_chain.py` was detecting "is this a memres
checkpoint?" with a try/except around `Qwen3MemResConfig.from_pretrained`.
Because `Qwen3MemResConfig` is a subclass of `Qwen3Config`, it loads
fine from a plain Qwen3 config.json and fills `memres_*` fields from
its own `__init__` defaults — the try never throws.  The subsequent
`overridable` merge then DROPPED CLI overrides for every architecture-
shape flag not in the explicit allow-list:
```
memres_mode, memres_writer_kind, memres_slot_positional,
memres_update_mode, memres_extract_source, memres_num_vectors,
memres_extraction_depth, memres_num_blocks, memres_slot_attention_iters
```
Net effect on every v13 cell so far:
- `--memres_mode simple_gate` → silently ran `attention_parity`
- `--memres_writer_kind slot_attention` → silently ran `original` (judge)
- `--memres_slot_positional` → silently ignored (no M_{in,judge}_pos)
- `--memres_extraction_depth 4` → silently ran L_E=0

That is, **v13c and v13a(buggy)** were actually just
`AP + writer_warmup + orthogonal-init` (no slot positional, no slot
attention, L_E=0) — NOT the full v13 stack we thought we were testing.
Explains why v13c's evidence_lift peaked at +0.22 during anneal and
collapsed post-warmup: it was running basically v11g + writer_warmup.

Fix: detect `from_memres_ckpt` by `base_cfg.model_type == "qwen3_memres"`
(from the raw config.json), not by from_pretrained success.  A plain
Qwen3 base now correctly falls through to the full merge, so every
`--memres_*` flag takes effect.

Commit: `src/train_chain.py::_build_model` (BUGFIX 2026-05-01).

### Effect on post-fix v13a restart (same script, clean relaunch)

|                       | v13a (buggy)           | v13a (fixed)         |
|-----------------------|-----------------------:|---------------------:|
| `mode` (actual)       | attention_parity       | **simple_gate**      |
| `writer_kind`         | original judge         | **slot_attention**   |
| `slot_positional`     | False                  | **True**             |
| L_E                   | 0                      | **4**                |
| trainable memres      | 9.76M                  | **28.91M**           |
| loss @ step 20        | 17.07                  | **6.53**             |
| loss @ step 80        | 13.25                  | **3.72**             |
| `gate_mean` @ phase 1 | 0.0000 (bug)           | **+0.4824**          |
| `memres_gate` grad    | 0.00e+00 (starved)     | 1.39e+13 (training)  |
| `write_gate` grad     | n/a                    | 2.21e+14 (training)  |

By step 80 of the fixed v13a the training loss on a FROZEN backbone
(phase 1) is already at 3.72 — within ~0.4 nats of the no-memory
floor — because the writer's gradient is now actually flowing into
`M_in`, `M_judge`, slot-attention `write_gate`, AND `memory_gate`.
The force-open `gate_mean = 0.482 = 0.5·tanh(2)` matches the
`_set_mem_bias` simple_gate branch spec exactly.

### Results affected (partial v13 stack only, NOT full v13)
- `chain_v13c_d4_ap_gh200` — completed 4000 steps, exit 0 but
  config was `AP + warmup + orth + no-pos + original-writer`,
  peak evidence_lift +0.22 at step 600 (warmup anneal), collapsed
  post-warmup (+0.0037 at step 4000, `D2 judge entropy=0.999`).
  Useful data on "writer_warmup alone can briefly produce
  content-specific memory but the collapse reasserts itself
  once the system is joint-trained".  NOT a valid test of the
  headline v13 stack.

### v13 campaign REORDER (2026-05-01 ~23:20 UTC-5) — AP-first per v3 pair-recipe evidence

Post-fix re-check of the routing prior (user-flagged 2026-05-01 ~23:15):
the **v3 pair-recipe manuscript** (`results/exp1_pair_recipe/manuscript.tex`
Table 2, head-to-head PG-19 trajectory, matched seed, matched compute)
showed `attention_parity (soft ±4 bias)` consistently beating
`simple_gate` by 1.6-3.8× on Δ_sh-m at every step, with AP @ step 2000
(+0.0272) already surpassing SG's full-budget asymptote at step 5200
(+0.0249).  *"The routing-pool variant achieves the scalar-gate's
asymptote in 2/5 the steps and is still improving when stopped."*

This is the **strongest routing-side prior we have from the paper
line-of-evidence**, and it's on the pair trainer where no collapse
cycle exists.  On the chain trainer AP has consistently collapsed
(v5 softparity, v7 all-AP, v8/9/11 all-AP, v12a) because once the
router closes early the writer is gradient-starved, feeds noise back
to the router, and the cycle reasserts.  That's the collapse mechanism
the v13 O/S/W levers are specifically designed to break.

The **full v13 AP stack has literally never been tested** -- v13c was
supposed to but the `_build_model` config-merge bug silently disabled
`slot_attention`, `slot_positional`, and L_E=4.  v13c actually ran as
"AP + writer_warmup + orthogonal init only" and showed the canonical
pattern: +0.22 peak at step 600 during anneal, then collapse to
+0.0037 by step 4000.

Decision: reorder the queue so the strongest hypothesis (AP + full
v13 stack) runs first.

### GH200 queue (2026-05-01 ~23:55 UTC-5, post-v13c2-early-signal)

v13c2 early diagnostics (step 400, mid-warmup): **evidence_lift =
+1.4085** — 6.4× any prior measurement in the project.  Phase 2
transition at step 500 produced gradient shock (grad_norm 6.5e8
pre-clip) and evidence_lift collapsed to -0.56 at step 600.  Decision:
scale warmup + anneal to let the writer reach a stable basin BEFORE
backbone unfreezes.  Added v13r as the headline candidate; v13c2
allowed to finish as "short-warmup baseline" data point.

| # | Run | State | Routing | Warmup+Anneal | Backbone | Corpus | Steps | Tests |
|---|---|---|---|---|---|---|---|---|
| 0 | v13c2 | **RUNNING** (step ~780) | attention_parity | 500+200 | trained (phase 2) | D4 | 4000 | short-warmup baseline; expected to show the phase-transition shock pattern for paper's "why warmup must scale" argument |
| 1 | **v13r** | **QUEUED NEXT** | attention_parity | **3000+1000** | trained (phase 2) | **mega (67k chains)** | **16000** | **HEADLINE CANDIDATE.** Scales warmup 6×, anneal 5× vs v13c2; 9.1 tokens/writer-param total |
| 2 | v13q | QUEUED | attention_parity | 600+300 | FROZEN | mega | 6000 | frozen-backbone regime ablation on mega |
| 3 | v13b | QUEUED | simple_gate | 500+200 | FROZEN | D4 | 4000 | SG ablation vs v13r's AP; stacks on v12d_frozen's +0.03 |
| 4 | v13d | QUEUED | simple_gate | none | trained | D4 | 4000 | no-warmup ablation |
| — | v13p | **DEPRECATED by v13r** | — | — | — | — | — | v13r is the headline; v13p would have been duplicative with different warmup |

#### chain_v13r_ap_mega_longwarmup_gh200 — QUEUED (next after v13c2)
- Script: `scripts/train_v13r_ap_mega_longwarmup_gh200.sh`
- Machine: GH200 GPU 0 (ETA start: 2026-05-02 ~02:00 UTC, after v13c2 completes)
- Runtime estimate: **~17 hours** (16000 steps × 3.8 sec/step on
  mega AP-full-v13 throughput; measured from v13c2 baseline 2.9
  sec/step on D4 window_k=3, scaled 1.3× for window_k=4 + longer
  mega chains).
- Step-budget rationale:
  * 3000 warmup: writer sees **49M tokens on frozen backbone**,
    1.7 tokens/writer-param during phase 1 alone (v13c2 saw 2.5M
    tokens = 0.086 tok/param before phase transition).  Motivated by
    user theory 2026-05-01 ~23:52: *"3000 steps of heating up and
    1000 steps of annealing and finish training with the full corpus."*
  * 1000 anneal: 5× v13c2's anneal to smooth the phase transition
    that produced grad_norm 6.5e8 at step 500→600 boundary.
  * 12000 post-anneal joint training: matches v13p's original
    paper-headline budget; 9.1 total tokens/writer-param
    (v13c2: 2.2, Chinchilla: ~20, v13p-as-queued: 9.1).
- Full v13 stack: AP routing + writer_warmup + orthogonal init +
  slot_positional + slot_attention writer + curriculum evidence/
  competition biases + burn_in_max=12 + carry_state + mega corpus.
- Decision triggers (16000-step schedule):
  * step 3000 (warmup end):   evidence_lift > +1.5 (beat v13c2's
    mid-warmup +1.4 on bigger corpus), alpha_mem_max > 0.2,
    judge row_entropy / log(2K) < 0.95
  * step 4000 (anneal end):   clipped grad_norm < 2, alpha_mem_max
    > 0.1, evidence_lift > +0.5 (at most 50% drop vs phase-1 peak)
  * step 6000:                evidence_lift > +0.8 (recovered past
    phase-2 shock)
  * step 10000:               pa_cb_dsh > +0.050 (PAPER HEADLINE
    THRESHOLD)
  * step 16000 (final):       pa_cb_dsh > +0.100, evidence_lift
    > +1.0 sustained over last 2k steps
  * KILL @ 4000 if evidence_lift < 0 for 3 consecutive evals (phase
    transition destroyed the content-specific memory permanently).
- If v13r clears the step-10000 trigger this is the paper's
  headline run; the v13p script is retired and v13b becomes the
  SG-ablation for the paper.

v13a (SG + trained) and the original v13c (partial-v13 AP) are both
MOVED to `cancelled/` / `done/` respectively and no longer on the
active path.

#### chain_v13c2_d4_ap_full_gh200 — TRAINING (2026-05-01 ~23:23 UTC-5)
- Script: `scripts/train_v13c2_d4_ap_full_gh200.sh`
- Machine: GH200 GPU 0, tmux `cwd-chain_v13c2_d4_ap_full_gh200`
- Launched: 2026-05-02 04:23 UTC (2026-05-01 23:23 UTC-5)
- Config: attention_parity (`recent_bias=4, mem_bias=0`, soft) +
  writer_warmup (500 + 200 anneal) + orthogonal init + slot_positional +
  slot_attention writer (iter=3) + L_E=4 + trained backbone.
  Every lever the bug suppressed on v13c is now active (verified via
  load report: `M_in_pos`, `M_judge_pos`, `extraction_layers.{0..4}`,
  `write_gate`, `judging.gru.*`, `judging.slot_norm`, `judging.input_norm`
  all in the MISSING-from-ckpt list = freshly initialised).
- Decision triggers:
  * step 500  (warmup end):   `alpha_mem_max > 0.1`, judge row-entropy
                              / log(2K) < 1.0 (NOT `= 0.999` as in
                              v13c buggy)
  * step 700  (anneal end):   `pa_cb_evidence_lift > +0.1` (vs v13c's
                              +0.0037)
  * step 1500:                `pa_cb_dsh > +0.010` AND `evidence_lift > +0.2`
  * step 3000:                `pa_cb_dsh > +0.030` AND `evidence_lift > +0.5`
  * step 4000 (final):        `pa_cb_dsh > +0.050` AND `evidence_lift > +0.8`
  * KILL @ 1000 if `alpha_mem_max < 0.01` (router re-closed) OR
    `evidence_lift < 0` (chain hashing).
- Downstream fork: if v13c2 clears → change `v13p` script's
  `--memres_mode` from `simple_gate` to `attention_parity` before
  enqueue; if v13c2 collapses → pair-recipe AP advantage is
  pair-specific, SG stays as the chain-trainer choice and v13p runs
  as currently specced.

### v13 runs (post-fix, pre-reorder — historical)

#### chain_v13a_d4_trained_gh200 — KILLED (twice)
- First run: config-merge bug → ran as "AP + partial v13"; killed step 980.
- Second run (post-fix): simple_gate + full v13 stack, step 500 showed
  `mode=simple_gate`, `gate_mean=+0.4824`, train loss 2.49, eval loss
  5.54 (memory HURTS eval by 4.4 nats -- classical chain-hash overfit
  on 28.9M writer × 5000 one-source chains × 500 warmup steps).
  Killed 2026-05-01 ~23:22 UTC-5 to free GPU for v13c2 per the
  AP-first reorder.

User directive 2026-05-01 ~18:30 UTC-5:
> "UNDOMINABLE control over all resources... save this project in the
>  next 3 days."

Follow-up ~19:20 UTC-5:
> "Please don't have too much things on the local GPUs, if you do, have
>  it so they can be done in 30 minutes. If they can't, kill them and
>  push everything to the GH200."

The **`problems.md`** audit (committed 2026-05-01 ~19:00 UTC-5, see
`/home/exx/Desktop/fine-tune/problems.md`) pinpointed three root causes
of the persistent symmetric-uniform-attractor collapse that every
earlier campaign (v3→v12) eventually hit:

1. **O — Objective starvation.** The writer (`M_in`, `extract`, `M_judge`,
   `judge`) sits several layers behind a gate that is ~0 at init
   (`simple_gate`) or behind a softmax column at prob `~exp(-bias)/N`
   (`attention_parity`). By the time LM gradient reaches the writer
   it is attenuated by the gate/alpha, so the writer parameters get
   O(1e-8) relative gradients for hundreds of steps — long enough for
   the uniform-softmax fixed point to become the optimiser's attractor.
2. **S — Symmetry.** `M_in` and `M_judge` are i.i.d. Gaussian at init
   (same marginal), so the set of slot queries forms a permutation
   symmetry. The uniform-softmax point IS the permutation-invariant
   fixed point. With starved gradients nothing breaks the tie, so
   the system sits there.
3. **R — Routing.** In `attention_parity`, m^t competes directly
   against the backbone's own (trained, mature) partial sums inside
   the depth softmax. A nascent noisy m^t structurally loses.

v13 attacks all three at once:
- **O.** New `writer_warmup` phase (500 steps) freezes backbone +
  embed + LM head, forces `mem_bias = 4` (`_set_mem_bias`) AND
  `memory_gate.gate ≈ 0.48` (simple_gate fix — see commit trail in
  `train_chain.py::_set_mem_bias`), and trains the entire memres
  subsystem (M_in, extract, M_judge, judge, readout, gate, router)
  directly against the LM objective. At step 500 the bias anneals
  to its configured init over 200 steps, then phase 2 is joint.
- **S.** `memres_queries_init=orthogonal` uses `nn.init.orthogonal_`
  on `M_in` / `M_judge` so the slot queries are an orthonormal basis
  from step 0 (BF16 is upcast to FP32 for the QR op, then cast back).
  `memres_slot_positional=True` adds a deterministic Fourier pattern
  as a learnable per-slot positional offset added to `q_seed` before
  expansion, giving every slot a unique identity that the optimiser
  cannot permute away.
- **R.** `--memres_mode simple_gate` takes m^t OUT of the depth
  softmax entirely; the backbone residual stream is untouched and
  m^t is added via the per-sublayer gate. This is the mode v9c used
  successfully before the spec-strict v10 migration.

Init-parity is preserved (`max|Δlogits| = 0` vs bare backbone for
both `simple_gate` and `attention_parity` with orth+pos init), so
the architectural changes do NOT create any warmup discontinuity.

### v13 run plan

| Run | Loc | Stack | Tests |
|---|---|---|---|
| v13c | GH200 GPU 0 (RUNNING) | AP + warmup + orth+pos + slot on D4 | "does warmup alone rescue AP, so spec Eq. 9 can stand?" |
| v13a | GH200 (QUEUED, next) | **HEADLINE**: SG + warmup + orth+pos + slot, trained backbone on D4 | "does the full v13 stack break the uniform attractor with a trainable backbone?" |
| v13b | GH200 (QUEUED) | SG + warmup + orth+pos + slot, **FROZEN** backbone on D4 | builds on v12d_d4_frozen's +0.03 evidence_lift — does v13 amplify it? |
| v13q | GH200 (QUEUED) | **AP + FROZEN + curriculum + mega corpus + 6k steps** | user-requested: composes four levers the other D4 cells don't reach; tests whether spec-strict Eq. 9 routing survives at scale |
| v13d | GH200 (QUEUED) | SG + orth+pos + slot, NO warmup on D4 | ablation: is warmup load-bearing, or do orth+pos+simple_gate alone suffice? |
| v13p | GH200 (HELD) | winning v13 config × LME mega × 16k steps | HEADLINE paper run, held until a v13 cell clears the decision triggers |

**Local H100s idle** per user directive — 4000 steps on D4 takes
~2-3 h per run, far exceeds the 30-min local-GPU cap. Everything
above 30 min now runs on GH200.

### v13 runs

#### chain_v13a_d4_trained_local — KILLED (superseded by chain_v13a_d4_trained_gh200)
- Script: `Scripts/train_v13a_d4_trained_local.sh`
- Machine: local H100 GPU 1
- Launched: 2026-05-01 ~19:15 UTC-5
- Killed: 2026-05-01 ~19:22 UTC-5 at step ~60
- Reason: 30-min local-GPU cap (user directive)
- Last signal: loss 16-17 (= `log(256)×callback_weight(3) ≈ 16.65`,
  expected at init since memory contribution = `gate × m^t = 0 × m^t = 0`);
  writer gradient order of magnitude normal (`|g_M_in| ~ 1e-4`);
  gate_mean 0 throughout — which revealed a **bug**: in `simple_gate`
  mode the router's `mem_bias` doesn't control the forward path, so
  forcing `mem_bias = 4` had no effect on memory injection. Fixed in
  `_set_mem_bias` (now also sets `memory_gate.gate = 0.5·tanh(bias/2)
  ≈ 0.48` for `simple_gate`).

#### chain_v13b_d4_frozen_local — KILLED (superseded by chain_v13b_d4_frozen_gh200)
- Script: `Scripts/train_v13b_d4_frozen_local.sh`
- Machine: local H100 GPU 0
- Launched: 2026-05-01 ~19:15 UTC-5
- Killed: 2026-05-01 ~19:22 UTC-5 at step ~60
- Reason: 30-min local-GPU cap (user directive)

#### chain_v13c_d4_ap_gh200 — TRAINING
- Script: `scripts/train_v13c_d4_ap_gh200.sh` (GH200 flat layout)
- Machine: GH200 GPU 0, tmux `cwd-chain_v13c_d4_ap_gh200`
- Launched: 2026-05-01 ~19:19 UTC-5
- Config: AP routing (`attention_parity`) + writer_warmup (bias=4) +
  orth+pos init + slot_attention writer + trained backbone on D4.
- Status @ 2026-05-01 ~19:27 UTC-5, step 220:
  loss 16.8 (= callback-weighted random-token baseline; expected
  during phase 1 because forced `mem_bias = 4` injects ~50% random
  m^t into the Eq. 9 softmax), writer gradients active
  (`|g_M_in| ≈ 5.5e-5`, `|g_extract| ≈ 6.0e+1`), `phase_aligned`
  metric improved enough at step 200 to trigger best-ckpt save
  → `output/chain_v13c_d4_ap_gh200/best/` on GH200. First ever
  "best" save within the first 200 steps on D4 for a memres run.
- Decision point: step 500 (warmup end) — if PA-eval isn't trending
  toward bare-baseline by step 1000, kill and fall through to v13a.

#### chain_v13a_d4_trained_gh200 — QUEUED (next after v13c)
- Script: `scripts/train_v13a_d4_trained_gh200.sh` (GH200 flat layout)
- Machine: GH200 GPU 0 (via watchdog queue)
- Enqueued: 2026-05-01 ~19:26 UTC-5
- Config: SG routing (`simple_gate`) + writer_warmup (bias=4, force
  `memory_gate=0.48` via the fix) + orth+pos init + slot_attention
  writer + trained backbone on D4.
- Decision triggers:
  * PA-eval NLL below bare baseline within 2000 steps
  * `evidence_lift > 0.1` by step 1500, `> 0.3` by step 3000
  * `gate_mean > 0.2` during phase 1 (force-opened), recovering
    under LM gradient after anneal
  * judge entropy falling from `log(8) ≈ 2.08` toward 1.0 by step 2000

#### chain_v13b_d4_frozen_gh200 — QUEUED (after v13a)
- Script: `scripts/train_v13b_d4_frozen_gh200.sh`
- Machine: GH200 GPU 0
- Enqueued: 2026-05-01 ~19:26 UTC-5
- Config: Same as v13a but `--freeze_backbone --lr_backbone 0` and
  `--writer_warmup_keep_backbone_frozen`. Tests whether v13 stacks
  correctly on the only prior campaign signal (`v12d_d4_frozen` had
  evidence_lift +0.03).
- Decision trigger: must beat v12d_d4_frozen by >3× evidence_lift.

#### chain_v13q_ap_frozen_curriculum_mega_gh200 — QUEUED (after v13b)
- Script: `scripts/train_v13q_ap_frozen_curriculum_mega_gh200.sh`
- Machine: GH200 GPU 0
- Enqueued: 2026-05-01 ~19:30 UTC-5 (user-requested, composes the
  four levers frozen-backbone × AP × curriculum × mega-corpus).
- Config: `attention_parity` routing (spec-strict Eq. 9) +
  `--freeze_backbone` + curriculum_evidence_bias=1.0 +
  curriculum_competition_bias=1.0 + burn_in=12 +
  `v11_mega_train_s512.pt` (899 MB, LME+MSC+PG19+TV+RealTalk) +
  6000 steps + writer_warmup 600 (anneal 300) + orth+pos +
  slot_attention. LR 1e-4 (2x frozen-backbone boost, same as v13p).
- Fills the gap between v13c (AP on small D4, 4k) and v13p (SG on
  mega, 16k): tests whether spec-compliant Eq. 9 routing survives at
  scale under the full v13 stack, with 3/8 the v13p budget.
- Decision triggers (6000-step AP-frozen schedule):
  * step 600  (end-warmup):  alpha_mem_max > 0.01, judge entropy
    (row) trending from log(2K)≈1.0 down toward 0.8
  * step 1000 (post-anneal): pa_cb_evidence_lift > +0.1 AND
    alpha_mem_max stays > 0.05
  * step 2000: pa_cb_dsh > +0.010, evidence_lift > +0.3
  * step 4000: pa_cb_dsh > +0.030, evidence_lift > +0.8
  * step 6000: pa_cb_dsh > +0.050, evidence_lift > +1.0
  * KILL @ step 1000 if alpha_mem_max < 0.005 (router re-closed
    despite frozen backbone -- AP is structurally unsalvageable).
  * KILL @ step 2000 if evidence_lift < 0 (memory is chain-identity
    hashing, not content).

#### chain_v13d_d4_ablate_warmup_gh200 — QUEUED (last, ablation)
- Script: `scripts/train_v13d_d4_ablate_warmup_gh200.sh`
- Machine: GH200 GPU 0
- Enqueued: 2026-05-01 ~19:19 UTC-5 (timestamp bumped to run after
  v13a/b so headline results land first)
- Config: SG + orth+pos + slot, **no** writer_warmup.
- Purpose: if v13d collapses like v12a but v13a/b clear, we have
  clean evidence that the warmup (O lever) is load-bearing.

#### chain_v13p_lme_mega_gh200 — HELD (spawn on v13 success)
- Script: `scripts/train_v13p_lme_mega_gh200.sh` (synced but not
  enqueued)
- Will be enqueued by hand once a v13 cell clears the decision
  triggers. Scales winning config to v11_mega × 16k steps.

### Local GPU state (2026-05-01 ~19:28 UTC-5)

GPU 0 / 1 / 2 all idle (14-17 MiB each). No training process on
any local GPU. Per user directive, local is reserved for future
≤30-min smoke / diagnostic work only.

---

## v12 campaign — slot-attention writer (2026-05-01 ~19:00 UTC)

User directive 2026-05-01 ~13:30 UTC-5 (~18:30 UTC):
> "Architectural only — replace the Judge with slot-attention (softmax
>  over slots, not over inputs, so slots are forced to specialize) AND
>  move m^t out of the depth softmax into a dedicated memory cross-
>  attention sublayer (so it stops competing with backbone block
>  summaries). Spec-strict. m^t MUST stay as foundational source b_{-1}
>  in the Block-AttnRes pool (Eq. 9). Only judge layer internals are
>  fair game."

Writer overhaul; routing left spec-strict per Eq. 9. Plan
[`memres_v12_slot_judge_a5c5b06d.plan.md`](../../memres_v12_slot_judge_a5c5b06d.plan.md).

### Diagnosis: the original writer is decision-less by construction

The v11 post-mortem ("v11 campaign — first 7 cells finished" §) found
that v11g's `MemoryBlock.judge` layer never makes a judgement — D2 on
`v11g/best` reports row_entropy / log(2K) = **0.999** (uniform), keep_mean
= **0.500** (no preference for prev vs new), eff_rank = **1.02** (all 128
slots collapsed to ~1 direction). v11r/v11q/v11l/v11h all confirmed this
isn't fixed by readout warmup, contrastive supervision, frozen backbone,
or norm-1 readout.

Why this is structural and not a training failure
([`paper_artifacts/eval/diag_v11g_best.json`](../../paper_artifacts/eval/diag_v11g_best.json)):

The original judge is a single softmax over the inputs axis:
```
attn[b, k, j] = softmax_j(M_judge[k] · W_K(P[j]) / sqrt(d))   for j in 0..2K-1
M_c[b, k]     = sum_j attn[b, k, j] · W_V(P[j])
```
At init, all M_judge rows are i.i.d. random, all P rows are i.i.d.
random, so `attn[b, k, j]` is approximately uniform `1/(2K)` — every
slot averages over all inputs and produces the same content-blind
output. The **symmetric uniform fixed point** is also a *gradient*
fixed point because permuting any two slots leaves both forward and
loss unchanged. There is no architectural pressure for slots to
specialise on disjoint content; only training noise can break the
symmetry, and the v11 audit showed that signal is too weak (writer
gradient ratio ~10^-5 of backbone) to do so over 4000 steps.

### Architectural change: Slot-Attention writer (Locatello 2020)

`SlotAttentionWriter` ([`src/modeling_memres.py`](../../src/modeling_memres.py)
lines 460–620) replaces `MemoryBlock.judge` (and optionally
`MemoryBlock.extract`) with the canonical Slot Attention update rule:
```
slots^(0) = M_judge.broadcast(B)
P         = [M_c^{t-1} || M_new]                 # (B, 2K, d)
for t in 1..T:                                    # T=3 iterations
    q          = W_Q(slot_norm(slots))            # (B, K, d)
    k          = W_K(input_norm(P))               # (B, 2K, d)
    v          = W_V(input_norm(P))               # (B, 2K, d)
    attn       = softmax(q kᵀ / sqrt(d), dim=-2)  # SOFTMAX OVER SLOTS
    attn       = attn / attn.sum(dim=-1)          # weighted-mean per slot
    updates    = attn @ v                         # (B, K, d)
    slots      = GRUCell(updates, slots)          # per-slot recurrent gate
return slots
```

Two changes vs the original judge that matter:

1. **Softmax over the slots axis (not inputs).** `softmax(scores, dim=-2)`
   means each input column j has exactly one unit of mass distributed
   across slots; if two slots have similar query projections they
   share that mass, but if either slot's projection drifts the other
   loses mass. This is the architectural pressure for slots to
   tile the input pool into K disjoint factors. Five years of
   object-centric / set-prediction literature have validated this
   primitive on exactly this problem.
2. **GRUCell per slot.** The keep-vs-write decision is made by a
   sigmoid-gated recurrent update on each slot independently rather
   than by a single softmax over [prev || new]. This is also what
   subsumes the Eq. 3-4 residual extraction stack when
   `--memres_writer_kind slot_attention_full`.

The PDF's "zero-sum semantic competition" claim (Section 2.1) is now
realised at the architectural level rather than by hopeful gradient.

### Spec adherence

Per the user directive, m^t **stays** as foundational source b_{-1}
in the Block-AttnRes pool (Eq. 9). The depth-softmax routing through
`BlockAttnResRouter` is unchanged. Only writer internals are fair game.
Init parity with the bare backbone is preserved at the gate / router
level (memory_gate=0 in simple_gate; alpha_mem ~ exp(-32)/N in
attention_parity); see [`results/eval/init_parity_test_v12.json`](../eval/init_parity_test_v12.json)
— **all 10 cases match expectation, 4 new slot-attention cases pass at
max|Δ| = 0.000e+00** (bit-exact equality).

### CLI surface

Single new flag (`src/train_chain.py`):
```
--memres_writer_kind {original,slot_attention,slot_attention_full}   default: original
--memres_slot_attention_iters INT                                    default: 3
```
The flag threads through `Qwen3MemResConfig` and `MemoryBlock.__init__`
(see [`src/modeling_memres.py`](../../src/modeling_memres.py) lines 250-310,
540-680). Architecture-shape; not overridable on warm-start from a
checkpoint with a different writer_kind.

### Validation sequence (D4 → LME → headline)

The v11r/v11q post-mortem established that LongMemEval-S corpora are
too noisy to differentiate architectural variants on the diagnostic
metrics. v12 validates on D4 first, then scales the winning architecture
onto LME with the v11g recipe, then composes with the v11p (frozen +
Chinchilla + mega) recipe for the headline.

| Cell      | Corpus         | Backbone | Writer kind          | Where    | Status |
| --------- | -------------- | -------- | -------------------- | -------- | ------ |
| `v12a`    | D4 synth       | trained  | slot_attention       | local 1  | RUNNING (step ~280, signals strong) |
| `v12b`    | D4 synth       | trained  | slot_attention_full  | local 1  | queued, conditional on v12a partial |
| `v12c`    | v11_lme_msc    | trained  | (winner)             | GH200    | script + queue ready |
| `v12d`    | v11_mega       | FROZEN   | slot_attention       | GH200    | enqueued behind v11m, v11p |

### Decision triggers (sharp, falsifiable)

**v12a-slot-judge-D4** (local GPU 1, started 2026-05-01 18:50 UTC):

| Step  | Pass criterion                                                                                  | KILL trigger                                                              |
| ----- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
|  200  | pa_cb_dnm > +0.5 AND D2 row_entropy_norm < 0.95                                                  | —                                                                         |
| 1000  | pa_cb_dnm > +2.0 AND pa_cb_evidence_lift > +1.0 AND judge_keep_mean ≠ 0.500 ± 0.05               | row_entropy_norm > 0.98 → slot attention also stays uniform; pivot to ground-truth-first scope |
| 2000  | pa_cb_ce_mem < 2.0 nats (~35% of log(256)=5.55)                                                  | —                                                                         |

**v12d-headline-frozen-chinchilla-mega** (GH200, queued):

Mirrors v11p decision triggers exactly so the architectural delta is
the only causal axis:
- step 1500: alpha_mem_max > 0 AND ‖m^t‖/‖embed‖ ∈ [0.3, 50]
- step 5000: alpha_mem_max > 1e-2 AND pa_cb_dnm > +0.05
- step 12500: pa_cb_dsh > +0.020 AND alpha_mem_max NOT decaying
- step 25000: standard Δ_sh-m > +0.010 (vs v11g step 1300 best of +0.0024)
- KILL: step 5000 with pa_cb_dsh < +0.005 → architecture + frozen +
  Chinchilla + mega together cannot move the headline metric → v13
  must consider routing intervention (m^t out of depth softmax,
  out-of-spec).

### Earliest evidence (v12a step 200, 2026-05-01 18:55 UTC)

[`logs/chain_v12a_slot_judge_d4_local.log`](../../logs/chain_v12a_slot_judge_d4_local.log) (v12a, local):

```
EVAL @ step 200: n=256 mem=2.7892 nomem=2.2461 shuffle=2.7906 oracle=3.2793
                 Δnm-m=-0.5431 Δsh-m=+0.0014 Δor-m=+0.4901
PA-EVAL @ step 200: n=64 cb_n=64
                    WS  Δnm-m=+0.9146 Δsh-m=-0.0032
                  | CB  Δnm-m=+0.3857 Δsh-m=+0.0205
EVID-EVAL @ step 200: n_ev=64 pa_cb_ce_mem=4.3274 pa_cb_ce_mem_floor=4.3247
                      Δnm-m_floor=+0.3885 evidence_lift=-0.0027
ROUTE @ step 200: mode=attention_parity α_mem_max=0.0201 α_mem_mean=0.0149
                  frac_open=0.00 top=[l12=0.0201, l13=0.0200, l20=0.0199]
READOUT @ step 200: ‖m^t‖ / ‖embed‖ mean=3.867 max=3.873
```

For comparison, v11g's BEST eval over 4000 steps was PA CB Δnm-m ≈
+0.027 at step ~1000 with α_mem_max ≈ 0.011 peaking at deep blocks
(l54, l53). v12a at step **200**:
- PA CB Δnm-m = **+0.3857** (~14x v11g best, after 0.05x the steps)
- PA CB Δsh-m = **+0.0205** (vs v11g ≈ 0.000 — *first time* the writer
  shows content-aware memory over shuffled memory)
- α_mem_max = **0.0201** (~2x v11g peak), peaks at l12/l13/l20
  (shallow/mid blocks, where memory readouts can bias contextual
  representations)
- WS Δnm-m = **+0.9146** (window-start sessions also benefit by
  ~0.9 nats — memory is helping early-session content too)

Caveats: D4 is synthetic, training-loss going down quickly is partly
template fitting; evidence_lift = -0.003 at step 200 is below noise so
no strong claim there yet. Step 1000 will tell us whether the lift
holds and whether judge_keep_mean has actually moved off 0.500.

### Parallel-running v11l-fix (LOCAL GPU 0, AP + frozen backbone)

While v12 was being implemented, v11l-fix relaunched the failed v11l
(`scripts/train_v11l_ap_frozen_backbone_gh200.sh` exited with the
GH200-flat-vs-local-src/ path bug; fixed locally as
`Scripts/train_v11l_ap_frozen_backbone_local.sh`). Closes the v11
audit's last untested cell (b: backbone co-evolution).

Step 600 readings (via
[`logs/chain_v11l_ap_frozen_backbone_local.log`](../../logs/chain_v11l_ap_frozen_backbone_local.log)):

```
PA-EVAL @ step 600: n=50 cb_n=50  WS Δnm-m=+0.0002 Δsh-m=-0.0001
                                | CB Δnm-m=-0.0023 Δsh-m=-0.0022
ROUTE @ step 600: alpha_mem_max=0.0005 alpha_mem_mean=0.0001 frac_open=0.00
                  top=[l35=0.0005, l43=0.0004, l41=0.0004]
```

Verdict at step 600: **(b) is REJECTED**. With the backbone frozen
(zero co-evolution by construction), α_mem_max collapses to 5e-4 and
PA CB Δsh-m goes negative. The decay cannot be backbone block
summaries crowding out memory in the depth softmax — block summaries
are stationary and memory still loses. This means the writer subsystem
itself is the bottleneck on LME with the original architecture, which
is exactly what v12 targets.

The v11 campaign's three orthogonal hypotheses are now:
- structural (decision-less judge) → confirmed (D2)
- gradient (backbone co-evolution) → **REJECTED (v11l-fix)**
- data (LME sparsity / chain confound) → confirmed (v11r/q)

So v12d (frozen + Chinchilla + mega + slot_attention) composes the two
remaining fixes (architectural + data) — gradient flow through the
backbone is no longer assumed to be the missing piece.

### Files modified for v12

- [`src/modeling_memres.py`](../../src/modeling_memres.py): added
  `SlotAttentionWriter` class; `MemoryBlock` accepts `writer_kind`,
  `slot_attention_iters` parameters; `_init_memres_params` calls
  `gru.reset_parameters()` for new module (HF generic init misses
  GRUCell — silent NaN until this fix); `Qwen3MemResConfig` exposes
  `memres_writer_kind`, `memres_slot_attention_iters`.
- [`src/train_chain.py`](../../src/train_chain.py): `--memres_writer_kind`
  and `--memres_slot_attention_iters` CLI flags wired through to config.
- [`tools/init_parity_test.py`](../../tools/init_parity_test.py): adds
  4 slot-attention parity cases; all pass max|Δ| = 0.000e+00.
- [`Scripts/train_v11l_ap_frozen_backbone_local.sh`](../../Scripts/train_v11l_ap_frozen_backbone_local.sh) (new local fix).
- [`Scripts/train_v12a_slot_judge_d4_local.sh`](../../Scripts/train_v12a_slot_judge_d4_local.sh) (new).
- [`Scripts/train_v12b_slot_extract_d4.sh`](../../Scripts/train_v12b_slot_extract_d4.sh) (new, conditional).
- [`Scripts/train_v12c_slot_writer_lme_gh200.sh`](../../Scripts/train_v12c_slot_writer_lme_gh200.sh) (new).
- [`Scripts/train_v12d_headline_frozen_chinchilla_mega_gh200.sh`](../../Scripts/train_v12d_headline_frozen_chinchilla_mega_gh200.sh) (new, queued).
- GH200 in-place: fixed v11m/v11p scripts (same `src/` path bug as
  v11l); synced modeling_memres.py + train_chain.py to GH200 root;
  enqueued v12d_slot_attention behind v11m + v11p.

### v12a grow-then-decay reproduces the v11g failure mode (step 800)

User directive 2026-05-01 ~14:35 UTC-5:
> "investigate the newest step of v12. I'm thinking a frozen vs non
>  frozen comparison study would be good. remove the v11p. I have lost
>  faith in it vs the slot idea."

**v12a-slot-judge-D4 trajectory through step 800:**

| step | α_mem_max | PA CB Δ_nm-m | PA CB Δ_sh-m | EVID Δ_nm-m_floor | evidence_lift | D2 row_ent_norm | D2 keep_mean | D2 eff_rank |
| ---- | --------- | ------------ | ------------ | ----------------- | ------------- | --------------- | ------------ | ----------- |
| 200  | 0.020     | **+0.386**   | **+0.021**   | +0.388            | -0.003        | 0.998           | 0.500        | 1.02        |
| 400  | 0.029     | +0.023       | +0.003       | +0.033            | -0.010        | 0.999           | 0.500        | 1.01        |
| 600  | 0.033     | -0.046       | -0.008       | -0.041            | -0.004        | 0.999           | 0.500        | 1.01        |
| 800  | 0.045     | -0.003       | -0.000       | +0.005            | -0.008        | 0.999           | 0.500        | 1.01        |

**The slot-attention writer also collapses to the symmetric uniform
fixed point.** D2 is decisive here: row_entropy_norm climbs from 0.998
back to 0.999 and eff_rank settles at 1.01, which means all 128 slots
are projecting to ~1 direction — exactly the same diagnostic
signature as v11g/v11h/v11l on the original judge. The PA CB Δ_nm-m
=+0.386 at step 200 was a real but transient signal: the slot-axis
softmax provides symmetry breaking *at init* but the GRUCell uses
shared weights across slots, so as training progresses the slot states
drift toward each other and the softmax returns to ~1/K (uniform across
slots), at which point the per-slot weighted-mean normalisation
produces uniform across inputs and we are back at the same fixed point.

α_mem_max keeps growing (0.020 → 0.045 over 4 evals), so the depth
router is recruiting more memory mass. The mass it recruits is
content-blind, so the model is at best neutral and at step 600
*actively hurt* (PA CB Δ_nm-m = -0.046) by reading from M_c.

D3-MC says M_c IS chain-specific (pair=0.015, near-orthogonal across
chains) and dynamic (Δ_step=1.55). So the writer is producing
*different* outputs per chain, but those outputs are not in
content-relevant directions — same v11r/q "chain-identity hash"
failure mode in disguise.

This is an honest correction to the earlier v12 narrative. The
single-knob slot-attention judge is necessary but not sufficient. A
deeper architectural fix is on the table for v13 (per-slot GRU
weights, slot-specific bias, or — if the user reverses on spec-
strictness — moving m^t out of the depth softmax). For v12, the
question becomes whether *backbone gradient flow* can break the
symmetry from above.

### v12d retargeted onto D4 (2026-05-01 ~19:50 UTC)

User directive 2026-05-01 ~14:42 UTC-5:
> "and queue a smaller corpus instead of chinchilla level. I was
>  curious about your earlier idea of building a synthetic dataset
>  that is stupid easy to find the answer of cross-session
>  remembrance ... I don't think that custom corpus ever got built.
>  Also, move everything from the H100s to the GH200. Also double
>  check the shell scripts, I won't be visiting the server for a
>  while."

**Surprise: D4 is exactly the "stupid easy" corpus.** Inspecting
`paper_artifacts/chains/synthd4_persona_callback_train_s512.pt`,
chain[0] reads:

```
session 0 (EVIDENCE):
  User: My favorite tool in the world is scissors.
  Assistant: Got it, your favorite tool is scissors. I'll remember that.
session 1: User: I keep losing track of my time. Assistant: Maybe try blocking your calendar more.
session 2: User: My printer is on the fritz again. Assistant: Always at the worst time, isn't it.
... five more distractor sessions ...
session 8 (CALLBACK):
  User: Quick question, what was my favorite tool again?
  Assistant: Your favorite tool is scissors.
```

That is exactly the "day 1 = I like red, day 6 = what's your favorite
color" structure the user described — 5000 chains × 9 sessions, 256-
item closed-set callback (tool, instrument, fruit, ...). Irreducible
callback CE floor under no-memory baseline = log(256) = 5.55 nats.
Ground truth in chain name. Already on GH200 (199 MB train, 20 MB val).

**Mega + chinchilla cancelled, D4 queued in its place.**

The old plan was 25k Chinchilla-budgeted steps on the v11 mega corpus
(~10x D4). Throwing 25k steps at a writer that hits the symmetric
uniform fixed point at step 800 on a 5000-chain *clean* corpus (v12a
finding) is ~50x compute waste. If the slot-attention writer cannot
move pa_cb_ce_mem toward log(256)=5.55 on D4 in 4000 steps, it has no
chance on mega in 25k. D4 is the right falsifiability surface.

| Cell | writer | backbone | corpus | budget | role |
| ---- | ------ | -------- | ------ | ------ | ---- |
| `v12d_d4_frozen`  | slot | frozen          | D4 | 4000 steps | matches v11l-fix regime; writer-only signal |
| `v12d_d4_trained` | slot | trained @ 2e-5  | D4 | 4000 steps | matches v12a regime; tests if LM gradient breaks symmetry |

Single axis: `--freeze_backbone --lr_backbone 0` vs `--lr_backbone 2e-5`.
Memres lr held at 5e-5 across both. 4000 steps × bs 4 × ga 2 × k=3 ≈
~50M tokens per arm, ~2 h wall-clock at 6.4k tok/s. Total study cost
~8 h on GH200 GPU 0 including both D5 epilogues.

**Decision triggers (D4-scaled, both arms identical):**
- step  200: pa_cb_dnm > +0.5 AND D2 row_entropy_norm < 0.95
- step 1000: pa_cb_dnm > +2.0 AND pa_cb_evidence_lift > +1.0
              AND D2 keep_mean ≠ 0.500 ± 0.05 AND D2 eff_rank > 4
- step 2000: pa_cb_ce_mem < 2.0 nats (~35% of log(256))
- step 4000: pa_cb_ce_mem < 1.0 nats (~18% of log(256)) — writer
              works in principle on the cleanest possible corpus
- KILL @ step 1000 in BOTH arms with row_entropy_norm > 0.98:
              v13 architectural pivot (per-slot untied GRU, slot-bias,
              or — if user reverses — out-of-spec routing).

**Migration logistics (H100 → GH200):**
- Killed local `v11l-fix` (PID 10658, step 1320, hypothesis already
  rejected at step 600 — see "v11l-fix verdict" §) and local `v12a`
  (PID 15268, step 860, collapse already documented at step 800).
- Cancelled GH200 queue items: `v12d_*_chinchilla_mega_*` (frozen and
  trained) and the original two D5 epilogues that pointed at the
  chinchilla-mega run names. Preserved in `cancelled/` for audit.
- Re-queued D4 versions in timestamp order:
  ```
  1777664000_chain_v12d_slot_attention_d4_frozen_gh200.json
  1777664100_chain_v12d_slot_attention_d4_trained_gh200.json
  1777664200_v12d_d5_ttt_epilogue_frozen.json     (now points to D4 run names)
  1777664300_v12d_d5_ttt_epilogue_trained.json    (now points to D4 run names)
  ```

**Script audit (since user is hands-off for a while):**
[`tools/audit_v12d_scripts.py`](../../tools/audit_v12d_scripts.py)
verifies on GH200 — every CLI flag in both training scripts and the
D5 epilogue resolves to a real argparse argument; both D4 corpus
files exist and load; `--memres_writer_kind` accepts `original /
slot_attention / slot_attention_full`; train_chain / modeling_memres /
presets all import cleanly under the `~/venv` env; every queued spec
references a script that exists. **6/6 PASS.**

First D4 cell came up at 19:46:36 UTC: step 40, loss 5.59 (already
near the log(256)=5.55 floor, expected — model hasn't learned
*memory* yet, just the callback template), 6.4k tok/s, 596M frozen +
9.8M memres training, no NaN, D1 gradient telemetry active. ETA ~2h
to step 4000.

### v11p cancelled, v12d frozen-vs-trained comparison study queued [SUPERSEDED by D4 retarget above]

User directive 2026-05-01 ~14:35 UTC-5 (same message as above):
> "remove the v11p. I have lost faith in it vs the slot idea."

**v11p killed.** Moved to `paper_tools/cloud_watchdog/cancelled/`.
v11p's role was to be the direct 1-axis counterfactual for v12d
(original-vs-slot writer with everything else fixed). After v12a's
step-800 D2 reading showed the slot writer hits the same fixed point,
that comparison loses its headline status — the "did the slot writer
beat the original at scale?" question is moot if the slot writer
collapses on its own diagnostic corpus.

**v12d redesign: frozen vs trained backbone, both with slot writer.**

The new headline study tests whether backbone gradient flow can break
the slot-attention symmetry by pushing M_c toward content-relevant
directions from the LM-loss side.

| Cell | writer | backbone | corpus | budget     | role |
| ---- | ------ | -------- | ------ | ---------- | ---- |
| `v12d_frozen`  | slot | frozen | mega | chinchilla | matches v11l-fix regime; isolates writer-only signal |
| `v12d_trained` | slot | trained @ 2e-5 | mega | chinchilla | matches v11g regime; tests if LM gradient breaks symmetry |

Single axis difference: `--freeze_backbone --lr_backbone 0` vs
`--lr_backbone 2e-5`. Memres lr held at 5e-5 (v11g-matched) on both
arms — v12d-frozen lost the lr=1e-4 bump from v11p so the comparison
is single-knob clean.

**Hypotheses, sharp:**

- *trained beats frozen on D2 row_entropy_norm and PA CB Δ_sh-m*:
  LM gradient through W_V_read and W_V_judge provides the symmetry-
  breaking pressure the writer cannot self-generate. v13 keeps
  backbone training and adds curriculum / auxiliary content-aware
  probing loss.
- *frozen beats trained*: backbone co-evolution still corrupts the
  channel even with the new writer; architectural fix needs to be
  deeper (untie GRU weights across slots, slot-specific bias) or
  spec-flexible (m^t out of depth softmax).
- *both collapse identically (D2 row_entropy_norm > 0.98 by step 5000
  in both arms)*: KILL. The slot-attention idea — as currently
  formulated — does not solve the writer's symmetric uniform fixed
  point. Pivot to v13 with one of: per-slot untied GRU, slot-bias,
  or out-of-spec routing.

**Decision triggers (mirror across both arms exactly):**
- step 1500: α_mem_max > 0 AND ‖m^t‖/‖embed‖ ∈ [0.3, 50]
- step 5000: α_mem_max > 1e-2 AND pa_cb_dnm > +0.05 AND
              D2 row_entropy_norm < 0.98
- step 12500: pa_cb_dsh > +0.020 AND α_mem_max NOT decaying
- step 25000: standard Δ_sh-m > +0.010
- KILL @ 5000: pa_cb_dsh < +0.005 AND row_entropy_norm > 0.98 in
  both arms → architectural pivot.

**GH200 queue, final (timestamp order):**
```
1777663472_chain_v12d_slot_attention_frozen_chinchilla_mega_gh200.json
1777663600_chain_v12d_slot_attention_trained_chinchilla_mega_gh200.json
1777663800_v12d_d5_ttt_epilogue_frozen.json
1777663900_v12d_d5_ttt_epilogue_trained.json
```

D5 epilogue is now per-arm (parameterized by frozen/trained).  Cross-
arm D5 comparison disambiguates *which writer (frozen-trained vs
trained-trained) encoded usable content for a deeper readout to
recover*.

### v11m cancelled (2026-05-01 ~19:30 UTC)

User directive 2026-05-01 ~14:30 UTC-5:
> "I have a feeling with these improvements v11 is suddenly obsolete,
>  does having v11m chinchilla make sense anymore?"

**Verdict: v11m killed. Moved to `paper_tools/cloud_watchdog/cancelled/`.**

v11m's stated single hypothesis (its own header): "the v11g grow-then-
decay isn't a routing-collapse... it's the predicted behaviour of a
9.7M-parameter from-scratch subnetwork that has only seen ~25% of its
Chinchilla token budget." This token-starvation hypothesis is now
decisively rejected by three independent findings:

1. **D2 on v11g/best**: row_entropy/log(2K) = 0.999, eff_rank = 1.02
   — this is the *symmetric uniform fixed point* of permutation-
   equivariant slot attention, a fixed point of gradient. 4× more
   steps cannot escape a gradient fixed point without external
   symmetry breaking.
2. **v11l-fix step 600** (running locally): with backbone frozen
   (zero co-evolution by construction), α_mem_max collapses to 5e-4
   and PA CB Δsh-m = -0.0022. The writer subsystem itself is the
   bottleneck; token budget is downstream of that.
3. **v12a step 200**: PA CB Δnm-m = +0.3857 in 200 steps with slot
   attention vs v11g's BEST of +0.027 in 4000 steps with the original
   judge. PA CB Δsh-m turned positive (+0.0205) for the first time
   in the campaign. Architectural change is qualitatively different
   behaviour, not a speedup.

v11m is also fully subsumed for ablation purposes:

| Cell | writer   | backbone | corpus | budget     | carry/k=4/burn | role                 |
| ---- | -------- | -------- | ------ | ---------- | -------------- | -------------------- |
| v11g | original | trained  | LME    | small      | no             | done, baseline       |
| v11l | original | frozen   | LME    | small      | no             | running, (b) test    |
| v11m | original | trained  | LME    | chinchilla | yes            | **CANCELLED**        |
| v11p | original | frozen   | mega   | chinchilla | yes            | comparator for v12d  |
| v12d | slot     | frozen   | mega   | chinchilla | yes            | headline             |

v11p is strictly more informative than v11m for the chinchilla
question: same budget, more diverse corpus, cleaner backbone regime
(frozen), and is the direct 1-axis counterfactual for v12d. If
chinchilla scaling on the original writer can save anything, v11p
will tell us; v11m would tell us less. Cancelling v11m saves ~13 h
of GH200 time → v12d starts ~13 h sooner.

GH200 queue post-cancel:
```
1777591646_chain_v11p_ap_frozen_chinchilla_mega_gh200.json
1777663472_chain_v12d_slot_attention_frozen_chinchilla_mega_gh200.json
1777663742_v12d_d5_ttt_epilogue.json
```

---

## v11 campaign — first 7 cells finished, post-mortem (2026-05-01 ~18:20 UTC)

User directive 2026-05-01 ~13:20 UTC-5 (~18:20 UTC):
> "check how v11 is running now"

22 hours after the v11 queue went live the GH200 has finished 7 of
the 9 enqueued cells, two failed at startup, one (`v11q`) is mid-run
on a broken regime, the watchdog/heartbeat have died, and the two
deferred Chinchilla cells (`v11m`, `v11p`) are stranded in the queue
because no daemon is alive to launch them. This section is the
ground-truth ledger of what each finished cell actually produced and
which audit hypotheses (P0–P5) survived contact with the data.

### Wall-clock & infrastructure state

* GH200 192.222.50.225, 1 × H200 (96 GB).
* `pgrep watchdog` and `pgrep heartbeat` both empty — daemons died
  at unknown time. v11q was launched manually in tmux session
  `cwd-chain_v11q_ap_contrastive_gh200` (created May 1 15:49 UTC).
* Queue contents: `1777591645_chain_v11m_…json`,
  `1777591646_chain_v11p_…json` — both with `.deferred_for_gpu`
  marker files dated Apr 30 23:27 UTC. **Will not auto-launch.**
* GPU state at observation: 42 % util, 66 GB / 98 GB used (v11q only).

### Finished-cell summary table (final-step PA-eval, EVID-eval, ROUTE)

| cell | wall-clock (UTC) | core change vs v11g | final α_mem_max | final PA CB Δ_nm-m | final PA CB Δ_sh-m | final `evidence_lift` | verdict |
|---|---|---|---:|---:|---:|---:|---|
| **v11g** baseline | Apr 30 21:46 → May 1 00:26 | — (reference: P0+P2, AP +4/0, gated, hidden_14, k=3) | 0.0093 | **−0.116** | −0.018 | n/a (no EVID-EVAL on this cell) | peak step 600 (PA CB Δ_nm-m=+0.030, Δ_sh-m=+0.016); decayed monotonically through step 4000 |
| **v11h** drop P2 (norm=1.0) | May 1 00:26 → 03:08 | `--memres_readout_norm_init 1.0` | 0.0086 | −0.002 | −0.002 | −0.013 | `‖m^t‖/‖embed‖ = 72`; AP softmax self-regulates magnitude; finishes at PA CB ≈ 0 (basically the same as v11g at step 4000) |
| **v11i** mem_bias=−4 | May 1 03:08 → 05:47 | `--router_mem_bias_init -4` | **0.0001** | +0.006 | +0.005 | +0.002 | router stays fully collapsed from step 0 to step 4000; PA CB ≈ 0 because memory never opens |
| **v11j** depth (k=4 + carry + burn=12) | May 1 05:47 → 09:13 | tests P5 alone | 0.0053 | −0.052 | +0.001 | +0.007 | adding depth *reduced* α_mem opening vs v11g and didn't move Δ_sh-m. P5 in isolation is a no-op. |
| **v11k** P0 reverted (old corpus) | May 1 09:13 → 12:07 | counterfactual on P0 (no `chain_evidence_positions`) | 0.0078 | **−0.265** | +0.027 | n/a | confirms P0 helps quantitatively (final Δ_nm-m −0.27 vs v11g's −0.12, ~2.3× worse) but doesn't change the *shape* of the failure |
| **v11l** frozen backbone | May 1 15:48 (3 sec) | `--freeze_backbone` | — | — | — | — | **FAILED at startup** — `scripts/train_v11l_…sh` invokes `python src/train_chain.py` but the file is at `train_chain.py` in CWD. Exit code 2. **Single-knob frozen-backbone hypothesis remains untested.** |
| **v11r** readout warmup + InfoNCE | May 1 12:07 → 15:48 | `--readout_warmup_steps 500` + `--readout_warmup_router_bias 4.0` + `--contrastive_infonce_weight 0.5` | 0.0101 | **+4.91** | **−1.14** | **−1.12** | broken regime (see §"Why v11r and v11q must not be promoted" below). Memory is *worse* than shuffle by 1.14 nats; oracle-evidence M_c is *worse* than random-haystack M_c by 1.12 nats. |
| **v11q** InfoNCE alone | May 1 15:49 → live (~step 2600/4000) | `--contrastive_infonce_weight 0.5` | 0.019 | **+13.04** | **−0.96** | **−0.57** | same broken regime as v11r. NCE diag/off gap is +6 to +14 (head learning chain-identity discrimination), but `pa_cb_ce_mem = 6.3 nats` (vs ~1.5 nats for normal LM regime) — the readout is destroying generic next-token prediction without helping the answer span. **Should be killed.** |
| **v11_4b_mega** | Apr 30 21:46 (5 sec) | qwen3-4b-xlarge, L_E=10, k=4 | — | — | — | — | **FAILED at startup** — CUDA OOM during `model.to(device)` (peak ~106 GB on a 96 GB card). Same OOM that killed the original v10 4B mega. Needs `--use_adam8bit`, `--freeze_backbone`, or a smaller backbone. |

### What the cross-cell evidence says about P0–P5

Treating the post-v10 audit's six hypotheses as falsifiable:

* **P0 (data: missing evidence labels) — confirmed real, magnitude
  modest.** v11g vs v11k differ in *only* the corpus
  (`v11_lme_msc_train_s512.pt` with `chain_evidence_positions` vs
  the legacy v10 corpus). PA CB Δ_nm-m moves from −0.265 → −0.116
  (factor of 2.3 in the right direction). The fix is necessary
  but not sufficient; the failure shape (peak-then-decay,
  Δ_sh-m ≈ 0) is unchanged.
* **P1 (router saturation) — causally confirmed.** v11i with
  `mem_bias = −4` parks at α_mem_max = 1e-4 from step 0 to step 4000;
  v11g/h/j/k/r with `mem_bias = 0` open α_mem to 5e-3 to 1.9e-2.
  This is the cleanest single-knob result of the campaign.
* **P2 (readout magnitude) — irrelevant for AP, only matters for
  `simple_gate`.** v11g (norm=0.05, ‖m^t‖=3.6) and v11h (norm=1.0,
  ‖m^t‖=72) finish at the same PA CB metrics. The depth softmax
  in `attention_parity` handles magnitude on its own. *Recommend
  dropping P2 from the v12 design constraints.*
* **P3 (PA-eval misalignment) — fixed and now informative.** All
  cells run on the same evidence-aware PA-eval; the metric now
  tracks training distribution faithfully. Useful as a leading
  indicator (v11g's step-600 peak was visible 3000 steps before
  the standard EVAL even moved).
* **P4 (gradient dilution) — partially addressed by `cb_loss_w=3`
  + InfoNCE.** v11q's NCE diag/off gap of +6 to +14 nats proves
  the gradient signal is now strong enough to shape M_c — but in
  the wrong direction (chain-identity instead of content). P4
  was real but the fix's direction matters.
* **P5 (recurrence depth mismatch) — no-op alone.** v11j with
  `window_k=4 + carry_state + burn_in_max=12` finished with
  *lower* α_mem opening (0.0053 vs v11g's 0.0093) and slightly
  more negative PA CB. Depth alone doesn't help; arguably it
  spreads the same gradient over more steps and hurts.

### `evidence_lift` is the smoking gun (new diagnostic, all v11 cells)

`evidence_lift = pa_cb_ce_mem(actual evidence session) − pa_cb_ce_mem(random haystack session)`.
A working memory should make this **strongly negative** (lower CE
when the right evidence is in M_c). Across the campaign:

| cell | final evidence_lift | reading |
|---|---:|---|
| v11h | −0.013 | flat — model has learned to compress *something* but not specifically the answer |
| v11i | +0.002 | flat (router collapsed; no signal at all) |
| v11j | +0.007 | flat (also no specific evidence binding) |
| v11r | **−1.12** | catastrophically *anti*-aligned: oracle-evidence M_c is 1.12 nats *worse* than random-session M_c |
| v11q | **−0.57** | same direction as v11r, weaker magnitude |

The v11r/v11q pattern is unambiguous: *aggressive contrastive
losses on M_c teach the writer to encode chain-identity rather
than chain-content.* The InfoNCE objective rewards
"M_c[i] uniquely predicts chain i's callback" but does not
specify *which* feature of chain i must drive the prediction —
and the easiest discriminator is a chain-identity hash, not the
answer text. Once the writer learns that hash, the readout
delivers a high-magnitude shoutout that destroys generic LM
coherence (`pa_cb_ce_mem` jumps from ~1.5 nats to 6–13 nats).

### Why v11r and v11q must not be promoted

v11r's PA CB Δ_nm-m=+4.91 and v11q's Δ_nm-m=+13.0 *look* like
wins, but they are not:

```text
v11q step 2600:  pa_cb_ce_mem  = 6.275
                 pa_cb_ce_nomem = 19.31  (= 6.275 + 13.04)
                 pa_cb_ce_shuffle = 5.31  (= 6.275 − 0.96)
```

`pa_cb_ce_nomem = 19.3 nats` is not a baseline humans can clear;
it is a model whose general next-token distribution has been
broken by the contrastive objective's interference with the
backbone. Δ_nm-m is a difference between two pathological
numbers. The Δ_sh-m signal is the only honest reading
("does memory beat a random other chain's memory?") and it is
**negative** for both cells.

**Recommended action:** kill v11q
(`tmux kill-session -t cwd-chain_v11q_ap_contrastive_gh200; pkill -9 -f train_chain.py`),
do not promote v11r or v11q checkpoints to anything downstream,
revert the InfoNCE knob to 0 in any v12 baseline.

### Outstanding ledger items

* **v11l** — relaunch after fixing the `src/train_chain.py` →
  `train_chain.py` path bug in `scripts/train_v11l_ap_frozen_backbone_gh200.sh`.
  The frozen-backbone hypothesis (mechanism (b) from the §"v11
  frozen-backbone + Chinchilla" section above) remains the only
  single-knob v11 cell that has not been tested.
* **v11_4b_mega** — relaunch with `--use_adam8bit` (or
  `--freeze_backbone`, or downscale to qwen3-1.7B) to fit in 96 GB.
  Same OOM that killed the v10 4B; nothing has changed at the
  optimizer-memory level.
* **v11m_chinchilla / v11p_frozen_chinchilla_mega** — restart the
  watchdog (`paper_tools/cloud_watchdog/watchdog.sh &
  paper_tools/cloud_watchdog/heartbeat.sh &`) or pull them out of
  the queue manually. Currently both have `.deferred_for_gpu`
  markers and zero progress.

### Headline conclusion (mechanism, not symptoms)

After eliminating P0/P2 (data + magnitude), confirming P1 (router
gating), and disconfirming P5 (depth alone), the failure mode
that survives is:

> **The writer is content-blind.** With the LM-only objective
> (v11g/h/i/j/k) it learns to compress *something* about each
> session but `evidence_lift ≈ 0` — using the actual evidence
> session is no better than a random haystack session. With
> dense contrastive supervision (v11r/q) it learns chain-identity
> and `evidence_lift` becomes *strongly negative*. Either way
> the writer never learns "extract the answer from this
> session" because no objective in the v11 pipeline tells it to.

This is consistent with the D5 audit recorded below
(readout-only fine-tuning recovers 48 % of callback CE — the
writer's M_c does carry chain-discriminative content, just not
*answer-specific* content). The most informative single next
cell is therefore not another curriculum tweak but a writer-only
warmup against an extractive objective (e.g. mask-and-recover
the answer span from the evidence session into M_c) before any
LM gradient flows. That is a different decomposition than v11r
tried (which warmed the *readout*, not the *writer*).

### Files / state observed (read-only)

```
GH200 paper_tools/cloud_watchdog/logs/chain_v11{g,h,i,j,k,r}_*.log  (all exit_code=0)
GH200 paper_tools/cloud_watchdog/logs/chain_v11l_*.log              (exit_code=2, 237 B)
GH200 paper_tools/cloud_watchdog/logs/chain_v11_4b_mega_*.log       (exit_code=0 but OOM at .to(device))
GH200 paper_tools/cloud_watchdog/logs/chain_v11q_*.log              (live, ~step 2600, 37 KB and growing)
GH200 paper_tools/cloud_watchdog/queue/                             (v11m, v11p, both .deferred_for_gpu since Apr 30 23:27)
GH200 tmux session cwd-chain_v11q_ap_contrastive_gh200              (running v11q manually)
```

---

## Diagnostic audit & v11r — readout-warmup architectural fix (2026-05-01 ~06:55 UTC)

User directive 2026-05-01 after v11i finished and v11j started:
> "I care much more that evidence of a working recipe surfaces so we
>  can mega-scale a project and clear of potential loopholes. For
>  example, the competition mechanism really bothers me, just as much
>  as 'memory condensation' is not working to a correct gradient."
> "Sure, do everything you just said" (re: D1-D5 diagnostics + an
>  architectural fix).

### Diagnostic toolkit shipped (D1-D5)

Five mechanism-level audits, all integrated into `src/train_chain.py`
or shipped as standalone tools:

| ID  | What it measures | How |
|-----|------------------|-----|
| D1  | Per-module gradient L2 norms (M_in / extract / M_judge / judge / readout / router / write_gate / backbone) at each `--log_every` step. Surfaces gradient starvation in the writer subsystem. | `--diagnose_grad_groups` flag; logs `grad/<group>` to wandb and prints `|g|/|g_bb|` ratios per step. |
| D2  | Judge attention decisiveness: row-entropy vs `log(2K)`, mean keep-vs-write mass, variance over rows, effective rank of the average judge attention pattern. | `--diagnose_memory_dynamics` flag; computed in `_memory_dynamics_eval`; uses `MemoryBlock.judge_attention(...)` (new helper in `modeling_memres.py`). |
| D3  | M_c stability per session step (`||M_c^t - M_c^{t-1}||_F / ||M_c^{t-1}||_F`); chain-distinguishability via pairwise normalized Frobenius distance between distinct-chain `M_c^T`s. Detects content-blind writer. | Same `_memory_dynamics_eval` pass. |
| D4  | Synthetic gold-standard task: 5000-chain persona-callback corpus, 256-item closed set, 9 sessions/chain (persona + 7 fillers + callback). Hard ground truth: `callback_ce → 0` only if memory works. | `tools/build_synthetic_persona_callback.py`; corpora at `paper_artifacts/chains/synthd4_persona_callback_{train,val}_s512.pt`. |
| D5  | TTT-on-readout disambiguator: freeze writer + router + LM head, train ONLY the readout (W_Q/W_K/W_V) for 300 steps. If callback CE drops, writer encoded the info; readout was the bottleneck. | `tools/d5_ttt_readout.py`. |

### Audit run on `chain_v11g_ap_baseline_gh200/best` (~Apr 30 22:42 UTC)

Eval-only invocation against the synthetic D4 corpus:

```text
ROUTE: mode=attention_parity  alpha_or_gate_max=0.0092  frac_open=0.00
READOUT: ||m^t||/||embed|| mean=77.23
D2-JUDGE: row_entropy=5.541 (uniform=5.545; norm=0.999) keep_mean=0.500
          keep_var=0.0000 eff_rank=1.02
D3-MC   : Δ_step mean=1.347 max=1.383 self||M||=1.000 pair=0.022
          (pair/self=0.022)
pa_cb_dnm = +0.373  pa_cb_dsh = +0.0023  pa_cb_evidence_lift = +0.016
                                                          (synth)
```

Recorded at `results/exp2_chain_recipe/v11g_diag_synth.json`.

D5 with 300 readout-only steps (lr 1e-3, batch 4):

```text
baseline callback_ce: 8.1989
final    callback_ce: 4.2629
Δ                   : -3.9359  (-48.0%)
VERDICT: LIKELY R: the writer encoded the information; the
                   readout was the bottleneck.
```

Recorded at `results/exp2_chain_recipe/v11g_d5_ttt.json`.

### Diagnosis (mechanism, not symptom)

1. The **judge** is decision-less (D2 entropy/uniform = 99.9 %; effective
   rank 1.02; keep mass exactly 0.500 across all rows). However, this
   is a *symptom* not the disease — the V-projection of the judge
   still carries chain-content (D5 functionally proves it).
2. The **writer** does encode chain-specific content (D5 unblocks
   48 % of the callback CE gap by tuning the readout *alone*). Its
   structural pair-distinguishability (D3 = 2.2 %) is small but
   information-theoretically sufficient.
3. The **readout** is the structural bottleneck. After 4 000 joint
   steps the readout has learned essentially random projections
   from M_c — the LM-loss signal it received was attenuated by the
   router's ~1 % alpha_mem (D5 demonstrates ~3.9 nats of recoverable
   callback CE remained on the table after 4 000 steps).
4. The **router** locked the memory pathway closed
   (`alpha_mem_max = 0.0092`, `frac_open = 0`) before the readout had
   time to converge. Cause: every step the readout's m^t worsens the
   LM loss, the router learns to route around it. Effect: the readout
   gets even less gradient. Lock-in.
5. v9c worked because its readout norm magnitude (`m^t`-norm/embed
   `~ 165` at step 500 in chain_v7_p0_simplegate; `~ 77` at step 4 000
   in v11g) was preserved by `simple_gate`'s scalar boost; it did
   *NOT* mean the writer was producing structurally distinct M_c —
   D3 says it almost certainly was content-blind by structural metrics
   in v9c too.

### Architectural fix shipped: scaffolded readout warmup (v11r)

`train_chain.py` — new CLI knobs:

| flag | default | purpose |
|------|---------|---------|
| `--readout_warmup_steps` | 0 (off) | Phase 1 length. Freezes writer + router + backbone + LM head; only the readout (`memory_readout.W_Q/W_K/W_V/out_norm`) trains. |
| `--readout_warmup_router_bias` | 4.0 | `mem_bias` value held throughout phase 1. Forces routing toward memory so the readout receives strong LM gradient. |
| `--readout_warmup_anneal_steps` | 200 | After phase 1, linearly anneal `mem_bias` from `--readout_warmup_router_bias` back to `--router_mem_bias_init` over this many steps. |

Implementation: `Trainer._readout_warmup_freeze`, `_readout_warmup_unfreeze`,
`_set_mem_bias`; called from `Trainer.fit` per-step. Smoke-tested on v11g/best
init: at end of phase 1 (step 4) all groups except readout received zero
gradient (D1 confirmed); at step 8 (post-anneal) the routing had already
opened to `alpha_mem_max = 0.0614` (vs 0.0092 on v11g/best — 6× higher) and
`frac_open = 0.11` (vs 0.0).

### v11r config (single-knob vs v11g)

Adds:
- `--readout_warmup_steps 500`
- `--readout_warmup_router_bias 4.0`
- `--readout_warmup_anneal_steps 200`
- `--contrastive_infonce_weight 0.5` with 500-step warmup (same as v11q)
- `--diagnose_grad_groups`
- `--diagnose_memory_dynamics`

Everything else identical to v11g (attention_parity +4/0, gated update,
hidden_14 extract source, P2 readout norm 0.05, P0 evidence curriculum,
4 000 steps, lr_memres=5e-5, lr_backbone=2e-5, callback_loss_weight=3.0).

### Decision triggers (v11r)

| step | required signal |
|------|-----------------|
| 200 (mid-warmup) | `nce_gap > +0.5` (readout learning to discriminate). |
| 500 (end-warmup, pre-anneal) | `||m^t||/||embed|| ∈ [0.3, 50]`; `judge_row_entropy_norm < 1.0` (D2 dropping). |
| 1000 (post-anneal) | `pa_cb_dsh > +0.020`, `pa_cb_evidence_lift > +0.020`, `alpha_mem_max > 0.05` (router did not re-close). |
| 2000 | `pa_cb_dsh > +0.050` (architectural path is real). |
| 4000 | `pa_cb_dsh > +0.100` (head-line goal). |

KILL @ step 1000 with `alpha_mem_max < 0.01`: warmup did not escape
lock-in; relaunch with longer warmup (1 000 steps) or stronger
force-open bias (+6).

### Queue position

GH200 watchdog queue order (oldest mtime first):
1. v11j — currently running.
2. v11k — next up (no_evidence ablation).
3. **v11r — slot 3, the architectural-fix experiment.**
4. v11l — frozen-backbone control.
5. v11q — InfoNCE alone (no warmup); ablation isolating supervision-vs-architecture.
6. v11m, v11p — Chinchilla scaling experiments (last, lowest-leverage given the audit).

Sources of v11r being a "load-bearing experiment":
1. It is the *only* current run that targets the diagnosed
   readout-router lock-in directly.
2. The diagnostic stack (D1-D5) ships inside v11r, so its log will
   double as a longitudinal dataset of how all five metrics evolve
   under the new training regime — the exact signal we need to
   decide whether to scale to 4 B / megacorpus or pivot the
   architecture again.

### Files added / changed (this iteration)

```
tools/build_synthetic_persona_callback.py    (D4 corpus generator)
tools/d5_ttt_readout.py                      (D5 standalone tool)
src/modeling_memres.py                       (MemoryBlock.judge_attention helper)
src/train_chain.py                           (D1/D2/D3 telemetry + fix A flags)
Scripts/train_v11r_ap_readout_warmup_gh200.sh  (v11r launcher)
paper_artifacts/chains/synthd4_persona_callback_train_s512.pt
paper_artifacts/chains/synthd4_persona_callback_val_s512.pt
results/exp2_chain_recipe/v11g_diag_synth.json   (audit data)
results/exp2_chain_recipe/v11g_d5_ttt.json       (audit data)
```

---

## v11 campaign — InfoNCE contrastive supervision (2026-05-01 ~06:07 UTC)

User directive 2026-05-01 ~05:54 UTC after the v11g/h/i diagnostic
review and the post-v11i full rundown:
> "add a contrastive loss to the GH200, queued"

### Hypothesis under test (v11q)

The v11g grow-then-decay pattern admits two compatible mechanisms:
backbone co-evolution (b) and readout/writer overfit (a) — but it
*also* admits a third axis we haven't isolated: the memory subsystem
is supervised only by next-token CE on callback tokens (<5% of
training tokens), routed through three softmaxes (judge → depth →
readout) before any gradient touches the writer. The training
signal is sparse, indirect, and noisy. v11q tests whether the
right fix is *dense supervision on Δ_sh-m directly*, before
attempting any architectural pivot (per-block memory refresh,
moving m^t out of the depth softmax, two-tier memory).

### What v11q changes (single-knob vs v11g)

`--contrastive_infonce_weight 0.5` with linear ramp 0.05 → 0.5
over the first 500 steps (`--contrastive_infonce_warmup_steps 500`,
`--contrastive_infonce_initial_weight 0.05`). Default temperature
1.0. Callback-only scoring (default `True`). Everything else
identical to v11g (`attention_parity` +4/0, P0 evidence-aware
curriculum, P2 readout norm 0.05, hidden_14 extract source,
`window_k=3`, batch=4, grad_accum=2, lr_memres=5e-5,
lr_backbone=2e-5, 4 000 steps, 200-step warmup).

### Code change shipped pre-launch

`train_chain.py` — new CLI knobs:

* `--contrastive_infonce_weight <float>` (default 0.0): for each
  batch element i, score last session i under every batch element
  j's M_c (B*B forward on the last session only — TBPTT chain is
  reused). Cross-entropy with diagonal=positive, gradient flows
  through M_c[j] for all j so the off-diagonal pressure pushes
  M_c[j] _away_ from chain i's content. Direct attack on
  Δ_sh-m ≈ 0.
* `--contrastive_infonce_temperature <float>` (default 1.0): NLL
  values are typically [1, 4] nats so T=1.0 is natural; lower T
  sharpens.
* `--contrastive_infonce_initial_weight <float>` (default
  =`--contrastive_infonce_weight`) and
  `--contrastive_infonce_warmup_steps <int>` (default 0): linear
  ramp.
* `--contrastive_infonce_callback_only` /
  `--no-contrastive_infonce_callback_only` (default True): score
  per-pair NLL only on callback-supervision tokens (with fallback
  to all valid tokens when no callback is present). Concentrates
  the contrastive signal on the tokens that actually require
  memory.

The InfoNCE block sits after the existing `--neg_chain_weight`
(single-negative hinge) and `--in_chain_contrast_weight`
(perturbation hinge) blocks. All three are independently
toggleable; v11q runs InfoNCE alone for clean attribution.

Side fix (carried in the same commit): the in-chain perturbation
slot variable was named `F`, which silently shadowed
`import torch.nn.functional as F` for the rest of `_train_step`.
Renamed to `slot_idx` so InfoNCE's `F.cross_entropy` works.
Smoke-tested locally on H100: 5 steps run cleanly,
`nce 1.46x diag 5.5 off 5.4 gap −0.08` at step 1 (uniform-class
baseline `ln(B=4) ≈ 1.386`, expected at init), throughput
~5k tok/s on H100 (vs 6k baseline; ~25% per-step overhead).

### v11q decision triggers (sharper than v11g; contrastive directly attacks Δ_sh-m)

* step 200  — `pa_cb_dsh > 0` AND `||m^t||/||embed|| ∈ [0.3, 50]`
  AND `nce_gap (off − diag) > 0`
* step 500  — `pa_cb_dsh > +0.005` AND `nce_gap > +0.05`
* step 1000 — `pa_cb_dsh > +0.020` (vs v11g's +0.005 — the
  contrastive objective should hit this faster because we're
  optimising it directly)
* step 2000 — `pa_cb_dsh > +0.030` AND standard `Δ_sh-m > +0.005`
* step 4000 — standard `Δ_sh-m > +0.010` (final calibration test)
* **KILL: step 1000 with `pa_cb_dsh < 0` OR `nce_gap ≤ 0`** —
  contrastive failed to make the readout content-specific even
  with dense supervision. The architecture is the wall, not the
  pedagogy. Pivot to the architectural axis (per-block memory
  refresh / move m^t out of depth softmax / two-tier memory).

### Why this is the load-bearing experiment

v11l (frozen backbone) tests whether mechanism (b) dominates;
v11p (frozen + Chinchilla + mega) is the headline scaling test
under that hypothesis. **Neither addresses the supervision-deficit
hypothesis.** v11q does. The three together form a diagnostic
fork that resolves the v11g grow-then-decay attribution:

* **v11l works (no decay) but v11q stays flat** → mechanism (b)
  dominates; backbone co-evolution is the structural problem and
  freezing is the answer. Ship the recipe with `--freeze_backbone`.
* **v11q works (Δ_sh-m positive at step 1000) but v11l decays** →
  pedagogy was the bottleneck; sparse callback-CE-only supervision
  was insufficient and dense contrastive supervision opens the
  channel even with the backbone trainable. Ship the recipe with
  contrastive aux loss and full backbone training.
* **Both work** → both axes contribute; combine them in the next
  cell (v11r).
* **Neither works** → the v11 architectural recipe (Block-AttnRes
  on m^t inside the same softmax as `b_k`) is structurally broken
  even with dense supervision and a stable backbone target.
  Architectural pivot required: B1 (per-block memory refresh) or
  B5 (two-tier memory). This is the cleanest single-experiment
  falsification of the paper's "implicit cognitive routing via
  depth softmax" claim we have in flight.

### Status (2026-05-01 ~06:11 UTC)

* **v11q_ap_contrastive** — ENQUEUED on GH200 watchdog. Spec
  `paper_tools/cloud_watchdog/queue/1777615619_chain_v11q_ap_contrastive_gh200.json`,
  launcher `scripts/train_v11q_ap_contrastive_gh200.sh`. mtime
  manually set to `2026-04-30 23:27:24.700 +0000` (between v11l
  and v11m) so the watchdog's `ls -1tr` ordering puts v11q in
  **slot 3** of the queue: v11j (running) → v11k → v11l → **v11q**
  → v11m → v11p. ETA-to-start: ~9 h (v11j ~5 h remaining +
  v11k 3 h + v11l 3 h). The l ↔ q diagnostic fork resolves in
  ~12.3 h after v11q starts (3.3 h to step 4000), well before the
  ~27 h Chinchilla cells launch.

---

## v11 campaign — frozen-backbone + Chinchilla expansion (2026-04-30 ~22:30 UTC)

User directive 2026-04-30 ~22:30 UTC after the v11g grow-then-decay
review:
> "It seems to me that a frozen backbone with only tuning the
> summarizer for memory is now a necessary ablation study, can you
> make one in the image of v11g and add it to the gh200 queue?
> remove the 4B model. And after that, add any tests you deem
> necessary onto the queue as well. If data is issue, make more
> data. I know from a fact that attention parity works better, so
> there's no need there. Think of a way we can apply the chincilla
> budget to our system."

### What v11g actually showed (the trigger for this expansion)

Single-cell PA-eval trajectory of `chain_v11g_ap_baseline_gh200` on
GH200:

| step | α_mem_max | top sublayers | PA CB Δnm-m | PA CB Δsh-m |
|---:|---:|---|---:|---:|
| 200 | 0.0047 | l54, l53, l13 | +0.0181 | −0.0010 |
| 400 | 0.0108 | l12, l54, l13 | **+0.0360 ← peak** | +0.0011 |
| 600 | 0.0124 | l12, l13, l11 | +0.0301 | +0.0156 |
| 800 | 0.0111 | l12, l11, l13 | +0.0152 | +0.0014 |
| 1000 | 0.0113 | l12, l13, l9 | **−0.0191** | −0.0099 |
| 1200 | 0.0104 | l12, l13, l11 | −0.0124 | −0.0006 |
| 1400 | 0.0102 | l13, l12, l11 | −0.0174 | +0.0218 |

`α_mem_max` peaks at step 600 then monotonically declines while
`α_mem_mean` keeps creeping up (0.0025 → 0.0034) — the depth router
is *redistributing* memory mass from a few sharp sublayers across
many shallow ones, not closing the channel. Top sublayers shifted
from {l54, l53} late at step 200 to {l11, l12, l13} early by step
400 and stayed there. This is the canonical attention_parity
"grow-then-decay" pattern that v8a/v8b/v11g all exhibit on the
matched-distribution PA-eval; v9c is the only run on record that
*didn't* decay (it bootstrapped slowly through step 1000 then
plateaued through step 4000 at +0.12 to +0.18 PA CB Δnm-m).

Two competing mechanisms are consistent with the observations:

* **(b) Backbone co-evolution** — as the trainable backbone fine-
  tunes its block-AttnRes summary heads (`b_0..b_{N-1}`) for the
  LME chain task, those heads become better-conditioned predictors,
  the depth softmax in `attention_parity` reallocates mass to them,
  and `m^t` loses its voice. This is structural to AP (block
  summaries compete inside the same softmax as memory).
* **(a) Readout / writer overfit** — the memory subsystem itself
  drifts toward a degenerate fit on the training distribution, and
  pa_cb_dnm reverses sign on the matched-distribution eval because
  the readout's content-conditioning learnt during steps 200–600
  has been over-specialised by step 1000.

Independently of which dominates, the memory subsystem is *literally
from-scratch*: the load report shows 18/18 memres tensors MISSING
from the pretrained Qwen3-0.6B checkpoint. With ~9.7 M from-scratch
parameters and v11g's ~49 M training tokens, the run is at ~25% of
the Chinchilla token budget for this subsystem. Both axes (frozen
backbone, Chinchilla token budget) therefore admit clean single-knob
tests.

### v11 second-wave cells (replaces 4B mega; queued 2026-04-30 ~22:35 UTC)

| cell | machine | corpus | step budget | trains | bias | special |
|---|---|---|---:|---|---|---|
| **v11l_ap_frozen_backbone** | GH200 | v11_lme_msc | 4 000 | memres-only | +4/0 | `--freeze_backbone`; otherwise IDENTICAL to v11g (single-knob ablation) |
| **v11m_ap_chinchilla** | GH200 | v11_lme_msc | 16 000 | memres + backbone | +4/0 | k=4 / 12 (resample), carry_state; 4× more steps + deeper TBPTT; ~262 M tokens (1.35× Chinchilla for memres) |
| **v11p_ap_frozen_chinchilla_mega** | GH200 | **v11_mega** (67 745 chains) | 25 000 | memres-only | +4/0 | `--freeze_backbone`, `--lr 1e-4` (2× v11g), k=4, carry_state, burn_in 12 resampled; ~410 M tokens (2.1× Chinchilla); the v11 HEADLINE replacement |

All three: `attention_parity`, `--router_recent_bias_init 4
--router_mem_bias_init 0`, `--memres_gate_init 0.0`,
`--memres_readout_norm_init 0.05`, `--curriculum_competition_bias 1.0`,
`--callback_loss_weight 3.0`, `--save_best_metric phase_aligned`. v11l
mirrors v11g's lr / warmup / window_k for clean single-knob comparison;
v11m and v11p apply the Chinchilla-budget axis (longer training, deeper
recurrence) to test the "memres is from-scratch and token-starved"
hypothesis.

### v11l decision triggers (single-knob vs v11g)
* step 200 — `α_mem_max > 0` AND `||m^t||/||embed|| ∈ [0.3, 50]`
* step 500 — `α_mem_max > 5e-3` AND `pa_cb_dnm > +0.005` AND
  `pa_cb_evidence_lift > +0.005`
* step 1000 — `α_mem_max > 1e-2` (not decaying — vs v11g step 1000
  which sat at 0.011 and was already declining from peak)
* step 2000 — standard `Δ_sh-m > +0.005`
* KILL: step 1000 with `α_mem_max < 1e-3`. Frozen backbone *and*
  attention_parity collapsed → rules out (b) entirely; the decay is
  internal to memres and the next move is readout-regularisation.

### v11m decision triggers (Chinchilla budget on full training)
* step 1000 — `α_mem_max > 5e-3` AND `pa_cb_dnm` not yet collapsed
  (matches v11g step 600–800)
* step 4000 — `pa_cb_dnm > +0.05` (4× more steps should give v9c-tier
  gain by here; v9c step 1400 was +0.053)
* step 8000 — `pa_cb_dsh > +0.010` (content-specific, not just
  unconditional pull)
* step 16000 — standard `Δ_sh-m > +0.005`
* KILL: step 4000 with `pa_cb_dnm < 0` AND `α_mem_max < 5e-3` →
  Chinchilla budget didn't fix the collapse; falsifies token-
  starvation as the dominant mechanism.

### v11p decision triggers (HEADLINE; frozen + Chinchilla + mega)
* step 1500 (~1.5h) — `α_mem_max > 0` AND `||m^t||/||embed|| ∈ [0.3, 50]`
* step 5000 (~5h) — `α_mem_max > 1e-2` AND `pa_cb_dnm > +0.05`
* step 12500 (~12h) — `pa_cb_dsh > +0.020` AND `α_mem_max` stable or
  growing (NOT decaying like v11g)
* step 25000 (~24h) — standard `Δ_sh-m > +0.010`
* KILL: step 5000 with `α_mem_max < 1e-3`. If frozen backbone +
  Chinchilla budget + mega corpus + softer bias + readout rescale all
  together cannot open the channel, the v11 architectural recipe is
  falsified and a routing intervention is required before any further
  scaling.

### Why these three cells (decision-tree compactness)

* If **v11l works (no decay) but v11m still decays** → mechanism (b)
  dominates: backbone co-evolution drives the AP softmax to crowd memory
  out. Forces a routing intervention (e.g., separate scalar gate that
  doesn't enter the same softmax as `b_k`, or a stop-gradient on
  block summaries during memory-update windows) before scaling.
* If **v11l decays AND v11m decays the same way** → mechanism (a)
  dominates: readout / writer overfit. The fix is regularisation /
  contrastive aux on memory writes, not architecture.
* If **v11m grows monotonically past step 8000 while v11g decayed by
  step 1400** → memres was simply token-starved; v11p will likely
  succeed too and the v11 recipe ships unchanged but with a 4×
  longer step budget.
* **v11p** is the cleanest configuration we can build with the
  existing flags: from-scratch memres on a stable backbone target
  (no co-evolution), 2.1× Chinchilla token budget, 10× larger corpus,
  deeper recurrence, longer warmup. If this *still* shows the
  grow-then-decay pattern, the v11 architectural recipe (with
  attention_parity locked) is structurally broken and we need to
  break the b_k vs m^t softmax competition before any further
  scaling work is meaningful.

### Removed: v11_4b_mega_gh200 (2026-04-30 ~22:30 UTC)

* **Status:** REMOVED from queue, launcher deleted.
* **Why:** died at startup with CUDA OOM (`model.to(device)` at the
  4B preset's 52 GB peak collided with leftover allocations from the
  watchdog launch race; even after the `gpu_is_busy` patch, the 4B
  run is unstable on a 94 GB GH200 once activations + grad checkpoints
  + AdamW state for the 4B backbone all land at once). The 4B was
  itself a backoff from 8B for the same reason.
* **Replaced by:** `v11p_ap_frozen_chinchilla_mega_gh200`. Same
  headline question (does the recipe scale?) reframed under the
  user-directed correction (frozen backbone, attention_parity
  decided): instead of testing "deeper backbone" we test "biggest
  from-scratch memres budget on a stable backbone target." The
  Qwen3-4B backbone wasn't load-bearing for any v11 hypothesis —
  every architectural lesson (P0–P5) was discovered on the 0.6B
  preset.
* **Spec file removed:** `paper_tools/cloud_watchdog/queue/1777585825_chain_v11_4b_mega_gh200.json`.
* **Launcher removed:** `Scripts/train_v11_4b_mega_gh200.sh`.

---

## v11 campaign — evidence-aware curriculum + bootstrap fix (2026-04-30 ~21:30 UTC)

User directive 2026-04-30 ~21:00 UTC after the v10 audit:
> "create 4-5 ablation studies on the GH200 for 0.6B and then the 'most
> make sense initializers' for 4B model to train. be sure to use a bias
> for the memory at +4 0. I know that the system from v3 definitively
> is better."

v10 was killed mid-training and audited (see [`README.md`](../../README.md#stop-everything-and-read-this-first--what-is-actually-broken-2026-04-30-post-v10-audit)
P0–P5). The audit revealed five compounding failures: (P0) the LME chain
builder dropped `answer_session_ids`/`has_answer` so the competition
curriculum was sampling "evidence" uniformly — 96%+ of training samples
had no real evidence to learn from; (P1+P2) `MemoryGate` initialised at
0.0 + `MemoryReadout.out_norm` at 1.0 made `‖m^t‖/‖embed‖ ≈ 73` while
the gate fed exactly 0.0 of it through, so the backbone learned a
suppression policy in the only window where the readout had any voice;
(P3) the phase-aligned eval picked the "evidence" session uniformly,
matching the broken curriculum and making honest measurement impossible;
(P5) standard-eval ran at recurrent depths the model had never seen
during training. v11 addresses P0/P2/P3 in code and P5 in two of the
ablation cells.

### Hypothesis under test (v11)

If the v3 (attention_parity) routing is "definitively better" as the
user asserts, then with the data fix (P0), the magnitude fix (P2), and
the user-directed softer bias (`+4 / 0`, ~50× more initial α_mem mass
than v3's `+4 / -4`), the memory channel will (a) open
(α_mem_max > 1e-2 by step 1000), (b) learn content-specific writes
(pa_cb_evidence_lift > 0 by step 500), and (c) start closing the
deployment-distribution gap (standard Δ_sh-m > +0.005 by step 2000).

### Code/data fixes shipped before launch (verified locally)

* **`paper_tools/build_conversational_callback_chains.py`** — LongMemEval
  loader now reads `answer_session_ids` and `haystack_session_ids` from
  raw JSON and emits `chain_evidence_positions` (post-filter haystack
  indices of answer sessions). `has_answer` turns inside the haystack
  also flip `session_callback_mask` to `True`, so callback up-weighting
  fires on the actual answer span instead of only the synthetic span.
* **`paper_tools/merge_chain_corpora.py`** + new
  **`Scripts/build_v11_corpora_remote.sh`** — concatenation utility
  preserves `chain_evidence_positions`; the remote builder swaps the
  legacy 450 LME chains in `v6_lme_msc_train_s512.pt` and
  `mega_train_s512.pt` for the new evidence-aware 450 LME chains
  (yielding `v11_lme_msc_train_s512.pt` and `v11_mega_train_s512.pt`).
  Counts: v11_lme_msc 6378 chains / 450 with evidence (898 evidence
  positions); v11_mega 67745 chains / 450 with evidence; both with
  `session_callback_mask` sum = 42201.
* **`train_chain.py`** — (1) `ChainCorpus` loads
  `chain_evidence_positions` (with backward-compat empty-list fallback);
  (2) `ChainSampler.sample_window` uses `ev_in_range` for evidence
  position picks under the competition curriculum, falling back to
  uniform only when no labels exist (so the curriculum is silently no-op
  on legacy corpora rather than crashing); (3) `_phase_aligned_eval`
  picks the evidence session from `ev_in_range` and emits four new
  honest metrics: `pa_cb_ce_mem_floor`, `pa_cb_ce_nomem_floor`,
  `pa_cb_dnm_floor`, `pa_cb_evidence_lift` (memory benefit on
  evidence-labelled callback minus uniform-baseline benefit); (4)
  `--memres_gate_init` and `--memres_readout_norm_init` flags wired
  through the `Qwen3MemResConfig` overrides (including the `from_pretrained`
  override path that was silently dropping them in the first launch).
* **`modeling_memres.py`** — `Qwen3MemResConfig` accepts the two new
  init knobs; `_init_memres_params` now applies
  `module.out_norm.weight.fill_(scale)` when constructing
  `MemoryReadout`. Verified live: with `--memres_gate_init 0.005` and
  `--memres_readout_norm_init 0.05` the trainer reports
  `gate_mean +0.0050 max 0.0050` from step 0 (vs `+0.0000` before the
  override-dropping bug fix).

### v11 cells

| cell | machine | corpus | routing | bias (recent / mem) | readout norm | window / burn-in | tests |
|---|---|---|---|---|---:|---|---|
| **v11_local** | local H100 GPU 0 | v11_lme_msc | simple_gate | n/a | 0.05 | k=3 / 0 | sister to v11g; isolates simple_gate vs attention_parity at fixed P0+P2 fixes; running since 2026-04-30 ~21:00 UTC, currently step 1200, loss 2.52 |
| **v11g_ap_baseline** | GH200 | v11_lme_msc | attention_parity | +4 / 0 | 0.05 | k=3 / 0 | THE BASELINE (user-directed); running since 2026-04-30 ~21:46 UTC |
| **v11h_ap_norm1** | GH200 | v11_lme_msc | attention_parity | +4 / 0 | **1.0** (revert P2) | k=3 / 0 | A/B for P2: does AP's depth softmax self-regulate magnitude or does the readout still need rescaling? |
| **v11i_ap_pm4** | GH200 | v11_lme_msc | attention_parity | +4 / **−4** (v3 default) | 0.05 | k=3 / 0 | A/B for the user's `+4 / 0` claim: with P0+P2 in place, does the v3 default bias still recover or stay collapsed? |
| **v11j_ap_carry_depth** | GH200 | v11_lme_msc | attention_parity | +4 / 0 | 0.05 | **k=4 / 12 (resample)** | tests P5: with deep recurrence in training, does the standard Δ_sh-m metric move? |
| **v11k_ap_no_evidence** | GH200 | **v6_lme_msc** (no evidence labels) | attention_parity | +4 / 0 | 0.05 | k=3 / 0 | A/B for P0: with everything else identical, does the absence of evidence labels collapse routing? |
| **v11_4b_mega** | GH200 | v11_mega (67k chains) | attention_parity | +4 / 0 | 0.05 | k=4 / 12 (resample) | THE 4B HEADLINE: composes user-directed initializers with all four v11 fixes (P0+P2+evidence-aware eval+depth recurrence) on Qwen3-4B-xlarge with L_E=10; 25k steps, 3-day budget |

All ablations: `--callback_loss_weight 3.0 --curriculum_competition_bias
1.0 --memres_extract_source hidden_14` (hidden_18 for the 4B which has
36 layers), `--save_best_metric phase_aligned`, eval every 200 steps for
0.6B and every 500 for 4B.

### Decision triggers (v11g_ap_baseline; the same gates apply to h/i/j/k
unless noted; the 4B trigger schedule is 5× scaled and is documented at
the top of `Scripts/train_v11_4b_mega_gh200.sh`)

* step 200 — `α_mem_max > 0` AND `‖m^t‖/‖embed‖ ∈ [0.3, 50]`
* step 500 — `α_mem_max > 5e-3` AND `pa_cb_dnm > +0.005` AND
  `pa_cb_evidence_lift > +0.005`
* step 1000 — `α_mem_max > 1e-2` AND `pa_cb_dnm > +0.020` AND
  `pa_cb_dsh > +0.005` (memory is content-specific, not just an
  unconditional pull)
* step 2000 — standard `Δ_sh-m > +0.005` (deployment-distribution gap
  closing for the j/4b cells; for the others standard Δ_sh-m is expected
  to lag because they don't train at depth)

### KILL conditions (v11g; same shape for the others)

* step 1000 with `α_mem_max < 1e-3`: AP collapsed even with P0+P2; for
  v11g this would falsify the v11 hypothesis on this corpus and we
  rerun v11g with simple_gate (the v11_local recipe). For v11i it
  *predicts* the user's claim that `+4 / -4` is structurally too tight.
* step 2000 with `pa_cb_evidence_lift ≤ 0`: memory is not content-
  specific even with evidence labels — escalate to a stronger router
  intervention (e.g. larger `--router_mem_bias_init` and/or auxiliary
  evidence loss).

### Compute layout

GH200 watchdog runs the 6 specs **sequentially** (single GPU). Cell
duration estimate (0.6B at ~10–13k tok/s on the GH200): ~3–4 h each →
the five 0.6B ablations take ~17 h, then the 4B headline runs for ~3
days. Total budget ~4 days, fits comfortably in the GH200 reservation.

The watchdog had a launch-race bug discovered on the first attempt: it
launched all six queued specs in the same poll iteration because the
first launch hadn't allocated GPU memory yet, so `gpu_is_busy` returned
false for all subsequent specs. Fixed by adding a `break` after a
successful launch in `paper_tools/cloud_watchdog/watchdog.sh` — the
watchdog now launches at most one job per 30 s poll, and the second
poll correctly defers when the first is using GPU memory.

The trainer scripts also had a `... 2>&1 | tee logs/...log` tail that
swallowed Python's exit code (tee always returns 0), so OOM crashes
were being marked as `done` instead of `failed`. The tee was redundant
with the watchdog's per-job log capture; stripped from all v11 GH200
scripts.

### Status (2026-04-30 ~21:50 UTC)

* **v11_local** — running on local H100 GPU 0, step 1200, loss 2.52,
  `gate_mean +0.0050` (P1+P2 init verified); first phase-aligned eval
  at step 200 already past, awaiting next eval cycle review. Not in the
  watchdog queue (runs locally via `nohup` in `Scripts/train_v11_evidence_aware_local.sh`).
* **v11g_ap_baseline** — running on GH200, step 120, loss 2.70.
* **v11{h,i,j,k}** + **v11_4b_mega** — queued on GH200 watchdog,
  deferred while v11g holds the GPU; will start in order as v11g
  finishes (`bash` script names match the cell names).
* The watchdog daemon is up under PID 149760 on GH200 and the heartbeat
  pushes status to ntfy topic `memres-e6ebdc70` every 30 min.

---

## v10 → v7 — folded into [`COMPREHENSIVE.md`](../../archive/COMPREHENSIVE.md) Part V

The v10 / v9 / v8 / v7 ledger entries — all KILLED, SUPERSEDED, or
audited — were folded into `archive/COMPREHENSIVE.md` Part V on
2026-05-01 per the "fold superseded entries when the ledger gets
crowded" convention below. Each subsection there preserves the
final result tables, verdicts, mechanism statements, and cited
filenames; the planning prose (hypotheses, decision triggers,
launch commands) was dropped because the corresponding cells have
all finished.

Last-eval / verdict summary (full tables in Part V):

| campaign | dates | verdict | leading checkpoint at end of campaign |
|---|---|---|---|
| **v10** (data-diversity 4B mega) | 2026-04-30 | KILLED mid-training; post-mortem audit identified P0/P1/P2/P3/P5 (five compounding causally-independent failures) → triggered v11 redesign | none promoted; mega corpus reused as `v11_mega` |
| **v9** (judge-competition curriculum) | 2026-04-30 | HEADLINE breakthrough: PA CB Δ_nm-m = +0.178 at step 1400 (24.4× v8b peak); v9c diverse confirmed data-diversity matters | `Runs/chain_v9_compete_lme_gh200/best` step 1400 (since superseded by v11g) |
| **v8** (PA-eval lens + RMSNorm fix) | 2026-04-29 → 30 | All three cells KILLED (v8a overfit @ 880, v8b overfit @ 2460, v8c stalled @ 2000); RMSNorm-on-readout retained; mechanism diagnoses (`attention_parity` collapse vs `simple_gate` explosion) survive | `Runs/chain_v8b_mixed_simplegate_rmsnorm/best` step 1200 (now archived) |
| **v7 P0** (compression curriculum + bias relax) | 2026-04-29 | All three cells KILLED 22:00 UTC after v8 lens-change; finding 1: `simple_gate` opens the channel where `attention_parity` cannot; finding 2: pure-P0 is a train/eval distribution mismatch trap | none promoted |

Standalone diagnostic JSONs preserved in `archive/eval/` and
`paper_artifacts/eval/diag_v7_*.json`. The v6 → v7 transition
narrative is in `COMPREHENSIVE.md` Part IV §"2026-04-29: v6 → v7
transition and the simple_gate finding".

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

1. Create / pick a `Scripts/train_*.sh` for the run. The script's
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
- Output dirs: always `Runs/<run_name>/{step-N, best/}`.
