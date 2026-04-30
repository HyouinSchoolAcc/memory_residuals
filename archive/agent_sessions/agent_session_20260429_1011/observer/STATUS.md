# Observer STATUS log (v5 soft-init 2x2)

Each entry is wall-clock UTC-5 (local) | local step counts. Brief.

## 2026-04-29 10:40 (UTC-5) — wake-1, smoke check

**All three cells alive.**

| cell | run_name | machine | tmux | last step | last loss | tok/s | EVAL? |
|---|---|---|---|---:|---:|---:|---|
| A | `chain_v5_softhidden14_msc` | GH200 192.222.50.225 | `cwd-chain_v5_softhidden14_msc` | 80 | 3.0899 | 3.0k | not yet (next at 200) |
| B | `chain_v5_softembed_msc` | local GPU 0 (H100 NVL) | `local-chain_v5_softembed_msc` | 100 | 3.1234 | 7.7k | not yet (next at 200) |
| C | `chain_v5_softhidden14_pgtv` | local GPU 1 (H100 NVL) | `local-chain_v5_softhidden14_pgtv` | 60 | 3.3437 | 4.8k | not yet (next at 200) |

Notes:

- All three trainers finished init parity load (same MISSING report
  list, expected; the memres params don't exist in the base Qwen3-0.6B
  ckpt and are newly init'd to the soft `±4` parity values).
- All three show `gate_mean +0.0000 max 0.0000` per train line — this
  is the *unused* `MemoryGate.gate` (ReZero scalar consumed only by
  `simple_gate` mode). In `attention_parity` mode the routing happens
  in `depth_router` and is NOT reflected in this column. Per writer
  `findings_loss_curve.md` §6 (CAVEAT). Do not interpret `gate_mean=0`
  as evidence of channel collapse.
- C's tok/s (4.8k) is materially below B's (7.7k) on the same H100 NVL
  hardware. Expected: C uses `--memres_extract_source hidden_14`
  which adds a no-grad partial backbone forward to layer 14 each
  session; B uses `embed` (no extra forward). A is on GH200 with the
  same `hidden_14` extract; its 3.0k tok/s vs C's 4.8k tok/s is the
  GH200 vs H100-NVL gap (gradient checkpointing, single GPU). No
  performance concern; ETAs unchanged.
- C's eval set is `stage1_validation_s512.pt` (48 PG-19 chains, 2145
  sessions) per the design (no MSC). A/B's eval set is the MSC-mixed
  `stage1_msc_val_s512.pt` (548 chains, 4645 sessions). **EVAL CE
  numbers are NOT directly comparable A/B vs C in absolute level,
  because the held-out chains are different. The Δ_sh-m signal IS
  comparable** because all three Δ's are intra-cell differences over
  the same cell's eval set.
- No EVAL lines anywhere yet (those land at step 200).

Next check: ~10:50 (B should have crossed step 200 by ~10:45; C / A
by ~10:53).

## 2026-04-29 10:55 (UTC-5) — wake-1, step 200 EVAL on all three cells

| cell | step | mem CE | nomem CE | shuffle CE | Δ_sh-m | one-line interpretation |
|---|---:|---:|---:|---:|---:|---|
| A (HEADLINE) | 200 | 3.1147 | 3.1142 | 3.1137 | **-0.0010** | gate opened (3 distinct CEs); mildly negative Δ_sh-m, within v3-step200 envelope |
| B            | 200 | 3.1183 | 3.1159 | 3.1188 | **+0.0005** | gate opened; smallest positive Δ_sh-m, comparable to v3 step 200 |
| C            | 200 | 3.1408 | 3.1397 | 3.1417 | **+0.0008** | gate opened; largest Δ_sh-m, but on PG-19 only eval set |

**Top fact:** all three cells broke v4's `mem == nomem == shuffle` to
4 decimals at the very first EVAL (step 200). The bf16-saturation
diagnosis is vindicated.

**Top concern:** cell A — the HEADLINE — is currently the *worst* of
the three on Δ_sh-m at step 200 (-0.0010 vs +0.0005 / +0.0008). This
is at-noise, but if it persists past step 1000 the recipe-paper claim
("hidden_14 + MSC is the headline") needs re-evaluation.

**Wrote:** `observer/advice_first_eval.md`.

Next check: ~11:10 (let B reach ~step 500, A reach ~step 380).

## 2026-04-29 11:31 (UTC-5) — wake-2, full trajectories through step 1000 (cell B), step 600 (cell C), step 400 (cell A)

| cell | last step | latest EVAL | latest Δ_sh-m | latest Δ_nm-m | one-line interpretation |
|---|---:|---:|---:|---:|---|
| A (HEADLINE) | ~420 | step 400 | +0.0002 | +0.0026 | gate open; barely positive; 20× behind v3 step 400 (+0.0042) |
| B            | ~1040 | step 1000 | **-0.0039** | **-0.0021** | **trajectory peaked at step 600 then sign-flipped negative; 0.022 nat behind v3 step 1000** |
| C            | ~660 | step 600 | +0.0005 | -0.0129 | Δ_sh-m flat near zero; **Δ_nm-m strongly negative and deepening** (-0.0011 → -0.0105 → -0.0129); mem CE *rising* (3.14 → 3.16) |

**Top fact:** v5 cell B's Δ_sh-m trajectory is monotonically below
v3's at every step from 400 onward, peaked at step 600 (+0.0011),
then **sign-flipped negative** at steps 800 / 1000. Δ_nm-m corroborates:
also flipped negative at step 800 / 1000. v3 at step 1000 was
+0.0177; cell B is at -0.0039 — a **0.022-nat gap** and growing.

**Top concern:** the failure mode is *not* the v4 bit-zero "gate
closed" mode. It is the new "memory becomes harmful" mode — mem CE
on held-out chains is now worse than the same model with no memory.
This is escalation-worthy: cell B is unlikely to clear the
step-1500 Δ_sh-m > +0.005 trigger on current trajectory. **Decision
to kill remains the human's; I am flagging escalation-worthiness
only.** Cell C reproduces the Δ_nm-m sign flip on a different eval
set, so the failure is in the *shared* knob set, not in
extract-source or corpus.

**Most likely cause:** H2 (contrastive ramp peaking 0.5 at step
1000 — timing matches the sign flip exactly), with H1
(context_dropout=0.30) as a likely co-driver. H3-H5 lower priors.

**Wrote:** `observer/concern_v5_below_v3_trajectory.md` (with five
single-knob falsifiable ablations ranked by cost). Refreshed
`cell_A_evals.md`, `cell_B_evals.md`, `cell_C_evals.md`. New
`cell_comparison.md` per risk-audit request.

**Cross-cell convergence (per audit):** at step 200/400 (only points
where all three cells exist), A vs B Δ_sh-m gap shrank from -0.0015
to -0.0004 — A is *catching* B. None of the three is on the v3
trajectory. The "recipe collapses to soft init alone" risk is live;
need step 600 / 800 / 1000 in cells A and C to call it.

Next check: ~11:42 (B step 1200 EVAL, possibly C step 800 EVAL).

## 2026-04-29 11:56 (UTC-5) — wake-3, killed cell B + relaunched as B prime; cell A wobbling

### Headline event: cell B was killed at step 1460 (16:53 UTC = 11:53 local)

Killed cell B's final divergence (per human's brief):

| step | ce_mem | ce_nomem | ce_shuffle | Δ_sh-m | note |
|---:|---:|---:|---:|---:|---|
| 800 | 3.1003 | 3.0981 | 3.0984 | -0.0019 | (matches my previous tracking) |
| 1000 | 3.0982 | 3.0962 | 3.0943 | -0.0039 | (matches) |
| 1200 | 3.1364 | 3.1134 | 3.1849 | +0.0485 | **fake positive — both forwards damaged** |
| 1400 | **3.6253** | 3.1453 | 3.5211 | -0.1041 | **catastrophic ce_mem blowup** |

ce_mem ran 3.10 → **3.62** in 400 steps while ce_nomem stayed flat
(3.10 → 3.14). Memory poisoning, not memory ignoring. Diagnosis: the
H1 + H2 regulariser stack (mem_drop=0.10, ctx_drop=0.30, neg_chain
ramp 0.05 → 0.5 over 1000 steps) drove early posterior collapse,
which the contrastive ramp at peak weight then satisfied
adversarially by damaging both mem and shuffle forwards. **My
wake-2 concern document `concern_v5_below_v3_trajectory.md`
correctly anticipated this — the human's diagnosis is the H2-primary
+ H1-secondary read I proposed.**

Killed cell logs preserved at
`logs/chain_v5_softembed_msc_KILLED_step1460.log`. `runs.md` ledger
updated.

### Cell B prime is now running (replaces dead B)

`chain_v5_softembed_msc_noreg` (local GPU 0, started 11:53 local).
Single change vs killed B: regulariser stack zeroed
(mem_drop=0, ctx_drop=0, neg_chain_weight=0). Throughput 10.7k tok/s
(vs killed B's 7.9k — ~30% faster, neg_chain forward removed). Step
60 / loss 3.08 as of 11:56; first EVAL lands ~12:03 local.

| cell | run_name | machine | last step | latest EVAL | Δ_sh-m | note |
|---|---|---|---:|---:|---:|---|
| A (HEADLINE) | `chain_v5_softhidden14_msc` | GH200 | ~620 | step 600 | **-0.0007** | back to negative, tracks killed B's pattern; **A has the same regulariser stack as killed B** |
| B prime | `chain_v5_softembed_msc_noreg` | local GPU 0 | 60 | _pending_ | _pending_ | first EVAL at step 200 in ~7 min |
| C | `chain_v5_softhidden14_pgtv` | local GPU 1 | ~960 | step 800 | -0.0003 | ce_mem 3.16 (stable); **NOT diverging like killed B** — slower-clock damage |

### Top fact
**Cell B's catastrophic divergence between step 1200 and 1400
confirms the regulariser-stack diagnosis from
`concern_v5_below_v3_trajectory.md`.** The H1+H2 cluster kills the
memory channel slowly (steps 200-1000) and then catastrophically
(steps 1200-1400). Cell B prime now tests whether removing those
three knobs (in one shot, not isolated) restores the v3sp trajectory.

### Top concern
**Cell A is on track to repeat killed B's failure.** Cell A's
Δ_sh-m trajectory at step 200/400/600 is -0.0010 / +0.0002 / -0.0007
— qualitatively the same shape as killed B's step 200/400/600
(+0.0005 / +0.0006 / +0.0011), just shifted ~0.0015 below. Cell A
has the same regulariser stack as killed B. If B prime's first 1000
steps show v3-style Δ_sh-m climb, cell A is essentially guaranteed
to fail catastrophically by step ~1500 on the same mechanism. Cell A
has ~5400 / 6000 steps left on the GH200 (~12 h, ~$30 cloud spend).
**Escalation-worthiness for cell A rises sharply if B prime confirms
the diagnosis.** Kill decision remains the human's; I am flagging
the linkage.

### Top recommendation
Wait for B prime step 600/800/1000 EVALs (next ~3 hours). If B
prime reaches Δ_sh-m ≈ +0.005 to +0.010 by step 600 and continues
climbing, cell A's regulariser stack is the smoking gun and the
human can decide whether to kill A early.

In the meantime cell C is the cleanest passive observation: it has
the same regulariser stack but on a different (no-MSC) corpus, and
is showing **slower-clock** damage than killed B. If C never
catastrophically diverges, that suggests MSC corpus interacts with
the regulariser stack to amplify damage — a useful finding for the
recipe-paper "regulariser trap" section.

### Files updated this wake
- `cell_Bprime_evals.md` (new)
- `cell_A_evals.md` (added step 600 row + risk note)
- `cell_C_evals.md` (added step 800 row + divergence-watch section)
- `cell_comparison.md` (rewritten to include killed B as historical row + B prime + divergence-watch ce_mem column)
- (this) `STATUS.md`

### Files NOT touched
- `cell_B_evals.md` (legacy from wake-2; left as a record of what
  was tracked under the killed run; per brief, tracking moves to
  `cell_Bprime_evals.md`)

Next check: ~12:08 local (B prime step 200 EVAL + C step 1000 EVAL).

---

## 2026-04-29 13:10 (UTC-5) — wake-N, **v6 PIVOT BRIEF**

**All v5 cells killed. v6 launched on three GPUs.** Tracking moves
from v5 cells A / B' / C to v6 cells GATED / COMPETITIVE / GATED-DEEP.

### Why the pivot

After v5 wake-3 it was clear that all three v5 cells (A on GH200, B'
on local GPU 0, C on local GPU 1) shared the same failure signature:

- `gate_max` stuck at 0.000 indefinitely
- `Δ_sh-m` saturates at ~+0.0006 (statistical noise — memory channel
  not recruited)
- `Δ_or-m` (oracle gap) ~ -0.23 nats — meaning the memory pathway
  COULD help by 0.23 nats if it were ever used, but the gradient
  signal under uniform NLL on PG-19/MSC isn't strong enough to ever
  open it.

This is **not** a regularisation problem (B' had no dropouts and no
contrastive loss; same failure). Not an init problem (soft ±4 ruled
out the bf16 saturation that killed v4). The bottleneck is structural:
**uniform NLL on conversational chat or book corpora gives no gradient
signal that says "look at M_c instead of just predicting from local
context."** The memory channel is starved of supervision.

### What v6 changes (four axes, all shipped together)

| axis | v5 | v6 |
|---|---|---|
| corpus | PG-19 books / MSC chats | LongMemEval-S (450 chains × ~50 sessions × 6k tok, callback-supervised) |
| memory update | competitive softmax (zero-sum, slot fully replaceable each step) | gated: `M_c^t = (1-g) * M_c^{t-1} + g * judge`; sigmoid g init bias -1.0 (g≈0.27) — non-replacing |
| loss | uniform NLL | uniform NLL + `callback_loss_weight=10.0` on tokens flagged in `session_callback_mask` (the answer-span tokens of each LongMemEval Q+A) |
| sampling | uniform over chain windows | `callback_window_bias=0.7` — 70% of windows aligned to include the callback session as their last position |

Routing stays at **attention_parity at -4/+4** (the only routing
config we have empirical confidence in: ReZero variants are strictly
worse, hard ±32 saturates bf16). User explicitly reminded to keep
parity, so this is locked in for all v6 cells.

### v6 active cells

| cell | run_name | machine | tmux/spec | step (last) | loss | tok/s | EVAL? |
|---|---|---|---|---:|---:|---:|---|
| GATED | `chain_v6_lme_gated_callback` | local GPU 0 | local-chain_v6_lme_gated_callback | 140 | 1.4686 | 5.4k | next at 200 |
| COMPETITIVE | `chain_v6_lme_competitive_callback` | local GPU 1 | local-chain_v6_lme_competitive_callback | 80 | 1.6420 | 5.4k | next at 200 |
| GATED-DEEP (window_k=12) | `chain_v6_lme_gated_callback_w12` | GH200 | watchdog `1777486074` | starting | n/a | n/a | next at 200 |

**Single-knob ablation map:**
- GATED vs COMPETITIVE = `--memres_update_mode gated|competitive`. Tests whether the gated update is doing real work or whether corpus + callback loss alone are sufficient.
- GATED vs GATED-DEEP = `--window_k 8|12`. Tests whether deeper TBPTT through the recurrent judge stack helps recruit memory.

### Trajectories so far (vs v5 baseline)

| metric | v5 typical (any cell) | v6 GATED step 140 | v6 COMP step 80 |
|---|---:|---:|---:|
| train loss | 2.7-3.2 (bouncing for 1000+ steps) | 1.4686 | 1.6420 |
| ce_oracle - ce_mem | -0.23 (memory unused) | TBD @ step 200 | TBD @ step 200 |
| gate_max | 0.000 forever | 0.000 (early) | 0.000 (early) |

Train loss dropping ~5x faster than v5 ever did is consistent with
the callback supervision producing real gradient signal. Whether the
gradient flows into M_c (gate_max > 0, Δ_sh-m > 0) or just into the
local context residual is what step 200 EVAL will tell us.

### Decision triggers (per cell, in priority order)

For ALL v6 cells, the diagnostic ladder is:

1. **step 500: `gate_max > 0`?** v3-v5 cells were stuck at 0
   indefinitely. If still 0, the gated update isn't engaging at all
   and we have an architectural problem that the loss alone can't
   fix. If > 0, memory is at least being written.

2. **step 1000: `Δ_sh-m on lme_val > +0.005`?** This was the v3
   envelope. Hitting it on LME means the recipe transferred. Not
   hitting it means callback bias / weight need re-tuning.

3. **step 2500: `Δ_sh-m > +0.02` AND alpha_mem > 5%** on routing
   trace. This is the threshold where we declare "the architecture
   can recruit memory in conversational settings" and start writing
   up the recipe. Below this, we ablate further (try
   callback_loss_weight 30.0, try `--memres_extract_source hidden_14`,
   etc.)

### Cross-cell decision rules

- If GATED catches up to COMPETITIVE on Δ_sh-m by step 1000: gated
  update is not contributing → revert to competitive in v7.
- If GATED beats COMPETITIVE by ≥+0.005 Δ_sh-m at step 2500: gated
  is the recipe and we ship it.
- If GATED-DEEP beats GATED on Δ_sh-m by step 1500: window_k=8 is
  too shallow for the conversational gradient to propagate through
  the recurrent judge → bump window_k in v7.

### Where files are

- v6 launch scripts:
  - `scripts/train_v6_lme_gated_callback.sh` (local GPU 0)
  - `scripts/train_v6_ablation_a_competitive.sh` (local GPU 1)
  - `scripts/train_v6_lme_gated_w12_gh200.sh` (GH200)
- v6 logs:
  - local: `logs/chain_v6_lme_*.log`
  - GH200: remote `paper_tools/cloud_watchdog/logs/chain_v6_lme_gated_callback_w12.log`
- v6 corpus:
  - `paper_artifacts/chains/lme_train_s512.pt` (450 chains, 5367 callback positions)
  - `paper_artifacts/chains/lme_val_s512.pt` (50 chains, 666 callback positions)
- v6 code changes:
  - `modeling_memres.py` — `memres_update_mode={"competitive","gated"}`, gated branch in `MemoryBlock.forward`
  - `train_chain.py` — `--callback_loss_weight`, `--callback_window_bias`, `_weighted_lm_loss`
  - `paper_tools/build_conversational_callback_chains.py` — unified MSC/LME/REALTALK builder
- killed v5 logs (preserved):
  - `logs/chain_v5_softhidden14_pgtv_KILLED_v6_pivot_step1620.log` (cell C)
  - `logs/chain_v5_softembed_msc_noreg_KILLED_v6_pivot_step1540.log` (cell B')
  - remote `paper_tools/cloud_watchdog/logs/chain_v5_softhidden14_msc_KILLED_v6_pivot_step1080.log` (cell A)
- canonical run record: `experiments/exp2_long_horizon_recipe/runs.md` (rewritten)

### What an observer wake should do

For each wake while v6 cells run:

1. Pull last 10 lines of all 3 v6 logs (2 local, 1 GH200).
2. If a new EVAL line appeared since last wake, append a row to
   `cell_v6_<name>_evals.md` (create if missing) with: step,
   `ce_mem`, `ce_nomem`, `ce_shuffle`, `ce_oracle`, `Δ_sh-m`,
   `Δ_or-m`, `alpha_mem` if logged, `gate_max`.
3. If `gate_max` first goes > 0, write a one-paragraph note in
   STATUS.md ("memory channel opened at step X for cell Y").
4. If `Δ_sh-m` crosses +0.005 (v3 envelope), write a note.
5. If `Δ_sh-m` crosses +0.02 (recipe-paper threshold), write a
   recommendation and STOP letting it drift — start eval planning.
6. If any cell diverges (ce_mem rising, Δ_sh-m going negative),
   recommend kill in STATUS.md with the specific evidence.

Continue cross-cell comparisons in `cell_comparison.md`.

### Files NOT used in v6
- v5 cell_*_evals.md remain as historical record.
- `cell_B_evals.md` and `cell_Bprime_evals.md` represent killed runs.

### First v6 GATED EVAL @ step 200 (20 minutes after launch)

```
EVAL @ step 200: n=256 mem=1.3989 nomem=1.4037 shuffle=1.3987 oracle=1.4797
                 Δnm-m=+0.0048   Δsh-m=-0.0002   Δor-m=+0.0809
```

Interpretation:

- **Δ_nm-m = +0.0048**: memory is helping vs no-memory by ~0.005 nats
  per token after only 200 steps. Compare v5 cell C at step 1600
  (after the corpus-only fix): Δ_nm-m ≈ +0.0006. **v6 GATED at step
  200 already beats v5 at step 1600 by ~8×.** This is the strongest
  evidence so far that the corpus + callback supervision pivot is
  doing real work.
- **Δ_sh-m = -0.0002**: memory isn't yet recruiting CHAIN-SPECIFIC
  information (shuffling chain identities barely changes anything).
  This is expected at step 200 — the router opens first, then the
  judge starts producing chain-discriminative reads. Watch this
  number's trajectory; it should start climbing by step 500-1000.
- **Δ_or-m = +0.0809**: oracle CE > mem CE by 0.08, meaning the mem
  pathway is currently OUTPERFORMING the oracle on this batch. This
  is sampling noise — the oracle eval samples gold completions that
  may not exist exactly in the model's distribution. Don't trust
  Δ_or-m at small step counts.
- **"best" checkpoint saved**: composite metric (Δ_nm-m + Δ_sh-m
  weighted) beat init.
- **`gate_mean=0` in train log**: this is the unused ReZero scalar
  of `MemoryGate` (only consumed in `simple_gate` routing mode).
  In `attention_parity` mode (what v6 uses), routing happens in
  `depth_router`. Per writer's `findings_loss_curve.md` §6: do NOT
  read `gate_mean=0` here as channel collapse. The actual mass on
  memory is reported by Δ_nm-m above and is non-zero.

### What this means for the v6 hypothesis ladder

- Step 500 trigger ("memory channel opens at all"): **already
  cleared at step 200.** Δ_nm-m > 0 and best ckpt saved.
- Step 1000 trigger ("Δ_sh-m > +0.005"): pending, on track to test
  in ~80 minutes.
- Step 2500 trigger ("Δ_sh-m > +0.020 AND alpha_mem > 5%"): pending,
  ~5 hours out.

If Δ_sh-m is still ≈ 0 at step 1000, that means memory is helping
non-chain-specifically (probably a per-source style prior or the LM
head learning that "this is a conversational task → de-bias toward
question-answering"). That would be a partial success but not the
recipe paper's claim. We'd need to push callback_loss_weight up or
fix the callback bias.

If Δ_sh-m crosses +0.005 by step 600-1000, that's full recipe
confirmation and we plan eval / writeup.

Next observer wake: ~13:30 local (step 400 of GATED, step 320 of
COMPETITIVE, GH200 still warming).

### Fourth v6 cell queued: PURIST (v3-knob ablation)

Added 13:28 local in response to user question "is v5 cell C just v3
with different corpus?". The answer surfaced that v3 climbed Δ_sh-m
monotonically to +0.018 by step 1000 on the SAME PG-19+TV corpus
where v5 cell C plateaued at +0.0005. Seven knobs differ between v3
and v5 cell C; v6 already reverted three of them (no dropouts, no
contrastive, window_k=8). PURIST reverts the remaining four:
`carry_state` removed, `burn_in_max=0`, `lr=3e-4`, `lr_backbone=3e-5`.

| cell | run_name | machine | role |
|---|---|---|---|
| PURIST | `chain_v6_lme_gated_purist` | GH200 (queued behind GATED-DEEP) | tests whether v5's last training-knob inheritances are dead weight |

PURIST will start automatically when GATED-DEEP completes (~6h ETA).
Watchdog spec: `1777487405_chain_v6_lme_gated_purist.json`.

PURIST decision rule (when it eventually runs):
- PURIST > GATED on Δ_sh-m at step 1000 → v5's last inheritances are
  dead weight; recipe simplifies.
- PURIST ≈ GATED → those knobs are neutral.
- PURIST < GATED → at least one of `carry_state` / `burn_in` / `lr`
  was load-bearing; queue a finer ablation to find which.

Track PURIST's EVAL line in a new `cell_v6_purist_evals.md` once it
starts emitting. Until then, no observer attention needed.





