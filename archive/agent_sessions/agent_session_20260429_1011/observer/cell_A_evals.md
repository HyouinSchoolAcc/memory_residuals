# Cell A — `chain_v5_softhidden14_msc` (GH200, hidden_14, PG19+TV+MSC)

Eval set: `stage1_msc_val_s512.pt` (548 chains, 4645 sessions),
`eval_n_chains=32`, `eval_window=4`, `score_tail_frac=1.0`.
Soft init: `mem_bias=-4`, `recent_bias=+4`.

v3 reference (`chain_v3_softparity_full`, `embed`, soft `±4`,
window_k=8, no dropouts, no contrastive ramp):
- step 200: Δ_sh-m = -0.0001
- step 400: Δ_sh-m = +0.0042
- step 1000: Δ_sh-m = +0.0177
- step 4400: Δ_sh-m = +0.0379

Decision triggers: step 1500 Δ_sh-m > +0.005 ; step 4000 > +0.020.

| step | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m | Δ_sh-m | Δ_or-m | interpretation |
|-----:|-------:|---------:|-----------:|----------:|-------:|-------:|-------:|----------------|
| 200 | 3.1147 | 3.1142 | 3.1137 | 2.9282 | -0.0005 | -0.0010 | -0.1865 | gate opened, three CEs distinct. v3 step 200 was -0.0001. Mildly worse than v3. |
| 400 | 3.0924 | 3.0950 | 3.0926 | 2.9204 | +0.0026 | +0.0002 | -0.1720 | barely above zero. v3 step 400 was +0.0042. **20× behind v3.** Δ_nm-m is now positive (+0.0026) — better than B's step-400 (+0.0027, ≈ same). |
| 600 | 3.1076 | 3.1071 | 3.1068 | 2.9251 | -0.0005 | **-0.0007** | -0.1825 | **back to negative.** Three CEs are nearly identical again (3.1076 / 3.1071 / 3.1068). v3 step 600 was +0.0089; **A is 13× behind v3 and trending wrong way**. Pattern: small positive → near-zero → small negative. **Tracks killed cell B's step 200/400/600 (+0.0005 / +0.0006 / +0.0011) but A is below B even there.** Cell A has the same regulariser stack as killed B — the cell B prime ablation will tell us whether to expect cell A to follow killed B into catastrophic divergence. |

Cell A is at step ~600; on the GH200 at 3.0k tok/s × ~8s/step pace,
step 800 EVAL lands ~27 min from now (~12:23 local).

## Risk note (added 11:56)

Cell A shares the regulariser stack (mem_drop=0.10, ctx_drop=0.30,
neg_chain ramp 0.05 → 0.5 over 1000 steps) that the killed cell B
diagnosis indicts. Cell A's step 200/400/600 trajectory
(-0.0010 / +0.0002 / -0.0007) qualitatively mirrors killed cell B's
step 200/400/600 (+0.0005 / +0.0006 / +0.0011), just with a baseline
offset (A is consistently ~0.0015 below B at matched step). If the
regulariser-stack diagnosis is correct, **cell A is on track to
catastrophically diverge between step 1200 and step 1400** the same
way killed B did. Cell A still has ~5400 / 6000 steps to run on the
GH200 (~12 h, ~$30 of cloud budget).

The diagnostic that will resolve this fastest is **cell B prime's
trajectory through step 600/800/1000.** If B prime climbs the v3sp
ladder, the regulariser hypothesis is confirmed and cell A is
escalation-worthy as well. If B prime is also stuck or negative, the
cause is elsewhere and cell A is no more at risk than the others.

I am not recommending kill of cell A. The next ~3 hours of evidence
from cell B prime will sharpen that question.

## Comparison to cells B and C at matched steps

At step 200 / 400 (the only steps that lap-overlap):

| step | A Δ_sh-m | B Δ_sh-m | C Δ_sh-m | A vs B (same eval set) |
|-----:|---------:|---------:|---------:|-----------------------|
| 200  | -0.0010  | +0.0005  | +0.0008  | A is 0.0015 behind B |
| 400  | +0.0002  | +0.0006  | +0.0003  | A is 0.0004 behind B |

A is closing the gap to B by step 400 (was -0.0015, now -0.0004),
which means cell A is now *not* the worst of the three on the
strictly-comparable A vs B contrast. Cell A's trajectory is
qualitatively the same as B's pre-step-600 — small positive that's
not climbing. Cell B has since gone negative; whether cell A follows
the same track in another 200-400 steps is the binding question.
