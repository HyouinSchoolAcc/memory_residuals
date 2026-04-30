# Cell C — `chain_v5_softhidden14_pgtv` (local GPU 1, hidden_14, PG19+TV no MSC)

Eval set: `stage1_validation_s512.pt` (48 PG-19 chains, 2145 sessions),
`eval_n_chains=32`, `eval_window=4`, `score_tail_frac=1.0`.
Soft init: `mem_bias=-4`, `recent_bias=+4`.

This is the only cell whose eval set excludes MSC dialogue, so its CE
absolute level will differ from A/B. Δ_sh-m is still meaningful but is
measured on books only.

| step | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m | Δ_sh-m | Δ_or-m | interpretation |
|-----:|-------:|---------:|-----------:|----------:|-------:|-------:|-------:|----------------|
| 200 | 3.1408 | 3.1397 | 3.1417 | 2.9396 | -0.0011 | +0.0008 | -0.2012 | gate opened, three CEs distinct. |
| 400 | 3.1383 | 3.1278 | 3.1386 | 2.9311 | -0.0105 | +0.0003 | -0.2072 | Δ_sh-m flat; **Δ_nm-m sharply negative (-0.0105)** — nomem strongly beats mem already. |
| 600 | 3.1650 | 3.1520 | 3.1655 | 2.9468 | -0.0129 | +0.0005 | -0.2181 | Δ_sh-m flat; Δ_nm-m **deepening negative (-0.0129)**. mem CE is *rising* (3.14 → 3.16) — eval is degrading. |
| 800 | 3.1594 | 3.1490 | 3.1591 | 2.9423 | -0.0104 | **-0.0003** | -0.2170 | **Δ_sh-m crossed zero negative.** Δ_nm-m partially recovers (-0.0129 → -0.0104), mem CE drops back slightly (3.165 → 3.159). Comparable to killed B's step 800 (Δ_sh-m = -0.0019) but C's deterioration is shallower — confirms user's "stable, not diverging — different trajectory shape than killed cell B". C is degrading slower than killed B did. |

## Cell C divergence-watch (per wake-3 brief)

User asked specifically: monitor cell C for the killed-B divergence
signature (`ce_mem` running away while `ce_nomem` stays flat).

Killed cell B's signature: ce_mem 3.10 (step 1000) → 3.14 (step 1200)
→ **3.62 (step 1400)**, while ce_nomem stayed at 3.10 → 3.11 → 3.14.
Sign: ce_mem grew by 0.52 in 400 steps while ce_nomem grew by 0.04.

Cell C ce_mem trajectory:
- step 200: 3.1408
- step 400: 3.1383 (-0.003 vs step 200)
- step 600: 3.1650 (+0.027 vs step 400)
- step 800: 3.1594 (-0.006 vs step 600)

**Cell C is NOT showing the killed-B divergence signature.** ce_mem
moves in the third decimal (3.14-3.17 range) and is currently
*decreasing* between step 600 and 800. This is regulariser-stack
damage of the "stuck near zero with mildly harmful memory" type, not
the "memory poisoning prediction" type that killed B. If anything is
going wrong in C it is on a slower clock than in killed B.

Continue monitoring step 1000 / 1200 of cell C. If ce_mem suddenly
jumps by > +0.3 nat between any two consecutive EVALs while ce_nomem
stays stable, raise the cell-C-diverging concern immediately.

## Trajectory shape

C's Δ_sh-m is essentially flat at ~+0.0005 across 200/400/600. Below
the v3 reference at *every* step, but C is on a different eval set
than v3, so the comparison is not perfectly clean. Compared to *cell
B's* trajectory (also v5, also against v3): C's Δ_sh-m has not
flipped sign yet, but Δ_nm-m has, and is deepening.

mem CE rising from 3.14 → 3.16 between step 200 and 600 is the
clearest red flag in cell C. The LM is *not* fitting better on the
held-out chain when memory is present, even as train loss is
descending normally. The Δ_or-m is also *deepening* (-0.20 → -0.22),
i.e. the oracle gap is *widening*.

Concerns: the same failure-mode signal that's killing cell B (memory
becoming actively harmful) appears to be present in cell C too, just
delayed by ~400 steps. Both cells share the dropout / contrastive
ramp / window_k=3 / carry_state / lr_backbone=2e-5 knob set; only
extract source and corpus differ. **This raises the probability that
the failure is in the *shared* knob set (hypotheses 1-3 / 5), not in
extract or corpus.**
