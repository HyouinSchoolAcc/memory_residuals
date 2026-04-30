# Cell comparison — Δ_sh-m and Δ_nm-m trajectories across A / B prime / C

Updated wake-3 (2026-04-29 11:56 UTC-5).

A and B prime share eval set `stage1_msc_val_s512.pt` (548 chains,
4645 sessions). Their Δ_sh-m and Δ_nm-m are directly comparable.

C uses `stage1_validation_s512.pt` (PG-19 only, 48 chains, 2145
sessions). Its CE absolute level differs from A / B prime; trajectory
shape is comparable, magnitudes are not.

The killed cell B trajectory is included as a strikethrough row to
mark the regulariser-stack failure pattern; B prime now replaces it.

## Δ_sh-m

| step | A Δ_sh-m | killed B (historical) | B prime Δ_sh-m | C Δ_sh-m | A − B′ | converging? |
|-----:|---------:|:---------------------:|---------------:|---------:|------:|:-----------:|
| 200  | -0.0010  | _+0.0005_ | _pending_ | +0.0008  | _pending_ | A < C |
| 400  | +0.0002  | _+0.0006_ | _pending_ | +0.0003  | _pending_ | A ≈ C |
| 600  | -0.0007  | _+0.0011_ | _pending_ | +0.0005  | _pending_ | A < C |
| 800  | _pending_ | _-0.0019_ | _pending_ | -0.0003 | _pending_ | — |
| 1000 | _pending_ | _-0.0039_ | _pending_ | _pending_ | _pending_ | — |
| 1200 | _pending_ | _+0.0485 (fake — both forwards damaged)_ | _pending_ | _pending_ | _pending_ | — |
| 1400 | _pending_ | _-0.1041 (catastrophic)_ | _pending_ | _pending_ | — |
| **v3sp ref** | -0.0001 / +0.0042 / +0.0089 / +0.0107 / +0.0177 (steps 200/400/600/800/1000) | — | — | — | — | — |

Killed B and v3sp diverged at step 400 already (B was +0.0006, v3sp
+0.0042 — 7× behind). By step 1000 the gap was 0.022 nat. The kill
diagnosis pinpoints the H1 + H2 regulariser stack (mem_drop + ctx_drop
+ neg_chain ramp) as the dominant cause; B prime tests that diagnosis.

## Δ_nm-m (mem helps vs no mem)

| step | A Δ_nm-m | killed B Δ_nm-m | B prime Δ_nm-m | C Δ_nm-m |
|-----:|---------:|:--------------:|---------------:|---------:|
| 200  | -0.0005  | _-0.0024_ | _pending_ | -0.0011 |
| 400  | +0.0026  | _+0.0027_ | _pending_ | -0.0105 |
| 600  | -0.0005  | _+0.0007_ | _pending_ | -0.0129 |
| 800  | _pending_ | _-0.0022_ | _pending_ | -0.0104 |
| 1000 | _pending_ | _-0.0021_ | _pending_ | _pending_ |

Cell A's Δ_nm-m at step 600 went back to -0.0005 from the step-400
positive +0.0026, mirroring killed B's late-stage flip (just earlier
in step count). Cell C's Δ_nm-m is the most strongly negative of all
cells — peaked at -0.0129 at step 600, partial recovery to -0.0104
at step 800.

## ce_mem absolute (divergence-watch column)

The killed cell B's diagnostic signature was ce_mem running away
(3.10 → 3.62) while ce_nomem stayed flat (3.10 → 3.14). Tracking that
across cells:

| step | A ce_mem | killed B ce_mem | B prime ce_mem | C ce_mem |
|-----:|---------:|:--------------:|---------------:|---------:|
| 200  | 3.1147   | _3.1183_ | _pending_ | 3.1408 |
| 400  | 3.0924   | _3.0863_ | _pending_ | 3.1383 |
| 600  | 3.1076   | _3.0987_ | _pending_ | 3.1650 |
| 800  | _pending_ | _3.1003_ | _pending_ | 3.1594 |
| 1000 | _pending_ | _3.0982_ | _pending_ | _pending_ |
| 1200 | _pending_ | _3.1364_ | _pending_ | _pending_ |
| 1400 | _pending_ | _**3.6253**_ | _pending_ | _pending_ |

Killed B's ce_mem stayed stable through step 1000 (3.10), wobbled at
1200 (3.14), then **catastrophically jumped to 3.62 at step 1400**.
A is at 3.11 / 3.09 / 3.11 (stable). C is at 3.14 / 3.14 / 3.17 / 3.16
(stable, different baseline). **No cell currently shows the killed-B
runaway signature.** Continue monitoring at each EVAL.

## Convergence question

**Are A and C moving identically?** (B prime hasn't produced an EVAL
yet so it's not in the comparison.) At steps where both have data:

| step | A Δ_sh-m | C Δ_sh-m | A on its eval set vs C on its eval set |
|-----:|---------:|---------:|---------------------------------------|
| 200  | -0.0010  | +0.0008  | C ahead by 0.0018 (different eval sets, qualitative) |
| 400  | +0.0002  | +0.0003  | nearly identical |
| 600  | -0.0007  | +0.0005  | C ahead by 0.0012 |

A and C are **qualitatively similar**: both stuck near zero with
small fluctuations across step 200/400/600. C's Δ_sh-m has not
sign-flipped in a sustained way; A's has wobbled around zero. Neither
is on the v3 trajectory. **None of the three working cells is
producing the recipe-paper headline signal yet.**

If B prime climbs the v3 ladder while A and C stay near zero, the
2x2 contribution survives in a *narrow* form: "MSC + hidden_14 needs
the regulariser-free training schedule; if you keep the regularisers
the new corpus and extract pieces yield no Δ_sh-m signal." That's a
publishable but smaller story than the original recipe paper claim.

If B prime is also stuck near zero, the regulariser stack was *not*
the dominant cause and another knob (carry_state / window_k / lr_bb)
is implicated. That's the F3 / F4 / F5 escalation path from
`concern_v5_below_v3_trajectory.md`.

## Update cadence

Next EVAL refreshes expected:

- B prime step 200 EVAL: ~12:03 local (~7 min from this wake)
- C step 1000 EVAL: ~12:08 local (~12 min)
- A step 800 EVAL: ~12:23 local (~27 min)
