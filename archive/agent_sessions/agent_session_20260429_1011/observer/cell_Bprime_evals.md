# Cell B prime — `chain_v5_softembed_msc_noreg` (local GPU 0, embed, PG19+TV+MSC)

Replaces killed cell B (`chain_v5_softembed_msc`). The killed cell's
trajectory and final state are recorded in
`logs/chain_v5_softembed_msc_KILLED_step1460.log` and the previous
version of `cell_B_evals.md` (now superseded; killed run lives in
`runs.md`).

Eval set: `stage1_msc_val_s512.pt` (548 chains, 4645 sessions),
`eval_n_chains=32`, `eval_window=4`, `score_tail_frac=1.0`.
Soft init: `mem_bias=-4`, `recent_bias=+4`.

## Differences from killed cell B (single change is the regulariser stack)

| knob | killed B | B prime |
|---|---|---|
| `--memory_dropout` | 0.10 | **0.0** |
| `--context_dropout` | 0.30 | **0.0** |
| `--neg_chain_weight` | 0.5 (ramp 0.05 → 0.5 over 1000 steps) | **0.0** |
| extract source | embed | embed |
| corpus | PG19+TV+MSC | PG19+TV+MSC |
| init | soft `±4` | soft `±4` |
| window_k | 3 | 3 |
| carry_state | on | on |
| lr / lr_backbone | 2e-4 / 2e-5 | 2e-4 / 2e-5 |

So B prime is **one ablation against killed B**: it removes the entire
regulariser stack (the H1+H2 cluster from `concern_v5_below_v3_trajectory.md`)
in one shot. It does not isolate H1 from H2 — it tells us "regulariser
stack vs no regulariser stack" only. If B prime recovers the v3
trajectory, it confirms the *combined* regulariser stack is the
dominant blocker; isolating H1 vs H2 individually would need follow-up.

## Knob parity with v3sp reference

B prime now differs from v3sp only in:

| knob | v3sp | B prime | mechanistic effect |
|---|---|---|---|
| corpus | PG19+TV (no MSC) | PG19+TV+MSC | adds dialogue training data |
| window_k | 8 | 3 | shorter unrolls |
| carry_state | off | **on** | TBPTT-style state carry |
| lr_backbone | 3e-6 (effective) | 2e-5 | 7× faster backbone movement |

So if B prime hits v3sp's trajectory by step 1000, the *delta* in the
recipe paper attributable to "MSC corpus + carry_state + lr_backbone +
window_k=3" is ≈ zero (or small). The story collapses to "soft init
alone is the active ingredient; everything else is regularisation
debt." If B prime is *better* than v3sp by step 1000, MSC + carry +
shorter window are net positive even without the regulariser stack.

## Throughput

step 60 already at 10.7k tok/s (vs killed B's 7.7-7.9k). Confirms
removing the neg_chain forward saves ~30% wallclock. ETA ~3.8 h is
realistic; expected finish ~20:40 UTC (~15:40 local).

## Trajectory (so far)

Started 11:53 local. As of 11:56 local, **step 60, loss 3.08, no
EVAL yet** (first EVAL lands at step 200, ETA ~10 more minutes).
soft-init signal at step 20 is loss 4.49, grad_norm 70 — matches
killed B's step 20 (4.69, 71.5) — gradient flow looks identical.

| step | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m | Δ_sh-m | Δ_or-m | interpretation |
|-----:|-------:|---------:|-----------:|----------:|-------:|-------:|-------:|----------------|
| _no EVALs landed yet — next at step 200 ~12:03 local_ | | | | | | | | |

## What I will record at step 200

The single most informative reading:

- **If Δ_sh-m at step 200 ≈ -0.0001 (matches v3sp)** → soft-init
  sanity reproduces; we now wait for step 1000.
- **If Δ_sh-m at step 200 is positive but small (~+0.001)** → similar
  to killed B at step 200 (+0.0005); the regulariser-removal hasn't
  kicked in yet (regularisers act over many steps, not single
  forward passes), watch step 600/800 instead.
- **If Δ_sh-m at step 200 is more negative (≤ -0.005)** → unusual;
  would suggest the regulariser stack was *masking* a different
  underlying problem, not causing it. Lower-prior outcome.

The binding readings are step 600 (where killed B peaked), step 800
(where killed B sign-flipped), and step 1000 (where killed B was
-0.0039 and v3sp was +0.0177). A monotone climb through those
matching v3sp's shape is the cleanest possible "regulariser stack
was the dominant cause" outcome.
