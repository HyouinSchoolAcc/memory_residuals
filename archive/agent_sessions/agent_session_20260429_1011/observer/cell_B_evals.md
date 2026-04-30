# Cell B — `chain_v5_softembed_msc` (local GPU 0, embed, PG19+TV+MSC)

Eval set: `stage1_msc_val_s512.pt` (548 chains, 4645 sessions),
`eval_n_chains=32`, `eval_window=4`, `score_tail_frac=1.0`.
Soft init: `mem_bias=-4`, `recent_bias=+4`.

This cell most closely matches the v3 `chain_v3_softparity_full`
configuration (same `embed` extract, same soft `±4`), differing only
in: window_k (3 vs 8), corpus (+MSC), dropouts (mem 0.1, ctx 0.3),
carry_state, contrastive ramp 0.05→0.5, lr_backbone (2e-5 vs 3e-6
effective).

v3 reference at the same init: step 200 Δ_sh-m=-0.0001,
step 400 +0.0042, step 600 +0.0089, step 800 +0.0107,
step 1000 +0.0177; step 4400 +0.0379.

| step | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m | Δ_sh-m | Δ_or-m | interpretation |
|-----:|-------:|---------:|-----------:|----------:|-------:|-------:|-------:|----------------|
| 200 | 3.1183 | 3.1159 | 3.1188 | 2.9294 | -0.0024 | +0.0005 | -0.1888 | gate opened, three CEs distinct; ≈ v3 step 200. |
| 400 | 3.0863 | 3.0891 | 3.0870 | 2.9156 | +0.0027 | +0.0006 | -0.1708 | flat — v3 was +0.0042. **6.7× behind v3.** |
| 600 | 3.0987 | 3.0994 | 3.0997 | 2.9193 | +0.0007 | +0.0011 | -0.1793 | barely climbing — v3 was +0.0089. **8× behind v3.** |
| 800 | 3.1003 | 3.0981 | 3.0984 | 2.9230 | -0.0022 | **-0.0019** | -0.1773 | **sign-flipped negative.** v3 was +0.0107. Δ_nm-m also flipped (-0.0022). |
| 1000 | 3.0982 | 3.0962 | 3.0943 | 2.9206 | -0.0021 | **-0.0039** | -0.1777 | **negative deepening.** v3 was +0.0177. v5 cell B is ~0.022 nat behind v3 trajectory and trending the wrong way. |

## Trajectory shape

```
step:    200    400    600    800    1000
v3:    -0.0001 +0.0042 +0.0089 +0.0107 +0.0177  (monotone climb)
v5(B): +0.0005 +0.0006 +0.0011 -0.0019 -0.0039  (peak at step 600, then descent)
```

Δ_nm-m trajectory shows the same pattern: it was +0.0027 at step 400,
+0.0007 at step 600, then **flipped to negative (-0.0022 / -0.0021) at
steps 800 / 1000**. This means the model's ce on a true memory chain
is now *worse* than the same model with no memory at all on the same
held-out chain — i.e. memory is now *harmful* to next-token prediction
on the in-trainer eval. This is the paper-ruining failure mode.

mem CE itself dropped from 3.12 (step 200) to 3.09 (step 400-1000), so
the LM is fitting; the memory channel is the part that has gone bad.

Decision-trigger 1 (step ~1500: Δ_sh-m > +0.005) is **highly likely
to fail** for cell B given the current trajectory. The next EVAL is at
step 1200 (~6 min from now), then 1400 (~12 min from now); decision
trigger reads at step ~1500, so the binding observation is step 1400
or step 1600.
