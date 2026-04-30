# Concern: memory channel is closed across the entire v4 headline run

**Severity:** high. This is the same failure mode v3 had (alpha_mem
nearly zero), reproduced in the v4 hidden_14 + MSC run, and the
mechanism is consistent with depth-router softmax saturation under
the hard ±32 init.

## Direct evidence (same as writer)

Across all 27 in-trainer EVAL lines from step 200 → step 5400 in
`monitor/training_log_1015.log`:

```
mem == nomem == shuffle (to 4 decimals)
Δ_nm-m = +0.0000
Δ_sh-m = +0.0000
```

ce_mem drifts 3.1231 → 3.0894 (≈0.034 nat) — that's the LM head
adapting to the PG-19+TV+MSC token mix, not memory recruitment.

Δ_or-m = -0.18 throughout, identical-to-noise across 5400 steps:
the architecture is leaving 0.18 nat on the table relative to the raw
4-session concat oracle.

## Mechanism (corroborates writer's analysis with one caveat)

The writer's softmax math is correct. With `mem_bias=-32` on memory
and `recent_bias=+32` on the most-recent source, the routing-softmax
mass on memory at init is

```
α_mem(0) = exp(-32) / [exp(+32) + (N-1) * exp(0) + exp(-32)]
        ≈ exp(-32) / exp(+32)
        = exp(-64)
        ≈ 1.6e-28
```

which is ~20 orders of magnitude below bf16's representable smallest
positive (≈6e-8). Backprop through `softmax_i * (1 - softmax_i)` gives
∂L/∂mem_bias proportional to α_mem ≈ 1.6e-28, which is bf16 zero.
**The contrastive loss therefore cannot move `mem_bias` off init.**
v3 with ±4 had α_mem(0) ≈ 1.7e-4 (using the full softmax including the
recent +4 source; the overnight writeup's `exp(-8)/56 ≈ 6e-5` is a
rough lower-bound that under-counts the recent_bias normalisation but
is the right order of magnitude). v3's gradient is non-vanishing
through bf16 and α_mem(4400) ≈ 4.7e-4 (per `routing_v3sp_val.json`).

The writer's analysis is correct on these mechanics. ✅

## Caveat the writer should note (and does not)

`gate_mean +0.0000 max 0.0000` printed every 20 steps in the train log
is reading `model.memory_gate.gate`
(`MemoryGate`, `modeling_memres.py:262`). That parameter is **only
consumed by the `simple_gate` memres mode**
(`modeling_memres.py:1077–1095`). In `attention_parity` mode
(line 1136 onwards) the forward pass uses `route_if_needed()` →
`depth_router.route()` and **never reads `memory_gate.gate`**. So the
log column is the unused ReZero scalar sitting at its zero init for
the whole run; it is not a "the gate has not moved" signal — there is
no gate to move in this mode.

This does not affect the writer's conclusion (which is supported by
the EVAL `Δ_sh-m = +0.0000` line, not by the gate column), but the
manuscript should not cite the gate column as evidence. The honest
in-trainer signal is `mem == nomem == shuffle`.

## What we still do not know (need routing_trace on a v4 ckpt)

The in-trainer EVAL only reports `Δ` to 4 decimals. We have not
directly measured α_mem on a v4 checkpoint. The expectation under the
hard-init theory is α_mem(v4, step 5000) ≈ 1e-26..1e-15 (a few orders
of magnitude above 1.6e-28 init due to bf16 noise and the `M_in` /
`M_judge` warmup); under any *competing* theory (e.g. dropouts caused
the collapse), α_mem(v4) might be 1e-5 to 1e-4 (small but non-zero)
and we'd need a different fix.

A 30-second `paper_tools/routing_trace.py` run on `step-5000` would
distinguish these. I am not running it right now because the GH200 is
busy with the live training and I'm read-only on remote; the v4
post-train pipeline will run it on `step-6000` automatically.

## What the next iteration should *also* prove

If the writer's diagnosis is correct, the next iteration should
demonstrate **routing-trace α_mem > 1e-3** at the equivalent of
step 4400 of v3, not just `Δ_sh-m > 0` in-trainer. The routing-trace
α_mem is the more falsifiable operational definition of "memory
channel open" than the in-trainer `Δ` (which is dominated by the
backbone's adaptation to corpus mix).
