# Findings — `alpha_mem` (memory routing mass) under hard ±32 init

## Direct in-trainer evidence (no extra script needed)

The trainer's per-eval line reports four cross-entropies on a 124-position
held-out batch under {mem, nomem, shuffle, oracle} memory injection.
At hard parity init (`mem_bias=-32, recent_bias=+32`):

```
EVAL @ step 200..5400: mem == nomem == shuffle to 4 decimals
```

This is the operational definition of "the depth router has zero
attention mass on `b_{-1}`": injecting any `M_c` (true / shuffled /
empty) does not change the next-token logits beyond bf16 round-off.

That's a **stronger** statement than the v3 routing-trace finding.
On v3 soft-parity (`±4`) at step 4400 we measured
`α_mem^true = 4.7e-4` with a `+4.79%` mem-vs-shuffle gap on PG-19 val
(see `paper_artifacts/eval/overnight_traces_writeup.md`). With hard
parity (`±32`) and an additional contrastive ramp, the gap is
**zero to 4 decimals on the in-trainer eval.**

## Why hard ±32 will not move

The depth router pool has N+1 sources (delta sources b_0..b_n
plus memory b_{-1}). At init the pseudo-query (post-LayerNorm running
hidden) feeds a softmax with biases:

- `b_{-1}` (memory): `mem_bias = -32`
- `b_{n-1}` (most recent): `recent_bias = +32`
- others: 0

Logits-into-softmax: `[-32, 0, 0, ..., 0, +32]`.
Softmax mass on `b_{-1}` ≈ `exp(-32 - 32) / (N + exp(-32 - 32))` ≈
`1.6e-28`. Backward gradient through that softmax onto `mem_bias` is
proportional to that mass — so `1.6e-28`, deep below bf16's
representable range (`≈6e-8`). The optimizer step on `mem_bias` is a
literal numerical zero.

A negative-chain contrastive loss differentiates "true M_c vs shuffled
M_c" by passing M_c through the readout into the residual stream
through the depth router. If the router puts `1.6e-28` mass on memory,
the contrastive loss has `1.6e-28` of gradient signal flowing back
to the router. **The contrastive ramp cannot rescue this.**

## What v3 did differently — and why it works ≈

v3 `chain_v3_softparity_full` uses `mem_bias=-4, recent_bias=+4`:

- Init mass on memory: `exp(-4-4) / (N + exp(-4-4))` ≈
  `exp(-8)/56 ≈ 6.2e-5`.
- Backward gradient through the softmax: `~6.2e-5`, well within
  bf16's reps. Optimizer can move the bias.

That's why v3 soft-init reached 7.6× the analytic floor (4.7e-4 at
step 4400); the v4 hard-init has not moved the gate in 5400 steps.

## Direct corroboration plan (deferred)

The cleanest empirical confirmation is to checkpoint-load step-5000
and run `paper_tools/routing_trace.py` to dump α_mem per sublayer.
That script exists in v3 (was used for the overnight sweep). For the
current session I am not launching new GPU work; the in-trainer
mem == nomem == shuffle to 4 decimals at *every* eval is sufficient
evidence that α_mem is essentially zero. We will run the routing trace
on the final step-6000 ckpt as part of the post-train pipeline.

## What this means for paper 2

This is now the headline empirical claim of the *recipe* paper:

> The init-parity-preserving design (hard ±32) is bit-exact at step 0
> by construction, but the gradient through the depth-router softmax
> at that init is below bf16 representability. The contrastive loss
> cannot drive memory recruitment off this saturation. We must either
> (a) use a soft init (±4) where contrastive gradient is non-vanishing,
> or (b) add an explicit auxiliary loss that bypasses the softmax
> saturation on memory routing.

The v4 hard-init result is **not a wasted run**: it is the empirical
case for *why the next iteration must use soft init*. It pins down the
failure mode that the v3 soft-init implicitly avoided but never named.

## Files referenced

- `~/memory_residuals/paper_tools/cloud_watchdog/logs/chain_v4_hidden14_msc.log`
- `paper_artifacts/eval/overnight_traces_writeup.md`
- `paper_artifacts/eval/chain_v3_training_summary.md`
- `experiments/exp1_drop_in_primitive/manuscript.tex` (Table 6 routing-mass)
