# v20 architecture — copy-paste spec

A self-contained description of the `v20a` / `v20b` cell architecture so it
can be ported faithfully to a new backbone. References are to
`src/modeling_memres.py` (the model) and the launch scripts
`Scripts/train_v20{a,b}_*.sh` (the training-side glue). Everything below
is shared by both v20 cells; v20a vs v20b only differ in two
hyperparameters and are listed at the end.

---

## 0. What we're doing and what memory residuals is trying to accomplish

**The goal.** Give a pretrained LLM a *persistent, learned, native*
memory — a fixed-size state matrix `M_c` that the model writes to as
each session ends and reads from on every forward pass — so that the
model can recall facts, decisions, and commitments across an
arbitrarily long sequence of sessions without ever re-attending to the
raw history and without bolting on a retrieval system.

**The thesis.** Long-horizon recall in LLMs should not be solved by
RAG, by re-attention over a growing context window, or by a hand-coded
controller. It should be solved by *adding the right architectural
primitive to the pretrained transformer* and training that primitive
end-to-end on the same loss the LM was already trained on (next-token
NLL on multi-session data). If that works, you get long-horizon recall
"for free" at inference: a constant `O(K·d)` memory and a constant
read/write cost per session, independent of how many sessions came
before.

**Concretely, what success looks like.** A frozen Qwen3-0.6B backbone
+ a small set of memory parameters (`MemoryBlock`, `MemoryReadout`,
`BlockAttnResRouter`, ~30M params total) trained on synthetic
multi-session chains where session N+1 contains a "callback" that can
only be answered by recalling content from session 1. The success
metric is `evidence_lift` — the next-token CE on the callback
*decreases* when `M_c` is supplied vs when the model sees only the
callback session, on a *frozen backbone*, on a corpus where the
backbone could not have memorised the answer (`synthd5_random_codes`,
random-string evidence). That is end-to-end, leak-free evidence that
memory is doing real work.

**What's hard about it (and why this took 20 cells).** All of the
hard problems are training-dynamics problems, not capacity problems.
The architecture was specified in the v3 era; the rest of the campaign
has been about getting the joint-training optimisation to actually
*use* it instead of collapsing onto one of several degenerate fixed
points:

- The **uniform-symmetric fixed point** of the writer — i.i.d.-init
  slot queries are permutation-equivariant, so the loss is symmetric
  and any all-slots-identical configuration is a gradient stationary
  point. Fixed *structurally* by orthogonal init + per-slot positional
  addresses + slot-axis softmax (slot attention).
- The **router-closes / writer-stays-random** chicken-and-egg —
  router closes around `α_mem ≈ 0` because the writer is content-blind
  → writer gets attenuated gradient → writer stays random. Fixed by
  the `α_mem` floor load-balance auxiliary.
- The **content-blind read-side** — under LM-NLL alone the LM head
  can satisfy callback CE with a chain-identity hash and the readout
  never learns to decode chain-specific content from `M_c`. Fixed by
  the F3 read-side probe head (v18+) and multi-layer readout
  refinement (v19+).
- The **m_t-dominates-LM** failure when the read-side is over-
  pressured (v19b: `‖m_t‖/‖embed‖ = 12.7`, `evidence_lift = −0.016`).
  This is what v20 is calibrating: probe weight vs floor weight, the
  two knobs that control how hard the read-side is pushed.

v20 is the active recipe-calibration cell; the architecture below is
the candidate to ship if either v20a or v20b clears
`evidence_lift > +0.05` end-to-end with `§5 ttt_lift_vs_floor` MIXED
or POS.

---

## 1. Why memory residuals at all (architectural properties)

Standard transformers handle long context by *re-attending* over raw
tokens. That is O(N²) in attention and forgets nothing voluntarily —
context = whatever fits in the window. We want a *constant-sized*
memory matrix `M_c ∈ R^{K×d}` that the model writes to and reads from
**natively**, with no retrieval index, no separate controller, no
hand-engineered gating heuristic. Concretely:

1. **Constant cost.** `M_c` has K=128 slots regardless of how many
   prior sessions have been compressed into it. Recall over a 100-turn
   dialogue costs the same as recall over a 2-turn dialogue.
2. **Native writes.** Each session's tokens are compressed into K slots
   by an attention primitive (Stage 1 *Extraction*) and then merged
   with the prior `M_c` by a second attention primitive (Stage 2
   *Judging*) that explicitly competes "keep the old slot" against
   "overwrite with the new candidate" — the *Forgetting Defense*.
3. **Native reads.** Each token in the current session cross-attends
   `M_c` to produce a per-position readout `m_t ∈ R^{S×d}`, which is
   shape-compatible with any attention layer's output and is injected
   into the existing residual stream of a pretrained transformer
   through the **Block Attention Residuals** depth router (Du et al.).
4. **Bit-exact init parity.** The augmented model is, by construction,
   numerically equal to the bare backbone at step 0. Memory only
   contributes once training has moved a gate / router off zero. So we
   can drop this onto any pretrained Qwen3 (or anything Qwen3-shaped)
   without breaking it.

The architecture, the failure modes, and the campaign log live in
`memory_residuals.pdf` (spec), `atn_residuals.pdf` (depth router), and
`README.md` §"Architectural priors" (settled findings — read those
before deviating). v20 is the current candidate recipe under that
spec.

---

## 2. v20 architecture — concise spec

Backbone: **Qwen3-0.6B**, frozen (`lr_backbone=0`, `--freeze_backbone`).
All learning happens in the memres subsystem and the F3 probe head.

### 2.1 Top-level forward (one training session t)

Given history token ids and current-session token ids:

```
C_t        = bare-backbone hidden state at layer 14   # no-grad partial fwd
M_new      = MemoryBlock.extract(C_t)                  # Stage 1, (B, K, d)
M_c^t      = MemoryBlock.judge(M_c^{t-1}, M_new)       # Stage 2, (B, K, d)
m_t        = MemoryReadout(inputs_embeds, M_c^t)       # (B, S, d)
hidden     = BlockAttnRes(inputs_embeds, m_t, layers)  # depth-routed forward
logits     = lm_head(RMSNorm(hidden))
```

Recurrent across sessions: `M_c^t` is the input to the next session's
`judge`. TBPTT window `k=3` (3 sessions retained in the graph).

### 2.2 MemoryBlock — writer (Section 2.1 of the paper)

K = 128 slots, hidden size = backbone d.

**Stage 1 — Extraction.** Inputs: `C_t = bare_backbone(input_ids)[layer_14]`,
detached, then RMSNormed (`memres_extract_input_norm=True`; otherwise
‖C‖ at layer 14 ≈ 50 and the writer's gradients explode).

```
C_n   = RMSNorm(C_t)                                  # (B, N, d)
M_in  = M_in_param + M_in_pos_param                   # (K, d), broadcast to B
E     = CrossAttn(M_in, C_n)                          # 1 layer (extraction_depth=0)
M_new = E                                             # (B, K, d)
```

`CrossAttn` = standard `(W_Q, W_K, W_V)` no-bias projections + softmax
on the inputs axis + value matmul. **No QK-LayerNorm in extraction.**

**Stage 2 — Judging.** SlotAttentionWriter (Locatello 2020), 3
iterations, **with QK-LayerNorm**.

```
M_judge       = M_judge_param + M_judge_pos_param         # (K, d), broadcast
P_judge       = concat(M_c^{t-1}, M_new, dim=1)           # (B, 2K, d)
slots         = M_judge
for it in range(3):
    Pn        = RMSNorm_in(P_judge)
    K_proj    = RMSNorm_k(W_K(Pn))                        # QK-LayerNorm
    V_proj    = W_V(Pn)
    sn        = RMSNorm_slot(slots)
    Q_proj    = RMSNorm_q(W_Q(sn))                        # QK-LayerNorm
    scores    = Q_proj @ K_proj.T * d^{-0.5}              # (B, K, 2K)
    attn      = softmax(scores, dim=-2)                   # softmax over SLOTS
    attn      = attn / attn.sum(dim=-1, keepdim=True)     # per-slot renorm
    updates   = attn @ V_proj                             # (B, K, d)
    slots     = GRUCell(updates, slots)                   # weight-tied across slots
M_c^t = RMSNorm_judge(slots)                              # final norm
```

Critical detail: softmax is along the **slots axis** (not the inputs
axis as in vanilla cross-attn). This is what gives the Forgetting
Defense — slots compete for input mass, so two slots cannot collapse to
the same content without losing it. Per-slot weighted-mean
normalisation makes each row a proper convex combination of the inputs.

`update_mode=gated` is set but **the external sigmoid `write_gate` is
bypassed** for slot-attention writers (the GRUCell already implements
per-slot keep/write; stacking the sigmoid was strictly harmful, see
README prior #6). `write_gate` parameters stay allocated for checkpoint
compatibility.

**Symmetry breaks (necessary; uniform fixed point is structural —
README prior #5).**
- `M_in`, `M_judge`: learnable `(K, d)` params, **orthogonal init**
  (`memres_queries_init=orthogonal`, init in fp32 then cast). Rows are
  pairwise orthogonal so slot i's query is structurally distinct from
  slot j's at step 0.
- `M_in_pos`, `M_judge_pos`: learnable `(K, d)` per-slot positional
  addresses, initialised to a deterministic Fourier pattern
  (`sin/cos(k / 10000^{2i/d})`) scaled by `d^{-0.5}`
  (`memres_slot_positional=True`). Added to the queries before the
  W_Q/W_K/W_V projections.

### 2.3 MemoryReadout — reader (Section 2.2, Eq. 6 + v19 refinement stack)

Per-position cross-attention from the current session's token
embeddings into `M_c`, **with depth=4 residual refinement layers**
stacked on the base layer (Perceiver-style iterative refinement).

```
# Base layer (Eq. 6).
Q   = W_Q(X)                                          # X = inputs_embeds
K_  = W_K(M_c)
V   = W_V(M_c)
m   = softmax(Q @ K_.T * d^{-0.5}, dim=-1) @ V        # (B, S, d)
m_t = out_norm(m)                                     # RMSNorm, weight init = 0.05

# 4 refinement layers (i = 0..3).
for i in range(4):
    q_in   = X + m_t                                  # condition on X *and* current m_t
    Q_i    = refine_W_Q[i](q_in)
    K_i    = refine_W_K[i](M_c)
    V_i    = refine_W_V[i](M_c)
    attn_i = softmax(Q_i @ K_i.T * d^{-0.5}, dim=-1)
    m_t    = m_t + refine_out_norm[i](attn_i @ V_i)   # residual; norm weight init = 0.05
```

The **0.05 RMSNorm-weight init** is load-bearing: it scales the readout
output 20× down so `‖m_t‖/‖embed‖` lands in a healthy range
(~3.85 at v18a). Without it the readout dominates the residual stream.

### 2.4 BlockAttnResRouter — depth-wise routing pool

Mode: `attention_parity`. Pool stores **cumulative hidden-state
checkpoints** (not deltas), so the most-recent source is exactly the
standard residual stream input.

For each routed sublayer i (one per attention or MLP transform; total
= 2L − 1 routing steps for an L-layer backbone, the very first
attention sublayer reads `inputs_embeds` directly):

```
pool = [m_t, h_0, h_block_1, ..., h_block_{n-1}, h_partial]   # b_{-1} prepended
       # m_t is included only if memory is present
scores = einsum("d,bsnd->bsn", w_i, RMSNorm(stack(pool)))     # w_i: (d,), zero-init
scores[..., 0] += mem_bias[i]                                 # b_{-1} bias
scores[..., -1] += recent_bias[i]                             # most-recent bias
alpha = softmax(scores, dim=-1)                               # (B, S, n_src)
h_pre = einsum("bsn,bsnd->bsd", alpha, stack(pool))
```

`memres_num_blocks=4`. Block boundaries partition the 2L sublayers
into 4 balanced groups; once a block completes its accumulated
checkpoint is appended to `pool`. Pool size is bounded by N+2
regardless of L.

**Init biases (v20-specific, soft parity).** Override the parity-init
defaults of (-32, +32) with:
- `router_mem_bias_init = 0`     (memory has no negative bias from step 0)
- `router_recent_bias_init = +4` (most-recent source mildly preferred)

This is the "soft ±4" regime from the v3 pair recipe — empirically
beats hard ±32 because the saturated softmax of hard parity has no
gradient on the pseudo-queries until a slow bias warmup completes.

`MemoryGate` (per-sublayer ReZero scalar) is allocated but unused on
the forward path in `attention_parity` mode (only `simple_gate` mode
uses it).

### 2.5 ReadoutProbeHead — F3 training-only auxiliary head

Independent head, **training-only**, never affects eval. Its purpose
is to install a chain-specific gradient channel through
`MemoryReadout`'s own W_Q/W_K/W_V (the §5 capacity probe falsified the
joint-trained read-side: 6/6 NEG with M_c TTT-able and read-side
frozen). Without it the writer trains content-blind under LM-NLL only.

```
class ReadoutProbeHead(nn.Module):
    norm = RMSNorm(d)
    proj = Linear(d, vocab_size, bias=False)          # NOT tied to embed_tokens

    def forward(m_t_at_cb):                           # (B, d)
        return proj(norm(m_t_at_cb))                  # (B, V)
```

At each training step the trainer locates the *first answer-token
position* in the callback session, reads `m_t` at that position from
the **same** `MemoryReadout` the LM used in the forward pass, runs it
through the probe head, and adds `cur_w * CE(probe_logits, answer_tid)`
to total loss. Gradient path: probe → MemoryReadout.W_Q/W_K/W_V → M_c
→ judge → extract.

`proj` is **not** tied to `embed_tokens` — tying would route probe
gradient back into the embedding table and pollute the LM head's input
distribution. At eval time the LM head is the sole reader of `m_t`, so
the probe never affects reported callback CE (anti-circularity).

Probe weight schedule: linear warmup from 0 to `--readout_probe_loss_weight`
over `--readout_probe_warmup_steps=200` steps.

### 2.6 alpha_mem floor — load-balance auxiliary

MoE-style load balance on the depth router's mass on `b_{-1}`. Without
it the router collapses to ignoring memory (`alpha_mem ≈ 1e-3`)
because local features dominate.

```
loss_floor = mean_l ReLU( target − mean_{B,S}(alpha_mem_l) )
total_loss += aux_weight * loss_floor
```

Sums over routed sublayers `l`. ReLU side: penalises only the
*downward* violation; once `mean(alpha_mem_l) ≥ target` the term has
zero gradient and the main LM-NLL is unimpeded.

### 2.7 Initialisation summary (the parts HF doesn't init for you)

`Qwen3PreTrainedModel._init_weights` handles `nn.Linear`,
`nn.Embedding`, RMSNorm, and rotary. The custom params and primitives
must be initialised by `_init_memres_params` (called from
`post_init`):

| param | init |
|---|---|
| `MemoryBlock.M_in`, `M_judge` | orthogonal (fp32 → cast); rows pairwise orthogonal |
| `MemoryBlock.M_in_pos`, `M_judge_pos` | Fourier `sin/cos(k/10000^{2i/d}) * d^{-0.5}` |
| `MemoryReadout.out_norm.weight` | `0.05` (scalar fill) |
| `MemoryReadout.refine_out_norm[i].weight` | `0.05` (each, same calibration) |
| `BlockAttnResRouter.w[i]` | zeros (uniform softmax over pool at step 0) |
| `BlockAttnResRouter.mem_bias[i]` | `router_mem_bias_init` = `0.0` |
| `BlockAttnResRouter.recent_bias[i]` | `router_recent_bias_init` = `+4.0` |
| `MemoryGate.gate[i]` | `0.0` (unused on the forward path in attention_parity) |
| `MemoryBlock.write_gate.weight/bias` | zero / -1 (allocated but unused under slot_attention) |
| `SlotAttentionWriter.gru` | `gru.reset_parameters()` (HF doesn't match `nn.GRUCell`) |
| `ReadoutProbeHead.norm.weight` | `1.0` (HF default) |
| `ReadoutProbeHead.proj.weight` | HF default Linear init (independent of `embed_tokens`) |

`MemoryReadout.W_V` stays at the default normal init (non-zero); init
parity is preserved by the RMSNorm-weight 0.05 (small magnitude) and
the router's `mem_bias=0 + recent_bias=+4` (soft parity, ~exp(4)/N
mass on the most-recent source at step 0).

---

## 3. v20 hyperparameters

Shared by v20a and v20b unless marked.

| group | flag | value |
|---|---|---|
| backbone | `--preset` | `qwen3-0.6b-large` |
| | `--freeze_backbone` `--lr_backbone 0` | yes |
| writer | `--memres_num_vectors` (default in code) | `K=128` |
| | `--memres_extract_source` | `hidden_14` |
| | `--memres_extract_input_norm` | on |
| | `--memres_extraction_depth` (default 0) | `0` |
| | `--memres_writer_kind` | `slot_attention` |
| | `--memres_slot_attention_iters` | `3` |
| | `--memres_queries_init` | `orthogonal` |
| | `--memres_slot_positional` | on |
| | `--memres_judge_qk_layernorm` | on |
| | `--memres_update_mode` | `gated` (external sigmoid bypassed) |
| reader | `--memres_readout_norm_init` | `0.05` |
| | `--memres_readout_depth` | `4` |
| router | `--memres_mode` | `attention_parity` |
| | `--memres_num_blocks` (default 4) | `4` |
| | `--router_mem_bias_init` | `0` |
| | `--router_recent_bias_init` | `4` |
| | `--memres_gate_init` | `0.0` |
| F3 probe | `--readout_probe_enabled` | on |
| | `--readout_probe_warmup_steps` | `200` |
| | `--readout_probe_loss_weight` | **v20a: 0.3 / v20b: 1.0** |
| α-floor | `--alpha_mem_floor_aux_weight` | **v20a: 0.5 / v20b: 0.05** |
| | `--alpha_mem_floor_target` | **v20a: 0.10 / v20b: 0.05** |
| writer warmup | `--writer_warmup_steps` | `0` (disabled; structural breaks carry the load) |
| training | `--lr` | `1e-4` |
| | `--steps` `--warmup` | `1500` `200` |
| | `--max_norm` | `1.0` |
| | `--batch_size` `--grad_accum` | `4` `2` |
| | `--window_k` | `3` (TBPTT length) |
| | `--memory_dropout` `--context_dropout` | `0.10` `0.05` |
| | `--callback_loss_weight` | `5.0` (5× weight on callback tokens) |
| | `--mask_evidence_session_loss` `--mask_padding_loss` | on |
| | `--curriculum_evidence_bias` | `1.0` |
| | `--score_tail_frac` `--burn_in_max` | `1.0` `0` |
| | `--neg_chain_weight` | `0.0` (no InfoNCE; v14 finding: harmful with slot writer) |
| safety | `--kill_on_memory_collapse` (after step 200) | on (exit 42 if pair/self < 0.01 or m_t/embed < 0.01) |

### v20a vs v20b — the single-variable ablation

Both share the architecture and *every* training knob above except:

| | v20a | v20b |
|---|---|---|
| `--readout_probe_loss_weight` | **0.3** | 1.0 |
| `--alpha_mem_floor_aux_weight` | 0.5 | **0.05** |
| `--alpha_mem_floor_target` | 0.10 | **0.05** |

Each is an isolation test of one suspected cause of v19b's
"memory dominates LM in the wrong direction" failure (v19b had
`evidence_lift = −0.016` with `‖m_t‖/‖embed‖ = 12.7`):

- **v20a** keeps the strong floor (0.5 / 0.10) but cuts probe
  weight 70%. If the *probe* was over-pressuring the readout into
  encoding chain-identity at the callback position, this should
  fix end-to-end `evidence_lift` while preserving §5 capacity.
- **v20b** keeps the standard probe (1.0) but reverts the floor to
  v18a strength (0.05 / 0.05). If the *floor* was opening the
  router faster than the readout could specialise, this should
  fix it instead.

Both succeed → redundant fixes; pick the better §5.
Only one succeeds → that's the operating-point variable.
Neither → F3+depth interaction is wrong; v21 candidates are ReZero-init
on `refine_W_V`, separate gradient clip on probe loss, or decay probe
weight to zero after step 500.

---

## 4. Porting checklist (drop into a new backbone)

1. Subclass the new backbone's `Config` to add the `memres_*`,
   `router_*`, and probe fields (see `Qwen3MemResConfig.__init__` for
   the canonical surface).
2. Pull `MemoryBlock`, `SlotAttentionWriter`, `CrossAttention`,
   `MemoryReadout`, `ReadoutProbeHead`, `BlockAttnResRouter`,
   `MemoryGate`, and `_init_memres_params` from
   `src/modeling_memres.py` verbatim. They are pure `nn.Module`s with
   no Qwen3-specific dependencies except `Qwen3RMSNorm` (any RMSNorm
   works; the `weight=1.0` HF default is what `_init_weights` assumes).
3. In the new model, replicate the `attention_parity` forward pass:
   replace each layer's `h = h + attn(LN(h))` and `h = h + mlp(LN(h))`
   with `h_pre = router.route(idx, pool)` then `h_post = h_pre +
   delta`, accumulating cumulative checkpoints into the pool (see
   `Qwen3MemResModel.forward` lines under `else: # Attention-pool`).
4. Wire `compute_memory(history_ids) → M_c` and the per-position
   readout `m_t = MemoryReadout(inputs_embeds, M_c)` exactly as in
   `Qwen3MemResModel.forward`.
5. Add the F3 probe loss block to your trainer (see `train_chain.py`
   §"v18 / F3" — it locates the callback position, recomputes `m_t`
   for selected rows, runs `readout_probe_head`, adds
   `weight * CE(logits, answer_tid)` to total loss).
6. Add the alpha_mem floor block (collect `alpha_trace` from the
   forward pass, `aux = mean_l ReLU(target − mean(alpha_l))`, weight
   by `aux_weight`).
7. Run `tools/init_parity_test.py` against the new backbone with
   `M_c=None`. It should report bit-exact logits vs the bare backbone
   (the soft-parity ±4 regime gives ~1e-3 logit drift; tighten to
   hard parity ±32 if you need bit-exactness for a unit test).
8. Smoke-train for 100 steps and confirm: `‖m_t‖/‖embed‖` is finite
   and < 50; `pair/self` is non-zero; no NaN gradients on
   `MemoryReadout.W_V` or `MemoryBlock.M_in`.
