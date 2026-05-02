"""
Memory Residuals — faithful to the paper's architecture.

Two key architectural advances over standard Qwen3:

1. MemoryBlock — Two-Stage QKV Competition (Section 2.1):

   Stage 1 (Extraction, Eq. 1 baseline; Eqs. 3-4 with L_E > 0):
       E^(0) = CrossAttn(M_in, C_t)                                  # (K, d)
       E^(ℓ) = E^(ℓ-1) + CrossAttn^(ℓ)(E^(ℓ-1), C_t)  for ℓ = 1..L_E # (K, d)
       M_new = E^(L_E)

   Stage 2 (Judging, Eqs. 2/5):
       P_judge = [M_c^{t-1} || M_new]                                 # (2K, d)
       M_c^t   = Softmax(M_judge W_Q^judge (P_judge W_K^judge)^T / sqrt(d)) P_judge W_V^judge

   The softmax across the 2K dimension creates zero-sum competition
   between old memory and new candidate (the Forgetting Defense).  Every
   intermediate state in the extraction lives in R^{K x d}, so the
   residual refinement is well-typed and L_E = 0 recovers Eq. 1.

2. Block AttnRes Depth Routing with Memory Injection (Section 2.2):

   Readout (Eq. 6) — per-position cross-attention over M_c with the
   current session's token embeddings as queries:
       m^t = Softmax(X W_Q^read (M_c^t W_K^read)^T / sqrt(d)) M_c^t W_V^read
       m^t in R^{S x d}  — matches the shape of any attention layer output.

   We adopt the Block variant of Attention Residuals [du2026attention]:
   the attention/MLP transforms are treated as depth-wise layers and
   partitioned into N blocks.  Within each block n, sublayer-output deltas
   are summed into a
   running partial sum b_n^i; once a block completes, b_n is pushed to
   the inter-block pool.  Memory is registered as an external foundational
   source b_{-1} := m^t, parallel to the embedding source b_0 := h_1.

   Per-position pool for the i-th layer of block n (Eq. 9):
       V_{n,i} = [b_{-1}, b_0, b_1, ..., b_{n-1}]                if i = 1
               | [b_{-1}, b_0, b_1, ..., b_{n-1}, b_n^{i-1}]     if i >= 2

   Routed input (Eq. 10):
       h_{n,i} = sum_{v in V_{n,i}} alpha_{v -> (n,i)} * v
       alpha_{v -> (n,i)} = phi(w_{n,i}, v) / sum_{v'} phi(w_{n,i}, v')
       phi(q, v) = exp(q^T RMSNorm(v))

   Pseudo-queries w_{n,i} are zero-initialized so initial weights are
   uniform across sources (per AttnRes §5).  The pool size is bounded by
   N+2 regardless of L, so memory and depth-wise communication scale as
   O(Nd) rather than O(Ld).

   When no memory is provided (m_t is None) b_{-1} is dropped from the
   pool, and the model is a standard Block AttnRes transformer.  Setting
   memres_num_blocks = 2 * num_hidden_layers recovers Full AttnRes; setting
   memres_num_blocks = 1 collapses to standard residuals (with b_{-1}
   and b_0 isolated as the only non-trivial sources).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.utils import auto_docstring, can_return_tuple

try:
    from transformers.utils.generic import merge_with_config_defaults
except ImportError:

    def merge_with_config_defaults(fn):
        return fn


try:
    from transformers.utils.output_capturing import capture_outputs
except ImportError:

    def capture_outputs(fn):
        return fn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# Canonical MemRes routing modes.  See `_normalise_memres_mode` for the
# legacy-string translation table that keeps old checkpoints / scripts
# working.
MEMRES_MODES: tuple[str, ...] = (
    # "Toy-example" path: keep the pretrained Qwen3 residual flow exactly
    # as-is and add the memory readout m^t into each sublayer input through
    # a per-sublayer ReZero-style scalar gate g_l (init 0).  The depth-wise
    # routing pool is unused on the forward path; only the gate, M_c, and
    # the readout participate in gradient flow.  Bit-exact init parity by
    # construction.  This is the simplification we run as a baseline.
    "simple_gate",
    # "Full implementation" path, no init parity: the depth-wise routing
    # pool stores per-sublayer *deltas* and m^t is one source among many.
    # Each routed sublayer's input is a softmax-weighted average over the
    # delta pool.  This matches the original AttnRes paper, but cannot
    # recover the bare residual stream from any one-hot softmax setting,
    # so step-0 logits are perturbed (~34 max|Δ| on Qwen3-0.6B) and the
    # optimiser has to absorb that perturbation before memory learning
    # starts.
    "attention_base",
    # "Full implementation" path, with init parity: the depth-wise routing
    # pool stores *cumulative* hidden-state checkpoints (b_0 = h_0,
    # b_k = h_k, partial = h_current) and the router is initialised with
    # mem_bias = -32 / recent_bias = +32 so the softmax is one-hot on the
    # most-recent slot.  Combined, this makes the routed sublayer input
    # equal to the bare residual stream input at step 0.  The cost is a
    # saturated softmax that needs bias warm-up to learn to mix non-recent
    # sources.
    "attention_parity",
)


_VALID_WRITER_KINDS = ("original", "slot_attention", "slot_attention_full")


def _normalise_memres_mode(
    memres_mode: str | None,
    block_attnres_parity_init: bool | None,
) -> str:
    """Translate any (mode, parity_init) tuple -- legacy or new -- to a
    canonical mode string in `MEMRES_MODES`.

    Legacy mapping (kept so existing config.json / shell scripts work):

        memres_mode="residual"                              -> "simple_gate"
        memres_mode="block_attnres", parity_init=False      -> "attention_base"
        memres_mode="block_attnres", parity_init=True       -> "attention_parity"

    A canonical input is returned unchanged.  When `memres_mode` is one of
    the new canonical names but `block_attnres_parity_init` is also given
    (e.g. an old script paired with a new mode flag), `parity_init` is
    silently ignored because the new mode already disambiguates.
    """
    if memres_mode in MEMRES_MODES:
        return memres_mode
    if memres_mode == "residual":
        return "simple_gate"
    if memres_mode == "block_attnres":
        # Preserve the legacy CLI default: when block_attnres was paired
        # with an unspecified parity_init flag (None), the trainers used
        # to default the flag to True.  Old shell scripts passing only
        # `--memres_mode block_attnres` therefore expected the parity
        # variant -- we keep that mapping here.
        if block_attnres_parity_init is None:
            block_attnres_parity_init = True
        return (
            "attention_parity"
            if bool(block_attnres_parity_init)
            else "attention_base"
        )
    raise ValueError(
        f"Unknown memres_mode {memres_mode!r}; expected one of "
        f"{MEMRES_MODES} (or legacy 'residual' / 'block_attnres')."
    )


class Qwen3MemResConfig(Qwen3Config):
    """Qwen3Config extended with Memory Residuals hyper-parameters."""

    model_type = "qwen3_memres"

    def __init__(
        self,
        memres_num_vectors: int = 128,
        memres_extraction_depth: int = 0,
        memres_num_blocks: int = 4,
        memres_mode: str = "attention_parity",
        block_attnres_parity_init: bool | None = None,
        # Optional explicit overrides for the BlockAttnResRouter's init
        # biases.  When None (default), the router falls back to mode-derived
        # defaults: attention_parity uses (-32, +32) for bit-exact init
        # parity at the cost of a saturated softmax; attention_base /
        # simple_gate uses (-8, 0).  Setting these to less extreme values
        # (e.g. -4 / +4) trades a small amount of init parity for a
        # softmax that has gradient signal on the pseudo-queries from
        # step 0, which lets the router actually learn to recruit memory.
        router_mem_bias_init: float | None = None,
        router_recent_bias_init: float | None = None,
        # Source of the per-token representation C_t fed to the extraction
        # stack.  Two modes:
        #   "embed"      -- C_t = embed_tokens(input_ids).  This is the
        #                   bag-of-token-embeddings extraction; the
        #                   compressor never sees contextual information
        #                   about the session and can only learn to attend
        #                   to a weighted average of token embeddings.
        #                   Empirically this collapses to style/lexical
        #                   memory (high in-trainer Δ_sh-m on PG-19 books,
        #                   null effect on dialogue).  Default historically
        #                   for back-compat.
        #   "hidden_<L>" -- run a no_grad bare-backbone forward and use
        #                   hidden_states[L] as C_t.  This gives the
        #                   compressor a contextualised representation of
        #                   the session (syntax, anaphora, entity binding)
        #                   so it can extract semantic content rather than
        #                   bag-of-tokens.  Cost: one extra backbone
        #                   forward per session boundary (~50% per-step
        #                   overhead on Qwen3-0.6B).
        memres_extract_source: str = "embed",
        # Recurrent memory update mode (Stage 2 of MemoryBlock):
        #   "competitive" -- M_c^t = judge(M_c^{t-1}, M_new) where judge is
        #                    a single softmax over [M_c^{t-1} || M_new].
        #                    Each slot of M_c^t is a softmax-weighted sum
        #                    of one row from M_c^{t-1} OR one row from M_new,
        #                    so a slot can be FULLY replaced with new info
        #                    in one step (zero-sum competition). Default,
        #                    matches all paper experiments through v5.
        #   "gated"       -- M_c^t = (1 - g) * M_c^{t-1} + g * judge(...)
        #                    where g in [0, 1]^{B x K x 1} is a learned per-
        #                    slot sigmoid gate computed from (M_c^{t-1}, M_new).
        #                    Init bias is -1.0 so g ~ 0.27 at step 0, giving
        #                    modest non-replacing writes. Lets the model
        #                    learn to *not* update slots that hold useful
        #                    long-horizon information rather than being
        #                    forced into a zero-sum overwrite every step.
        memres_update_mode: str = "competitive",
        # v11 (2026-04-30): bootstrap-fix knobs to unstuck the
        # gate/readout/writer chicken-and-egg (README Stop everything
        # P1 + P2). Defaults preserve pre-v11 init exactly so saved
        # checkpoints still load.
        # ``memres_gate_init``: scalar init for ``MemoryGate.gate``
        # (used in simple_gate routing). 0.0 is the historical default
        # but produces zero forward influence and zero gradient on
        # gate at step 0; a small positive value (e.g. 0.005) breaks
        # the chicken-and-egg by letting the LM gradient flow back
        # to the readout from step 1.
        # ``memres_readout_norm_init``: scalar init for
        # ``MemoryReadout.out_norm.weight``. 1.0 is the HF
        # Qwen3RMSNorm default and produces ||m^t||/||embed|| ~ 73,
        # which puts the gate's useful operating range at
        # [0, ~0.014] -- too narrow for AdamW's natural step. A
        # smaller value (e.g. 0.05) downscales the readout output
        # by 20x so the gate operates in [0, ~0.3], a healthier
        # regime for the optimizer.
        memres_gate_init: float = 0.0,
        memres_readout_norm_init: float = 1.0,
        # v12 (2026-05-01): writer-subsystem architectural switch.
        # Controls the implementation of MemoryBlock.judge (and
        # optionally MemoryBlock.extract).
        #   "original"            -- Eq. 1-2/5 single-pass cross-attention
        #                            judge (v1-v11 baseline).
        #   "slot_attention"      -- Stage 2 judge replaced by an
        #                            iterative SlotAttentionWriter (softmax
        #                            over slots + GRU keep/write per slot).
        #                            Targets the v11g D2 finding that the
        #                            original judge is decision-less
        #                            (row_entropy/log(2K) = 0.999).
        #   "slot_attention_full" -- Both Stage 1 (extraction) and Stage 2
        #                            (judging) use SlotAttentionWriter.
        # ``memres_slot_attention_iters`` is the number of iterations
        # inside the Stage-2 SlotAttentionWriter (canonical Slot Attention
        # default is 3).  Stage 1 uses 1 + memres_extraction_depth
        # iterations to subsume Eq. 3-4 layer count.
        memres_writer_kind: str = "original",
        memres_slot_attention_iters: int = 3,
        # v13 (2026-05-01): explicit symmetry-breaking in the writer's
        # learnable query seeds to escape the D2-confirmed permutation-
        # equivariant uniform fixed point (see problems.md §2b).
        #
        # ``memres_queries_init``:
        #   "normal"     -- i.i.d. N(0, d^{-1/2}). Rows are permutation-
        #                   equivariant under identity; with the judge
        #                   softmax permutation-invariant in its output,
        #                   the loss is itself permutation-symmetric in
        #                   M_judge and any uniform configuration is a
        #                   gradient fixed point. Default, preserves
        #                   v1-v12 behaviour.
        #   "orthogonal" -- rows of M_in and M_judge are mutually
        #                   orthogonal (nn.init.orthogonal_). Slot i's
        #                   projection q_i is orthogonal to slot j's
        #                   q_j at init, so attn[:,i,:] and attn[:,j,:]
        #                   are structurally distinct. The permutation
        #                   symmetry is broken at t=0; once broken,
        #                   distinct per-slot gradients maintain the
        #                   break.
        #
        # ``memres_slot_positional``: when True, add a fixed per-slot
        # index embedding (learnable, but initialised to a deterministic
        # Fourier pattern) to M_in and M_judge at forward time.  This
        # gives every slot a non-shared, non-random "address" that
        # survives adversarial parameter drift.  Zero-cost at parity
        # because the addition is followed by the existing W_Q/W_K/W_V
        # projections.
        memres_queries_init: str = "normal",
        memres_slot_positional: bool = False,
        # v14 (2026-05-02): attention-entropy-collapse stabiliser on the
        # Stage-2 judge.  Diagnostic evidence: v13r D2-JUDGE @ step 10000
        # had row_entropy / log(2K) = 0.988 (near-uniform judge), which
        # is the exact attention-entropy-collapse pathology identified
        # in Zhai et al. 2023.  QK-LayerNorm (post-projection RMSNorm on
        # Q and K) decouples attention-logit magnitude from W_Q/W_K
        # spectral norm so the softmax can find sharper distributions
        # without spectral-norm regularisation.  Standard in Qwen,
        # Gemma, DeepSeek-V3.  Applied only to the judge (writer_kind-
        # agnostic -- both ``original`` and ``slot_attention`` paths
        # pick it up via the ``judging`` sub-module).  Extraction is
        # left untouched because the D1 / D3 diagnostics do NOT show
        # entropy collapse in extraction; keeping the intervention
        # targeted reduces the risk surface of the v14 campaign.
        memres_judge_qk_layernorm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memres_num_vectors = memres_num_vectors
        # L_E: number of Perceiver-style refinement layers stacked on top of
        # the initial M_in cross-attention (Eq. 4).  L_E = 0 recovers Eq. 1.
        self.memres_extraction_depth = memres_extraction_depth
        # N: number of Block AttnRes blocks partitioning the attention/MLP
        # sublayers.  N == 2 * num_hidden_layers recovers Full AttnRes; N == 1 is
        # a standard residual stream with b_{-1} and b_0 as the only
        # non-trivial sources.  Defaults to 4 (well-suited to small/medium
        # research models, ~3 attn/MLP sublayers per block at L=12).
        self.memres_num_blocks = memres_num_blocks
        # Canonical routing mode (see MEMRES_MODES above for the three
        # variants).  Legacy values (memres_mode="residual" /
        # "block_attnres" + block_attnres_parity_init) are silently
        # translated so saved checkpoints and existing shell scripts keep
        # working.
        self.memres_mode = _normalise_memres_mode(
            memres_mode, block_attnres_parity_init
        )
        # Derived alias kept for backward compatibility with downstream
        # tooling that reads `config.block_attnres_parity_init` directly
        # (eval scripts, init_parity_test.py, older notebooks).  Always
        # consistent with the canonical mode.
        self.block_attnres_parity_init = (
            self.memres_mode == "attention_parity"
        )
        # Optional router-bias overrides; None means "use mode-derived
        # default" (resolved in Qwen3MemResModel.__init__).
        self.router_mem_bias_init = router_mem_bias_init
        self.router_recent_bias_init = router_recent_bias_init
        # Extraction source policy (see __init__ docstring above).
        self.memres_extract_source = memres_extract_source
        # Recurrent memory update mode (see __init__ docstring above).
        if memres_update_mode not in ("competitive", "gated"):
            raise ValueError(
                f"memres_update_mode must be 'competitive' or 'gated'; "
                f"got {memres_update_mode!r}"
            )
        self.memres_update_mode = memres_update_mode
        # v11 bootstrap-fix knobs (see __init__ docstring above).
        self.memres_gate_init = float(memres_gate_init)
        self.memres_readout_norm_init = float(memres_readout_norm_init)
        # v12 writer-kind switch (see __init__ docstring above).
        if memres_writer_kind not in _VALID_WRITER_KINDS:
            raise ValueError(
                f"memres_writer_kind must be one of {_VALID_WRITER_KINDS}; "
                f"got {memres_writer_kind!r}"
            )
        self.memres_writer_kind = memres_writer_kind
        self.memres_slot_attention_iters = int(memres_slot_attention_iters)
        if memres_queries_init not in ("normal", "orthogonal"):
            raise ValueError(
                f"memres_queries_init must be 'normal' or 'orthogonal'; "
                f"got {memres_queries_init!r}"
            )
        self.memres_queries_init = memres_queries_init
        self.memres_slot_positional = bool(memres_slot_positional)
        # v14 judge stabiliser (see __init__ docstring above).
        self.memres_judge_qk_layernorm = bool(memres_judge_qk_layernorm)


# ---------------------------------------------------------------------------
# Weight init helper
# ---------------------------------------------------------------------------


class MemoryGate(nn.Module):
    """Per-sublayer ReZero-style gate for residual memory injection.

    For each routed sublayer i we maintain a scalar gate g_i.  When memory is
    present the routed input is

        h_pre = h_residual + g_i * m^t

    All g_i are initialised to ``init`` (default 0) so the augmented model
    collapses onto the bare backbone at step 0.  The gate is a *raw* scalar
    (no sigmoid), so its gradient is exactly  m^t * grad_out  -- as soon as
    W_V_read has produced a non-zero readout the gate can move off zero
    without the saturating-sigmoid problem.  The shared gate provides a
    clean depth-wise profile of which sublayers actually recruit memory
    once training has progressed, which is what the routing diagnostic
    measures.
    """

    def __init__(self, num_routing_steps: int, init: float = 0.0):
        super().__init__()
        self._init = init
        self.gate = nn.Parameter(torch.empty(num_routing_steps))

    def alpha(self, idx: int) -> torch.Tensor:
        return self.gate[idx]


def _init_memres_params(module: nn.Module, hidden_size: int) -> None:
    """Initialize raw nn.Parameter attributes and key projections on MemRes modules.

    HF's `_init_weights` only touches Linear/Embedding/Norms generically, so
    our custom Parameters (M_in, M_judge, depth_router.w[...]) stay
    uninitialized (NaN garbage) after `from_pretrained` unless we init them
    explicitly.  The MemoryReadout.W_V projection additionally needs a *zero*
    initialization so that at step 0 the per-position readout m^t equals zero
    and the augmented model is behaviourally identical to the pretrained
    backbone (the ReZero/Fixup-style trick recommended in
    TRAINING_PLAYBOOK.md and also called out in PITFALLS.md §1 as the single
    most important init knob to avoid channel collapse).

    Per AttnRes §5, all pseudo-query vectors w_{n,i} are zero-initialized so
    the initial softmax over the value pool is uniform across sources; this
    reduces the model to an equal-weight average at the start of training and
    prevents training volatility.
    """
    std = hidden_size**-0.5
    if isinstance(module, MemoryBlock):
        kind = getattr(module, "_queries_init_kind", "normal")
        if kind == "orthogonal":
            # Orthogonal rows -> slots are structurally distinct from
            # step 0.  `nn.init.orthogonal_` uses QR decomposition
            # internally, which is not implemented for bfloat16; compute
            # in float32 and cast back to preserve model dtype.
            with torch.no_grad():
                for p in (module.M_in, module.M_judge):
                    buf = torch.empty(
                        p.shape, dtype=torch.float32, device=p.device
                    )
                    nn.init.orthogonal_(buf)
                    # Rescale to match the normal init's typical row
                    # norm (~1.0) so downstream W_Q/W_K/W_V see the
                    # same activation scale.  orthogonal_ already gives
                    # unit row norms, so this is a no-op for shape
                    # K <= d; kept explicit for clarity.
                    p.data.copy_(buf.to(p.dtype))
        else:
            nn.init.normal_(module.M_in, std=std)
            nn.init.normal_(module.M_judge, std=std)
        # Per-slot positional address.  Fourier pattern indexed by slot
        # position: pos[k, 2i] = sin(k / 10000^(2i/d)),
        # pos[k, 2i+1] = cos(...).  Small magnitude (scaled by std) so
        # it survives gradient updates but doesn't dominate M_in/M_judge
        # initially.  Learnable, so the model can refine it.
        if (getattr(module, "_slot_positional", False)
                and module.M_in_pos is not None
                and module.M_judge_pos is not None):
            K = module.num_vectors
            d = hidden_size
            with torch.no_grad():
                pos = torch.zeros(K, d)
                positions = torch.arange(K, dtype=torch.float32).unsqueeze(1)
                div = torch.exp(
                    torch.arange(0, d, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d)
                )
                pos[:, 0::2] = torch.sin(positions * div)
                pos[:, 1::2] = torch.cos(positions * div)
                pos.mul_(std)  # downscale to match M_in/M_judge magnitude
                module.M_in_pos.data.copy_(pos)
                module.M_judge_pos.data.copy_(pos)
        # In gated mode, override HF's generic Linear init with the
        # zero-weight + -1 bias init that gives g ~ 0.27 at step 0.
        if getattr(module, "update_mode", "competitive") == "gated":
            nn.init.zeros_(module.write_gate.weight)
            nn.init.constant_(module.write_gate.bias, -1.0)
    elif isinstance(module, MemoryReadout):
        # Read-out V projection stays at the default normal init so m^t has
        # well-scaled non-zero magnitude at step 0; the augmented model is
        # still identical to the bare backbone because the per-sublayer
        # MemoryGate is zero-initialised, so  h + gate * m^t == h .  This
        # gives a non-zero gradient signal back into the gate from step 1
        # (the saturating-sigmoid alternative collapses gate gradients).
        # v11 (2026-04-30): apply ``out_norm_init`` to the RMSNorm
        # weight so the readout output magnitude is calibrated to
        # the gate's natural operating range. Init=1.0 reproduces
        # the v8/v9/v10 default; init=0.05 produces ||m^t||/||embed||
        # ~ 3.6 instead of ~73, so a gate of ~0.3 (vs ~0.014) is the
        # "useful" operating point -- well within AdamW's natural
        # step size.
        scale = float(getattr(module, "_out_norm_init", 1.0))
        with torch.no_grad():
            module.out_norm.weight.fill_(scale)
    elif isinstance(module, BlockAttnResRouter):
        for w in module.w:
            nn.init.zeros_(w)
        # Strong negative init for the per-sublayer memory-score bias so the
        # initial alpha_mem ~ exp(mem_bias)/N is effectively zero.  This is
        # what makes the augmented model collapse onto the base model at
        # step 0 even though the depth-wise pseudo-queries are zero
        # (otherwise uniform softmax dilutes other sources by 1/N).
        nn.init.constant_(module.mem_bias, module._mem_bias_init)
        # Default 0 (uniform softmax over non-memory sources).  When set to
        # a large positive value via `recent_bias_init`, the softmax
        # collapses onto the most-recent source -- combined with a
        # cumulative-state value pool this is what lets block_attnres mode
        # achieve forward init-parity with the bare backbone.
        nn.init.constant_(module.recent_bias, module._recent_bias_init)
    elif isinstance(module, MemoryGate):
        nn.init.constant_(module.gate, module._init)
    elif isinstance(module, SlotAttentionWriter):
        # HuggingFace's generic ``_init_weights`` only matches Linear /
        # Embedding / Norm / RotaryEmbedding submodules, so the GRUCell
        # inside the SlotAttentionWriter is left at the meta-device
        # NaN values that ``from_pretrained`` allocates -- the model
        # then produces NaN logits at the first forward pass.  We call
        # ``reset_parameters()`` on the GRUCell to apply PyTorch's
        # standard uniform_(-sqrt(1/h), sqrt(1/h)) init to all four
        # GRUCell tensors (weight_ih, weight_hh, bias_ih, bias_hh).
        module.gru.reset_parameters()


# ---------------------------------------------------------------------------
# Cross-attention primitive
# ---------------------------------------------------------------------------


class CrossAttention(nn.Module):
    """Single cross-attention: queries attend to context key-values.

    ``qk_layernorm`` (v14): when True, insert an RMSNorm on the Q and K
    projections' outputs before the inner product.  Motivated by the
    D2-JUDGE diagnostic in v13 showing row_entropy / log(2K) > 0.95
    (near-uniform judge attention), which corresponds to the
    low-spectral-norm / low-logit-magnitude attention-entropy pathology
    identified in Zhai et al. 2023 ("Stabilizing Transformer Training
    by Preventing Attention Entropy Collapse").  QK-LayerNorm
    decouples the attention-logit magnitude from the W_Q / W_K weight
    norm: logits become  d^{-1/2} * <rmsnorm(W_Q q), rmsnorm(W_K k)>
    which is insensitive to the spectral norm of the projections and
    lets the softmax find sharper distributions without requiring
    carefully tuned weight decay or spectral constraints on W_Q/W_K.
    Cheaper than full σReparam and has become the standard fix (used
    in Qwen, Gemma, DeepSeek-V3 attention blocks).
    """

    def __init__(self, hidden_size: int, qk_layernorm: bool = False):
        super().__init__()
        self.scale = hidden_size**-0.5
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        if qk_layernorm:
            self.q_norm: nn.Module = Qwen3RMSNorm(hidden_size, eps=1e-6)
            self.k_norm: nn.Module = Qwen3RMSNorm(hidden_size, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        Q = self.q_norm(self.W_Q(queries))
        K = self.k_norm(self.W_K(context))
        V = self.W_V(context)
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ V


# ---------------------------------------------------------------------------
# Slot-Attention writer (v12)  --  iterative slot-competitive attention.
# ---------------------------------------------------------------------------


class SlotAttentionWriter(nn.Module):
    """Iterative slot-competitive attention (Locatello et al. 2020).

    Used by the v12 writer subsystem in two places:
      * MemoryBlock.judging  (Stage 2 / Eq. 5):
            slots^(0) = M_judge,          P = [M_c^{t-1} || M_new]   (B, 2K, d)
      * MemoryBlock.extract  (Stage 1 / Eq. 3-4) when writer_kind is
        ``slot_attention_full``:
            slots^(0) = M_in,             P = C_t                    (B, N, d)

    Why this replaces the plain cross-attention judge
    --------------------------------------------------
    The PDF's Section 2.1 ("Forgetting Defense Insight") promises a
    "zero-sum semantic competition" between M_c^{t-1} and the newly
    extracted candidate.  The original ``CrossAttention`` judge takes
    softmax over the *inputs* axis, so each of the K query slots
    independently averages over the 2K inputs.  At init, every query is
    a randomly-projected vector and the softmax is near-uniform, which
    means every slot tends to the same content-blind average -- there
    is no architectural pressure for slots to specialise on disjoint
    pieces of the input.  This is the symmetric uniform fixed point
    that the v11g D2 audit measured at row_entropy / log(2K) = 0.999
    and effective rank 1.02 across 4000 training steps.

    Slot Attention restructures the softmax to be over the *slots* axis
    (i.e., across K).  The competition is then explicit: if two slots
    project similarly under W_Q, they share each input's softmax mass --
    but as soon as either slot's projection drifts, the other loses
    that mass.  After the per-row weighted-mean normalisation each slot
    receives a proper convex combination of the inputs that selected it,
    so the K slots tile the input pool into K disjoint factors.  Five
    years of object-centric / set-prediction literature have validated
    this primitive on exactly this problem.

    Init parity with the bare backbone is preserved at the *model* level
    by the downstream gate / router init (memory_gate=0 in simple_gate
    mode, alpha_mem ~ exp(-32)/N in attention_parity), which zeros m^t's
    contribution to h regardless of what the writer outputs at init.
    See ``tools/init_parity_test.py`` for the slot-attention case.

    Parameter accounting (per writer instance, hidden_size=d, K slots):
      * W_Q, W_K, W_V  (nn.Linear no-bias)  -- same as CrossAttention
      * GRUCell        (3 d^2 + 3 d^2 = 6 d^2 + 6 d, vs CrossAttention's 3 d^2)
      * 2 RMSNorms     (slot_norm + input_norm; d each)
    Roughly 3x the parameters of one CrossAttention pass; comparable to
    a single transformer layer at the same hidden size.
    """

    def __init__(
        self,
        hidden_size: int,
        num_iters: int = 3,
        eps: float = 1e-6,
        qk_layernorm: bool = False,
    ):
        super().__init__()
        if num_iters < 1:
            raise ValueError(
                f"num_iters must be >= 1; got {num_iters!r}"
            )
        self.hidden_size = hidden_size
        self.num_iters = int(num_iters)
        self.scale = hidden_size**-0.5
        self.eps = eps
        # Same Q/K/V projection budget as a CrossAttention judge.
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        # Per-slot recurrent update.  GRUCell is the canonical Slot
        # Attention update (Locatello 2020); it weight-ties across slots
        # but maintains independent hidden state per slot through the
        # iterations.
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        # Pre-projection normalisations (canonical SA places LayerNorm
        # before W_Q on slots and before W_K/V on inputs).  We use
        # RMSNorm to stay consistent with the rest of the architecture.
        self.slot_norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.input_norm = Qwen3RMSNorm(hidden_size, eps=eps)
        # v14 QK-LayerNorm: post-projection RMSNorm on Q/K.  See
        # CrossAttention.__init__ for motivation.  Note the slot_norm /
        # input_norm above are PRE-projection norms on the slot state
        # and input pool; qk_layernorm is POST-projection, on the Q and
        # K images, which is where the attention-entropy-collapse
        # literature places it.  Cheap and orthogonal.
        if qk_layernorm:
            self.q_norm: nn.Module = Qwen3RMSNorm(hidden_size, eps=eps)
            self.k_norm: nn.Module = Qwen3RMSNorm(hidden_size, eps=eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _attention(
        self,
        slots: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the (B, K, M) slot-axis-normalised attention weights.

        Two normalisations:
          1. ``softmax(scores, dim=-2)``: softmax along the SLOTS axis
             (K), so each input column j has its mass distributed across
             slots with sum 1.  This is what makes the slots compete:
             any input row can only contribute total mass 1, split among
             slots according to their query-key alignment.
          2. ``attn / attn.sum(dim=-1, keepdim=True)``: per-slot weighted-
             mean normalisation -- guarantees each slot's row sums to
             exactly 1, so attn @ v is a proper convex combination of
             the inputs (and is shape-compatible with the original
             D2 keep-vs-write diagnostic).
        """
        scores = (slots @ k.transpose(-2, -1)) * self.scale  # (B, K, M)
        # Softmax over SLOTS axis (this is the architectural pivot).
        attn = F.softmax(scores, dim=-2)
        # Per-slot weighted-mean renormalisation.
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
        return attn

    def forward(
        self,
        slots: torch.Tensor,
        P: torch.Tensor,
    ) -> torch.Tensor:
        """Iterative slot-competitive attention.

        Args:
            slots: (B, K, d) initial slot state -- typically the
                   broadcast of a learnable parameter (M_judge or M_in).
            P:     (B, M, d) input pool to attend over.

        Returns:
            slots_final: (B, K, d) updated slot state after num_iters
            iterations.
        """
        B, K, d = slots.shape
        Pn = self.input_norm(P)
        k_proj = self.k_norm(self.W_K(Pn))
        v_proj = self.W_V(Pn)
        for _ in range(self.num_iters):
            sn = self.slot_norm(slots)
            q_proj = self.q_norm(self.W_Q(sn))
            attn = self._attention(q_proj, k_proj)  # (B, K, M)
            updates = attn @ v_proj                 # (B, K, d)
            # Per-slot GRU update (slots and updates flatten to (B*K, d);
            # K independent recurrent states share the same gating weights).
            slots = self.gru(
                updates.reshape(B * K, d),
                slots.reshape(B * K, d),
            ).reshape(B, K, d)
        return slots

    @torch.no_grad()
    def attention(
        self,
        slots: torch.Tensor,
        P: torch.Tensor,
    ) -> torch.Tensor:
        """Diagnostic: final-iteration normalised attention (B, K, M).

        Used by the D2 audit to measure judge decisiveness.  Returned in
        the same (B, K, 2K) shape convention as
        ``MemoryBlock.judge_attention`` for the original writer, so the
        keep/write split (attn[:, :, :K] vs attn[:, :, K:]) and the
        per-row entropy reductions in the audit code apply unchanged.
        """
        B, K, d = slots.shape
        Pn = self.input_norm(P)
        k_proj = self.k_norm(self.W_K(Pn))
        v_proj = self.W_V(Pn)
        attn: torch.Tensor | None = None
        for it in range(self.num_iters):
            sn = self.slot_norm(slots)
            q_proj = self.q_norm(self.W_Q(sn))
            attn = self._attention(q_proj, k_proj)
            if it < self.num_iters - 1:
                updates = attn @ v_proj
                slots = self.gru(
                    updates.reshape(B * K, d),
                    slots.reshape(B * K, d),
                ).reshape(B, K, d)
        assert attn is not None
        return attn


# ---------------------------------------------------------------------------
# Memory Block: Two-Stage QKV Competition  (Section 2.1)
# ---------------------------------------------------------------------------


class MemoryBlock(nn.Module):
    """
    Two-Stage QKV Competition (Section 2.1).

    Stage 1 — Extraction (Eq. 1 baseline; Eqs. 3-4 with L_E > 0):
        The raw session C_t (shape (B, N, d)) is compressed into a
        K-slot candidate M_new via a Perceiver-style refinement stack.
        The initial cross-attention uses fixed learnable queries M_in;
        each refinement layer lets the evolving latent state re-query
        the raw session.  All intermediate states live in R^{K x d},
        so the residual refinement is well-typed.

            E^(0) = CrossAttn(M_in, C_t)
            E^(ℓ) = E^(ℓ-1) + CrossAttn^(ℓ)(E^(ℓ-1), C_t), ℓ = 1..L_E
            M_new = E^(L_E)

    Stage 2 — Judging (Eqs. 2/5):
        P_judge = [M_c^{t-1} || M_new] in R^{2K x d}.  Learnable judging
        queries M_judge (K x d) attend over P_judge so old and new
        compete in a zero-sum softmax.  Single round — the heavy lifting
        has already happened in the extraction stack.

    Writer kinds (v12)
    ------------------
    ``writer_kind`` selects the implementation of Stage 1 / Stage 2:

      * ``original`` (default): Stage 1 is a stack of plain
        ``CrossAttention`` layers (paper Eq. 3-4); Stage 2 is a single
        ``CrossAttention`` competition (Eq. 5).  This is the v1-v11
        baseline.
      * ``slot_attention``: Stage 1 unchanged; Stage 2 replaced by an
        iterative ``SlotAttentionWriter`` (softmax over slots, GRU
        keep/write gate per slot).  Targets the D2-confirmed
        decision-less judge (v11g row_entropy/log(2K) = 0.999).
      * ``slot_attention_full``: both Stage 1 and Stage 2 use
        ``SlotAttentionWriter``.  Tests whether the extraction stage
        also benefits from the slot-axis-softmax inductive bias.
    """

    def __init__(
        self,
        hidden_size: int,
        num_vectors: int,
        extraction_depth: int = 0,
        eps: float = 1e-6,
        update_mode: str = "competitive",
        writer_kind: str = "original",
        slot_attention_iters: int = 3,
        queries_init: str = "normal",
        slot_positional: bool = False,
        judge_qk_layernorm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_vectors = num_vectors
        self.extraction_depth = extraction_depth
        if update_mode not in ("competitive", "gated"):
            raise ValueError(
                f"update_mode must be 'competitive' or 'gated'; got {update_mode!r}"
            )
        if writer_kind not in _VALID_WRITER_KINDS:
            raise ValueError(
                f"writer_kind must be one of {_VALID_WRITER_KINDS}; "
                f"got {writer_kind!r}"
            )
        if queries_init not in ("normal", "orthogonal"):
            raise ValueError(
                f"queries_init must be 'normal' or 'orthogonal'; "
                f"got {queries_init!r}"
            )
        self.update_mode = update_mode
        self.writer_kind = writer_kind
        self.slot_attention_iters = int(slot_attention_iters)
        # v13 symmetry-break knobs.  Threaded here so that
        # _init_memres_params (called by HF from_pretrained after
        # construction) can see them on the module.
        self._queries_init_kind = queries_init
        self._slot_positional = bool(slot_positional)

        # Stage 1: learnable extraction queries M_in  (K x d)
        self.M_in = nn.Parameter(torch.empty(num_vectors, hidden_size))
        nn.init.normal_(self.M_in, std=hidden_size**-0.5)
        # v13: optional per-slot positional address.  Applied additively
        # before the W_Q/W_K/W_V projections in extract() / judge().
        # Initialised in _init_memres_params to a deterministic per-slot
        # Fourier pattern with small magnitude (~d^{-0.5}), so parity
        # with the bare backbone is preserved (M_in/M_judge remain
        # unused at step 0 because gate/router init zeros their
        # downstream contribution).
        if slot_positional:
            self.M_in_pos = nn.Parameter(
                torch.empty(num_vectors, hidden_size)
            )
        else:
            self.register_parameter("M_in_pos", None)
        if writer_kind == "slot_attention_full":
            # A single SlotAttentionWriter does (1 + L_E) iterations of
            # slot-competitive refinement over C_t, replacing the stack
            # of CrossAttention layers.  Iterations subsume the Eq. 3-4
            # residual refinement (the GRU IS the residual update on the
            # slot state), so we collapse the L_E layer stack into one
            # iterative module.  ``extraction_layers`` is left as None
            # for type-symmetry with the original path.
            self.extract_block = SlotAttentionWriter(
                hidden_size,
                num_iters=1 + extraction_depth,
                eps=eps,
            )
            self.extraction_layers: nn.ModuleList | None = None
        else:
            # Initial compression + L_E refinement layers (Eq. 3-4).
            # All share the same (K, d) -> (K, d) latent shape.
            self.extraction_layers = nn.ModuleList(
                [CrossAttention(hidden_size) for _ in range(1 + extraction_depth)]
            )
            self.extract_block: SlotAttentionWriter | None = None

        # Stage 2: learnable judging queries M_judge  (K x d)
        self.M_judge = nn.Parameter(torch.empty(num_vectors, hidden_size))
        nn.init.normal_(self.M_judge, std=hidden_size**-0.5)
        if slot_positional:
            self.M_judge_pos = nn.Parameter(
                torch.empty(num_vectors, hidden_size)
            )
        else:
            self.register_parameter("M_judge_pos", None)
        if writer_kind in ("slot_attention", "slot_attention_full"):
            self.judging: nn.Module = SlotAttentionWriter(
                hidden_size,
                num_iters=self.slot_attention_iters,
                eps=eps,
                qk_layernorm=bool(judge_qk_layernorm),
            )
        else:
            self.judging = CrossAttention(
                hidden_size, qk_layernorm=bool(judge_qk_layernorm)
            )
        # Per PITFALLS §2 and TRAINING_PLAYBOOK §"RMSNorm placement", apply
        # RMSNorm to M_c after the judge step so that repeated application
        # over many sessions does not let the recurrent state drift out of
        # the well-conditioned region the readout was trained on.  Without
        # this, eval over chains longer than the training TBPTT window k
        # exhibits unbounded growth of |M_c|_F and the readout becomes
        # numerically degenerate.
        self.judge_norm = Qwen3RMSNorm(hidden_size, eps=eps)

        # In "gated" mode, add a per-slot sigmoid write gate so that the
        # recurrent update can choose to KEEP an existing slot (g ~ 0)
        # rather than overwrite it with the judge output. The gate is a
        # function of the prior slot M_c^{t-1} and the new candidate M_new,
        # so the model can learn content-dependent forgetting / writing.
        # Init bias is -1.0 so g ~ 0.27 at step 0: modest writes early,
        # most of the prior memory is preserved (which is also the
        # behaviour we want with longer chains).
        if update_mode == "gated":
            self.write_gate = nn.Linear(2 * hidden_size, 1, bias=True)
            nn.init.zeros_(self.write_gate.weight)
            nn.init.constant_(self.write_gate.bias, -1.0)

    def extract(self, C: torch.Tensor) -> torch.Tensor:
        """Stage 1: refine M_in queries over C_t for 1 + L_E layers.

        Initial layer compresses N -> K via M_in queries (Eq. 1 / 3).
        Subsequent layers refine the (K, d) state by letting it re-query
        C_t with a residual connection (Eq. 4).

        For ``writer_kind == "slot_attention_full"`` the L_E + 1
        CrossAttention layers are collapsed into a single
        ``SlotAttentionWriter`` whose internal iteration count plays
        the role of the layer count -- Eq. 4's residual refinement is
        absorbed into the per-slot GRU update.
        """
        B = C.size(0)
        q_seed = self.M_in
        if self.M_in_pos is not None:
            q_seed = q_seed + self.M_in_pos
        M_in = q_seed.unsqueeze(0).expand(B, -1, -1)
        if self.extract_block is not None:
            return self.extract_block(M_in, C)
        assert self.extraction_layers is not None
        E = self.extraction_layers[0](M_in, C)  # (B, K, d)
        for layer in self.extraction_layers[1:]:
            E = E + layer(E, C)
        return E

    def judge(self, M_c_prev: torch.Tensor, M_new: torch.Tensor) -> torch.Tensor:
        """Stage 2: single-round competition over [M_c^{t-1} || M_new]."""
        B = M_new.size(0)
        P_judge = torch.cat([M_c_prev, M_new], dim=1)  # (B, 2K, d)
        q_seed = self.M_judge
        if self.M_judge_pos is not None:
            q_seed = q_seed + self.M_judge_pos
        M_judge = q_seed.unsqueeze(0).expand(B, -1, -1)
        if isinstance(self.judging, SlotAttentionWriter):
            # Slot-competitive iterative judge (v12).  M_judge serves as
            # the initial slot state; the GRUCell takes care of the
            # keep-vs-write gating across iterations.
            out = self.judging(M_judge, P_judge)
        else:
            out = self.judging(M_judge, P_judge)
        # RMSNorm bounds the recurrent state.  See __init__ for rationale.
        return self.judge_norm(out)

    @torch.no_grad()
    def judge_attention(
        self, M_c_prev: torch.Tensor, M_new: torch.Tensor
    ) -> torch.Tensor:
        """Diagnostic: return raw (B, K, 2K) judge attention weights.

        Mirrors ``judge`` but returns the attention probabilities rather
        than the projected output.  Used by the D2 audit:

            attn[:, :, :K] -> mass put on the prior memory  (KEEP)
            attn[:, :, K:] -> mass put on the new candidate (WRITE)

        Per-row entropy answers "did the judge make a decision at all?":
        log(2K) means uniform attention (no judging).  Mean keep mass
        answers "did the writer overwrite or preserve?".

        For both writer kinds the returned tensor has each row summing
        to 1, so ``keep_mean = attn[:, :, :K].sum(dim=-1).mean()`` and
        per-row entropy reductions in the audit code apply uniformly.
        """
        B, K, d = M_new.shape
        P = torch.cat([M_c_prev, M_new], dim=1)  # (B, 2K, d)
        q_seed = self.M_judge
        if self.M_judge_pos is not None:
            q_seed = q_seed + self.M_judge_pos
        M_judge = q_seed.unsqueeze(0).expand(B, -1, -1)
        if isinstance(self.judging, SlotAttentionWriter):
            return self.judging.attention(M_judge, P)
        # Original judge: softmax over the *inputs* axis.
        judge = self.judging
        Q = judge.W_Q(M_judge)
        Kk = judge.W_K(P)
        scores = (Q @ Kk.transpose(-2, -1)) * judge.scale
        return F.softmax(scores, dim=-1)

    def forward(
        self,
        C: torch.Tensor,
        M_c_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full two-stage memory update.

        Args:
            C: (B, N, d) — session token representations.
            M_c_prev: (B, K, d) — prior memory state; None -> zeros.
        Returns:
            M_c: (B, K, d) — updated memory state.
        """
        M_new = self.extract(C)
        if M_c_prev is None:
            M_c_prev = torch.zeros_like(M_new)
        candidate = self.judge(M_c_prev, M_new)
        if self.update_mode == "competitive":
            return candidate
        # Gated mode: per-slot sigmoid gate g in [0,1]^{B x K x 1}.
        # M_c^t = (1 - g) * M_c^{t-1} + g * candidate.
        # When the prior is the zero matrix (first session, no warm-start),
        # this collapses to g * candidate — same gradient signal as the
        # competitive path on step t=0, so the gate doesn't suppress the
        # model's first ever memory write.
        gate_input = torch.cat([M_c_prev, M_new], dim=-1)  # (B, K, 2d)
        g = torch.sigmoid(self.write_gate(gate_input))     # (B, K, 1)
        return (1 - g) * M_c_prev + g * candidate


# ---------------------------------------------------------------------------
# Memory Readout  (Section 2.2, Eq. 6)
# ---------------------------------------------------------------------------


class MemoryReadout(nn.Module):
    """Per-position cross-attention readout over M_c (Eq. 6).

    Each position in the current session queries the K memory slots
    independently, producing m^t of shape (B, S, d) — identical to the
    shape of any standard attention layer output, so it drops directly
    into the depth-wise routing pool at v_0 without broadcasting.

        m^t = RMSNorm( Softmax(X W_Q (M_c W_K)^T / sqrt(d)) M_c W_V )

    Why the RMSNorm on the output (added in v8, see
    ``results/eval/diag_v7_*.json`` for the diagnostic data
    that motivated it):

      The paper's spec assumes the foundational sources of the depth
      pool (b_{-1}=m^t, b_0=h_1, b_k=block summaries) are commensurate
      in scale because they all live in R^{S x d}. The code never
      enforced this for m^t directly: there is RMSNorm on M_c (the
      writer's output) and RMSNorm in the BlockAttnResRouter for
      *score* computation, but no RMSNorm on m^t itself. Consequently,
      the scale of m^t = (attn @ V) where V = W_V @ M_c drifts to
      whatever ||W_V^read|| the optimizer happens to find:

        - In ``attention_parity`` mode at soft ±4 the per-sublayer
          alpha_mem sits at ~4e-4, the gradient on W_V^read is
          ~1000x weaker than on backbone params, ``weight_decay=0.1``
          erodes W_V^read each step, ||W_V^read|| -> 0, ||m^t|| -> 0,
          and the architecture causally collapses to the bare
          backbone (verified empirically on chain_v7_p0_softerbias
          step-2000: ||m^t||/||embed|| = 1.66e-10, pa_cb_dsh = 0.0
          to floating-point precision).
        - In ``simple_gate`` mode the gate gives W_V^read direct LM
          gradient, so ||W_V^read|| can grow without bound. On
          chain_v7_p0_simplegate step-500 we measured
          ||m^t||/||embed|| = 165, which dominates the residual
          stream and pushes Δ_nm-m on callback tokens to -0.66.

      RMSNorm on the readout output rescales m^t back to ~sqrt(d),
      matching the embedding norm and making memory injection
      commensurate with the LM contribution at every depth. The
      learnable scalar weight (initialized to 1) lets the model still
      modulate the magnitude over training, but bounded.

    Init parity is preserved by both downstream injection paths:
    ``simple_gate`` has gate=0 at init so h + 0*RMSNorm(m^t) = h;
    ``attention_parity`` has alpha_mem ~ exp(-mem_bias)/N at init so
    the contribution to h is multiplied by an effectively-zero
    softmax weight regardless of m^t magnitude.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        out_norm_init: float = 1.0,
    ):
        super().__init__()
        self.scale = hidden_size**-0.5
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        # See class docstring above for the empirical justification.
        # weight init defaults to 1.0 (HF _init_weights for
        # Qwen3RMSNorm). v11 (2026-04-30) accepts a smaller init
        # (e.g. 0.05) to scale the readout output down by 20x so the
        # gate's useful operating range moves from [0, ~0.014] to
        # [0, ~0.3] -- a far healthier regime for AdamW. The
        # parameter remains learnable, so the model can still grow
        # the readout magnitude over training as needed.
        self.out_norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self._out_norm_init = float(out_norm_init)

    def forward(self, X: torch.Tensor, M_c: torch.Tensor) -> torch.Tensor:
        """X: (B, S, d) queries;  M_c: (B, K, d) memory slots  ->  (B, S, d)"""
        Q = self.W_Q(X)
        K = self.W_K(M_c)
        V = self.W_V(M_c)
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        return self.out_norm(attn @ V)


# ---------------------------------------------------------------------------
# Block Attention Residuals Router  (Section 2.2, Eqs. 7-10)
# ---------------------------------------------------------------------------


class BlockAttnResRouter(nn.Module):
    """
    Block Attention Residuals routing with optional memory injection.

    For routing step idx (sublayer i of block n), the value pool is

        V_{n,i} = [b_{-1}, b_0, b_1, ..., b_{n-1}]               if i == 1
                | [b_{-1}, b_0, b_1, ..., b_{n-1}, b_n^{i-1}]    if i >= 2

    where b_{-1} = m^t is the (optional) memory source, b_0 = h_1 is the
    token embedding, b_k are completed prior block summaries, and b_n^{i-1}
    is the running intra-block partial sum.  b_{-1} is omitted when memory
    is absent.  The routed hidden state is

        h_{n,i} = sum_{v in V_{n,i}}  alpha_{v -> (n,i)} * v
        alpha_{v -> (n,i)} = phi(w_{n,i}, v) / sum_{v'} phi(w_{n,i}, v')
        phi(q, v) = exp(q^T RMSNorm(v))

    Pseudo-queries w_{n,i} are zero-initialized so the initial pool weights
    are uniform across sources (AttnRes Section 5).
    """

    def __init__(
        self,
        hidden_size: int,
        num_routing_steps: int,
        eps: float = 1e-6,
        mem_bias_init: float = -8.0,
        recent_bias_init: float = 0.0,
    ):
        super().__init__()
        self.w = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_routing_steps)]
        )
        # Per-sublayer learnable scalar additive bias on the memory source's
        # routing score.  Initialized in `_init_memres_params` to a strong
        # negative number so alpha_mem ~ exp(mem_bias)/N at step 0; this is
        # what allows the augmented model to collapse onto the base model
        # at step 0 even with zero pseudo-queries (otherwise the uniform
        # softmax over an N+1 source pool dilutes other sources by 1/N).
        # As training progresses the model can move this bias up.
        self._mem_bias_init = mem_bias_init
        self.mem_bias = nn.Parameter(torch.empty(num_routing_steps))
        # Per-sublayer additive bias on the *last* (most recent) source in
        # the value pool.  Default 0 reproduces the original AttnRes
        # softmax over uniformly-treated sources.  When initialized to a
        # large positive number (e.g. +8) and the value pool stores
        # cumulative hidden-state checkpoints (Qwen3MemResConfig
        # `block_attnres_parity_init=True`), the softmax is one-hot on the
        # most recent cumulative source which equals the standard residual
        # stream input -- giving forward-identity to the bare backbone at
        # step 0.  The trade-off is a saturated softmax that requires a
        # bias warm-up to relax before the pseudo-queries can route
        # anywhere else.
        self._recent_bias_init = recent_bias_init
        self.recent_bias = nn.Parameter(torch.empty(num_routing_steps))
        self.norm = Qwen3RMSNorm(hidden_size, eps=eps)

    def route(
        self,
        router_idx: int,
        sources: list[torch.Tensor],
        has_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            router_idx: index into self.w (0 for layer 1, 1 for layer 2, ...).
            sources: list of (B, S, d) tensors in pool order
                     (b_{-1}? , b_0, b_1, ..., b_{n-1}, b_n^{i-1}?).
            has_memory: True iff sources[0] is b_{-1} = m^t.

        Returns:
            h: (B, S, d) routed hidden state.
            alpha_mem: (B, S) attention mass on b_{-1}, or None if no memory.
        """
        w = self.w[router_idx]  # (d,)

        stacked = torch.stack(sources, dim=2)  # (B, S, n_src, d)
        normed = self.norm(stacked)
        scores = torch.einsum("d,bsnd->bsn", w, normed)  # (B, S, n_src)
        # Build a (n_src,)-shaped bias vector containing the per-source
        # init biases: mem_bias on the b_{-1} slot, recent_bias on the
        # final slot (the most-recent source in the pool), zeros elsewhere.
        n_src = scores.shape[-1]
        bias = torch.zeros(n_src, dtype=scores.dtype, device=scores.device)
        if has_memory:
            bias[0] = self.mem_bias[router_idx].to(bias.dtype)
        bias[-1] = bias[-1] + self.recent_bias[router_idx].to(bias.dtype)
        scores = scores + bias
        alphas = F.softmax(scores, dim=-1)  # (B, S, n_src)
        h = torch.einsum("bsn,bsnd->bsd", alphas, stacked)  # (B, S, d)

        alpha_mem = alphas[:, :, 0] if has_memory else None
        return h, alpha_mem


DepthWiseRouter = BlockAttnResRouter


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class Qwen3MemResDecoderLayer(GradientCheckpointingLayer):
    """Qwen3 decoder block whose attention and MLP transforms can be routed
    as separate AttnRes layers.

    Attention Residuals treats each self-attention or MLP transform as an
    individual depth-wise source.  The standard ``forward`` remains available
    for compatibility, while the MemRes backbone calls ``attention_delta`` and
    ``mlp_delta`` directly so routing happens before both sublayers.
    """

    def __init__(self, config: Qwen3MemResConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    def attention_delta(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return attn_out

    def mlp_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.post_attention_layernorm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Compatibility path: vanilla Qwen residual flow inside the decoder
        # block. MemRes routing uses attention_delta/mlp_delta directly.
        attn_out = self.attention_delta(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + attn_out

        hidden_states = hidden_states + self.mlp_delta(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------


class Qwen3MemResModel(Qwen3PreTrainedModel):
    """Qwen3 backbone with Memory Residuals (Sections 2.1 + 2.2)."""

    config_class = Qwen3MemResConfig

    def __init__(self, config: Qwen3MemResConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MemResDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        # ── Memory components ──
        # Invariant: MemoryBlock and MemoryReadout both use
        # config.hidden_size (= d) as their internal dimension.  This makes
        # the readout output m^t shape-compatible with any decoder layer's
        # output (B, S, d), which is what lets b_{-1} := m^t drop into the
        # Block AttnRes value pool alongside b_0, b_1, ... without any
        # broadcasting or projection.  Do not change these to a different
        # size unless the routing pool's stack op is also restructured.
        self.memory_block = MemoryBlock(
            config.hidden_size,
            config.memres_num_vectors,
            config.memres_extraction_depth,
            update_mode=getattr(config, "memres_update_mode", "competitive"),
            writer_kind=getattr(config, "memres_writer_kind", "original"),
            slot_attention_iters=getattr(
                config, "memres_slot_attention_iters", 3
            ),
            queries_init=getattr(config, "memres_queries_init", "normal"),
            slot_positional=getattr(config, "memres_slot_positional", False),
            judge_qk_layernorm=getattr(
                config, "memres_judge_qk_layernorm", False
            ),
        )
        self.memory_readout = MemoryReadout(
            config.hidden_size,
            eps=config.rms_norm_eps,
            out_norm_init=getattr(config, "memres_readout_norm_init", 1.0),
        )
        assert self.memory_block.hidden_size == config.hidden_size
        assert self.memory_readout.W_V.out_features == config.hidden_size
        # Block AttnRes router: one pseudo-query w_{n,i} per routed sublayer
        # after the first. AttnRes treats attention and MLP as separate layers;
        # the first attention sublayer receives b_0 = inputs_embeds directly.
        # In "simple_gate" mode the router is unused on the forward path but
        # the parameters are still constructed so checkpoints stay shape-stable
        # across modes.
        # When the canonical mode is "attention_parity" we configure the router
        # with a strong positive bias on the most-recent source and a strongly
        # negative mem_bias so the softmax is one-hot at step 0; combined with
        # the cumulative-state value pool (see forward()) this makes the
        # attention pool forward-identical to the bare backbone.  The
        # +/- 32 magnitude is chosen so that alpha_other ~ exp(-32)/N is
        # well below bf16 precision (~1e-14) for every off-recent source.
        # Empirically (results/eval/init_parity_test.json) this
        # is what's needed to keep the depth-compounded perturbation
        # under the 1e-3 logit tolerance across 2L = 56 routed sublayers
        # of Qwen3-0.6B; +/- 16 leaves ~3e-7 mass per off-source which
        # compounds to ~3e-1 in the final logits because the routed
        # error feeds back into the next sublayer's input.
        parity_init = (
            getattr(config, "memres_mode", "attention_parity")
            == "attention_parity"
        )
        # Resolve router init biases.  Explicit overrides on the config
        # (e.g. softened parity at -4/+4) take precedence; otherwise we
        # use the mode-derived defaults documented above.
        mem_bias_default = -32.0 if parity_init else -8.0
        recent_bias_default = 32.0 if parity_init else 0.0
        mem_bias_init = getattr(config, "router_mem_bias_init", None)
        recent_bias_init = getattr(config, "router_recent_bias_init", None)
        if mem_bias_init is None:
            mem_bias_init = mem_bias_default
        if recent_bias_init is None:
            recent_bias_init = recent_bias_default
        self.depth_router = BlockAttnResRouter(
            config.hidden_size,
            num_routing_steps=max(2 * config.num_hidden_layers - 1, 0),
            eps=config.rms_norm_eps,
            mem_bias_init=float(mem_bias_init),
            recent_bias_init=float(recent_bias_init),
        )
        # Per-sublayer gate for residual memory injection.  We have one gate
        # per attention or MLP transform (so 2 * num_hidden_layers gates),
        # mirroring the depth profile used by the routing diagnostic.
        self.memory_gate = MemoryGate(
            num_routing_steps=max(2 * config.num_hidden_layers, 1),
            init=float(getattr(config, "memres_gate_init", 0.0)),
        )

        # Block partition over AttnRes sublayers, not decoder blocks.  Boundaries
        # are balanced so the number of completed block summaries is exactly N
        # (capped by the number of sublayers), preserving the paper's N+2 pool
        # bound even when total_sublayers is not divisible by N.
        total_sublayers = max(2 * config.num_hidden_layers, 1)
        num_blocks = min(max(config.memres_num_blocks, 1), total_sublayers)
        self._num_memres_blocks = num_blocks
        self._block_end_sublayers = {
            (total_sublayers * i + num_blocks - 1) // num_blocks
            for i in range(1, num_blocks + 1)
        }

        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        _init_memres_params(module, self.config.hidden_size)

    # -- convenience helpers ------------------------------------------------

    def compress_session(
        self,
        C: torch.Tensor,
        M_c_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Two-stage compression: extract + judge. Returns M_c (B, K, d)."""
        return self.memory_block(C, M_c_prev)

    def readout_memory(self, X: torch.Tensor, M_c: torch.Tensor) -> torch.Tensor:
        """Per-position readout: X (B, S, d) queries M_c (B, K, d) -> (B, S, d)."""
        return self.memory_readout(X, M_c)

    def _resolve_extract_layer(self) -> int:
        """Return the integer layer index to use for the extract source.

        - "embed"       -> -1   (raw token-embedding lookup; legacy)
        - "hidden_<L>"  ->  L   (mid-layer contextualised hidden state;
                                 L is interpreted as a 1-indexed layer
                                 of the backbone, matching the position
                                 in BaseModelOutputWithPast.hidden_states
                                 -- index 0 is post-embedding, index L is
                                 the output of the L-th decoder layer).
        """
        src = getattr(self.config, "memres_extract_source", "embed")
        if src == "embed":
            return -1
        if src.startswith("hidden_"):
            try:
                return int(src.split("_", 1)[1])
            except ValueError as e:
                raise ValueError(
                    f"Unparsable memres_extract_source={src!r}; "
                    f"expected 'embed' or 'hidden_<int>'."
                ) from e
        raise ValueError(
            f"Unknown memres_extract_source={src!r}; "
            f"expected 'embed' or 'hidden_<int>'."
        )

    @torch.no_grad()
    def extract_source(
        self,
        input_ids: torch.LongTensor,
        layer: int | None = None,
    ) -> torch.Tensor:
        """Compute the per-token representation C_t fed to MemoryBlock.extract.

        The default behaviour is governed by ``config.memres_extract_source``:

        - ``"embed"`` returns the raw token-embedding lookup (legacy
          bag-of-token-embeddings extraction).  No backbone forward is run
          and the result is gradient-tracked through the embedding table
          (matching prior chain-trainer behaviour).
        - ``"hidden_<L>"`` runs a no_grad standard-residual partial forward
          (no memory injection, no Block-AttnRes routing -- just the bare
          ``h <- h + attn(h) + mlp(h)`` recurrence) over the first ``L``
          decoder layers and returns the post-layer-L hidden state.  This
          gives the compressor a contextualised representation of the
          session (syntax, anaphora, entity binding) so it can extract
          *semantic content* rather than bag-of-tokens.  The result is
          detached so the extract step adds gradient only through the
          M_in / extraction-layer parameters and the judge step, never
          through the backbone for *this* session's extraction.

        The early-exit at layer L means cost scales as L/L_total of one
        backbone forward (e.g. layer 14 on a 28-layer Qwen3-0.6B is ~50%
        of one full forward pass per session boundary).

        Why a *standard-residual* forward and not the Block-AttnRes
        routing forward with ``M_c=None``?  Because at parity init the two
        are bit-identical, and after training the routing pool's learned
        pseudo-queries reflect *how the model recruits memory*, which is
        exactly the signal we do not want to bake into the extract source
        (it would couple extract with read-side learning in a way that
        confounds the empirical claim that "contextualised extract beats
        bag-of-token extract"). Standard-residual extract gives a clean,
        bare-backbone-shaped representation throughout training.
        """
        if layer is None:
            layer = self._resolve_extract_layer()
        if layer < 0:
            # Legacy bag-of-token-embeddings path.  NOT wrapped in
            # no_grad: the embedding table is shared with the LM head and
            # gradients should flow through it the same way they do in
            # the pre-extract_source codebase.
            with torch.enable_grad():
                return self.embed_tokens(input_ids)

        # Bare-backbone partial forward over layers [0, layer).
        # Save and restore train mode so any layer-internal dropout uses
        # eval semantics (extract should be deterministic given input_ids).
        was_training = self.training
        self.eval()
        try:
            n_layers = len(self.layers)
            if not (1 <= layer <= n_layers):
                raise ValueError(
                    f"extract_source layer={layer} out of range [1, {n_layers}] "
                    f"for this backbone."
                )

            inputs_embeds = self.embed_tokens(input_ids)
            B, S = inputs_embeds.shape[:2]
            device = inputs_embeds.device

            # Position ids and rotary embeddings.
            position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

            # Plain causal mask (additive bf16 bias).  The backbone's
            # standard mask machinery wants more arguments than we have
            # context for here, so we build the simple full-causal mask
            # by hand -- which is what the chain trainer also does
            # (single full_attention type).
            neg_inf = torch.finfo(inputs_embeds.dtype).min
            causal_bias = torch.zeros(
                1, 1, S, S, device=device, dtype=inputs_embeds.dtype
            )
            upper = torch.triu(
                torch.ones(S, S, device=device), diagonal=1
            ).bool()
            causal_bias[:, :, upper] = neg_inf

            h = inputs_embeds
            for i in range(layer):
                lyr = self.layers[i]
                # Standard residual recurrence: h = h + attn(h) + mlp(h).
                # No Block-AttnRes routing pool, no memory injection -- this
                # is the "what bare Qwen3 thinks of this session" signal.
                attn_delta = lyr.attention_delta(
                    hidden_states=h,
                    attention_mask=causal_bias,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=position_embeddings,
                )
                h = h + attn_delta
                mlp_delta = lyr.mlp_delta(h)
                h = h + mlp_delta
        finally:
            if was_training:
                self.train()
        return h.detach()

    def compute_memory(
        self,
        history_ids: torch.Tensor,
        M_c_prev: torch.Tensor | None = None,
        detach_embeddings: bool = False,
    ) -> torch.Tensor:
        """history_ids (B, N) -> M_c (B, K, d).

        Returns only the persistent memory matrix M_c.  The per-position
        readout m^t depends on the *current* session and is therefore
        computed inside ``forward`` from ``M_c`` and ``inputs_embeds``.

        Honours ``config.memres_extract_source``: if set to a
        ``"hidden_<L>"`` policy, the bare-backbone hidden state at layer
        L is used instead of raw token embeddings.

        When detach_embeddings is True the embedding lookup is detached so
        gradients flow through the memory block but not through the shared
        embedding table for the history tokens.  Only meaningful in the
        legacy ``"embed"`` extract source (the hidden-state path always
        detaches; see ``extract_source``).
        """
        C = self.extract_source(history_ids)
        if detach_embeddings and self._resolve_extract_layer() < 0:
            C = C.detach()
        return self.compress_session(C, M_c_prev)

    # -- forward ------------------------------------------------------------

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring(
        custom_args="""
        history_ids (`torch.LongTensor` of shape `(batch_size, history_length)`, *optional*):
            Token ids of past session(s). Embedded and compressed into `M_c` by the MemoryBlock.
            Ignored when `M_c` is passed directly.
        M_c (`torch.Tensor` of shape `(batch_size, num_vectors, hidden_size)`, *optional*):
            Pre-computed memory matrix. Takes precedence over `history_ids`. The per-position
            readout m^t is recomputed each forward from (`inputs_embeds`, `M_c`) because it is
            query-dependent on the current session.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Positions used to update and read the generation cache.
        collect_alpha_trace (`bool`, *optional*, defaults to `False`):
            When True, attach `alpha_trace` (list of `(B, S)` tensors, one per routing layer)
            to the output.
        """
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        history_ids: torch.LongTensor | None = None,
        M_c: torch.Tensor | None = None,
        collect_alpha_trace: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # ── Resolve memory: history_ids -> M_c ──
        if M_c is None and history_ids is not None:
            C = self.embed_tokens(history_ids)
            M_c = self.compress_session(C)

        # Per-position readout (Eq. 6): m^t = CrossAttn(inputs_embeds, M_c)
        m_t = self.readout_memory(inputs_embeds, M_c) if M_c is not None else None

        # ── Standard cache / position setup ──
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen,
                past_seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # ── Causal mask ──
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            import inspect as _inspect

            _embed_kw = (
                "input_embeds"
                if "input_embeds" in _inspect.signature(create_causal_mask).parameters
                else "inputs_embeds"
            )
            mask_kwargs = {
                "config": self.config,
                _embed_kw: inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # ── Layer loop dispatch ──
        # In "simple_gate" mode we keep the standard pretrained residual flow
        #     h_post = h_pre + f(h_pre)
        # and inject the memory readout m^t additively at each sublayer
        # input through a per-sublayer ReZero-style gate
        #     h_pre = h_post_prev + sigma(g_l) * m^t.
        # In "attention_base" / "attention_parity" mode we recover the
        # original Block Attention Residuals routing (Eqs. 9-10), with m^t
        # entering as b_{-1}.
        b_minus_1 = m_t  # may be None
        has_memory = b_minus_1 is not None
        alpha_trace: list | None = [] if collect_alpha_trace else None
        memres_mode = _normalise_memres_mode(
            getattr(self.config, "memres_mode", "simple_gate"),
            getattr(self.config, "block_attnres_parity_init", None),
        )

        if memres_mode == "simple_gate":
            h_post = inputs_embeds
            sublayer_idx = 0

            def inject_mem(h: torch.Tensor) -> torch.Tensor:
                nonlocal sublayer_idx
                if has_memory:
                    alpha = self.memory_gate.alpha(sublayer_idx).to(h.dtype)
                    out = h + alpha * b_minus_1
                    if alpha_trace is not None:
                        # Broadcast the scalar gate as an (B, S) tensor for
                        # symmetry with the block_attnres path.
                        alpha_trace.append(
                            alpha.expand(h.shape[0], h.shape[1]).detach()
                        )
                else:
                    out = h
                sublayer_idx += 1
                return out

            for layer_idx, layer in enumerate(self.layers):
                layer_mask = causal_mask_mapping[layer.attention_type]

                # Attention sublayer.
                h_pre = inject_mem(h_post)
                if self.gradient_checkpointing and self.training:
                    attn_delta = self._gradient_checkpointing_func(
                        layer.attention_delta,
                        h_pre,
                        layer_mask,
                        position_ids,
                        past_key_values,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    attn_delta = layer.attention_delta(
                        hidden_states=h_pre,
                        attention_mask=layer_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                h_post = h_pre + attn_delta

                # MLP sublayer.
                h_pre = inject_mem(h_post)
                if self.gradient_checkpointing and self.training:
                    mlp_delta = self._gradient_checkpointing_func(
                        layer.mlp_delta, h_pre
                    )
                else:
                    mlp_delta = layer.mlp_delta(h_pre)
                h_post = h_pre + mlp_delta

            hidden_states = self.norm(h_post)
        else:
            # Attention-pool routing (Eqs. 9-10).
            #
            # Two pool conventions are supported.  In "attention_base" the
            # pool stores *deltas* -- b_0 is the embedding and each
            # completed block summary / running partial sum is a sum of
            # sublayer deltas.  In this regime the standard residual stream
            #     h_{n,i-1} = b_0 + sum_k b_k + b_n^{i-1}
            # is decomposed across multiple slots, so a softmax that is
            # constrained to weights summing to 1 cannot reconstruct it
            # from any one-hot init -- forward init-parity with the bare
            # backbone is therefore impossible in this regime.
            #
            # In "attention_parity" mode the pool stores *cumulative
            # hidden-state checkpoints*
            #     b_0 = h_0,   b_k = h_k (after block k),   partial = h_curr
            # so the most-recent source IS the standard residual stream
            # input to the next sublayer.  Combined with the router's
            # `recent_bias` init this makes the augmented model
            # forward-identical to the bare backbone at step 0 even in
            # attention-pool mode.
            parity_init = memres_mode == "attention_parity"
            b_0 = inputs_embeds
            completed_blocks: list[torch.Tensor] = []
            partial_sum: torch.Tensor | None = None

            def value_pool() -> list[torch.Tensor]:
                pool: list[torch.Tensor] = []
                if has_memory:
                    pool.append(b_minus_1)
                pool.append(b_0)
                pool.extend(completed_blocks)
                if partial_sum is not None:
                    pool.append(partial_sum)
                return pool

            h_pre = inputs_embeds
            h_post = inputs_embeds
            sublayer_idx = 0

            def route_if_needed() -> torch.Tensor:
                nonlocal sublayer_idx
                if sublayer_idx == 0:
                    return inputs_embeds
                routed, alpha_mem = self.depth_router.route(
                    sublayer_idx - 1, value_pool(), has_memory=has_memory
                )
                if alpha_trace is not None and alpha_mem is not None:
                    alpha_trace.append(alpha_mem)
                return routed

            def accumulate(delta: torch.Tensor, h_post_after: torch.Tensor) -> None:
                nonlocal partial_sum, sublayer_idx
                if parity_init:
                    # Cumulative checkpoint -- partial_sum tracks h itself.
                    partial_sum = h_post_after
                else:
                    partial_sum = (
                        delta if partial_sum is None else partial_sum + delta
                    )
                sublayer_idx += 1
                if sublayer_idx in self._block_end_sublayers:
                    completed_blocks.append(partial_sum)
                    partial_sum = None

            for layer_idx, layer in enumerate(self.layers):
                layer_mask = causal_mask_mapping[layer.attention_type]
                h_pre = route_if_needed()
                if self.gradient_checkpointing and self.training:
                    attn_delta = self._gradient_checkpointing_func(
                        layer.attention_delta,
                        h_pre,
                        layer_mask,
                        position_ids,
                        past_key_values,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    attn_delta = layer.attention_delta(
                        hidden_states=h_pre,
                        attention_mask=layer_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                h_post = h_pre + attn_delta
                accumulate(attn_delta, h_post)

                h_pre = route_if_needed()
                if self.gradient_checkpointing and self.training:
                    mlp_delta = self._gradient_checkpointing_func(
                        layer.mlp_delta, h_pre
                    )
                else:
                    mlp_delta = layer.mlp_delta(h_pre)
                h_post = h_pre + mlp_delta
                accumulate(mlp_delta, h_post)

            hidden_states = self.norm(h_post)

        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        if alpha_trace is not None:
            out.alpha_trace = alpha_trace
        return out


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------


class Qwen3MemResForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3 causal LM with Memory Residuals.

    Quick-start::

        # Training (history_ids automatically compressed to M_c)
        out = model(input_ids=current, labels=current, history_ids=prev_session)

        # Efficient multi-turn generation: pre-compute M_c once; the
        # per-position readout m^t is recomputed each forward from M_c
        # and the current session's embeddings.
        M_c = model.model.compute_memory(history_ids)
        tokens = model.generate(input_ids=prompt, M_c=M_c, max_new_tokens=200)

        # Recurrent multi-session update
        M_c_1 = model.model.compute_memory(session_1_ids)
        M_c_2 = model.model.compute_memory(session_2_ids, M_c_prev=M_c_1)
        tokens = model.generate(input_ids=prompt, M_c=M_c_2, max_new_tokens=200)
    """

    config_class = Qwen3MemResConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen3MemResConfig):
        super().__init__(config)
        self.model = Qwen3MemResModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        _init_memres_params(module, self.config.hidden_size)

    @can_return_tuple
    @auto_docstring(
        custom_args="""
        history_ids (`torch.LongTensor` of shape `(batch_size, history_length)`, *optional*):
            Past-session token ids, compressed to memory via the backbone's MemoryBlock.
        M_c (`torch.Tensor` of shape `(batch_size, num_vectors, hidden_size)`, *optional*):
            Pre-computed memory matrix. The per-position readout m^t is computed inside
            the backbone forward.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Positions used to update and read the generation cache.
        collect_alpha_trace (`bool`, *optional*, defaults to `False`):
            Expose per-layer memory routing mass as `alpha_trace` on the output.
        """
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        history_ids: torch.LongTensor | None = None,
        M_c: torch.Tensor | None = None,
        collect_alpha_trace: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            history_ids=history_ids,
            M_c=M_c,
            collect_alpha_trace=collect_alpha_trace,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_idx = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
        if getattr(outputs, "alpha_trace", None) is not None:
            result.alpha_trace = outputs.alpha_trace
        return result
