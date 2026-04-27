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


class Qwen3MemResConfig(Qwen3Config):
    """Qwen3Config extended with Memory Residuals hyper-parameters."""

    model_type = "qwen3_memres"

    def __init__(
        self,
        memres_num_vectors: int = 128,
        memres_extraction_depth: int = 0,
        memres_num_blocks: int = 4,
        memres_mode: str = "residual",
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
        # Routing mode.  "residual" (default) preserves the pretrained
        # backbone's standard residual flow and injects memory through a
        # per-sublayer gated additive contribution
        #     h_pre_l = h_residual_l + sigma(g_l) * m^t
        # where the per-sublayer gate scalar g_l is initialised to a strong
        # negative number (sigma(g_l) ~ 0) so the augmented model is
        # behaviourally identical to the bare Qwen3 backbone at step 0.  This
        # is the recommended mode whenever the backbone is pretrained.
        # "block_attnres" recovers the original Block Attention Residuals
        # routing (Eqs. 9-10) and is intended for from-scratch ablations.
        if memres_mode not in {"residual", "block_attnres"}:
            raise ValueError(
                f"Unknown memres_mode {memres_mode!r}; choose 'residual' or "
                "'block_attnres'."
            )
        self.memres_mode = memres_mode


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
        nn.init.normal_(module.M_in, std=std)
        nn.init.normal_(module.M_judge, std=std)
    elif isinstance(module, MemoryReadout):
        # Read-out V projection stays at the default normal init so m^t has
        # well-scaled non-zero magnitude at step 0; the augmented model is
        # still identical to the bare backbone because the per-sublayer
        # MemoryGate is zero-initialised, so  h + gate * m^t == h .  This
        # gives a non-zero gradient signal back into the gate from step 1
        # (the saturating-sigmoid alternative collapses gate gradients).
        pass
    elif isinstance(module, BlockAttnResRouter):
        for w in module.w:
            nn.init.zeros_(w)
        # Strong negative init for the per-sublayer memory-score bias so the
        # initial alpha_mem ~ exp(mem_bias)/N is effectively zero.  This is
        # what makes the augmented model collapse onto the base model at
        # step 0 even though the depth-wise pseudo-queries are zero
        # (otherwise uniform softmax dilutes other sources by 1/N).
        nn.init.constant_(module.mem_bias, module._mem_bias_init)
    elif isinstance(module, MemoryGate):
        nn.init.constant_(module.gate, module._init)


# ---------------------------------------------------------------------------
# Cross-attention primitive
# ---------------------------------------------------------------------------


class CrossAttention(nn.Module):
    """Single cross-attention: queries attend to context key-values."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = hidden_size**-0.5
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(queries)
        K = self.W_K(context)
        V = self.W_V(context)
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ V


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
    """

    def __init__(
        self,
        hidden_size: int,
        num_vectors: int,
        extraction_depth: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_vectors = num_vectors
        self.extraction_depth = extraction_depth

        # Stage 1: learnable extraction queries M_in  (K x d)
        self.M_in = nn.Parameter(torch.empty(num_vectors, hidden_size))
        nn.init.normal_(self.M_in, std=hidden_size**-0.5)
        # Initial compression + L_E refinement layers (Eq. 3-4).
        # All share the same (K, d) -> (K, d) latent shape.
        self.extraction_layers = nn.ModuleList(
            [CrossAttention(hidden_size) for _ in range(1 + extraction_depth)]
        )

        # Stage 2: learnable judging queries M_judge  (K x d)
        self.M_judge = nn.Parameter(torch.empty(num_vectors, hidden_size))
        nn.init.normal_(self.M_judge, std=hidden_size**-0.5)
        self.judging = CrossAttention(hidden_size)
        # Per PITFALLS §2 and TRAINING_PLAYBOOK §"RMSNorm placement", apply
        # RMSNorm to M_c after the judge step so that repeated application
        # over many sessions does not let the recurrent state drift out of
        # the well-conditioned region the readout was trained on.  Without
        # this, eval over chains longer than the training TBPTT window k
        # exhibits unbounded growth of |M_c|_F and the readout becomes
        # numerically degenerate.
        self.judge_norm = Qwen3RMSNorm(hidden_size, eps=eps)

    def extract(self, C: torch.Tensor) -> torch.Tensor:
        """Stage 1: refine M_in queries over C_t for 1 + L_E layers.

        Initial layer compresses N -> K via M_in queries (Eq. 1 / 3).
        Subsequent layers refine the (K, d) state by letting it re-query
        C_t with a residual connection (Eq. 4).
        """
        B = C.size(0)
        M_in = self.M_in.unsqueeze(0).expand(B, -1, -1)
        E = self.extraction_layers[0](M_in, C)  # (B, K, d)
        for layer in self.extraction_layers[1:]:
            E = E + layer(E, C)
        return E

    def judge(self, M_c_prev: torch.Tensor, M_new: torch.Tensor) -> torch.Tensor:
        """Stage 2: single-round competition over [M_c^{t-1} || M_new]."""
        B = M_new.size(0)
        P_judge = torch.cat([M_c_prev, M_new], dim=1)  # (B, 2K, d)
        M_judge = self.M_judge.unsqueeze(0).expand(B, -1, -1)
        out = self.judging(M_judge, P_judge)
        # RMSNorm bounds the recurrent state.  See __init__ for rationale.
        return self.judge_norm(out)

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
        return self.judge(M_c_prev, M_new)


# ---------------------------------------------------------------------------
# Memory Readout  (Section 2.2, Eq. 6)
# ---------------------------------------------------------------------------


class MemoryReadout(nn.Module):
    """Per-position cross-attention readout over M_c (Eq. 6).

    Each position in the current session queries the K memory slots
    independently, producing m^t of shape (B, S, d) — identical to the
    shape of any standard attention layer output, so it drops directly
    into the depth-wise routing pool at v_0 without broadcasting.

        m^t = Softmax(X W_Q (M_c W_K)^T / sqrt(d)) M_c W_V
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = hidden_size**-0.5
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, X: torch.Tensor, M_c: torch.Tensor) -> torch.Tensor:
        """X: (B, S, d) queries;  M_c: (B, K, d) memory slots  ->  (B, S, d)"""
        Q = self.W_Q(X)
        K = self.W_K(M_c)
        V = self.W_V(M_c)
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ V


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
        if has_memory:
            # Additive bias on the memory source (index 0).  This shifts the
            # softmax distribution towards zero memory weight at init so that
            # the augmented model is behaviourally identical to the base
            # model when W_V_read = 0; gradients can later push the bias up.
            mem_bias = self.mem_bias[router_idx]
            bias_vec = torch.zeros_like(scores[..., :1])
            bias_vec = bias_vec + mem_bias
            zeros_rest = torch.zeros_like(scores[..., 1:])
            score_bias = torch.cat([bias_vec, zeros_rest], dim=-1)
            scores = scores + score_bias
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
        )
        self.memory_readout = MemoryReadout(config.hidden_size)
        assert self.memory_block.hidden_size == config.hidden_size
        assert self.memory_readout.W_V.out_features == config.hidden_size
        # Block AttnRes router: one pseudo-query w_{n,i} per routed sublayer
        # after the first. AttnRes treats attention and MLP as separate layers;
        # the first attention sublayer receives b_0 = inputs_embeds directly.
        # In "residual" mode the router is unused on the forward path but the
        # parameters are still constructed so checkpoints stay shape-stable
        # across modes.
        self.depth_router = BlockAttnResRouter(
            config.hidden_size,
            num_routing_steps=max(2 * config.num_hidden_layers - 1, 0),
            eps=config.rms_norm_eps,
        )
        # Per-sublayer gate for residual memory injection.  We have one gate
        # per attention or MLP transform (so 2 * num_hidden_layers gates),
        # mirroring the depth profile used by the routing diagnostic.
        self.memory_gate = MemoryGate(
            num_routing_steps=max(2 * config.num_hidden_layers, 1),
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

        When detach_embeddings is True the embedding lookup is detached so
        gradients flow through the memory block but not through the shared
        embedding table for the history tokens.  It is False by default to
        preserve the paper's end-to-end differentiability claim.
        """
        C = self.embed_tokens(history_ids)
        if detach_embeddings:
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
        # In "residual" mode we keep the standard pretrained residual flow
        #     h_post = h_pre + f(h_pre)
        # and inject the memory readout m^t additively at each sublayer
        # input through a per-sublayer ReZero-style gate
        #     h_pre = h_post_prev + sigma(g_l) * m^t.
        # In "block_attnres" mode we recover the original Block Attention
        # Residuals routing (Eqs. 9-10), with m^t entering as b_{-1}.
        b_minus_1 = m_t  # may be None
        has_memory = b_minus_1 is not None
        alpha_trace: list | None = [] if collect_alpha_trace else None
        memres_mode = getattr(self.config, "memres_mode", "residual")

        if memres_mode == "residual":
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
            # Block AttnRes routing (Eqs. 9-10).
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

            def accumulate(delta: torch.Tensor) -> None:
                nonlocal partial_sum, sublayer_idx
                partial_sum = delta if partial_sum is None else partial_sum + delta
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
                accumulate(attn_delta)

                h_pre = route_if_needed()
                if self.gradient_checkpointing and self.training:
                    mlp_delta = self._gradient_checkpointing_func(
                        layer.mlp_delta, h_pre
                    )
                else:
                    mlp_delta = layer.mlp_delta(h_pre)
                h_post = h_pre + mlp_delta
                accumulate(mlp_delta)

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
