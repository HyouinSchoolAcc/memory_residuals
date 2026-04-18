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

2. Depth-Wise Residual Stream Injection (Section 2.2):

   Readout (Eq. 6) — per-position cross-attention over M_c with the
   current session's token embeddings as queries:
       m^t = Softmax(X W_Q^read (M_c^t W_K^read)^T / sqrt(d)) M_c^t W_V^read
       m^t in R^{S x d}  — matches the shape of any attention layer output.

   Register as foundational source (Eq. 7):
       v_0 := m^t  in R^{S x d}

   Depth-wise attention (Eq. 8):
       h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
       alpha_{i->l} = phi(w_l, k_i) / sum_j phi(w_l, k_j)
       phi(q, k) = exp(q^T RMSNorm(k))

   Each layer's input is a softmax-weighted mix of ALL preceding layer
   outputs plus the memory vector.  During filler turns the local layer
   outputs dominate; during callbacks mass shifts onto v_0 = m^t.

When no memory is provided (m_t is None) the model reduces to vanilla Qwen3.
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memres_num_vectors = memres_num_vectors
        # L_E: number of Perceiver-style refinement layers stacked on top of
        # the initial M_in cross-attention (Eq. 4).  L_E = 0 recovers Eq. 1.
        self.memres_extraction_depth = memres_extraction_depth


# ---------------------------------------------------------------------------
# Weight init helper
# ---------------------------------------------------------------------------


def _init_memres_params(module: nn.Module, hidden_size: int) -> None:
    """Initialize raw nn.Parameter attributes on MemRes modules.

    HF's `_init_weights` only touches Linear/Embedding/Norms, so our custom
    Parameters (M_in, M_judge, depth_router.w[...]) stay uninitialized (NaN
    garbage) after `from_pretrained` unless we init them explicitly.
    """
    std = hidden_size**-0.5
    if isinstance(module, MemoryBlock):
        nn.init.normal_(module.M_in, std=std)
        nn.init.normal_(module.M_judge, std=std)
    elif isinstance(module, DepthWiseRouter):
        for w in module.w:
            nn.init.normal_(w, std=0.01)


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
        return self.judging(M_judge, P_judge)

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
# Depth-Wise Router  (Section 2.2, Eq. 7-9)
# ---------------------------------------------------------------------------


class DepthWiseRouter(nn.Module):
    """
    Depth-wise attention routing (Eq. 8).

    h_l = sum_{i=0}^{l-1}  alpha_{i->l} * v_i

    alpha_{i->l} = phi(w_l, k_i) / sum_j phi(w_l, k_j)
    phi(q, k)    = exp(q^T RMSNorm(k))

    w_l in R^d is a learned pseudo-query for routing layer l.
    """

    def __init__(self, hidden_size: int, num_routing_layers: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size)) for _ in range(num_routing_layers)]
        )
        for w in self.w:
            nn.init.normal_(w, std=0.01)
        self.norm = Qwen3RMSNorm(hidden_size, eps=eps)

    def route(
        self,
        router_idx: int,
        sources: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute h_l and memory-attention mass alpha_{0->l}.

        Args:
            router_idx: index into self.w  (0 for the second decoder layer,
                        1 for the third, etc.)
            sources: [v_0, v_1, ..., v_l] where v_0 = m^t (B,S,d) and the
                     rest are layer outputs (B,S,d).
        Returns:
            h_l:       (B, S, d) — routed input for the next layer.
            alpha_mem: (B, S)    — attention mass on v_0 (memory).
        """
        w = self.w[router_idx]  # (d,)

        # Stack sources -> (B, S, num_sources, d)
        stacked = torch.stack(sources, dim=2)

        # phi(w, k_i) = exp(w^T RMSNorm(v_i))
        normed = self.norm(stacked)
        scores = torch.einsum("d,bsnd->bsn", w, normed)  # (B, S, N_src)

        # softmax over sources for each (batch, position)
        alphas = F.softmax(scores, dim=-1)  # (B, S, N_src)

        # weighted sum
        h_l = torch.einsum("bsn,bsnd->bsd", alphas, stacked)  # (B, S, d)

        # memory mass = alpha on source 0
        alpha_mem = alphas[:, :, 0]  # (B, S)

        return h_l, alpha_mem


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class Qwen3MemResDecoderLayer(GradientCheckpointingLayer):
    """Standard Qwen3 decoder layer (self-attention + MLP with internal
    residual connections).  Depth-wise routing across layers is handled by
    the model backbone, not by individual layers."""

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
        # ── Attention sublayer with residual ──
        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + attn_out

        # ── MLP sublayer with residual ──
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
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
        self.memory_block = MemoryBlock(
            config.hidden_size,
            config.memres_num_vectors,
            config.memres_extraction_depth,
        )
        self.memory_readout = MemoryReadout(config.hidden_size)
        self.depth_router = DepthWiseRouter(
            config.hidden_size,
            num_routing_layers=max(config.num_hidden_layers - 1, 0),
            eps=config.rms_norm_eps,
        )

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
        detach_embeddings: bool = True,
    ) -> torch.Tensor:
        """history_ids (B, N) -> M_c (B, K, d).

        Returns only the persistent memory matrix M_c.  The per-position
        readout m^t depends on the *current* session and is therefore
        computed inside ``forward`` from ``M_c`` and ``inputs_embeds``.

        When detach_embeddings is True the embedding lookup is detached so
        gradients flow through the memory block but not through the shared
        embedding table for the history tokens (saves memory).
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

        # ── Forward through layers with depth-wise routing ──
        use_routing = m_t is not None
        alpha_trace: list | None = [] if collect_alpha_trace else None

        if use_routing:
            # v_0 := m^t is already (B, S, d) — Eq. 7, no broadcasting needed.
            sources: list[torch.Tensor] = [m_t]

        # Layer 0 always receives embeddings directly
        hidden_states = inputs_embeds

        for layer_idx, layer in enumerate(self.layers):
            # Depth-wise routing for layers 1+ when memory is present (Eq. 8)
            if use_routing and layer_idx > 0:
                hidden_states, alpha_mem = self.depth_router.route(
                    layer_idx - 1, sources
                )
                if alpha_trace is not None:
                    alpha_trace.append(alpha_mem)

            # Run standard transformer layer
            layer_mask = causal_mask_mapping[layer.attention_type]
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    layer_mask,
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            # Store layer output as source v_{l+1} for subsequent routing
            if use_routing:
                sources.append(hidden_states)

        hidden_states = self.norm(hidden_states)

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
