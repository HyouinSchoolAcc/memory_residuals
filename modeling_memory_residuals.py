"""
Memory Residuals: End-to-End Differentiable Lifelong Memory for Conversational Agents

Earlier reference variant — kept for comparison.  The canonical implementation
lives in modeling_memres.py.

Two advances over standard Qwen3:

1. MemoryBlock — Two-Stage QKV Competition (Section 2.1):

   Stage 1 (Extraction, Eq. 1):
       E_t = Softmax(M_in W_Q^ext (C_t W_K^ext)^T / sqrt(d)) C_t W_V^ext

   Stage 2 (Judging, Eq. 2):
       P_judge = [M_c^{t-1} || E_t]
       M_c^t   = Softmax(M_judge W_Q^judge (P_judge W_K^judge)^T / sqrt(d)) P_judge W_V^judge

   Zero-sum softmax across 2K forces old-vs-new competition (Forgetting Defense).

2. Depth-Wise Residual Stream Injection (Section 2.2):

   Readout (Eq. 6):   m^t = Softmax(r^T M_c^T / sqrt(d)) M_c
   Foundation (Eq. 7): v_0 := m^t
   Routing (Eq. 8):    h_l = sum alpha_{i->l} v_i ,  phi(q,k) = exp(q^T RMSNorm(k))

When memory is absent the model reduces to vanilla Qwen3.
"""

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
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class MemResConfig(Qwen3Config):
    """Qwen3Config extended with Memory Residuals hyper-parameters."""

    model_type = "qwen3_memres"

    def __init__(
        self,
        memres_num_vectors: int = 128,
        memres_judging_depth: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memres_num_vectors = memres_num_vectors
        self.memres_judging_depth = memres_judging_depth


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
# Memory block — Two-Stage QKV Competition (Section 2.1)
# ---------------------------------------------------------------------------


class MemoryBlock(nn.Module):
    """
    Two-Stage QKV Competition (Section 2.1).

    Stage 1 — Extraction (Eq. 1):
        Compress session C_t into candidate E_t via learnable M_in queries.

    Stage 2 — Judging (Eq. 2):
        P_judge = [M_c^{t-1} || E_t], M_c^t = CrossAttn(M_judge, P_judge).
        Softmax over 2K forces zero-sum competition (Forgetting Defense).

    Optional multi-layer judging depth L_J (Section 2.1.1, Eq. 3-5).
    """

    def __init__(self, hidden_size: int, num_vectors: int, judging_depth: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_vectors = num_vectors
        self.judging_depth = judging_depth

        # Stage 1: extraction queries M_in (K x d)
        self.M_in = nn.Parameter(torch.empty(num_vectors, hidden_size))
        nn.init.normal_(self.M_in, std=hidden_size**-0.5)
        self.extraction = CrossAttention(hidden_size)

        # Stage 2: judging queries M_judge (K x d)
        self.M_judge = nn.Parameter(torch.empty(num_vectors, hidden_size))
        nn.init.normal_(self.M_judge, std=hidden_size**-0.5)

        if judging_depth <= 1:
            self.judging = CrossAttention(hidden_size)
        else:
            self.judging_layers = nn.ModuleList(
                [CrossAttention(hidden_size) for _ in range(judging_depth)]
            )
            self.readout = CrossAttention(hidden_size)

    def forward(
        self, C: torch.Tensor, M_c_prev: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            C: (B, N, d) — session token representations
            M_c_prev: (B, K, d) — prior memory state; None -> zeros
        Returns:
            M_c: (B, K, d) — updated memory state
        """
        B = C.size(0)

        # Stage 1: Extraction
        M_in = self.M_in.unsqueeze(0).expand(B, -1, -1)
        E_t = self.extraction(M_in, C)

        # Stage 2: Judging
        if M_c_prev is None:
            M_c_prev = torch.zeros_like(E_t)
        P_judge = torch.cat([M_c_prev, E_t], dim=1)  # (B, 2K, d)
        M_judge = self.M_judge.unsqueeze(0).expand(B, -1, -1)

        if self.judging_depth <= 1:
            return self.judging(M_judge, P_judge)

        for layer in self.judging_layers:
            P_judge = P_judge + layer(M_judge, P_judge)
        return self.readout(M_judge, P_judge)


# ---------------------------------------------------------------------------
# Memory Readout (Eq. 6)
# ---------------------------------------------------------------------------


class MemoryReadout(nn.Module):
    """m^t = Softmax(r^T M_c^T / sqrt(d)) M_c"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = hidden_size**-0.5
        self.r = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.r, std=hidden_size**-0.5)

    def forward(self, M_c: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("d,bkd->bk", self.r, M_c) * self.scale
        attn = F.softmax(scores, dim=-1)
        return torch.einsum("bk,bkd->bd", attn, M_c)


# ---------------------------------------------------------------------------
# Depth-Wise Router (Eq. 7-9)
# ---------------------------------------------------------------------------


class DepthWiseRouter(nn.Module):
    """phi(q,k) = exp(q^T RMSNorm(k)), w_l per routing layer."""

    def __init__(self, hidden_size: int, num_routing_layers: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size)) for _ in range(num_routing_layers)]
        )
        for w in self.w:
            nn.init.normal_(w, std=0.01)
        self.norm = Qwen3RMSNorm(hidden_size, eps=eps)

    def route(
        self, router_idx: int, sources: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.w[router_idx]
        stacked = torch.stack(sources, dim=2)  # (B, S, N_src, d)
        normed = self.norm(stacked)
        scores = torch.einsum("d,bsnd->bsn", w, normed)
        alphas = F.softmax(scores, dim=-1)
        h_l = torch.einsum("bsn,bsnd->bsd", alphas, stacked)
        alpha_mem = alphas[:, :, 0]
        return h_l, alpha_mem


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class MemResDecoderLayer(GradientCheckpointingLayer):
    """Standard Qwen3 decoder layer. Depth-wise routing is external."""

    def __init__(self, config: MemResConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
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
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------


class MemResModel(Qwen3PreTrainedModel):
    """Qwen3 backbone with Memory Residuals."""

    config_class = MemResConfig

    def __init__(self, config: MemResConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [MemResDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.memory_block = MemoryBlock(
            config.hidden_size, config.memres_num_vectors, config.memres_judging_depth
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

    def compute_memory(
        self, history_ids: torch.Tensor, M_c_prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C = self.embed_tokens(history_ids)
        M_c = self.memory_block(C, M_c_prev)
        m_t = self.memory_readout(M_c)
        return M_c, m_t

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
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
        m_t: torch.Tensor | None = None,
        collect_alpha_trace: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        history_ids: (B, N) past-session tokens -> compressed to memory.
        M_c: (B, K, d) pre-computed memory matrix.
        m_t: (B, d) pre-computed readout vector.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if m_t is None:
            if M_c is None and history_ids is not None:
                C = self.embed_tokens(history_ids)
                M_c = self.memory_block(C)
            if M_c is not None:
                m_t = self.memory_readout(M_c)

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

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = dict(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        B, S, d = inputs_embeds.shape
        use_routing = m_t is not None
        alpha_trace: list | None = [] if collect_alpha_trace else None

        if use_routing:
            v_0 = m_t.unsqueeze(1).expand(B, S, d)
            sources: list[torch.Tensor] = [v_0]

        hidden_states = inputs_embeds

        for layer_idx, layer in enumerate(self.layers):
            if use_routing and layer_idx > 0:
                hidden_states, alpha_mem = self.depth_router.route(
                    layer_idx - 1, sources
                )
                if alpha_trace is not None:
                    alpha_trace.append(alpha_mem)

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


class MemResForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3 causal LM with Memory Residuals.

    Quick-start::
        out = model(input_ids=current, labels=current, history_ids=prev_session)

        M_c, m_t = model.model.compute_memory(history_ids)
        tokens = model.generate(input_ids=prompt, m_t=m_t, max_new_tokens=200)
    """

    config_class = MemResConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MemResConfig):
        super().__init__(config)
        self.model = MemResModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
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
        m_t: torch.Tensor | None = None,
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
            m_t=m_t,
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
