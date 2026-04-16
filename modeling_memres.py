"""
Qwen3 with Memory Residuals (MemRes).

A MemoryBlock compresses a long history C ∈ R^(N×d) into a fixed-size
M_c ∈ R^(K×d) via cross-attention with K learnable latent queries.

Each decoder layer then pools hidden states with M_c:
    V_pool = [H || M_c],  α = softmax(H W_Q · (V_pool W_K)^T / √d)
    H_tilde = α · V_pool,  H_out = H_tilde + SelfAttn(LN(H_tilde))

Causal mask covers the local S×S block only; memory columns are globally
visible. When memory_state is None the model reduces to vanilla Qwen3.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
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
from transformers.utils import auto_docstring, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs


class Qwen3MemResConfig(Qwen3Config):
    model_type = "qwen3_memres"

    def __init__(
        self,
        memres_num_memory_vectors: int = 128,
        memres_num_memory_heads: int = 8,
        memres_apply_at: str = "both",
        memres_use_gate: bool = False,
        memres_gate_init: float = -2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memres_num_memory_vectors = memres_num_memory_vectors
        self.memres_num_memory_heads = memres_num_memory_heads
        self.memres_apply_at = memres_apply_at
        self.memres_use_gate = memres_use_gate
        self.memres_gate_init = memres_gate_init


class MemoryBlock(nn.Module):
    """Compress C ∈ R^(B×N×d) into M_c ∈ R^(B×K×d) via multi-head cross-attention
    with K learnable latent queries."""

    def __init__(
        self,
        hidden_size: int,
        num_memory_vectors: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_memory_vectors = num_memory_vectors
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.memory_queries = nn.Parameter(torch.zeros(num_memory_vectors, hidden_size))
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_norm = Qwen3RMSNorm(hidden_size, eps=norm_eps)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, l, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, h * dh)

    def forward(self, history_hidden: torch.Tensor) -> torch.Tensor:
        b = history_hidden.shape[0]
        queries = self.memory_queries.unsqueeze(0).expand(b, -1, -1)

        q = self._split_heads(self.q_proj(queries))
        k = self._split_heads(self.k_proj(history_hidden))
        v = self._split_heads(self.v_proj(history_hidden))

        scale = 1.0 / math.sqrt(self.head_dim)
        weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = self.out_proj(self._merge_heads(torch.matmul(weights, v)))
        return self.out_norm(out)

    @torch.no_grad()
    def blend(
        self,
        existing_memory: torch.Tensor,
        new_hidden: torch.Tensor,
        blend_alpha: float = 0.5,
    ) -> torch.Tensor:
        """Inference-time incremental update: M_c' = (1-α) M_c + α compress(new)."""
        return (1.0 - blend_alpha) * existing_memory + blend_alpha * self.forward(
            new_hidden
        )


class MemResProjections(nn.Module):
    """Memory-residual attention site: pools hidden_states with memory_state
    through a softmax over [H || M_c], with causal mask on the local block."""

    def __init__(
        self,
        hidden_size: int,
        use_gate: bool,
        gate_init: float,
        norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pre_norm = Qwen3RMSNorm(hidden_size, eps=norm_eps)

        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, gate_init)

    def _attention_logits(
        self, normed: torch.Tensor, memory_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_pool = torch.cat([normed, memory_state], dim=1)
        q = self.q_proj(normed)
        k = self.k_proj(v_pool)
        scale = 1.0 / math.sqrt(self.hidden_size)
        return torch.matmul(q, k.transpose(-2, -1)) * scale, v_pool

    @staticmethod
    def _apply_causal_mask(logits: torch.Tensor, S: int, K: int) -> torch.Tensor:
        local = torch.ones(S, S, dtype=torch.bool, device=logits.device).tril()
        mem = torch.ones(S, K, dtype=torch.bool, device=logits.device)
        mask = torch.cat([local, mem], dim=1)
        return logits.masked_fill(~mask, float("-inf"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        K = memory_state.shape[1]
        normed = self.pre_norm(hidden_states)

        logits, v_pool = self._attention_logits(normed, memory_state)
        if S > 1:
            logits = self._apply_causal_mask(logits, S, K)
        if attention_mask is not None:
            am = (
                attention_mask.squeeze(1)
                if attention_mask.dim() == 4
                else attention_mask
            )
            pad = torch.zeros(B, S, K, dtype=am.dtype, device=am.device)
            logits = logits + torch.cat([am, pad], dim=-1)

        h_tilde = torch.matmul(F.softmax(logits, dim=-1), v_pool)

        if self.use_gate:
            alpha = torch.sigmoid(self.gate(hidden_states))
            return (1.0 - alpha) * hidden_states + alpha * h_tilde
        return h_tilde

    @torch.no_grad()
    def memory_mass(
        self, hidden_states: torch.Tensor, memory_state: torch.Tensor
    ) -> torch.Tensor:
        """Per-position fraction of α routed to M_c columns. Returns (B, S)."""
        S = hidden_states.shape[1]
        K = memory_state.shape[1]
        normed = self.pre_norm(hidden_states)
        logits, _ = self._attention_logits(normed, memory_state)
        logits = self._apply_causal_mask(logits, S, K)
        return F.softmax(logits, dim=-1)[..., S:].sum(dim=-1)


class Qwen3MemResDecoderLayer(GradientCheckpointingLayer):
    """Standard Qwen3 decoder layer with memory-residual attention inserted
    before self-attention and/or MLP."""

    def __init__(self, config: Qwen3MemResConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

        apply_at = config.memres_apply_at
        self.use_memres_attn = apply_at in ("attn", "both")
        self.use_memres_mlp = apply_at in ("mlp", "both")

        proj_kwargs = dict(
            hidden_size=config.hidden_size,
            use_gate=config.memres_use_gate,
            gate_init=config.memres_gate_init,
            norm_eps=config.rms_norm_eps,
        )
        if self.use_memres_attn:
            self.memres_attn = MemResProjections(**proj_kwargs)
        if self.use_memres_mlp:
            self.memres_mlp = MemResProjections(**proj_kwargs)

    def _maybe_memres(
        self,
        projections: MemResProjections | None,
        hidden_states: torch.Tensor,
        memory_state: torch.Tensor | None,
        alpha_trace: list | None,
    ) -> torch.Tensor:
        if memory_state is None or projections is None:
            return hidden_states
        if alpha_trace is not None:
            alpha_trace.append(projections.memory_mass(hidden_states, memory_state))
        return projections(hidden_states, memory_state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        memory_state: torch.Tensor | None = None,
        alpha_trace: list | None = None,
        **kwargs,
    ) -> torch.Tensor:
        pre_attn = self._maybe_memres(
            getattr(self, "memres_attn", None) if self.use_memres_attn else None,
            hidden_states,
            memory_state,
            alpha_trace,
        )
        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(pre_attn),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = pre_attn + attn_out

        pre_mlp = self._maybe_memres(
            getattr(self, "memres_mlp", None) if self.use_memres_mlp else None,
            hidden_states,
            memory_state,
            alpha_trace,
        )
        return pre_mlp + self.mlp(self.post_attention_layernorm(pre_mlp))


class Qwen3MemResModel(Qwen3PreTrainedModel):
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

        self.memory_block = MemoryBlock(
            hidden_size=config.hidden_size,
            num_memory_vectors=config.memres_num_memory_vectors,
            num_heads=config.memres_num_memory_heads,
            norm_eps=config.rms_norm_eps,
        )

        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()

    def compress_history(self, history_hidden: torch.Tensor) -> torch.Tensor:
        return self.memory_block(history_hidden)

    def blend_memory(
        self,
        existing_memory: torch.Tensor,
        new_hidden: torch.Tensor,
        blend_alpha: float = 0.5,
    ) -> torch.Tensor:
        return self.memory_block.blend(existing_memory, new_hidden, blend_alpha)

    def _init_weights(self, module):
        super()._init_weights(module)
        # Zero-init on BOTH W_Q^res and W_K^res yields ∇W_Q = ∇W_K = 0 at step 0
        # (bilinear dead zone). Parent's small-normal init is kept for q/k.
        if isinstance(module, MemResProjections):
            if module.use_gate:
                nn.init.zeros_(module.gate.weight)
                nn.init.constant_(module.gate.bias, self.config.memres_gate_init)
        elif isinstance(module, MemoryBlock):
            std = getattr(self.config, "initializer_range", 0.02)
            nn.init.normal_(module.memory_queries, mean=0.0, std=std)

    def _build_causal_mask(
        self,
        attention_mask,
        inputs_embeds,
        cache_position,
        past_key_values,
        position_ids,
    ):
        if isinstance(attention_mask, dict):
            return attention_mask
        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            mapping["sliding_attention"] = create_sliding_window_causal_mask(
                **mask_kwargs
            )
        return mapping

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
        memory_state: torch.Tensor | None = None,
        collect_alpha_trace: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        Args:
            memory_state: optional (B, K, d) compressed memory. When None the
                layer stack reduces to plain Qwen3.
            collect_alpha_trace: when True, attach ``alpha_trace`` (list of
                (B, S) tensors, one per MemRes site) to the output.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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

        causal_mask_mapping = self._build_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, position_ids
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        alpha_trace: list | None = [] if collect_alpha_trace else None
        hidden_states = inputs_embeds
        for layer in self.layers:
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
                    memory_state,
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
                    memory_state=memory_state,
                    alpha_trace=alpha_trace,
                )

        hidden_states = self.norm(hidden_states)
        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        if alpha_trace is not None:
            out.alpha_trace = alpha_trace
        return out


class Qwen3MemResForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3 causal-LM with Memory Residuals.

    Typical multi-session usage::

        with torch.no_grad():
            h = model.model(input_ids=session1_ids).last_hidden_state
        m_c = model.model.compress_history(h.detach())
        out = model(input_ids=session2_ids, labels=session2_ids, memory_state=m_c)
    """

    config_class = Qwen3MemResConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen3MemResConfig):
        super().__init__(config)
        self.model = Qwen3MemResModel(config)
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
        memory_state: torch.Tensor | None = None,
        collect_alpha_trace: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            memory_state: optional (B, K, d) compressed memory.
            collect_alpha_trace: when True, expose ``alpha_trace`` on output.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            memory_state=memory_state,
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
