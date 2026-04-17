"""
Memory Residuals: End-to-End Differentiable Lifelong Memory for Conversational Agents

Two advances over standard Qwen3:

1. MemoryBlock — compresses multi-session history C ∈ ℝ^(N×d) into a fixed-size
   persistent memory state M_c ∈ ℝ^(K×d) via cross-attention with learnable latent
   queries M_in ∈ ℝ^(K×d):

       M_c = Softmax(M_in W_Q (C W_K)ᵀ / √d) · C W_V

2. MemResDecoderLayer — injects M_c into the residual stream right before each
   attention sublayer's QKV projection (Section 2.2):

       V_pool = [H_{l-1} ‖ M_c]  ∈ ℝ^((S+K)×d)
       α      = Softmax(H_{l-1} W_Q^res (V_pool W_K^res)ᵀ / √d)  ∈ ℝ^(S×(S+K))
       H̃_l   = α V_pool                                           ∈ ℝ^(S×d)
       H_l    = H̃_l + SelfAttn(LN(H̃_l), ...)

   For filler turns softmax weights stay on the local H_{l-1} tokens; for
   memory-intensive turns ("rewrite like the emails from last Tuesday") weights
   shift onto M_c, pulling lifelong context directly into QKV.
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

    def __init__(self, memres_num_vectors: int = 128, **kwargs):
        super().__init__(**kwargs)
        # K: number of latent memory vectors (fixed regardless of history length)
        self.memres_num_vectors = memres_num_vectors


# ---------------------------------------------------------------------------
# Memory block
# ---------------------------------------------------------------------------


class MemoryBlock(nn.Module):
    """
    Learnable summarizing memory block (Section 2.1).

    Compresses multi-session history C ∈ ℝ^(N×d) into M_c ∈ ℝ^(K×d).
    K is fixed (e.g. 128 vectors) — the KV cache doesn't grow with N.

    The network is forced to distill only the most predictive user-history
    information. No fragmentation; dimensionally aligned with the native layers.
    """

    def __init__(self, hidden_size: int, num_vectors: int):
        super().__init__()
        self.scale = hidden_size**-0.5
        # M_in ∈ ℝ^(K×d): the persistent latent query bank, initialised with
        # small random values so each vector starts as a distinct memory slot.
        self.M_in = nn.Parameter(torch.empty(num_vectors, hidden_size))
        nn.init.normal_(self.M_in, std=hidden_size**-0.5)

        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C: (B, N, d) — multi-session history token representations
        Returns:
            M_c: (B, K, d) — compressed memory state
        """
        B = C.size(0)
        M_in = self.M_in.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)
        Q = self.W_Q(M_in)  # (B, K, d)
        K = self.W_K(C)  # (B, N, d)
        V = self.W_V(C)  # (B, N, d)
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # (B, K, N)
        return attn @ V  # (B, K, d)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class MemResDecoderLayer(GradientCheckpointingLayer):
    """
    Qwen3 decoder layer with Memory Attention Residuals (Section 2.2).

    The attention sublayer receives H̃_l (attention-weighted blend of local tokens
    and M_c) instead of raw H_{l-1}, so memory is queried per-turn, per-layer,
    and per-token — not via a hard retrieval gate.

    MLP uses a standard residual
    """

    def __init__(self, config: MemResConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.scale = config.hidden_size**-0.5

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

        # Memory residual projections W_Q^res and W_K^res.
        # Zero-init → Q=0, K=0 → uniform softmax → H̃ = mean(V_pool) at init.
        # Gradients immediately flow from the first forward pass.
        self.mem_res_W_Q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mem_res_W_K = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        nn.init.zeros_(self.mem_res_W_Q.weight)
        nn.init.zeros_(self.mem_res_W_K.weight)

    def _memory_attn_res(
        self,
        hidden_states: torch.Tensor,  # (B, S, d) — H_{l-1}
        M_c: torch.Tensor,  # (B, K, d) — compressed memory state
    ) -> torch.Tensor:
        """Compute H̃_l = α · V_pool (Section 2.2, Eq. 3–5)."""
        V_pool = torch.cat([hidden_states, M_c], dim=1)  # (B, S+K, d)
        Q = self.mem_res_W_Q(hidden_states)  # (B, S, d)
        K = self.mem_res_W_K(V_pool)  # (B, S+K, d)
        alpha = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # (B, S, S+K)
        return alpha @ V_pool  # (B, S, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        M_c: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # ── Attention sublayer with memory residual ──
        H_tilde = (
            self._memory_attn_res(hidden_states, M_c)
            if M_c is not None
            else hidden_states
        )

        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(H_tilde),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = H_tilde + attn_out

        # ── MLP sublayer (standard pre-LN residual) ──
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
        # Single shared memory block — one M_c per forward pass, used at all layers.
        self.memory_block = MemoryBlock(config.hidden_size, config.memres_num_vectors)

        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

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
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        history_ids: (B, N) token ids of previous session(s). If provided, they
            are embedded and compressed into M_c by the MemoryBlock. Ignored when
            M_c is passed directly (allows pre-computing M_c once for multi-turn
            generation).
        M_c: (B, K, d) pre-computed memory state. Takes precedence over history_ids.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute M_c from history if not pre-computed by the caller
        if M_c is None and history_ids is not None:
            C = self.embed_tokens(history_ids)  # (B, N, d)
            M_c = self.memory_block(C)  # (B, K, d)

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
        hidden_states = inputs_embeds

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    M_c,
                    causal_mask_mapping[layer.attention_type],
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    M_c=M_c,
                    attention_mask=causal_mask_mapping[layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------


class MemResForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3 causal LM with Memory Residuals.

    Quick-start:
        # Training (history_ids automatically compressed to M_c)
        out = model(input_ids=current, labels=current, history_ids=prev_session)

        # Efficient multi-turn generation (pre-compute M_c once)
        C = model.model.embed_tokens(history_ids)
        M_c = model.model.memory_block(C)
        tokens = model.generate(input_ids=prompt, M_c=M_c, max_new_tokens=200)
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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
