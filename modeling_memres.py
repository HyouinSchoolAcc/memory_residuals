"""
Qwen3 with Memory Residuals (MemRes).

Extends Block Attention Residuals with cross-session episodic memory.
Previous conversation history is compressed into a fixed-size latent
state M_c via a learnable summarizing memory block, then reintroduced
into the residual stream via a dynamically gated cross-attention at
each decoder layer.

Architecture:
    - MemoryBlock: C ∈ R^(N×d)  →  M_c ∈ R^(K×d)
      Cross-attention with K learnable query vectors M_in.
      Applied once per session transition (offline compression).

    - MemoryResidualCrossAttn: per-layer gated cross-attention to M_c.
      MemUpdate = CrossAttn(H_l, M_c, M_c)
      α = σ(W_g H_l + b_g)          ← input-dependent per-dim gate
      residual += α ⊙ MemUpdate

    - When memory_state is None, the model is identical to AttnRes.

Session workflow:
    # After session 1
    m_c = model.compress_history(session1_hidden_states)  # (B, K, d)

    # During session 2
    outputs = model(input_ids, memory_state=m_c)

Based on:
    "Memory Residuals" (Yueze Liu, 2026)
    "Attention Residuals" (Kimi Team, arXiv:2603.15031)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import can_return_tuple, auto_docstring
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from modeling_attnres import (
    Qwen3AttnResConfig,
    Qwen3AttnResDecoderLayer,
    Qwen3PreTrainedModel,
    block_attn_res,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Qwen3MemResConfig(Qwen3AttnResConfig):
    """Qwen3AttnResConfig extended with MemRes hyper-parameters."""

    model_type = "qwen3_memres"

    def __init__(
        self,
        memres_num_memory_vectors: int = 128,
        memres_num_heads: int = 8,
        # Where to inject memory cross-attention: "attn", "mlp", or "both"
        memres_apply_at: str = "attn",
        # Initial bias on gate logit: σ(-2) ≈ 0.12  →  small initial mixing
        memres_gate_init: float = -2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memres_num_memory_vectors = memres_num_memory_vectors
        self.memres_num_heads = memres_num_heads
        self.memres_apply_at = memres_apply_at
        self.memres_gate_init = memres_gate_init


# ---------------------------------------------------------------------------
# MemoryBlock: history → M_c
# ---------------------------------------------------------------------------

class MemoryBlock(nn.Module):
    """
    Compresses multi-session conversation history into K latent memory vectors.

    M_c = CrossAttn(M_in, C, C)

    where M_in ∈ R^(K×d) are learnable query vectors and C ∈ R^(N×d)
    is the encoded history (e.g., final hidden states from the previous session).

    Because K is fixed and small (e.g., 128), the KV cache for past conversations
    does not grow indefinitely; the cross-attention is forced to distill only the
    most predictive information about the user and past sessions.
    """

    def __init__(
        self,
        hidden_size: int,
        num_memory_vectors: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_size = hidden_size
        self.num_memory_vectors = num_memory_vectors
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Learnable query vectors M_in ∈ R^(K×d) — initialized to zero per AttnRes convention
        self.memory_queries = nn.Parameter(torch.zeros(num_memory_vectors, hidden_size))

        # Cross-attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm = Qwen3RMSNorm(hidden_size, eps=norm_eps)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d) → (B, H, L, dh)"""
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, L, dh) → (B, L, d)"""
        B, H, L, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * dh)

    def forward(self, history_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_hidden: (B, N, d) — encoded history from the previous session.
                            Typically the final hidden states after applying the model
                            to the previous conversation turns.
        Returns:
            memory_state: (B, K, d) — compressed latent memory.
        """
        B, N, _ = history_hidden.shape

        # Expand learnable queries to batch dimension: (B, K, d)
        queries = self.memory_queries.unsqueeze(0).expand(B, -1, -1)

        Q = self.q_proj(queries)           # (B, K, d)
        K = self.k_proj(history_hidden)    # (B, N, d)
        V = self.v_proj(history_hidden)    # (B, N, d)

        Q, K, V = self._split_heads(Q), self._split_heads(K), self._split_heads(V)

        # Scaled dot-product attention: Q attends over history positions
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, K, N)
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, V)  # (B, H, K, dh)
        out = self._merge_heads(out)          # (B, K, d)

        return self.norm(self.out_proj(out))  # (B, K, d)

    def update(
        self,
        existing_memory: torch.Tensor,
        new_hidden: torch.Tensor,
        blend_alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Incrementally update an existing memory state with new hidden states.

        Blends the current memory with a freshly compressed version:
            M_c_new = (1 - α) * M_c_existing + α * compress(new_hidden)

        Args:
            existing_memory: (B, K, d) — current memory state.
            new_hidden:      (B, N, d) — new hidden states to incorporate.
            blend_alpha:     blend weight for the new information (0 = no update, 1 = full replace).
        Returns:
            updated_memory: (B, K, d)
        """
        new_memory = self.forward(new_hidden)
        return (1.0 - blend_alpha) * existing_memory + blend_alpha * new_memory


# ---------------------------------------------------------------------------
# Per-layer memory cross-attention with learned sigmoid gate
# ---------------------------------------------------------------------------

class MemoryResidualCrossAttn(nn.Module):
    """
    Gated cross-attention to the episodic memory state M_c.

    At each decoder layer:
        MemUpdate = CrossAttn(H_l, M_c, M_c)
        α = σ(W_g H_l + b_g)        ← per-token, per-dim gate
        output = α ⊙ MemUpdate

    The gate α is initialized near zero (σ(-2) ≈ 0.12), so the model starts
    close to standard behavior and learns when to open the memory channel.
    When the current turn requires no historical context, α → 0 and the
    memory lookup is effectively bypassed.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        gate_init: float = -2.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Cross-attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Input-dependent per-dim sigmoid gate
        # W=0, b=gate_init ensures near-zero gate at init → stable loading of pretrained weights
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_init)

        self.norm = Qwen3RMSNorm(hidden_size, eps=norm_eps)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * dh)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, d) — current token representations.
            memory_state:  (B, K, d) — compressed episodic memory from MemoryBlock.
        Returns:
            gated_update:  (B, T, d) — memory contribution to the residual stream.
        """
        B, T, D = hidden_states.shape

        Q = self.q_proj(hidden_states)  # (B, T, d)
        K = self.k_proj(memory_state)   # (B, K, d)
        V = self.v_proj(memory_state)   # (B, K, d)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, K)
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, V)  # (B, H, T, dh)
        mem_update = self.out_proj(self.norm(self._merge_heads(out)))  # (B, T, d)

        # Input-dependent gate: routes memory only when the current turn requires it
        alpha = torch.sigmoid(self.gate_proj(hidden_states))  # (B, T, d)

        return alpha * mem_update


# ---------------------------------------------------------------------------
# Memory-augmented decoder layer
# ---------------------------------------------------------------------------

class Qwen3MemResDecoderLayer(Qwen3AttnResDecoderLayer):
    """
    Qwen3 decoder layer with both Block AttnRes (depth-wise) and
    Memory Residuals (cross-session episodic memory).

    If memory_state is None (no history provided), the layer is identical
    to the standard Qwen3AttnResDecoderLayer.

    Forward signature extends the parent with:
        memory_state: Optional (B, K, d) tensor from MemoryBlock.
    """

    def __init__(self, config: Qwen3MemResConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        apply_at = getattr(config, "memres_apply_at", "attn")
        num_heads = getattr(config, "memres_num_heads", 8)
        gate_init = getattr(config, "memres_gate_init", -2.0)
        norm_eps = config.rms_norm_eps

        self.memres_apply_at = apply_at

        # Add memory cross-attention modules at the requested sublayer(s)
        if apply_at in ("attn", "both"):
            self.mem_cross_attn = MemoryResidualCrossAttn(
                config.hidden_size, num_heads, gate_init, norm_eps
            )
        if apply_at in ("mlp", "both"):
            self.mem_cross_attn_mlp = MemoryResidualCrossAttn(
                config.hidden_size, num_heads, gate_init, norm_eps
            )

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        memory_state: torch.Tensor | None = None,   # (B, K, d) or None
        **kwargs,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        entropy_accum = kwargs.pop("entropy_accum", None)

        # ---- Attention sublayer ----
        if entropy_accum is not None:
            h_attn, ent = block_attn_res(
                blocks, partial_block,
                self.attn_res_proj, self.attn_res_norm, self.attn_res_bias,
                return_entropy=True,
            )
            entropy_accum.append(ent)
        else:
            h_attn = block_attn_res(
                blocks, partial_block,
                self.attn_res_proj, self.attn_res_norm, self.attn_res_bias,
            )
        h = self._apply_gate(partial_block, h_attn, "attn")

        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(h),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        partial_block = partial_block + attn_out

        # Memory cross-attention after attention sublayer
        if memory_state is not None and self.memres_apply_at in ("attn", "both"):
            partial_block = partial_block + self.mem_cross_attn(partial_block, memory_state)

        # Full mode: record post-attention state in history
        if self.attnres_mode == "full":
            blocks = blocks + [partial_block]

        # ---- MLP sublayer ----
        if entropy_accum is not None:
            h_attn, ent = block_attn_res(
                blocks, partial_block,
                self.mlp_res_proj, self.mlp_res_norm, self.mlp_res_bias,
                return_entropy=True,
            )
            entropy_accum.append(ent)
        else:
            h_attn = block_attn_res(
                blocks, partial_block,
                self.mlp_res_proj, self.mlp_res_norm, self.mlp_res_bias,
            )
        h = self._apply_gate(partial_block, h_attn, "mlp")

        mlp_out = self.mlp(self.post_attention_layernorm(h))
        partial_block = partial_block + mlp_out

        # Memory cross-attention after MLP sublayer
        if memory_state is not None and self.memres_apply_at in ("mlp", "both"):
            partial_block = partial_block + self.mem_cross_attn_mlp(partial_block, memory_state)

        if self.attnres_mode == "full" or self.is_block_boundary:
            blocks = blocks + [partial_block]

        return blocks, partial_block


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------

class Qwen3MemResModel(Qwen3PreTrainedModel):
    """
    Qwen3 backbone with Block AttnRes + episodic Memory Residuals.

    Accepts an optional `memory_state` tensor (B, K, d) from MemoryBlock.
    When None, behaves identically to Qwen3AttnResModel.
    """

    config_class = Qwen3MemResConfig

    def __init__(self, config: Qwen3MemResConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3MemResDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        # Memory compression block — encodes historical hidden states into M_c
        self.memory_block = MemoryBlock(
            hidden_size=config.hidden_size,
            num_memory_vectors=config.memres_num_memory_vectors,
            num_heads=config.memres_num_heads,
            norm_eps=config.rms_norm_eps,
        )

        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()

    def compress_history(self, history_hidden: torch.Tensor) -> torch.Tensor:
        """
        Compress previous session hidden states into a fixed-size memory state.

        Args:
            history_hidden: (B, N, d) — final hidden states from a previous session.
                            Use the `last_hidden_state` from a prior model forward pass.
        Returns:
            memory_state: (B, K, d) — compact latent memory to pass into the next session.
        """
        return self.memory_block(history_hidden)

    def update_memory(
        self,
        existing_memory: torch.Tensor,
        new_hidden: torch.Tensor,
        blend_alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Incrementally blend new hidden states into an existing memory state.

        Args:
            existing_memory: (B, K, d)
            new_hidden:      (B, N, d)
            blend_alpha:     weight given to new information (0=no change, 1=full replace).
        Returns:
            updated_memory: (B, K, d)
        """
        return self.memory_block.update(existing_memory, new_hidden, blend_alpha)

    def _init_weights(self, module):
        # Parent resets all nn.Linear weights/biases with a normal distribution
        # and zeros out biases. We re-apply our custom initializations afterward.
        super()._init_weights(module)

        if isinstance(module, Qwen3MemResDecoderLayer):
            # AttnRes depth-attention initialization
            gate_type = getattr(self.config, "attnres_gate_type", "bias")
            if gate_type == "sigmoid_scalar":
                module.attn_res_gate_logit.data.fill_(-2.0)
                module.mlp_res_gate_logit.data.fill_(-2.0)
                module.attn_res_bias.data.fill_(0.0)
                module.mlp_res_bias.data.fill_(0.0)
            elif gate_type == "sigmoid_vector":
                nn.init.zeros_(module.attn_res_gate_proj.weight)
                nn.init.constant_(module.attn_res_gate_proj.bias, -2.0)
                nn.init.zeros_(module.mlp_res_gate_proj.weight)
                nn.init.constant_(module.mlp_res_gate_proj.bias, -2.0)
                module.attn_res_bias.data.fill_(0.0)
                module.mlp_res_bias.data.fill_(0.0)
            elif gate_type == "learnable_alpha":
                module.attn_res_alpha.data.fill_(0.0)
                module.mlp_res_alpha.data.fill_(0.0)
                module.attn_res_bias.data.fill_(0.0)
                module.mlp_res_bias.data.fill_(0.0)
            else:
                bias_init = getattr(self.config, "attnres_recency_bias_init", 10.0)
                module.attn_res_bias.data.fill_(bias_init)
                module.mlp_res_bias.data.fill_(bias_init)

            # MemRes sigmoid gate initialization — re-apply after parent resets biases to 0
            gate_init = getattr(self.config, "memres_gate_init", -2.0)
            apply_at = getattr(self.config, "memres_apply_at", "attn")
            if apply_at in ("attn", "both") and hasattr(module, "mem_cross_attn"):
                nn.init.zeros_(module.mem_cross_attn.gate_proj.weight)
                nn.init.constant_(module.mem_cross_attn.gate_proj.bias, gate_init)
            if apply_at in ("mlp", "both") and hasattr(module, "mem_cross_attn_mlp"):
                nn.init.zeros_(module.mem_cross_attn_mlp.gate_proj.weight)
                nn.init.constant_(module.mem_cross_attn_mlp.gate_proj.bias, gate_init)

        elif isinstance(module, MemoryBlock):
            # Keep memory queries at zero init (per AttnRes convention — uniform attention at start)
            nn.init.zeros_(module.memory_queries)

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
        memory_state: torch.Tensor | None = None,   # (B, K, d) from compress_history()
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
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
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Block AttnRes state
        blocks: list[torch.Tensor] = [inputs_embeds]
        partial_block: torch.Tensor = inputs_embeds

        entropy_lambda = kwargs.pop("entropy_lambda", 0.0)
        entropy_accum = [] if entropy_lambda > 0 else None

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                blocks, partial_block = self._gradient_checkpointing_func(
                    layer.__call__,
                    blocks,
                    partial_block,
                    causal_mask_mapping[layer.attention_type],
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    memory_state,
                )
            else:
                blocks, partial_block = layer(
                    blocks=blocks,
                    partial_block=partial_block,
                    attention_mask=causal_mask_mapping[layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    memory_state=memory_state,
                    entropy_accum=entropy_accum,
                )

        hidden_states = self.norm(partial_block)

        attnres_entropy = None
        if entropy_accum:
            attnres_entropy = torch.stack(entropy_accum).mean()

        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        out.attnres_entropy = attnres_entropy
        return out


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------

class Qwen3MemResForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """
    Qwen3 causal LM with Block AttnRes + Memory Residuals.

    Usage — single session (no memory, identical to AttnRes):
        outputs = model(input_ids=input_ids, labels=labels)

    Usage — multi-session with persistent memory:
        # End of session 1: compress to memory
        with torch.no_grad():
            h = model.model(input_ids=session1_ids).last_hidden_state
            m_c = model.model.compress_history(h)  # (B, K, d)

        # Session 2: pass memory into the model
        outputs = model(input_ids=session2_ids, memory_state=m_c, labels=labels)

        # Optionally update memory mid-session
        with torch.no_grad():
            h_new = model.model(input_ids=new_ids, memory_state=m_c).last_hidden_state
            m_c = model.model.update_memory(m_c, h_new, blend_alpha=0.3)
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
        memory_state: torch.Tensor | None = None,  # (B, K, d) — episodic memory
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
            memory_state=memory_state,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

            entropy_lambda = kwargs.get("entropy_lambda", 0.0)
            attnres_entropy = getattr(outputs, "attnres_entropy", None)
            if entropy_lambda > 0 and attnres_entropy is not None:
                loss = loss - entropy_lambda * attnres_entropy

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
