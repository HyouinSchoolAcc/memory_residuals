"""
Named training presets for Memory Residuals.

Each preset pins (a) the pretrained Qwen3 backbone to attach MemRes onto,
(b) the MemoryBlock capacity (extraction depth `L_E`, slot count `K`), and
(c) the Block AttnRes block count `N`.  The MemoryBlock and MemoryReadout
always operate at `config.hidden_size` (`d`), so the readout output shape
`(B, S, d)` is identical to any decoder layer's output by construction.

Pick a preset on the trainer command line via `--preset`; it overrides the
manual size flags (`--pretrained`, `--memres_num_vectors`,
`--memres_extraction_depth`, `--memres_num_blocks`).

Approximate sizes (vocab=151936, tied embeddings, full bf16 inference):

    preset             d    L     K  L_E  N   MemBlock  vs 1 layer    TOTAL
    qwen3-0.6b-small  1024  28   128   0  8     6.55M      0.62x      0.606B
    qwen3-0.6b-large  1024  28   128   4  8    19.14M      1.42x      0.618B
    qwen3-8b-small    4096  36   128   0  8   101.71M      0.79x      7.721B
    qwen3-8b-large    4096  36   128   4  8   303.04M      1.83x      7.922B

Notes:
    - "7B" is not part of the Qwen3 release line (the family is 0.6B, 1.7B,
      4B, 8B, 14B, 32B); the 7B-class slot is filled here by Qwen3-8B,
      which is the closest analog and the one Qwen Team recommends as the
      "small flagship" tier.  Swap `pretrained` to "Qwen/Qwen2.5-7B" if you
      strictly need the 7B parameter count.
    - "small" vs "large" varies `L_E` only; `K` and `N` are constant so the
      readout-vs-attention dimensional invariant is preserved across all
      four presets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    pretrained: str
    memres_num_vectors: int
    memres_extraction_depth: int
    memres_num_blocks: int
    description: str


PRESETS: dict[str, Preset] = {
    "qwen3-0.6b-small": Preset(
        pretrained="Qwen/Qwen3-0.6B",
        memres_num_vectors=128,
        memres_extraction_depth=0,
        memres_num_blocks=8,
        description=(
            "Qwen3-0.6B backbone with a small MemoryBlock "
            "(L_E=0, single-pass extraction, ~0.62x of one transformer layer)."
        ),
    ),
    "qwen3-0.6b-large": Preset(
        pretrained="Qwen/Qwen3-0.6B",
        memres_num_vectors=128,
        memres_extraction_depth=4,
        memres_num_blocks=8,
        description=(
            "Qwen3-0.6B backbone with a large MemoryBlock "
            "(L_E=4, five-layer Perceiver-style refinement stack, "
            "~1.42x of one transformer layer)."
        ),
    ),
    "qwen3-8b-small": Preset(
        pretrained="Qwen/Qwen3-8B",
        memres_num_vectors=128,
        memres_extraction_depth=0,
        memres_num_blocks=8,
        description=(
            "Qwen3-8B backbone with a small MemoryBlock "
            "(L_E=0, single-pass extraction, ~0.79x of one transformer layer)."
        ),
    ),
    "qwen3-8b-large": Preset(
        pretrained="Qwen/Qwen3-8B",
        memres_num_vectors=128,
        memres_extraction_depth=4,
        memres_num_blocks=8,
        description=(
            "Qwen3-8B backbone with a large MemoryBlock "
            "(L_E=4, five-layer Perceiver-style refinement stack, "
            "~1.83x of one transformer layer)."
        ),
    ),
    # v10 presets: L_E=10 deep extraction stack, for the mega-corpus 8B run.
    # Rationale: v9c showed data diversity is the load-bearing signal axis
    # (LME-only and PG-19-only always had signal; MSC-heavy mixes dilute).
    # A deeper extraction stack lets the writer compact the richer
    # input distribution into K=128 slots without losing salient structure.
    "qwen3-4b-xlarge": Preset(
        pretrained="Qwen/Qwen3-4B",
        memres_num_vectors=128,
        memres_extraction_depth=10,
        memres_num_blocks=8,
        description=(
            "Qwen3-4B (d=2560, 36 layers) with an XL MemoryBlock "
            "(L_E=10, eleven-layer Perceiver-style refinement stack). "
            "Primary v10 preset for the GH200 3-day mega-corpus run: "
            "~4.3B total, ~52 GB HBM under full AdamW, fits comfortably "
            "on the 96 GB GH200 with room for activations + grad "
            "checkpointing. Upgrade to qwen3-8b-xlarge only if/when "
            "bitsandbytes is installed (required to fit 8B+L_E=10 full "
            "training on a single 96 GB GPU)."
        ),
    ),
    "qwen3-8b-xlarge": Preset(
        pretrained="Qwen/Qwen3-8B",
        memres_num_vectors=128,
        memres_extraction_depth=10,
        memres_num_blocks=8,
        description=(
            "Qwen3-8B (d=4096, 36 layers) with an XL MemoryBlock "
            "(L_E=10, eleven-layer Perceiver-style refinement stack). "
            "~8.8B total. Does NOT fit single 96 GB GPU under full "
            "AdamW (~106 GB peak); use qwen3-4b-xlarge on GH200 unless "
            "bitsandbytes AdamW8bit is installed (then pass "
            "--use_adam8bit)."
        ),
    ),
}


def apply_preset(args, preset_name: str) -> None:
    """Mutate `args` in place so trainer flags reflect the selected preset.

    Manual flags are overwritten with the preset values; this keeps preset
    runs reproducible regardless of what the CLI defaults happen to be.
    """
    if preset_name not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"Unknown preset {preset_name!r}; choose one of: {valid}"
        )
    p = PRESETS[preset_name]
    args.pretrained = p.pretrained
    args.memres_num_vectors = p.memres_num_vectors
    args.memres_extraction_depth = p.memres_extraction_depth
    args.memres_num_blocks = p.memres_num_blocks
