"""
Visualize Block Attention Residual softmax weights in the style of
the Kimi team's "Attention Residuals: Layer Dependencies" figure.

Y-axis: every sublayer (Attn, MLP) for each layer
X-axis: source blocks (Embed, Block 1, ..., Block N, Partial)
Color:  viridis, showing Residual Magnitude (softmax weight)

Usage:
    python visualize.py \
        --model_path output/scratch-block-d1024-L28-20k/step-18000
"""

import argparse
import sys
import os

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Local import
from modeling_attnres import Qwen3AttnResForCausalLM, Qwen3AttnResConfig
from transformers import AutoTokenizer


def compute_softmax_weights(blocks, partial_block, proj, norm, recency_bias):
    """Compute AttnRes softmax weights."""
    V = torch.stack(blocks + [partial_block], dim=0)
    K = norm(V)
    query = proj.weight.view(-1)
    logits = torch.einsum("d, n b t d -> n b t", query, K)
    logits[-1] = logits[-1] + recency_bias
    weights = logits.softmax(dim=0)
    return weights.float().mean(dim=(1, 2)).detach().cpu().numpy()


def extract_attnres_weights(model, input_ids):
    """Run forward pass and capture softmax weights from every sublayer."""
    import modeling_attnres as attnres_mod
    captured = {"attn": {}, "mlp": {}}

    original_forwards = {}
    for layer_idx, layer in enumerate(model.model.layers):
        original_forwards[layer_idx] = layer.forward

        def make_patched_forward(orig_forward, lyr, idx):
            def patched_forward(blocks, partial_block, **kwargs):
                # Attention sublayer weights
                w_attn = compute_softmax_weights(
                    blocks, partial_block,
                    lyr.attn_res_proj, lyr.attn_res_norm, lyr.attn_res_bias
                )
                captured["attn"][idx] = {
                    "block_weights": w_attn[:-1],
                    "partial_weight": w_attn[-1],
                    "num_blocks": len(blocks),
                }

                # Run attention sublayer
                h = attnres_mod.block_attn_res(
                    blocks, partial_block,
                    lyr.attn_res_proj, lyr.attn_res_norm, lyr.attn_res_bias
                )
                attn_out, _ = lyr.self_attn(
                    hidden_states=lyr.input_layernorm(h),
                    attention_mask=kwargs.get("attention_mask"),
                    position_ids=kwargs.get("position_ids"),
                    past_key_values=kwargs.get("past_key_values"),
                    use_cache=kwargs.get("use_cache", False),
                    cache_position=kwargs.get("cache_position"),
                    position_embeddings=kwargs.get("position_embeddings"),
                )
                post_attn_partial = partial_block + attn_out

                # Full mode: append post-attn state to history
                if lyr.attnres_mode == "full":
                    blocks = blocks + [post_attn_partial]

                # MLP sublayer weights (with updated blocks in full mode)
                w_mlp = compute_softmax_weights(
                    blocks, post_attn_partial,
                    lyr.mlp_res_proj, lyr.mlp_res_norm, lyr.mlp_res_bias
                )
                captured["mlp"][idx] = {
                    "block_weights": w_mlp[:-1],
                    "partial_weight": w_mlp[-1],
                    "num_blocks": len(blocks),
                }

                # Run MLP sublayer
                h = attnres_mod.block_attn_res(
                    blocks, post_attn_partial,
                    lyr.mlp_res_proj, lyr.mlp_res_norm, lyr.mlp_res_bias
                )
                mlp_out = lyr.mlp(lyr.post_attention_layernorm(h))
                final_partial = post_attn_partial + mlp_out

                # Full mode: always append; Block mode: only at boundaries
                if lyr.attnres_mode == "full" or lyr.is_block_boundary:
                    blocks = blocks + [final_partial]

                return blocks, final_partial
            return patched_forward

        layer.forward = make_patched_forward(original_forwards[layer_idx], layer, layer_idx)

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for layer_idx, layer in enumerate(model.model.layers):
            layer.forward = original_forwards[layer_idx]

    return captured


def plot_kimi_style(captured, num_layers, layers_per_block, num_blocks, model_name, output_path):
    """
    Create a Kimi-paper-style triangular heatmap.
    Rows = sublayers (Attn_0, MLP_0, Attn_1, MLP_1, ...)
    Cols = source positions (all weights concatenated: block_weights + partial_weight)
    """
    num_sublayers = num_layers * 2  # Attn + MLP per layer

    # Total columns = max number of sources any sublayer attends to
    # (block_weights length + 1 for partial)
    max_sources = 0
    for sublayer in ["attn", "mlp"]:
        for d in captured[sublayer].values():
            n = len(d["block_weights"]) + 1  # +1 for partial
            max_sources = max(max_sources, n)
    total_cols = max_sources

    # Build the matrix: rows = sublayers, cols = all sources in order
    # Each row's weights are [source_0, source_1, ..., source_k, partial]
    # We place them left-aligned so source indices align across rows.
    matrix = np.full((num_sublayers, total_cols), np.nan)

    for layer_idx in range(num_layers):
        attn_row = layer_idx * 2
        mlp_row = layer_idx * 2 + 1

        if layer_idx in captured["attn"]:
            d = captured["attn"][layer_idx]
            all_weights = np.concatenate([d["block_weights"], [d["partial_weight"]]])
            matrix[attn_row, :len(all_weights)] = all_weights

        if layer_idx in captured["mlp"]:
            d = captured["mlp"][layer_idx]
            all_weights = np.concatenate([d["block_weights"], [d["partial_weight"]]])
            matrix[mlp_row, :len(all_weights)] = all_weights

    # --- Plotting ---
    is_full_mode = total_cols > 20
    fig_width = max(12, total_cols * 0.5 + 4) if is_full_mode else 12
    fig, ax = plt.subplots(figsize=(fig_width, 18))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.08)

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#1a1a2e")  # dark background for unavailable blocks

    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=0.7,
                   interpolation="nearest", origin="upper")

    # --- Block boundary lines ---
    for b in range(1, num_blocks + 1):
        boundary = b * layers_per_block * 2  # sublayer index
        if boundary < num_sublayers:
            ax.axhline(y=boundary - 0.5, color="white", linewidth=0.5,
                      linestyle="-", alpha=0.3)

    # Vertical line before Partial column (only for block mode)
    if not is_full_mode:
        ax.axvline(x=total_cols - 1.5, color="white", linewidth=0.8,
                  linestyle="-", alpha=0.4)

    # --- Y-axis labels (sublayer type + layer number) ---
    y_labels = []
    y_colors = []
    for i in range(num_layers):
        y_labels.append(f"Attn")
        y_colors.append("#4CAF50")  # green for attention
        y_labels.append(f"MLP")
        y_colors.append("#FF9800")  # orange for MLP

    # Build tick labels like "Attn 0", "MLP 0", "Attn 1", ...
    tick_labels = []
    for i in range(num_sublayers):
        layer_num = i // 2
        sublayer_type = y_labels[i]
        tick_labels.append(f"{sublayer_type} {layer_num}")

    ax.set_yticks(range(num_sublayers))
    ax.set_yticklabels(tick_labels, fontsize=5)

    # Color the y-axis tick labels
    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(y_colors[i])
        tick.set_fontweight("bold")

    # --- X-axis labels ---
    # In full mode, each source column is a sublayer output;
    # columns are not fixed — each row has its own partial at the end.
    # We label by source index; for full mode, odd=post-attn, even=post-MLP.
    x_labels = []
    x_colors_list = []
    is_full_mode = total_cols > 20
    for j in range(total_cols):
        if j == 0:
            x_labels.append("EMB")
            x_colors_list.append("#9C27B0")
        elif is_full_mode:
            layer_num = (j - 1) // 2
            sublayer = "A" if (j - 1) % 2 == 0 else "M"
            x_labels.append(f"{sublayer}{layer_num}")
            x_colors_list.append("#4CAF50" if sublayer == "A" else "#FF9800")
        else:
            x_labels.append(f"B{j}")
            x_colors_list.append("#2196F3")

    ax.set_xticks(range(total_cols))
    fontsize_x = 3.5 if is_full_mode else 6.5
    ax.set_xticklabels(x_labels, fontsize=fontsize_x, rotation=90)
    for j, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(x_colors_list[j])
        tick.set_fontweight("bold")

    # --- Text annotations for significant weights (skip in full mode — too dense) ---
    if not is_full_mode:
        for i in range(num_sublayers):
            for j in range(total_cols):
                val = matrix[i, j]
                if not np.isnan(val) and val > 0.08:
                    color = "white" if val < 0.35 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=4, color=color, fontweight="bold")

    # --- Colorbar ---
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label="Residual Magnitude")
    cbar.ax.tick_params(labelsize=8)

    # --- Title ---
    ax.set_title(
        f"Attention Residuals: Layer Dependencies\n{model_name}",
        fontsize=13, fontweight="bold", pad=20
    )

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor="#4CAF50", label="Attention sublayer"),
        mpatches.Patch(facecolor="#FF9800", label="MLP sublayer"),
        mpatches.Patch(facecolor="#9C27B0", label="Embedding (Block 0)"),
        mpatches.Patch(facecolor="#2196F3", label="Completed Block"),
        mpatches.Patch(facecolor="#E91E63", label="Partial Block"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
             framealpha=0.9, edgecolor="gray")

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize AttnRes (Kimi paper style)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text", type=str,
                       default="The theory of general relativity, proposed by Albert Einstein in 1915, "
                               "describes gravity as the curvature of spacetime caused by mass and energy. "
                               "This revolutionary framework replaced Newton's law of universal gravitation "
                               "and has been confirmed by numerous experiments.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    if args.output is None:
        args.output = f"attnres_deps_{model_name}.png"

    print(f"Loading model from {args.model_path}...")
    model = Qwen3AttnResForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    input_ids = tokenizer(args.text, return_tensors="pt")["input_ids"]
    print(f"Input: {args.text[:80]}... ({input_ids.shape[1]} tokens)")

    num_layers = model.config.num_hidden_layers
    num_blocks = model.config.attnres_num_blocks
    layers_per_block = model.model.layers[0].layers_per_block

    print(f"Model: {num_layers} layers, {num_blocks} blocks, {layers_per_block} layers/block")

    print("Running forward pass...")
    captured = extract_attnres_weights(model, input_ids)
    print(f"Captured {len(captured['attn'])} attn + {len(captured['mlp'])} mlp sublayers")

    plot_kimi_style(captured, num_layers, layers_per_block, num_blocks, model_name, args.output)


if __name__ == "__main__":
    main()
