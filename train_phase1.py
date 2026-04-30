#!/usr/bin/env python3
"""Phase-1 multi-GPU trainer for Memory Residuals.

This is the production trainer used to drive the empirical Memory Residuals
paper.  Compared to the legacy ``train_memres.py``:

* Multi-GPU via ``torchrun`` and DDP for sub-1B backbones, or FSDP (full
  sharding) for the 8B preset.  Activations are bf16 throughout.
* Regularizers: memory dropout (zero-out ``m_t``), context dropout (mask the
  first half of the current session), and memory-gate detach for ablations.
* Periodic in-loop evaluation on a held-out pairs JSONL file.  Reports
  memory-on, memory-off, shuffled-history, and per-sublayer mean alpha_mem.
* Saves: best-by-validation-CE checkpoint, last checkpoint, full optimizer
  state for continuation.

The data format is the JSONL produced by
``archive/tools/prepare_pairs.py`` (``history``, ``current`` keys per
line). History compresses into ``M_c`` once per step and the loss is taken
on the current session. (The pair-trainer pipeline is dormant; v6 work
uses ``train_chain.py`` and ``tools/build_conversational_callback_chains.py``.)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from functools import partial

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig, AutoTokenizer

from modeling_memres import (
    Qwen3MemResConfig,
    Qwen3MemResDecoderLayer,
    Qwen3MemResForCausalLM,
    _normalise_memres_mode,
)
from presets import PRESETS, apply_preset


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument("--pretrained", default=None)
    p.add_argument(
        "--memres_mode",
        # Canonical names: simple_gate / attention_base / attention_parity.
        # Legacy names "residual" / "block_attnres" are accepted and folded
        # into the canonical set after parsing so old shell scripts and
        # saved checkpoints keep working.
        choices=(
            "simple_gate", "attention_base", "attention_parity",
            "residual", "block_attnres",
        ),
        default="attention_parity",
        help="simple_gate = ReZero-style scalar-gate injection (toy/baseline); "
             "attention_base = full Block AttnRes routing pool, delta sources, "
             "no init parity; "
             "attention_parity = full Block AttnRes pool, cumulative sources, "
             "step-0 logits bit-identical to the bare backbone (default). "
             "Legacy: 'residual' -> simple_gate, "
             "'block_attnres' -> attention_parity (or attention_base when "
             "paired with --no-block_attnres_parity_init).",
    )
    p.add_argument(
        "--block_attnres_parity_init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="DEPRECATED -- use --memres_mode attention_base / "
        "attention_parity directly. Only honoured when --memres_mode is the "
        "legacy 'block_attnres' string; ignored otherwise.",
    )
    # Router init-bias overrides (None -> mode-derived defaults: attention_parity
    # uses (-32, +32), other modes use (-8, 0)).  Soften (e.g. -4 / +4) to
    # trade init parity for non-saturated router gradients at step 0.
    p.add_argument(
        "--router_mem_bias_init", type=float, default=None,
        help="Override BlockAttnResRouter.mem_bias init value.",
    )
    p.add_argument(
        "--router_recent_bias_init", type=float, default=None,
        help="Override BlockAttnResRouter.recent_bias init value.",
    )

    # Backbone overrides for from-scratch runs
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_kv_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1536)
    p.add_argument("--head_dim", type=int, default=None)

    # MemRes knobs (override preset)
    p.add_argument("--memres_num_vectors", type=int, default=128)
    p.add_argument("--memres_extraction_depth", type=int, default=0)
    p.add_argument("--memres_num_blocks", type=int, default=8)

    # Data. The pair corpora are archived under archive/datasets/pairs/;
    # this trainer is dormant (v6 work uses train_chain.py). If you want to
    # reproduce the run3 pair-warmup, restore the corpora and pass the path.
    p.add_argument(
        "--train_data",
        default="archive/datasets/pairs/stage1_train_h1024_c512.pt",
        help="Path to a pre-tokenized .pt file produced by "
        "archive/tools/pretokenize_pairs.py (history_ids, current_ids).",
    )
    p.add_argument(
        "--eval_data",
        default="archive/datasets/pairs/stage1_validation_h1024_c512.pt",
    )
    p.add_argument(
        "--history_len",
        type=int,
        default=1024,
        help="Trim/pad pre-tokenized history to this length (must be <= "
        "history dimension of the .pt file).",
    )
    p.add_argument("--current_len", type=int, default=512)

    # Optimization
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument(
        "--lr_memres",
        type=float,
        default=None,
        help="Optional separate LR for the MemRes modules; defaults to --lr "
        "if unset.  Higher LRs on fresh memory params + lower LRs on the "
        "pretrained backbone is the standard recipe.",
    )
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.1)

    # Regularizers
    p.add_argument(
        "--memory_dropout",
        type=float,
        default=0.10,
        help="Per-step probability of zeroing M_c (drops the memory channel "
        "for that micro-batch).",
    )
    p.add_argument(
        "--context_dropout",
        type=float,
        default=0.30,
        help="Per-step probability of masking the first half of the current "
        "session in the attention mask, forcing reliance on memory.",
    )

    # Behaviour
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--detach_history_embeddings", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--shard_strategy",
        choices=("ddp", "fsdp_full"),
        default="ddp",
        help="ddp replicates the model on each GPU (fast for <=1B); "
        "fsdp_full shards parameters/grads/optimizer state across GPUs "
        "(required for 8B-class backbones).",
    )

    # IO + logging
    p.add_argument("--out_dir", default="output/phase1")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_n", type=int, default=64)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument(
        "--resume", default=None, help="Path to checkpoint dir to resume from."
    )

    args = p.parse_args()
    if args.preset is not None:
        apply_preset(args, args.preset)
    if args.lr_memres is None:
        args.lr_memres = args.lr
    args.memres_mode = _normalise_memres_mode(
        args.memres_mode, args.block_attnres_parity_init
    )
    args.block_attnres_parity_init = args.memres_mode == "attention_parity"
    return args


# ---------------------------------------------------------------------------
# Streaming JSONL pair loader
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PairBatch:
    history_ids: torch.Tensor  # (B, H)
    current_ids: torch.Tensor  # (B, C+1)
    shuffled_history_ids: torch.Tensor  # (B, H), pairs from another sample


class TensorPairLoader:
    """Loads a pre-tokenized .pt file and iterates rank-sharded batches.

    Expected file layout (produced by ``tools/pretokenize_pairs.py``):
        {"history_ids": LongTensor(N, H), "current_ids": LongTensor(N, C+1),
         "history_len": H, "current_len": C, "tokenizer": str}
    """

    def __init__(
        self,
        path: Path,
        rank: int,
        world_size: int,
        seed: int,
        shuffle: bool = True,
    ):
        self.path = Path(path)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        blob = torch.load(self.path, map_location="cpu", weights_only=False)
        self.history_ids: torch.Tensor = blob["history_ids"]
        self.current_ids: torch.Tensor = blob["current_ids"]
        self.history_len = int(blob.get("history_len", self.history_ids.shape[1]))
        self.current_len = int(
            blob.get("current_len", self.current_ids.shape[1] - 1)
        )
        n = self.history_ids.shape[0]
        self.indices = list(range(n))[rank::world_size]

    def iter_samples(self):
        rng = random.Random(self.seed + self.rank)
        while True:
            order = list(self.indices)
            if self.shuffle:
                rng.shuffle(order)
            n = len(order)
            for pos, idx in enumerate(order):
                shuffled_idx = order[(pos + 1) % n]
                yield PairBatch(
                    history_ids=self.history_ids[idx : idx + 1],
                    current_ids=self.current_ids[idx : idx + 1],
                    shuffled_history_ids=self.history_ids[
                        shuffled_idx : shuffled_idx + 1
                    ],
                )
            if not self.shuffle:
                return

    def iter_batches(self, batch_size: int):
        buf_h, buf_c, buf_sh = [], [], []
        for sample in self.iter_samples():
            buf_h.append(sample.history_ids)
            buf_c.append(sample.current_ids)
            buf_sh.append(sample.shuffled_history_ids)
            if len(buf_h) == batch_size:
                yield PairBatch(
                    history_ids=torch.cat(buf_h, 0),
                    current_ids=torch.cat(buf_c, 0),
                    shuffled_history_ids=torch.cat(buf_sh, 0),
                )
                buf_h.clear()
                buf_c.clear()
                buf_sh.clear()


# ---------------------------------------------------------------------------
# Causal mask with optional context dropout
# ---------------------------------------------------------------------------


def make_attn_mask_dict(
    seq_len: int,
    cutoff: int,
    device,
    dtype=torch.bfloat16,
) -> dict:
    """Build a ``{"full_attention": (1, 1, S, S) mask}`` dict that combines
    the standard causal mask with a *context-dropout* block: positions >=
    cutoff cannot attend to positions < cutoff.  The Qwen3MemRes backbone
    short-circuits its own ``create_causal_mask`` call when
    ``attention_mask`` is already a dict, so this is the cleanest way to
    inject an additive bias.

    Allowed iff i >= j (causal) AND not (i >= cutoff and j < cutoff)
    """
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    # Causal (i < j blocked).
    upper = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    mask[:, :, upper] = neg_inf
    # Context dropout (i >= cutoff, j < cutoff blocked).
    if cutoff > 0:
        mask[:, :, cutoff:, :cutoff] = neg_inf
    return {"full_attention": mask}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def cosine_with_warmup(step: int, warmup: int, total: int, lr_min_ratio: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cos


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._init_distributed()
        self._resolve_names()
        torch.manual_seed(args.seed + self.rank)
        random.seed(args.seed + self.rank)

        self.is_main = self.rank == 0
        self.use_wandb = self._init_wandb()
        self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path())

        if self.is_main:
            print(self._banner())

        self.model = self._build_model()
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self._apply_freeze()
        self.wrapped = self._wrap_model()

        self.optimizer, self.scheduler = self._build_optimizer()
        if self.is_main:
            print(f"  loading train data {args.train_data} ...", flush=True)
        self.train_loader = TensorPairLoader(
            args.train_data,
            self.rank,
            self.world_size,
            args.seed,
            shuffle=True,
        )
        if self.is_main:
            print(
                f"  loaded train: {self.train_loader.history_ids.shape} "
                f"+ {self.train_loader.current_ids.shape}",
                flush=True,
            )
            print(f"  loading eval data {args.eval_data} ...", flush=True)
        self.eval_loader = TensorPairLoader(
            args.eval_data,
            0,
            1,
            args.seed + 999,
            shuffle=False,
        )
        if self.is_main:
            print(
                f"  loaded eval: {self.eval_loader.history_ids.shape}",
                flush=True,
            )
        os.makedirs(args.out_dir, exist_ok=True)

        self.best_eval_ce = float("inf")
        self.global_step = 0

        if args.resume:
            self._load_checkpoint(args.resume)

    # ---- setup helpers ----

    def _init_distributed(self) -> None:
        if dist.is_available() and "WORLD_SIZE" in os.environ:
            dist.init_process_group("nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

    def _resolve_names(self) -> None:
        a = self.args
        if a.run_name is None:
            tag = a.preset or f"d{a.hidden_size}-L{a.num_layers}"
            a.run_name = f"phase1-{tag}-K{a.memres_num_vectors}-{a.steps // 1000}k"
        if a.out_dir is None:
            a.out_dir = f"./output/{a.run_name}"

    def _tokenizer_path(self) -> str:
        if self.args.pretrained is not None:
            return self.args.pretrained
        return "Qwen/Qwen3-0.6B"

    def _banner(self) -> str:
        a = self.args
        return (
            f"\n=== Memory Residuals Phase 1 ===\n"
            f"  run_name        : {a.run_name}\n"
            f"  preset          : {a.preset}\n"
            f"  pretrained      : {a.pretrained}\n"
            f"  memres_mode     : {a.memres_mode}\n"
            f"  K, L_E, N       : {a.memres_num_vectors}, "
            f"{a.memres_extraction_depth}, {a.memres_num_blocks}\n"
            f"  history/current : {a.history_len}/{a.current_len}\n"
            f"  steps           : {a.steps}, bs={a.batch_size}, "
            f"grad_accum={a.grad_accum}\n"
            f"  lr              : memres={a.lr_memres}, backbone={a.lr_backbone}\n"
            f"  reg             : mem_drop={a.memory_dropout}, "
            f"ctx_drop={a.context_dropout}\n"
            f"  shard_strategy  : {a.shard_strategy}\n"
            f"  world_size      : {self.world_size}\n"
            f"  out_dir         : {a.out_dir}"
        )

    def _init_wandb(self) -> bool:
        if not self.is_main or not self.args.wandb_project:
            return False
        try:
            import wandb

            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=self.args.run_name,
                config=vars(self.args),
            )
            return True
        except Exception as exc:
            print(f"W&B init failed ({exc}); continuing without logging")
            return False

    def _build_model(self) -> Qwen3MemResForCausalLM:
        a = self.args
        memres_kwargs = dict(
            memres_num_vectors=a.memres_num_vectors,
            memres_extraction_depth=a.memres_extraction_depth,
            memres_num_blocks=a.memres_num_blocks,
            memres_mode=a.memres_mode,
            block_attnres_parity_init=a.block_attnres_parity_init,
            router_mem_bias_init=a.router_mem_bias_init,
            router_recent_bias_init=a.router_recent_bias_init,
        )
        if a.pretrained:
            base_cfg = AutoConfig.from_pretrained(a.pretrained)
            cfg = Qwen3MemResConfig(**{**base_cfg.to_dict(), **memres_kwargs})
            return Qwen3MemResForCausalLM.from_pretrained(
                a.pretrained, config=cfg, dtype=torch.bfloat16
            ).to(self.device)
        head_dim = a.head_dim or (a.hidden_size // a.num_heads)
        cfg = Qwen3MemResConfig(
            vocab_size=151936,
            hidden_size=a.hidden_size,
            num_hidden_layers=a.num_layers,
            num_attention_heads=a.num_heads,
            num_key_value_heads=a.num_kv_heads,
            intermediate_size=a.intermediate_size,
            max_position_embeddings=(a.history_len + a.current_len) * 2,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            head_dim=head_dim,
            **memres_kwargs,
        )
        return (
            Qwen3MemResForCausalLM(cfg).to(dtype=torch.bfloat16).to(self.device)
        )

    def _apply_freeze(self) -> None:
        # In simple_gate mode the depth_router is constructed (so checkpoint
        # shapes stay stable across modes) but unused on the forward path,
        # so always disable its gradients to keep the optimizer / DDP graph
        # tight.
        if self.args.memres_mode == "simple_gate":
            for name, p in self.model.named_parameters():
                if "depth_router" in name:
                    p.requires_grad = False

        if not self.args.freeze_backbone:
            return
        markers = ("memory_block", "memory_readout", "depth_router", "memory_gate")
        n_train = 0
        n_total = 0
        for name, p in self.model.named_parameters():
            n_total += p.numel()
            if any(m in name for m in markers):
                if not (
                    self.args.memres_mode == "simple_gate"
                    and "depth_router" in name
                ):
                    p.requires_grad = True
                    n_train += p.numel()
            else:
                p.requires_grad = False
        if self.is_main:
            print(
                f"  freeze_backbone : trainable {n_train / 1e6:.2f}M / "
                f"total {n_total / 1e9:.3f}B"
            )

    def _wrap_model(self):
        if self.world_size <= 1:
            return self.model
        if self.args.shard_strategy == "fsdp_full":
            mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Qwen3MemResDecoderLayer},
            )
            return FSDP(
                self.model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=wrap_policy,
                mixed_precision=mp,
                device_id=self.local_rank,
                use_orig_params=True,
            )
        return DDP(
            self.model,
            device_ids=[self.local_rank],
            # In "simple_gate" mode the depth_router is constructed but
            # unused on the forward path; some routes also drop the memory
            # channel under memory_dropout.  Setting find_unused_parameters
            # =True avoids hangs on all_reduce when graph topology varies.
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    def _model(self) -> Qwen3MemResForCausalLM:
        return self.wrapped.module if hasattr(self.wrapped, "module") else self.wrapped

    def _build_optimizer(self):
        a = self.args
        backbone_params = []
        memres_params = []
        markers = ("memory_block", "memory_readout", "depth_router", "memory_gate")
        named = self.wrapped.named_parameters() if hasattr(
            self.wrapped, "named_parameters"
        ) else self.model.named_parameters()
        for name, p in named:
            if not p.requires_grad:
                continue
            if any(m in name for m in markers):
                memres_params.append(p)
            else:
                backbone_params.append(p)
        groups = []
        if memres_params:
            groups.append(
                {
                    "params": memres_params,
                    "lr": a.lr_memres,
                    "name": "memres",
                }
            )
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": a.lr_backbone,
                    "name": "backbone",
                }
            )
        if self.is_main:
            n_mem = sum(p.numel() for p in memres_params) / 1e6
            n_bb = sum(p.numel() for p in backbone_params) / 1e6
            print(
                f"  param groups    : memres={n_mem:.1f}M @ {a.lr_memres}, "
                f"backbone={n_bb:.1f}M @ {a.lr_backbone}"
            )
        opt = AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.weight_decay)
        sched = LambdaLR(
            opt,
            lr_lambda=lambda step: cosine_with_warmup(
                step, a.warmup, a.steps, a.lr_min_ratio
            ),
        )
        return opt, sched

    # ---- training step ----

    def _train_step(self, batch: PairBatch, step_rng: random.Random) -> dict:
        a = self.args
        h_ids = batch.history_ids.to(self.device)
        c_ids = batch.current_ids.to(self.device)

        input_ids = c_ids[:, :-1]
        labels = input_ids

        # Memory dropout: with probability p, drop memory entirely (M_c=None).
        drop_mem = step_rng.random() < a.memory_dropout
        if drop_mem:
            M_c = None
        else:
            M_c = self._model().model.compute_memory(
                h_ids, detach_embeddings=a.detach_history_embeddings
            )

        # Context dropout: occasionally block the suffix of the current
        # session from attending to the prefix.  This forces the suffix
        # tokens' predictions to rely on the memory channel rather than
        # sequence context, training the cognitive-routing property
        # explicitly.
        drop_ctx = step_rng.random() < a.context_dropout
        attn_mask = None
        if drop_ctx and input_ids.shape[1] >= 4:
            cutoff = step_rng.randint(1, input_ids.shape[1] // 2)
            attn_mask = make_attn_mask_dict(
                input_ids.shape[1], cutoff, self.device, dtype=torch.bfloat16
            )

        out = self.wrapped(
            input_ids=input_ids,
            labels=labels,
            M_c=M_c,
            attention_mask=attn_mask,
        )
        loss = out.loss / a.grad_accum
        loss.backward()
        return {
            "loss": loss.item() * a.grad_accum,
            "drop_mem": drop_mem,
            "drop_ctx": drop_ctx,
        }

    @torch.no_grad()
    def evaluate(self) -> dict:
        a = self.args
        # Use the unwrapped model for eval so DDP doesn't (a) try to sync
        # gradients on no_grad forward passes and (b) drop ``collect_alpha_trace``
        # from kwargs.  All ranks still drive identical eval data so DDP's
        # internal state remains consistent for the next training step.
        model = self._model()
        model.eval()
        loader = self.eval_loader
        ce_mem, ce_no, ce_shuffle = [], [], []
        gate_alphas: list[torch.Tensor] = []
        n_seen = 0
        gen = loader.iter_samples()
        for sample in gen:
            if n_seen >= a.eval_n:
                break
            n_seen += 1
            h_ids = sample.history_ids.to(self.device)
            c_ids = sample.current_ids.to(self.device)
            sh_ids = sample.shuffled_history_ids.to(self.device)
            input_ids = c_ids[:, :-1]
            labels = input_ids
            M_c = model.model.compute_memory(h_ids)
            M_sh = model.model.compute_memory(sh_ids)
            ce_mem.append(
                model(input_ids=input_ids, labels=labels, M_c=M_c).loss.item()
            )
            ce_no.append(
                model(input_ids=input_ids, labels=labels, M_c=None).loss.item()
            )
            ce_shuffle.append(
                model(input_ids=input_ids, labels=labels, M_c=M_sh).loss.item()
            )
            out = model(
                input_ids=input_ids[:, : min(64, input_ids.shape[1])],
                M_c=M_c,
                collect_alpha_trace=True,
            )
            if getattr(out, "alpha_trace", None):
                gate_alphas.append(
                    torch.stack(
                        [a_layer.float().mean() for a_layer in out.alpha_trace]
                    )
                )
        model.train()
        if not ce_mem:
            return {}
        m = sum(ce_mem) / len(ce_mem)
        n = sum(ce_no) / len(ce_no)
        s = sum(ce_shuffle) / len(ce_shuffle)
        mean_alpha = (
            torch.stack(gate_alphas).mean(dim=0) if gate_alphas else torch.tensor([])
        )
        return {
            "n": len(ce_mem),
            "ce_mem": m,
            "ce_nomem": n,
            "ce_shuffle": s,
            "ce_nomem_minus_mem": n - m,
            "ce_shuffle_minus_mem": s - m,
            "mean_alpha_per_sublayer": mean_alpha.tolist(),
            "mean_alpha_overall": (
                float(mean_alpha.mean()) if mean_alpha.numel() else float("nan")
            ),
        }

    def _save(self, tag: str, eval_metrics: dict | None = None) -> None:
        if not self.is_main:
            return
        ckpt = Path(self.args.out_dir) / tag
        ckpt.mkdir(parents=True, exist_ok=True)
        self._model().save_pretrained(ckpt)
        self.tokenizer.save_pretrained(ckpt)
        torch.save(
            {
                "step": self.global_step,
                "best_eval_ce": self.best_eval_ce,
            },
            ckpt / "trainer_state.pt",
        )
        if eval_metrics is not None:
            (ckpt / "eval_metrics.json").write_text(
                json.dumps(eval_metrics, indent=2), encoding="utf-8"
            )
        print(f"Saved checkpoint -> {ckpt}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = Path(path) / "trainer_state.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu")
            self.global_step = state.get("step", 0)
            self.best_eval_ce = state.get("best_eval_ce", float("inf"))
            if self.is_main:
                print(f"Resumed from step {self.global_step}")

    def _log(
        self,
        step: int,
        accum_loss: float,
        grad_norm: float,
        tokens_seen: int,
        t0: float,
        drop_stats: dict,
    ) -> None:
        # All ranks must participate in the all_reduce (collective op);
        # only rank 0 actually prints.
        loss_t = torch.tensor(accum_loss, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        if not self.is_main:
            return
        elapsed = time.time() - t0
        tok_sec = tokens_seen * self.world_size / max(elapsed, 1e-6)
        lr_now = [g["lr"] for g in self.optimizer.param_groups]
        avg_loss = loss_t.item()
        gate_mean = (
            self._model().model.memory_gate.gate.float().mean().item()
            if hasattr(self._model().model, "memory_gate")
            else float("nan")
        )
        gate_max = (
            self._model().model.memory_gate.gate.float().abs().max().item()
            if hasattr(self._model().model, "memory_gate")
            else float("nan")
        )
        print(
            f"step {step:6d} | loss {avg_loss:.4f} | lrs {lr_now} | "
            f"grad_norm {grad_norm:.3f} | mem_drop {drop_stats['mem_drop_rate']:.2f} "
            f"| ctx_drop {drop_stats['ctx_drop_rate']:.2f} | "
            f"gate_mean {gate_mean:+.4f} gate_max {gate_max:.4f} | "
            f"{tok_sec / 1e3:.1f}k tok/s"
        )
        if self.use_wandb:
            import wandb

            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/lr_memres": lr_now[0],
                    "train/lr_backbone": lr_now[-1],
                    "train/grad_norm": grad_norm,
                    "train/tok_per_s": tok_sec,
                    "train/mem_drop_rate": drop_stats["mem_drop_rate"],
                    "train/ctx_drop_rate": drop_stats["ctx_drop_rate"],
                    "train/gate_mean": gate_mean,
                    "train/gate_abs_max": gate_max,
                },
                step=step,
            )

    def fit(self) -> None:
        a = self.args
        if self.is_main:
            print("  entering training loop", flush=True)
        self.wrapped.train()
        self.optimizer.zero_grad(set_to_none=True)
        rng = random.Random(a.seed + self.rank * 7919)

        accum_loss = 0.0
        accum_step = 0
        tokens_seen = 0
        n_mem_drop = 0
        n_ctx_drop = 0
        n_seen = 0
        t0 = time.time()

        loader = self.train_loader.iter_batches(a.batch_size)
        while self.global_step < a.steps:
            try:
                batch = next(loader)
            except StopIteration:
                break

            stats = self._train_step(batch, rng)
            accum_loss += stats["loss"]
            n_mem_drop += int(stats["drop_mem"])
            n_ctx_drop += int(stats["drop_ctx"])
            n_seen += 1
            tokens_seen += a.current_len * a.batch_size
            accum_step += 1
            if accum_step < a.grad_accum:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.wrapped.parameters(), a.max_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            accum_step = 0

            if self.global_step % a.log_every == 0:
                drop_stats = {
                    "mem_drop_rate": n_mem_drop / max(n_seen, 1),
                    "ctx_drop_rate": n_ctx_drop / max(n_seen, 1),
                }
                self._log(
                    self.global_step,
                    accum_loss / a.grad_accum,
                    float(grad_norm),
                    tokens_seen,
                    t0,
                    drop_stats,
                )
                t0 = time.time()
                tokens_seen = 0
                n_mem_drop = 0
                n_ctx_drop = 0
                n_seen = 0
            accum_loss = 0.0

            if self.global_step % a.eval_every == 0:
                metrics = self.evaluate()
                if metrics and self.is_main:
                    print(
                        f"  EVAL @ step {self.global_step}: "
                        f"n={metrics['n']} "
                        f"mem={metrics['ce_mem']:.4f} "
                        f"nomem={metrics['ce_nomem']:.4f} "
                        f"shuffle={metrics['ce_shuffle']:.4f} "
                        f"Δ_nm-m={metrics['ce_nomem_minus_mem']:+.4f} "
                        f"Δ_sh-m={metrics['ce_shuffle_minus_mem']:+.4f} "
                        f"alpha={metrics['mean_alpha_overall']:.4f}"
                    )
                    if self.use_wandb:
                        import wandb

                        wandb.log(
                            {
                                "eval/ce_mem": metrics["ce_mem"],
                                "eval/ce_nomem": metrics["ce_nomem"],
                                "eval/ce_shuffle": metrics["ce_shuffle"],
                                "eval/delta_nomem_minus_mem": metrics[
                                    "ce_nomem_minus_mem"
                                ],
                                "eval/delta_shuffle_minus_mem": metrics[
                                    "ce_shuffle_minus_mem"
                                ],
                                "eval/mean_alpha_overall": metrics[
                                    "mean_alpha_overall"
                                ],
                            },
                            step=self.global_step,
                        )
                    if metrics["ce_mem"] < self.best_eval_ce:
                        self.best_eval_ce = metrics["ce_mem"]
                        self._save("best", eval_metrics=metrics)

            if self.global_step % a.save_every == 0:
                self._save(f"step-{self.global_step}")

        # Final eval + save.
        final_metrics = self.evaluate()
        self._save("final", eval_metrics=final_metrics)
        if self.is_main and self.use_wandb:
            import wandb

            wandb.finish()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    Trainer(args).fit()


if __name__ == "__main__":
    main()
