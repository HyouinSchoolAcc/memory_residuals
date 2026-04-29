#!/usr/bin/env python3
"""Recurrent chain trainer for Memory Residuals (Phase 1, the real one).

This is the trainer that actually exercises the recurrent state $M_c$.  Each
training step:

    1.  Sample a chain (a PG-19 book or a TV show) from the pre-tokenised
        chain corpus produced by ``paper_tools/pretokenize_chains.py``.
    2.  Sample a window of $k$ consecutive sessions inside that chain.
    3.  Initialise $M_c \\leftarrow 0$ (or carry the detached $M_c$ from the
        previous window if --carry_state is set).
    4.  Iterate the window left-to-right.  At session $t$:
            (a) Forward with the *current* $M_c$, compute next-token NLL on
                session $t$.  This is the loss the optimizer actually sees.
            (b) Update memory: $M_c \\leftarrow$ judge($M_c$, extract($C_t$)).
                The judge step compares the *previous* memory against the new
                candidate so old and new compete for slots, exactly per the
                paper.
    5.  Backprop $\\sum_t L_t / k$ through all $k$ forward passes and all $k$
        recurrent judge steps.

This is *not* the ``train_phase1.py`` setup, where the four-session history
was concatenated into a single text field and compressed once per step.
That setup is a useful warm-up because it teaches the readout/extraction
parameters to be useful at all, but it does not train the recurrent
competition.  This trainer does.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig, AutoTokenizer

from modeling_memres import (
    Qwen3MemResConfig,
    Qwen3MemResForCausalLM,
    _normalise_memres_mode,
)
from presets import PRESETS, apply_preset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument("--pretrained", default=None)
    p.add_argument(
        "--memres_mode",
        # Canonical names: simple_gate / attention_base / attention_parity.
        # Legacy names "residual" / "block_attnres" are accepted and folded
        # into the canonical set after parsing (see _normalise_memres_args)
        # so old shell scripts and saved checkpoints keep working.
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
    p.add_argument("--init_from", default=None,
                   help="Optional: warm-start from a Phase-0 (pair-trained) checkpoint")

    # Router init-bias overrides.  Default (None / None) -> mode-derived
    # defaults: attention_parity uses (-32, +32) for bit-exact init parity
    # with a saturated softmax; attention_base / simple_gate uses (-8, 0).
    # Setting these to less extreme magnitudes (e.g. -4 / +4) softens the
    # parity init: step-0 logits drift slightly from bare Qwen3, but the
    # softmax has non-trivial gradient on the pseudo-queries from step 0
    # and the router can actually learn to recruit memory.
    p.add_argument(
        "--router_mem_bias_init", type=float, default=None,
        help="Override BlockAttnResRouter.mem_bias init value. "
             "Default: -32 (attention_parity) / -8 (other modes).",
    )
    p.add_argument(
        "--router_recent_bias_init", type=float, default=None,
        help="Override BlockAttnResRouter.recent_bias init value. "
             "Default: +32 (attention_parity) / 0 (other modes).",
    )

    # MemRes hyper-params (overridden by --preset)
    p.add_argument("--memres_num_vectors", type=int, default=128)
    p.add_argument("--memres_extraction_depth", type=int, default=0)
    p.add_argument("--memres_num_blocks", type=int, default=8)
    p.add_argument(
        "--memres_extract_source", default="embed",
        help="Source for the per-token representation C_t fed to the "
             "extraction stack. 'embed' = bag-of-token-embeddings (legacy "
             "default); 'hidden_<L>' = bare-backbone hidden state at "
             "layer L (no_grad, detached) -- contextualised representation "
             "that carries syntax/anaphora/entity binding into M_in's "
             "cross-attention. 'hidden_14' is the recommended default for "
             "a 28-layer Qwen3 backbone (mid-stack, semantic).",
    )

    # Data
    p.add_argument("--train_chains",
                   default="paper_artifacts/chains/stage1_train_s512.pt")
    p.add_argument("--eval_chains",
                   default="paper_artifacts/chains/stage1_validation_s512.pt")
    p.add_argument("--session_len", type=int, default=512,
                   help="Must match the .pt file's session_len.")
    p.add_argument("--window_k", type=int, default=4,
                   help="TBPTT window size: # sessions backpropped through "
                        "the recurrent judge stack per step.")
    p.add_argument("--source_weights",
                   default=None,
                   help="Optional JSON mapping source -> upweight, e.g. "
                        "'{\"pg19\":1.0,\"tv\":4.0,\"msc\":3.0}'.  "
                        "Source is detected from chain_name prefix: "
                        "'msc_' -> 'msc', leading digit -> 'pg19', "
                        "anything else -> 'tv'.  Default weights are "
                        "{pg19:1.0, tv:4.0, msc:3.0}.")

    # Optimization
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Number of independent chains processed in parallel "
                        "per step (NOT effective batch).")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_backbone", type=float, default=3e-6)
    p.add_argument("--lr_min_ratio", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.1)

    # Regularisers
    p.add_argument("--memory_dropout", type=float, default=0.10,
                   help="Per-session probability of zeroing M_c on the "
                        "*read* path (the judge update still runs).")
    p.add_argument("--context_dropout", type=float, default=0.30,
                   help="Per-session probability of masking the prefix of "
                        "the current session, forcing memory reliance.")
    p.add_argument("--carry_state", action="store_true",
                   help="If set, persist M_c across windows (detached). "
                        "Otherwise each window starts from zero.")
    p.add_argument("--neg_chain_weight", type=float, default=0.0,
                   help="Per-step weight on the negative-chain replay "
                        "auxiliary loss.  At each step we draw a *different* "
                        "chain's first session, build a shuffle M_c from it, "
                        "score the matched chain's last TBPTT session under "
                        "that shuffle memory, and add  alpha * max(0, "
                        "L_match - L_shuffle + margin)  to the total loss. "
                        "This is the PITFALLS.md §3 prescription against "
                        "style-only / shortcut-learning collapse.  0 disables. "
                        "When --neg_chain_warmup_steps > 0 this is the FINAL "
                        "weight, ramped linearly from --neg_chain_initial_weight."
                   )
    p.add_argument("--neg_chain_initial_weight", type=float, default=None,
                   help="Initial value of neg_chain_weight at step 0; ramps "
                        "linearly to --neg_chain_weight over "
                        "--neg_chain_warmup_steps. Default: equal to "
                        "--neg_chain_weight (no ramp).")
    p.add_argument("--neg_chain_warmup_steps", type=int, default=0,
                   help="Number of steps over which to linearly ramp "
                        "neg_chain_weight from initial -> final. The "
                        "COMPREHENSIVE.md §4.1 Phase B curriculum uses "
                        "0.05 -> 0.2 -> 0.5 in three phases; in this "
                        "trainer we fold that into a single linear ramp.")
    p.add_argument("--neg_chain_margin", type=float, default=0.05,
                   help="Margin for the negative-chain auxiliary loss.")

    # Intra-chain perturbation contrastive loss.  A *strictly stronger*
    # version of the inter-chain neg_chain_weight loss:
    #
    # - neg_chain_weight: contrasts same-chain last session under
    #   different-chain M_c.  Catches "memory learned style" but not
    #   "memory drops salient facts under interference from sessions
    #   between fact and recall".
    # - in_chain_contrast_weight: contrasts same-chain last session
    #   under same-chain M_c with one earlier session swapped for a
    #   random distractor.  The contrast concentrates on fact-recall
    #   tokens because most of the last-session NLL is invariant to
    #   the swap.  Gradient flows through the judge step at every
    #   intermediate session, putting direct pressure on the two-stage
    #   QKV competition to *preserve* fact-bearing channels even while
    #   sessions in between are pushing irrelevant updates.
    p.add_argument("--in_chain_contrast_weight", type=float, default=0.0,
                   help="Weight on the intra-chain perturbation "
                        "contrastive loss.  0 disables.  See block "
                        "comment in the trainer for the full spec.")
    p.add_argument("--in_chain_contrast_initial_weight", type=float, default=None,
                   help="Initial value of in_chain_contrast_weight at "
                        "step 0; ramps linearly to "
                        "--in_chain_contrast_weight over "
                        "--in_chain_contrast_warmup_steps. "
                        "Default: equal to --in_chain_contrast_weight (no ramp).")
    p.add_argument("--in_chain_contrast_warmup_steps", type=int, default=0,
                   help="Linear warmup for in_chain_contrast_weight.")
    p.add_argument("--in_chain_contrast_margin", type=float, default=0.05,
                   help="Margin for the in-chain contrastive loss.")
    p.add_argument("--in_chain_perturb_strategy",
                   choices=("session_zero", "random_earlier"),
                   default="random_earlier",
                   help="Which session in the window to swap for the "
                        "perturbed M_c. session_zero: always perturb "
                        "session 0 (the persona prefix in MSC). "
                        "random_earlier (default): perturb a random "
                        "session in [0, window_k-2] each step, so "
                        "every position gets pressured over training.")
    p.add_argument("--burn_in_max", type=int, default=12,
                   help="Maximum number of sessions to unroll under "
                        "no_grad before the TBPTT window starts.  This lets "
                        "the model see M_c states from realistic recurrence "
                        "depths during training; needed because eval-time "
                        "chains are 20+ sessions long while TBPTT is only "
                        "k sessions deep. With --burn_in_resample, the actual "
                        "burn-in is sampled per-step in {0..burn_in_max} "
                        "(LRMT failure-mode defense, COMPREHENSIVE §4.1).")
    p.add_argument("--burn_in_resample", action="store_true",
                   help="If set, resample burn-in per chain in {0,4,8,...,burn_in_max} "
                        "so the model sees M_c states from a range of recurrence "
                        "depths during training.")
    p.add_argument("--save_best_metric", choices=("ce_mem", "composite"),
                   default="composite",
                   help="ce_mem: legacy; minimised. composite: maximise "
                        "delta_nomem_minus_mem + 2*delta_shuffle_minus_mem "
                        "(penalises both channel collapse and shortcut "
                        "learning). Default: composite.")

    # Conversational-pipeline knobs
    p.add_argument("--mask_padding_loss", action="store_true",
                   help="Mask EOS-padding positions from the LM loss. "
                        "Essential for conversational corpora (MSC) where "
                        "sessions are ~150 tokens padded with EOS to "
                        "session_len; without masking ~70%% of the loss "
                        "is EOS-on-EOS noise. Detected by the first EOS "
                        "in each row of input_ids -- safe iff the "
                        "tokeniser only emits EOS as trailing padding "
                        "(which is what pretokenize_chains.py does).")
    p.add_argument("--score_tail_frac", type=float, default=1.0,
                   help="Score only the last fraction of each session's "
                        "*non-padding* content.  E.g. 0.5 means only the "
                        "second half of each session contributes to the "
                        "LM loss; the first half is treated as context. "
                        "Concentrates gradient on the response tail "
                        "where memory should matter most.  Default 1.0 "
                        "= legacy (score the entire content).")

    # IO + logging
    p.add_argument("--out_dir", default="output/chain_run")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_n_chains", type=int, default=24,
                   help="# eval chains; we use the *full* prefix per chain "
                        "(up to a clamp) to score the last few sessions.")
    p.add_argument("--eval_window", type=int, default=4,
                   help="Score the last N sessions of each eval chain "
                        "(M_c built sequentially over all preceding sessions "
                        "regardless of this clamp).")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")

    a = p.parse_args()
    if a.preset is not None:
        apply_preset(a, a.preset)
    a.memres_mode = _normalise_memres_mode(
        a.memres_mode, a.block_attnres_parity_init
    )
    a.block_attnres_parity_init = a.memres_mode == "attention_parity"
    return a


def cosine_with_warmup(step: int, warmup: int, total: int, lr_min_ratio: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cos


# ---------------------------------------------------------------------------
# Chain dataset
# ---------------------------------------------------------------------------


class ChainCorpus:
    """Stores one big (N_sessions, S) tensor and per-chain metadata.

    The training sampler picks a chain (with optional source up-weighting)
    and a random window inside it.
    """

    def __init__(self, path: Path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.session_ids: torch.Tensor = blob["session_ids"].long()
        self.chain_starts: torch.Tensor = blob["chain_starts"]
        self.chain_lengths: torch.Tensor = blob["chain_lengths"]
        self.chain_names: list[str] = blob["chain_names"]
        self.session_len: int = int(blob["session_len"])
        self.tokenizer: str = blob["tokenizer"]

    def __len__(self) -> int:
        return len(self.chain_starts)

    def chain_session_at(self, chain_idx: int, position: int) -> torch.Tensor:
        start = int(self.chain_starts[chain_idx])
        return self.session_ids[start + position]

    def chain_window(self, chain_idx: int, start: int, k: int) -> torch.Tensor:
        s = int(self.chain_starts[chain_idx])
        return self.session_ids[s + start : s + start + k]

    def chain_prefix(self, chain_idx: int, end: int) -> torch.Tensor:
        s = int(self.chain_starts[chain_idx])
        return self.session_ids[s : s + end]


def _detect_source(name: str) -> str:
    """Detect the corpus source of a chain from its name.

    'msc_<id>'  -> 'msc'   (multi-session chat dialogues)
    leading digit -> 'pg19'   (PG-19 books are named '<book_id>')
    anything else -> 'tv'    (TV episode chains have show-name prefixes)
    """
    if name.startswith("msc_"):
        return "msc"
    if name[:1].isdigit():
        return "pg19"
    return "tv"


class ChainSampler:
    def __init__(
        self,
        corpus: ChainCorpus,
        rank: int,
        world_size: int,
        seed: int,
        window_k: int,
        source_weights: dict[str, float] | None = None,
    ):
        self.corpus = corpus
        self.rank = rank
        self.world_size = world_size
        self.window_k = window_k
        self.rng = random.Random(seed + rank * 7919)

        # Default source up-weights.  PG-19 has ~218k chains so it dominates
        # by token count; TV (~30 chains) and MSC (~4k chains) get
        # multiplicative up-weight on per-chain length so the sampler
        # actually visits them.
        default_w = {"pg19": 1.0, "tv": 4.0, "msc": 3.0}
        if source_weights:
            default_w.update(source_weights)

        weights: list[float] = []
        source_counts: dict[str, int] = {}
        for ci, name in enumerate(corpus.chain_names):
            length = int(corpus.chain_lengths[ci])
            src = _detect_source(name)
            source_counts[src] = source_counts.get(src, 0) + 1
            # Eligibility: a chain of length L can produce a window of
            # window_k sessions iff L >= window_k (start=0 is always
            # legal even when no burn-in prefix is available).  This is
            # critical for short conversational chains (MSC): 3-session
            # dialogues are eligible at window_k=3 with start=0.
            if length < window_k:
                weights.append(0.0)
                continue
            w = default_w.get(src, 1.0)
            weights.append(w * length)
        self._source_counts = source_counts
        self._effective_weights = default_w
        self.weights = weights
        self.cum = []
        running = 0.0
        for w in weights:
            running += w
            self.cum.append(running)
        self.total = running
        self.eligible = sum(1 for w in weights if w > 0)
        if self.total <= 0:
            raise RuntimeError("No eligible chains for sampling")

    def sample_window(self) -> tuple[int, int, torch.Tensor, torch.Tensor | None]:
        r = self.rng.random() * self.total
        # binary search
        lo, hi = 0, len(self.cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r <= self.cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        chain_idx = lo
        length = int(self.corpus.chain_lengths[chain_idx])
        max_start = length - self.window_k
        start = self.rng.randint(0, max_start)
        window = self.corpus.chain_window(chain_idx, start, self.window_k)
        # Burn-in: optionally include the chain prefix [0:start] for a
        # no-grad recurrent unroll before TBPTT.  Returns None when start=0.
        burn_in = (
            self.corpus.chain_window(chain_idx, 0, start) if start > 0 else None
        )
        return chain_idx, start, window, burn_in


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


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
        # EOS token id used by pretokenize_chains.py to right-pad sessions
        # to session_len.  Conversational corpora (MSC) have sessions
        # ~150 tokens padded to 512, so masking padding from the LM loss
        # is essential -- otherwise ~70%% of the loss is the easy
        # EOS-on-EOS prediction and gradient signal on real content
        # collapses.
        self.eos_id = int(self.tokenizer.eos_token_id)

        if self.is_main:
            print(self._banner())

        self.model = self._build_model()
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self._apply_freeze()
        self.wrapped = self._wrap_model()
        self.optimizer, self.scheduler = self._build_optimizer()

        if self.is_main:
            print(f"  loading train chains: {args.train_chains}", flush=True)
        self.train_corpus = ChainCorpus(Path(args.train_chains))
        sw = None
        if args.source_weights:
            try:
                sw = json.loads(args.source_weights)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"--source_weights must be valid JSON; got {args.source_weights!r}"
                ) from e
        self.train_sampler = ChainSampler(
            self.train_corpus, self.rank, self.world_size, args.seed,
            args.window_k, source_weights=sw,
        )
        if self.is_main:
            print(
                f"  train: {len(self.train_corpus)} chains, "
                f"{self.train_sampler.eligible} eligible "
                f"(window_k={args.window_k}); "
                f"source counts={self.train_sampler._source_counts}; "
                f"weights={self.train_sampler._effective_weights}",
                flush=True,
            )

        if self.is_main:
            print(f"  loading eval chains: {args.eval_chains}", flush=True)
        self.eval_corpus = ChainCorpus(Path(args.eval_chains))
        if self.is_main:
            print(
                f"  eval: {len(self.eval_corpus)} chains, "
                f"sessions: {self.eval_corpus.session_ids.shape[0]}",
                flush=True,
            )

        os.makedirs(args.out_dir, exist_ok=True)
        self.best_eval_ce = float("inf")
        self.global_step = 0

        if args.init_from:
            if self.is_main:
                print(f"  warm-start (memres params only) from {args.init_from}",
                      flush=True)
            self._load_memres_warm_start(args.init_from)

    # ------------------------------------------------------------------
    # Distributed / model setup
    # ------------------------------------------------------------------

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
            tag = a.preset or "scratch"
            a.run_name = f"chain-{tag}-K{a.memres_num_vectors}-k{a.window_k}-{a.steps // 1000}k"

    def _tokenizer_path(self) -> str:
        if self.args.pretrained:
            return self.args.pretrained
        return "Qwen/Qwen3-0.6B"

    def _banner(self) -> str:
        a = self.args
        return (
            f"\n=== Memory Residuals -- recurrent chain trainer ===\n"
            f"  run_name        : {a.run_name}\n"
            f"  preset          : {a.preset}\n"
            f"  pretrained      : {a.pretrained}\n"
            f"  K, L_E, N       : {a.memres_num_vectors}, "
            f"{a.memres_extraction_depth}, {a.memres_num_blocks}\n"
            f"  session_len     : {a.session_len}, window_k={a.window_k}\n"
            f"  steps           : {a.steps}, bs={a.batch_size}, "
            f"grad_accum={a.grad_accum}\n"
            f"  lr              : memres={a.lr}, backbone={a.lr_backbone}\n"
            f"  reg             : mem_drop={a.memory_dropout}, "
            f"ctx_drop={a.context_dropout}, carry_state={a.carry_state}\n"
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
            print(f"W&B init failed ({exc})")
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
            memres_extract_source=a.memres_extract_source,
        )
        if a.pretrained:
            base_cfg = AutoConfig.from_pretrained(a.pretrained)
            cfg = Qwen3MemResConfig(**{**base_cfg.to_dict(), **memres_kwargs})
            return Qwen3MemResForCausalLM.from_pretrained(
                a.pretrained, config=cfg, dtype=torch.bfloat16
            ).to(self.device)
        raise ValueError("Chain trainer requires --pretrained or --preset")

    def _load_memres_warm_start(self, path: str) -> None:
        from safetensors.torch import load_file
        ckpt = Path(path) / "model.safetensors"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        state = load_file(str(ckpt))
        memres_keys = [k for k in state if any(
            m in k for m in ("memory_block", "memory_readout",
                              "memory_gate", "depth_router")
        )]
        target = self.model.state_dict()
        loaded = 0
        for k in memres_keys:
            if k in target and target[k].shape == state[k].shape:
                target[k] = state[k].to(target[k].dtype).to(target[k].device)
                loaded += 1
        self.model.load_state_dict(target)
        if self.is_main:
            print(f"  warm-started {loaded}/{len(memres_keys)} memres params")

    def _apply_freeze(self) -> None:
        # In simple_gate mode the depth_router is constructed (so checkpoint
        # shapes stay stable across modes) but unused on the forward path,
        # so its parameters never receive gradient and we freeze them
        # explicitly to keep the optimiser / DDP graph tight.
        if self.args.memres_mode == "simple_gate":
            for name, p in self.model.named_parameters():
                if "depth_router" in name:
                    p.requires_grad = False

    def _wrap_model(self):
        if self.world_size <= 1:
            return self.model
        # ``static_graph=True`` is required for chain TBPTT: the same memres
        # parameters (M_in, M_judge, gates, readout W_*) participate in
        # multiple forward passes inside a single backward, which makes DDP's
        # default ``find_unused_parameters`` machinery throw a "marked as
        # ready twice" error.
        ddp = DDP(
            self.model,
            device_ids=[self.local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,
        )
        return ddp

    def _model(self) -> Qwen3MemResForCausalLM:
        return self.wrapped.module if hasattr(self.wrapped, "module") else self.wrapped

    def _build_optimizer(self):
        a = self.args
        markers = ("memory_block", "memory_readout", "depth_router", "memory_gate")
        memres_params, backbone_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(m in name for m in markers):
                memres_params.append(p)
            else:
                backbone_params.append(p)
        groups = []
        if memres_params:
            groups.append({"params": memres_params, "lr": a.lr, "name": "memres"})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": a.lr_backbone,
                           "name": "backbone"})
        if self.is_main:
            print(
                f"  param groups    : memres={sum(p.numel() for p in memres_params)/1e6:.1f}M "
                f"@ {a.lr}, backbone={sum(p.numel() for p in backbone_params)/1e6:.1f}M "
                f"@ {a.lr_backbone}"
            )
        opt = AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.weight_decay)
        sched = LambdaLR(
            opt,
            lr_lambda=lambda s: cosine_with_warmup(s, a.warmup, a.steps, a.lr_min_ratio),
        )
        return opt, sched

    # ------------------------------------------------------------------
    # Recurrent training step
    # ------------------------------------------------------------------

    def _make_attn_mask(self, seq_len: int, cutoff: int) -> dict | None:
        if cutoff <= 0:
            return None
        neg_inf = torch.finfo(torch.bfloat16).min
        mask = torch.zeros(1, 1, seq_len, seq_len, device=self.device,
                           dtype=torch.bfloat16)
        upper = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device), diagonal=1
        ).bool()
        mask[:, :, upper] = neg_inf
        mask[:, :, cutoff:, :cutoff] = neg_inf
        return {"full_attention": mask}

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Construct loss labels honouring --mask_padding_loss / --score_tail_frac.

        Logic:
        - If neither knob is set, labels == input_ids (legacy behaviour).
        - With --mask_padding_loss, every position where input_ids == eos_id
          is set to -100 so it does not contribute to the LM loss.  This
          assumes EOS only appears as trailing right-padding (true for
          pretokenize_chains.py output).
        - With --score_tail_frac F < 1.0, the first (1-F) fraction of each
          row's *non-padding* content is also masked to -100, so only the
          tail F contributes to loss.  Computed per-row from the row's
          actual content length.
        """
        a = self.args
        if not a.mask_padding_loss and a.score_tail_frac >= 1.0:
            return input_ids
        labels = input_ids.clone()
        is_pad = input_ids == self.eos_id  # (B, S) bool
        if a.mask_padding_loss:
            labels = labels.masked_fill(is_pad, -100)
        if a.score_tail_frac < 1.0:
            # content_len[bi] = number of non-padding tokens in row bi.
            # When mask_padding_loss is also set, this matches the number
            # of positions still contributing to loss after masking.
            # When it's not set, we still use the same definition so the
            # tail fraction is over real content (not over padded length).
            content_len = (~is_pad).long().sum(dim=1)
            B = input_ids.shape[0]
            for bi in range(B):
                cl = int(content_len[bi].item())
                if cl <= 0:
                    continue
                head_cut = int(cl * (1.0 - a.score_tail_frac))
                if head_cut > 0:
                    labels[bi, :head_cut] = -100
        return labels

    def _train_step(self, rng: random.Random,
                    carry: torch.Tensor | None) -> tuple[float, torch.Tensor | None]:
        a = self.args
        # Sample one chain window per micro-batch element.  Cap the burn-in
        # length at ``--burn_in_max`` for memory; longer prefixes are simply
        # tail-truncated (we keep the most recent burn_in_max sessions).
        windows = []
        burn_ins: list[torch.Tensor | None] = []
        # Long-horizon training protocol: when --burn_in_resample is set, draw
        # a per-chain burn-in length from {0, 4, 8, ..., burn_in_max} so the
        # judge step sees M_c at a range of recurrence depths each batch.
        # Otherwise, behave as before: cap at burn_in_max, full prefix.
        for _ in range(a.batch_size):
            _, _, w, b = self.train_sampler.sample_window()
            if a.burn_in_resample and a.burn_in_max > 0 and b is not None:
                step = max(4, a.burn_in_max // 5)
                choices = [0] + list(range(step, a.burn_in_max + 1, step))
                pick = rng.choice(choices)
                if pick == 0 or b.shape[0] == 0:
                    b = None
                else:
                    b = b[-pick:]
            elif b is not None and a.burn_in_max > 0 and b.shape[0] > a.burn_in_max:
                b = b[-a.burn_in_max:]
            elif a.burn_in_max == 0:
                b = None
            windows.append(w)
            burn_ins.append(b)
        # (B, k, S)
        window = torch.stack(windows).to(self.device)

        model = self._model()
        cfg = model.config
        K, d = cfg.memres_num_vectors, cfg.hidden_size
        B = window.shape[0]

        # Initial memory: zero tensor (NOT None) so the graph topology is
        # constant across iterations -- DDP static_graph=True requires this.
        if carry is not None and a.carry_state:
            M_c = carry.detach()
        else:
            M_c = torch.zeros(B, K, d, device=self.device, dtype=torch.bfloat16)

        # Snapshot M_c at the START of the TBPTT window (after burn-in,
        # after carry-state).  The intra-chain contrastive loss
        # re-builds a perturbed M_c starting from this exact state, so
        # positive and perturbed only differ in one swapped session.
        # Captured here before the main loop mutates M_c.
        M_c_window_start: torch.Tensor | None = None

        # Burn-in: process each batch element's prefix under no_grad, building
        # a per-element M_c at the right recurrence depth, then stack.
        # Non-uniform burn-in lengths require we process each element separately.
        if any(b is not None for b in burn_ins):
            with torch.no_grad():
                burn_M = []
                for bi in range(B):
                    m = M_c[bi : bi + 1]
                    bp = burn_ins[bi]
                    if bp is not None:
                        bp = bp.to(self.device)
                        for j in range(bp.shape[0]):
                            sess_j = bp[j].unsqueeze(0)
                            C_j = model.model.extract_source(sess_j[:, :-1])
                            m = model.model.compress_session(C_j, m)
                    burn_M.append(m)
                M_c = torch.cat(burn_M, dim=0).detach()
        M_c_window_start = M_c.detach().clone()

        # Pre-compute a trivial causal mask we can always pass; we'll
        # optionally OR-in a context-dropout block.  The MemRes backbone
        # accepts a dict {"full_attention": (1, 1, S, S)} additive bias.
        seq = window.shape[2] - 1   # we feed [:, :-1]
        neg_inf = torch.finfo(torch.bfloat16).min
        causal_only = torch.zeros(
            1, 1, seq, seq, device=self.device, dtype=torch.bfloat16
        )
        upper = torch.triu(
            torch.ones(seq, seq, device=self.device), diagonal=1
        ).bool()
        causal_only[:, :, upper] = neg_inf
        causal_mask_dict = {"full_attention": causal_only}

        losses = []
        n_drop_mem = 0
        n_drop_ctx = 0
        last_input_ids = None
        last_labels = None
        last_M_c_pre = None  # M_c BEFORE the last session's update, for contrast.
        for t in range(a.window_k):
            session = window[:, t]                         # (B, S)
            input_ids = session[:, :-1]
            labels = self._build_labels(input_ids)

            # Memory dropout: zero out M_c (don't change topology).
            drop_mem = rng.random() < a.memory_dropout
            if drop_mem:
                read_M = torch.zeros_like(M_c)
                n_drop_mem += 1
            else:
                read_M = M_c

            # Context dropout: always pass a mask dict; sometimes block prefix.
            drop_ctx = rng.random() < a.context_dropout
            if drop_ctx and seq >= 4:
                cutoff = rng.randint(1, seq // 2)
                m = causal_only.clone()
                m[:, :, cutoff:, :cutoff] = neg_inf
                attn_mask = {"full_attention": m}
                n_drop_ctx += 1
            else:
                attn_mask = causal_mask_dict

            out = self.wrapped(
                input_ids=input_ids,
                labels=labels,
                M_c=read_M,
                attention_mask=attn_mask,
            )
            losses.append(out.loss)

            # Snapshot M_c right before the *last* session's update so we can
            # use it (and a paired shuffle M_c) for the negative-chain loss.
            if t == a.window_k - 1:
                last_input_ids = input_ids
                last_labels = labels
                last_M_c_pre = read_M  # what the loss above conditioned on

            # Recurrent memory update: M_c <- judge(M_c, extract(C_t)).
            # extract_source honours config.memres_extract_source: 'embed'
            # for legacy bag-of-token-embeddings, 'hidden_<L>' for the
            # contextualised mid-layer hidden state path.
            C_t = model.model.extract_source(input_ids)
            M_c = model.model.compress_session(C_t, M_c)

        loss_match = torch.stack(losses).mean()
        total_loss = loss_match

        # Negative-chain contrastive auxiliary loss (PITFALLS §3).  Build a
        # shuffle M_c from a randomly-chosen *different* chain (any chain,
        # any random window) and score the same matched-chain last session
        # under it.  We push the matched loss to be at least `margin` below
        # the shuffle loss; otherwise we add the gap to the total loss.
        if a.neg_chain_warmup_steps > 0:
            init_w = a.neg_chain_initial_weight
            if init_w is None:
                init_w = a.neg_chain_weight
            ramp = min(1.0, self.global_step / max(1, a.neg_chain_warmup_steps))
            cur_neg_weight = init_w + (a.neg_chain_weight - init_w) * ramp
        else:
            cur_neg_weight = a.neg_chain_weight
        if cur_neg_weight > 0.0:
            with torch.no_grad():
                shuffle_windows = []
                for _ in range(a.batch_size):
                    _, _, sw, sb = self.train_sampler.sample_window()
                    if sb is not None and a.burn_in_max > 0 and sb.shape[0] > a.burn_in_max:
                        sb = sb[-a.burn_in_max:]
                    elif a.burn_in_max == 0:
                        sb = None
                    shuffle_windows.append((sw, sb))
                # Build shuffle M_c per batch element.
                M_sh_list = []
                for bi, (sw, sb) in enumerate(shuffle_windows):
                    m = torch.zeros(1, K, d, device=self.device, dtype=torch.bfloat16)
                    if sb is not None:
                        sb = sb.to(self.device)
                        for j in range(sb.shape[0]):
                            sj = sb[j].unsqueeze(0)
                            Cj = model.model.extract_source(sj[:, :-1])
                            m = model.model.compress_session(Cj, m)
                    sw = sw.to(self.device)
                    for j in range(sw.shape[0]):
                        sj = sw[j].unsqueeze(0)
                        Cj = model.model.extract_source(sj[:, :-1])
                        m = model.model.compress_session(Cj, m)
                    M_sh_list.append(m)
                M_sh = torch.cat(M_sh_list, 0).detach()
            # Score the matched chain's last session under M_sh.  This loss
            # SHOULD be larger than loss_match -- if it isn't, memory is not
            # chain-specific and we get a positive contrastive penalty.
            out_sh = self.wrapped(
                input_ids=last_input_ids,
                labels=last_labels,
                M_c=M_sh,
                attention_mask=causal_mask_dict,
            )
            loss_shuffle = out_sh.loss
            margin_loss = (loss_match - loss_shuffle + a.neg_chain_margin).clamp(min=0.0)
            total_loss = total_loss + cur_neg_weight * margin_loss

        # ------------------------------------------------------------
        # Intra-chain perturbation contrastive loss (the "did the fact
        # survive interference?" loss).
        #
        # Build a perturbed M_c by re-running TBPTT through the same
        # window with ONE earlier session swapped for a random
        # other-chain session.  Score the matched chain's last session
        # under both positive and perturbed M_c; require positive to be
        # at least `margin` lower in NLL.
        #
        # Gradient through the perturbed-build pressures the judge step
        # at every intermediate session to PRESERVE channels carrying
        # information that turns out to matter at the recall position
        # -- the QKV competition is forced to defend M_c_prev when
        # M_new (current session) does not contain the salient content.
        #
        # Cost: one extra TBPTT chain build (with grad) + one extra
        # forward+backward on the recall session.  ~30%% per-step
        # overhead at window_k=3.
        if a.in_chain_contrast_warmup_steps > 0:
            ic_init = a.in_chain_contrast_initial_weight
            if ic_init is None:
                ic_init = a.in_chain_contrast_weight
            ic_ramp = min(
                1.0,
                self.global_step / max(1, a.in_chain_contrast_warmup_steps),
            )
            cur_ic_weight = ic_init + (a.in_chain_contrast_weight - ic_init) * ic_ramp
        else:
            cur_ic_weight = a.in_chain_contrast_weight
        if cur_ic_weight > 0.0 and a.window_k >= 2 and M_c_window_start is not None:
            # 1. Pick the perturbation slot.  random_earlier rotates the
            #    pressure across all earlier slots so the model can't
            #    learn to "skip" a fixed position.
            if a.in_chain_perturb_strategy == "session_zero":
                F = 0
            else:  # random_earlier
                F = rng.randint(0, max(0, a.window_k - 2))

            # 2. Sample a distractor session per batch element by
            #    drawing a random other-chain window and taking its
            #    session 0.  no_grad: just data plumbing, no params.
            with torch.no_grad():
                distractor_list = []
                for _ in range(B):
                    _, _, dw, _ = self.train_sampler.sample_window()
                    distractor_list.append(dw[0])
                distractor_F = torch.stack(distractor_list).to(self.device)

            # 3. Re-build M_c through the window with session F swapped.
            #    WITH grad -- this is what carries the recall signal
            #    backwards into the intermediate judge / extract steps.
            M_c_pert = M_c_window_start.clone()
            for t in range(a.window_k - 1):
                if t == F:
                    sess_t = distractor_F
                else:
                    sess_t = window[:, t]
                ids_t = sess_t[:, :-1]
                C_t_pert = model.model.extract_source(ids_t)
                M_c_pert = model.model.compress_session(C_t_pert, M_c_pert)

            # 4. Score the recall session under the perturbed M_c.
            out_pert = self.wrapped(
                input_ids=last_input_ids,
                labels=last_labels,
                M_c=M_c_pert,
                attention_mask=causal_mask_dict,
            )
            loss_pert = out_pert.loss

            # 5. Margin loss on the recall position only.  We use
            #    losses[-1] (the last-session loss with the same
            #    dropout state as the recall forward) so positive and
            #    perturbed are compared like-for-like.
            loss_recall = losses[-1]
            ic_margin = (
                loss_recall - loss_pert + a.in_chain_contrast_margin
            ).clamp(min=0.0)
            total_loss = total_loss + cur_ic_weight * ic_margin

        total_loss = total_loss / a.grad_accum
        total_loss.backward()
        return float(loss_match.item()), M_c

    # ------------------------------------------------------------------
    # Recurrent evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict:
        a = self.args
        model = self._model()
        model.eval()
        ev = self.eval_corpus

        ce_mem, ce_no, ce_shuffle, ce_oracle = [], [], [], []
        for chain_idx in range(min(a.eval_n_chains, len(ev))):
            length = int(ev.chain_lengths[chain_idx])
            if length < a.eval_window + 1:
                continue
            # Compress sessions [0..end-1] into M_c, evaluate session [end].
            # We unroll the entire chain prefix recurrently for the full
            # cumulative-memory benefit.
            M_c = None
            # Score the LAST eval_window sessions (each conditioned on the
            # M_c built from all *strictly earlier* sessions).
            score_starts = range(length - a.eval_window, length)
            for end in range(length):
                sess = ev.chain_session_at(chain_idx, end).to(self.device).unsqueeze(0)
                input_ids = sess[:, :-1]
                if end in score_starts:
                    labels = input_ids
                    # Memory on
                    out_mem = model(input_ids=input_ids, labels=labels, M_c=M_c)
                    ce_mem.append(out_mem.loss.item())
                    # Memory off
                    out_no = model(input_ids=input_ids, labels=labels, M_c=None)
                    ce_no.append(out_no.loss.item())
                    # Shuffled memory: build a memory from a different chain's
                    # prefix of the same length end.
                    if M_c is not None:
                        other_idx = (chain_idx + 1) % len(ev)
                        other_len = int(ev.chain_lengths[other_idx])
                        clamp = min(end, other_len - 1)
                        if clamp > 0:
                            M_sh = None
                            for j in range(clamp):
                                osess = ev.chain_session_at(other_idx, j).to(self.device).unsqueeze(0)
                                C_o = model.model.extract_source(osess[:, :-1])
                                M_sh = model.model.compress_session(C_o, M_sh)
                            out_sh = model(input_ids=input_ids, labels=labels, M_c=M_sh)
                            ce_shuffle.append(out_sh.loss.item())

                    # Oracle: concat prior k-1 sessions + current as raw ctx.
                    # This is the "uncompressed full history" upper bound.
                    prior = ev.chain_window(
                        chain_idx, max(0, end - (a.eval_window - 1)), a.eval_window - 1
                    ).to(self.device)
                    if prior.numel() > 0:
                        prior_flat = prior.flatten().unsqueeze(0)  # (1, kp*S)
                        full = torch.cat([prior_flat, input_ids], dim=1)
                        labels_o = full.clone()
                        labels_o[:, : prior_flat.shape[1]] = -100
                        out_or = model(
                            input_ids=full, labels=labels_o, M_c=None
                        )
                        ce_oracle.append(out_or.loss.item())

                # Recurrent memory update is mandatory regardless of scoring.
                C_t = model.model.extract_source(sess[:, :-1])
                M_c = model.model.compress_session(C_t, M_c)
        model.train()

        def mean(xs):
            return float(sum(xs) / len(xs)) if xs else float("nan")

        return {
            "n_scored": len(ce_mem),
            "ce_mem": mean(ce_mem),
            "ce_nomem": mean(ce_no),
            "ce_shuffle": mean(ce_shuffle),
            "ce_oracle_concat": mean(ce_oracle),
            "delta_nomem_minus_mem": mean(ce_no) - mean(ce_mem),
            "delta_shuffle_minus_mem": mean(ce_shuffle) - mean(ce_mem),
            "delta_oracle_minus_mem": mean(ce_oracle) - mean(ce_mem),
        }

    def _save(self, tag: str, eval_metrics: dict | None = None) -> None:
        if not self.is_main:
            return
        ckpt = Path(self.args.out_dir) / tag
        ckpt.mkdir(parents=True, exist_ok=True)
        self._model().save_pretrained(ckpt)
        self.tokenizer.save_pretrained(ckpt)
        if eval_metrics:
            (ckpt / "eval_metrics.json").write_text(
                json.dumps(eval_metrics, indent=2), encoding="utf-8"
            )
        print(f"Saved checkpoint -> {ckpt}", flush=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        a = self.args
        self.wrapped.train()
        rng = random.Random(a.seed + self.rank * 7919 + 13)
        carry = None

        accum_loss = 0.0
        accum_step = 0
        tokens_seen = 0
        t0 = time.time()

        if self.is_main:
            print("  entering training loop", flush=True)

        self.optimizer.zero_grad(set_to_none=True)
        while self.global_step < a.steps:
            loss_val, carry = self._train_step(rng, carry)
            accum_loss += loss_val
            tokens_seen += a.window_k * a.session_len * a.batch_size
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
                # All-reduce loss across ranks (collective op).
                loss_t = torch.tensor(accum_loss / a.grad_accum, device=self.device)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                if self.is_main:
                    elapsed = time.time() - t0
                    tok_sec = tokens_seen * self.world_size / max(elapsed, 1e-6)
                    lr_now = [g["lr"] for g in self.optimizer.param_groups]
                    gate_mean = self._model().model.memory_gate.gate.float().mean().item()
                    gate_max = self._model().model.memory_gate.gate.float().abs().max().item()
                    print(
                        f"step {self.global_step:6d} | loss {loss_t.item():.4f} | "
                        f"lrs {lr_now} | grad_norm {float(grad_norm):.3f} | "
                        f"gate_mean {gate_mean:+.4f} max {gate_max:.4f} | "
                        f"{tok_sec/1e3:.1f}k tok/s",
                        flush=True,
                    )
                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": loss_t.item(),
                            "train/lr_memres": lr_now[0],
                            "train/lr_backbone": lr_now[-1],
                            "train/grad_norm": float(grad_norm),
                            "train/tok_per_s": tok_sec,
                            "train/gate_mean": gate_mean,
                            "train/gate_abs_max": gate_max,
                        }, step=self.global_step)
                t0 = time.time()
                tokens_seen = 0
            accum_loss = 0.0

            if self.global_step % a.eval_every == 0:
                metrics = self.evaluate()
                if metrics and self.is_main:
                    print(
                        f"  EVAL @ step {self.global_step}: n={metrics['n_scored']} "
                        f"mem={metrics['ce_mem']:.4f} nomem={metrics['ce_nomem']:.4f} "
                        f"shuffle={metrics['ce_shuffle']:.4f} oracle={metrics['ce_oracle_concat']:.4f} "
                        f"Δnm-m={metrics['delta_nomem_minus_mem']:+.4f} "
                        f"Δsh-m={metrics['delta_shuffle_minus_mem']:+.4f} "
                        f"Δor-m={metrics['delta_oracle_minus_mem']:+.4f}",
                        flush=True,
                    )
                    if self.use_wandb:
                        import wandb
                        wandb.log({f"eval/{k}": v for k, v in metrics.items()
                                   if isinstance(v, (int, float))},
                                  step=self.global_step)
                    # COMPREHENSIVE.md §4.1: minimising ce_mem alone is
                    # gameable by channel collapse (the model can null memory
                    # and still drop ce_mem by overfitting the bare backbone).
                    # The composite (Δ_nm-m + 2·Δ_sh-m) penalises both
                    # collapse (Δ_nm-m -> 0) and shortcut learning
                    # (Δ_sh-m -> 0 or negative). We negate so "lower is
                    # better" semantics of best_eval_ce remain.
                    if a.save_best_metric == "ce_mem":
                        score = metrics["ce_mem"]
                    else:
                        d_nm = metrics.get("delta_nomem_minus_mem", 0.0) or 0.0
                        d_sh = metrics.get("delta_shuffle_minus_mem", 0.0) or 0.0
                        score = -(d_nm + 2.0 * d_sh)
                    if score < self.best_eval_ce:
                        self.best_eval_ce = score
                        self._save("best", eval_metrics={
                            **metrics,
                            "_save_best_metric": a.save_best_metric,
                            "_save_best_score": score,
                        })

            if self.global_step % a.save_every == 0:
                self._save(f"step-{self.global_step}")

        final_metrics = self.evaluate()
        self._save("final", eval_metrics=final_metrics)
        if self.is_main and self.use_wandb:
            import wandb
            wandb.finish()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    Trainer(parse_args()).fit()


if __name__ == "__main__":
    main()
