#!/usr/bin/env python3
"""AUDIT A3-CLOUD step 2+4: Build empirical P(item|category) on train side,
then score val callbacks under that template prior.

- Reads synthd4v2_persona_callback_{train,val}_s512.pt.
- Extracts (category, item) per chain from chain_names (format:
  synthetic_persona_callback_<ci:05d>_<cat>_<item>_n<N>ev). Falls back
  to parsing the callback-session text if the name is missing.
- Builds P(item|category) over the closed 8x32 set with Laplace alpha=0.5.
- Saves the prior as a pickle and as JSON counts.
- For val, computes CE of the answer item under P(item|category) then
  distributes the negative-log prob across the callback-mask tokens. Our
  simplest defensible choice: assign the entire item-CE to the FIRST
  callback-mask token and 0 to the rest. Mean per-chain CE over n=128.

Run:
    python tools/audit_a3_template_prior_cloud.py
"""
from __future__ import annotations
import json, math, pickle, re, sys
from pathlib import Path
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
N_VAL = 128
ALPHA = 0.5

CATEGORIES = ["color", "fruit", "animal", "object", "sport", "tool", "instrument", "hobby"]
ITEM_TYPE_PHRASE = {
    "color": "color", "fruit": "fruit", "animal": "animal", "object": "thing",
    "sport": "sport", "tool": "tool", "instrument": "instrument", "hobby": "hobby",
}

CLOSED_SET = {
    "color": ["red","blue","green","yellow","purple","orange","pink","brown","black","white","gray","cyan","magenta","violet","indigo","teal","maroon","navy","olive","lime","scarlet","amber","coral","ivory","crimson","azure","beige","khaki","salmon","turquoise","lavender","mint"],
    "fruit": ["apple","banana","cherry","grape","orange","peach","pear","plum","strawberry","watermelon","pineapple","mango","kiwi","papaya","guava","lemon","lime","blueberry","raspberry","blackberry","cranberry","apricot","fig","pomegranate","coconut","avocado","lychee","passionfruit","tangerine","grapefruit","nectarine","persimmon"],
    "animal": ["dog","cat","horse","cow","sheep","pig","chicken","duck","rabbit","hamster","fox","wolf","bear","deer","elk","moose","lion","tiger","leopard","cheetah","elephant","giraffe","zebra","rhino","monkey","gorilla","panda","koala","kangaroo","penguin","dolphin","whale"],
    "object": ["chair","table","lamp","couch","bed","desk","shelf","mirror","clock","vase","rug","curtain","pillow","blanket","basket","bucket","ladder","broom","shovel","hammer","saw","wrench","drill","nail","candle","torch","telescope","microscope","compass","globe","atlas","map"],
    "sport": ["soccer","basketball","baseball","football","tennis","golf","hockey","rugby","cricket","volleyball","swimming","cycling","running","skiing","snowboarding","surfing","boxing","wrestling","judo","karate","fencing","archery","rowing","sailing","climbing","hiking","skating","dancing","yoga","pilates","diving","kayaking"],
    "tool": ["scissors","knife","spoon","fork","spatula","whisk","tongs","ladle","grater","peeler","rolling_pin","strainer","thermometer","scale","timer","blender","toaster","kettle","fryer","mixer","grinder","sharpener","stapler","ruler","calculator","tape","pen","pencil","marker","eraser","notebook","calendar"],
    "instrument": ["piano","guitar","violin","drums","flute","trumpet","saxophone","clarinet","trombone","harp","cello","viola","oboe","bassoon","accordion","harmonica","ukulele","banjo","mandolin","sitar","marimba","xylophone","tuba","tambourine","bagpipes","didgeridoo","kazoo","ocarina","recorder","synthesizer","organ","keyboard"],
    "hobby": ["reading","writing","drawing","painting","sculpting","knitting","sewing","crochet","gardening","cooking","baking","fishing","hunting","camping","birdwatching","stargazing","photography","filmmaking","podcasting","blogging","vlogging","gaming","puzzling","origami","calligraphy","pottery","woodworking","metalworking","beekeeping","brewing","winemaking","collecting"],
}


def _flatten_closed_set():
    seen = set()
    out = []
    for cat, items in CLOSED_SET.items():
        for it in items:
            if it in seen:
                continue
            seen.add(it)
            out.append((cat, it))
    return out


CLOSED_PAIRS = _flatten_closed_set()
CLOSED_BY_CAT = defaultdict(list)
for _c, _i in CLOSED_PAIRS:
    CLOSED_BY_CAT[_c].append(_i)

NAME_RE = re.compile(r"^synthetic_persona_callback_\d+_([a-z]+)_(.+)_n\d+ev$")


def parse_name(name: str):
    m = NAME_RE.match(name)
    if not m:
        return None, None
    cat, item = m.group(1), m.group(2)
    if cat in CATEGORIES and item in CLOSED_BY_CAT[cat]:
        return cat, item
    # item may have contained "_" – ensure we pick a real item in that cat
    if cat in CATEGORIES:
        for it in CLOSED_BY_CAT[cat]:
            if it == item:
                return cat, it
    return None, None


def extract_pairs(blob):
    names = blob.get("chain_names", None)
    n = int(blob["chain_starts"].shape[0])
    pairs = []
    misses = 0
    for ci in range(n):
        cat, item = (None, None)
        if names is not None and ci < len(names):
            cat, item = parse_name(str(names[ci]))
        if cat is None:
            misses += 1
        pairs.append((cat, item))
    return pairs, misses


def build_prior(train_blob):
    pairs, misses = extract_pairs(train_blob)
    counts = defaultdict(lambda: Counter())
    n_chains_used = 0
    for cat, item in pairs:
        if cat is None:
            continue
        counts[cat][item] += 1
        n_chains_used += 1
    # Laplace with alpha=ALPHA over the full 32-item per-category support.
    prior = {}
    counts_dump = {}
    for cat in CATEGORIES:
        support = CLOSED_BY_CAT[cat]
        k = len(support)
        c = counts.get(cat, Counter())
        total = sum(c.values())
        denom = total + ALPHA * k
        p = {it: (c.get(it, 0) + ALPHA) / denom for it in support}
        prior[cat] = p
        counts_dump[cat] = {
            "n_train_chains": total,
            "support_size": k,
            "per_item_counts": {it: c.get(it, 0) for it in support},
        }
    return prior, counts_dump, n_chains_used, misses


def score_val(val_blob, prior, tokenizer):
    sess = val_blob["session_ids"]
    starts = val_blob["chain_starts"].tolist()
    cb_pos_all = val_blob["chain_callback_position"].tolist()
    cb_mask_all = val_blob["session_callback_mask"]
    names = val_blob.get("chain_names", None)
    n = min(N_VAL, len(starts))
    per_chain = []
    ce_list = []
    first_tok_ce = []
    n_multi = 0
    item_tok_lens = []
    for ci in range(n):
        st = starts[ci]
        cb_pos = cb_pos_all[ci]
        ids = sess[st + cb_pos]
        msk = cb_mask_all[st + cb_pos]
        n_tok = int(msk.sum().item())
        item_tok_lens.append(n_tok)
        cat, item = (None, None)
        if names is not None and ci < len(names):
            cat, item = parse_name(str(names[ci]))
        if cat is None:
            ce_list.append(float("nan"))
            continue
        p = prior[cat][item]
        ce_item = -math.log(p)  # nats
        # Distribute onto the n_tok mask positions – entire CE on 1st, 0 else.
        # Mean-over-mask equals ce_item / n_tok. (Matches how audit_base_prior
        # averages per-position.)
        if n_tok <= 0:
            ce_list.append(float("nan")); continue
        per_token_mean = ce_item / n_tok
        ce_list.append(per_token_mean)
        first_tok_ce.append(ce_item)
        if n_tok > 1:
            n_multi += 1
        per_chain.append({
            "chain_id": str(names[ci]) if names is not None else f"c{ci}",
            "callback_category": cat,
            "callback_item": item,
            "n_callback_tok": n_tok,
            "ce_template_prior_per_token": per_token_mean,
            "ce_template_prior_item": ce_item,
            "prior_prob": p,
        })
    import statistics as S
    good = [x for x in ce_list if not (isinstance(x, float) and math.isnan(x))]
    return {
        "n_chains_scored": len(good),
        "ce_template_prior_per_tok_mean": S.mean(good) if good else float("nan"),
        "ce_template_prior_item_mean": S.mean(first_tok_ce) if first_tok_ce else float("nan"),
        "avg_item_tok_len": S.mean(item_tok_lens) if item_tok_lens else float("nan"),
        "n_multi_tok_items": n_multi,
        "alpha": ALPHA,
        "item_split_choice": "entire CE on 1st callback-mask token; 0 on subsequent",
        "per_chain": per_chain,
    }


def main():
    train_path = ROOT / "paper_artifacts/chains/synthd4v2_persona_callback_train_s512.pt"
    val_path = ROOT / "paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt"
    print(f"Loading train {train_path} ...", flush=True)
    train = torch.load(train_path, weights_only=False, map_location="cpu")
    print(f"Loading val   {val_path} ...", flush=True)
    val = torch.load(val_path, weights_only=False, map_location="cpu")

    prior, counts_dump, n_train, misses_train = build_prior(train)
    print(f"Train pairs used: n={n_train}, name-parse-misses={misses_train}", flush=True)

    # Save prior pickle
    out_dir = ROOT / "results/exp2_chain_recipe"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "audit_a3_template_prior.pkl", "wb") as f:
        pickle.dump({
            "prior": prior,
            "counts": counts_dump,
            "categories": CATEGORIES,
            "alpha": ALPHA,
        }, f)

    tok_name = train.get("tokenizer", "Qwen/Qwen3-0.6B")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
    except Exception:
        tokenizer = None

    res = score_val(val, prior, tokenizer)
    res["train_counts"] = counts_dump
    res["train_name_parse_misses"] = misses_train
    res["n_train_chains_used"] = n_train
    res["tokenizer"] = tok_name
    res["train_corpus"] = str(train_path)
    res["val_corpus"] = str(val_path)

    out_json = out_dir / "audit_a3_template_prior_cloud.json"
    out_json.write_text(json.dumps(res, indent=2))
    print(f"Saved -> {out_json}")
    print(f"CE_template_prior (per-token mean over n={res['n_chains_scored']}): "
          f"{res['ce_template_prior_per_tok_mean']:.4f} nats")
    print(f"CE_template_prior (per-ITEM mean)                       : "
          f"{res['ce_template_prior_item_mean']:.4f} nats")
    print(f"avg_item_tok_len={res['avg_item_tok_len']:.2f}  "
          f"multi_tok_items={res['n_multi_tok_items']}/{res['n_chains_scored']}")
    print(f"log(32)={math.log(32):.4f}  log(256)={math.log(256):.4f}")


if __name__ == "__main__":
    main()
