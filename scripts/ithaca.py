#!/usr/bin/env python3
"""Ithaca-for-runes: a bidirectional multitask transformer over younger futhark.

DeepMind's Ithaca restores damaged ancient Greek inscriptions and attributes
them in space and time. The same three tasks map cleanly onto the Rundata
corpus — and unlike a causal LM, restoration needs to see runes on *both* sides
of a gap, so this is a bidirectional encoder with three heads:

  1. restoration  — masked-rune modelling (BERT-style). Self-supervised, so the
                    training signal is unbounded; at inference the real lacunae
                    (Rundata '-' / '...') become [MASK] and the model fills them.
  2. period       — coarse dating: U (pre-Viking) / V (Viking) / M (medieval),
                    from the Rundata dating prefix. 100% label coverage.
  3. region       — geographic attribution from the signature prefix
                    (U=Uppland, Ög=Östergötland, DR=Denmark, N=Norway, ...),
                    11 classes + 'other'. ~94% coverage, essentially free labels.

The restoration head is also the scoring engine for ranking competing readings
of contested strings (see scripts/rank_readings.py) — the original runeGPT goal.

Usage:
    uv run python scripts/ithaca.py                  # train, ~few min on Apple Silicon
    uv run python scripts/ithaca.py --iters 4000
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parent.parent
PAD, MASK = 0, 1                      # reserved special-token ids
PERIODS = ["U", "V", "M"]            # pre-Viking / Viking / medieval
REGION_MIN = 50                      # signature prefixes rarer than this -> 'other'


def region_of(signature: str) -> str:
    m = re.match(r"([A-ZÖÄÅa-zäöå]+)", signature or "")
    return m.group(1) if m else "?"


def period_of(dating: str) -> str | None:
    d = (dating or "").strip()
    return d[0] if d and d[0] in PERIODS else None


def load(block_size: int):
    rows = [json.loads(l) for l in open(ROOT / "data" / "corpus.jsonl")]
    rows = [r for r in rows if r["runes"]]

    # vocab: special tokens first, then the rune characters
    chars = sorted({c for r in rows for c in r["runes"]})
    itos = ["[PAD]", "[MASK]"] + chars
    stoi = {c: i for i, c in enumerate(itos)}

    # region label set: prefixes with >= REGION_MIN inscriptions, else 'other'
    counts = Counter(region_of(r["signature"]) for r in rows)
    regions = sorted([k for k, v in counts.items() if v >= REGION_MIN]) + ["other"]
    ridx = {r: i for i, r in enumerate(regions)}

    samples = []
    for r in rows:
        ids = [stoi[c] for c in r["runes"]][:block_size]
        if not ids:
            continue
        reg = region_of(r["signature"])
        samples.append(
            {
                "ids": ids,
                "period": PERIODS.index(period_of(r["dating"])) if period_of(r["dating"]) else -1,
                "region": ridx.get(reg, ridx["other"]),
                "sig": r["signature"],
            }
        )

    rng = np.random.default_rng(42)
    rng.shuffle(samples)
    n_val = max(1, len(samples) // 10)
    return samples[n_val:], samples[:n_val], itos, regions


def batch(samples, bs, block_size, vocab_size, rng):
    pick = rng.integers(0, len(samples), size=bs)
    x = np.zeros((bs, block_size), dtype=np.int32)
    pad_mask = np.zeros((bs, block_size), dtype=np.float32)  # 1 where real token
    tgt = np.full((bs, block_size), -1, dtype=np.int32)      # restoration targets
    period = np.array([samples[i]["period"] for i in pick], dtype=np.int32)
    region = np.array([samples[i]["region"] for i in pick], dtype=np.int32)
    for b, i in enumerate(pick):
        ids = samples[i]["ids"]
        n = len(ids)
        x[b, :n] = ids
        pad_mask[b, :n] = 1.0
        # BERT-style masking on 15% of real positions
        k = max(1, int(round(0.15 * n)))
        pos = rng.choice(n, size=k, replace=False)
        for p in pos:
            tgt[b, p] = ids[p]
            roll = rng.random()
            if roll < 0.8:
                x[b, p] = MASK
            elif roll < 0.9:
                x[b, p] = rng.integers(2, vocab_size)  # random rune
            # else: keep original (10%)
    return mx.array(x), mx.array(pad_mask), mx.array(tgt), mx.array(period), mx.array(region)


class Block(nn.Module):
    def __init__(self, dims, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, heads)
        self.ln2 = nn.LayerNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, 4 * dims), nn.GELU(), nn.Linear(4 * dims, dims), nn.Dropout(dropout)
        )
        self.drop = nn.Dropout(dropout)

    def __call__(self, x, mask):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, mask))      # bidirectional: padding mask only
        return x + self.mlp(self.ln2(x))


class IthacaRunes(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.pos = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = [Block(cfg["n_embd"], cfg["n_head"], cfg["dropout"]) for _ in range(cfg["n_layer"])]
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.restore = nn.Linear(cfg["n_embd"], cfg["vocab_size"])  # masked-rune head
        self.period = nn.Linear(cfg["n_embd"], len(PERIODS))         # dating head
        self.region = nn.Linear(cfg["n_embd"], cfg["n_region"])      # geo head

    def torso(self, idx, pad_mask):
        L = idx.shape[1]
        # additive key-padding mask (B,1,1,L): -inf where PAD, broadcast over heads/queries
        add = mx.where(pad_mask[:, None, None, :] > 0, 0.0, -1e9)
        x = self.drop(self.tok(idx) + self.pos(mx.arange(L)))
        for blk in self.blocks:
            x = blk(x, add)
        return self.ln_f(x)

    def __call__(self, idx, pad_mask):
        h = self.torso(idx, pad_mask)
        # mean-pool over real tokens for the sequence-level heads
        denom = mx.maximum(pad_mask.sum(axis=1, keepdims=True), 1.0)
        pooled = (h * pad_mask[:, :, None]).sum(axis=1) / denom
        return self.restore(h), self.period(pooled), self.region(pooled)


def _masked_ce(logits, targets, valid):
    # MLX lacks boolean indexing: clamp invalid targets, then zero their loss.
    safe = mx.maximum(targets, 0)
    ce = nn.losses.cross_entropy(logits, safe, reduction="none")
    denom = mx.maximum(valid.sum(), 1.0)
    return (ce * valid).sum() / denom


def loss_fn(model, x, pad_mask, tgt, period, region):
    rlogits, plogits, glogits = model(x, pad_mask)
    V = rlogits.shape[-1]
    flat_t = tgt.reshape(-1)
    r_ce = _masked_ce(rlogits.reshape(-1, V), flat_t, (flat_t >= 0).astype(mx.float32))
    p_ce = _masked_ce(plogits, period, (period >= 0).astype(mx.float32))
    g_ce = nn.losses.cross_entropy(glogits, region, reduction="mean")
    return r_ce + 0.5 * p_ce + 0.5 * g_ce, (r_ce, p_ce, g_ce)


def evaluate(model, val, bs, block_size, V, rng, iters=30):
    model.eval()
    racc = pacc = gacc = 0.0
    for _ in range(iters):
        x, pm, tgt, per, reg = batch(val, bs, block_size, V, rng)
        rl, pl, gl = model(x, pm)
        ft = tgt.reshape(-1)
        rv = (ft >= 0).astype(mx.float32)
        rhit = (rl.reshape(-1, V).argmax(-1) == mx.maximum(ft, 0)).astype(mx.float32)
        racc += (rhit * rv).sum().item() / max(rv.sum().item(), 1)
        pv = (per >= 0).astype(mx.float32)
        phit = (pl.argmax(-1) == mx.maximum(per, 0)).astype(mx.float32)
        pacc += (phit * pv).sum().item() / max(pv.sum().item(), 1)
        gacc += (gl.argmax(-1) == reg).astype(mx.float32).mean().item()
    model.train()
    return racc / iters, pacc / iters, gacc / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints-ithaca")
    args = p.parse_args()

    train, val, itos, regions = load(args.block_size)
    cfg = {
        "vocab_size": len(itos), "block_size": args.block_size,
        "n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
        "dropout": args.dropout, "n_region": len(regions),
    }
    model = IthacaRunes(cfg)
    mx.eval(model.parameters())
    n = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"vocab {len(itos)} | regions {len(regions)} | train {len(train)} val {len(val)} | {n/1e6:.2f}M params")

    sched = optim.cosine_decay(args.lr, args.iters)
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.1)
    lg = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(0)
    V = len(itos)

    args.out.mkdir(exist_ok=True)
    best = -1.0
    t0 = time.time()
    for it in range(1, args.iters + 1):
        x, pm, tgt, per, reg = batch(train, args.batch_size, args.block_size, V, rng)
        (loss, parts), grads = lg(model, x, pm, tgt, per, reg)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if it % args.eval_every == 0 or it == args.iters:
            racc, pacc, gacc = evaluate(model, val, args.batch_size, args.block_size, V, rng)
            r, pp, g = (c.item() for c in parts)
            score = racc + pacc + gacc
            star = ""
            if score > best:
                best = score
                model.save_weights(str(args.out / "model.safetensors"))
                (args.out / "config.json").write_text(json.dumps({"cfg": cfg, "itos": itos, "regions": regions}))
                star = " *"
            print(f"iter {it:5d} | loss {loss.item():.3f} (r{r:.2f} p{pp:.2f} g{g:.2f}) | "
                  f"val restore {racc:.1%} period {pacc:.1%} region {gacc:.1%}{star}", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {args.out}/")


if __name__ == "__main__":
    main()
