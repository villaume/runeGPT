#!/usr/bin/env python3
"""Train a char-level GPT on runes, where every token IS a rune.

nanoGPT's shakespeare_char recipe, ported to MLX so it runs on the Mac GPU.
The corpus (data/runes.txt, from scripts/build_corpus.py) is ~320k chars of
Unicode Younger Futhark with a ~33-symbol vocabulary.

Usage:
    uv run python scripts/train.py                 # defaults, ~3000 iters
    uv run python scripts/train.py --iters 5000 --n-layer 6 --n-embd 384
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parent.parent


class Block(nn.Module):
    def __init__(self, dims: int, heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, heads)
        self.ln2 = nn.LayerNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, 4 * dims),
            nn.GELU(),
            nn.Linear(4 * dims, dims),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, mask))
        return x + self.mlp(self.ln2(x))


class RuneGPT(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = [
            Block(cfg["n_embd"], cfg["n_head"], cfg["dropout"])
            for _ in range(cfg["n_layer"])
        ]
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)

    def __call__(self, idx: mx.array) -> mx.array:
        L = idx.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(mx.arange(L)))
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))


def load_data(block_size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    text = (ROOT / "data" / "runes.txt").read_text()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    # split by inscription (line) so val isn't just one geographic region
    lines = text.splitlines()
    rng = np.random.default_rng(42)
    rng.shuffle(lines)
    n_val = max(1, len(lines) // 10)
    val_text = "\n".join(lines[:n_val]) + "\n"
    train_text = "\n".join(lines[n_val:]) + "\n"
    encode = lambda s: np.array([stoi[c] for c in s], dtype=np.int32)
    return encode(train_text), encode(val_text), vocab


def get_batch(data: np.ndarray, batch_size: int, block_size: int) -> tuple[mx.array, mx.array]:
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return mx.array(x), mx.array(y)


def loss_fn(model: RuneGPT, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean"
    )


def estimate_loss(model, data, batch_size, block_size, iters=40) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch_size, block_size)
        losses.append(loss_fn(model, x, y).item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    args = p.parse_args()

    train_data, val_data, vocab = load_data(args.block_size)
    cfg = {
        "vocab_size": len(vocab),
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
    }
    model = RuneGPT(cfg)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"vocab {len(vocab)} | train {len(train_data):,} val {len(val_data):,} chars | {n_params/1e6:.2f}M params")

    schedule = optim.join_schedules(
        [
            optim.linear_schedule(0.0, args.lr, args.warmup),
            optim.cosine_decay(args.lr, args.iters - args.warmup),
        ],
        [args.warmup],
    )
    opt = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    args.out.mkdir(exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    for it in range(1, args.iters + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size)
        loss, grads = loss_and_grad(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

        if it % args.eval_every == 0 or it == args.iters:
            val = estimate_loss(model, val_data, args.batch_size, args.block_size)
            marker = ""
            if val < best_val:
                best_val = val
                model.save_weights(str(args.out / "model.safetensors"))
                (args.out / "config.json").write_text(
                    json.dumps({"cfg": cfg, "vocab": vocab, "val_loss": val})
                )
                marker = " *"
            print(
                f"iter {it:5d} | train {loss.item():.4f} | val {val:.4f} | "
                f"{(time.time()-t0)/it*1000:.0f} ms/iter{marker}",
                flush=True,
            )

    print(f"done in {time.time()-t0:.0f}s | best val loss {best_val:.4f} -> {args.out}/")


if __name__ == "__main__":
    main()
