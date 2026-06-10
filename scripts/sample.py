#!/usr/bin/env python3
"""Sample fake runestone inscriptions from a trained runeGPT checkpoint.

Usage:
    uv run python scripts/sample.py                       # 5 inscriptions
    uv run python scripts/sample.py -n 10 --temperature 0.9
    uv run python scripts/sample.py --prompt "ᚴᚢᚦᚱᚢᚾ ᛬ ᛚᛁᛏ"
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import RuneGPT

ROOT = Path(__file__).resolve().parent.parent


def generate(model, cfg, prompt_ids, max_new, temperature, top_k, eot):
    ids = list(prompt_ids)
    for _ in range(max_new):
        ctx = mx.array([ids[-cfg["block_size"] :]])
        logits = np.array(model(ctx)[0, -1]) / temperature
        if top_k:
            cutoff = np.sort(logits)[-top_k]
            logits[logits < cutoff] = -np.inf
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        nxt = int(np.random.choice(len(probs), p=probs))
        if nxt == eot:
            break
        ids.append(nxt)
    return ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--num", type=int, default=5)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    meta = json.loads((args.checkpoint / "config.json").read_text())
    cfg, vocab = meta["cfg"], meta["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    model = RuneGPT(cfg)
    model.load_weights(str(args.checkpoint / "model.safetensors"))
    model.eval()

    eot = stoi["\n"]  # newline = end of inscription
    prompt_ids = [eot] + [stoi[c] for c in args.prompt]
    for _ in range(args.num):
        ids = generate(model, cfg, prompt_ids, args.max_new, args.temperature, args.top_k, eot)
        print("".join(vocab[i] for i in ids[1:]))
        print()


if __name__ == "__main__":
    main()
