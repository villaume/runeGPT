"""Train the AlphaZero-lite tafl net by self-play — milestone 4.

    uv run python scripts/tafl_az_train.py                       # brandub smoke run
    uv run python scripts/tafl_az_train.py --rounds 20 --games-per-round 24

Each round: play self-play games into a replay buffer, take gradient steps on
(policy=visit-distribution, value=game-outcome), then gauge strength by a score
(win + ½·draw) vs. the random and (shallow) alpha-beta agents. The default is a
*smoke* run: it validates the whole loop — the training loss falls steadily and
MCTS solves tactics — but a tiny net at low sim counts stays draw-heavy and is
NOT a strong player. Reaching a *trustworthy oracle* (to re-run the balance
sweeps) is a longer, larger run — bump --rounds/--games-per-round/--sims/--channels
and let it cook. That scaling-up is deliberately left as future work.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tafl.agents import AlphaBetaAgent, RandomAgent
from tafl.az.agent import AZAgent
from tafl.az.net import TaflNet
from tafl.az.selfplay import play_selfplay_game
from tafl.engine import Tafl
from tafl.rules import REFERENCE_RULES
from tafl.selfplay import play_game

ROOT = Path(__file__).resolve().parent.parent


def loss_fn(net, x, pi, z):
    logits, value = net(x)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    policy_loss = -(pi * logp).sum(axis=-1).mean()
    value_loss = ((value - z) ** 2).mean()
    return policy_loss + value_loss


def score_vs(net, game, opponent, *, az_sims, games):
    """Score of the AZ agent vs. ``opponent`` as (win + ½·draw)/games, split half
    as attacker / half as defender. Returns (score, win_frac, draw_frac)."""
    az = AZAgent(net, sims=az_sims)
    wins = draws = 0
    for i in range(games):
        py_rng = random.Random(1000 + i)
        az_side = "attackers" if i % 2 == 0 else "defenders"
        a, d = (az, opponent) if az_side == "attackers" else (opponent, az)
        r = play_game(game, a, d, rng=py_rng, max_plies=200)
        if r.winner == az_side:
            wins += 1
        elif r.winner == "draw":
            draws += 1
    return (wins + 0.5 * draws) / games, wins / games, draws / games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="brandub", choices=list(REFERENCE_RULES))
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--games-per-round", type=int, default=12)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--buffer", type=int, default=20000)
    ap.add_argument("--eval-games", type=int, default=8)
    ap.add_argument("--eval-sims", type=int, default=32)
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints-az")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    game = Tafl(REFERENCE_RULES[args.board]())
    net = TaflNet(game.n, channels=args.channels, blocks=args.blocks)
    mx.eval(net.parameters())
    from mlx.utils import tree_flatten
    nparams = sum(v.size for _, v in tree_flatten(net.parameters()))
    opt = optim.AdamW(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(net, loss_fn)
    buffer: deque = deque(maxlen=args.buffer)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(exist_ok=True)

    print(f"AZ training on {args.board} ({game.n}x{game.n}) | net {nparams/1e3:.0f}k params | "
          f"action space {net.A}\n")

    for rd in range(1, args.rounds + 1):
        t0 = time.time()
        results = {"attackers": 0, "defenders": 0, "draw": 0}
        for _ in range(args.games_per_round):
            samples, res = play_selfplay_game(net, game, sims=args.sims, rng=rng)
            buffer.extend(samples)
            results[res] += 1

        losses = []
        for _ in range(args.train_steps):
            if len(buffer) < args.batch_size:
                break
            batch = [buffer[i] for i in rng.integers(0, len(buffer), args.batch_size)]
            x = mx.array(np.stack([b[0] for b in batch]))
            pi = mx.array(np.stack([b[1] for b in batch]))
            z = mx.array(np.array([b[2] for b in batch], dtype=np.float32))
            loss, grads = loss_and_grad(net, x, pi, z)
            opt.update(net, grads)
            mx.eval(net.parameters(), opt.state)
            losses.append(loss.item())

        sc_rand, w_rand, d_rand = score_vs(net, game, RandomAgent(random.Random(rd)),
                                           az_sims=args.eval_sims, games=args.eval_games)
        sc_ab, _, _ = score_vs(net, game, AlphaBetaAgent(depth=1),
                               az_sims=args.eval_sims, games=args.eval_games)
        net.save_weights(str(args.out / f"{args.board}.safetensors"))
        print(f"round {rd:2d} | buffer {len(buffer):5d} | loss {np.mean(losses):.3f} | "
              f"vs random: score {sc_rand:4.0%} (win {w_rand:.0%}/draw {d_rand:.0%}) | "
              f"vs AB1 score {sc_ab:4.0%} | {time.time()-t0:.0f}s")

    print(f"\nweights -> {args.out / (args.board + '.safetensors')}")


if __name__ == "__main__":
    main()
