"""Generate self-play training samples (state, policy target, outcome).

Each move runs an MCTS search; the visit distribution becomes the policy target,
and the game's final outcome (from each recorded position's mover perspective)
becomes the value target. Early moves are sampled with temperature for opening
diversity; later moves go to the most-visited child. Dirichlet noise at the root
keeps self-play exploring.
"""

from __future__ import annotations

import numpy as np

from ..engine import Tafl
from .encoding import encode_state
from .mcts import policy_target, run_mcts, sample_move


def play_selfplay_game(
    net, game: Tafl, *, sims: int = 64, c_puct: float = 1.5,
    temp_moves: int = 12, dirichlet: float = 0.3, max_plies: int = 200,
    rng: np.random.Generator | None = None,
):
    """Play one game; return (samples, result). Each sample is
    (planes[n,n,C], pi[A], z) with z in {-1, 0, +1} from that mover's view."""
    rng = rng or np.random.default_rng()
    s = game.initial_state()
    recorded = []                                  # (planes, pi, to_move)
    result = "draw"
    for ply in range(max_plies):
        res = game.result(s)
        if res is not None:
            result = res
            break
        root = run_mcts(net, game, s, sims=sims, c_puct=c_puct,
                        dirichlet=dirichlet, rng=rng)
        recorded.append((encode_state(game, s), policy_target(root, game.n), s.to_move))
        move = sample_move(root, 1.0 if ply < temp_moves else 0.0, rng)
        s = game.apply(s, move)

    samples = []
    for planes, pi, to_move in recorded:
        z = 0.0 if result == "draw" else (1.0 if result == to_move else -1.0)
        samples.append((planes, pi, np.float32(z)))
    return samples, result
