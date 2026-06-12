"""An MCTS+net player with the same ``choose(game, state)`` interface as the
alpha-beta and random agents, so it drops straight into the existing self-play /
balance harness (``tafl.selfplay.play_game``)."""

from __future__ import annotations

from ..engine import Move, State, Tafl
from .mcts import best_move, run_mcts


class AZAgent:
    def __init__(self, net, *, sims: int = 128, c_puct: float = 1.5):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct

    def choose(self, game: Tafl, state: State) -> Move:
        root = run_mcts(self.net, game, state, sims=self.sims,
                        c_puct=self.c_puct, dirichlet=0.0)
        return best_move(root)
