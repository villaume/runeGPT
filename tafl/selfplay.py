"""Play complete games between two agents and report the outcome.

Tafl under two deterministic agents yields exactly one game, so to get a *balance
distribution* we diversify the opening: the first few plies are random (shared by
both sides), then the agents take over. Sweeping the RNG seed then samples a
spread of games from one rule set -- the standard way to wring a win-rate out of
deterministic engines.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .engine import State, Tafl
from .rules import ATTACKERS, DEFENDERS


@dataclass
class GameResult:
    winner: str          # ATTACKERS, DEFENDERS, or "draw"
    plies: int
    reason: str          # "king_escaped" / "king_captured" / "no_moves" / "repetition" / "ply_limit"


def _has_agent(a) -> bool:
    return a is not None and hasattr(a, "choose")


def play_game(
    game: Tafl,
    attacker,
    defender,
    *,
    rng: random.Random,
    opening_random_plies: int = 0,
    max_plies: int | None = None,
) -> GameResult:
    max_plies = max_plies if max_plies is not None else game.rules.max_plies
    state = game.initial_state()
    seen: Counter[State] = Counter()

    for ply in range(max_plies):
        res = game.result(state)
        if res is not None:
            return GameResult(res, ply, _reason_for(game, state, res))

        seen[state] += 1
        if seen[state] >= 3:                      # threefold repetition
            return GameResult(_repetition_winner(game), ply, "repetition")

        agent = attacker if state.to_move == ATTACKERS else defender
        if ply < opening_random_plies or not _has_agent(agent):
            move = rng.choice(game.legal_moves(state))
        else:
            move = agent.choose(game, state)
        state = game.apply(state, move)

    return GameResult("draw", max_plies, "ply_limit")


def _reason_for(game: Tafl, state: State, winner: str) -> str:
    if game.king_square(state.board) is None:
        return "king_captured"
    kp = game.king_square(state.board)
    if kp and game._king_on_goal(*kp):
        return "king_escaped"
    return "no_moves"


def _repetition_winner(game: Tafl) -> str:
    mode = game.rules.repetition
    if mode == "loss_attacker":
        return DEFENDERS
    if mode == "loss_mover":
        return DEFENDERS                          # attackers are the side that perpetuates; convention
    return "draw"
