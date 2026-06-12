"""Aggregate self-play games into the balance metrics from DESIGN section 4.

The hypothesis: the historical rule set lives in the high-balance, decisive,
non-trivial region. This module turns a batch of games into the numbers that
locate a rule set in that space. ``agent_sensitivity`` (does the result hold up
under stronger play?) is computed by ``compare_strengths`` rather than stored on
a single batch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .agents import AlphaBetaAgent
from .engine import Tafl
from .rules import ATTACKERS, DEFENDERS, TaflRules
from .selfplay import play_game


@dataclass
class Balance:
    games: int
    attacker_winrate: float
    defender_winrate: float
    draw_rate: float
    mean_plies: float
    win_balance: float        # |attacker - defender|; 0 = perfectly even
    decisiveness: float       # 1 - draw_rate

    def __str__(self) -> str:
        return (f"atk {self.attacker_winrate:5.1%}  def {self.defender_winrate:5.1%}  "
                f"draw {self.draw_rate:5.1%}  |imbalance {self.win_balance:5.1%}|  "
                f"decisive {self.decisiveness:5.1%}  len {self.mean_plies:5.1f}")


def run_balance(
    rules: TaflRules,
    *,
    games: int = 24,
    depth: int = 3,
    opening_random_plies: int = 4,
    max_plies: int = 300,
    seed: int = 0,
) -> Balance:
    game = Tafl(rules)
    atk = AlphaBetaAgent(depth)
    dfn = AlphaBetaAgent(depth)
    wins = {ATTACKERS: 0, DEFENDERS: 0, "draw": 0}
    total_plies = 0
    for i in range(games):
        rng = random.Random(seed * 100_000 + i)
        r = play_game(game, atk, dfn, rng=rng,
                      opening_random_plies=opening_random_plies, max_plies=max_plies)
        wins[r.winner] += 1
        total_plies += r.plies
    aw = wins[ATTACKERS] / games
    dw = wins[DEFENDERS] / games
    dr = wins["draw"] / games
    return Balance(
        games=games,
        attacker_winrate=aw,
        defender_winrate=dw,
        draw_rate=dr,
        mean_plies=total_plies / games,
        win_balance=abs(aw - dw),
        decisiveness=1 - dr,
    )


def sweep(variants: dict[str, TaflRules], **kw) -> dict[str, Balance]:
    """Run ``run_balance`` over a named set of rule sets. Keyword args are passed
    straight through (games, depth, opening_random_plies, max_plies, seed)."""
    return {name: run_balance(rules, **kw) for name, rules in variants.items()}


def compare_strengths(rules: TaflRules, depths=(1, 3), **kw) -> dict[int, Balance]:
    """Run the same rule set at several search depths. If the win-rate swings a lot
    between weak and strong play, the rule set is only "balanced" for weak players --
    a red flag, not a real equilibrium (DESIGN section 4, agent_sensitivity)."""
    return {d: run_balance(rules, depth=d, **kw) for d in depths}
