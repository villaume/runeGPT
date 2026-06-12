"""Balance sweep over brandub (7x7) rule variants — milestone 2.

Plays alpha-beta vs. alpha-beta self-play games (randomised openings) for each
candidate rule set and prints a balance table. The thesis: the historical rules
sit in the balanced, decisive region; a variant one side wins almost always is
probably not what was played.

    uv run python scripts/tafl_balance.py                  # default sweep
    uv run python scripts/tafl_balance.py --games 40 --depth 3
    uv run python scripts/tafl_balance.py --sensitivity    # weak vs strong play

brandub is small enough that a few plies of alpha-beta is a serious player, which
is why it anchors ground truth before any neural net (see tafl/DESIGN.md).
"""

from __future__ import annotations

import argparse
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tafl.balance import compare_strengths, run_balance
from tafl.rules import CornerRule, ThroneRule, TaflRules


def brandub(**over) -> TaflRules:
    base = dict(board_size=7, layout="brandub", king_goal="corner")
    base.update(over)
    return TaflRules(**base)


# Candidate rule sets: each toggles one contested mechanic against a baseline.
VARIANTS: dict[str, TaflRules] = {
    "four_sides + hostile throne":   brandub(king_capture="four_sides",
                                             throne=ThroneRule(hostile_to_king=True)),
    "four_sides + plain throne":     brandub(king_capture="four_sides",
                                             throne=ThroneRule()),
    "edge_counts (wall captures)":   brandub(king_capture="edge_counts",
                                             throne=ThroneRule(hostile_to_king=True)),
    "two_sides (king like a man)":   brandub(king_capture="two_sides",
                                             throne=ThroneRule()),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--opening", type=int, default=4, help="random opening plies")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--sensitivity", action="store_true",
                    help="also report weak (depth 1) vs strong (--depth) play per variant")
    args = ap.parse_args()

    print(f"brandub balance sweep — {args.games} games/variant, depth {args.depth}, "
          f"{args.opening} random opening plies\n")
    header = f"{'variant':30s}  {'result':>0}"
    print(header)
    print("-" * 92)
    t0 = time.time()
    for name, rules in VARIANTS.items():
        b = run_balance(rules, games=args.games, depth=args.depth,
                        opening_random_plies=args.opening, max_plies=args.max_plies)
        print(f"{name:30s}  {b}")
    print("-" * 92)
    print(f"({time.time() - t0:.1f}s)")

    if args.sensitivity:
        print("\nagent sensitivity (does the balance survive stronger play?)")
        print("-" * 92)
        for name, rules in VARIANTS.items():
            res = compare_strengths(rules, depths=(1, args.depth), games=args.games,
                                    opening_random_plies=args.opening, max_plies=args.max_plies)
            weak, strong = res[1], res[args.depth]
            swing = abs(weak.defender_winrate - strong.defender_winrate)
            print(f"{name:30s}  depth1 def {weak.defender_winrate:5.1%} -> "
                  f"depth{args.depth} def {strong.defender_winrate:5.1%}   swing {swing:5.1%}")


if __name__ == "__main__":
    main()
