"""13x13 great hnefatafl balance sweep — extending the study to the big board.

The 13x13 setup is itself contested (24+12+1 here, the common 'great cross'
reading; a denser 32+16+1 also circulates). On top of that we sweep the same
contested mechanics as the smaller boards — king escape (corner vs edge) and king
capture (four-sides / weak / edge-counts) — to see whether a larger board, with
more room for the king to run, shifts where the balanced region sits.

    uv run python scripts/tafl_hnefatafl13.py
    uv run python scripts/tafl_hnefatafl13.py --games 12 --depth 3

Heavy caveat (see tafl/DESIGN.md §7, and louder here): depth-3 alpha-beta on a
13x13 board is a *weak* oracle — coordinating 24 attackers into an encirclement
is far beyond a 3-ply horizon — so defender win-rates are very soft upper bounds.
This sweep maps the qualitative landscape; the milestone-4 net is needed to trust
13x13 numbers at all.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tafl.balance import run_balance
from tafl.rules import CornerRule, ThroneRule, TaflRules

_OPEN_CORNERS = CornerRule(soldiers_may_stop=True, hostile_to_attackers=False,
                           hostile_to_defenders=False, hostile_to_king=False)
_HOSTILE_THRONE = ThroneRule(hostile_to_king=True, hostile_to_attackers=True,
                             hostile_to_defenders=True)


def hnef(**over) -> TaflRules:
    base = dict(board_size=13, layout="hnefatafl13")
    base.update(over)
    return TaflRules(**base)


VARIANTS: dict[str, TaflRules] = {
    "corner + 4-side king (ref)": hnef(king_goal="corner", king_capture="four_sides",
                                       throne=_HOSTILE_THRONE, corners=CornerRule(),
                                       shieldwall=True),
    "corner + weak king":         hnef(king_goal="corner", king_capture="two_sides",
                                       throne=ThroneRule(), corners=CornerRule(),
                                       shieldwall=True),
    "corner + edge-counts king":  hnef(king_goal="corner", king_capture="edge_counts",
                                       throne=_HOSTILE_THRONE, corners=CornerRule(),
                                       shieldwall=True),
    "edge escape + 4-side king":  hnef(king_goal="any_edge", king_capture="four_sides",
                                       throne=_HOSTILE_THRONE, corners=_OPEN_CORNERS),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--opening", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=130)
    args = ap.parse_args()

    print(f"hnefatafl 13x13 balance sweep — {args.games} games/variant, depth {args.depth}, "
          f"{args.opening} random opening plies\n")
    print(f"{'variant':30s}  result")
    print("-" * 96)
    t0 = time.time()
    for name, rules in VARIANTS.items():
        b = run_balance(rules, games=args.games, depth=args.depth,
                        opening_random_plies=args.opening, max_plies=args.max_plies)
        print(f"{name:30s}  {b}")
    print("-" * 96)
    print(f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
