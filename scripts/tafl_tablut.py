"""Tablut (9x9) balance sweep — milestone 3: Linnaeus vs. Smith.

The only near-complete historical tafl account is Linnaeus's 1732 record of Sámi
*tablut*; the 1811 Smith translation muddied it. The two big contested axes are
**how the king escapes** (Linnaeus: to any edge) and **how the king is captured**
(four sides, or — on a weak reading — custodially like an ordinary man). This
sweep scores a small landscape across both axes and asks which reconstructions
land in the balanced, decisive region.

    uv run python scripts/tafl_tablut.py
    uv run python scripts/tafl_tablut.py --games 30 --depth 3
    uv run python scripts/tafl_tablut.py --sensitivity

Caveats (see tafl/DESIGN.md §7): depth-3 alpha-beta is a modest oracle and the
encircling attacker is the harder side to search, so attacker win-rates read low;
these are balance *estimates*, and the named reconstructions are typical readings,
not asserted as the historical truth.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tafl.balance import compare_strengths, run_balance
from tafl.rules import CornerRule, ThroneRule, TaflRules

# corners as Linnaeus's tablut treats them: ordinary squares (no corner escape,
# no corner capture) — used by the edge-escape variants.
_OPEN_CORNERS = CornerRule(soldiers_may_stop=True, hostile_to_attackers=False,
                           hostile_to_defenders=False, hostile_to_king=False)


def tablut(**over) -> TaflRules:
    base = dict(board_size=9, layout="tablut")
    base.update(over)
    return TaflRules(**base)


VARIANTS: dict[str, TaflRules] = {
    # the modern faithful-to-Linnaeus reading: edge escape, king taken on four sides
    "Linnaeus: edge + 4-side king": tablut(
        king_goal="any_edge", king_capture="four_sides",
        throne=ThroneRule(hostile_to_king=True), corners=_OPEN_CORNERS),
    # the Smith-1811 weak-king reading: edge escape, king captured like an ordinary man
    "Smith 1811: edge + weak king": tablut(
        king_goal="any_edge", king_capture="two_sides",
        throne=ThroneRule(), corners=_OPEN_CORNERS),
    # edge escape but the wall helps take the king (edge counts)
    "edge + edge-counts king": tablut(
        king_goal="any_edge", king_capture="edge_counts",
        throne=ThroneRule(hostile_to_king=True), corners=_OPEN_CORNERS),
    # the later 'fetlar-ised' reading: corner escape, strong king
    "corner + 4-side king": tablut(
        king_goal="corner", king_capture="four_sides",
        throne=ThroneRule(hostile_to_king=True), corners=CornerRule()),
    # corner escape with a weak king
    "corner + weak king": tablut(
        king_goal="corner", king_capture="two_sides",
        throne=ThroneRule(), corners=CornerRule()),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--opening", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=180)
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args()

    print(f"tablut (9x9) balance sweep — {args.games} games/variant, depth {args.depth}, "
          f"{args.opening} random opening plies\n")
    print(f"{'variant':32s}  result")
    print("-" * 96)
    t0 = time.time()
    for name, rules in VARIANTS.items():
        b = run_balance(rules, games=args.games, depth=args.depth,
                        opening_random_plies=args.opening, max_plies=args.max_plies)
        print(f"{name:32s}  {b}")
    print("-" * 96)
    print(f"({time.time() - t0:.1f}s)")

    if args.sensitivity:
        print("\nagent sensitivity (does the balance survive stronger play?)")
        print("-" * 96)
        for name, rules in VARIANTS.items():
            res = compare_strengths(rules, depths=(1, args.depth), games=args.games,
                                    opening_random_plies=args.opening, max_plies=args.max_plies)
            weak, strong = res[1], res[args.depth]
            swing = abs(weak.defender_winrate - strong.defender_winrate)
            print(f"{name:32s}  depth1 def {weak.defender_winrate:5.1%} -> "
                  f"depth{args.depth} def {strong.defender_winrate:5.1%}   swing {swing:5.1%}")


if __name__ == "__main__":
    main()
