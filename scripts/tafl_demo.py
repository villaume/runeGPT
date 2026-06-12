"""Play a random game of a named tafl rule set and print the board.

    uv run python scripts/tafl_demo.py                 # brandub, random rollout
    uv run python scripts/tafl_demo.py fetlar --seed 7
    uv run python scripts/tafl_demo.py --list

Milestone 1 has no real agents yet, so both sides move at random -- this is just
a sanity window onto the engine and the renderer. The balance search (later
milestones) replaces ``random`` with MCTS / minimax players.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tafl.engine import Tafl
from tafl.rules import REFERENCE_RULES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ruleset", nargs="?", default="brandub", choices=list(REFERENCE_RULES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--list", action="store_true", help="list reference rule sets and exit")
    args = ap.parse_args()

    if args.list:
        for name, builder in REFERENCE_RULES.items():
            r = builder()
            print(f"{name:18s} {r.board_size}x{r.board_size}  "
                  f"king_capture={r.king_capture}  goal={r.king_goal}")
        return

    g = Tafl(REFERENCE_RULES[args.ruleset]())
    rng = random.Random(args.seed)
    s = g.initial_state()
    print(f"=== {args.ruleset} ({g.n}x{g.n}) — random rollout, seed {args.seed} ===\n")
    print(g.render(s), "\n")

    ply = 0
    while ply < args.max_plies:
        res = g.result(s)
        if res is not None:
            break
        s = g.apply(s, rng.choice(g.legal_moves(s)))
        ply += 1

    res = g.result(s) or "draw (ply limit)"
    print(g.render(s))
    print(f"\nresult after {ply} plies: {res}")


if __name__ == "__main__":
    main()
