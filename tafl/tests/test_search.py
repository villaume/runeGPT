"""Tests for the search agent and the balance harness (milestone 2).

Stdlib/assert-based like test_engine.py: run with
``uv run python -m tafl.tests.test_search``. Positions are tiny so the searches
are instant.
"""

from __future__ import annotations

import random

from tafl.agents import AlphaBetaAgent, RandomAgent
from tafl.agents.minimax import MATE_THRESHOLD, search_root
from tafl.balance import run_balance
from tafl.engine import State, Tafl
from tafl.rules import ATTACKER, ATTACKERS, DEFENDER, DEFENDERS, EMPTY, KING, TaflRules, brandub_7x7

_LAYOUT_BY_SIZE = {7: "brandub", 9: "tablut", 11: "fetlar"}


def make_game(size: int = 7, **overrides) -> Tafl:
    return Tafl(TaflRules(board_size=size, layout=_LAYOUT_BY_SIZE[size], **overrides))


def board_from(game: Tafl, pieces: dict[tuple[int, int], int]) -> tuple[int, ...]:
    b = [EMPTY] * (game.n * game.n)
    for (r, c), p in pieces.items():
        b[game.idx(r, c)] = p
    return tuple(b)


def test_agent_takes_the_winning_king_escape():
    g = make_game(7, king_goal="corner", king_capture="four_sides")
    board = board_from(g, {(0, 1): KING, (3, 3): ATTACKER, (6, 5): ATTACKER})
    s = State(board, DEFENDERS)
    move = AlphaBetaAgent(depth=2).choose(g, s)
    ns = g.apply(s, move)
    assert g.result(ns) == DEFENDERS                     # it found the corner run


def test_agent_captures_a_king_it_can_take():
    g = make_game(7, king_capture="four_sides")
    board = board_from(g, {
        (3, 3): KING, (2, 3): ATTACKER, (4, 3): ATTACKER, (3, 2): ATTACKER, (3, 6): ATTACKER,
    })
    s = State(board, ATTACKERS)
    move = AlphaBetaAgent(depth=2).choose(g, s)
    ns = g.apply(s, move)
    assert g.king_square(ns.board) is None               # closed the box


def test_search_reports_a_mate_value_for_the_winning_side():
    g = make_game(7, king_capture="four_sides")
    board = board_from(g, {
        (3, 3): KING, (2, 3): ATTACKER, (4, 3): ATTACKER, (3, 2): ATTACKER, (3, 6): ATTACKER,
    })
    s = State(board, ATTACKERS)
    _move, value, _nodes = search_root(g, s, depth=2)
    assert value >= MATE_THRESHOLD


def test_agent_prefers_a_faster_mate():
    # king one step from a corner; the agent must not dither (which would risk a draw)
    g = make_game(7, king_goal="corner")
    board = board_from(g, {(1, 0): KING, (3, 3): ATTACKER, (5, 5): ATTACKER})
    s = State(board, DEFENDERS)
    _move, value, _ = search_root(g, s, depth=4)
    assert value >= MATE_THRESHOLD                       # sees the forced escape


def test_random_agents_game_terminates():
    g = Tafl(brandub_7x7())
    from tafl.selfplay import play_game
    r = play_game(g, RandomAgent(random.Random(1)), RandomAgent(random.Random(2)),
                  rng=random.Random(3), max_plies=400)
    assert r.winner in (ATTACKERS, DEFENDERS, "draw")
    assert r.plies >= 1


def test_balance_rates_are_a_distribution():
    b = run_balance(brandub_7x7(), games=4, depth=1, opening_random_plies=2, max_plies=80, seed=0)
    total = b.attacker_winrate + b.defender_winrate + b.draw_rate
    assert abs(total - 1.0) < 1e-9
    assert 0.0 <= b.win_balance <= 1.0
    assert b.mean_plies >= 1


# --- manual runner ----------------------------------------------------------
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e or 'assertion failed'}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
