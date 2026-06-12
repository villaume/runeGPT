"""Engine-correctness tests for the generic tafl engine.

Pure stdlib (assert-based) so it runs with ``uv run python -m tafl.tests.test_engine``
with no extra deps, and it is also pytest-discoverable (``test_*`` functions) if
pytest is later added.

Each test builds an explicit position and checks one mechanic. This is the
milestone-1 contract: the *engine* is correct. Reproducing published *win-rates*
needs the search agents (milestone 3) and is out of scope here.
"""

from __future__ import annotations

import random

from tafl.engine import State, Tafl
from tafl.rules import (
    ATTACKER,
    ATTACKERS,
    CornerRule,
    DEFENDER,
    DEFENDERS,
    EMPTY,
    KING,
    ThroneRule,
    TaflRules,
    brandub_7x7,
    fetlar_11x11,
    tablut_linnaeus_9x9,
)

_LAYOUT_BY_SIZE = {7: "brandub", 9: "tablut", 11: "fetlar"}


def make_game(size: int = 7, **overrides) -> Tafl:
    """A Tafl on a board of ``size`` with rule overrides; used for hand-built
    positions (we never call ``initial_state`` on these)."""
    rules = TaflRules(board_size=size, layout=_LAYOUT_BY_SIZE[size], **overrides)
    return Tafl(rules)


def board_from(game: Tafl, pieces: dict[tuple[int, int], int]) -> tuple[int, ...]:
    b = [EMPTY] * (game.n * game.n)
    for (r, c), p in pieces.items():
        b[game.idx(r, c)] = p
    return tuple(b)


def count(board: tuple[int, ...], piece: int) -> int:
    return sum(1 for x in board if x == piece)


# --- setup ------------------------------------------------------------------
def test_initial_positions_have_expected_piece_counts():
    for builder, atk, dfn in [
        (brandub_7x7, 8, 4),
        (tablut_linnaeus_9x9, 16, 8),
        (fetlar_11x11, 24, 12),
    ]:
        g = Tafl(builder())
        s = g.initial_state()
        assert count(s.board, ATTACKER) == atk
        assert count(s.board, DEFENDER) == dfn
        assert count(s.board, KING) == 1
        assert s.to_move == ATTACKERS                      # attackers move first


# --- movement ---------------------------------------------------------------
def test_rook_slides_and_is_blocked_by_pieces():
    g = make_game(7)
    board = board_from(g, {(3, 1): ATTACKER, (3, 4): DEFENDER, (5, 5): KING})
    s = State(board, ATTACKERS)
    moves = {m for m in g.legal_moves(s) if m[:2] == (3, 1)}
    dests = {(r, c) for _, _, r, c in moves}
    # rightward it may reach (3,2),(3,3-blocked? throne) ... must stop before the defender at (3,4)
    assert (3, 2) in dests
    assert (3, 4) not in dests                              # cannot land on a piece
    assert (3, 5) not in dests                              # cannot jump over the defender


def test_soldier_cannot_stop_on_corner_or_throne_but_passes_throne():
    g = make_game(7)
    board = board_from(g, {(3, 5): ATTACKER, (0, 3): ATTACKER, (6, 6): KING})
    s = State(board, ATTACKERS)
    dests = {(r, c) for _, _, r, c in g.legal_moves(s)}
    assert (0, 0) not in dests                              # corner restricted to king
    assert (3, 3) not in dests                              # throne restricted to king
    assert (3, 2) in dests                                  # but the soldier passes through the empty throne


# --- captures ---------------------------------------------------------------
def test_basic_custodial_capture():
    g = make_game(7)
    board = board_from(g, {(0, 2): ATTACKER, (3, 4): ATTACKER, (3, 3): DEFENDER, (5, 5): KING})
    s = State(board, ATTACKERS)
    ns = g.apply(s, (0, 2, 3, 2))                           # A . D A  -> defender pinned
    assert ns.board[g.idx(3, 3)] == EMPTY
    assert ns.board[g.idx(3, 2)] == ATTACKER


def test_moving_between_two_enemies_is_safe():
    g = make_game(7)
    board = board_from(g, {(3, 2): ATTACKER, (3, 4): ATTACKER, (0, 3): DEFENDER, (5, 5): KING})
    s = State(board, DEFENDERS)
    ns = g.apply(s, (0, 3, 3, 3))                           # defender steps between two attackers
    assert ns.board[g.idx(3, 3)] == DEFENDER               # not suicide
    assert ns.board[g.idx(3, 2)] == ATTACKER
    assert ns.board[g.idx(3, 4)] == ATTACKER


def test_double_capture_in_one_move():
    g = make_game(7)
    board = board_from(g, {
        (1, 4): ATTACKER, (3, 2): ATTACKER, (3, 6): ATTACKER,   # anchors + mover
        (2, 4): DEFENDER, (3, 3): DEFENDER, (5, 5): KING,
    })
    s = State(board, ATTACKERS)
    ns = g.apply(s, (3, 6, 3, 4))                           # lands beside both defenders
    assert ns.board[g.idx(2, 4)] == EMPTY
    assert ns.board[g.idx(3, 3)] == EMPTY
    assert count(ns.board, DEFENDER) == 0


def test_corner_acts_as_capture_anvil():
    g = make_game(7)                                        # corners hostile_to_defenders by default
    board = board_from(g, {(2, 2): ATTACKER, (0, 1): DEFENDER, (5, 5): KING})
    s = State(board, ATTACKERS)
    ns = g.apply(s, (2, 2, 0, 2))                           # A . D corner  -> defender pinned against (0,0)
    assert ns.board[g.idx(0, 1)] == EMPTY


# --- king escape ------------------------------------------------------------
def test_king_escapes_to_corner():
    g = make_game(7, king_goal="corner")
    board = board_from(g, {(0, 1): KING, (6, 6): ATTACKER})
    s = State(board, DEFENDERS)
    assert (0, 1, 0, 0) in g.legal_moves(s)
    ns = g.apply(s, (0, 1, 0, 0))
    assert g.result(ns) == DEFENDERS


def test_king_escapes_to_edge_when_goal_is_any_edge():
    g = make_game(9, king_goal="any_edge")
    board = board_from(g, {(4, 1): KING, (0, 0): ATTACKER})
    s = State(board, DEFENDERS)
    ns = g.apply(s, (4, 1, 4, 0))
    assert g.result(ns) == DEFENDERS


# --- king capture -----------------------------------------------------------
def test_king_captured_on_four_sides():
    g = make_game(7, king_capture="four_sides")
    board = board_from(g, {
        (3, 3): KING, (2, 3): ATTACKER, (4, 3): ATTACKER, (3, 2): ATTACKER, (3, 6): ATTACKER,
    })
    s = State(board, ATTACKERS)
    ns = g.apply(s, (3, 6, 3, 4))                           # closes the fourth side
    assert g.king_square(ns.board) is None
    assert g.result(ns) == ATTACKERS


def test_king_captured_on_two_sides_in_that_mode():
    g = make_game(7, king_capture="two_sides")
    board = board_from(g, {(3, 3): KING, (3, 2): ATTACKER, (3, 6): ATTACKER})
    s = State(board, ATTACKERS)
    ns = g.apply(s, (3, 6, 3, 4))                           # pins king horizontally
    assert g.king_square(ns.board) is None


def test_edge_counts_mode_takes_king_on_the_wall():
    pieces = {(0, 3): KING, (0, 2): ATTACKER, (0, 4): ATTACKER, (3, 3): ATTACKER}
    move = (3, 3, 1, 3)                                     # complete the third real side

    g_edge = make_game(7, king_capture="edge_counts")
    ns = g_edge.apply(State(board_from(g_edge, pieces), ATTACKERS), move)
    assert g_edge.king_square(ns.board) is None             # off-board wall counts -> captured

    g_four = make_game(7, king_capture="four_sides")
    ns2 = g_four.apply(State(board_from(g_four, pieces), ATTACKERS), move)
    assert g_four.king_square(ns2.board) is not None        # same position, wall does NOT count -> safe


def test_throne_is_hostile_side_for_king_capture():
    g = make_game(7, king_capture="four_sides", throne=ThroneRule(hostile_to_king=True))
    # king beside the empty throne (3,3); throne supplies the fourth hostile side
    board = board_from(g, {
        (3, 4): KING, (2, 4): ATTACKER, (4, 4): ATTACKER, (3, 5): ATTACKER, (0, 1): ATTACKER,
    })
    s = State(board, ATTACKERS)
    ns = g.apply(s, (0, 1, 0, 4))                           # a quiet attacker move; throne already pins left side
    # left neighbour of king is throne (3,3) -> hostile_to_king closes the box
    assert g.king_square(ns.board) is None


# --- terminal: no legal moves ----------------------------------------------
def test_side_with_no_moves_loses():
    g = make_game(7)
    board = board_from(g, {
        (3, 3): KING, (2, 3): ATTACKER, (4, 3): ATTACKER, (3, 2): ATTACKER, (3, 4): ATTACKER,
    })
    s = State(board, DEFENDERS)                             # defenders to move, king fully boxed, no other men
    assert g.legal_moves(s) == []
    assert g.result(s) == ATTACKERS


# --- shieldwall -------------------------------------------------------------
def test_shieldwall_captures_a_bracketed_edge_row():
    g = make_game(11, shieldwall=True)
    board = board_from(g, {
        (0, 2): DEFENDER,                                   # left bracket
        (0, 3): ATTACKER, (0, 4): ATTACKER, (0, 5): ATTACKER,   # the wall, on the top edge
        (1, 3): DEFENDER, (1, 4): DEFENDER, (1, 5): DEFENDER,   # backing them from the inside
        (2, 6): DEFENDER,                                   # the mover
        (5, 5): KING,
    })
    s = State(board, DEFENDERS)
    ns = g.apply(s, (2, 6, 0, 6))                           # closes the right bracket
    assert ns.board[g.idx(0, 3)] == EMPTY
    assert ns.board[g.idx(0, 4)] == EMPTY
    assert ns.board[g.idx(0, 5)] == EMPTY


# --- smoke: random games terminate cleanly ---------------------------------
def test_random_playouts_terminate_with_a_valid_result():
    rng = random.Random(1234)
    for builder in (brandub_7x7, tablut_linnaeus_9x9, fetlar_11x11):
        g = Tafl(builder())
        for _ in range(8):
            s = g.initial_state()
            for _ply in range(g.rules.max_plies):
                res = g.result(s)
                if res is not None:
                    break
                moves = g.legal_moves(s)
                s = g.apply(s, rng.choice(moves))
            res = g.result(s)
            assert res in (ATTACKERS, DEFENDERS) or _ply == g.rules.max_plies - 1


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
        except Exception as e:  # noqa: BLE001 -- surface engine bugs as test failures
            failed += 1
            print(f"  ERR  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
