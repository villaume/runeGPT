"""Heuristic evaluation for non-terminal positions.

Always scored from the **defenders'** point of view (higher = better for the
king's side); the search negates as needed. The terms are deliberately simple
and legible -- this is a cutoff heuristic for alpha-beta, not a learned value
net (that arrives with the AlphaZero-lite milestone). The weights are hand-set
and only need to be *good enough that the search plays sensibly*, since the
balance signal we care about comes from near-perfect play, not from the eval.
"""

from __future__ import annotations

from .engine import Tafl, State
from .rules import ATTACKER, DEFENDER, EMPTY, KING

# component weights (defenders' perspective)
W_MATERIAL_DEF = 8       # each defender soldier
W_MATERIAL_ATK = 6       # each attacker soldier (they start more numerous)
W_KING_GOAL = 5          # per step closer the king is to its goal
W_KING_MOBILITY = 2      # squares the king can move to
W_ATK_SURROUND = 12      # each attacker orthogonally adjacent to the king
W_NEAR_ESCAPE = 250      # king has a clear one-move run to a winning square

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _king_goal_distance(game: Tafl, kr: int, kc: int) -> int:
    """Manhattan distance from the king to its nearest winning square."""
    n = game.n
    if game.rules.king_goal == "corner":
        targets = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        return min(abs(kr - tr) + abs(kc - tc) for tr, tc in targets)
    # any_edge: distance to the nearest wall
    return min(kr, kc, n - 1 - kr, n - 1 - kc)


def _king_escape_squares(game: Tafl, state: State, kr: int, kc: int) -> tuple[int, int]:
    """(mobility, winning_moves): how many squares the king can reach in one move,
    and how many of those are immediate wins. Two distinct winning moves on the
    defenders' turn is effectively a victory (attackers can block only one)."""
    board = state.board
    mobility = winning = 0
    for dr, dc in _DIRS:
        r, c = kr + dr, kc + dc
        while game.in_bounds(r, c) and board[game.idx(r, c)] == EMPTY:
            if game._can_stop(KING, r, c, board):
                mobility += 1
                if game._king_on_goal(r, c):
                    winning += 1
            r, c = r + dr, c + dc
    return mobility, winning


def evaluate(game: Tafl, state: State) -> int:
    """Static evaluation, defenders' perspective. Assumes a non-terminal state."""
    board = state.board
    score = 0
    n_def = n_atk = 0
    for piece in board:
        if piece == DEFENDER:
            n_def += 1
        elif piece == ATTACKER:
            n_atk += 1
    score += W_MATERIAL_DEF * n_def - W_MATERIAL_ATK * n_atk

    kp = game.king_square(board)
    if kp is None:
        return score                      # defensive: terminal handled by search
    kr, kc = kp

    score -= W_KING_GOAL * _king_goal_distance(game, kr, kc)

    mobility, winning = _king_escape_squares(game, state, kr, kc)
    score += W_KING_MOBILITY * mobility
    if winning >= 1:
        # one open run is a threat; two is unstoppable on the defenders' turn
        score += W_NEAR_ESCAPE * (winning + (winning >= 2))

    adj_attackers = 0
    for dr, dc in _DIRS:
        r, c = kr + dr, kc + dc
        if game.in_bounds(r, c) and board[game.idx(r, c)] == ATTACKER:
            adj_attackers += 1
    score -= W_ATK_SURROUND * adj_attackers

    return score
