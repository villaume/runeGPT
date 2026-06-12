"""Negamax + alpha-beta search with a transposition table.

This is the "instrument" of the balance search (DESIGN section 2): it must be
strong enough that an imbalance we measure is the *game's*, not the agent's. For
7x7 brandub a few plies of well-ordered alpha-beta already plays a serious game,
which is why brandub is the right place to anchor ground truth before any neural
net.

Design notes
------------
* **Negamax**: one routine, values always from the side-to-move's perspective.
* **Mate scoring** is ``MATE - ply`` so the search prefers faster wins and slower
  losses -- crucial in self-play, or a winning side dithers into a repetition draw.
  Mate-valued nodes are *not* written to the TT (their value is path-dependent),
  keeping the table sound.
* **Move ordering**: TT best-move first, then a cheap tactical key (king runs for
  the defenders, crowding the king for the attackers). Good ordering is most of
  alpha-beta's value.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..engine import Move, State, Tafl
from ..eval import evaluate
from ..rules import ATTACKERS, KING

MATE = 1_000_000
MATE_THRESHOLD = 900_000          # |value| above this is a forced mate, not heuristic

# TT entry flags
EXACT, LOWER, UPPER = 0, 1, 2
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class _TTEntry:
    depth: int
    flag: int
    value: int
    best: Move | None


class _Searcher:
    def __init__(self, game: Tafl):
        self.game = game
        self.tt: dict[State, _TTEntry] = {}
        self.nodes = 0

    # --- terminal probe that avoids result()'s extra legal_moves() call -------
    def _terminal(self, state: State, ply: int):
        g = self.game
        kp = g.king_square(state.board)
        if kp is None:                                   # king captured -> attackers win
            return _mate_for(ATTACKERS, state.to_move, ply)
        if g._king_on_goal(*kp):                          # king escaped -> defenders win
            from ..rules import DEFENDERS
            return _mate_for(DEFENDERS, state.to_move, ply)
        return None

    def _ordered_moves(self, state: State, tt_best: Move | None) -> list[Move]:
        g, board = self.game, state.board
        kp = g.king_square(board)
        attackers_turn = state.to_move == ATTACKERS

        def key(m: Move) -> int:
            _, _, r, c = m
            s = 0
            if kp is not None:
                kr, kc = kp
                adj = abs(r - kr) + abs(c - kc) == 1
                if attackers_turn and adj:
                    s += 3                                # crowd the king
                if not attackers_turn and board[g.idx(m[0], m[1])] == KING:
                    s += 2                                # move the king...
                    if g.is_edge(r, c) or g.is_corner(r, c):
                        s += 4                            # ...toward the rim
            return s

        moves = g.legal_moves(state)
        moves.sort(key=key, reverse=True)
        if tt_best is not None and tt_best in moves:
            moves.remove(tt_best)
            moves.insert(0, tt_best)
        return moves

    def negamax(self, state: State, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        term = self._terminal(state, ply)
        if term is not None:
            return term

        alpha_orig = alpha
        entry = self.tt.get(state)
        tt_best = entry.best if entry else None
        if entry and entry.depth >= depth:
            if entry.flag == EXACT:
                return entry.value
            if entry.flag == LOWER:
                alpha = max(alpha, entry.value)
            elif entry.flag == UPPER:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        if depth == 0:
            v = evaluate(self.game, state)
            return v if state.to_move == _DEF else -v

        moves = self._ordered_moves(state, tt_best)
        if not moves:                                     # no legal move -> side to move loses
            return -MATE + ply

        best = -MATE - 1
        best_move = None
        for m in moves:
            child = self.game.apply(state, m)
            v = -self.negamax(child, depth - 1, -beta, -alpha, ply + 1)
            if v > best:
                best, best_move = v, m
            alpha = max(alpha, v)
            if alpha >= beta:
                break

        if abs(best) < MATE_THRESHOLD:                    # don't cache path-dependent mates
            flag = EXACT
            if best <= alpha_orig:
                flag = UPPER
            elif best >= beta:
                flag = LOWER
            self.tt[state] = _TTEntry(depth, flag, best, best_move)
        return best


# defenders side constant, imported lazily to avoid a cycle at module top
from ..rules import DEFENDERS as _DEF  # noqa: E402


def _mate_for(winner: str, to_move: str, ply: int) -> int:
    base = MATE - ply
    return base if winner == to_move else -(MATE - ply)


def search_root(game: Tafl, state: State, depth: int) -> tuple[Move | None, int, int]:
    """Iterative-deepening search with the best move tracked at the root itself
    (mate-valued nodes are intentionally not cached, so we can't read the value
    back from the TT). Returns (best_move, value, nodes), value from the
    side-to-move's perspective."""
    s = _Searcher(game)
    if not game.legal_moves(state):
        return None, -MATE, 0
    best_move: Move | None = None
    value = 0
    for d in range(1, depth + 1):
        alpha = -MATE - 1
        local_best, local_move = -MATE - 1, None
        for m in s._ordered_moves(state, best_move):      # previous iteration's best leads ordering
            v = -s.negamax(game.apply(state, m), d - 1, -(MATE + 1), -alpha, 1)
            if v > local_best:
                local_best, local_move = v, m
            alpha = max(alpha, v)
        best_move, value = local_move, local_best
    return best_move, value, s.nodes


class AlphaBetaAgent:
    """Fixed-depth alpha-beta player."""

    def __init__(self, depth: int = 3):
        self.depth = depth

    def choose(self, game: Tafl, state: State) -> Move:
        move, _, _ = search_root(game, state, self.depth)
        if move is None:
            raise ValueError("no legal move (terminal state passed to agent)")
        return move


class RandomAgent:
    """Uniform-random legal mover -- the weak baseline for sensitivity checks."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def choose(self, game: Tafl, state: State) -> Move:
        return self.rng.choice(game.legal_moves(state))
