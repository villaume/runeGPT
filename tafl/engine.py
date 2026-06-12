"""Generic, rule-set-agnostic tafl engine.

One implementation, parameterised entirely by a ``TaflRules`` object. Nothing
here knows the name "brandub" or "fetlar" -- it knows board size, capture mode,
goal, and which squares are hostile, and reads all of that off the rules.

State is immutable and hashable (the board is a tuple), so positions can be
dropped straight into a set/dict for repetition detection and, later, for
transposition tables in the search agents.

Conventions
-----------
* Coordinates are ``(row, col)``, 0-indexed; the flat board index is
  ``row*size + col``.
* ``Move`` is ``(from_row, from_col, to_row, to_col)``.
* ``result()`` returns ``"attackers"``, ``"defenders"``, ``"draw"`` or ``None``
  (game ongoing). A side that cannot move loses.
* King capture is resolved by removing the king from the board, so a captured
  king simply means "no king present" downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from .layouts import LAYOUTS
from .rules import (
    ATTACKER,
    ATTACKERS,
    DEFENDER,
    DEFENDERS,
    EMPTY,
    KING,
    TaflRules,
)

Move = tuple[int, int, int, int]
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class State:
    board: tuple[int, ...]
    to_move: str            # ATTACKERS or DEFENDERS

    def __hash__(self) -> int:          # board+turn identity for repetition tables
        return hash((self.board, self.to_move))


def side_of(piece: int) -> str | None:
    if piece == ATTACKER:
        return ATTACKERS
    if piece in (DEFENDER, KING):
        return DEFENDERS
    return None


class Tafl:
    """A playable game defined by a single ``TaflRules``."""

    def __init__(self, rules: TaflRules):
        if rules.exit_fort:
            raise NotImplementedError(
                "exit_fort (edge-fort king escape) is a later milestone; "
                "set exit_fort=False"
            )
        if rules.layout not in LAYOUTS:
            raise ValueError(f"unknown layout {rules.layout!r}")
        self.rules = rules
        self.n = rules.board_size
        c = self.n // 2
        self.throne_rc = (c, c)
        self.corners_rc = frozenset(
            {(0, 0), (0, self.n - 1), (self.n - 1, 0), (self.n - 1, self.n - 1)}
        )

    # --- basic geometry -----------------------------------------------------
    def idx(self, r: int, c: int) -> int:
        return r * self.n + c

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    def is_corner(self, r: int, c: int) -> bool:
        return (r, c) in self.corners_rc

    def is_throne(self, r: int, c: int) -> bool:
        return (r, c) == self.throne_rc

    def is_edge(self, r: int, c: int) -> bool:
        return r == 0 or c == 0 or r == self.n - 1 or c == self.n - 1

    # --- setup --------------------------------------------------------------
    def initial_state(self) -> State:
        return State(board=LAYOUTS[self.rules.layout](), to_move=ATTACKERS)

    def king_square(self, board: tuple[int, ...]) -> tuple[int, int] | None:
        try:
            i = board.index(KING)
        except ValueError:
            return None
        return divmod(i, self.n)

    # --- movement -----------------------------------------------------------
    def _can_stop(self, piece: int, r: int, c: int, board: tuple[int, ...]) -> bool:
        """May ``piece`` end its move on the empty square (r, c)?"""
        if board[self.idx(r, c)] != EMPTY:
            return False
        is_king = piece == KING
        if self.is_corner(r, c) and not is_king and not self.rules.corners.soldiers_may_stop:
            return False
        if self.is_throne(r, c) and not is_king and not self.rules.throne.soldiers_may_stop:
            return False
        return True

    def _blocks_passage(self, piece: int, r: int, c: int, board: tuple[int, ...]) -> bool:
        """Is the (empty) square (r, c) impassable to ``piece`` sliding through it?"""
        if self.is_throne(r, c) and piece != KING and self.rules.throne.blocks_passage:
            return True
        return False

    def legal_moves(self, state: State) -> list[Move]:
        board, side = state.board, state.to_move
        moves: list[Move] = []
        for i, piece in enumerate(board):
            if side_of(piece) != side:
                continue
            r0, c0 = divmod(i, self.n)
            for dr, dc in _DIRS:
                r, c = r0 + dr, c0 + dc
                while self.in_bounds(r, c) and board[self.idx(r, c)] == EMPTY:
                    if self._blocks_passage(piece, r, c, board):
                        break
                    if self._can_stop(piece, r, c, board):
                        moves.append((r0, c0, r, c))
                    r, c = r + dr, c + dc
        return moves

    # --- captures -----------------------------------------------------------
    def _hostile_square_for(self, side: str, r: int, c: int, board: tuple[int, ...]) -> bool:
        """Does the special square (r, c) act as a capturing anvil for ``side``
        (i.e. it can pin an *enemy* of ``side`` against a friendly piece)?"""
        enemy_is_attacker = side == DEFENDERS
        if self.is_corner(r, c):
            return (self.rules.corners.hostile_to_attackers if enemy_is_attacker
                    else self.rules.corners.hostile_to_defenders)
        if self.is_throne(r, c):
            # Aage/WTF: the throne is hostile to the attackers *always* (even while the
            # king occupies it), but hostile to the defenders only when empty.
            if enemy_is_attacker:
                return self.rules.throne.hostile_to_attackers
            return self.rules.throne.hostile_to_defenders and board[self.idx(r, c)] == EMPTY
        return False

    def _is_anchor(self, side: str, r: int, c: int, board: tuple[int, ...]) -> bool:
        """Can square (r, c) serve as the far end of a custodial capture by
        ``side`` -- a friendly piece or a hostile special square."""
        if not self.in_bounds(r, c):
            return False
        if side_of(board[self.idx(r, c)]) == side:
            return True
        return self._hostile_square_for(side, r, c, board)

    def _apply_custodial(self, board: list[int], to_r: int, to_c: int, side: str) -> None:
        """Resolve soldier captures triggered by ``side`` moving a piece to
        (to_r, to_c). Mutates ``board`` in place. The king is never taken here --
        it has its own rule."""
        enemy = ATTACKER if side == DEFENDERS else DEFENDER
        for dr, dc in _DIRS:
            ar, ac = to_r + dr, to_c + dc
            if not self.in_bounds(ar, ac) or board[self.idx(ar, ac)] != enemy:
                continue
            if self._is_anchor(side, ar + dr, ac + dc, tuple(board)):
                board[self.idx(ar, ac)] = EMPTY

    # --- shieldwall ---------------------------------------------------------
    def _apply_shieldwall(self, board: list[int], to_r: int, to_c: int, side: str) -> None:
        """A row of enemy soldiers pinned against the board edge and bracketed at
        both ends by friendly pieces (or a corner) is captured as a group.

        Only triggered when the moving piece lands on the same edge as the wall,
        completing one of its brackets. Conservative implementation: it captures
        only a straight contiguous line along the edge the mover just joined.
        """
        if not self.rules.shieldwall or not self.is_edge(to_r, to_c):
            return
        enemy = ATTACKER if side == DEFENDERS else DEFENDER
        # Edge orientation: which axis runs *along* the wall.
        on_top_bottom = to_r in (0, self.n - 1)
        on_left_right = to_c in (0, self.n - 1)
        for along in _edge_axes(on_top_bottom, on_left_right):
            inward = _inward(to_r, to_c, self.n)
            self._scan_wall(board, to_r, to_c, along, inward, enemy, side)

    def _scan_wall(self, board, to_r, to_c, along, inward, enemy, side) -> None:
        dr, dc = along
        for direction in (1, -1):
            line: list[int] = []
            r, c = to_r + dr * direction, to_c + dc * direction
            # walk the contiguous run of bracketed enemies along the edge
            while self.in_bounds(r, c) and board[self.idx(r, c)] == enemy:
                ir, ic = r + inward[0], c + inward[1]
                if not self.in_bounds(ir, ic) or board[self.idx(ir, ic)] not in (
                    DEFENDER, KING, ATTACKER,
                ) or side_of(board[self.idx(ir, ic)]) != side:
                    line = []  # not backed by a friendly piece on the inside -> no wall
                    break
                line.append(self.idx(r, c))
                r, c = r + dr * direction, c + dc * direction
            if not line:
                continue
            # far end must be a friendly piece or a corner to close the bracket
            if (self.in_bounds(r, c) and side_of(board[self.idx(r, c)]) == side) or \
               self.is_corner(r, c):
                for i in line:
                    board[i] = EMPTY

    # --- king capture -------------------------------------------------------
    def _king_side_hostile(self, kr, kc, dr, dc, board) -> bool:
        r, c = kr + dr, kc + dc
        if not self.in_bounds(r, c):
            return self.rules.king_capture == "edge_counts"
        if board[self.idx(r, c)] == ATTACKER:
            return True
        if self.is_throne(r, c) and board[self.idx(r, c)] == EMPTY and self.rules.throne.hostile_to_king:
            return True
        if self.is_corner(r, c) and self.rules.corners.hostile_to_king:
            return True
        return False

    def _king_captured(self, board: tuple[int, ...]) -> bool:
        pos = self.king_square(board)
        if pos is None:
            return False
        kr, kc = pos
        mode = self.rules.king_capture
        if mode in ("four_sides", "edge_counts"):
            return all(self._king_side_hostile(kr, kc, dr, dc, board) for dr, dc in _DIRS)
        if mode == "two_sides_strong_throne":
            # Aage/WTF reading: the king is taken custodially (two sides) in the open,
            # but needs a full surround on/around the throne. He is strong ON the throne
            # always; whether being merely ADJACENT also strengthens him is governed by
            # throne.hostile_to_king (true for Tablut -> 3 sides; false for Brandubh).
            on_throne = (kr, kc) == self.throne_rc
            adj_throne = any((kr + dr, kc + dc) == self.throne_rc for dr, dc in _DIRS)
            if on_throne or (adj_throne and self.rules.throne.hostile_to_king):
                return all(self._king_side_hostile(kr, kc, dr, dc, board) for dr, dc in _DIRS)
        # two_sides (and the open-board case above): custodial along either axis
        horiz = (self._king_side_hostile(kr, kc, 0, -1, board)
                 and self._king_side_hostile(kr, kc, 0, 1, board))
        vert = (self._king_side_hostile(kr, kc, -1, 0, board)
                and self._king_side_hostile(kr, kc, 1, 0, board))
        return horiz or vert

    # --- transitions --------------------------------------------------------
    def apply(self, state: State, move: Move) -> State:
        r0, c0, r1, c1 = move
        board = list(state.board)
        piece = board[self.idx(r0, c0)]
        board[self.idx(r0, c0)] = EMPTY
        board[self.idx(r1, c1)] = piece

        # the mover's captures (king does nothing if unarmed)
        if not (piece == KING and not self.rules.king_armed):
            self._apply_custodial(board, r1, c1, state.to_move)
            self._apply_shieldwall(board, r1, c1, state.to_move)

        # king capture only ever happens on the attackers' move
        if state.to_move == ATTACKERS and self._king_captured(tuple(board)):
            kp = self.king_square(tuple(board))
            if kp is not None:
                board[self.idx(*kp)] = EMPTY

        nxt = DEFENDERS if state.to_move == ATTACKERS else ATTACKERS
        return State(board=tuple(board), to_move=nxt)

    # --- terminal -----------------------------------------------------------
    def result(self, state: State) -> str | None:
        if self.king_square(state.board) is None:
            return ATTACKERS                              # king captured
        kp = self.king_square(state.board)
        if self._king_on_goal(*kp):
            return DEFENDERS
        if not self.legal_moves(state):
            return DEFENDERS if state.to_move == ATTACKERS else ATTACKERS
        return None

    def _king_on_goal(self, kr: int, kc: int) -> bool:
        if self.rules.king_goal == "corner":
            return self.is_corner(kr, kc)
        return self.is_edge(kr, kc)

    # --- debug --------------------------------------------------------------
    def render(self, state: State) -> str:
        glyph = {EMPTY: ".", ATTACKER: "A", DEFENDER: "D", KING: "K"}
        rows = []
        for r in range(self.n):
            cells = []
            for c in range(self.n):
                p = state.board[self.idx(r, c)]
                ch = glyph[p]
                if p == EMPTY and self.is_throne(r, c):
                    ch = "_"
                elif p == EMPTY and self.is_corner(r, c):
                    ch = "x"
                cells.append(ch)
            rows.append(" ".join(cells))
        return "\n".join(rows)


def _edge_axes(on_top_bottom: bool, on_left_right: bool):
    """The axis that runs *along* the edge the mover sits on (a corner is on two)."""
    axes = []
    if on_top_bottom:
        axes.append((0, 1))
    if on_left_right:
        axes.append((1, 0))
    return axes


def _inward(r: int, c: int, n: int) -> tuple[int, int]:
    """Unit vector pointing from edge square (r, c) toward the board interior."""
    if r == 0:
        return (1, 0)
    if r == n - 1:
        return (-1, 0)
    if c == 0:
        return (0, 1)
    return (0, -1)
