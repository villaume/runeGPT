"""Starting positions for the named board sizes.

Each builder returns a flat ``tuple`` of length ``size*size`` of piece codes
(index = row*size + col). Coordinate lists are written out explicitly rather
than generated, because the symmetric-but-irregular attacker formations are
where reconstructions actually differ -- being able to read the exact squares
matters more than terseness.
"""

from __future__ import annotations

from .rules import EMPTY, ATTACKER, DEFENDER, KING


def _build(size: int, king, defenders, attackers) -> tuple[int, ...]:
    board = [EMPTY] * (size * size)
    for (r, c) in attackers:
        board[r * size + c] = ATTACKER
    for (r, c) in defenders:
        board[r * size + c] = DEFENDER
    kr, kc = king
    board[kr * size + kc] = KING
    return tuple(board)


def brandub() -> tuple[int, ...]:
    king = (3, 3)
    defenders = [(2, 3), (4, 3), (3, 2), (3, 4)]
    attackers = [(0, 3), (1, 3), (5, 3), (6, 3), (3, 0), (3, 1), (3, 5), (3, 6)]
    return _build(7, king, defenders, attackers)


def tablut() -> tuple[int, ...]:
    king = (4, 4)
    defenders = [(4, 2), (4, 3), (4, 5), (4, 6), (2, 4), (3, 4), (5, 4), (6, 4)]
    attackers = [
        (0, 3), (0, 4), (0, 5), (1, 4),     # top
        (8, 3), (8, 4), (8, 5), (7, 4),     # bottom
        (3, 0), (4, 0), (5, 0), (4, 1),     # left
        (3, 8), (4, 8), (5, 8), (4, 7),     # right
    ]
    return _build(9, king, defenders, attackers)


def fetlar() -> tuple[int, ...]:
    king = (5, 5)
    defenders = [
        (5, 3), (5, 4), (5, 6), (5, 7),
        (3, 5), (4, 5), (6, 5), (7, 5),
        (4, 4), (4, 6), (6, 4), (6, 6),
    ]
    attackers = [
        (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 5),         # top
        (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (9, 5),    # bottom
        (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (5, 1),         # left
        (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (5, 9),    # right
    ]
    return _build(11, king, defenders, attackers)


LAYOUTS = {
    "brandub": brandub,
    "tablut": tablut,
    "fetlar": fetlar,
}
