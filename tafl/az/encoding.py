"""State and move encoding for the AlphaZero-lite net. Pure NumPy, no MLX — so it
is fast to import and trivial to unit-test.

Move encoding (AlphaZero "queen-ray" style, but tafl pieces only slide
orthogonally): an action is (from-square, direction, distance). With 4 directions
and distances 1..n-1 the action space is ``n*n*4*(n-1)`` — 1176 for 7x7, 2592 for
9x9. The policy head emits one logit per action; illegal actions are masked before
the softmax.

State encoding: ``(n, n, C)`` float planes (channels-last, matching mlx Conv2d):
attackers, defenders, king, throne marker, corner marker, and a side-to-move plane.
Piece planes are *not* swapped by side — tafl's two sides have genuinely different
roles — so the side-to-move plane carries whose turn it is.
"""

from __future__ import annotations

import numpy as np

from ..engine import Move, State, Tafl
from ..rules import ATTACKER, ATTACKERS, DEFENDER, KING

# fixed direction order: up, down, left, right (index 0..3)
AZ_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
NUM_PLANES = 6


def action_size(n: int) -> int:
    return n * n * 4 * (n - 1)


def encode_move(move: Move, n: int) -> int:
    r0, c0, r1, c1 = move
    dr = (r1 > r0) - (r1 < r0)
    dc = (c1 > c0) - (c1 < c0)
    dist = abs(r1 - r0) + abs(c1 - c0)          # orthogonal: one delta is zero
    direction = AZ_DIRS.index((dr, dc))
    return ((r0 * n + c0) * 4 + direction) * (n - 1) + (dist - 1)


def decode_action(action: int, n: int) -> Move:
    dist = action % (n - 1) + 1
    action //= (n - 1)
    direction = action % 4
    square = action // 4
    r0, c0 = divmod(square, n)
    dr, dc = AZ_DIRS[direction]
    return (r0, c0, r0 + dr * dist, c0 + dc * dist)


def legal_action_indices(game: Tafl, state: State) -> tuple[list[Move], list[int]]:
    moves = game.legal_moves(state)
    return moves, [encode_move(m, game.n) for m in moves]


def legal_mask(game: Tafl, state: State) -> np.ndarray:
    mask = np.zeros(action_size(game.n), dtype=bool)
    for m in game.legal_moves(state):
        mask[encode_move(m, game.n)] = True
    return mask


def encode_state(game: Tafl, state: State) -> np.ndarray:
    n = game.n
    planes = np.zeros((n, n, NUM_PLANES), dtype=np.float32)
    for i, piece in enumerate(state.board):
        r, c = divmod(i, n)
        if piece == ATTACKER:
            planes[r, c, 0] = 1.0
        elif piece == DEFENDER:
            planes[r, c, 1] = 1.0
        elif piece == KING:
            planes[r, c, 2] = 1.0
    tr, tc = game.throne_rc
    planes[tr, tc, 3] = 1.0
    for (r, c) in game.corners_rc:
        planes[r, c, 4] = 1.0
    if state.to_move == ATTACKERS:
        planes[:, :, 5] = 1.0
    return planes
