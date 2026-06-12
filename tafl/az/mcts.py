"""PUCT Monte-Carlo Tree Search driven by the net's policy + value (no rollouts).

Each node stores visit count N and total value W *from the perspective of the
side to move at that node*; backup flips the sign every ply, and a parent scores
a child by ``-child.Q`` (the child's value is the opponent's). Leaves that are
terminal use the true game result instead of the net — exact where it counts.
"""

from __future__ import annotations

import math

import numpy as np

from ..engine import State, Tafl
from .encoding import action_size, encode_move, encode_state

try:  # MLX is only needed when a real net is used; keep import lazy-friendly
    import mlx.core as mx
except Exception:  # pragma: no cover
    mx = None


class Node:
    __slots__ = ("prior", "N", "W", "children")

    def __init__(self, prior: float):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children: dict | None = None      # None until expanded

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0


def evaluate(net, game: Tafl, state: State):
    """Run the net on one state. Returns (legal_moves, priors_over_moves, value),
    value from the side-to-move's perspective."""
    planes = encode_state(game, state)
    logits, value = net(mx.array(planes[None]))
    mx.eval(logits, value)
    logits = np.array(logits[0])
    moves = game.legal_moves(state)
    if not moves:
        return moves, np.array([]), float(np.array(value)[0])
    idx = np.array([encode_move(m, game.n) for m in moves])
    ml = logits[idx]
    ml = ml - ml.max()
    pr = np.exp(ml)
    pr /= pr.sum()
    return moves, pr, float(np.array(value)[0])


def _terminal_value(result: str, to_move: str) -> float:
    if result == "draw":
        return 0.0
    return 1.0 if result == to_move else -1.0


def _expand(node: Node, net, game: Tafl, state: State) -> float:
    moves, priors, value = evaluate(net, game, state)
    node.children = {m: Node(float(p)) for m, p in zip(moves, priors)}
    return value


def _select(node: Node, c_puct: float):
    sqrt_n = math.sqrt(node.N) if node.N else 0.0
    best, best_move, best_child = -1e18, None, None
    for move, child in node.children.items():
        u = c_puct * child.prior * sqrt_n / (1 + child.N)
        score = -child.Q + u
        if score > best:
            best, best_move, best_child = score, move, child
    return best_move, best_child


def run_mcts(net, game: Tafl, root_state: State, *, sims: int = 64,
             c_puct: float = 1.5, dirichlet: float = 0.0, eps: float = 0.25,
             rng: np.random.Generator | None = None) -> Node:
    root = Node(0.0)
    _expand(root, net, game, root_state)
    if dirichlet > 0 and root.children:
        rng = rng or np.random.default_rng()
        noise = rng.dirichlet([dirichlet] * len(root.children))
        for child, nz in zip(root.children.values(), noise):
            child.prior = (1 - eps) * child.prior + eps * float(nz)

    for _ in range(sims):
        node, s, path = root, root_state, [root]
        while node.children:
            move, node = _select(node, c_puct)
            s = game.apply(s, move)
            path.append(node)
        result = game.result(s)
        value = _terminal_value(result, s.to_move) if result is not None \
            else _expand(node, net, game, s)
        for nd in reversed(path):           # backup, flipping sign each ply
            nd.N += 1
            nd.W += value
            value = -value
    return root


def policy_target(root: Node, n: int) -> np.ndarray:
    """Visit-count distribution over the full action space — the policy training
    target π."""
    pi = np.zeros(action_size(n), dtype=np.float32)
    for move, child in root.children.items():
        pi[encode_move(move, n)] = child.N
    total = pi.sum()
    if total > 0:
        pi /= total
    return pi


def best_move(root: Node):
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]


def sample_move(root: Node, temperature: float, rng: np.random.Generator):
    moves = list(root.children)
    counts = np.array([root.children[m].N for m in moves], dtype=np.float64)
    if temperature <= 1e-6 or counts.sum() == 0:
        return moves[int(counts.argmax())]
    probs = counts ** (1.0 / temperature)
    probs /= probs.sum()
    return moves[int(rng.choice(len(moves), p=probs))]
