"""Tests for the AlphaZero-lite stack (milestone 4): encoding, net, MCTS,
self-play, and one training step. Tiny nets and low sim counts keep it fast.

Run: ``uv run python -m tafl.tests.test_az``.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from tafl.az.encoding import (
    NUM_PLANES, action_size, decode_action, encode_move, encode_state, legal_mask,
)
from tafl.az.mcts import best_move, run_mcts
from tafl.az.net import TaflNet
from tafl.az.selfplay import play_selfplay_game
from tafl.engine import State, Tafl
from tafl.rules import ATTACKER, EMPTY, KING, TaflRules, brandub_7x7, tablut_linnaeus_9x9


def _board(g, pieces):
    b = [EMPTY] * (g.n * g.n)
    for (r, c), v in pieces.items():
        b[g.idx(r, c)] = v
    return tuple(b)


def test_move_encode_decode_roundtrip():
    for builder in (brandub_7x7, tablut_linnaeus_9x9):
        g = Tafl(builder())
        for m in g.legal_moves(g.initial_state()):
            assert decode_action(encode_move(m, g.n), g.n) == m


def test_legal_mask_matches_move_list():
    g = Tafl(brandub_7x7())
    s = g.initial_state()
    mask = legal_mask(g, s)
    assert mask.sum() == len(g.legal_moves(s))
    assert mask.shape[0] == action_size(g.n)


def test_encode_state_planes():
    g = Tafl(brandub_7x7())
    planes = encode_state(g, g.initial_state())
    assert planes.shape == (g.n, g.n, NUM_PLANES)
    assert planes[3, 3, 3] == 1.0                        # throne plane marks the centre
    assert planes[0, 0, 4] == 1.0                        # corner plane marks corners
    assert planes[:, :, 5].mean() == 1.0                 # attackers to move -> side plane all ones
    assert planes[:, :, 2].sum() == 1.0                  # exactly one king


def test_net_forward_shapes():
    g = Tafl(brandub_7x7())
    net = TaflNet(g.n, channels=8, blocks=1)
    mx.eval(net.parameters())
    x = mx.array(encode_state(g, g.initial_state())[None])
    policy, value = net(x)
    mx.eval(policy, value)
    assert policy.shape == (1, action_size(g.n))
    assert value.shape == (1,)
    assert -1.0 <= float(value[0]) <= 1.0


def test_mcts_finds_mate_and_escape_even_untrained():
    # Terminal-value backup should surface a 1-ply win regardless of net priors.
    gm = Tafl(TaflRules(board_size=7, layout="brandub", king_capture="four_sides"))
    net = TaflNet(7, channels=8, blocks=1)
    mx.eval(net.parameters())
    # attacker mate-in-1 (close the fourth side of the king)
    s = State(_board(gm, {(3, 3): KING, (2, 3): ATTACKER, (4, 3): ATTACKER,
                          (3, 2): ATTACKER, (3, 6): ATTACKER}), "attackers")
    mv = best_move(run_mcts(net, gm, s, sims=48, dirichlet=0.0))
    assert gm.king_square(gm.apply(s, mv).board) is None
    # defender mate-in-1 (run the king to a corner)
    gd = Tafl(TaflRules(board_size=7, layout="brandub", king_goal="corner"))
    s2 = State(_board(gd, {(0, 1): KING, (6, 6): ATTACKER, (5, 5): ATTACKER}), "defenders")
    mv2 = best_move(run_mcts(net, gd, s2, sims=48, dirichlet=0.0))
    assert gd.result(gd.apply(s2, mv2)) == "defenders"


def test_selfplay_produces_valid_samples():
    g = Tafl(brandub_7x7())
    net = TaflNet(g.n, channels=8, blocks=1)
    mx.eval(net.parameters())
    samples, result = play_selfplay_game(net, g, sims=12, temp_moves=4, max_plies=50,
                                         rng=np.random.default_rng(0))
    assert len(samples) >= 1
    planes, pi, z = samples[0]
    assert planes.shape == (g.n, g.n, NUM_PLANES)
    assert pi.shape == (action_size(g.n),) and abs(pi.sum() - 1.0) < 1e-4
    assert all(float(s[2]) in (-1.0, 0.0, 1.0) for s in samples)
    assert result in ("attackers", "defenders", "draw")


def test_training_step_reduces_loss_on_a_fixed_batch():
    import mlx.nn as nn
    import mlx.optimizers as optim

    g = Tafl(brandub_7x7())
    net = TaflNet(g.n, channels=8, blocks=1)
    mx.eval(net.parameters())
    samples, _ = play_selfplay_game(net, g, sims=12, temp_moves=4, max_plies=40,
                                    rng=np.random.default_rng(1))
    x = mx.array(np.stack([s[0] for s in samples]))
    pi = mx.array(np.stack([s[1] for s in samples]))
    z = mx.array(np.array([s[2] for s in samples], dtype=np.float32))

    def loss_fn(net):
        logits, value = net(x)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -(pi * logp).sum(axis=-1).mean() + ((value - z) ** 2).mean()

    lg = nn.value_and_grad(net, loss_fn)
    opt = optim.AdamW(learning_rate=5e-3)
    first = lg(net)[0].item()
    last = first
    for _ in range(25):
        loss, grads = lg(net)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)
        last = loss.item()
    assert last < first                                   # overfits the fixed batch


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
