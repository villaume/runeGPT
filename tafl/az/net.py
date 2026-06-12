"""A small residual conv net with AlphaZero's two heads (policy + value), in MLX.

Deliberately tiny — a handful of residual blocks over a 32-channel trunk — so
self-play on the small boards runs in reasonable time on Apple Silicon, in the
same "minutes not days" spirit as the rune GPT. Channels-last (NHWC) to match
mlx's Conv2d. No normalisation layers in this first cut: it keeps the net robust
with the small, shifting batches that early self-play produces.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .encoding import NUM_PLANES, action_size


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(self.c1(x))
        return nn.relu(x + self.c2(h))


class TaflNet(nn.Module):
    """Input: (B, n, n, NUM_PLANES). Outputs: policy logits (B, A) and value (B,)."""

    def __init__(self, n: int, channels: int = 32, blocks: int = 4):
        super().__init__()
        self.n = n
        self.A = action_size(n)
        self.stem = nn.Conv2d(NUM_PLANES, channels, kernel_size=3, padding=1)
        self.blocks = [_ResBlock(channels) for _ in range(blocks)]

        self.p_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.p_fc = nn.Linear(2 * n * n, self.A)

        self.v_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.v_fc1 = nn.Linear(n * n, channels)
        self.v_fc2 = nn.Linear(channels, 1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        b = x.shape[0]
        h = nn.relu(self.stem(x))
        for blk in self.blocks:
            h = blk(h)

        p = nn.relu(self.p_conv(h)).reshape(b, -1)
        policy = self.p_fc(p)

        v = nn.relu(self.v_conv(h)).reshape(b, -1)
        v = nn.relu(self.v_fc1(v))
        value = mx.tanh(self.v_fc2(v)).reshape(b)
        return policy, value
