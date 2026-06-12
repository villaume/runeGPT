"""AlphaZero-lite for tafl (milestone 4): a small MLX policy+value net, PUCT MCTS,
and a self-play training loop — the stronger oracle the alpha-beta sweeps kept
asking for. See tafl/DESIGN.md milestone 4."""

from .agent import AZAgent
from .encoding import action_size, decode_action, encode_move, encode_state, legal_mask
from .mcts import best_move, policy_target, run_mcts, sample_move
from .net import TaflNet
from .selfplay import play_selfplay_game

__all__ = [
    "TaflNet", "AZAgent",
    "run_mcts", "best_move", "sample_move", "policy_target",
    "encode_state", "encode_move", "decode_action", "legal_mask", "action_size",
    "play_selfplay_game",
]
