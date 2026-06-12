"""tafl -- reconstructing Hnefatafl rules by self-play balance search.

Milestone 1: a generic, rule-set-agnostic engine plus named reference rule sets.
See ``tafl/DESIGN.md`` for the full plan.
"""

from .engine import Move, State, Tafl, side_of
from .rules import (
    ATTACKERS,
    DEFENDERS,
    CornerRule,
    REFERENCE_RULES,
    TaflRules,
    ThroneRule,
    brandub_7x7,
    copenhagen_11x11,
    fetlar_11x11,
    hnefatafl_13x13,
    tablut_historical_9x9,
    tablut_linnaeus_9x9,
    tablut_smith_1811_9x9,
)

__all__ = [
    "Tafl", "State", "Move", "side_of",
    "TaflRules", "ThroneRule", "CornerRule",
    "ATTACKERS", "DEFENDERS",
    "REFERENCE_RULES",
    "brandub_7x7", "tablut_linnaeus_9x9", "tablut_smith_1811_9x9",
    "tablut_historical_9x9", "fetlar_11x11", "copenhagen_11x11", "hnefatafl_13x13",
]
