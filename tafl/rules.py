"""Rule-config schema for the tafl family + named reference rule sets.

A single ``TaflRules`` object fully parameterises the generic engine in
``engine.py`` -- there is no per-variant code. Each axis below is one of the
historically *undetermined* mechanics (see ``tafl/DESIGN.md`` section 1); a
"rule set" is one choice per axis. The whole point of the project is to sweep
this space and score each combination for balance, so the schema is the
primary artifact.

The named rule sets at the bottom are *reconstructions*, not gospel. They give
the engine tests something concrete to exercise and later give the balance
search its seed points. Where a choice is contested (e.g. how the Tablut king
is captured) the docstring says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# --- piece codes (shared with engine.py) ------------------------------------
EMPTY, ATTACKER, DEFENDER, KING = 0, 1, 2, 3

ATTACKERS = "attackers"
DEFENDERS = "defenders"


@dataclass(frozen=True)
class ThroneRule:
    """Behaviour of the central throne square once the king has left it."""

    soldiers_may_stop: bool = False     # may an attacker/defender end its move on the throne?
    blocks_passage: bool = False        # does the empty throne block movement through it (soldiers)?
    hostile_to_attackers: bool = False  # empty throne acts as a capturing anvil against attackers
    hostile_to_defenders: bool = False  # empty throne acts as a capturing anvil against defenders
    hostile_to_king: bool = False       # empty throne counts as a hostile *side* for king capture


@dataclass(frozen=True)
class CornerRule:
    """Behaviour of the four corner squares."""

    soldiers_may_stop: bool = False     # corners restricted to the king if False
    hostile_to_attackers: bool = True   # corner acts as a capturing anvil against attackers
    hostile_to_defenders: bool = True   # corner acts as a capturing anvil against defenders
    hostile_to_king: bool = True        # corner counts as a hostile side for king capture


# King-capture mode -- the single most contested mechanic.
#   "four_sides"  : king taken only when all four orthogonal neighbours are hostile
#                   (attacker or hostile square). Off-board never counts, so the
#                   king is *safe on the board edge*. Strong-king variant.
#   "edge_counts" : as four_sides, but the board edge counts as a hostile side, so
#                   a king on the wall falls to 3 attackers and in a corner to 2.
#   "two_sides"   : king taken custodially like an ordinary soldier (two opposite
#                   sides). Strongly attacker-favoured.
KingCapture = Literal["four_sides", "edge_counts", "two_sides"]

# King's win condition.
#   "corner"   : king must reach one of the four corners.
#   "any_edge" : king wins on reaching any edge square.
KingGoal = Literal["corner", "any_edge"]

# Repetition handling (perpetual positions).
Repetition = Literal["draw", "loss_attacker", "loss_mover"]


@dataclass(frozen=True)
class TaflRules:
    board_size: int
    layout: str                         # named starting layout (validated against board_size)
    king_capture: KingCapture = "four_sides"
    king_goal: KingGoal = "corner"
    king_armed: bool = True             # may the king take part in captures it initiates?
    throne: ThroneRule = field(default_factory=ThroneRule)
    corners: CornerRule = field(default_factory=CornerRule)
    shieldwall: bool = False            # edge-row line captures
    exit_fort: bool = False             # king escapes via an edge fort (not yet implemented)
    repetition: Repetition = "draw"
    max_plies: int = 1000               # safety bound for self-play; over => draw

    def __post_init__(self) -> None:
        if self.board_size % 2 == 0:
            raise ValueError("tafl boards are odd-sided so the throne is central")
        if self.board_size < 5:
            raise ValueError("board too small for a tafl start position")


# --- named reference rule sets ----------------------------------------------
# These instantiate TaflRules for the four families the tests and the eventual
# balance search lean on. They are deliberately *typical modern reconstructions*
# and are labelled as such; the historical truth is exactly what we are trying
# to triangulate, so none of these is asserted to be canonical.

def brandub_7x7() -> TaflRules:
    """7x7 Irish brandub. Common modern reconstruction: corner escape, king taken
    on four sides, empty throne hostile to the king (so a king beside it falls to
    three)."""
    return TaflRules(
        board_size=7,
        layout="brandub",
        king_capture="four_sides",
        king_goal="corner",
        throne=ThroneRule(hostile_to_king=True),
        corners=CornerRule(),
    )


def tablut_linnaeus_9x9() -> TaflRules:
    """9x9 Tablut, faithful-as-possible to Linnaeus (1732): king escapes to any
    edge. We use the four-sides king capture here -- the *balanced* reading that
    modern reconstructors favour."""
    return TaflRules(
        board_size=9,
        layout="tablut",
        king_capture="four_sides",
        king_goal="any_edge",
        throne=ThroneRule(hostile_to_king=True),
        corners=CornerRule(soldiers_may_stop=True, hostile_to_attackers=False,
                           hostile_to_defenders=False, hostile_to_king=False),
    )


def tablut_smith_1811_9x9() -> TaflRules:
    """9x9 Tablut as it came down through Smith's 1811 mistranslation: edge escape
    *and* a king captured like an ordinary man (two sides). Expected to test as
    badly imbalanced -- it is included precisely as the negative control for the
    balance search."""
    return TaflRules(
        board_size=9,
        layout="tablut",
        king_capture="two_sides",
        king_goal="any_edge",
        throne=ThroneRule(),
        corners=CornerRule(soldiers_may_stop=True, hostile_to_attackers=False,
                           hostile_to_defenders=False, hostile_to_king=False),
    )


def fetlar_11x11() -> TaflRules:
    """11x11 Fetlar (Aage Nielsen community reconstruction): corner escape, strong
    king taken on four sides, hostile throne and corners, shieldwall captures on.
    Exit forts are part of the full Copenhagen ruleset but not yet implemented."""
    return TaflRules(
        board_size=11,
        layout="fetlar",
        king_capture="four_sides",
        king_goal="corner",
        throne=ThroneRule(hostile_to_king=True, hostile_to_attackers=True,
                          hostile_to_defenders=True),
        corners=CornerRule(),
        shieldwall=True,
    )


def hnefatafl_13x13() -> TaflRules:
    """13x13 great hnefatafl (24+12+1). Like Fetlar but on the larger board: corner
    escape, strong king taken on four sides, hostile throne and corners, shieldwall.
    Whether the big board needs a *weak* king to stay playable is exactly what the
    13x13 sweep tests."""
    return TaflRules(
        board_size=13,
        layout="hnefatafl13",
        king_capture="four_sides",
        king_goal="corner",
        throne=ThroneRule(hostile_to_king=True, hostile_to_attackers=True,
                          hostile_to_defenders=True),
        corners=CornerRule(),
        shieldwall=True,
    )


REFERENCE_RULES = {
    "brandub": brandub_7x7,
    "tablut_linnaeus": tablut_linnaeus_9x9,
    "tablut_smith_1811": tablut_smith_1811_9x9,
    "fetlar": fetlar_11x11,
    "hnefatafl13": hnefatafl_13x13,
}
