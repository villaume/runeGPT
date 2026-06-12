# tafl/ — reconstructing Hnefatafl rules by self-play balance search

A design doc. No code yet. The premise: we can't *recover* lost tafl rules from game
records (none survive), but we can ask **which candidate rule sets produce a balanced,
strategically deep game** — on the assumption that a game played for centuries wasn't a
forced win for either side. Strong self-play agents score each rule set; archaeology
constrains the search space. This is the same logic the Digital Ludeme Project (Ludii,
Cameron Browne et al., Maastricht) uses to reconstruct ancient games.

This is a cousin of the rest of runeGPT: another "small model on Norse material" problem,
reusing the MLX/nanoGPT muscle for an AlphaZero-lite net instead of a char-GPT.

---

## 1. The historical uncertainty we're modelling

No complete original rule set survives. Evidence:

- **Tablut (9×9)** — Linnaeus, *Iter Lapponicum* (1732), Sámi game. The *only* near-complete
  historical account of any tafl game. The 1811 English translation (Smith) introduced
  ambiguities that polluted later reconstructions.
- **Tawlbwrdd (Welsh)** — manuscript gives piece counts, not mechanics.
- **Saga references** (Hervarar saga riddle, etc.) name the game, don't teach it.
- **Archaeological boards** at 7×7 (Ballinderry), 9×9, 11×11, 13×13, 19×19 — sizes/counts only.

The genuinely undetermined mechanics (each becomes a config axis in §3):

1. **King capture** — surrounded on 4 orthogonal sides? 2 sides (like a soldier)? against
   board edge counts as a side?
2. **King's goal** — reach a corner, or reach any edge square?
3. **Throne (center) square** — hostile to soldiers? Restricted (only king enters)? Hostile to
   the king's own side? Empty-throne-as-capture-anvil?
4. **Corner squares** — restricted? Hostile (act as a capturing wall)?
5. **Armed king** — can the king participate in captures, or only soldiers?
6. **Special captures** — shieldwall (edge-row capture of a line), edge-fort exits.
7. **Piece counts & starting layout** per board size (constrained by archaeology, not free).
8. **Repetition / perpetual** rule — draw, or loss for the perpetrator (usually attacker).

A "rule set" is one choice per axis. The space is a few thousand combinations; archaeology
and a sanity filter (no illegal/contradictory combos) prune it hard.

---

## 2. Why self-play, and why *not* vanilla Q-learning

- **No data to imitate.** No move records survive, so this is not supervised/imitation
  learning. It's mechanism evaluation via self-play.
- **Tabular Q-learning is out.** Even 7×7 tafl has an astronomical state space, and it's a
  two-player adversarial game — tabular Q approximates the wrong object.
- **The signal we want is a *balance statistic*, not a policy.** For each rule set we want
  P(attacker win), P(king win), draw rate, mean game length, decisiveness, branching factor,
  comeback frequency. The agent is just the instrument that produces those numbers; it must be
  *strong enough that the imbalance is the game's, not the agent's*.

Algorithm ladder, by board size:

| Board | Approach | Why |
|------|----------|-----|
| **7×7 brandub** | Retrograde/endgame analysis + alpha-beta; aim near-**solved** | Small enough for ground-truth balance, not estimates. **Start here.** |
| **9×9 tablut** | AlphaZero-lite (MCTS + small conv/residual net), self-play | Standard, correct tool for adversarial perfect-info games; reuses MLX. |
| **11×11+** | Same net, more sims; or strong minimax + handcrafted eval over many games | Full AlphaZero may be overkill for a balance estimate. |

DQN/Q-learning-with-approximation is *possible* but strictly worse: it doesn't give the clean
self-play balance signal MCTS does. If the user specifically wants "a Q-learning algo," the
honest mapping is **AlphaZero-style value/policy self-play** — Q-learning's adversarial cousin.

---

## 3. The rule-config schema (the heart of it)

A single dataclass / dict that any candidate rule set instantiates. The engine is one
generic tafl implementation parameterised entirely by this object — no per-variant code.

```python
@dataclass(frozen=True)
class TaflRules:
    board_size: int                 # 7, 9, 11, 13
    start_layout: str               # named layout id, validated against board_size
    # --- piece counts derived from layout, but asserted against archaeology ---
    king_capture: Literal["four_sides", "two_sides", "anvil_throne", "edge_counts"]
    king_armed: bool                # king helps capture?
    king_goal: Literal["corner", "any_edge"]
    throne: ThroneRule              # blocked_to_soldiers, hostile_to_soldiers,
                                    # hostile_to_king_side, empty_is_hostile, re_enterable
    corners: CornerRule             # restricted / hostile-as-wall / normal
    shieldwall: bool                # edge-row line captures
    exit_fort: bool                 # king escapes via edge fort (Copenhagen rule)
    repetition: Literal["draw", "loss_mover", "loss_attacker"]
```

Named reference rule sets to validate the engine against (these are *known* and let us check
the engine reproduces published win-rates before trusting it on unknown combos):

- `brandub_7x7` (common modern reconstruction)
- `tablut_linnaeus` (faithful-as-possible to the 1732 text)
- `tablut_smith_1811` (the mistranslated one — should test as imbalanced; nice sanity check)
- `fetlar_11x11`, `copenhagen_11x11` (Aage Nielsen community, deliberately balanced)

---

## 4. Balance metrics (what makes a rule set "plausible")

Per rule set, from N self-play games between two strong, equally-resourced agents:

- **`win_balance`** — |P(attacker) − P(king)|; lower is better. Target band ≈ 0.45–0.55 each.
- **`decisiveness`** — 1 − draw_rate; a game that's mostly draws under strong play is suspect.
- **`depth`** — mean plies + branching factor; trivially short = shallow = implausible.
- **`comeback_rate`** — fraction of games where the eval lead changes sides late; proxy for
  "strategically alive," distinguishes a real game from a slippery-slope.
- **`agent_sensitivity`** — does win-rate move a lot between weak and strong agents? If a rule
  set only balances for *weak* play, it's not a balanced game (it's a coin flip dressed up).

Output: a ranked table of rule sets. The hypothesis — *the historical rules sit in the
high-balance, high-depth, decisive region, intersected with what archaeology allows.* We
report a region, not a single answer, and say so.

---

## 5. Repo layout (when we build)

```
tafl/
  DESIGN.md              # this file
  rules.py               # TaflRules schema + named reference rule sets
  engine.py              # generic move-gen, captures, terminal tests — fully param by rules
  agents/
    minimax.py           # alpha-beta + eval, the baseline instrument
    mcts.py              # PUCT search
    net_mlx.py           # small residual net (policy+value), MLX — reuses train.py muscle
  selfplay.py            # run N games, collect TaflRules -> metrics
  balance.py             # the metrics in §4, ranked-table output
  solve_brandub.py       # retrograde/endgame for 7x7 ground truth
  tests/                 # engine reproduces published win-rates for named rule sets
scripts/
  tafl_search.py         # CLI: sweep rule-config space, write tafl/results.json
docs/
  (optional) a tafl tab on the Atlas: interactive board + the balance heatmap
```

Reuses from the existing project: MLX net plumbing from `scripts/train.py`, the "tiny model,
Apple Silicon, minutes not days" ethos, and the Atlas/Pages pattern for a visual payoff.

---

## 6. Milestones

1. **Engine + reference rule sets + tests.** ✅ **Done.** Generic engine (`engine.py`),
   `TaflRules` schema + 4 reference sets (`rules.py`), starting layouts (`layouts.py`), and a
   16-test correctness suite (`tests/test_engine.py`, all passing). Reproducing published
   *win-rates* is deferred to milestone 3 (it needs the search agents); milestone 1's contract
   is engine *correctness* — captures, king-capture modes, escapes, shieldwall, terminals.
2. **Brandub near-solve.** ✅ **Done (alpha-beta, not yet a full game-theoretic solve).**
   Negamax + alpha-beta + transposition table (`agents/minimax.py`), heuristic eval
   (`eval.py`), randomised-opening self-play (`selfplay.py`), balance metrics (`balance.py`),
   and a brandub variant sweep (`scripts/tafl_balance.py`). First real balance signal below.
3. **Minimax self-play sweep on 9×9.** ✅ **Done.** Tablut variant sweep across the two
   contested axes (king escape, king capture) in `scripts/tafl_tablut.py`, reusing the
   milestone-2 machinery via `balance.sweep`. Result in §6b — and it is *not* the tidy
   "Linnaeus balanced, Smith broken" story we guessed; both literal readings are imbalanced.
   *Extension (done):* the same sweep on **13×13 great hnefatafl** (`scripts/tafl_hnefatafl13.py`,
   `hnefatafl_13x13` reference + `hnefatafl13` layout). Result in §6c — edge escape breaks on the
   big board, corner escape is needed, and the weak king is again the balanced choice.
4. **AlphaZero-lite (MLX)** for 9×9/11×11 to confirm the minimax ranking holds under stronger
   play (the `agent_sensitivity` check). ✅ **Machinery built (`tafl/az/`); oracle-strength
   training is future work.** Encoding, a small MLX policy+value net, PUCT MCTS, a self-play
   training loop, and an `AZAgent` that drops into the existing harness — all tested. The smoke
   run learns (loss falls, MCTS solves tactics) but is not yet strong enough to re-run the
   sweeps; see §9.
5. **Write-up + optional Atlas tab.** The deliverable is a *report on plausible rule regions*,
   citing Ludii/DLP and Aage Nielsen, not a claim to have "found the rules."

---

## 6a. First balance signal (milestone 2, brandub 7×7)

Alpha-beta vs. alpha-beta, depth 3, 30 randomised-opening games per variant:

| brandub variant | attacker | defender | draw | imbalance | decisive |
|---|---|---|---|---|---|
| four_sides + hostile throne | 0% | 63% | 37% | 63% | 63% |
| four_sides + plain throne | 0% | 63% | 37% | 63% | 63% |
| edge_counts (wall captures) | 3% | 60% | 37% | 57% | 63% |
| **two_sides (king like a man)** | **43%** | **47%** | **10%** | **3%** | **90%** |

The contested mechanic is exactly the king-capture rule, and it dominates balance. A **strong
king** (taken only on four sides) makes 7×7 a near-guaranteed defender win/draw — attackers
essentially can't win. A **weak king** (captured custodially like an ordinary man, the
Linnaeus/Smith reading) gives a near-50/50, highly decisive game. Sensitivity backs this up:
under stronger search the two_sides line *converges toward parity* (def 30%→45% from depth 1→3)
— the fingerprint of a real equilibrium — while the four-sides lines stay lopsided.

This matches the historical debate (small boards need a weak king to be playable) and, more to
the point, validates the whole premise: rule choices produce *measurable, separable* balance
signatures. **Caveat:** depth 3 is a modest oracle and the encircling attacker is the harder
side to search, so "0% attacker" partly reflects search difficulty; firming it up needs the
deeper search / AlphaZero-lite of later milestones (and a true brandub solve as ground truth).

---

## 6b. Tablut signal (milestone 3, 9×9 Linnaeus vs. Smith)

Alpha-beta self-play, 24 randomised-opening games/variant, at depth 1 and depth 3:

| tablut variant | atk@3 | **def@3** | draw@3 | def@1 → def@3 |
|---|---|---|---|---|
| Linnaeus: edge + 4-side king | 0% | **83%** | 17% | 92% → 83% |
| Smith 1811: edge + weak king | 62% | **21%** | 17% | 54% → 21% |
| edge + edge-counts king | 0% | **83%** | 17% | 92% → 83% |
| corner + 4-side king | 0% | **58%** | 42% | 83% → 58% |
| corner + weak king | 79% | **12%** | 8% | 33% → 12% |

The clean hypothesis (Linnaeus balanced, Smith broken) is **wrong**, and that's the interesting
part. On 9×9 *neither literal reading is balanced*: the faithful four-side king is heavily
king-favoured (83% def), while the weak "king like a man" reading flips hard to the attackers.
The balanced point sits **between** the two historical readings — which is exactly why
real reconstructors (Aage Nielsen et al.) kept adjusting tablut rather than playing it as
recorded. The model reproduces the *reason the reconstruction problem exists*.

**Direction of travel (sensitivity).** Deeper search helps the attacker in *every* variant
(def win-rate falls d1→d3 across the board), because building an encirclement needs more
look-ahead than running for the edge. So: the weak-king readings are *robustly* too generous
to attackers (they only get worse for the king under stronger play), while the four-side
king readings are king-favoured but *closing* — `corner + 4-side king` drops 83%→58% def and
is the fastest-converging candidate for a balanced region. The likely historical answer is a
strong-ish king plus a balancing convention the bare text omits (e.g. extra restrictions on
the throne, or the attacker's first-move tempo), not either literal extreme.

**Caveat, louder at 9×9 than 7×7:** depth-3 attackers play below the level needed to convert
an encirclement, so every defender win-rate here is an *upper bound* that will compress under
the deeper / AlphaZero-lite oracle of milestone 4. We can already trust the *ordering* and the
*direction*; we cannot yet name the single balanced rule set.

---

## 6c. Big-board signal (13×13 great hnefatafl, 24+12+1)

Alpha-beta self-play, depth 3, 10 randomised-opening games/variant. The 13×13 setup is itself
contested — this uses the common 24+12+1 "great cross" reading (a denser 32+16+1 also circulates).

| 13×13 variant | atk | def | draw | imbalance | decisive |
|---|---|---|---|---|---|
| corner + 4-side king (ref) | 0% | 40% | 60% | 40% | 40% |
| **corner + weak king** | 40% | 30% | 30% | **10%** | 70% |
| corner + edge-counts king | 0% | 30% | 70% | 30% | 30% |
| edge escape + 4-side king | 0% | 80% | 20% | 80% | 80% |

Two things carry over and one is new:

1. **Edge escape does not scale.** On 13×13 "reach any edge" is trivial for the king (def 80%,
   games end in ~19 plies) — the big board has too much rim. This is concrete support for the
   historical pattern that *large* hnefatafl uses **corner** escape while *small* tablut (9×9)
   could use edge escape: the goal has to get harder as the board grows or the king just walks out.
2. **Among corner-escape variants the weak king is again the most balanced and decisive**
   (10% imbalance, 70% decisive), echoing 7×7 and 9×9. The board got 3.4× bigger and the same
   mechanic still controls balance.
3. **The strong-king variants are draw-heavy (60–70%)** — but this is the milestone-4 caveat at
   its loudest. Coordinating 24 attackers into an encirclement on a 13×13 board is *far* beyond a
   3-ply horizon, so the strong-king "can't convert" reading is unreliable: those draws are the
   weak oracle failing, not necessarily the rules. 13×13 numbers are the softest in the project
   and should not be trusted past their ordering until the stronger search arrives.

Cross-board summary: the king-capture rule is the master balance dial at every size, and bigger
boards push toward harder escape (corner) and/or an easier-to-take king to stay playable.

---

## 7. Honest limitations (state these in any write-up)

- Balance is *necessary, not sufficient* — a balanced rule set isn't proven historical, just
  not ruled out. We narrow, we don't decide.
- Agent strength caps confidence: a wrong "this side always wins" might be our agent's weakness.
  Hence the brandub ground-truth anchor and `agent_sensitivity`.
- "Fun/depth" metrics are proxies; medieval players' taste isn't ours.
- Prior art exists and should be credited up front (Ludii / Digital Ludeme Project; Aage
  Nielsen's tafl reconstructions). We'd be replicating their *method* on a focused slice, with
  a clean MLX implementation — not inventing it.
```

---

## 8. Rule fidelity vs. the World Tafl Federation / Aage Nielsen

Audit of our reference rule sets against the authoritative modern rules at
[aagenielsen.dk/tafl_rules.php](https://aagenielsen.dk/tafl_rules.php) (Fetlar, Copenhagen,
Historical Hnefatafl/Tablut 9×9, and the WTF Brandubh leaflet). Differences found and fixed:

| # | our rule set (before) | authoritative rule | resolution |
|---|---|---|---|
| 1 | **brandub**: strong king (four sides) | WTF Brandubh: **weak king** — taken like an ordinary man (two sides) *except* a full surround on the throne; throne never hostile to the king but always to the attackers; the repeating side loses | **fixed** — new `two_sides_strong_throne` capture mode; `brandub_7x7` rebuilt faithfully |
| 2 | **fetlar**: `shieldwall=True` | Fetlar has **no shieldwall** (it's a Copenhagen rule); Fetlar repetitions are draws | **fixed** — shieldwall off; added a faithful `copenhagen_11x11` (shieldwall on, attacker loses on repetition) |
| 3 | **tablut**: only `linnaeus` (four-side) and `smith` (two-side) | WTF Historical Hnefatafl: weak king, but full surround on the throne and three sides beside it; edge escape; attacker loses on repetition | **added** `tablut_historical_9x9` — the authoritative reading, which sits *between* our two bracket readings |
| 4 | throne hostile only when empty (both sides) | throne hostile to attackers **always**, to defenders only when empty | **fixed** in `_hostile_square_for` (practically equivalent — a king on the throne already anchors defender captures — but now explicit) |
| 5 | repetition default = draw everywhere | Tablut/Copenhagen: loss for attackers; Brandubh: loss for the repeater; Fetlar: draw | **fixed** — set per reference |
| 6 | Copenhagen **exit-fort** win | a king who reaches the edge in an unbreakable fort wins | **not yet implemented** — engine still guards `exit_fort`; `copenhagen_11x11` leaves it off, documented |
| 7 | **hnefatafl13** (our 13×13) | **not in the WTF/Aage canon at all** — no 13×13 on the site | left as-is, explicitly flagged as a Ludii/other-source, contested setup |

**A nice convergence.** The authoritative rules independently land in the *intermediate* region our
search flagged as balanced. After the fix, at depth 3:

| rule set | atk | def | decisive | vs. our brackets |
|---|---|---|---|---|
| brandub (WTF weak/strong-throne) | 33% | 67% | 100% | between strong (0/63, draws) and pure-weak (43/47) |
| tablut_historical (WTF) | 17% | 83% | 100% | more decisive than `linnaeus` (0/83, draws), still king-leaning at d3 |

Both authoritative rules trade the draw-heavy, attacker-can't-win character of a pure strong king
for a fully *decisive* game with the king modestly favoured — exactly the kind of middle ground the
milestone-2/3 sweeps pointed to. The depth-3 caveat still applies (attackers underplayed, so the
king-leaning numbers are upper bounds), so this is corroboration of the *method*, not a final verdict.

---

## 9. Milestone 4 — AlphaZero-lite (machinery built; oracle-strength is future work)

The recurring caveat across §6a–§6c is that fixed-depth alpha-beta underplays the encircling
attacker, especially as boards grow. The fix is a search that *learns* to plan. `tafl/az/` builds
the full AlphaZero-style stack, in the project's MLX idiom:

- **`encoding.py`** — state → `(n, n, 6)` planes (attacker/defender/king/throne/corner/side); moves
  → a "queen-ray" action space `n·n·4·(n-1)` with exact encode/decode round-trips.
- **`net.py`** — a small residual conv net (MLX, channels-last) with policy + value heads.
- **`mcts.py`** — PUCT MCTS using net priors + value, no rollouts; terminal leaves use the true
  result; Dirichlet root noise for self-play.
- **`selfplay.py`** — games → (planes, visit-distribution π, outcome z) training samples.
- **`agent.py`** — `AZAgent.choose(game, state)`, interface-compatible with the alpha-beta/random
  agents, so it drops straight into `tafl.selfplay.play_game` and the balance harness.
- **`scripts/tafl_az_train.py`** — the self-play training loop.

**Status — honest.** The loop is validated end-to-end (8 tests: encode/decode, masking, net
shapes, MCTS solving mate-in-1 even untrained, self-play sample validity, a training step that
overfits a fixed batch). A short brandub smoke run shows the **training loss falling steadily
(≈6.9 → 3.3 over 8 rounds)** and the agent beating random more often than not. But at 175k params /
32 sims / 8 rounds it is a *weak, draw-prone* player — self-play is draw-heavy and the win-rate vs.
baselines is noisy, not climbing convincingly. **It is not yet the trustworthy oracle**, so the
§6 sweeps have *not* been re-run with it.

**What "future work" concretely means here:** a bigger net and many more self-play games / sims /
rounds, plus batched MCTS leaf-evaluation (the current per-leaf net call is the speed bottleneck) so
9×9–13×13 self-play is affordable. Only once `AZAgent` clearly beats `AlphaBetaAgent(depth≥3)` should
we trust it to re-run the balance sweeps and finally separate "the rules are imbalanced" from "our
search was too weak" — the question every earlier milestone deferred to this one.

