# Transport tuner migration — design

2026-08-01. Migrate the transport tuner — the third SWG tuner, developed in
`no-intersections/j-vacuum-2026-07-31/transport_tuner.py` (v4) — into
`supervillain/generator/no_intersection/surface_worm/tuners.py` beside
`SectorWeightTuner` and `PairUmbrellaTuner`, on `feature/surface-worm`.

Evidence of success (the precondition for this migration): j-vacuum NOTES
finding 14 — the v4 design scores **4.20 J flips/Mmove warm-started, above
the cap-16 flatness baseline of 3.52** — with every recipe ingredient earned
by a documented failure (v1: pressure-only stopping fooled by a bad flatten;
v2: targeted tuning degrades the learned histogram; findings 12/13: single-
seed scores are basin roulette; finding 13: cold scores are nucleation
roulette).

## What it does

Finding 7: every J-changing excursion carries charge through a wide-open
surface and hits the hard w(D) cap — the cap chosen for Θ-measurement health
throttles exactly the excursions that transport. This tuner chooses the cap
by the transport physics: grow the cap stage by stage, flattening w(D) at
each stage, measuring top-of-range pressure, the transport-corner dwell
(D ≥ cornerFrac·cap with Q > 0 — the finding-7 proxy), and vacuum returns
(the Θ-health constraint); keep the best stage by corner dwell subject to the
return floor; stop on constraint-binds, corner-dwell turnover (two
consecutive caps below half the best), or capMax.

## Class

`TransportTuner(S, intersectionFugacity, openSurfaceFugacity=0.09,
targetFraction=0.8, pCob=0.2, pairUmbrella=None, cap0=12, capStep=6,
capMax=64, pressureFrac=0.15, pressureEps=0.02, cornerFrac=0.75,
returnFloor=50, stageSeeds=3, retries=2, tuneTargetFraction=0.0,
tuneIterations=20, tuneTicks=3000, measureTicks=4000, stride=200,
equilibrate=2_000_000, damping=0.8, seed=None)`

- `.tune() -> SectorWeights` — the v4 loop, ported by content from the
  notebook source's `stage()` + `__main__` decision logic:
  - per cap: flatten via an internal `SectorWeightTuner` constructed with
    `targetFraction=tuneTargetFraction` (untargeted by default — v2's
    lesson, stated in the docstring);
  - stage measurement with `stageSeeds` independent equilibrations (library
    gas, `measure=False`, `FState`, `sweep`), aggregated as MEAN corner
    dwell / MINIMUM returns (finding-12 lesson, stated in the docstring);
    per-seed seeds derived as `seed + cap + 271*s`, per-attempt tune seeds
    offset by `7919*attempt`, matching the source;
  - retries (up to `retries` extra tune attempts per cap) only when corner
    dwell falls below half the global best;
  - keep-best-by-corner among stages meeting the return floor; two-strike
    turnover stop; constraint-binds verdict when min-returns < returnFloor;
    capMax exhaustion verdict.
  - `.history` — list of per-stage diag dicts (`cap`, `pressure`,
    `returns` (min), `cornerFrac` (mean), `cornerSpread`, `vacuumFrac`,
    `topFloor`); `.verdict` — the stopping-reason string; `.best` — the
    chosen stage's diag dict. Raises `RuntimeError` (not SystemExit) when
    no cap satisfies the return floor.
  - The chosen cap is carried by the returned table's length (`.cap`), the
    same convention as everywhere else; no separate cap return.
- `.score_flips(warmStartConfiguration=None, budget=40_000_000) ->
  (flips, moves, chargedFraction)` — the validation arm: a gas at the
  chosen design; warm-started via `FState.from_configuration(S,
  warmStartConfiguration)` when given (cold otherwise, with the
  nucleation-roulette caveat in the docstring — cold scores measure the
  basin draw); advances in 20-move batches; at each `legal_vacuum` visit
  reads J via `FState.intersection_winding()` (replacing the notebook's
  primitive_2form + reconstruct_n + IntersectionWinding.Villain — same
  integers, audited) and counts changes. Raises `RuntimeError` if called
  before `tune()`.
- Docstrings carry the earned history and the known limitation: large-cap
  stages need tune-budget scaling (v4 chose cap 12 while brute-force cap-24
  flattening reached 5.40 flips/Mmove); budget scaling is the next lever,
  not decision logic.

**Not ported:** the design-npz writer and the CLI — notebook driver
concerns; `SectorWeights` is `ReadWriteable`.

**Export:** `TransportTuner` joins the public re-export in
`no_intersection/__init__.py` and the subpackage `__init__.py` (the spec'd
"third tuner slot").

## Tests (`test/test_surface_worm_tuners.py`)

1. `tune()` smoke at tiny budgets (cap0=6, capStep=4, capMax=10,
   stageSeeds=1, tuneIterations=2, tuneTicks/measureTicks a few hundred,
   equilibrate small): returns `SectorWeights`, `.history` non-empty with
   monotone cap trajectory, `.verdict` set, `.best` in `.history`.
2. Verdict-logic unit test with a monkeypatched `_stage`: drive all three
   exits (constraint-binds, turnover after two weak caps, capMax) and the
   retry path (dead corner triggers extra attempts; healthy does not),
   asserting the chosen table is the best-by-corner among floor-passing
   stages.
3. `score_flips` smoke at minimal budget: return contract
   `(int, int, float)`, works cold and with a warm configuration (reuse a
   configuration emitted by a tiny gas run), raises before `tune()`.

## Process

Same SDD discipline as the main migration: one task (brief = this spec),
implementer + spec/quality review + fix loop, on `feature/surface-worm`;
test invocation `NUMBA_CACHE_DIR=/tmp/numba_sessions/swg-migrate
.venv/bin/python -m pytest test/test_surface_worm_tuners.py -v` plus the
full surface_worm suite; commit trailer per house rules. Port source
read-only: `/Users/evanberkowitz/physics/supervillain/no-intersections/
j-vacuum-2026-07-31/transport_tuner.py`.
