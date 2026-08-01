# Transport Tuner Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the v4 transport tuner into `surface_worm/tuners.py` as `TransportTuner`, per the spec `docs/superpowers/specs/2026-08-01-transport-tuner-design.md`.

**Architecture:** One new class beside the two existing tuners; the flatten step delegates to `SectorWeightTuner`; stage measurement drives a `measure=False` library gas over `FState`s; the scoring arm reads J via `FState.intersection_winding()` at `legal_vacuum` visits.

**Tech Stack:** numpy, the existing surface_worm modules, pytest.

## Global Constraints

- Branch `feature/surface-worm` in `/Users/evanberkowitz/physics/supervillain/library`.
- Port source (read-only, port by content): `/Users/evanberkowitz/physics/supervillain/no-intersections/j-vacuum-2026-07-31/transport_tuner.py` — `stage()` (lines 51–101), `score_flips()` (104–153), and the `__main__` decision loop (220–266). The spec (its "Class" section) is the binding interface; the source is the binding algorithm.
- Seed derivations verbatim from the source: stage-seed `seed + cap + 271*s`; per-attempt tune-seed offset `7919*attempt`.
- Test runner: `NUMBA_CACHE_DIR=/tmp/numba_sessions/swg-migrate .venv/bin/python -m pytest test/test_surface_worm_tuners.py -v`, then the full `-k surface_worm` suite.
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- `seed=None` must remain valid (derivations then start from a drawn integer: `seed = int(np.random.default_rng().integers(2**31))` when None, once, in the constructor).

---

### Task 1: `TransportTuner`

**Files:**
- Modify: `supervillain/generator/no_intersection/surface_worm/tuners.py` (append the class)
- Modify: `supervillain/generator/no_intersection/surface_worm/__init__.py` (add `TransportTuner` to the tuners import/exports)
- Modify: `supervillain/generator/no_intersection/__init__.py` (add `TransportTuner` to the re-export line)
- Test: `test/test_surface_worm_tuners.py` (append)

**Interfaces:**
- Consumes: `SectorWeightTuner(S, intersectionFugacity, cap, openSurfaceFugacity=..., iterations=..., ticks=..., stride=..., damping=..., pCob=..., targetFraction=..., seed=...)` with `.tune() -> SectorWeights`; `SurfaceWormGas(S, sectorWeights=..., intersectionFugacity=..., targetFraction=..., pairUmbrella=..., measure=False, seed=..., rng=...)` with `.sweep(state, nmoves, pCob=...)`; `FState(S)` / `FState.from_configuration(S, configuration)` with `.D`, `.Q`, `.legal_vacuum`, `.intersection_winding()`; `PairUmbrella.off(N)`.
- Produces: `TransportTuner` with the spec's constructor signature, `.tune() -> SectorWeights`, `.history` (list of dicts with keys `cap, pressure, returns, cornerFrac, cornerSpread, vacuumFrac, topFloor`), `.verdict` (str), `.best` (dict), `.score_flips(warmStartConfiguration=None, budget=40_000_000) -> (int, int, float)`.

- [ ] **Step 1: Write the failing tests** (append to `test/test_surface_worm_tuners.py`)

```python
# ---- TransportTuner -------------------------------------------------------
from supervillain.generator.no_intersection.surface_worm.tuners import TransportTuner


def test_transport_tuner_smoke():
    t = TransportTuner(_S(), intersectionFugacity=0.3, openSurfaceFugacity=0.2,
                       targetFraction=0.0, pCob=0.5,
                       cap0=6, capStep=4, capMax=10, stageSeeds=1, retries=0,
                       returnFloor=1, tuneIterations=2, tuneTicks=120,
                       measureTicks=200, stride=40, equilibrate=4_000, seed=3)
    w = t.tune()
    from supervillain.generator.no_intersection.surface_worm.weights import SectorWeights
    assert isinstance(w, SectorWeights)
    assert t.history and t.verdict and t.best in t.history
    caps = [h['cap'] for h in t.history]
    assert caps == sorted(caps)
    for key in ('cap', 'pressure', 'returns', 'cornerFrac', 'cornerSpread',
                'vacuumFrac', 'topFloor'):
        assert key in t.history[0]


def test_transport_tuner_verdict_logic(monkeypatch):
    # Drive the decision loop with scripted stages: all three exits + retry path.
    from supervillain.generator.no_intersection.surface_worm.weights import SectorWeights

    def scripted(results):
        t = TransportTuner(_S(), intersectionFugacity=0.3, openSurfaceFugacity=0.2,
                           cap0=6, capStep=2, capMax=12, stageSeeds=1, retries=1,
                           returnFloor=10, seed=1)
        calls = {'n': 0}
        def fake_stage(cap, attempt):
            r = results[min(calls['n'], len(results) - 1)]; calls['n'] += 1
            diag = dict(cap=cap, pressure=0.1, returns=r['returns'],
                        cornerFrac=r['corner'], cornerSpread=0.0,
                        vacuumFrac=0.3, topFloor=1)
            return SectorWeights.fugacity(0.2, cap=cap), diag
        monkeypatch.setattr(t, '_stage', fake_stage)
        return t, calls

    # constraint binds: second cap fails the floor -> verdict mentions floor,
    # chosen table is the first (best floor-passing) stage
    t, _ = scripted([{'returns': 50, 'corner': 0.10},
                     {'returns': 3,  'corner': 0.90}])
    w = t.tune()
    assert 'floor' in t.verdict or 'constraint' in t.verdict
    assert t.best['cap'] == 6 and w.cap == 6

    # turnover: two consecutive caps below half the best
    t, _ = scripted([{'returns': 50, 'corner': 0.20},
                     {'returns': 50, 'corner': 0.05},
                     {'returns': 50, 'corner': 0.05}])
    t.tune()
    assert 'turnover' in t.verdict
    assert t.best['cap'] == 6

    # capMax exhaustion
    t, _ = scripted([{'returns': 50, 'corner': 0.10},
                     {'returns': 50, 'corner': 0.11},
                     {'returns': 50, 'corner': 0.12},
                     {'returns': 50, 'corner': 0.13}])
    t.tune()
    assert 'capMax' in t.verdict
    assert t.best['cap'] == 12

    # retry path: a dead-corner stage triggers a second attempt at the same cap
    t, calls = scripted([{'returns': 50, 'corner': 0.20},   # cap 6, healthy best
                         {'returns': 50, 'corner': 0.0},    # cap 8 attempt 0: dead
                         {'returns': 50, 'corner': 0.15},   # cap 8 attempt 1 (retry)
                         {'returns': 50, 'corner': 0.0},    # cap 10 attempt 0: dead
                         {'returns': 50, 'corner': 0.0},    # cap 10 attempt 1: dead
                         {'returns': 50, 'corner': 0.0},    # cap 12 attempt 0
                         {'returns': 50, 'corner': 0.0}])   # cap 12 attempt 1
    t.tune()
    assert calls['n'] == 7          # retries actually fired
    assert t.best['cap'] == 6

    # no cap satisfies the floor -> RuntimeError
    t, _ = scripted([{'returns': 1, 'corner': 0.10}])
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        t.tune()


def test_transport_tuner_score_flips():
    t = TransportTuner(_S(), intersectionFugacity=0.3, openSurfaceFugacity=0.2,
                       targetFraction=0.0, pCob=0.5,
                       cap0=6, capStep=4, capMax=6, stageSeeds=1, retries=0,
                       returnFloor=1, tuneIterations=2, tuneTicks=120,
                       measureTicks=200, stride=40, equilibrate=4_000, seed=4)
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        t.score_flips(budget=100)          # before tune()
    t.tune()
    flips, moves, charged = t.score_flips(budget=4_000)
    assert isinstance(flips, int) and moves >= 4_000 and 0.0 <= charged <= 1.0
```

- [ ] **Step 2: Run to verify failure** — `NUMBA_CACHE_DIR=/tmp/numba_sessions/swg-migrate .venv/bin/python -m pytest test/test_surface_worm_tuners.py -v`; expected: ImportError on `TransportTuner`.
- [ ] **Step 3: Implement `TransportTuner`** in `tuners.py`, per the spec's Class section verbatim (constructor signature, `.tune()`, `.history`/`.verdict`/`.best`, `.score_flips`). Structure it as: `_stage(cap, attempt)` (private; the source's `stage()` — internal `SectorWeightTuner` with `targetFraction=self.tuneTargetFraction` and seed `self.seed + cap + 7919*attempt`; then `stageSeeds` measurement chains, gas seeds `self.seed + cap + 271*s + 7919*attempt`, each `FState(S)` + `sweep(equilibrate)` then `measureTicks` × `sweep(stride)` reading `state.D`/`state.Q`; aggregate mean-corner/min-returns exactly as source lines 91–101); `tune()` (the source's decision loop, lines 220–266, with `RuntimeError` replacing `SystemExit` and `self.history/self.verdict/self.best` set); `score_flips()` (source lines 104–153 with `FState.intersection_winding()` at `state.legal_vacuum` replacing the reconstruction block, warm start via `FState.from_configuration` when `warmStartConfiguration` is not None, and the returned tuple `(flips, moves, chargedFraction)`). Docstrings per the spec's earned-history requirements. Add `TransportTuner` to both `__init__.py` export sites.
- [ ] **Step 4: Run to verify pass**, then the full `-k surface_worm` suite (no regressions) and `test/test_surface_worm_api.py` (exports).
- [ ] **Step 5: Commit** — `surface-worm: TransportTuner (cap-growing transport tuner, v4 recipe)` + trailer.
