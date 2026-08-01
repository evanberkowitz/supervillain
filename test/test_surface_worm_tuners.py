import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.tuners import (
    SectorWeightTuner, PairUmbrellaTuner)
from supervillain.generator.no_intersection.surface_worm.weights import (
    SectorWeights, PairUmbrella)

def _S():
    return supervillain.action.NoIntersections(Lattice(4, 4), kappa=0.2)

def test_sector_weight_tuner_smoke():
    t = SectorWeightTuner(_S(), intersectionFugacity=0.3, cap=8,
                          iterations=3, ticks=300, stride=50, seed=1)
    w = t.tune()
    assert isinstance(w, SectorWeights) and w.cap == 8
    assert len(t.history) == 3 and 'flatness' in t.history[0]

def test_pair_umbrella_tuner_smoke():
    w = SectorWeights.fugacity(0.2, cap=8)
    t = PairUmbrellaTuner(_S(), sectorWeights=w, intersectionFugacity=0.3,
                          targetFraction=0.5, pCob=0.5,
                          iterations=2, ticks=300, stride=50, equilibrate=5_000, seed=2)
    u = t.tune()
    assert isinstance(u, PairUmbrella) and u.N == 4
    assert len(t.history) == 2

def test_pair_umbrella_tuner_converged_branch_rebuilds_from_updated_shape():
    # Regression test for a review finding: on convergence, tune() must return
    # table(shape, offset) built from the JUST-UPDATED shape, not the `umbrella`
    # that was actually measured this iteration (built from the PRE-update
    # shape). The 2-iteration smoke above never converges, so it can't catch a
    # regression here -- this test forces convergence on iteration 0 by faking
    # `measure` (skips the expensive gas run entirely) and `update` (returns a
    # table with a distinctive marker), so the two candidate return values are
    # unmistakably different and we can assert which one tune() actually picked.
    S = _S()
    N = S.Lattice.N
    w = SectorWeights.fugacity(0.2, cap=8)
    t = PairUmbrellaTuner(S, sectorWeights=w, intersectionFugacity=0.3,
                          targetFraction=0.5, pCob=0.5,
                          iterations=5, ticks=10, stride=5, equilibrate=10, seed=3)
    t.minCount = 1     # trivially satisfied by the faked harvest below
    shells = t.achievable_shells(N)

    fakeHarvest = {
        'PairSeparationTicks': np.zeros(N ** 2 + 1),
        'Ticks': 100,
        'VacuumTicks': 50,
        'VacuumReturns': t.excursionFloor + 1,   # not collapsed
        'MaxPairSeparationSquared': int(shells.max()),
    }
    fakeHarvest['PairSeparationTicks'][shells] = 5   # every shell hit, well above minCount=1

    measured = []
    def fake_measure(umbrella, seed, ticks, stride, equilibrate):
        measured.append(umbrella)
        return fakeHarvest
    # Non-constant, so mean-zeroing it (as tune() does to every shape) still leaves
    # a distinctive signal -- a constant marker would mean-zero to all zeros, which
    # is indistinguishable from the pre-update (also all-zero, PairUmbrella.off)
    # shape and would defeat the "NOT measured[0]" check below.
    marker = -777.0 * np.arange(1, N ** 2 + 2, dtype=float)
    def fake_update(umbrella, occupancy, totalTicks, shellsArg):
        return PairUmbrella(marker, N)
    t.measure = fake_measure
    t.update = fake_update

    result = t.tune()

    assert len(t.history) == 1
    assert t.history[0]['converged']
    offset = t.history[0]['offset']
    expectedShape = marker.copy()
    expectedShape[shells] -= expectedShape[shells].mean()
    # The returned table carries the marker (i.e. was rebuilt from the
    # just-updated shape at the bisector's offset)...
    assert np.allclose(result.logWeight[shells], expectedShape[shells] + offset)
    assert np.allclose(result.logWeight[np.setdiff1d(np.arange(N ** 2 + 1), shells)], 0.0)
    # ...and is NOT the table that was actually measured this iteration (the bug
    # this test guards against: returning `umbrella` instead of the rebuild).
    assert not np.allclose(result.logWeight, measured[0].logWeight)


# ---- TransportTuner -------------------------------------------------------
from supervillain.generator.no_intersection.surface_worm.tuners import TransportTuner


def test_transport_tuner_smoke():
    # measureTicks bumped from the brief's 200 to 8_000 (still ~3s wall time).
    # `_stage`'s `returns` counter gates on the simple D==0 and Q==0 -- no periods
    # check -- and at openSurfaceFugacity=0.2 (4x the emit tests' healthy 0.05,
    # test_surface_worm_emit.py) combined with intersectionFugacity=0.3, Q sits at a
    # nonzero equilibrium (measured mean ~13-15, minimum observed 2, over thousands
    # of ticks) while D=0 only a minority of the time (measured P(D=0) ~0.30 at this
    # openSurfaceFugacity, vs 1.0 at 0.05) -- so the simple joint event D=0 AND Q=0
    # is rare on its own, and 200 ticks is not enough to see one reliably for every
    # seed. Not a correctness bug -- see the tuner's docstring and task-1-report.md
    # for the measurement.
    t = TransportTuner(_S(), intersectionFugacity=0.3, openSurfaceFugacity=0.2,
                       targetFraction=0.0, pCob=0.5,
                       cap0=6, capStep=4, capMax=10, stageSeeds=1, retries=0,
                       returnFloor=1, tuneIterations=2, tuneTicks=120,
                       measureTicks=8_000, stride=40, equilibrate=4_000, seed=3)
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
    # measureTicks bumped from 200 to 8_000 -- same D=0-and-Q=0-is-rare budget
    # reason as test_transport_tuner_smoke above; tune() must succeed here too
    # before score_flips() has a design to validate.
    t = TransportTuner(_S(), intersectionFugacity=0.3, openSurfaceFugacity=0.2,
                       targetFraction=0.0, pCob=0.5,
                       cap0=6, capStep=4, capMax=6, stageSeeds=1, retries=0,
                       returnFloor=1, tuneIterations=2, tuneTicks=120,
                       measureTicks=8_000, stride=40, equilibrate=4_000, seed=4)
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        t.score_flips(budget=100)          # before tune()
    t.tune()
    flips, moves, charged = t.score_flips(budget=4_000)
    assert isinstance(flips, int) and moves >= 4_000 and 0.0 <= charged <= 1.0


def test_transport_tuner_rejects_negative_retries():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        TransportTuner(_S(), intersectionFugacity=0.3, openSurfaceFugacity=0.2,
                       retries=-1, seed=1)
