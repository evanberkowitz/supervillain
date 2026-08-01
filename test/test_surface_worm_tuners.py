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
