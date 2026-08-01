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
