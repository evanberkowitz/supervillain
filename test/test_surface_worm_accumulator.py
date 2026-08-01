import numpy as np
import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.accumulator import CorrelatorAccumulator
from supervillain.generator.no_intersection.surface_worm.state import FState

def _S(kappa=0.2):
    return supervillain.action.NoIntersections(Lattice(4, 4), kappa=kappa)

def test_vacuum_and_class_classification():
    acc = CorrelatorAccumulator(4, 0.3)
    st = FState(_S())                      # legal vacuum
    acc.tick(st)
    F = np.zeros((6,)+(4,)*4, dtype=np.int64); F[0, 0, 0, :, :] = 1
    acc.tick(FState(_S(), F))              # closed, q=0, class 1: EXCLUDED
    h = acc.harvest()
    assert h['VacuumTicks'] == 1 and h['ClosedTicks'] == 1
    assert h['NontrivialClassTicks'] == 1

def test_pair_bin_divides_umbrella_out():
    from supervillain.generator.no_intersection.surface_worm.weights import PairUmbrella
    acc = CorrelatorAccumulator(4, 0.3)
    u = PairUmbrella(np.log(np.full(17, 2.0)), 4)   # w2 = 2 everywhere
    acc.pairUmbrella = u
    st = FState(_S())
    # manufacture a +/-1 pair state at separation 1 by hand.  FState.counts is a
    # dict {'D': int, 'Q': int} (not an indexable [D, Q] array like the audit's
    # cfg), so the brief's `st.counts[1] = 2` becomes `st.counts['Q'] = 2` --
    # the assertions below are otherwise identical to the brief.
    st.q[0, 0, 0, 0] = 1; st.q[1, 0, 0, 0] = -1
    st.counts['Q'] = 2
    st.chargeSites = {(0, 0, 0, 0): 1, (1, 0, 0, 0): -1}
    st.absoluteCharge = 2; st.squaredCharge = 2
    acc.tick(st)
    h = acc.harvest()
    dx = tuple(np.argwhere(h['Theta_Theta'] != 0)[0])
    V = 4 ** 4
    assert np.isclose(h['Theta_Theta'][dx], (1 / 2.0) / (V * 0.3 ** 2))

def test_harvest_resets():
    acc = CorrelatorAccumulator(4, 0.3)
    acc.tick(FState(_S()))
    acc.harvest()
    assert acc.harvest()['VacuumTicks'] == 0
