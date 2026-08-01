import numpy as np
import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.accumulator import CorrelatorAccumulator
from supervillain.generator.no_intersection.surface_worm.gas import SurfaceWormGas
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

def test_measure_true_through_generator_protocol():
    # Integration gate for the gas wiring (not just the accumulator in isolation):
    # a measuring gas must drive Ensemble.generate end to end, with harvest keys
    # landing as inline observables and no KeyError from Configurations.__setitem__
    # on an unregistered field.  Vacuum-healthy point reused from Task 8's helper
    # (test_surface_worm_sweep.py's _gas defaults): kappa=0.2 with a strongly
    # suppressive bare fugacity keeps the chain hugging the closed shell.
    N = 4
    S = supervillain.action.NoIntersections(Lattice(N, N), kappa=0.2)
    g = SurfaceWormGas(S, openSurfaceFugacity=0.05, intersectionFugacity=0.1,
                       ticksPerStep=200, stride=100, measure=True,
                       seed=9, rng=np.random.default_rng(9))
    # Shared object, not a copy: the accumulator's per-bin division by w_2 must
    # undo exactly the bias the acceptance introduced (weights.py's warning).
    assert g.accumulator.pairUmbrella is g.pairUmbrella

    g.equilibrate(20_000)
    e = supervillain.Ensemble(S).generate(3, g, start='cold')

    Ticks = np.asarray(e.Ticks)
    VacuumTicks = np.asarray(e.VacuumTicks)
    ThetaTheta = np.asarray(e.Theta_Theta)
    NontrivialClassTicks = np.asarray(e.NontrivialClassTicks)
    assert Ticks.shape == (3,)
    assert VacuumTicks.shape == (3,)
    assert ThetaTheta.shape == (3, N, N, N, N)
    assert NontrivialClassTicks.shape == (3,)
    # Every emitted row actually ticked the accumulator ticksPerStep times or more
    # (the legal-vacuum wait loop only ever adds ticks, never removes them).
    assert (Ticks >= 200).all()
