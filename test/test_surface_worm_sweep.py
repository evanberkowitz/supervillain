import numpy as np
import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.gas import SurfaceWormGas
from supervillain.generator.no_intersection.surface_worm.state import FState

def test_sweep_invariants_through_charged_excursions():
    S = supervillain.action.NoIntersections(Lattice(4, 4), kappa=0.03)
    g = SurfaceWormGas(S, openSurfaceFugacity=0.09, intersectionFugacity=0.1,
                       targetFraction=0.8, seed=12, measure=False)
    st = FState(S)
    g.sweep(st, 200_000)
    st.refresh_charge()
    assert st.check()          # periods, winding, D, Q, dF, q, G all match recompute

def test_sweep_and_reference_agree_statistically():
    # same target, two kernels: mean D and mean Q agree loosely over short runs
    S = supervillain.action.NoIntersections(Lattice(4, 4), kappa=0.2)
    # rng= must be passed explicitly (not just seed=): SurfaceWormGas.__init__ only
    # threads seed into the compiled kernel's own RNG (_nb_seed_val, used by
    # sweep's numba path); self.rng -- what sweep_reference actually draws from --
    # defaults to an UNSEEDED np.random.default_rng() when rng is omitted. Without
    # this, "seed=22" does not make sweep_reference reproducible at all: two
    # independent runs of this exact test produced D=12 and D=4 after the same
    # 5,000 moves. With rng seeded, the original 40_000-move/40-sample budget and
    # seeds (21, 22) agree comfortably and reproducibly (verified by rerunning
    # twice: identical numba=20.65, ref=21.77 both times) -- the apparent
    # autocorrelation-driven flakiness was this missing seed, not slow D-mixing.
    def meanD(sweeper, seed, moves=40_000, samples=40):
        g = SurfaceWormGas(S, openSurfaceFugacity=0.2, intersectionFugacity=0.3,
                           seed=seed, rng=np.random.default_rng(seed), measure=False)
        st = FState(S); out = []
        for _ in range(samples):
            getattr(g, sweeper)(st, moves // samples)
            out.append(st.D)
        return np.mean(out), np.std(out) / len(out) ** 0.5
    m1, e1 = meanD('sweep', 21)
    m2, e2 = meanD('sweep_reference', 22)
    assert abs(m1 - m2) < 5 * np.hypot(e1, e2) + 0.5
