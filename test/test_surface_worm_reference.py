import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.gas import SurfaceWormGas
from supervillain.generator.no_intersection.surface_worm.state import FState

def _gas(N=4, kappa=0.2, **kw):
    S = supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)
    kw.setdefault('openSurfaceFugacity', 0.2)
    kw.setdefault('intersectionFugacity', 0.3)
    # sectorWeightCap default (64) is a hard wall (SectorWeights.fugacity: see
    # test_hard_wall_is_minus_inf) -- fine under Metropolis-gated sweeps (the -inf
    # weight simply blocks the proposal), but test_plaquette_acceptance_matches_
    # global_recompute below APPLIES every move unconditionally to explore varied
    # states, and D = #{dF != 0} grows by up to 4 per toggle (cellsPerPlaquette).
    # With 60 unconditional toggles at N=4 (V=256, so 4*V=1024 distinct cube
    # identities) that deterministically pushes D past a cap of 64 for the fixed
    # seed used there, and once BOTH the before- and after-toggle states sit past
    # the cap, `_log_extended_weight` returns -inf for each and the test's own
    # `after - before` is -inf-(-inf) = nan -- not a sign of any acceptance-vs-
    # recompute disagreement (verified: with the cap raised the same walk matches
    # to ~5e-14), just the cap binding where the diagnostic walk never intended to
    # probe it. Raised well above the 4*60=240 ceiling that many unconditional
    # toggles can possibly reach, so the cap never binds in that test.
    kw.setdefault('sectorWeightCap', 512)
    return S, SurfaceWormGas(S, seed=5, **kw)

def test_constructor_guardrails():
    S = supervillain.action.NoIntersections(Lattice(4, 4), kappa=0.2)
    with pytest.raises(ValueError):
        SurfaceWormGas(S)                                        # neither price
    with pytest.raises(ValueError):
        SurfaceWormGas(S, openSurfaceFugacity=0.1,
                       sectorWeights=object())                   # both prices
    with pytest.raises(ValueError):
        SurfaceWormGas(S, openSurfaceFugacity=0.1, targetFraction=1.0)

def test_tilt_is_always_on():
    S, g = _gas()
    assert g._windingCoefficient == pytest.approx(2 * np.pi**2 * 0.2 / 4**4)

def test_plaquette_acceptance_matches_global_recompute():
    S, g = _gas()
    st = FState(S)
    rng = np.random.default_rng(3)
    twopi2k = 2 * np.pi**2 * g.kappa
    for _ in range(60):
        c = int(rng.integers(6)); x = tuple(int(v) for v in rng.integers(4, size=4))
        s = 1 if rng.random() < 0.5 else -1
        before = g._log_extended_weight(st.F)
        lnA, dD, dQ, cube_new, q_new, idx = g._plaquette_log_acceptance(st, c, x, s)
        # apply unconditionally to walk into varied states
        g._apply_plaquette(st, c, x, s, dD, dQ, cube_new, q_new, idx)
        after = g._log_extended_weight(st.F)
        # the acceptance exponent must equal the true Delta log pi_ext up to the
        # Hastings proposal correction, which the global recompute does not carry;
        # with targetFraction=0 that correction is identically zero.
        assert lnA == pytest.approx(after - before, abs=1e-8)

def test_invariants_after_reference_sweep():
    S, g = _gas()
    st = FState(S)
    g.sweep_reference(st, 3000)
    assert st.check()
