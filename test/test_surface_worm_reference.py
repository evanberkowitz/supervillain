import itertools

import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.gas import SurfaceWormGas
from supervillain.generator.no_intersection.surface_worm.state import FState
from supervillain.generator.no_intersection.surface_worm.weights import PairUmbrella

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

def test_coboundary_candidates_carry_umbrella_and_match_global_recompute():
    # The compiled kernel's cob_log_umbrella warns that the coboundary heatbath is
    # the move that transports the +/-1 pair (it changes q at fixed dF), so leaving
    # w_2 out of _coboundary_log_weights would target a different stationary
    # distribution than the plaquette move (and than the compiled kernel) whenever
    # the umbrella is not the identity.  Walk a chain under a NON-identity umbrella,
    # find (mu, y) draws whose umbrella contribution genuinely VARIES across the
    # enumerated Delta window (cheap: only _coboundary_umbrella_log_weight, no
    # recompute), and only then pay for the expensive check: every enumerated
    # candidate's full log-weight against an independent global recompute
    # (_log_extended_weight, which now also carries w_2).  Requiring genuine
    # variation is what makes this a real regression test for the fix rather than
    # one that could pass vacuously with w_2 silently omitted (most touched (mu, y)
    # draws leave the umbrella contribution constant across their small window, so
    # a naive "aff is non-empty" gate is not enough -- confirmed by running this
    # exact test against the pre-fix code, which failed here).
    N = 4
    umbrella = PairUmbrella(np.linspace(0, 0.5, N ** 2 + 1), N)
    S, g = _gas(N=N, pairUmbrella=umbrella)
    st = FState(S)
    for _ in range(50):
        g.sweep_reference(st, 200)
        if st.Q >= 2:
            break
    assert st.Q >= 2, 'chain never visited the pair sector; the test would not exercise w_2'

    checked = 0
    for mu in range(4):
        for y in itertools.product(range(N), repeat=4):
            deltas, logw, plq, aff, shift = g._coboundary_log_weights(st, mu, y)
            if not aff:
                continue
            w2vals = np.array([g._coboundary_umbrella_log_weight(st, aff, int(D)) for D in deltas])
            if len(set(np.round(w2vals, 10))) <= 1:
                continue  # umbrella term is constant across this window; not a useful check
            recompute = np.empty(len(deltas))
            for i, Delta in enumerate(deltas):
                Fshift = st.F.copy()
                for (pc, xp, sign) in plq:
                    Fshift[(pc,) + xp] += int(Delta) * sign
                recompute[i] = g._log_extended_weight(Fshift)
            # Only relative log-weights are meaningful (this feeds an exact-
            # conditional softmax draw, not a Metropolis ratio against the
            # current state), so compare differences against the first candidate.
            assert np.allclose(logw - logw[0], recompute - recompute[0], atol=1e-8)
            checked += 1
            if checked >= 5:
                break
        if checked >= 5:
            break
    assert checked > 0, 'no (mu, y) draw exercised a varying pair-separation umbrella term'
