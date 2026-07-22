#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.lattice import delta
from supervillain.analysis import autocorrelation_time


def _action(D=2, N=6, kappa=0.5, W=1):
    L = supervillain.lattice.Lattice(D=D, N=N)
    return supervillain.action.Worldline(L, kappa=kappa, W=W)


def _randomized(S, seed=0, sweeps=15):
    # Build a nontrivial *valid* configuration (δm = 0) from cold with the trusted Metropolis
    # updaters: CoexactUpdate keeps δm = 0, VortexUpdate stirs v, WrappingUpdate stirs wrapping.
    from supervillain.generator.worldline import CoexactUpdate, VortexUpdate, WrappingUpdate
    cfg = S.configurations(1)[0]
    cu = CoexactUpdate(S, interval_t=2); cu.rng = np.random.default_rng(seed)
    vu = VortexUpdate(S, interval_v=2); vu.rng = np.random.default_rng(seed + 1)
    wu = WrappingUpdate(S, interval_w=2); wu.rng = np.random.default_rng(seed + 2)
    for _ in range(sweeps):
        cfg = cu.step(cfg)
        cfg = vu.step(cfg)
        cfg = wu.step(cfg)
    return cfg


def test_wrapping_heatbath_shapes():
    S = _action(D=2, N=6)
    G = supervillain.generator.worldline.WrappingHeatbath(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['m'].shape == cfg['m'].shape
    assert result['v'].shape == cfg['v'].shape


def test_wrapping_heatbath_leaves_v_and_delta_m_untouched():
    # Each cycle change is a closed loop current, so δm (and the constraint δm = 0) is exactly
    # preserved, and v is never touched.
    for D, N, W in [(2, 6, 1), (2, 6, 2), (3, 4, 1), (2, 6, float('inf'))]:
        S = _action(D=D, N=N, kappa=0.7, W=W)
        G = supervillain.generator.worldline.WrappingHeatbath(S, rng=np.random.default_rng(0))
        cfg = _randomized(S, seed=3)

        assert (np.asarray(delta(cfg['m'])) == 0).all()   # starting config is valid

        v_before = np.asarray(cfg['v']).copy()
        result = G.step(cfg)

        assert (np.asarray(result['v']) == v_before).all()
        assert (np.asarray(delta(result['m'])) == 0).all()


def test_wrapping_heatbath_conditional_matches_discrete_gaussian():
    # The per-cycle draw must reproduce the exact discrete Gaussian p(Δ) ∝ exp(-½ a (Δ-Δ*)²),
    # curvature a = N/κ.  A large κ/N widens σ = √(κ/N) so the pmf spreads over several integers.
    S = _action(D=2, N=6, kappa=24.0)
    G = supervillain.generator.worldline.WrappingHeatbath(S, rng=np.random.default_rng(7))
    a = G.curvature
    dstar = np.full(40000, 0.35)
    draws = G._draw(dstar)

    ks = np.arange(draws.min(), draws.max() + 1)
    logw = -0.5 * a * (ks - 0.35)**2
    pmf = np.exp(logw - logw.max()); pmf /= pmf.sum()
    emp = np.array([(draws == k).mean() for k in ks])
    assert np.abs(emp - pmf).max() < 0.01


def test_wrapping_heatbath_conditional_mutation_fails():
    # Teeth: comparing the draws against a wrong curvature must fail.
    S = _action(D=2, N=6, kappa=24.0)
    G = supervillain.generator.worldline.WrappingHeatbath(S, rng=np.random.default_rng(8))
    dstar = np.full(40000, 0.35)
    draws = G._draw(dstar)
    ks = np.arange(draws.min(), draws.max() + 1)
    logw = -0.5 * (2 * G.curvature) * (ks - 0.35)**2   # wrong: doubled curvature
    pmf = np.exp(logw - logw.max()); pmf /= pmf.sum()
    emp = np.array([(draws == k).mean() for k in ks])
    assert np.abs(emp - pmf).max() > 0.01


def test_wrapping_heatbath_conditional_matches_action():
    # Gold check of the *physics* (curvature a = N/κ AND center Δ* = -A/N): for a single cycle,
    # the heatbath's model conditional must match the true Boltzmann conditional computed
    # directly from the Worldline action, exp(-S[m + Δ·cycle]).  Deterministic.
    for W in (1, 2):
        S = _action(D=2, N=6, kappa=24.0, W=W)
        L = S.Lattice
        G = supervillain.generator.worldline.WrappingHeatbath(S)
        cfg = _randomized(S, seed=5)

        f = np.asarray(cfg['m']) - np.asarray(delta(cfg['v'])) / S._W

        # One μ=0 cycle at perpendicular origin: unit current on every link of that loop.  The
        # generator's center for this cycle is dstar = -(sum of f along the loop)/N.
        mu = 0
        u = np.zeros_like(f)
        perp = tuple(slice(None) if i == mu else 0 for i in range(L.D))
        u[mu][perp] = 1
        dstar = float(-(f[mu].sum(axis=mu))[tuple(0 for i in range(L.D) if i != mu)] / G.cycle_links)

        xs = np.arange(-10, 11)
        Strue = np.array([0.5 / S.kappa * np.sum((f + x * u)**2) for x in xs])
        Ptrue = np.exp(-(Strue - Strue.min())); Ptrue /= Ptrue.sum()

        Smodel = 0.5 * G.curvature * (xs - dstar)**2
        Pmodel = np.exp(-(Smodel - Smodel.min())); Pmodel /= Pmodel.sum()

        assert np.abs(Ptrue - Pmodel).max() < 1e-10


def _mean_action_density(S, generator, steps, therm):
    e = supervillain.Ensemble(S).generate(steps, generator, start='cold').cut(therm)
    a = np.asarray(e.ActionDensity)
    tau = max(autocorrelation_time(a), 1)
    return a.mean(), a.std(ddof=1) * np.sqrt(2 * tau / len(a))


def _mn_kernels(S, seed):
    # Shared vortex + coexact-m kernels held identical between the two chains, so only the
    # wrapping kernel differs.
    from supervillain.generator.worldline import VortexUpdate, CoexactUpdate
    vu = VortexUpdate(S); vu.rng = np.random.default_rng(seed)
    cu = CoexactUpdate(S); cu.rng = np.random.default_rng(seed + 100)
    return vu, cu


def test_wrapping_heatbath_matches_wrapping_update():
    # Match-existing: swap only the wrapping kernel, holding shared FAST vortex/coexact kernels
    # so both ensembles are genuinely decorrelated.  ⟨ActionDensity⟩ must agree.
    import supervillain.generator.combining as C
    from supervillain.generator.worldline import WrappingUpdate, WrappingHeatbath

    for W in (1, 2):
        S = _action(D=2, N=6, kappa=0.5, W=W)

        wu = WrappingUpdate(S); wu.rng = np.random.default_rng(11)
        metro = C.Sequentially((*_mn_kernels(S, 1), wu))
        heat = C.Sequentially((*_mn_kernels(S, 2),
                               WrappingHeatbath(S, rng=np.random.default_rng(3))))

        m_mean, m_err = _mean_action_density(S, metro, 4000, 1000)
        h_mean, h_err = _mean_action_density(S, heat, 4000, 1000)
        err = np.hypot(m_err, h_err)
        assert abs(m_mean - h_mean) < 6 * err
