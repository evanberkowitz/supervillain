#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.lattice import Form, d, delta, delta_sparse
from supervillain.analysis import autocorrelation_time


def _action(D=2, N=6, kappa=0.5, W=1):
    L = supervillain.lattice.Lattice(D=D, N=N)
    return supervillain.action.Worldline(L, kappa=kappa, W=W)


def _randomized(S, seed=0, sweeps=15):
    # Build a nontrivial *valid* configuration (δm = 0) starting from cold by applying the
    # trusted Metropolis updaters: CoexactUpdate keeps δm = 0, VortexUpdate stirs v.
    from supervillain.generator.worldline import CoexactUpdate, VortexUpdate
    cfg = S.configurations(1)[0]
    cu = CoexactUpdate(S, interval_t=2); cu.rng = np.random.default_rng(seed)
    vu = VortexUpdate(S, interval_v=2); vu.rng = np.random.default_rng(seed + 1)
    for _ in range(sweeps):
        cfg = cu.step(cfg)
        cfg = vu.step(cfg)
    return cfg


def test_coexact_heatbath_shapes():
    S = _action(D=2, N=6)
    G = supervillain.generator.worldline.CoexactHeatbath(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['m'].shape == cfg['m'].shape
    assert result['v'].shape == cfg['v'].shape


def test_coexact_heatbath_leaves_v_and_delta_m_untouched():
    # Δm = δt is coexact, so δm (and the constraint δm = 0) is exactly preserved, and v is
    # not touched at all.
    for D, N, W in [(2, 6, 1), (2, 6, 2), (3, 4, 1)]:
        S = _action(D=D, N=N, kappa=0.7, W=W)
        G = supervillain.generator.worldline.CoexactHeatbath(S, rng=np.random.default_rng(0))
        cfg = _randomized(S, seed=3)

        # The starting configuration is valid.
        assert (np.asarray(delta(cfg['m'])) == 0).all()

        v_before = np.asarray(cfg['v']).copy()
        result = G.step(cfg)

        assert (np.asarray(result['v']) == v_before).all()
        # δm stays *exactly* 0 (integer m, δ²t = 0).
        assert (np.asarray(delta(result['m'])) == 0).all()


def test_coexact_heatbath_conditional_matches_discrete_gaussian():
    # The per-plaquette draw must reproduce the exact discrete Gaussian
    # p(k) ∝ exp(-½ a (k-k*)²).  Large κ widens σ = √κ/2 so the pmf spreads over several integers.
    S = _action(D=2, N=6, kappa=9.0)
    G = supervillain.generator.worldline.CoexactHeatbath(S, rng=np.random.default_rng(7))
    a = G.curvature
    kstar = np.full(40000, 0.35)
    draws = G._draw(kstar)

    lo, hi = draws.min(), draws.max()
    ks = np.arange(lo, hi + 1)
    logw = -0.5 * a * (ks - 0.35)**2
    pmf = np.exp(logw - logw.max()); pmf /= pmf.sum()
    emp = np.array([(draws == k).mean() for k in ks])
    assert np.abs(emp - pmf).max() < 0.01


def test_coexact_heatbath_conditional_mutation_fails():
    # Teeth: comparing the draws against a wrong curvature must fail.
    S = _action(D=2, N=6, kappa=9.0)
    G = supervillain.generator.worldline.CoexactHeatbath(S, rng=np.random.default_rng(8))
    kstar = np.full(40000, 0.35)
    draws = G._draw(kstar)
    ks = np.arange(draws.min(), draws.max() + 1)
    logw = -0.5 * (2 * G.curvature) * (ks - 0.35)**2   # wrong: doubled curvature
    pmf = np.exp(logw - logw.max()); pmf /= pmf.sum()
    emp = np.array([(draws == k).mean() for k in ks])
    assert np.abs(emp - pmf).max() > 0.01


def test_coexact_heatbath_conditional_matches_action():
    # Gold check of the *physics* (curvature a = 4/κ AND mean μ = -(df)_p/4): for a single
    # plaquette, the heatbath's model conditional must match the true Boltzmann conditional
    # computed directly from the Worldline action, exp(-S[m + δt_x]).  Deterministic.
    for W in (1, 2):
        S = _action(D=2, N=6, kappa=9.0, W=W)
        L = S.Lattice
        G = supervillain.generator.worldline.CoexactHeatbath(S)
        cfg = _randomized(S, seed=5)

        f = np.asarray(cfg['m']) - np.asarray(delta(cfg['v'])) / S._W
        g = np.asarray(d(Form(f, degree=1, lattice=L)))

        comp_idx = 0
        site = (0,) * L.D
        single = tuple(np.array([c]) for c in site)          # a single-plaquette "color"
        u = delta_sparse(L, 2, comp_idx, single, np.array([1]))   # δt for unit t at that plaquette

        xs = np.arange(-8, 9)
        # Reduced action (drop the x-independent constant offset, which cancels on normalizing).
        Strue = np.array([0.5 / S.kappa * np.sum((f + x * u)**2) for x in xs])
        Ptrue = np.exp(-(Strue - Strue.min())); Ptrue /= Ptrue.sum()

        mu = -g[comp_idx][site] / 4.0
        Smodel = 0.5 * G.curvature * (xs - mu)**2
        Pmodel = np.exp(-(Smodel - Smodel.min())); Pmodel /= Pmodel.sum()

        assert np.abs(Ptrue - Pmodel).max() < 1e-10


def _mean_action_density(S, generator, steps, therm, seed):
    e = supervillain.Ensemble(S).generate(steps, generator, start='cold').cut(therm)
    a = np.asarray(e.ActionDensity)
    tau = max(autocorrelation_time(a), 1)
    return a.mean(), a.std(ddof=1) * np.sqrt(2 * tau / len(a))


def _v_kernels(S, seed):
    # Shared v-sector / wrapping kernels held identical between the two chains, so only the
    # coexact-m kernel differs.
    from supervillain.generator.worldline import VortexUpdate, WrappingUpdate
    vu = VortexUpdate(S); vu.rng = np.random.default_rng(seed)
    wu = WrappingUpdate(S); wu.rng = np.random.default_rng(seed + 100)
    return vu, wu


def test_coexact_heatbath_matches_coexact_update():
    # Match-existing: swap only the coexact-m kernel, holding shared FAST v/wrapping kernels
    # (VortexUpdate + WrappingUpdate) so both ensembles are genuinely decorrelated (an
    # autocorrelation-blind reference would manufacture spurious disagreement).
    # ⟨ActionDensity⟩ must agree.
    import supervillain.generator.combining as C
    from supervillain.generator.worldline import CoexactUpdate, CoexactHeatbath

    for W in (1, 2):
        S = _action(D=2, N=6, kappa=0.5, W=W)

        cu = CoexactUpdate(S); cu.rng = np.random.default_rng(11)
        metro = C.Sequentially((*_v_kernels(S, 1), cu))
        heat = C.Sequentially((*_v_kernels(S, 2),
                               CoexactHeatbath(S, rng=np.random.default_rng(3))))

        m_mean, m_err = _mean_action_density(S, metro, 4000, 1000, 10)
        h_mean, h_err = _mean_action_density(S, heat, 4000, 1000, 20)
        err = np.hypot(m_err, h_err)
        assert abs(m_mean - h_mean) < 6 * err
