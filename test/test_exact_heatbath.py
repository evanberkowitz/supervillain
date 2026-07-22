#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.lattice import Form, d
from supervillain.analysis import autocorrelation_time


def _action(D=2, N=6, kappa=0.5, W=1):
    L = supervillain.lattice.Lattice(D=D, N=N)
    return supervillain.action.Villain(L, kappa=kappa, W=W)


def test_exact_heatbath_shapes():
    S = _action(D=2, N=6)
    G = supervillain.generator.villain.ExactHeatbath(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['phi'].shape == cfg['phi'].shape
    assert result['n'].shape == cfg['n'].shape


def test_exact_heatbath_leaves_phi_and_dn_untouched():
    # Δn = dz is closed, so dn (and the winding constraint) is exactly preserved, and φ
    # is not touched at all.
    for D, N in [(2, 6), (3, 4)]:
        S = _action(D=D, N=N, kappa=0.7)
        G = supervillain.generator.villain.ExactHeatbath(S, rng=np.random.default_rng(0))
        cfg = S.configurations(1)[0]
        cfg['phi'] = cfg['phi'] + Form(
            np.random.default_rng(1).normal(size=np.asarray(cfg['phi']).shape),
            degree=0, lattice=S.Lattice)
        cfg['n'] = cfg['n'] + np.random.default_rng(2).integers(-1, 2, size=np.asarray(cfg['n']).shape)
        phi_before = np.asarray(cfg['phi']).copy()
        dn_before = np.asarray(d(cfg['n'])).copy()
        result = G.step(cfg)
        assert (np.asarray(result['phi']) == phi_before).all()
        assert np.allclose(np.asarray(d(result['n'])), dn_before)


def test_exact_heatbath_conditional_matches_discrete_gaussian():
    # The per-site draw must reproduce the exact discrete Gaussian
    # p(k) ∝ exp(-½ a (k-k*)²).  Low κ widens σ so the pmf spreads over several integers.
    S = _action(D=2, N=6, kappa=0.02)
    G = supervillain.generator.villain.ExactHeatbath(S, rng=np.random.default_rng(7))
    a = G.curvature
    kstar = np.full(40000, 0.35)
    draws = G._draw(kstar)

    lo, hi = draws.min(), draws.max()
    ks = np.arange(lo, hi + 1)
    logw = -0.5 * a * (ks - 0.35)**2
    pmf = np.exp(logw - logw.max()); pmf /= pmf.sum()
    emp = np.array([(draws == k).mean() for k in ks])
    assert np.abs(emp - pmf).max() < 0.01


def test_exact_heatbath_conditional_mutation_fails():
    # Teeth: comparing the draws against a wrong curvature must fail.
    S = _action(D=2, N=6, kappa=0.02)
    G = supervillain.generator.villain.ExactHeatbath(S, rng=np.random.default_rng(8))
    kstar = np.full(40000, 0.35)
    draws = G._draw(kstar)
    ks = np.arange(draws.min(), draws.max() + 1)
    logw = -0.5 * (2 * G.curvature) * (ks - 0.35)**2   # wrong: doubled curvature
    pmf = np.exp(logw - logw.max()); pmf /= pmf.sum()
    emp = np.array([(draws == k).mean() for k in ks])
    assert np.abs(emp - pmf).max() > 0.01


def _mean_action_density(S, generator, steps, therm, seed):
    e = supervillain.Ensemble(S).generate(steps, generator, start='cold').cut(therm)
    a = np.asarray(e.ActionDensity)
    tau = max(autocorrelation_time(a), 1)
    return a.mean(), a.std(ddof=1) * np.sqrt(2 * tau / len(a))


def test_exact_heatbath_matches_exact_update():
    # Match-existing: swap only the n-exact kernel, holding a shared FAST φ kernel
    # (SiteHeatbath) so both ensembles are genuinely decorrelated (an autocorrelation-blind
    # reference would manufacture spurious disagreement).  ⟨ActionDensity⟩ must agree.
    import supervillain.generator.combining as C
    from supervillain.generator.villain import SiteHeatbath, ExactUpdate, ExactHeatbath
    S = _action(D=2, N=6, kappa=0.5)

    metro = C.Sequentially((SiteHeatbath(S, rng=np.random.default_rng(1)), ExactUpdate(S)))
    heat = C.Sequentially((SiteHeatbath(S, rng=np.random.default_rng(2)),
                           ExactHeatbath(S, rng=np.random.default_rng(3))))

    m_mean, m_err = _mean_action_density(S, metro, 4000, 1000, 10)
    h_mean, h_err = _mean_action_density(S, heat, 4000, 1000, 20)
    err = np.hypot(m_err, h_err)
    assert abs(m_mean - h_mean) < 6 * err
