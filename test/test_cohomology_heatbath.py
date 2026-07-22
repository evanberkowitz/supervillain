#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.lattice import Form, d
from supervillain.analysis import autocorrelation_time


def _action(D=2, N=6, kappa=0.5, W=1):
    L = supervillain.lattice.Lattice(D=D, N=N)
    return supervillain.action.Villain(L, kappa=kappa, W=W)


def test_cohomology_heatbath_shapes():
    S = _action(D=2, N=6)
    G = supervillain.generator.villain.CohomologyHeatbath(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['phi'].shape == cfg['phi'].shape
    assert result['n'].shape == cfg['n'].shape


def test_cohomology_heatbath_leaves_phi_and_dn_untouched():
    # Δn is constant on a slice, so d(Δn)=0 and dn (the constraint) is exactly preserved;
    # φ is untouched.
    for D, N in [(2, 6), (3, 4)]:
        S = _action(D=D, N=N, kappa=0.3)
        G = supervillain.generator.villain.CohomologyHeatbath(S, rng=np.random.default_rng(0))
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


def test_cohomology_heatbath_conditional_matches_discrete_gaussian():
    # Small κ, small N widen σ so the holonomy conditional spreads over several integers.
    S = _action(D=2, N=4, kappa=0.02)
    G = supervillain.generator.villain.CohomologyHeatbath(S, rng=np.random.default_rng(7))
    a = G.curvature
    hstar = np.full(40000, 0.4)
    draws = G._draw(hstar)
    ks = np.arange(draws.min(), draws.max() + 1)
    logw = -0.5 * a * (ks - 0.4)**2
    pmf = np.exp(logw - logw.max()); pmf /= pmf.sum()
    emp = np.array([(draws == k).mean() for k in ks])
    assert np.abs(emp - pmf).max() < 0.01


def test_cohomology_heatbath_conditional_mutation_fails():
    S = _action(D=2, N=4, kappa=0.02)
    G = supervillain.generator.villain.CohomologyHeatbath(S, rng=np.random.default_rng(8))
    hstar = np.full(40000, 0.4)
    draws = G._draw(hstar)
    ks = np.arange(draws.min(), draws.max() + 1)
    logw = -0.5 * (2 * G.curvature) * (ks - 0.4)**2
    pmf = np.exp(logw - logw.max()); pmf /= pmf.sum()
    emp = np.array([(draws == k).mean() for k in ks])
    assert np.abs(emp - pmf).max() > 0.01


def _mean(e, obs):
    a = np.asarray(getattr(e, obs))
    tau = max(autocorrelation_time(a), 1)
    return a.mean(), a.std(ddof=1) * np.sqrt(2 * tau / len(a))


def test_cohomology_heatbath_matches_cohomology_update():
    # Match-existing on the winding sector: swap only the holonomy kernel, sharing a fast
    # SiteHeatbath φ-kernel (and LinkHeatbath so the sector is genuinely explored).  Small
    # κ keeps the winding sector active.  ⟨WindingSquared⟩ and ⟨ActionDensity⟩ must agree.
    import supervillain.generator.combining as C
    from supervillain.generator.villain import (
        SiteHeatbath, LinkHeatbath, CohomologyUpdate, CohomologyHeatbath)
    S = _action(D=2, N=4, kappa=0.1)

    metro = C.Sequentially((SiteHeatbath(S, rng=np.random.default_rng(1)),
                            LinkHeatbath(S, rng=np.random.default_rng(2)),
                            CohomologyUpdate(S)))
    heat = C.Sequentially((SiteHeatbath(S, rng=np.random.default_rng(3)),
                           LinkHeatbath(S, rng=np.random.default_rng(4)),
                           CohomologyHeatbath(S, rng=np.random.default_rng(5))))

    em = supervillain.Ensemble(S).generate(6000, metro, start='cold').cut(1500)
    eh = supervillain.Ensemble(S).generate(6000, heat, start='cold').cut(1500)

    for obs in ('ActionDensity', 'WindingSquared'):
        m_mean, m_err = _mean(em, obs)
        h_mean, h_err = _mean(eh, obs)
        err = np.hypot(m_err, h_err)
        assert abs(m_mean - h_mean) < 6 * err, f'{obs}: {m_mean}±{m_err} vs {h_mean}±{h_err}'
