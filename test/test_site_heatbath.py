#!/usr/bin/env python

import numpy as np
import supervillain


def _action(D=2, N=6, kappa=0.5, W=1):
    L = supervillain.lattice.Lattice(D=D, N=N)
    return supervillain.action.Villain(L, kappa=kappa, W=W)


def test_site_heatbath_D2_shapes():
    S = _action(D=2, N=6)
    G = supervillain.generator.villain.SiteHeatbath(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['phi'].shape == cfg['phi'].shape
    assert result['n'].shape == cfg['n'].shape


def test_site_heatbath_D3_works():
    S = _action(D=3, N=4)
    G = supervillain.generator.villain.SiteHeatbath(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['phi'].shape == cfg['phi'].shape


def test_site_heatbath_leaves_n_untouched():
    S = _action(D=2, N=6)
    G = supervillain.generator.villain.SiteHeatbath(S, rng=np.random.default_rng(0))
    cfg = S.configurations(1)[0]
    cfg['n'] = cfg['n'] + 1  # some nonzero n
    before = np.asarray(cfg['n']).copy()
    result = G.step(cfg)
    assert (np.asarray(result['n']) == before).all()


def test_site_heatbath_equipartition_at_zero_n():
    # <S> = ½(V-1) at n=0, independent of κ — the exact-conditional signature.
    S = _action(D=2, N=6, kappa=0.7)
    L = S.Lattice
    G = supervillain.generator.villain.SiteHeatbath(S, rng=np.random.default_rng(2026))
    e = supervillain.Ensemble(S).generate(2000, G, start='cold').cut(400)
    from supervillain.lattice import Form
    from supervillain.analysis import autocorrelation_time
    phi = np.asarray(e.configuration.phi)
    n = np.asarray(e.configuration.n)
    Svals = np.array([S(Form(phi[i], degree=0, lattice=L),
                        Form(n[i], degree=1, lattice=L)) for i in range(len(e))])
    target = 0.5 * (L.N ** L.D - 1)
    tau = autocorrelation_time(Svals)
    sem = Svals.std(ddof=1) * np.sqrt(2 * tau / len(Svals))
    assert abs(Svals.mean() - target) < 6 * sem
