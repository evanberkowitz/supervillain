#!/usr/bin/env python

import numpy as np
import supervillain


def _action(D=2, N=6, kappa=0.1, W=1):
    L = supervillain.lattice.Lattice(D=D, N=N)
    return supervillain.action.Villain(L, kappa=kappa, W=W)


def test_link_heatbath_D2_shapes():
    S = _action(D=2, N=6)
    G = supervillain.generator.villain.LinkHeatbath(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['phi'].shape == cfg['phi'].shape
    assert result['n'].shape == cfg['n'].shape


def test_link_heatbath_D3_works():
    S = _action(D=3, N=4)
    G = supervillain.generator.villain.LinkHeatbath(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['n'].shape == cfg['n'].shape


def test_link_heatbath_leaves_phi_untouched():
    S = _action(D=2, N=6)
    G = supervillain.generator.villain.LinkHeatbath(S, rng=np.random.default_rng(0))
    cfg = S.configurations(1)[0]
    before = np.asarray(cfg['phi']).copy()
    result = G.step(cfg)
    assert (np.asarray(result['phi']) == before).all()


def test_link_heatbath_W2_preserves_constraint_and_sector():
    S = _action(D=2, N=6, kappa=0.1, W=2)
    G = supervillain.generator.villain.LinkHeatbath(S, rng=np.random.default_rng(3))
    cfg = S.configurations(1)[0]  # cold: n=0, dn≡0 (mod W)
    for _ in range(20):
        cfg = G.step(cfg)
        # Moves are multiples of W, so from n=0 every n stays ≡ 0 (mod W)...
        assert (np.asarray(cfg['n']) % S.W == 0).all()
        # ...and the winding constraint holds.
        assert S.valid(cfg)


def test_link_heatbath_exact_n2_at_zero_phi():
    # At φ=0, W=1 the per-link marginal is P(n=k) ∝ exp(-2π²κ k²); check <n²>.
    kappa = 0.05
    S = _action(D=2, N=6, kappa=kappa, W=1)
    G = supervillain.generator.villain.LinkHeatbath(S, rng=np.random.default_rng(2026))
    e = supervillain.Ensemble(S).generate(600, G, start='cold').cut(50)
    k = np.arange(-12, 13)
    w = np.exp(-0.5 * kappa * (2 * np.pi * k) ** 2)
    target = (k**2 * w).sum() / w.sum()
    n2 = np.asarray(e.configuration.n).ravel().astype(float) ** 2
    sem = n2.std(ddof=1) / np.sqrt(n2.size)
    assert abs(n2.mean() - target) < 6 * sem
