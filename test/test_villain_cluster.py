#!/usr/bin/env python

import numpy as np
import pytest
import supervillain
from supervillain.action import Villain, NoIntersections
from supervillain.lattice import Lattice, Form, d
from supervillain.analysis import autocorrelation_time


def _action(D=2, N=8, kappa=0.7, W=1):
    return Villain(Lattice(D, N), kappa=kappa, W=W)


def test_wolff_rejects_constrained_actions():
    # The boundary n -> -n flip breaks dn ≡ 0 (mod W) for W>1 and the dn∧dn=0 constraint of
    # NoIntersections, so the constructor must reject both.
    with pytest.raises(ValueError):
        supervillain.generator.villain.VillainWolff(_action(W=2))
    with pytest.raises(ValueError):
        supervillain.generator.villain.VillainWolff(NoIntersections(Lattice(4, 4), kappa=0.3))


def test_wolff_not_ergodic_alone_from_zero_n():
    # It only reflects existing structure: from n = 0 the flip n -> -n keeps n = 0
    # (−0 = 0), so it can never build vortex/winding content on its own.
    S = _action(D=2, N=8, kappa=0.5)
    G = supervillain.generator.villain.VillainWolff(S, rng=np.random.default_rng(1))
    cfg = S.configurations(1)[0]           # cold: n = 0
    cfg['phi'] = cfg['phi'] + Form(
        np.random.default_rng(2).normal(size=np.asarray(cfg['phi']).shape),
        degree=0, lattice=S.Lattice)
    for _ in range(50):
        cfg = G.step(cfg)
        assert (np.asarray(cfg['n']) == 0).all()


def test_wolff_wrapping_reflection_flips_the_holonomy():
    # A cluster that wraps a cycle realizes n -> -n on it, flipping the H^1 holonomy
    # w_μ -> -w_μ: the move is NOT confined to a winding sector.  Build a clean winding
    # config (n_0 = 1 everywhere gives dn = 0 and holonomy w_0 = N, w_1 = 0), reflect the
    # whole lattice, and confirm the holonomy flips sign.
    S = _action(D=2, N=6, kappa=0.7)
    L = S.Lattice
    n = L.form(1)
    n[0] = 1                                # unit worldline winding in direction 0
    assert np.allclose(np.asarray(d(n)), 0)  # dn = 0, a clean winding config
    holo = lambda n: np.array([np.asarray(n)[0, :, 0].sum(), np.asarray(n)[1, 0, :].sum()])
    w = holo(n)
    n_reflected = Form(-np.asarray(n), degree=1, lattice=L)   # whole-lattice cluster: n -> -n
    assert np.array_equal(holo(n_reflected), -w) and not np.array_equal(w, -w)


def test_wolff_whole_lattice_reflection_is_a_symmetry():
    # Reflecting every site (φ→2r−φ) and every link (n→−n) sends θ→−θ, leaving S invariant.
    S = _action(D=2, N=6, kappa=0.7)
    L = S.Lattice
    rng = np.random.default_rng(0)
    phi = L.form(0); phi[...] = rng.normal(size=phi.shape)
    n = L.form(1); n[...] = rng.integers(-2, 3, size=n.shape)
    r = 0.4
    phi2 = Form(2 * r - np.asarray(phi), degree=0, lattice=L)
    n2 = Form(-np.asarray(n), degree=1, lattice=L)
    assert abs(S(phi2, n2) - S(phi, n)) < 1e-8 * (1 + abs(S(phi, n)))


def test_wolff_changes_the_configuration():
    S = _action(D=2, N=8, kappa=0.7)
    G = supervillain.generator.villain.VillainWolff(S, rng=np.random.default_rng(1))
    cfg = S.configurations(1)[0]
    cfg['phi'] = cfg['phi'] + Form(
        np.random.default_rng(2).normal(size=np.asarray(cfg['phi']).shape),
        degree=0, lattice=S.Lattice)
    before = np.asarray(cfg['phi']).copy()
    after = np.asarray(G.step(cfg)['phi'])
    assert not np.allclose(after, before)  # some cluster reflected


def _mean(e, obs):
    a = np.asarray(getattr(e, obs))
    tau = max(autocorrelation_time(a), 1)
    return a.mean(), a.std(ddof=1) * np.sqrt(2 * tau / len(a))


def test_wolff_preserves_the_distribution():
    # Correctness: interleaving the Wolff cluster with the exact heatbaths must NOT change
    # the sampled distribution vs the heatbaths alone.  Compare ⟨ActionDensity⟩ and
    # ⟨SpinMagnetizationSquared⟩ with τ-aware errors; use a shared fast kernel so both
    # ensembles are genuinely decorrelated.
    import supervillain.generator.combining as C
    from supervillain.generator.villain import SiteHeatbath, LinkHeatbath, VillainWolff
    S = _action(D=2, N=12, kappa=0.7)

    local = C.Sequentially((SiteHeatbath(S, rng=np.random.default_rng(1)),
                            LinkHeatbath(S, rng=np.random.default_rng(2))))
    withcluster = C.Sequentially((SiteHeatbath(S, rng=np.random.default_rng(3)),
                                  LinkHeatbath(S, rng=np.random.default_rng(4)),
                                  VillainWolff(S, rng=np.random.default_rng(5))))

    el = supervillain.Ensemble(S).generate(5000, local, start='cold').cut(1200)
    ec = supervillain.Ensemble(S).generate(5000, withcluster, start='cold').cut(1200)

    for obs in ('ActionDensity', 'SpinMagnetizationSquared'):
        lm, le = _mean(el, obs)
        cm, ce = _mean(ec, obs)
        err = np.hypot(le, ce)
        assert abs(lm - cm) < 6 * err, f'{obs}: local {lm}±{le} vs cluster {cm}±{ce}'
