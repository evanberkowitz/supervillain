#!/usr/bin/env python

r'''The FourierSiteHeatbath draws φ from its exact joint conditional.

The sharp checks are deterministic: the conditional mean really is the
stationary point of the action (to machine precision, in any dimension), and
n is untouched.  The distributional checks are equipartition --- of the whole
action at n = 0, and of the *fluctuation* about the mean at n ≠ 0, which is
what pins the noise normalization --- plus agreement with the site-by-site
SiteHeatbath, which samples the same conditional.
'''

import numpy as np
import pytest

import supervillain
from supervillain.lattice import Form, d, delta, laplacian


def _action(D=2, N=6, kappa=0.5, W=1):
    L = supervillain.lattice.Lattice(D=D, N=N)
    return supervillain.action.Villain(L, kappa=kappa, W=W)


def _S(S, cfg):
    L = S.Lattice
    return float(S(Form(np.asarray(cfg['phi']), degree=0, lattice=L),
                   Form(np.asarray(cfg['n']), degree=1, lattice=L)))


def _randomize(S, cfg, seed):
    r = np.random.default_rng(seed)
    L = S.Lattice
    cfg = dict(cfg)
    cfg['phi'] = cfg['phi'] + Form(r.normal(size=np.asarray(cfg['phi']).shape),
                                   degree=0, lattice=L)
    cfg['n'] = cfg['n'] + r.integers(-2, 3, size=np.asarray(cfg['n']).shape)
    return cfg


def test_requires_villain():
    with pytest.raises(ValueError):
        supervillain.generator.villain.FourierSiteHeatbath('not an action')


def test_shapes_and_n_untouched():
    S = _action(D=2, N=6)
    G = supervillain.generator.villain.FourierSiteHeatbath(
        S, rng=np.random.default_rng(0))
    cfg = _randomize(S, S.configurations(1)[0], seed=1)
    before = np.asarray(cfg['n']).copy()
    out = G.step(cfg)
    assert out['phi'].shape == cfg['phi'].shape
    assert out['n'].shape == cfg['n'].shape
    assert np.array_equal(np.asarray(out['n']), before)


@pytest.mark.parametrize('D,N', [(2, 6), (3, 4), (2, 5), (4, 4)])
def test_conditional_mean_is_stationary(D, N):
    r'''δ(dφ̄ − 2πn) = 0: the mean really is the minimizer, in any D and for odd N.'''
    S = _action(D=D, N=N, kappa=0.7)
    G = supervillain.generator.villain.FourierSiteHeatbath(
        S, rng=np.random.default_rng(2))
    cfg = _randomize(S, S.configurations(1)[0], seed=3)
    mean = G.conditional_mean(cfg)
    force = np.asarray(delta(d(mean) - 2 * np.pi * cfg['n']))
    assert np.abs(force).max() < 1e-8 * max(1., np.abs(np.asarray(cfg['n'])).max())


@pytest.mark.parametrize('D,N', [(2, 6), (3, 4)])
def test_conditional_mean_minimizes_the_action(D, N):
    r'''No perturbation of φ̄ lowers the action.'''
    S = _action(D=D, N=N, kappa=0.7)
    G = supervillain.generator.villain.FourierSiteHeatbath(
        S, rng=np.random.default_rng(4))
    cfg = _randomize(S, S.configurations(1)[0], seed=5)
    mean = G.conditional_mean(cfg)
    best = _S(S, cfg | {'phi': mean})
    r = np.random.default_rng(6)
    for _ in range(5):
        for scale in (0.05, -0.05):
            bumped = mean + Form(scale * r.normal(size=mean.shape),
                                 degree=0, lattice=S.Lattice)
            assert _S(S, cfg | {'phi': bumped}) > best


@pytest.mark.parametrize('D,N', [(2, 6), (3, 4), (2, 5)])
def test_equipartition_at_zero_n(D, N):
    r'''At n = 0 the action is a pure Gaussian: ⟨S⟩ = (V−1)/2, one half per
    non-constant mode, independent of κ.'''
    S = _action(D=D, N=N, kappa=0.3)
    G = supervillain.generator.villain.FourierSiteHeatbath(
        S, rng=np.random.default_rng(7))
    cfg = S.configurations(1)[0]
    draws = np.array([_S(S, (cfg := G.step(cfg))) for _ in range(400)])
    expected = 0.5 * (S.Lattice.sites - 1)
    err = draws.std() / np.sqrt(len(draws))
    assert abs(draws.mean() - expected) < 5 * err


def test_fluctuation_equipartition_with_flux():
    r'''The noise normalization, isolated from the mean: at any n the
    fluctuation about φ̄ carries (κ/2)‖d(φ−φ̄)‖² with mean (V−1)/2.'''
    S = _action(D=2, N=6, kappa=0.4)
    G = supervillain.generator.villain.FourierSiteHeatbath(
        S, rng=np.random.default_rng(8))
    cfg = _randomize(S, S.configurations(1)[0], seed=9)
    mean = np.asarray(G.conditional_mean(cfg))
    vals = []
    for _ in range(400):
        cfg = G.step(cfg)
        eta = np.asarray(cfg['phi']) - mean
        vals.append(0.5 * S.kappa * float((np.asarray(d(Form(
            eta, degree=0, lattice=S.Lattice)))**2).sum()))
    vals = np.array(vals)
    expected = 0.5 * (S.Lattice.sites - 1)
    err = vals.std() / np.sqrt(len(vals))
    assert abs(vals.mean() - expected) < 5 * err


def test_agrees_with_site_heatbath():
    r'''Both sample P(φ|n); at fixed n their action distributions must agree.'''
    S = _action(D=2, N=6, kappa=0.6)
    cfg0 = _randomize(S, S.configurations(1)[0], seed=10)
    n = cfg0['n']

    fourier = supervillain.generator.villain.FourierSiteHeatbath(
        S, rng=np.random.default_rng(11))
    site = supervillain.generator.villain.SiteHeatbath(
        S, rng=np.random.default_rng(12))

    a = dict(cfg0)
    fast = np.array([_S(S, (a := fourier.step(a))) for _ in range(600)])

    b = dict(cfg0)
    for _ in range(50):            # burn in the site sweeper
        b = site.step(b)
    slow = np.array([_S(S, (b := site.step(b))) for _ in range(600)])

    assert np.array_equal(np.asarray(a['n']), np.asarray(n))
    # the site sweeper is correlated, so give it a generous effective sample size
    err = np.hypot(fast.std() / np.sqrt(len(fast)),
                   slow.std() / np.sqrt(len(slow) / 20))
    assert abs(fast.mean() - slow.mean()) < 5 * err


def test_successive_draws_are_independent():
    r'''The point of the generator: the φ sector has no autocorrelation.'''
    S = _action(D=2, N=6, kappa=0.6)
    G = supervillain.generator.villain.FourierSiteHeatbath(
        S, rng=np.random.default_rng(13))
    cfg = _randomize(S, S.configurations(1)[0], seed=14)
    draws = np.array([_S(S, (cfg := G.step(cfg))) for _ in range(2000)])
    c = draws - draws.mean()
    rho = float((c[:-1] * c[1:]).mean() / (c * c).mean())
    assert abs(rho) < 5 / np.sqrt(len(draws))


def test_constant_mode_is_left_alone():
    r'''The flat direction is not touched, so the mean of φ is preserved.'''
    S = _action(D=2, N=6, kappa=0.5)
    G = supervillain.generator.villain.FourierSiteHeatbath(
        S, rng=np.random.default_rng(15))
    cfg = _randomize(S, S.configurations(1)[0], seed=16)
    before = np.asarray(cfg['phi']).mean()
    for _ in range(5):
        cfg = G.step(cfg)
        assert abs(np.asarray(cfg['phi']).mean() - before) < 1e-8


def test_works_for_no_intersections():
    r'''NoIntersections is a Villain whose extra term is φ-independent, so the
    same conditional applies and the constraint cannot be disturbed.'''
    L = supervillain.lattice.Lattice(D=4, N=4)
    S = supervillain.action.NoIntersections(L, kappa=0.3)
    G = supervillain.generator.villain.FourierSiteHeatbath(
        S, rng=np.random.default_rng(17))
    cfg = S.configurations(1)[0]
    before = np.asarray(cfg['n']).copy()
    for _ in range(5):
        cfg = G.step(cfg)
    assert np.array_equal(np.asarray(cfg['n']), before)


@pytest.mark.parametrize('D,N', [(2, 6), (3, 4), (2, 5), (4, 4)])
def test_symbol_is_the_lattice_laplacian(D, N):
    r'''The generator reads its Fourier symbol off the lattice's own
    :func:`~.laplacian` rather than rewriting the eigenvalues by hand.  Check it
    against the analytic 4 Σ sin²(k/2), and check that using it really inverts
    the lattice operator on a random source.'''
    S = _action(D=D, N=N, kappa=0.5)
    G = supervillain.generator.villain.FourierSiteHeatbath(
        S, rng=np.random.default_rng(18))
    L = S.Lattice

    k = 2 * np.pi * np.fft.fftfreq(N)
    analytic = sum(4 * np.sin(K / 2)**2
                   for K in np.meshgrid(*(D * (k,)), indexing='ij'))
    inverse = np.where(analytic > 1e-9 * analytic.max(),
                       1. / np.where(analytic > 1e-9 * analytic.max(), analytic, 1.), 0.)
    assert np.allclose(G._inverse_laplacian, inverse)

    # and it is a genuine inverse: Δ₀⁻¹Δ₀ f = f up to the constant mode
    r = np.random.default_rng(19)
    f = L.zeros(0)
    f[...] = r.normal(size=f.shape)
    f = f - np.asarray(f).mean()
    back = L.ifft(L.fft(laplacian(f)) * G._inverse_laplacian).real
    assert np.abs(back - np.asarray(f)).max() < 1e-10
