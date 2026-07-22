#!/usr/bin/env python

import numpy as np
import pytest

import supervillain
from supervillain.lattice import Lattice, delta, d, Form
from supervillain.generator.worldline import (
    VortexHeatbath, VortexOverrelaxation, VortexUpdate, CoexactUpdate, WrappingUpdate,
)
from supervillain.generator.combining import Sequentially


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(D, N, kappa, W):
    return supervillain.action.Worldline(Lattice(D=D, N=N), kappa=kappa, W=W)


def _random_valid_configuration(S, seed):
    r'''
    A configuration with a non-trivial constraint-satisfying $m=\delta t$ (so
    $\delta m = \delta\delta t = 0$) and a random $v$ (integer for finite $W$,
    real for $W=\infty$).
    '''
    L = S.Lattice
    rng = np.random.default_rng(seed)

    t = L.form(2, dtype=int)
    t[:] = rng.integers(-2, 3, size=t.shape)
    m = delta(t)

    if S.W < float('inf'):
        v = L.form(2, dtype=int)
        v[:] = rng.integers(-3, 4, size=v.shape)
    else:
        v = L.form(2, dtype=float)
        v[:] = rng.normal(size=v.shape)

    return {'m': m, 'v': v}


def _seeded(generator, seed):
    generator.rng = np.random.default_rng(seed)
    return generator


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('D', [2, 3])
@pytest.mark.parametrize('W', [1, 3, float('inf')])
def test_heatbath_shapes_and_m_untouched(D, W):
    S = _action(D, N=4, kappa=0.7, W=W)
    cfg = _random_valid_configuration(S, seed=0)
    g = _seeded(VortexHeatbath(S), seed=1)

    out = g.step(cfg)

    assert out['m'].shape == cfg['m'].shape
    assert out['v'].shape == cfg['v'].shape
    # m is untouched, both in value and in the constraint δm = 0.
    assert np.array_equal(np.asarray(out['m']), np.asarray(cfg['m']))
    assert (np.asarray(delta(out['m'])) == 0).all()
    assert S.valid(out)
    # v really moved.
    assert not np.array_equal(np.asarray(out['v']), np.asarray(cfg['v']))


@pytest.mark.parametrize('D', [2, 3])
def test_overrelaxation_shapes_and_m_untouched(D):
    S = _action(D, N=4, kappa=0.7, W=float('inf'))
    cfg = _random_valid_configuration(S, seed=0)
    g = _seeded(VortexOverrelaxation(S), seed=1)

    out = g.step(cfg)

    assert out['m'].shape == cfg['m'].shape
    assert out['v'].shape == cfg['v'].shape
    assert np.array_equal(np.asarray(out['m']), np.asarray(cfg['m']))
    assert (np.asarray(delta(out['m'])) == 0).all()
    assert S.valid(out)


def test_overrelaxation_requires_infinite_W():
    S = _action(2, N=4, kappa=0.7, W=3)
    with pytest.raises(ValueError):
        VortexOverrelaxation(S)


def test_heatbath_requires_worldline():
    L = Lattice(D=2, N=4)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    with pytest.raises(ValueError):
        VortexHeatbath(S)


# ---------------------------------------------------------------------------
# VortexHeatbath finite-W: exact discrete-Gaussian conditional
# ---------------------------------------------------------------------------

def _single_plaquette_target(S):
    r'''First plaquette of component 0 in the first checkerboard color.'''
    L = S.Lattice
    color = L.checkerboarding[0]
    comp = 0
    coord = tuple(int(c[0]) for c in color)
    return comp, coord


def _generator_conditional(S, cfg, comp, coord):
    r'''
    The Gaussian conditional the heatbath uses for one plaquette on the frozen
    background ``cfg``: the mean of the *new* value and the curvature ``a``.
    Uses the signed boundary sum $(df)_p = d(f)_p$ and $a = 4/(\kappa \bar W^2)$.
    '''
    W = S._W
    f = Form(np.asarray(cfg['m'], dtype=float) - np.asarray(delta(cfg['v'])) / W, degree=1, lattice=S.Lattice)
    dfp = np.asarray(d(f))[comp][coord]
    mean = cfg['v'][comp][coord] + (W / 4.0) * dfp
    a = 4.0 / (S.kappa * W**2)
    return mean, a


def test_heatbath_conditional_pmf_finite_W():
    # κ, W chosen so σ = W√κ/2 ≈ 1.1 spans a few integers.
    S = _action(2, N=4, kappa=5.0, W=1)
    cfg = _random_valid_configuration(S, seed=2)
    comp, coord = _single_plaquette_target(S)

    mean, a = _generator_conditional(S, cfg, comp, coord)

    # Enumerate a generous window of integer values for the target plaquette.
    ks = np.arange(int(np.floor(mean)) - 12, int(np.floor(mean)) + 13)

    # (1) Exact conditional straight from the action: p(k) ∝ exp(-S| v_p = k).
    def action_at(k):
        v = cfg['v'].copy()
        v[comp][coord] = k
        return 0.5 / S.kappa * np.sum((np.asarray(cfg['m']) - np.asarray(delta(v)) / S._W)**2)

    logp = -np.array([action_at(k) for k in ks])
    exact = np.exp(logp - logp.max()); exact /= exact.sum()

    # (2) The generator's discrete Gaussian with mean and curvature a.
    logg = -0.5 * a * (ks - mean)**2
    gen = np.exp(logg - logg.max()); gen /= gen.sum()

    # The generator's conditional must equal the exact action conditional.
    assert np.allclose(gen, exact, atol=1e-9)

    # Mutation check: a wrong curvature no longer matches the exact conditional.
    logbad = -0.5 * (2.0 * a) * (ks - mean)**2
    bad = np.exp(logbad - logbad.max()); bad /= bad.sum()
    assert np.abs(bad - exact).max() > 1e-2

    # (3) The sampler actually draws from that conditional: histogram gen.step().
    M = 40000
    g = _seeded(VortexHeatbath(S), seed=7)
    samples = np.empty(M, dtype=int)
    for i in range(M):
        samples[i] = g.step(cfg)['v'][comp][coord]

    counts = np.array([(samples == k).sum() for k in ks], dtype=float)
    freq = counts / M
    # Frequencies match the exact pmf within a few standard errors.
    assert np.abs(freq - exact).max() < 0.02


# ---------------------------------------------------------------------------
# VortexHeatbath W=∞: continuous Gaussian mean and variance
# ---------------------------------------------------------------------------

def test_heatbath_continuous_mean_and_variance():
    S = _action(2, N=4, kappa=0.8, W=float('inf'))
    cfg = _random_valid_configuration(S, seed=3)
    comp, coord = _single_plaquette_target(S)

    mean, a = _generator_conditional(S, cfg, comp, coord)
    variance = 1.0 / a
    assert np.isclose(variance, np.pi**2 * S.kappa)   # 1/a = π²κ

    M = 40000
    g = _seeded(VortexHeatbath(S), seed=11)
    samples = np.array([g.step(cfg)['v'][comp][coord] for _ in range(M)])

    # ~1/√M relative precision; give it a comfortable band.
    assert np.isclose(samples.mean(), mean, atol=5 * np.sqrt(variance / M))
    assert np.isclose(samples.var(), variance, rtol=0.05)


# ---------------------------------------------------------------------------
# VortexOverrelaxation W=∞: microcanonical (action preserving)
# ---------------------------------------------------------------------------

def test_overrelaxation_preserves_action():
    S = _action(2, N=5, kappa=0.9, W=float('inf'))
    cfg = _random_valid_configuration(S, seed=4)
    g = _seeded(VortexOverrelaxation(S), seed=5)

    before = S(**cfg)
    out = g.step(cfg)
    after = S(**out)

    assert np.isclose(before, after, atol=1e-8)


def test_overrelaxation_is_not_the_identity():
    S = _action(2, N=5, kappa=0.9, W=float('inf'))
    cfg = _random_valid_configuration(S, seed=6)
    g = _seeded(VortexOverrelaxation(S), seed=8)

    out1 = g.step(cfg)
    assert not np.allclose(np.asarray(out1['v']), np.asarray(cfg['v']))

    # The composite sweep is a product of non-commuting reflections, so two
    # applications do not return to the starting configuration.
    out2 = g.step(out1)
    assert not np.allclose(np.asarray(out2['v']), np.asarray(cfg['v']))


# ---------------------------------------------------------------------------
# match-existing: agree with the trusted VortexUpdate on ⟨ActionDensity⟩
# ---------------------------------------------------------------------------

def _action_density_estimate(S, v_kernel, seed, configurations=3000):
    r'''
    Run an ergodic chain that shares the m-updaters (CoexactUpdate + WrappingUpdate)
    and differs only in the v-kernel, then estimate ⟨ActionDensity⟩ with an
    autocorrelation-aware (bootstrap) error.
    '''
    m1 = _seeded(CoexactUpdate(S), seed + 1)
    m2 = _seeded(WrappingUpdate(S), seed + 2)
    G = Sequentially((m1, m2, _seeded(v_kernel, seed + 3)))

    E = supervillain.Ensemble(S).generate(configurations, G, start='cold')
    E.measure()
    tau = int(np.ceil(E.autocorrelation_time()))
    e = E.cut(20 * tau).every(2 * tau)
    b = supervillain.analysis.Bootstrap(e, 100)
    return b.estimate('ActionDensity')


@pytest.mark.parametrize('W', [1, float('inf')])
def test_heatbath_matches_vortex_update(W):
    S = _action(2, N=4, kappa=0.5, W=W)

    val_hb, err_hb = _action_density_estimate(S, VortexHeatbath(S), seed=100)
    val_mu, err_mu = _action_density_estimate(S, VortexUpdate(S), seed=200)

    combined = np.sqrt(err_hb**2 + err_mu**2)
    assert np.abs(val_hb - val_mu) < 6 * combined
