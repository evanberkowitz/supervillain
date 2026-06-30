#!/usr/bin/env python

import numpy as np
import pytest

import supervillain
from supervillain.batch import Batch
from supervillain.lattice import d, wedge
from supervillain.observable.topological import (
    TopologicalChargeDensity,
    TopologicalCharge,
    TopologicalChargeDensitySquared,
    TopologicalTwoPoint,
    Topological_Topological,
)


def unit_charge_dipole(L):
    r'''Return an integer 1-form whose charge consists of one +1/-1 pair.'''
    n = L.zeros(1, dtype=int)
    origin = L.origin

    n[L.comp_index[1][(0,)]][origin] = 1
    shifted = list(origin)
    shifted[0] = 1
    n[L.comp_index[1][(1,)]][tuple(shifted)] = 1

    return n


def charge_density(S, n):
    r'''The per-configuration charge field Q_x, the ingredient the downstream
    observables consume.'''
    return TopologicalChargeDensity.Villain(S, n)


def brute_force_correlation(L, f, g):
    r'''An independent, FFT-free real-space evaluation of the translation-averaged
    correlator (1/Λ) ∑_x f(x) g(x-Δx), matching the convention of
    :meth:`~supervillain.lattice.Lattice.correlation`.'''
    f = np.asarray(f)
    g = np.asarray(g)
    axes = tuple(range(L.D))
    C = np.zeros(L.dims)
    for shift in np.ndindex(*L.dims):
        C[shift] = (f * np.roll(g, shift, axis=axes)).sum() / L.sites
    return C


# ---------------------------------------------------------------------------
# TopologicalChargeDensity: the per-configuration field Q_x, the ingredient
# from which the other observables are built.  It is the only observable that
# depends on n; the rest consume it.
# ---------------------------------------------------------------------------

def test_topological_charge_density_equals_exact_form_d_of_n_wedge_dn():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    rng = np.random.default_rng(20260629)
    n = L.zeros(1, dtype=int)
    n[...] = rng.integers(-2, 3, size=n.shape)

    density = TopologicalChargeDensity.Villain(S, n)
    assert density.shape == L.dims

    # Independent path: Q = dn∧dn = d(n∧dn) by the Leibniz rule (since ddn = 0).
    # The observable forms dn first and then wedges; here we wedge first (n∧dn)
    # and then take d, a genuinely different sequence of operations.
    exact_form = np.asarray(d(wedge(n, d(n)))).sum(axis=0)
    assert np.array_equal(density, exact_form)


def test_topological_charge_density_is_bilinear():
    # Q = dn∧dn is quadratic in n, so scaling n by an integer c scales Q by c².
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    rng = np.random.default_rng(31415)
    n = L.zeros(1, dtype=int)
    n[...] = rng.integers(-2, 3, size=n.shape)

    base = TopologicalChargeDensity.Villain(S, n)
    for c in (2, 3, -2):
        scaled = TopologicalChargeDensity.Villain(S, c * n)
        assert np.array_equal(scaled, c ** 2 * base)


def test_topological_charge_density_for_dipole():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    density = TopologicalChargeDensity.Villain(S, unit_charge_dipole(L))

    assert np.array_equal(np.sort(density[density != 0]), [-1, 1])
    assert density.sum() == 0


@pytest.mark.parametrize('D', [3, 5])
def test_topological_charge_density_rejects_non_four_dimensions(D):
    # The topological charge dn∧dn is a 4-form; only in D=4 is it the top form
    # whose integral is the global charge.  TopologicalChargeDensity is the
    # gatekeeper: the downstream observables consume it and inherit the restriction.
    L = supervillain.lattice.Lattice(D=D, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)

    with pytest.raises(NotImplementedError):
        TopologicalChargeDensity.Villain(S, L.zeros(1, dtype=int))


@pytest.mark.parametrize('D', [3, 5])
@pytest.mark.parametrize('name', [
    'TopologicalChargeDensity',
    'TopologicalCharge',
    'TopologicalChargeDensitySquared',
    'TopologicalTwoPoint',
])
def test_topological_observables_reject_non_four_dimensions_via_ensemble(D, name):
    # Measuring any topological observable on a non-4D ensemble fails because the
    # TopologicalChargeDensity ingredient raises.  (ensemble.measure swallows the
    # NotImplementedError and skips, so we trigger the measurement via attribute
    # access, which propagates it.)
    L = supervillain.lattice.Lattice(D=D, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    configurations = S.configurations(1)
    configurations[0] = {'phi': L.zeros(0), 'n': L.zeros(1, dtype=int)}
    ensemble = supervillain.Ensemble(S).from_configurations(configurations)

    with pytest.raises(NotImplementedError):
        getattr(ensemble, name)

    # measure() skips unimplemented observables rather than raising.
    assert name not in ensemble.measure([name])


# ---------------------------------------------------------------------------
# Ensemble integration: measurement, caching, and registration.
# ---------------------------------------------------------------------------

def test_topological_charge_density_squared_integrates_with_ensemble():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    configurations = S.configurations(2)
    configurations[0] = {'phi': L.zeros(0), 'n': L.zeros(1, dtype=int)}
    configurations[1] = {'phi': L.zeros(0), 'n': unit_charge_dipole(L)}
    ensemble = supervillain.Ensemble(S).from_configurations(configurations)

    assert (
        supervillain.observable.TopologicalChargeDensitySquared
        is TopologicalChargeDensitySquared
    )
    assert (
        supervillain.observables['TopologicalChargeDensitySquared']
        is TopologicalChargeDensitySquared
    )

    measured = ensemble.measure(['TopologicalChargeDensitySquared'])
    values = measured['TopologicalChargeDensitySquared']

    assert isinstance(values, Batch)
    assert values.shape == (2,)
    # vacuum → 0; unit dipole → Q² = 1 on two four-cells → 2/Λ.
    assert np.array_equal(Batch.as_array(values), [0, 2 / L.cells_of_degree[4]])
    assert ensemble.TopologicalChargeDensitySquared is values
    assert 'TopologicalChargeDensitySquared' in ensemble.measured
    # The density ingredient was measured and cached as a side effect.
    assert 'TopologicalChargeDensity' in ensemble.measured
    assert TopologicalChargeDensitySquared.autocorrelation(ensemble)


# ---------------------------------------------------------------------------
# TopologicalCharge: the signed total Q_total = ∑_x Q_x, a per-configuration
# Observable that sums the density.  Identically zero.
# ---------------------------------------------------------------------------

def test_topological_charge_sums_the_density_to_zero():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    rng = np.random.default_rng(20260629)
    n = L.zeros(1, dtype=int)
    n[...] = rng.integers(-2, 3, size=n.shape)

    for cfg in (L.zeros(1, dtype=int), unit_charge_dipole(L), n):
        density = charge_density(S, cfg)
        # Independent total via the exact form Q = d(n∧dn): a sum of an exact
        # form over the closed lattice, so it must vanish identically.
        exact_total = np.asarray(d(wedge(cfg, d(cfg)))).sum()
        assert TopologicalCharge.Villain(S, density) == exact_total
        assert TopologicalCharge.Villain(S, density) == 0


# ---------------------------------------------------------------------------
# TopologicalChargeDensitySquared: the same-site value ⟨Q_x²⟩.
# ---------------------------------------------------------------------------

def test_topological_charge_density_squared_vacuum_is_zero():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)

    assert TopologicalChargeDensitySquared.Villain(S, charge_density(S, L.zeros(1, dtype=int))) == 0


def test_topological_charge_density_squared_counts_unit_charge_dipole():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    density = charge_density(S, unit_charge_dipole(L))

    # Q is ±1 on two 4-cells, so Q² is 1 on two cells and the mean is 2/Λ.
    assert TopologicalChargeDensitySquared.Villain(S, density) == 2 / L.cells_of_degree[4]


def test_topological_charge_density_squared_equals_brute_force_origin_correlation():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    rng = np.random.default_rng(11)
    n = L.zeros(1, dtype=int)
    n[...] = rng.integers(-2, 3, size=n.shape)
    density = charge_density(S, n)

    # Λ⁻¹∑_x Q_x² is the same-site value of the real-space correlator;
    # check it against the independent brute-force correlation at Δx = 0.
    expected = brute_force_correlation(L, density, density)[L.origin]
    assert np.isclose(TopologicalChargeDensitySquared.Villain(S, density), expected)
    assert expected > 0


# ---------------------------------------------------------------------------
# TopologicalTwoPoint: the Δx-resolved correlator ⟨Q_x Q_{x-Δx}⟩.
# ---------------------------------------------------------------------------

def test_topological_two_point_matches_brute_force_real_space():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    rng = np.random.default_rng(7)
    n = L.zeros(1, dtype=int)
    n[...] = rng.integers(-2, 3, size=n.shape)
    density = charge_density(S, n)

    two_point = np.asarray(TopologicalTwoPoint.Villain(S, density))
    assert two_point.shape == L.dims

    # Independent of the FFT-accelerated L.correlation: brute-force real-space sum.
    assert np.allclose(two_point, brute_force_correlation(L, density, density))


def test_topological_two_point_origin_equals_charge_density_squared():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    rng = np.random.default_rng(8)
    n = L.zeros(1, dtype=int)
    n[...] = rng.integers(-2, 3, size=n.shape)
    density = charge_density(S, n)

    two_point = np.asarray(TopologicalTwoPoint.Villain(S, density))
    assert np.isclose(two_point[L.origin], TopologicalChargeDensitySquared.Villain(S, density))


def test_topological_two_point_sums_to_zero_because_total_charge_vanishes():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    rng = np.random.default_rng(9)
    n = L.zeros(1, dtype=int)
    n[...] = rng.integers(-2, 3, size=n.shape)

    # ∑_Δx ⟨Q_x Q_{x-Δx}⟩ = (1/Λ)(∑_x Q_x)² = 0 since Q_total = 0.
    two_point = np.asarray(TopologicalTwoPoint.Villain(S, charge_density(S, n)))
    assert np.isclose(two_point.sum(), 0)


# ---------------------------------------------------------------------------
# Topological_Topological: connected correlator = TwoPoint − ⟨Q⟩⊗⟨Q⟩ correlation.
# ---------------------------------------------------------------------------

def test_topological_topological_subtracts_density_disconnected_piece():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    rng = np.random.default_rng(99)
    n = L.zeros(1, dtype=int)
    n[...] = rng.integers(-2, 3, size=n.shape)

    density = charge_density(S, n)
    two_point = np.asarray(TopologicalTwoPoint.Villain(S, density))

    # Independent disconnected piece via the brute-force real-space correlation.
    expected = two_point - brute_force_correlation(L, density, density)
    result = np.asarray(Topological_Topological.default(S, two_point, density))
    assert np.allclose(result, expected)


def test_topological_topological_equals_two_point_when_density_expectation_vanishes():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    rng = np.random.default_rng(100)
    n = L.zeros(1, dtype=int)
    n[...] = rng.integers(-2, 3, size=n.shape)

    two_point = np.asarray(TopologicalTwoPoint.Villain(S, charge_density(S, n)))
    # With ⟨Q⟩ = 0 the disconnected piece vanishes and the connected correlator
    # is exactly the two-point function.
    zero_density = charge_density(S, L.zeros(1, dtype=int))
    result = np.asarray(Topological_Topological.default(S, two_point, zero_density))
    assert np.array_equal(result, two_point)


# ---------------------------------------------------------------------------
# W is no longer restricted to 1: any finite W works (charge quantized in W²),
# and W=∞ (dn=0) measures a well-defined zero.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('W', [2, 3])
def test_topological_observables_allow_finite_W_greater_than_one(W):
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=W)

    # W·(unit dipole) has dn ∈ Wℤ, a valid W>1 configuration; its charge scales by W².
    n = W * unit_charge_dipole(L)
    assert S.valid({'n': n})

    density = charge_density(S, n)
    assert np.abs(density).sum() == 2 * W ** 2
    assert density.sum() == 0

    assert TopologicalCharge.Villain(S, density) == 0
    assert TopologicalChargeDensitySquared.Villain(S, density) == 2 * W ** 4 / L.cells_of_degree[4]
    assert np.asarray(TopologicalTwoPoint.Villain(S, density)).shape == L.dims


def test_topological_observables_W_infinite_measure_zero():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=float('inf'))

    # W=∞ forces dn=0, so every valid configuration has vanishing charge.
    n = L.zeros(1, dtype=int)
    assert S.valid({'n': n})

    density = charge_density(S, n)
    assert np.all(density == 0)
    assert TopologicalCharge.Villain(S, density) == 0
    assert TopologicalChargeDensitySquared.Villain(S, density) == 0
    assert np.all(np.asarray(TopologicalTwoPoint.Villain(S, density)) == 0)


# ---------------------------------------------------------------------------
# Gauge invariance of the new observables.
# ---------------------------------------------------------------------------

def test_new_topological_observables_are_gauge_invariant():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    n = unit_charge_dipole(L)
    phi = L.zeros(0)
    k = L.zeros(0, dtype=int)
    rng = np.random.default_rng(4321)
    k[...] = rng.integers(-3, 4, size=k.shape)

    transformed = S.gauge_transform({'phi': phi, 'n': n}, k)
    density = charge_density(S, n)
    density_transformed = charge_density(S, transformed['n'])

    assert np.array_equal(density_transformed, density)
    assert TopologicalChargeDensitySquared.Villain(S, density_transformed) == TopologicalChargeDensitySquared.Villain(S, density)
    assert np.allclose(
        np.asarray(TopologicalTwoPoint.Villain(S, density_transformed)),
        np.asarray(TopologicalTwoPoint.Villain(S, density)),
    )


# ---------------------------------------------------------------------------
# Registration and autocorrelation inclusion.
# ---------------------------------------------------------------------------

def _two_configuration_ensemble(S):
    L = S.Lattice
    configurations = S.configurations(2)
    configurations[0] = {'phi': L.zeros(0), 'n': L.zeros(1, dtype=int)}
    configurations[1] = {'phi': L.zeros(0), 'n': L.zeros(1, dtype=int)}
    return supervillain.Ensemble(S).from_configurations(configurations)


def test_new_topological_observables_are_registered():
    assert supervillain.observables['TopologicalChargeDensity'] is TopologicalChargeDensity
    assert supervillain.observables['TopologicalCharge'] is TopologicalCharge
    assert supervillain.observables['TopologicalChargeDensitySquared'] is TopologicalChargeDensitySquared
    assert supervillain.observables['TopologicalTwoPoint'] is TopologicalTwoPoint
    assert supervillain.derivedQuantities['Topological_Topological'] is Topological_Topological


def test_topological_charge_density_squared_autocorrelation_included_for_finite_W():
    L = supervillain.lattice.Lattice(D=4, N=3)
    for W in (1, 2):
        S = supervillain.action.Villain(L, kappa=0.5, W=W)
        ensemble = _two_configuration_ensemble(S)
        assert TopologicalChargeDensitySquared.autocorrelation(ensemble)


def test_topological_charge_density_squared_autocorrelation_excluded_for_W_infinite():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=float('inf'))
    ensemble = _two_configuration_ensemble(S)

    assert not TopologicalChargeDensitySquared.autocorrelation(ensemble)


def test_total_and_density_excluded_from_autocorrelation():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    ensemble = _two_configuration_ensemble(S)

    # The total charge is identically zero (no variance), and the density is
    # field-valued (not a Scalar), so neither enters the autocorrelation-time
    # computation.
    assert not TopologicalCharge.autocorrelation(ensemble)
    assert not TopologicalChargeDensity.autocorrelation(ensemble)
