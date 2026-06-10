#!/usr/bin/env python

import numpy as np
import pytest

import supervillain
import supervillain.layout as layout
from supervillain.compact import d
from supervillain.compact.villain import Villain as CompactVillain


@pytest.fixture
def lattice():
    return supervillain.Lattice2D(5)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_layout_round_trip_phi(lattice, rng):
    phi = rng.normal(size=lattice.dims)
    form = layout.to_form(phi, degree=0, lattice2d=lattice)
    back = layout.from_form(form)
    assert np.allclose(phi, back)


def test_layout_round_trip_n(lattice, rng):
    n = rng.integers(-2, 3, size=(lattice.dim,) + lattice.dims)
    form = layout.to_form(n, degree=1, lattice2d=lattice, dtype=int)
    back = layout.from_form(form)
    assert np.array_equal(n, back)


def test_exterior_derivative_agreement(lattice, rng):
    phi = rng.normal(size=lattice.dims)
    d_prod = lattice.d(0, phi)
    d_compact = np.asarray(d(layout.to_form(phi, degree=0, lattice2d=lattice)))
    assert np.allclose(d_prod, d_compact)


def test_villain_action_agreement(lattice, rng):
    phi = rng.normal(size=lattice.dims)
    n = rng.integers(-2, 3, size=(lattice.dim,) + lattice.dims)

    kappa = 1.7
    S_prod = supervillain.Villain(lattice, kappa)(phi, n)

    phi_f = layout.to_form(phi, degree=0, lattice2d=lattice)
    n_f = layout.to_form(n, degree=1, lattice2d=lattice, dtype=int)
    S_compact = CompactVillain(kappa)(phi_f, n_f)

    assert S_prod == pytest.approx(S_compact)


def test_metropolis_delta_agreement(lattice, rng):
    phi = rng.normal(size=lattice.dims)
    n = rng.integers(-2, 3, size=(lattice.dim,) + lattice.dims)
    kappa = 2.0
    S = supervillain.Villain(lattice, kappa)

    phi_f = layout.to_form(phi, degree=0, lattice2d=lattice)
    n_f = layout.to_form(n, degree=1, lattice2d=lattice, dtype=int)
    Sc = CompactVillain(kappa)

    dphi = rng.normal(scale=0.1, size=lattice.dims)
    dn = rng.integers(-1, 2, size=(lattice.dim,) + lattice.dims)

    dS_prod = S(phi + dphi, n + dn) - S(phi, n)

    dphi_f = layout.to_form(phi + dphi, degree=0, lattice2d=lattice)
    dn_f = layout.to_form(n + dn, degree=1, lattice2d=lattice, dtype=int)
    dS_compact = Sc(dphi_f, dn_f) - Sc(phi_f, n_f)

    assert dS_prod == pytest.approx(dS_compact)
