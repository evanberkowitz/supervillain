#!/usr/bin/env python
# Tests that the layout bridge, exterior derivative, and Villain action all behave consistently.

import numpy as np
import pytest

import supervillain
import supervillain.layout as layout
from supervillain.lattice.compact import Lattice, d


@pytest.fixture
def lattice():
    return Lattice(D=2, N=5)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_layout_round_trip_phi(lattice, rng):
    phi = lattice.zeros(0)
    phi[0] = rng.normal(size=(lattice.N,) * lattice.D)
    form = layout.to_form(np.asarray(phi[0]), degree=0, lattice2d=supervillain.Lattice2D(lattice.N))
    back = layout.from_form(form)
    assert np.allclose(phi[0], back)


def test_layout_round_trip_n(lattice, rng):
    n = lattice.zeros(1, dtype=int)
    n[:] = rng.integers(-2, 3, size=n.shape)
    L2D = supervillain.Lattice2D(lattice.N)
    form = layout.to_form(np.asarray(n), degree=1, lattice2d=L2D, dtype=int)
    back = layout.from_form(form)
    assert np.array_equal(n, back)


def test_exterior_derivative_agreement(lattice, rng):
    L2D = supervillain.Lattice2D(lattice.N)
    phi_arr = rng.normal(size=L2D.dims)
    d_prod = L2D.d(0, phi_arr)
    phi_f = layout.to_form(phi_arr, degree=0, lattice2d=L2D)
    d_compact = np.asarray(d(phi_f))
    assert np.allclose(d_prod, d_compact)


def test_villain_action_value(lattice, rng):
    phi = lattice.zeros(0)
    phi[0] = rng.normal(size=(lattice.N,) * lattice.D)
    n = lattice.zeros(1, dtype=int)
    n[:] = rng.integers(-2, 3, size=n.shape)

    kappa = 1.7
    S = supervillain.Villain(lattice, kappa)
    computed = S(phi, n)
    expected = (kappa / 2) * float(((d(phi) - 2 * np.pi * n) ** 2).sum())
    assert computed == pytest.approx(expected)


def test_metropolis_delta(lattice, rng):
    phi = lattice.zeros(0)
    phi[0] = rng.normal(size=(lattice.N,) * lattice.D)
    n = lattice.zeros(1, dtype=int)
    n[:] = rng.integers(-2, 3, size=n.shape)

    kappa = 2.0
    S = supervillain.Villain(lattice, kappa)

    dphi = lattice.zeros(0)
    dphi[0] = rng.normal(scale=0.1, size=(lattice.N,) * lattice.D)
    dn = lattice.zeros(1, dtype=int)
    dn[:] = rng.integers(-1, 2, size=dn.shape)

    dS = S(phi + dphi, n + dn) - S(phi, n)
    expected = (kappa / 2) * float(
        ((d(phi + dphi) - 2 * np.pi * (n + dn)) ** 2).sum()
        - ((d(phi) - 2 * np.pi * n) ** 2).sum()
    )
    assert dS == pytest.approx(expected)
