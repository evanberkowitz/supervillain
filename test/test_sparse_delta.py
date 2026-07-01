#!/usr/bin/env python
"""Dedicated tests for delta_sparse — the single-component, single-color δ.

delta_sparse must be bit-identical to the dense delta() of the equivalent dense
form, across dimensions, form degrees, components, colors, and dtypes; and it
must support incremental accumulation into an existing δ array (the use that
lets the Worldline updates maintain δv without recomputing it)."""

import numpy as np
import pytest

from supervillain.lattice import Lattice, delta, delta_sparse


# A broad sweep: D = 2..5, a few N (even and odd -> different checkerboardings),
# every degree that has a codifferential, both integer and floating dtypes.
_DNp = [
    (D, N, p)
    for D in range(2, 6)
    for N in (3, 4, 5)
    for p in range(1, D + 1)
]


def _dense_from_sparse(L, p, component, color, values):
    """A dense p-form that is `values` on `component` at `color`, zero elsewhere."""
    f = L.zeros(p, dtype=values.dtype)
    f[component][color] = values
    return f


@pytest.mark.parametrize("D,N,p", _DNp)
@pytest.mark.parametrize("dtype", [int, float])
def test_delta_sparse_matches_dense_every_component_and_color(D, N, p, dtype):
    L = Lattice(D=D, N=N)
    rng = np.random.default_rng(D * 1000 + N * 10 + p)
    n_comps = len(L.components[p])
    for component in range(n_comps):
        for color in L.checkerboarding:
            k = len(color[0])
            values = (rng.integers(-4, 5, k).astype(int) if dtype is int
                      else rng.standard_normal(k))
            f = _dense_from_sparse(L, p, component, color, values)
            dense = np.asarray(delta(f))
            sparse = delta_sparse(L, p, component, color, values)
            assert (dense == sparse).all(), \
                f"D={D} N={N} p={p} comp={component}: max|Δ|={np.abs(dense - sparse).max()}"


@pytest.mark.parametrize("D,N,p", _DNp)
def test_delta_sparse_preserves_integer_dtype(D, N, p):
    L = Lattice(D=D, N=N)
    color = L.checkerboarding[0]
    values = np.ones(len(color[0]), dtype=int)
    out = delta_sparse(L, p, 0, color, values)
    assert out.dtype == np.dtype(int)


@pytest.mark.parametrize("D,N,p", _DNp)
def test_delta_sparse_accumulates_like_linearity(D, N, p):
    # The incremental-maintenance use: δ(v + Δv) == δ(v) + δ(Δv), with the second
    # term added in place by delta_sparse(out=...).  This is exactly how the
    # Worldline updates keep δv current after an accepted change.
    L = Lattice(D=D, N=N)
    rng = np.random.default_rng(7 * D + N + p)
    v = L.zeros(p, dtype=int)
    v[...] = rng.integers(-3, 4, v.shape)
    component = rng.integers(0, len(L.components[p]))
    color = L.checkerboarding[rng.integers(0, len(L.checkerboarding))]
    values = rng.integers(-4, 5, len(color[0]))

    delta_v = np.asarray(delta(v)).copy()
    delta_sparse(L, p, int(component), color, values, out=delta_v)

    change = _dense_from_sparse(L, p, int(component), color, values)
    expected = np.asarray(delta(v + change))
    assert (delta_v == expected).all()


@pytest.mark.parametrize("D,N,p", _DNp)
def test_delta_sparse_returns_the_out_array(D, N, p):
    L = Lattice(D=D, N=N)
    color = L.checkerboarding[0]
    values = np.ones(len(color[0]), dtype=float)
    out = np.zeros((len(L.components[p - 1]),) + (N,) * D)
    returned = delta_sparse(L, p, 0, color, values, out=out)
    assert returned is out


def test_delta_sparse_rejects_degree_zero():
    L = Lattice(D=3, N=4)
    with pytest.raises(ValueError):
        delta_sparse(L, 0, 0, L.checkerboarding[0], np.ones(1))


def test_delta_sparse_zero_values_give_zero():
    L = Lattice(D=4, N=5)
    color = L.checkerboarding[0]
    out = delta_sparse(L, 2, 0, color, np.zeros(len(color[0])))
    assert (out == 0).all()
