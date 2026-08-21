#!/usr/bin/env python
"""Dedicated tests for the sparse form operators d_sparse, coface_sum_at, face_sum_at.

Two flavors:
  - d_sparse is input-sparse (like delta_sparse): δ/d of a form supported on one
    component+color, bit-identical to the dense operator, supporting in-place
    accumulation for incremental field maintenance.
  - coface_sum_at / face_sum_at are output-sparse: the reduction evaluated at a
    single output component+color, bit-identical to indexing the dense reduction."""

import numpy as np
import pytest

from supervillain.lattice import (
    Lattice, d, d_sparse, coface_sum_at, face_sum_at,
)


def _dense_from_sparse(L, p, component, color, values):
    f = L.zeros(p, dtype=values.dtype)
    f[component][color] = values
    return f


# d raises degree, so degrees 0 .. D-1.
_DNp_d = [(D, N, p) for D in range(2, 6) for N in (3, 4, 5) for p in range(0, D)]


@pytest.mark.parametrize("D,N,p", _DNp_d)
@pytest.mark.parametrize("dtype", [int, float])
def test_d_sparse_matches_dense_every_component_and_color(D, N, p, dtype):
    L = Lattice(D=D, N=N)
    rng = np.random.default_rng(D * 100 + N * 10 + p)
    for component in range(len(L.components[p])):
        for color in L.checkerboarding:
            k = len(color[0])
            values = (rng.integers(-4, 5, k).astype(int) if dtype is int
                      else rng.standard_normal(k))
            f = _dense_from_sparse(L, p, component, color, values)
            dense = np.asarray(d(f))
            sparse = d_sparse(L, p, component, color, values)
            assert (dense == sparse).all(), \
                f"D={D} N={N} p={p} comp={component}: max|Δ|={np.abs(dense - sparse).max()}"


@pytest.mark.parametrize("D,N,p", _DNp_d)
def test_d_sparse_accumulates_like_linearity(D, N, p):
    # d(v + Δv) == d(v) + d(Δv), with the second term added in place — the
    # incremental dφ / n maintenance the Villain updates rely on.
    L = Lattice(D=D, N=N)
    rng = np.random.default_rng(5 * D + N + p)
    v = L.zeros(p, dtype=int)
    v[...] = rng.integers(-3, 4, v.shape)
    component = int(rng.integers(0, len(L.components[p])))
    color = L.checkerboarding[int(rng.integers(0, len(L.checkerboarding)))]
    values = rng.integers(-4, 5, len(color[0]))

    d_v = np.asarray(d(v)).copy()
    d_sparse(L, p, component, color, values, out=d_v)

    change = _dense_from_sparse(L, p, component, color, values)
    assert (d_v == np.asarray(d(v + change))).all()


@pytest.mark.parametrize("D,N,p", _DNp_d)
def test_d_sparse_preserves_integer_dtype(D, N, p):
    L = Lattice(D=D, N=N)
    color = L.checkerboarding[0]
    out = d_sparse(L, p, 0, color, np.ones(len(color[0]), dtype=int))
    assert out.dtype == np.dtype(int)


def test_d_sparse_rejects_top_degree():
    L = Lattice(D=3, N=4)
    with pytest.raises(ValueError):
        d_sparse(L, 3, 0, L.checkerboarding[0], np.ones(1))


# coface_sum_at: p -> p+1, so the output component indexes components[p+1].
@pytest.mark.parametrize("D,N,p", _DNp_d)
@pytest.mark.parametrize("dtype", [int, float])
def test_coface_sum_at_matches_dense(D, N, p, dtype):
    L = Lattice(D=D, N=N)
    rng = np.random.default_rng(D * 7 + N + p)
    data = (rng.integers(-4, 5, (len(L.components[p]),) + L.dims) if dtype is int
            else rng.standard_normal((len(L.components[p]),) + L.dims))
    f = L.form(p, dtype=dtype); f[...] = data
    dense = np.asarray(f.coface_sum())
    for component in range(len(L.components[p + 1])):
        for color in L.checkerboarding:
            got = coface_sum_at(f, component, color)
            assert (dense[component][color] == got).all()
            assert got.dtype == np.dtype(dtype)


# face_sum_at: p -> p-1, degrees 1 .. D.
@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 6) for N in (3, 4, 5) for p in range(1, D + 1)])
@pytest.mark.parametrize("dtype", [int, float])
def test_face_sum_at_matches_dense(D, N, p, dtype):
    L = Lattice(D=D, N=N)
    rng = np.random.default_rng(D * 13 + N + p)
    data = (rng.integers(-4, 5, (len(L.components[p]),) + L.dims) if dtype is int
            else rng.standard_normal((len(L.components[p]),) + L.dims))
    f = L.form(p, dtype=dtype); f[...] = data
    dense = np.asarray(f.face_sum())
    for component in range(len(L.components[p - 1])):
        for color in L.checkerboarding:
            got = face_sum_at(f, component, color)
            assert (dense[component][color] == got).all()
            assert got.dtype == np.dtype(dtype)
