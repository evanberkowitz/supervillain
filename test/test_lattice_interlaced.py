#!/usr/bin/env python

import numpy as np
import pytest

import supervillain.lattice.interlaced as il


def _random_shift(D, rng):
    return tuple(int(s) for s in rng.integers(-3, 4, size=D))


# ---------------------------------------------------------------------------
# Translation: push / pull  (period is 2N in every interlaced direction)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_push_push_inverse(D, N, p):
    lat = il.Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p))
    assert (il.push(il.push(f, s), tuple(-x for x in s)) == f).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_push_pull_inverse(D, N, p):
    lat = il.Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p + 1))
    assert (il.push(il.pull(f, s), s) == f).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_pull_push_inverse(D, N, p):
    lat = il.Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p + 2))
    assert (il.pull(il.push(f, s), s) == f).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_pull_pull_inverse(D, N, p):
    lat = il.Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p + 3))
    assert (il.pull(il.pull(f, s), tuple(-x for x in s)) == f).all()


@pytest.mark.parametrize("D,N,p,direction", [(D, N, p, direction) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1) for direction in range(D)])
def test_push_period(D, N, p, direction):
    lat = il.Lattice(D=D, N=N)
    f = lat.random(p)
    s = tuple(2 * N if k == direction else 0 for k in range(D))
    assert (il.push(f, s) == f).all()


@pytest.mark.parametrize("D,N,p,direction", [(D, N, p, direction) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1) for direction in range(D)])
def test_pull_period(D, N, p, direction):
    lat = il.Lattice(D=D, N=N)
    f = lat.random(p)
    s = tuple(2 * N if k == direction else 0 for k in range(D))
    assert (il.pull(f, s) == f).all()


# ---------------------------------------------------------------------------
# Exterior derivative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D)])
def test_d_nilpotent(D, N, p):
    lat = il.Lattice(D=D, N=N)
    f = lat.random(p)
    assert np.isclose(il.d(il.d(f)), 0).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D)])
def test_d_nonzero(D, N, p):
    lat = il.Lattice(D=D, N=N)
    assert il.d(lat.random(p)).any()


# ---------------------------------------------------------------------------
# Codifferential
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_delta_nilpotent(D, N, p):
    lat = il.Lattice(D=D, N=N)
    f = lat.random(p)
    assert np.isclose(il.delta(il.delta(f)), 0).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_delta_nonzero(D, N, p):
    lat = il.Lattice(D=D, N=N)
    assert il.delta(lat.random(p)).any()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_adjointness(D, N, p):
    # After the delta sign fix, ⟨da, b⟩ = +⟨a, δb⟩ in the interlaced picture too.
    lat = il.Lattice(D=D, N=N)
    rng = np.random.default_rng(D * 100 + N * 10 + p)
    a = lat.random(p)
    b = lat.random(p + 1) if p < D else np.zeros((2 * N,) * D)
    if p == D:
        pytest.skip("d not defined on top form")
    lhs = (il.d(a) * b).sum()
    rhs = (a * il.delta(b)).sum()
    assert np.isclose(lhs, rhs)


# ---------------------------------------------------------------------------
# Hodge star
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_hodge_inner_product(D, N, p):
    lat = il.Lattice(D=D, N=N)
    a = lat.random(p); b = lat.random(p)
    assert np.isclose((a * b).sum(), il.wedge(p, D - p, a, il.star(b)).sum())


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_star_nonzero(D, N, p):
    lat = il.Lattice(D=D, N=N)
    assert il.star(lat.random(p)).any()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_star_d_star_equals_shifted_delta(D, N, p):
    # Same identity as compact but shift is +2*(1,...,1) in interlaced coords
    # (= +1 in physical coords), because the interlaced star pushes by (+1,...,+1).
    lat = il.Lattice(D=D, N=N)
    rng = np.random.default_rng(D * 100 + N * 10 + p)
    f = lat.random(p)
    sign = (-1) ** (D * (p - 1) + 1)
    lhs = il.star(il.d(il.star(f)))
    rhs = il.delta(f).copy()
    for ax in range(D):
        rhs = np.roll(rhs, +2, axis=ax)
    assert np.allclose(lhs, sign * rhs)


# ---------------------------------------------------------------------------
# Wedge product
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,n,m", [(D, N, n, m) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) if n + m <= D])
def test_wedge_nonzero(D, N, n, m):
    lat = il.Lattice(D=D, N=N)
    assert il.wedge(n, m, lat.random(n), lat.random(m)).any()


@pytest.mark.parametrize("D,N,n,m", [(D, N, n, m) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) if n + m <= D])
def test_wedge_bilinear(D, N, n, m):
    lat = il.Lattice(D=D, N=N)
    a = lat.random(n); b = lat.random(n); c = lat.random(m); e = lat.random(m)
    assert np.isclose(il.wedge(n, m, a + b, c), il.wedge(n, m, a, c) + il.wedge(n, m, b, c)).all()
    assert np.isclose(il.wedge(n, m, a, c + e), il.wedge(n, m, a, c) + il.wedge(n, m, a, e)).all()


@pytest.mark.parametrize("D,N,n,m,q", [(D, N, n, m, q) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) for q in range(D + 1) if n + m + q <= D])
def test_wedge_associative(D, N, n, m, q):
    lat = il.Lattice(D=D, N=N)
    a = lat.random(n); b = lat.random(m); c = lat.random(q)
    assert np.isclose(il.wedge(n + m, q, il.wedge(n, m, a, b), c), il.wedge(n, m + q, a, il.wedge(m, q, b, c))).all()


@pytest.mark.parametrize("D,N,n,m", [(D, N, n, m) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) if n + m + 1 <= D])
def test_leibniz_rule(D, N, n, m):
    lat = il.Lattice(D=D, N=N)
    a = lat.random(n); b = lat.random(m)
    LHS = il.d(il.wedge(n, m, a, b))
    RHS = il.wedge(n + 1, m, il.d(a), b) + (-1)**n * il.wedge(n, m + 1, a, il.d(b))
    assert np.isclose(LHS, RHS).all()
