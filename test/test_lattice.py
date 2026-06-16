#!/usr/bin/env python

import tempfile

import h5py as h5
import numpy as np
import pytest

from supervillain.lattice import Lattice, Form, d, delta, star, wedge, push, pull
from supervillain.lattice.two_dimensional import Lattice2D
import supervillain.lattice.interlaced as il


# ---------------------------------------------------------------------------
# Lattice round-trips
# ---------------------------------------------------------------------------

# N=3: every site is origin or boundary — a useful edge case.
# N=4: even N, exercises Nyquist-frequency wrapping in mod.
# N=5: interior sites {-2,-1,0,1,2} exercise the generic path.
# D=6, N=5 → 5^6 = 15625 sites, still fast for a pure reshape test.
@pytest.mark.parametrize("D,N", [(D, N) for D in range(2, 7) for N in (3, 4, 5)])
def test_linearize_coordinatize_roundtrip(D, N):
    L = Lattice(D=D, N=N)
    v = np.random.default_rng(D * 10 + N).standard_normal(L.dims)
    assert np.allclose(v, L.coordinatize(L.linearize(v)))


@pytest.mark.parametrize("D,N", [(D, N) for D in range(2, 7) for N in (3, 4, 5)])
def test_linearize_coordinatize_roundtrip_with_batch(D, N):
    # A leading batch dimension must survive the round-trip untouched.
    L = Lattice(D=D, N=N)
    v = np.random.default_rng(D * 10 + N + 100).standard_normal((7, *L.dims))
    assert np.allclose(v, L.coordinatize(L.linearize(v)))


@pytest.mark.parametrize("D,N", [(D, N) for D in range(2, 7) for N in (3, 4, 5)])
def test_distance_squared_from_origin_matches_R_squared(D, N):
    # distance_squared(x, 0) must equal R_squared at every site.
    L = Lattice(D=D, N=N)
    origin = np.zeros(D, dtype=int)
    for coord in L.coordinates:
        assert L.distance_squared(coord, origin) == np.sum(coord**2)


@pytest.mark.parametrize("D,N", [(D, N) for D in range(2, 7) for N in (3, 4, 5)])
def test_distance_squared_symmetry(D, N):
    # d(a, b) == d(b, a)
    L = Lattice(D=D, N=N)
    rng = np.random.default_rng(D * 10 + N + 200)
    a, b = rng.integers(-N, N, size=(2, D))
    assert L.distance_squared(a, b) == L.distance_squared(b, a)


@pytest.mark.parametrize("N", (3, 4, 5))
def test_distance_squared_pbc(N):
    # On a 1D lattice of size N the two sites furthest apart in naive integer
    # distance are actually only floor(N/2) steps apart on the torus.
    L = Lattice(D=1, N=N)
    # The site at coordinate N//2 is at most N//2 steps from the origin.
    far = np.array([N // 2])
    assert L.distance_squared(far, [0]) == (N // 2) ** 2


@pytest.mark.parametrize("D,N", [(D, N) for D in range(2, 7) for N in (3, 4, 5)])
def test_distance_squared_batch(D, N):
    # Batched call must give the same result as calling site by site.
    L = Lattice(D=D, N=N)
    origin = np.zeros(D, dtype=int)
    batch = L.distance_squared(L.coordinates, origin)
    assert batch.shape == (L.sites,)
    for i, coord in enumerate(L.coordinates):
        assert batch[i] == L.distance_squared(coord, origin)


@pytest.mark.parametrize("D,N", [(D, N) for D in range(2, 5) for N in (3, 4, 5)])
def test_lattice_h5_roundtrip(D, N):
    L = Lattice(D=D, N=N)
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        with h5.File(f.name, 'w') as hf:
            L.to_h5(hf.create_group('lattice'))
            L2 = Lattice.from_h5(hf['lattice'])
    assert L2.D == L.D
    assert L2.N == L.N
    assert np.array_equal(L2.coords, L.coords)
    assert np.array_equal(L2.coordinates, L.coordinates)


@pytest.mark.parametrize("N", (3, 4, 5))
def test_lattice2d_h5_roundtrip(N):
    L = Lattice2D(N)
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        with h5.File(f.name, 'w') as hf:
            L.to_h5(hf.create_group('lattice'))
            L2 = Lattice2D.from_h5(hf['lattice'])
    assert L2.D == L.D
    assert L2.N == L.N
    assert L2.nt == L.nt
    assert L2.nx == L.nx
    assert np.array_equal(L2.coords, L.coords)


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_compact_to_interlaced(D, N, p):
    lat = Lattice(D=D, N=N)
    a = lat.random(p)
    big = a.to_interlaced()
    odds = np.mod(np.mgrid[(slice(0, 2 * N),) * D], 2).sum(axis=0)
    assert np.all(big[odds != p] == 0)
    assert not np.all(big[odds == p] == 0)


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_compact_to_interlaced_roundtrip(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    assert (f == Form.from_interlaced(p, f.to_interlaced())).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_interlaced_to_compact_roundtrip(D, N, p):
    lat = il.Lattice(D=D, N=N)
    data = lat.random(p)
    assert (Form.from_interlaced(p, data).to_interlaced() == data).all()


# ---------------------------------------------------------------------------
# Translation: push / pull
# ---------------------------------------------------------------------------

def _random_shift(D, rng):
    return tuple(int(s) for s in rng.integers(-3, 4, size=D))


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_push_push_inverse(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p))
    assert (push(push(f, s), tuple(-x for x in s)) == f).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_push_pull_inverse(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p + 1))
    assert (push(pull(f, s), s) == f).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_pull_push_inverse(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p + 2))
    assert (pull(push(f, s), s) == f).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_pull_pull_inverse(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p + 3))
    assert (pull(pull(f, s), tuple(-x for x in s)) == f).all()


@pytest.mark.parametrize("D,N,p,direction", [(D, N, p, direction) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1) for direction in range(D)])
def test_push_period(D, N, p, direction):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = tuple(N if k == direction else 0 for k in range(D))
    assert (push(f, s) == f).all()


@pytest.mark.parametrize("D,N,p,direction", [(D, N, p, direction) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1) for direction in range(D)])
def test_pull_period(D, N, p, direction):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = tuple(N if k == direction else 0 for k in range(D))
    assert (pull(f, s) == f).all()


# ---------------------------------------------------------------------------
# Exterior derivative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D)])
def test_d_nilpotent(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    assert np.isclose(np.asarray(d(d(f))), 0).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D)])
def test_d_nonzero(D, N, p):
    lat = Lattice(D=D, N=N)
    assert np.asarray(d(lat.random(p))).any()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D)])
def test_compact_d_matches_interlaced(D, N, p):
    lat = Lattice(D=D, N=N)
    a = lat.random(p)
    assert (d(a).to_interlaced() == il.d(a.to_interlaced())).all()


# ---------------------------------------------------------------------------
# Codifferential
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_delta_nilpotent(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    assert np.isclose(np.asarray(delta(delta(f))), 0).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_delta_nonzero(D, N, p):
    lat = Lattice(D=D, N=N)
    assert np.asarray(delta(lat.random(p))).any()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D)])
def test_compact_adjointness(D, N, p):
    lat = Lattice(D=D, N=N)
    a = lat.random(p)
    b = lat.random(p + 1)
    assert np.isclose((d(a) * b).sum(), (a * delta(b)).sum())


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_compact_delta_matches_interlaced(D, N, p):
    lat = Lattice(D=D, N=N)
    a = lat.random(p)
    assert (delta(a).to_interlaced() == il.delta(a.to_interlaced())).all()


# ---------------------------------------------------------------------------
# Hodge star
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_hodge_inner_product(D, N, p):
    lat = Lattice(D=D, N=N)
    a = lat.random(p); b = lat.random(p)
    assert np.isclose((a * b).sum(), wedge(a, star(b)).sum())


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_star_nonzero(D, N, p):
    lat = Lattice(D=D, N=N)
    assert np.asarray(star(lat.random(p))).any()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_star_d_star_equals_shifted_delta(D, N, p):
    # On this compact lattice the continuum identity δ = (-1)^{D(p-1)+1} ★d★
    # acquires a spatial shift: ★d★f = (-1)^{D(p-1)+1} · push(δf, (1,...,1))
    # where push(f, (1,...,1))[n] = f[n − (1,...,1)].
    lat = Lattice(D=D, N=N)
    rng = np.random.default_rng(D * 100 + N * 10 + p)
    f = lat.random(p)
    sign = (-1) ** (D * (p - 1) + 1)
    lhs = np.asarray(star(d(star(f))))
    rhs = sign * np.asarray(push(delta(f), (1,) * D))
    assert np.allclose(lhs, rhs)


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_compact_star_matches_interlaced(D, N, p):
    lat = Lattice(D=D, N=N)
    a = lat.random(p)
    assert (star(a).to_interlaced() == il.star(a.to_interlaced())).all()


# ---------------------------------------------------------------------------
# Wedge product
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,n,m", [(D, N, n, m) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) if n + m <= D])
def test_wedge_nonzero(D, N, n, m):
    lat = Lattice(D=D, N=N)
    assert np.asarray(wedge(lat.random(n), lat.random(m))).any()


@pytest.mark.parametrize("D,N,n,m", [(D, N, n, m) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) if n + m <= D])
def test_wedge_bilinear(D, N, n, m):
    lat = Lattice(D=D, N=N)
    a = lat.random(n); b = lat.random(n); c = lat.random(m); e = lat.random(m)
    assert np.isclose(wedge(a + b, c), wedge(a, c) + wedge(b, c)).all()
    assert np.isclose(wedge(a, c + e), wedge(a, c) + wedge(a, e)).all()


@pytest.mark.parametrize("D,N,n,m,q", [(D, N, n, m, q) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) for q in range(D + 1) if n + m + q <= D])
def test_wedge_associative(D, N, n, m, q):
    lat = Lattice(D=D, N=N)
    a = lat.random(n); b = lat.random(m); c = lat.random(q)
    assert np.isclose(np.asarray(wedge(wedge(a, b), c)), np.asarray(wedge(a, wedge(b, c)))).all()


@pytest.mark.parametrize("D,N,n,m", [(D, N, n, m) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) if n + m + 1 <= D])
def test_leibniz_rule(D, N, n, m):
    lat = Lattice(D=D, N=N)
    a = lat.random(n); b = lat.random(m)
    LHS = d(wedge(a, b))
    RHS = wedge(d(a), b) + (-1)**n * wedge(a, d(b))
    assert np.isclose(np.asarray(LHS), np.asarray(RHS)).all()


@pytest.mark.parametrize("D,N,p,q", [(D, N, p, q) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1) for q in range(D + 1) if p + q <= D])
def test_compact_wedge_matches_interlaced(D, N, p, q):
    lat = Lattice(D=D, N=N)
    a = lat.random(p); b = lat.random(q)
    assert np.isclose(wedge(a, b).to_interlaced(), il.wedge(p, q, a.to_interlaced(), b.to_interlaced())).all()
