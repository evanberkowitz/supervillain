#!/usr/bin/env python

import tempfile

import h5py as h5
import numpy as np
import pytest

from supervillain.lattice.compact import Lattice, d, delta, star, push
from supervillain.lattice.two_dimensional import Lattice2D
import supervillain.lattice.interlaced as il


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
def test_interlaced_adjointness(D, N, p):
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


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_interlaced_star_d_star_equals_shifted_delta(D, N, p):
    # Same identity as compact but shift is +2*(1,...,1) in interlaced coords
    # (= +1 in physical coords), because the interlaced star pushes by (+1,...,+1).
    lat = il.Lattice(D=D, N=N)
    rng = np.random.default_rng(D * 100 + N * 10 + p)
    f = lat.random(p)
    sign = (-1) ** (D * (p - 1) + 1)
    lhs = il.star(D - p + 1, il.d(il.star(p, f)))
    rhs = il.delta(f).copy()
    for ax in range(D):
        rhs = np.roll(rhs, +2, axis=ax)
    assert np.allclose(lhs, sign * rhs)


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
