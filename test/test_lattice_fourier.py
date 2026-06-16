#!/usr/bin/env python

r"""
Tests for :mod:`supervillain.lattice.fourier` — the spectral operator module.

This file is a close analog of ``test_lattice_interlaced.py``, with the
following differences reflecting the spectral formulation:

1. Push/pull period is ``N`` (not ``2N``): the compact layout has physical
   period N, whereas the interlaced layout has period 2N per direction.
2. ``test_star_d_star_equals_delta``: the identity
   $\delta = (-1)^{D(k+1)+1}\,\star\,d\,\star$ holds **without** any
   translational shift (compare to the shifted version in both compact and
   interlaced tests).
3. ``test_wedge_anticommutative``: the pointwise wedge satisfies
   $(a \wedge b) = (-1)^{nm}(b \wedge a)$ exactly at every site — a property
   that fails for both the compact and interlaced finite-difference wedges.
4. **No** ``test_leibniz_rule``: the Leibniz rule $d(a\wedge b) = da\wedge b
   + (-1)^n a\wedge db$ does not hold at finite $N$ for the spectral d with
   pointwise wedge.  The proof requires $q_k = q_m + q_{k-m}$ for all
   frequency indices, which fails due to Nyquist aliasing (two modes at $\pm\pi$
   convolve to DC but $(-\pi)+(-\pi)\neq 0$).  The failure is $O(1)$ for
   random fields; it vanishes in the $N\to\infty$ limit for smooth forms.
"""

import numpy as np
import pytest

import supervillain.lattice.fourier as fl
from supervillain.lattice.compact import Lattice


def _random_shift(D, rng):
    return tuple(int(s) for s in rng.integers(-3, 4, size=D))


# ---------------------------------------------------------------------------
# Translation: push / pull  (period is N in every compact direction)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_push_push_inverse(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p))
    assert (fl.push(fl.push(f, s), tuple(-x for x in s)) == f).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_push_pull_inverse(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p + 1))
    assert (fl.push(fl.pull(f, s), s) == f).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_pull_push_inverse(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p + 2))
    assert (fl.pull(fl.push(f, s), s) == f).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_pull_pull_inverse(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = _random_shift(D, np.random.default_rng(D * 100 + N * 10 + p + 3))
    assert (fl.pull(fl.pull(f, s), tuple(-x for x in s)) == f).all()


@pytest.mark.parametrize("D,N,p,direction", [(D, N, p, direction) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1) for direction in range(D)])
def test_push_period(D, N, p, direction):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = tuple(N if k == direction else 0 for k in range(D))
    assert (fl.push(f, s) == f).all()


@pytest.mark.parametrize("D,N,p,direction", [(D, N, p, direction) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1) for direction in range(D)])
def test_pull_period(D, N, p, direction):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    s = tuple(N if k == direction else 0 for k in range(D))
    assert (fl.pull(f, s) == f).all()


# ---------------------------------------------------------------------------
# Exterior derivative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D)])
def test_d_nilpotent(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    assert np.isclose(np.asarray(fl.d(fl.d(f))), 0).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D)])
def test_d_nonzero(D, N, p):
    lat = Lattice(D=D, N=N)
    assert np.asarray(fl.d(lat.random(p))).any()


# ---------------------------------------------------------------------------
# Codifferential
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_delta_nilpotent(D, N, p):
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    assert np.isclose(np.asarray(fl.delta(fl.delta(f))), 0).all()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_delta_nonzero(D, N, p):
    lat = Lattice(D=D, N=N)
    assert np.asarray(fl.delta(lat.random(p))).any()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_adjointness(D, N, p):
    lat = Lattice(D=D, N=N)
    a = lat.random(p)
    b = lat.random(p + 1) if p < D else np.zeros(lat.zeros(D).shape)
    if p == D:
        pytest.skip("d not defined on top form")
    lhs = (fl.d(a) * b).sum()
    rhs = (a * fl.delta(b)).sum()
    assert np.isclose(lhs, rhs)


# ---------------------------------------------------------------------------
# Hodge star
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_hodge_inner_product(D, N, p):
    lat = Lattice(D=D, N=N)
    a = lat.random(p)
    b = lat.random(p)
    assert np.isclose((a * b).sum(), fl.wedge(a, fl.star(b)).sum())


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(D + 1)])
def test_star_nonzero(D, N, p):
    lat = Lattice(D=D, N=N)
    assert np.asarray(fl.star(lat.random(p))).any()


@pytest.mark.parametrize("D,N,p", [(D, N, p) for D in range(2, 5) for N in (3, 4, 5) for p in range(1, D + 1)])
def test_star_d_star_equals_delta(D, N, p):
    # With spectral d and pointwise ★, the continuum identity holds exactly:
    # ★d★ f = (-1)^{D(p-1)+1} δf   with NO translational shift.
    lat = Lattice(D=D, N=N)
    f = lat.random(p)
    sign = (-1) ** (D * (p - 1) + 1)
    lhs = np.asarray(fl.star(fl.d(fl.star(f))))
    rhs = sign * np.asarray(fl.delta(f))
    assert np.allclose(lhs, rhs)


# ---------------------------------------------------------------------------
# Wedge product
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,N,n,m", [(D, N, n, m) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) if n + m <= D])
def test_wedge_nonzero(D, N, n, m):
    lat = Lattice(D=D, N=N)
    assert np.asarray(fl.wedge(lat.random(n), lat.random(m))).any()


@pytest.mark.parametrize("D,N,n,m", [(D, N, n, m) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) if n + m <= D])
def test_wedge_bilinear(D, N, n, m):
    lat = Lattice(D=D, N=N)
    a = lat.random(n); b = lat.random(n); c = lat.random(m); e = lat.random(m)
    assert np.isclose(fl.wedge(a + b, c), fl.wedge(a, c) + fl.wedge(b, c)).all()
    assert np.isclose(fl.wedge(a, c + e), fl.wedge(a, c) + fl.wedge(a, e)).all()


@pytest.mark.parametrize("D,N,n,m,q", [(D, N, n, m, q) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) for q in range(D + 1) if n + m + q <= D])
def test_wedge_associative(D, N, n, m, q):
    lat = Lattice(D=D, N=N)
    a = lat.random(n); b = lat.random(m); c = lat.random(q)
    assert np.isclose(fl.wedge(fl.wedge(a, b), c), fl.wedge(a, fl.wedge(b, c))).all()


@pytest.mark.parametrize("D,N,n,m", [(D, N, n, m) for D in range(2, 5) for N in (3, 4, 5) for n in range(D + 1) for m in range(D + 1) if n + m <= D])
def test_wedge_anticommutative(D, N, n, m):
    # The pointwise wedge is exactly anti-commutative: a∧b = (-1)^{nm} b∧a
    # at every site, with no approximation. This fails for the finite-difference wedge.
    lat = Lattice(D=D, N=N)
    a = lat.random(n); b = lat.random(m)
    assert np.isclose(fl.wedge(a, b), (-1)**(n * m) * fl.wedge(b, a)).all()
