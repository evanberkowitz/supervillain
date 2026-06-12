#!/usr/bin/env python
# coding: utf-8
#
# interlaced.py — Differential forms on a D-dimensional hypercubic lattice,
# stored in the interlaced representation.
#
# REPRESENTATION
# --------------
# A p-form field on an N^D lattice is stored in a (2N)^D array.
# Site (x_0, …, x_{D-1}) belongs to the p-form if exactly p of the
# x_k are odd.  All other array elements are zero.
#
# D is not a global constant.  Every operator function infers D from data.ndim
# (since the array shape is (2N,)*D).  A per-D cache stores the
# permutation-index tables so they are built at most once per dimension.

from itertools import permutations, product, cycle
from math import comb
import numpy as np


# ---------------------------------------------------------------------------
# Per-D structure cache
# ---------------------------------------------------------------------------
# All operators need:
#   ε[k]   — k-th standard basis vector (shift by 1 in direction k)
#   odd[p] — for each p-form component, the tuple of odd direction indices
#   even[p]— the tuple of even direction indices
#   idx[p] — the corresponding multi-dimensional numpy slice
#
# These depend only on D, so we compute them once and cache.

_cache = {}

def _structures(D):
    """
    Return (ε, odd, even, idx) for dimension D, building and caching on
    first call.

    odd[p]  — tuple of tuples; odd[p][i] lists the directions that are
               odd for the i-th p-form component.
    even[p] — same but for even directions.
    idx[p]  — tuple of tuples of slices; idx[p][i] selects the i-th
               p-form component from a (2N,…,2N) array.
    """
    if D not in _cache:
        ε = np.eye(D, dtype=int)

        # All 0/1 patterns of length D with exactly p ones, sorted.
        start = {
            p: tuple(sorted(set(permutations((0,)*(D-p) + (1,)*p))))
            for p in range(D + 1)
        }

        odd_dirs  = {
            p: tuple(tuple(k for k, j in enumerate(J) if j == 1) for J in S)
            for p, S in start.items()
        }
        even_dirs = {
            p: tuple(tuple(k for k, j in enumerate(J) if j == 0) for J in S)
            for p, S in start.items()
        }
        # slice(1,None,2) picks odd indices; slice(0,None,2) picks even.
        slices = {
            p: tuple(tuple(slice(j, None, 2) for j in J) for J in S)
            for p, S in start.items()
        }

        _cache[D] = (ε, odd_dirs, even_dirs, slices)

    return _cache[D]


# ---------------------------------------------------------------------------
# Push and pull  (lattice translations)
# ---------------------------------------------------------------------------

def push(data, shift):
    """
    Translate data by `shift` steps: result[n + shift] = data[n]  (periodic).

    shift — tuple or array of D integers, one per spatial direction.
    D is inferred from data.ndim.
    """
    result = data
    for axis, s in enumerate(shift):
        if s:
            result = np.roll(result, shift=s, axis=axis)
    return result


def pull(data, shift):
    """
    Pull data from `shift` steps away: result[n] = data[n + shift].

    Equivalent to push with the sign of shift reversed.
    """
    return push(data, tuple(-s for s in shift))


# ---------------------------------------------------------------------------
# Exterior derivative  d : Ω^p → Ω^{p+1}
# ---------------------------------------------------------------------------

def d(data):
    """
    Exterior derivative.  Infers D from data.ndim.

    For each (p+1)-form output site with odd directions O = (o_0, …, o_p),
    the contribution is

        Σ_{j}  σ_j · (pull(data, ε[o_j]) − pull(data, −ε[o_j]))[I]

    where σ alternates +1, −1 over the elements of O and I is the
    multi-dimensional slice selecting those sites.

    ε, odd, and idx are fetched from the per-D cache rather than globals.
    """
    D = data.ndim
    ε, odd, even, idx = _structures(D)
    out = np.zeros_like(data)
    for n in range(1, D + 1):
        for O, I in zip(odd[n], idx[n]):
            for σ, o in zip(cycle((+1, -1)), O):
                out[I] += σ * (pull(data, ε[o]) - pull(data, -ε[o]))[I]
    return out


# ---------------------------------------------------------------------------
# Codifferential  δ : Ω^p → Ω^{p-1}
# ---------------------------------------------------------------------------

def delta(data):
    """
    Codifferential (formal adjoint of d).  Infers D from data.ndim.

    For each (p-1)-form output site with even directions E = (e_0, …),
    the contribution at position i in E from direction e is

        (−1)^(e − i) · (pull(data, ε[e]) − pull(data, −ε[e]))[I]

    """
    D = data.ndim
    ε, odd, even, idx = _structures(D)
    out = np.zeros_like(data)
    for n in range(0, D):
        for E, I in zip(even[n], idx[n]):
            for i, e in enumerate(E):
                out[I] -= (-1)**(e - i) * (pull(data, ε[e]) - pull(data, -ε[e]))[I]
    return out

δ = delta


# ---------------------------------------------------------------------------
# Wedge product  ∧ : Ω^n × Ω^m → Ω^{n+m}
# ---------------------------------------------------------------------------

def _assign_sign(t):
    """
    Sign of a 0/1 assignment tuple.

    Counts the inversions where a 1 (→ a-shift) precedes a 0 (→ b-shift).
    """
    zeros_remaining = t.count(0)
    inversions = 0
    for x in t:
        if x == 0:
            zeros_remaining -= 1
        else:
            inversions += zeros_remaining
    return (-1) ** inversions


# Cache of assignment tuples per (D, n, m) so permutations are built once.
_assign_cache = {}

def _assignments(D, n, m):
    key = (D, n, m)
    if key not in _assign_cache:
        _assign_cache[key] = tuple(set(permutations((0,)*n + (1,)*m)))
    return _assign_cache[key]


def wedge(n, m, a, b):
    """
    Wedge product of an n-form a and an m-form b.  Infers D from a.ndim.

    For each (n+m)-form output site with odd directions O and each way to
    assign those directions to a (assignment=1) or b (assignment=0):

        w[I] += sign · pull(a, −a_shift) · pull(b, +b_shift)

    where I selects the (n+m)-form sites and sign is the inversion count
    of the 0/1 assignment tuple.
    """
    D = a.ndim
    ε, odd, even, idx = _structures(D)
    w = np.zeros_like(a)
    for O, ASSIGN in product(odd[n + m], _assignments(D, n, m)):
        sign    = _assign_sign(ASSIGN)
        a_shift = np.zeros(D, dtype=int)
        b_shift = np.zeros(D, dtype=int)
        for o, asgn in zip(O, ASSIGN):
            if asgn == 1:
                a_shift[o] += 1
            else:
                b_shift[o] += 1
        I = tuple(slice(1 if i in O else 0, None, 2) for i in range(D))
        w[I] += (sign * pull(a, -a_shift) * pull(b, +b_shift))[I]
    return w


# ---------------------------------------------------------------------------
# Hodge star  ★ : Ω^p → Ω^{D-p}
# ---------------------------------------------------------------------------

def _perm_sign(seq):
    """Sign of the permutation that sorts a sequence of distinct integers."""
    return (-1) ** sum(
        1 for i in range(len(seq)) for j in range(i + 1, len(seq))
        if seq[i] > seq[j]
    )


def star(p, data):
    """
    Hodge star of a p-form, returning a (D-p)-form.  Infers D from data.ndim.

    The spatial shift push(data, (+1,)*D) maps every p-form site to the
    corresponding (D-p)-form site (it flips all parities).  Each output
    component J then receives the permutation sign σ(I, J), where I is
    the complement of J:

        result[J_sites] = σ(I, J) · shifted[J_sites]

    σ(I, J) = sign of the permutation sorting the concatenation (I, J).

    For p=0 and p=D all signs happen to be +1, which is why a bare push
    without signs passes those cases but fails for intermediate degrees.
    """
    D = data.ndim
    ε, odd, even, idx = _structures(D)
    shifted = push(data, (+1,) * D)
    result  = np.zeros_like(data)
    for J_dirs, J_idx in zip(odd[D - p], idx[D - p]):
        I_dirs = tuple(k for k in range(D) if k not in set(J_dirs))
        sign   = _perm_sign(I_dirs + J_dirs)
        result[J_idx] = sign * shifted[J_idx]
    return result


# ---------------------------------------------------------------------------
# Lattice  — factory for interlaced forms
# ---------------------------------------------------------------------------

class Lattice:
    """
    A D-dimensional hypercubic lattice of size N, in the interlaced layout.

    All forms are (2N,)*D arrays.  lat.random(p) returns a p-form with
    random values at p-form sites and zero everywhere else.
    """

    def __init__(self, D, N):
        self.D = D
        self.N = N
        _structures(D)          # warm the cache

        # Precompute the p-form filters: form[p] = 1 at p-form sites, 0 elsewhere.
        TXYZ = np.mgrid[(slice(0, 2 * N),) * D]
        odds = np.mod(TXYZ, 2).sum(axis=0)
        self._form = np.zeros((D + 1,) + (2 * N,) * D)
        for p in range(D + 1):
            self._form[p] = np.where(odds == p, 1, 0)

    def form(self, p):
        """Return the p-form filter: 1 at p-form sites, 0 elsewhere."""
        return self._form[p]

    def zeros(self, p, dtype=float):
        """Zero p-form array, shape (2N,)*D."""
        return np.zeros((2 * self.N,) * self.D, dtype=dtype)

    def random(self, p):
        """Random p-form: uniform [0,1) at p-form sites, 0 elsewhere."""
        return self._form[p] * np.random.random((2 * self.N,) * self.D)

    def __repr__(self):
        return f"Lattice(D={self.D}, N={self.N})"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from itertools import product as iproduct

    parser = argparse.ArgumentParser(description="Test interlaced differential forms.")
    parser.add_argument('--D', type=int, default=4)
    parser.add_argument('--N', type=int, default=6)
    args = parser.parse_args()

    D = args.D
    N = args.N
    lat = Lattice(D, N)
    print(f"Testing on {lat}\n")

    print("NILPOTENCY OF d")
    for p in range(D + 1):
        a  = lat.random(p)
        da = d(a)   if p < D     else None
        dda= d(da)  if p < D - 1 else None
        if p < D - 1:
            assert not np.isclose(da, 0).all()
            assert np.isclose(dda, 0).all()
            print(f"  ✅ d² = 0 on {p}-forms")
        elif p == D - 1:
            assert not np.isclose(da, 0).all()
            print(f"  ✅ d of {p}-form is non-zero")
        else:
            print(f"  ✅ d not applied to top {p}-form")

    print("\nNILPOTENCY OF δ")
    for p in range(D + 1):
        a   = lat.random(p)
        da  = delta(a)   if p > 0 else None
        dda = delta(da)  if p > 1 else None
        if p > 1:
            assert not np.isclose(da, 0).all()
            assert np.isclose(dda, 0).all()
            print(f"  ✅ δ² = 0 on {p}-forms")
        elif p == 1:
            assert not np.isclose(da, 0).all()
            print(f"  ✅ δ of {p}-form is non-zero")
        else:
            print(f"  ✅ δ not applied to 0-form")

    print("\nADJOINTNESS  Σ (da)·b = −Σ a·(δb)")
    for p in range(D):
        a = lat.random(p)
        b = lat.random(p + 1)
        assert np.isclose(+(d(a) * b).sum(), -(a * delta(b)).sum())
        print(f"  ✅ Σ (da)·b = −Σ a·(δb)  for p={p}")

    print("\nBILINEARITY")
    for n, m in iproduct(range(D + 1), repeat=2):
        if n + m > D: continue
        a=lat.random(n); b=lat.random(n); c=lat.random(m); e=lat.random(m)
        assert np.isclose(wedge(n,m,a+b,c), wedge(n,m,a,c)+wedge(n,m,b,c)).all()
        assert np.isclose(wedge(n,m,a,c+e), wedge(n,m,a,c)+wedge(n,m,a,e)).all()
        print(f"  ✅ ∧ bilinear for {n}∧{m}")

    print("\nLEIBNIZ RULE")
    for n, m in iproduct(range(D + 1), repeat=2):
        if n + m + 1 > D: continue
        a = lat.random(n); b = lat.random(m)
        LHS = d(wedge(n, m, a, b))
        RHS = wedge(n+1, m, d(a), b) + (-1)**n * wedge(n, m+1, a, d(b))
        assert not np.isclose(LHS, 0).all()
        assert np.isclose(LHS, RHS).all()
        print(f"  ✅ d(a∧b) = da∧b + (−1)^{n} a∧db  for {n}∧{m}")

    print("\nASSOCIATIVITY")
    for n, m, p in iproduct(range(D), repeat=3):
        if n + m + p > D: continue
        a=lat.random(n); b=lat.random(m); c=lat.random(p)
        LHS = wedge(n+m, p, wedge(n,m,a,b), c)
        RHS = wedge(n, m+p, a, wedge(m,p,b,c))
        assert not np.isclose(LHS, 0).all()
        assert np.isclose(LHS, RHS).all()
        print(f"  ✅ (a∧b)∧c = a∧(b∧c)  for {n},{m},{p}")

    print("\nHODGE STAR  Σ a·b = Σ (a∧★b)")
    for p in range(D + 1):
        a = lat.random(p); b = lat.random(p)
        LHS = (a * b).sum()
        RHS = (wedge(p, D-p, a, star(p, b)) * lat.form(D)).sum()
        assert np.isclose(LHS, RHS), f"p={p}: {LHS} vs {RHS}"
        print(f"  ✅ Σ a·b = Σ (a∧★b)  for p={p}")

    print("\nAll tests passed.")
