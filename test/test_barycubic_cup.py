#!/usr/bin/env python
r'''
Cup and Steenrod products, the cohomology ring, and symmetries of the barycubic lattice
(SPEC §7 items 9–12).

Every fine simplex carries the total rank order of its vertices, so the Alexander–Whitney cup
$\cup = \cup_0$ and Steenrod's higher products $\cup_i$ are defined by the interval-cut formula of
SPEC §5.5.  What is tested, and why:

* **∪_0** (item 9): bilinear, translation covariant, literally Alexander–Whitney on rank-ordered
  vertices (checked with explicit vertex lookups, including the top-form/0-form rule), Leibniz and
  strict associativity exactly on integers, and out-of-range degrees raise.
* **∪_i** (item 10): the Steenrod coboundary identity for every legal (p, q, i), over the integers
  and mod 2; Steenrod squares of mod-2 cocycles are cocycles; the pointwise law $a \cup_p b = ab$
  for every p **including p = D** — the sign of $\cup_D$ is invisible to every legal Steenrod
  identity inside a D-dimensional complex (SPEC §5.5 "Scope of forced", review finding 1), so this
  is its only guard; and the cached sign table equals the closed-form rule entry by entry.
* **Cohomology ring** (item 11): the winding 1-cochains built from *lifted* link displacements are
  closed with period N, and $\int w_{\pi(0)} \cup \cdots \cup w_{\pi(D-1)} = {\rm sgn}(\pi) N^D$; the
  D = 1, N = 2 seam is a regression test for the common lift (SPEC §2.2).
* **Symmetries** (item 12): the point group and translations (``transform``) commute with $d$ and
  $\cup_i$ and preserve weights, with ε picking up $\det g$; the half-diagonal translation
  (``half_shift``) commutes with $d$, mirrors weights $M(R) = M(D-R)$, and — because it reverses rank
  order — obeys the order-reversal law for even $i$ and for odd diagonal $p = q = i$, but not for
  other odd $i$ (asserted).  The fundamental cycle satisfies $\partial[M] = 0$.

Integer random cochains with fixed seeds are used throughout, so every identity is checked exactly.
The integer windings used below are $u_\mu = 2 w_\mu = \Delta\xi_\mu$ (doubled coordinates), so that
the ring identity reads $\int u_{\pi(0)} \cup \cdots = {\rm sgn}(\pi)(2N)^D$ in exact integers.
'''

import itertools
import math

import numpy as np
import pytest

import barycubic_reference as ref
from barycubic_reference import DN_ALL, DN_CUP, lattice, int_form, is_zero, roll_sites, wrap
from supervillain.lattice import barycubic as bc


def _cases(dns, degree_filter):
    return [(D, N, p, q, i) for D, N in dns for p, q, i in ref.legal_cup_degrees(D) if degree_filter(D, p, q, i)]


def _d(f):
    r'''$d$, or ``None`` for the scalar 0 returned at top degree.'''
    out = bc.d(f)
    return None if np.ndim(out) == 0 else out


def _cup(a, b, i):
    r'''$a \cup_i b$, or ``None`` when a factor is ``None``, $i < 0$ (so $\cup_{-1} \equiv 0$), or the degree exceeds D.'''
    if a is None or b is None or i < 0:
        return None
    if i > min(a.degree, b.degree) or a.degree + b.degree - i > a.lattice.D:
        return None
    return bc.cup(a, b, i)


def _total(terms, like_shape):
    out = np.zeros(like_shape, dtype=np.int64)
    for c, t in terms:
        if t is not None:
            out = out + c * np.asarray(t)
    return out


def _index(L, p):
    return {k: j for j, k in enumerate(ref._key(np.asarray(L.canonical_labels(p))))}


def _face_values(L, f, vertices):
    r'''Values of the form ``f`` on the simplices with the given rank-ordered vertex lists ``(M, p+1, D)``.'''
    idx = _index(L, f.degree)
    return np.asarray(f).reshape(-1)[[idx[k] for k in ref._key(np.asarray(vertices) % (2 * L.N))]]


def _winding(L, mu):
    r'''Integer winding 1-cochain $u_\mu(e) = \Delta\xi_\mu(e)$ from the *lifted* rank-ordered link vertices.'''
    V = np.asarray(L.local_vertices(1), dtype=np.int64)
    per_label = V[:, 1, mu] - V[:, 0, mu]
    data = np.broadcast_to(per_label.reshape((-1,) + (1,) * L.D), (len(per_label),) + (L.N,) * L.D)
    return bc.BarycubicForm(np.array(data), degree=1, lattice=L)


# =============================================================================================
# Item 9: the cup product ∪_0
# =============================================================================================

@pytest.mark.parametrize("D,N", DN_CUP)
def test_cup_bilinear_for_every_i(D, N):
    r'''$(a + 2a') \cup_i b = a \cup_i b + 2\,a' \cup_i b$ and likewise in the second slot, exactly, for every legal (p, q, i).'''
    # SPEC §7 item 9.
    L = lattice(D, N)
    for p, q, i in ref.legal_cup_degrees(D):
        a, a2, b, b2 = int_form(L, p, 1, q, i), int_form(L, p, 2, q, i), int_form(L, q, 3, p, i), int_form(L, q, 4, p, i)
        prod = bc.cup(a, b, i)
        assert isinstance(prod, bc.BarycubicForm) and prod.degree == p + q - i
        assert np.issubdtype(prod.dtype, np.integer)
        assert np.array_equal(np.asarray(bc.cup(a + 2 * a2, b, i)), np.asarray(prod) + 2 * np.asarray(bc.cup(a2, b, i)))
        assert np.array_equal(np.asarray(bc.cup(a, b + 2 * b2, i)), np.asarray(prod) + 2 * np.asarray(bc.cup(a, b2, i)))


@pytest.mark.parametrize("D,N", DN_CUP)
def test_cup_translation_covariant(D, N):
    r'''Translating both factors by a lattice vector translates $a \cup_i b$ by the same vector.'''
    # SPEC §7 item 9.
    L = lattice(D, N)
    shift = tuple((k + 1) % N for k in range(D))
    for p, q, i in ref.legal_cup_degrees(D):
        a, b = int_form(L, p, 5, q, i), int_form(L, q, 6, p, i)
        moved = bc.cup(roll_sites(a, shift), roll_sites(b, shift), i)
        assert np.array_equal(np.asarray(moved), np.asarray(roll_sites(bc.cup(a, b, i), shift)))


@pytest.mark.parametrize("D,N", DN_CUP)
def test_cup_is_alexander_whitney(D, N):
    r'''$(a \cup b)(v_0 … v_{p+q}) = a(v_0 … v_p)\, b(v_p … v_{p+q})$ on every simplex, found by explicit vertex lookups.'''
    # SPEC §5.4.
    L = lattice(D, N)
    for p in range(D + 1):
        for q in range(D + 1 - p):
            a, b = int_form(L, p, 7, q), int_form(L, q, 8, p)
            simplices = np.asarray(L.canonical_labels(p + q), dtype=np.int64)
            expected = _face_values(L, a, simplices[:, :p + 1]) * _face_values(L, b, simplices[:, p:])
            assert np.array_equal(np.asarray(bc.cup(a, b)).reshape(-1), expected)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_cup_top_form_with_zero_form(D, N):
    r'''For a top form a and a 0-form f: $(a \cup f)(\sigma) = a(\sigma) f(v_D)$ and $(f \cup a)(\sigma) = f(v_0)\, a(\sigma)$.'''
    # SPEC §7 item 9 (the i = 0 analogue of test_top_cup_is_pointwise_product).
    L = lattice(D, N)
    a, f = int_form(L, D, 9), int_form(L, 0, 10)
    tops = np.asarray(L.canonical_labels(D), dtype=np.int64)
    flat = np.asarray(a).reshape(-1)
    assert np.array_equal(np.asarray(bc.cup(a, f)).reshape(-1), flat * _face_values(L, f, tops[:, D:]))
    assert np.array_equal(np.asarray(bc.cup(f, a)).reshape(-1), _face_values(L, f, tops[:, :1]) * flat)


@pytest.mark.parametrize("D,N", [(1, 2), (2, 3), (3, 2), (4, 2)])
def test_cup_out_of_range_raises(D, N):
    r'''``cup`` raises ``ValueError`` unless $0 \le i \le \min(p, q)$ and $p + q - i \le D$.'''
    # SPEC §5.4, §5.5.
    L = lattice(D, N)
    for p in range(D + 1):
        for q in range(D + 1):
            a, b = int_form(L, p, 11), int_form(L, q, 12)
            for i in (-1, min(p, q) + 1):
                with pytest.raises(ValueError):
                    bc.cup(a, b, i)
            for i in range(min(p, q) + 1):
                if p + q - i > D:
                    with pytest.raises(ValueError):
                        bc.cup(a, b, i)


@pytest.mark.parametrize("D,N", DN_CUP)
def test_cup_leibniz(D, N):
    r'''$d(a \cup b) = da \cup b + (-1)^p\, a \cup db$ exactly on integers.'''
    # SPEC §7 item 9 (port of test_lattice.py::test_leibniz_rule).
    L = lattice(D, N)
    for p in range(D):
        for q in range(D - p):
            a, b = int_form(L, p, 13, q), int_form(L, q, 14, p)
            lhs = np.asarray(bc.d(bc.cup(a, b)))
            rhs = _total([(1, _cup(_d(a), b, 0)), ((-1) ** p, _cup(a, _d(b), 0))], lhs.shape)
            assert np.array_equal(lhs, rhs)


@pytest.mark.parametrize("D,N", DN_CUP)
def test_cup_associative(D, N):
    r'''$(a \cup b) \cup c = a \cup (b \cup c)$ exactly on integers.'''
    # SPEC §7 item 9 (port of test_lattice.py::test_wedge_associative).
    L = lattice(D, N)
    for p, q, r in itertools.product(range(D + 1), repeat=3):
        if p + q + r <= D:
            a, b, c = int_form(L, p, 15, q, r), int_form(L, q, 16, p, r), int_form(L, r, 17, p, q)
            assert np.array_equal(np.asarray(bc.cup(bc.cup(a, b), c)), np.asarray(bc.cup(a, bc.cup(b, c))))


# =============================================================================================
# Item 10: Steenrod products ∪_i
# =============================================================================================

# The identity has degree n + 1 = p + q − i + 1; it is vacuous once that exceeds D.
STEENROD_CASES = _cases(DN_CUP, lambda D, p, q, i: p + q - i <= D - 1)


def _steenrod_residual(a, b, i, sign=True):
    r'''
    $d(a \cup_i b) - [da \cup_i b + (-1)^p a \cup_i db + (-1)^{p+q-i} a \cup_{i-1} b + (-1)^{pq+p+q} b \cup_{i-1} a]$,
    with $\cup_{-1} \equiv 0$ and terms of degree > D equal to 0 (all signs +1 when ``sign`` is False).
    '''
    p, q = a.degree, b.degree
    s = (lambda e: (-1) ** e) if sign else (lambda e: 1)
    lhs = np.asarray(bc.d(bc.cup(a, b, i)))
    rhs = _total([(1, _cup(_d(a), b, i)),
                  (s(p), _cup(a, _d(b), i)),
                  (s(p + q - i), _cup(a, b, i - 1)),
                  (s(p * q + p + q), _cup(b, a, i - 1))], lhs.shape)
    return lhs - rhs


@pytest.mark.parametrize("D,N,p,q,i", STEENROD_CASES)
def test_steenrod_identity_integers(D, N, p, q, i):
    r'''The Steenrod coboundary identity holds exactly over the integers.'''
    # SPEC §5.5 (the fork's convention, test_cup_products.py::_steenrod_residual), §7 item 10.
    L = lattice(D, N)
    a, b = int_form(L, p, 20, q, i), int_form(L, q, 21, p, i)
    assert not _steenrod_residual(a, b, i).any()


@pytest.mark.parametrize("D,N,p,q,i", STEENROD_CASES)
def test_steenrod_identity_mod2(D, N, p, q, i):
    r'''Mod 2 the identity reads $d(a \cup_i b) = da \cup_i b + a \cup_i db + a \cup_{i-1} b + b \cup_{i-1} a$.'''
    # SPEC §7 item 10.
    L = lattice(D, N)
    a, b = int_form(L, p, 22, q, i, low=0, high=2), int_form(L, q, 23, p, i, low=0, high=2)
    assert not (_steenrod_residual(a, b, i, sign=False) % 2).any()


@pytest.mark.parametrize("D,N", DN_CUP)
def test_steenrod_squares_of_cocycles_are_cocycles(D, N):
    r'''For a mod-2 cocycle a of degree p, $Sq^{p-i} a = a \cup_i a$ is again a mod-2 cocycle.'''
    # SPEC §7 item 10.  Cocycles: a product of windings (nontrivial class) plus a coboundary.
    L = lattice(D, N)
    for p in range(1, D):
        a = _winding(L, 0)
        for mu in range(1, p):
            a = bc.cup(a, _winding(L, mu))
        a = wrap(np.asarray(a) + np.asarray(bc.d(int_form(L, p - 1, 24))), a)
        assert not (np.asarray(bc.d(a)) % 2).any()
        for i in range(p + 1):
            if 2 * p - i <= D - 1:
                assert not (np.asarray(bc.d(bc.cup(a, a, i))) % 2).any(), (p, i)


@pytest.mark.parametrize("D,N", DN_CUP)
def test_top_steenrod_product_is_pointwise(D, N):
    r'''$a \cup_p b = a\,b$ pointwise for p-forms, for every p **including p = D** (the only guard on the sign of $\cup_D$).'''
    # SPEC §5.5 "Scope of forced", review finding 1 — mandatory for every p.
    L = lattice(D, N)
    for p in range(D + 1):
        a, b = int_form(L, p, 25), int_form(L, p, 26)
        assert np.array_equal(np.asarray(bc.cup(a, b, p)), np.asarray(a) * np.asarray(b))


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_cup_table_matches_closed_form(D):
    r'''Every entry of ``cup_table(p, q, i)`` — term set, face offsets in $\{0,1\}^D$, and sign — equals the closed-form interval-cut rule.'''
    # SPEC §5.5 (frozen sign rule), §7 item 10.  Expected terms derived from local_vertices only.
    L = lattice(D, 2)
    for p, q, i in ref.legal_cup_degrees(D):
        table = np.asarray(L.cup_table(p, q, i))
        assert table.dtype == np.int64 and table.shape[1] == 2 * D + 4
        got = {}
        for row in table:
            key = (int(row[0]), int(row[1]), tuple(int(t) for t in row[2:2 + D]),
                   int(row[2 + D]), tuple(int(t) for t in row[3 + D:3 + 2 * D]))
            got[key] = got.get(key, 0) + int(row[-1])
        got = {k: v for k, v in got.items() if v}
        assert got == ref.expected_cup_table(L.local_vertices, p, q, i), (p, q, i)


# =============================================================================================
# Item 11: the cohomology ring
# =============================================================================================

@pytest.mark.parametrize("D,N", DN_ALL)
def test_winding_cochains_are_closed_with_period_N(D, N):
    r'''$u_\mu = \Delta\xi_\mu$ (lifted) is closed, and its sum around the $\mu$ cycle through $\xi = t\,e_\mu$ is $2N$, i.e. $w_\mu = u_\mu/2$ has period N.'''
    # SPEC §2.2 (common lift), §7 item 11.
    L = lattice(D, N)
    links = _index(L, 1)
    for mu in range(D):
        u = _winding(L, mu)
        if D > 1:
            assert is_zero(bc.d(u))
        flat = np.asarray(u).reshape(-1)
        period = 0
        for t in range(2 * N):
            here, there = np.zeros(D, dtype=np.int64), np.zeros(D, dtype=np.int64)
            here[mu], there[mu] = t % (2 * N), (t + 1) % (2 * N)
            # A link between an even and an odd point along the axis; rank order puts the even one first.
            forward = t % 2 == 0
            pair = np.stack([here, there] if forward else [there, here])
            value = flat[links[ref._key(pair[None])[0]]]
            period += value if forward else -value
        assert period == 2 * N


def test_seam_regression_d1_n2():
    r'''D = 1, N = 2: the link stored as (0, 3) has lifted displacement −1, not the raw +3; the winding has period 2 and $\int w = N$.'''
    # SPEC §2.2 example, §7 item 11.
    L = lattice(1, 2)
    labels = np.asarray(L.canonical_labels(1), dtype=np.int64).reshape(-1, 2)
    row = int(np.flatnonzero((labels[:, 0] == 0) & (labels[:, 1] == 3))[0])
    assert labels[row, 1] - labels[row, 0] == 3
    u = _winding(L, 0)
    assert np.asarray(u).reshape(-1)[row] == -1
    assert bc.integrate(u) == 2 * 2       # ∫ u = 2N, so ∫ w = N = 2


@pytest.mark.parametrize("D,N", DN_ALL)
def test_cohomology_ring(D, N):
    r'''$\int u_{\pi(0)} \cup \cdots \cup u_{\pi(D-1)} = {\rm sgn}(\pi)\,(2N)^D$ for every permutation π, i.e. $\int w_{\pi(0)} \cup \cdots = {\rm sgn}(\pi) N^D$.'''
    # SPEC §7 item 11.  Left-nested products, with prefixes cached.
    L = lattice(D, N)
    u = [_winding(L, mu) for mu in range(D)]
    prefix = {(): None}
    for perm in itertools.permutations(range(D)):
        for k in range(1, D + 1):
            key = perm[:k]
            if key not in prefix:
                prefix[key] = u[key[0]] if k == 1 else bc.cup(prefix[key[:-1]], u[key[-1]])
        sign = round(np.linalg.det(np.eye(D)[list(perm)]))
        assert bc.integrate(prefix[perm]) == sign * (2 * N) ** D, perm


# =============================================================================================
# Item 12: symmetries
# =============================================================================================

def _group_elements(D, N):
    r'''A few point-group-and-translation elements (perm, flips, shift), including the identity and a pure reflection.'''
    rng = np.random.default_rng([D, N, 300])
    out = [(tuple(range(D)), (1,) * D, (0,) * D),
           (tuple(range(D)), (-1,) + (1,) * (D - 1), (0,) * D)]
    for _ in range(3):
        out.append((tuple(int(k) for k in rng.permutation(D)),
                    tuple(int(s) for s in rng.choice([-1, 1], D)),
                    tuple(int(s) for s in rng.integers(0, N, D))))
    return out


def _det(perm, flips):
    return round(np.linalg.det(np.eye(len(perm))[list(perm)])) * int(np.prod(flips))


def _weight_form(L, p):
    W = np.asarray(L.hodge_weights(p), dtype=float).reshape((-1,) + (1,) * L.D)
    return bc.BarycubicForm(np.broadcast_to(W, (len(W),) + (L.N,) * L.D).copy(), degree=p, lattice=L)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_transform_is_pullback(D, N):
    r'''$(g^* f)(\sigma) = f(g\sigma)$ with $g\xi = {\rm flips} \odot \xi_{\rm perm} + 2\,{\rm shift}$ on doubled coordinates; no sign.'''
    # SPEC §6.2.  Checked label by label through canonical labels.
    L = lattice(D, N)
    for p in range(D + 1):
        f = int_form(L, p, 30)
        labels = np.asarray(L.canonical_labels(p), dtype=np.int64)
        for perm, flips, shift in _group_elements(D, N):
            image = np.array(flips) * labels[:, :, list(perm)] + 2 * np.array(shift)
            expected = _face_values(L, f, image)
            assert np.array_equal(np.asarray(bc.transform(f, perm, flips, shift)).reshape(-1), expected)


@pytest.mark.parametrize("D,N", DN_CUP)
def test_transform_commutes_with_d_and_cup(D, N):
    r'''$g^* d = d g^*$ and $g^*(a \cup_i b) = g^*a \cup_i g^*b$ exactly, for every legal (p, q, i).'''
    # SPEC §6.2, §7 item 12.
    L = lattice(D, N)
    for perm, flips, shift in _group_elements(D, N)[1:3]:
        g = lambda f: bc.transform(f, perm, flips, shift)  # noqa: E731
        for p in range(D):
            f = int_form(L, p, 31)
            assert np.array_equal(np.asarray(g(bc.d(f))), np.asarray(bc.d(g(f))))
        for p, q, i in ref.legal_cup_degrees(D):
            a, b = int_form(L, p, 32, q, i), int_form(L, q, 33, p, i)
            assert np.array_equal(np.asarray(g(bc.cup(a, b, i))), np.asarray(bc.cup(g(a), g(b), i))), (p, q, i)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_transform_preserves_weights_and_scales_orientation(D, N):
    r'''Weights are invariant, $M(g\sigma) = M(\sigma)$; the orientation picks up $\det g$, so $\int g^* f = \det g \int f$.'''
    # SPEC §6.2, §7 item 12.
    L = lattice(D, N)
    for perm, flips, shift in _group_elements(D, N):
        det = _det(perm, flips)
        for p in range(D + 1):
            W = _weight_form(L, p)
            assert np.array_equal(np.asarray(bc.transform(W, perm, flips, shift)), np.asarray(W))
        eps = L.orientation
        assert np.array_equal(np.asarray(bc.transform(eps, perm, flips, shift)), det * np.asarray(eps))
        top = int_form(L, D, 34)
        assert bc.integrate(bc.transform(top, perm, flips, shift)) == det * bc.integrate(top)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_half_shift_is_signed_pullback(D, N):
    r'''$(\tau^* f)(\sigma) = (-1)^{\binom{p+1}{2}} f(\tau\sigma)$, $\tau\xi = \xi + (1,…,1)$, with $\tau\sigma$ re-sorted into rank order (reversed).'''
    # SPEC §6.2.
    L = lattice(D, N)
    for p in range(D + 1):
        f = int_form(L, p, 35)
        labels = np.asarray(L.canonical_labels(p), dtype=np.int64)
        expected = (-1) ** math.comb(p + 1, 2) * _face_values(L, f, labels[:, ::-1] + 1)
        assert np.array_equal(np.asarray(bc.half_shift(f)).reshape(-1), expected)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_half_shift_twice_is_unit_translation(D, N):
    r'''$\tau^2$ is the translation by one site along the diagonal: $(\tau^*)^2 f[c, x] = f[c, x + (1,…,1)]$.'''
    # SPEC §6.2 (consistency of the half-diagonal translation).
    L = lattice(D, N)
    for p in range(D + 1):
        f = int_form(L, p, 36)
        assert np.array_equal(np.asarray(bc.half_shift(bc.half_shift(f))), np.asarray(roll_sites(f, (-1,) * D)))


@pytest.mark.parametrize("D,N", DN_ALL)
def test_half_shift_commutes_with_d_and_mirrors_weights(D, N):
    r'''$\tau^* d = d \tau^*$ exactly, and $M(\tau\sigma) = M(\sigma)$, i.e. $M(R) = M(D - R)$.'''
    # SPEC §6.2, §7 item 12.
    L = lattice(D, N)
    for p in range(D):
        f = int_form(L, p, 37)
        assert np.array_equal(np.asarray(bc.half_shift(bc.d(f))), np.asarray(bc.d(bc.half_shift(f))))
    for p in range(D + 1):
        W = _weight_form(L, p)
        sign = (-1) ** math.comb(p + 1, 2)
        assert np.allclose(sign * np.asarray(bc.half_shift(W)), np.asarray(W), rtol=1e-14, atol=0)


@pytest.mark.parametrize("D,N", DN_CUP)
def test_half_shift_reversal_law(D, N):
    r'''
    $\tau^*(a \cup_i b) = (-1)^{pq + i + \binom{i+1}{2}}\, \tau^*b \cup_i \tau^*a$ holds for every even i and for odd
    $p = q = i$; for D ≥ 2 at least one off-diagonal odd-i case fails.
    '''
    # SPEC §5.5 "Order reversal", §6.2, review finding 5.
    L = lattice(D, N)
    failures = []
    for p, q, i in ref.legal_cup_degrees(D):
        a, b = int_form(L, p, 38, q, i), int_form(L, q, 39, p, i)
        c = (-1) ** (p * q + i + math.comb(i + 1, 2))
        holds = np.array_equal(np.asarray(bc.half_shift(bc.cup(a, b, i))),
                               c * np.asarray(bc.cup(bc.half_shift(b), bc.half_shift(a), i)))
        if i % 2 == 0 or p == q == i:
            assert holds, (p, q, i)
        elif not holds:
            failures.append((p, q, i))
    if D >= 2:
        assert failures, "the odd-i reversal law must fail off the diagonal"


@pytest.mark.parametrize("D,N", DN_ALL)
def test_fundamental_cycle_is_closed(D, N):
    r'''$\partial[M] = d_{D-1}^\top \varepsilon = 0$, so Stokes holds: $\int d f = 0$ for every (D−1)-form.'''
    # SPEC §3, §7 item 12.
    L = lattice(D, N)
    dmat = L.export_complex(D - 1)['d']
    eps = np.asarray(L.export_complex(D)['orientation']).reshape(-1)
    assert np.array_equal(eps, np.asarray(L.orientation).reshape(-1))
    assert not (dmat.T @ eps).any()
    assert bc.integrate(bc.d(int_form(L, D - 1, 40))) == 0

