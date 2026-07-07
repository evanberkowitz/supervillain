#!/usr/bin/env python
r"""
Exact Alexander-polynomial machinery: the certification tool of the movie workflow.

Two independent calculators, both exact over $\mathbb{Z}[t^{\pm 1}]$ (no floats in
any invariant; floats appear only in the generic projection, whose genericity is
asserted):

1. ``alexander(loop)`` --- the classical Alexander polynomial of a closed lattice
   loop in the 3-torus (e.g. a slice loop of a vortex sheet, ordered by
   ``extract_loop`` from a slice current).  Pipeline: generic rotation → planar
   diagram (crossings with over/under from height and signs from tangents) → the
   Wirtinger/Fox Alexander matrix → determinant of a first minor by fraction-free
   Bareiss elimination over $\mathbb{Z}[t]$ → normalization to $\Delta(1) = \pm 1$
   (asserted: a failure means the input was not a single embedded loop).

2. ``fox_alexander(word)`` --- the Alexander polynomial of a knot-group presentation
   $\langle a, b \mid b = w\,a\,w^{-1} \rangle$ with both generators meridians, via
   the free Fox derivative abelianized at $a, b \mapsto t$: $\Delta \doteq
   \partial r/\partial a$.  This is the presentation a 1-fusion ribbon 2-knot's
   group takes, with $w$ the band's through-passage word --- so it certifies the
   *2-knot* while ``alexander`` certifies its middle *slice*.

Polynomials are dicts {exponent: coefficient}; equality is up to units
$\pm t^{k}$ (``normalize``).

Self-tests (run the module):
  * unknot rectangle → Δ = 1;
  * the spatial trefoil of torus_knotted.py (a slice of the knotted torus) →
    Δ = t² − t + 1, through the full loop-extraction + diagram pipeline;
  * Fox: the trefoil word w = ab reproduces t² − t + 1; w = b gives 1 (a band
    sliding through only its own target circle fuses trivially).

Run from example/no-intersection/:

    uv run python alexander.py
"""

import numpy as np


# ─────────────────────────────────────────────── exact Laurent-polynomial ring

def normalize(p):
    """Strip the unit ±t^k: lowest exponent to 0, leading coefficient positive."""
    p = {e: c for e, c in p.items() if c}
    if not p:
        return {}
    lo = min(p)
    p = {e - lo: c for e, c in p.items()}
    if p[max(p)] < 0:
        p = {e: -c for e, c in p.items()}
    return p


def padd(p, q):
    out = dict(p)
    for e, c in q.items():
        out[e] = out.get(e, 0) + c
    return {e: c for e, c in out.items() if c}


def pmul(p, q):
    out = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            out[e1 + e2] = out.get(e1 + e2, 0) + c1 * c2
    return {e: c for e, c in out.items() if c}


def psub(p, q):
    return padd(p, {e: -c for e, c in q.items()})


def pdiv(p, q):
    """Exact division in Z[t, 1/t]; asserts the remainder vanishes."""
    if not p:
        return {}
    assert q, 'division by zero polynomial'
    out = {}
    r = dict(p)
    eq, cq = max(q), q[max(q)]
    while r:
        er, cr = max(r), r[max(r)]
        assert cr % cq == 0, 'inexact polynomial division'
        f = {er - eq: cr // cq}
        out = padd(out, f)
        r = psub(r, pmul(f, q))
    return out


def bareiss_det(M):
    """Fraction-free determinant of a matrix of Laurent polynomials (exact)."""
    n = len(M)
    if n == 0:
        return {0: 1}
    M = [row[:] for row in M]
    sign = 1
    prev = {0: 1}
    for k in range(n - 1):
        if not M[k][k]:
            swap = next((i for i in range(k + 1, n) if M[i][k]), None)
            if swap is None:
                return {}
            M[k], M[swap] = M[swap], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = pdiv(psub(pmul(M[i][j], M[k][k]), pmul(M[i][k], M[k][j])),
                               prev)
        prev = M[k][k]
    det = M[n - 1][n - 1]
    return {e: sign * c for e, c in det.items()}


# ─────────────────────────────────────── loops → diagrams → Alexander matrix

def extract_loop(j):
    """Order a slice current's single loop into a closed list of dual vertices.

    The dual edge of j_i(x) joins cube bases x − ê_i and x, directed by the sign;
    conservation makes out-degree 1 at every touched vertex for an embedded loop."""
    N = j.shape[1]
    succ = {}
    for i in range(3):
        for site in np.argwhere(j[i] != 0):
            v = tuple(int(x) for x in site)
            u = list(v)
            u[i] = (u[i] - 1) % N
            u = tuple(u)
            tail, head = (u, v) if j[(i,) + tuple(site)] > 0 else (v, u)
            assert tail not in succ, 'current is not a single embedded loop'
            succ[tail] = head
    assert succ, 'empty current'
    start = next(iter(succ))
    loop, v = [start], succ[start]
    while v != start:
        loop.append(v)
        v = succ[v]
    assert len(loop) == len(succ), 'current has more than one loop'
    return loop


def _unwrap(loop, N):
    """Lift the closed torus loop to ℝ³ (it must be null-homologous)."""
    pts = [np.array(loop[0], dtype=float)]
    for a, b in zip(loop, loop[1:] + loop[:1]):
        step = (np.array(b) - np.array(a) + N // 2) % N - N // 2
        pts.append(pts[-1] + step)
    assert np.allclose(pts[0], pts[-1]), 'loop wraps the torus: no diagram'
    return pts[:-1]


def diagram(loop, N):
    """Crossings of a generic planar projection: a list of
    (over_param, under_param, sign) with params measured along the loop."""
    rng = np.random.default_rng(20260704)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    pts = [R @ p for p in _unwrap(loop, N)]
    n = len(pts)
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    crossings = []
    for i in range(n):
        for j in range(i + 1, n):
            if j in (i, (i + 1) % n) or i == (j + 1) % n:
                continue
            (p, p2), (q, q2) = segs[i], segs[j]
            d1, d2 = p2 - p, q2 - q
            denom = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(denom) < 1e-9:
                # parallel in projection; genericity of R keeps them disjoint
                continue
            rhs = q[:2] - p[:2]
            u = (rhs[0] * d2[1] - rhs[1] * d2[0]) / denom
            v = (rhs[0] * d1[1] - rhs[1] * d1[0]) / denom
            if not (0 < u < 1 and 0 < v < 1):
                assert not (-1e-7 < u < 1 + 1e-7 and -1e-7 < v < 1 + 1e-7) or \
                    (1e-7 < u < 1 - 1e-7 and 1e-7 < v < 1 - 1e-7), \
                    'degenerate crossing at a vertex: change the projection seed'
                continue
            z1 = p[2] + u * d1[2]
            z2 = q[2] + v * d2[2]
            assert abs(z1 - z2) > 1e-9, 'projection not generic'
            # Crossing sign: det(tangent_over, tangent_under) with both tangents
            # along the traversal.  denom = det(d_i, d_j).
            if z1 > z2:
                crossings.append((i + u, j + v, 1 if denom > 0 else -1))
            else:
                crossings.append((j + v, i + u, 1 if -denom > 0 else -1))
    return crossings


def alexander(loop, N):
    """The Alexander polynomial of a closed lattice loop, normalized so that the
    lowest exponent is 0 and the leading coefficient positive; Δ(1) = ±1 asserted."""
    cross = diagram(loop, N)
    if not cross:
        return {0: 1}
    unders = sorted(c[1] for c in cross)
    m = len(unders)

    def arc(param):
        """Arc index of a point at loop parameter ``param``: arcs are the intervals
        between consecutive underpasses (cyclically)."""
        import bisect
        return bisect.bisect_left(unders, param) % m

    rows = []
    for over_p, under_p, sign_ in cross:
        over = arc(over_p)
        k = unders.index(under_p)
        # arc() labels the interval ENDING at u_k as arc k, so the strand arriving
        # at this underpass is arc k and the strand leaving it is arc k+1.
        under_in, under_out = k, (k + 1) % m
        # Wirtinger x_out = x_o^ε x_in x_o^{-ε}; Fox derivative abelianized:
        # ε=+1: (1−t)·o + t·in − out;  ε=−1 (times the unit t): (t−1)·o + in − t·out.
        row = [dict() for _ in range(m)]

        def add(col, poly):
            row[col] = padd(row[col], poly)

        if sign_ > 0:
            add(over, {0: 1, 1: -1})
            add(under_in, {1: 1})
            add(under_out, {0: -1})
        else:
            add(over, {1: 1, 0: -1})
            add(under_in, {0: 1})
            add(under_out, {1: -1})
        rows.append(row)
    minor = [row[:-1] for row in rows[:-1]]
    delta = normalize(bareiss_det(minor))
    check = sum(c for c in delta.values()) if delta else 0
    assert abs(check) == 1, f'Δ(1) = {check} ≠ ±1: not a knot diagram'
    return delta


# ────────────────────────────────────────────── Fox calculus for band words

def fox_alexander(word):
    """Δ(t) of the knot group ⟨a, b | b = w a w⁻¹⟩ via the Fox derivative
    ∂r/∂a of r = w a w⁻¹ b⁻¹, abelianized at a, b ↦ t.  ``word`` is a list of
    (generator, ±1) letters for w, e.g. [('a', 1), ('b', 1)]."""
    r = list(word) + [('a', 1)] + [(g, -e) for g, e in reversed(word)] + [('b', -1)]
    deriv = {}
    prefix = 0          # abelianized exponent sum of the prefix
    for g, e in r:
        if g == 'a':
            if e > 0:
                deriv = padd(deriv, {prefix: 1})
            else:
                deriv = padd(deriv, {prefix - 1: -1})
        prefix += e
    return normalize(deriv)


def poly_str(p):
    if not p:
        return '0'
    return ' + '.join(f'{c}·t^{e}' if e else f'{c}'
                      for e, c in sorted(p.items())).replace('+ -', '− ')


# ─────────────────────────────────────────────────────────────── self-tests

if __name__ == '__main__':
    from supervillain.lattice import Lattice, d
    from spun_hopf_link import slice_current

    # Unknot: a rectangular loop.
    N = 8
    j = np.zeros((3, N, N, N), dtype=int)
    for x in (1, 2, 3):
        j[0, x, 1, 3] += 1
        j[0, x, 4, 3] -= 1
    for y in (1, 2, 3):
        j[1, 4, y, 3] += 1
        j[1, 1, y, 3] -= 1
    for i in range(3):
        j[i] = np.roll(j[i], 1, axis=i)
    loop = extract_loop(j)
    delta = alexander(loop, N)
    print(f'unknot rectangle: Δ = {poly_str(delta)}   '
          f'{"✓" if delta == {0: 1} else "✗"}')

    # The spatial trefoil: a slice of torus_knotted's knotted torus.
    import torus_knotted
    L = Lattice(4, 8)
    n, _, _ = torus_knotted.configuration(L, grid=5, shift=2)
    F = np.asarray(d(n))
    loop = extract_loop(slice_current(F, 0))
    delta = alexander(loop, 8)
    trefoil = {0: 1, 1: -1, 2: 1}
    print(f'torus_knotted slice ({len(loop)} vertices): Δ = {poly_str(delta)}   '
          f'{"✓ (trefoil)" if delta == trefoil else "✗"}')

    # Fox calculus: the band word w = ab gives the trefoil group; w = b is trivial.
    for word, label, want in ((( ('a', 1), ('b', 1)), 'w = ab', trefoil),
                              ((('b', 1),), 'w = b', {0: 1})):
        delta = fox_alexander(list(word))
        print(f'fox {label}: Δ = {poly_str(delta)}   '
              f'{"✓" if delta == want else "✗"}')
