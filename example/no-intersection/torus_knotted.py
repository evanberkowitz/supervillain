#!/usr/bin/env python
r"""
Build a valid NoIntersections configuration whose vortex sheet is a KNOTTED TORUS.

Recipe (ergodicity.md, open question 6): realize a torus knot γ = T(k, g−k) as a closed
cubic-lattice polygon from its cyclic grid diagram (grid size g, shift k; the default
g=5, k=2 is the trefoil), lay it in the (x1, x2, x3) sublattice, hang a Dirac sheet below
it (vertical curtains under every horizontal strand + a winding-number filling of the
projected diagram at the base level), and let m be the sheet's crossing-number 1-form,
so that f = dm is Poincaré-dual to γ.  Then extend to T⁴ with no x0-component and no
x0-dependence:

    n_i(x0, x⃗) = m_i(x⃗)  for i ∈ {1,2,3},      n_0 = 0 .

F = dn is dual to Σ = γ × S¹_{x0} --- a knotted torus --- and the configuration is
EXACTLY valid: F_{0ν} = 0, and every complementary plane pair in F ∧ F contains a (0ν)
factor, so q = dn ∧ dn ≡ 0 identically (the same mechanism that validates the
single-direction configurations of genus.py).  Knottedness is inherited from the diagram
(π₁ of the complement is the torus-knot group × ℤ); the intrinsic topology is a torus,
which genus.topology confirms (χ = 0) --- χ cannot see knotting, only the embedding does.

Grid diagrams make the polygon robust and scalable: column i carries a vertical strand
from row i to row (i+k) mod g, at a HIGHER x3-level than the horizontal row strands, so
every crossing is automatically vertical-over-horizontal and the polygon is embedded by
construction.  gcd(g, k) = 1 gives a single component; g=5, k=2 → trefoil T(2,3);
g=7, k=2 → cinquefoil T(2,5); --scale s stretches all three directions by s.

The Dirac-sheet sign conventions are SELF-CALIBRATED: the four relative signs (two
curtain orientations, base filling, winding) are searched until f = dm is supported
exactly on the plaquettes pierced by γ with |f| = 1 (a global flip also passes, so two
of the sixteen combinations do).

Verifications performed:
  1. γ is a closed, self-avoiding polygon (reported length);
  2. f = dm is supported exactly on the plaquettes dual to γ's steps, with |f| = 1;
  3. q ≡ 0 exactly on the assembled 4D configuration (charge());
  4. the dual sheet is a single manifold component with χ = 0 (genus 1), no junctions.

Run from example/no-intersection/ (imports genus.py; corridors.py under --worm):

    uv run python torus_knotted.py [--N 8] [--scale 1] [--grid 5] [--shift 2] [--worm]

Importable: ``configuration(L, grid, shift, scale, origin)`` returns the 1-form n, for
use as a Markov-chain start or a census/corridor target.
"""

import argparse
from itertools import product
from math import gcd

import numpy as np

from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.charge import charge
from genus import topology

# Spatial plane pairs in component order, matching a 3D exterior derivative below.
PLANES3 = ((0, 1), (0, 2), (1, 2))
# The plane (index into PLANES3) pierced by a dual step along each spatial axis.
PLANE_OF_AXIS = {0: 2, 1: 1, 2: 0}


# ─────────────────────────────────────────────────────────── the knotted polygon

def grid_knot(g=5, k=2, scale=1, origin=(1, 1, 1)):
    """The torus knot T(k, g−k) as a closed lattice polygon, as a list of unit steps.

    Cyclic grid diagram: column i has its X marker in row i and its O marker in row
    (i+k) mod g.  Horizontal (row) strands run at dual height o3; vertical (column)
    strands at o3 + scale, so verticals pass OVER horizontals everywhere.  Steps are
    ``(start_vertex, axis, sign)`` with vertices in dual-integer coordinates (the dual
    point is start + (1/2, 1/2, 1/2)).
    """
    if gcd(g, k) != 1:
        raise ValueError(f'gcd(grid, shift) = gcd({g}, {k}) ≠ 1 gives a link, not a knot')
    o1, o2, o3 = origin
    s = scale
    x1 = lambda i: o1 + s * i
    x2 = lambda j: o2 + s * j
    zrow, zcol = o3, o3 + s

    steps = []

    def walk(p, target):
        p = list(p)
        for ax in range(3):
            delta = target[ax] - p[ax]
            sgn = 1 if delta > 0 else -1
            for _ in range(abs(delta)):
                steps.append((tuple(p), ax, sgn))
                p[ax] += sgn
        return tuple(p)

    c = 0
    start = (x1(0), x2(0), zrow)
    p = start
    for _ in range(g):
        nc = (c + k) % g
        for target in (
            (x1(c), x2(c), zcol),        # ascend at the X marker
            (x1(c), x2(nc), zcol),       # run along column c to row nc
            (x1(c), x2(nc), zrow),       # descend at the O marker
            (x1(nc), x2(nc), zrow),      # run along row nc to column nc's X marker
        ):
            p = walk(p, target)
        c = nc
    assert p == start, 'polygon failed to close'

    vertices = [v for v, _, _ in steps]
    assert len(set(vertices)) == len(vertices), 'polygon is not self-avoiding'
    return steps


# ───────────────────────────────────────────────────────────── the Dirac sheet

def build_m(steps, N, s1, s2, s3):
    """The crossing-number 1-form m of the Dirac sheet hanging below the polygon.

    Curtains: below each horizontal strand step (heights 1..h, walls at half-integer
    transverse position) --- these are crossed by primal links perpendicular to the
    strand.  Base filling: the winding number w(x1, x2) of the projected diagram fills
    the plane between x3 = 0 and 1, crossed by the vertical links at x3 = 0.  The three
    relative signs (s1, s2, s3) are calibrated by verify_f; exactly two of the eight
    combinations pass, and they are global flips of each other.
    """
    m = np.zeros((3, N, N, N), dtype=int)
    w = np.zeros((N, N), dtype=int)
    for (v1, v2, h), ax, sgn in steps:
        if ax == 1:                                    # column strand step (Δx2)
            q = v2 + 1 if sgn > 0 else v2              # the crossed integer x2 line
            m[0, v1, q, 1:h + 1] += s1 * sgn           # wall ⟂ x1-links
            w[:v1 + 1, q] += sgn                       # ray casting for the winding
        elif ax == 0:                                  # row strand step (Δx1)
            q = v1 + 1 if sgn > 0 else v1
            m[1, q, v2, 1:h + 1] += s2 * sgn           # wall ⟂ x2-links
        # ax == 2 (connectors): curtains are degenerate; handled by seam cancellation.
    m[2, :, :, 0] = s3 * w
    return m


def d3(m):
    """3D exterior derivative: f_{ij} = ∂_i m_j − ∂_j m_i with forward differences."""
    f = np.zeros_like(m)
    for a, (i, j) in enumerate(PLANES3):
        f[a] = (np.roll(m[j], -1, axis=i) - m[j]) - (np.roll(m[i], -1, axis=j) - m[i])
    return f


def expected_support(steps):
    """The plaquettes the polygon pierces: a dual step from v along +ê_k crosses the
    complementary-plane plaquette based at v + ê_k (at v itself for a −ê_k step)."""
    exp = {}
    for v, ax, sgn in steps:
        base = list(v)
        if sgn > 0:
            base[ax] += 1
        exp[(PLANE_OF_AXIS[ax], tuple(base))] = sgn
    return exp


def verify_f(f, steps):
    """Is f = dm supported exactly on the pierced plaquettes, with |f| = 1?"""
    exp = expected_support(steps)
    got = {(int(idx[0]), tuple(int(t) for t in idx[1:])): int(f[tuple(idx)])
           for idx in np.argwhere(f != 0)}
    ok = set(got) == set(exp) and all(abs(v) == 1 for v in got.values())
    return ok, exp, got


def calibrate(steps, N):
    """Find the sign convention for which the Dirac sheet's boundary is exactly γ."""
    passing = []
    for signs in product((1, -1), repeat=3):
        m = build_m(steps, N, *signs)
        ok, _, _ = verify_f(d3(m), steps)
        if ok:
            passing.append(signs)
    if not passing:
        raise AssertionError('no sign convention makes ∂(sheet) = γ; construction bug')
    return passing


# ─────────────────────────────────────────────────────────────── 4D assembly

def configuration(L, grid=5, shift=2, scale=1, origin=(1, 1, 1)):
    """A valid 1-form n on the 4D lattice L whose vortex sheet is the knotted torus
    T(shift, grid−shift) × S¹."""
    N = L.N
    steps = grid_knot(grid, shift, scale, origin)

    extent = max(max(v) for v, _, _ in steps)
    if extent + 2 > N:
        raise ValueError(f'knot extent {extent} does not fit N = {N} with clearance; '
                         f'need N ≥ {extent + 2}')

    signs = calibrate(steps, N)
    m = build_m(steps, N, *signs[0])

    n = L.zeros(1, dtype=int)
    arr = np.asarray(n)
    for i in range(3):
        arr[i + 1][:] = m[i]        # broadcast over x0: no x0 component, no x0 dependence
    return n, steps, signs


NAMED_TORI = r"""
named tori --- any gcd(g, k) = 1 works; the knot is T(k, g−k), whose grid number is
g = k + (g−k), and the footprint needs N ≥ scale·(g−1) + 3:

  --grid 5 --shift 1    unknot      T(1,4)          (the control; any k=1 is unknotted)
  --grid 5 --shift 2    trefoil     T(2,3) = 3_1    N ≥ 7
  --grid 7 --shift 2    cinquefoil  T(2,5) = 5_1    N ≥ 9   (Solomon's seal)
  --grid 9 --shift 2    septafoil   T(2,7) = 7_1    N ≥ 11
  --grid 7 --shift 3                T(3,4) = 8_19   N ≥ 9
  --grid 8 --shift 3                T(3,5) = 10_124 N ≥ 10
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, epilog=NAMED_TORI,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--N', type=int, default=8, help='lattice size (default 8)')
    parser.add_argument('--grid', type=int, default=5,
                        help='grid number g (default 5; see the named tori below)')
    parser.add_argument('--shift', type=int, default=2,
                        help='cyclic shift k: the knot is T(k, g−k) (default 2: trefoil)')
    parser.add_argument('--scale', type=int, default=1, help='stretch factor (default 1)')
    parser.add_argument('--worm', action='store_true',
                        help='also run the IntersectionWorm template census on this background')
    args = parser.parse_args()

    L = Lattice(4, args.N)
    n, steps, signs = configuration(L, grid=args.grid, shift=args.shift, scale=args.scale)

    print(f'Torus knot T({args.shift}, {args.grid - args.shift}) from the cyclic grid '
          f'diagram (g={args.grid}, k={args.shift}), scale {args.scale}, on N={args.N}.')
    print(f'  polygon: {len(steps)} unit steps, closed and self-avoiding ✓')
    print(f'  Dirac-sheet sign calibration: {len(signs)} passing convention(s) '
          f'{signs} (a global flip always pairs) ✓')
    print(f'  f = dm supported exactly on the {len(steps)} pierced plaquettes, |f| = 1 ✓')

    q = np.asarray(charge(n))
    assert not q.any()
    print(f'  q = dn ∧ dn ≡ 0 exactly (F_0ν = 0 kills every complementary pair) ✓')

    comps, junctions, shared = topology(n)
    print(f'  sheet topology: {len(comps)} component(s); junctions {junctions}; '
          f'shared vertices {shared}')
    for c in comps:
        print(f"    F={c['F']}  E={c['E']}  V={c['V']}  χ={c['chi']}  genus={c['genus']}")
    ok = (len(comps) == 1 and junctions == 0 and shared == 0
          and comps[0]['chi'] == 0 and comps[0]['genus'] == 1)
    print(f'  single embedded torus: {"✓" if ok else "✗ UNEXPECTED"}')
    print()
    print(f'This is a valid configuration whose sheet is a knotted torus γ × S¹.')
    print(f'χ sees only the intrinsic torus; the knotting lives in the embedding.')

    if args.worm:
        from corridors import worm_census
        print()
        print('IntersectionWorm template census on this background …')
        legal, any_dipole, total = worm_census(n)
        print(f'  {legal} legal first steps of {total} placements; '
              f'{any_dipole} produce a ± dipole somewhere')
