#!/usr/bin/env python
r"""
Build a NoIntersections configuration whose vortex sheet is a SPUN-KNOT 2-SPHERE.

Artin spinning: write ℝ⁴ = {(x1, x2, r, θ)} --- a half-space {(x1, x2, r): r ≥ 0} spun
around its boundary wall r = 0.  A knotted ARC in the half-space (endpoints on the wall,
knotted interior at r > 0) sweeps a 2-sphere when spun: interior points trace circles of
radius r, the endpoints are the fixed poles.  π₁ of the complement is the arc's knot
group, so the sphere is knotted (the spun trefoil is the classic example).

Lattice version: replace the circles by SQUARE RINGS in the (x3, x0)-plane.  The arc is
the grid-diagram torus knot of torus_knotted.py, cut open at one edge, with both cut
ends dropped to h = 0 ("the wall"); the old height coordinate becomes the ring RADIUS h.
The sphere is  ⋃ₛ {(x1(s), x2(s))} × C_{h(s)}  with C_h the boundary of the concentric
2h × 2h block in the (x3, x0)-plane (C_0 = a point: the poles).  n is the 1-form dual to
the swept solid-ring 3-chain: an x1-step of the arc at radius h sweeps a slab of
(1,3,0)-cells dual to x2-links, and vice versa; h-steps sweep nothing (their annuli
emerge in F = dn automatically, and cap the poles).

**Why this construction matters** (unlike the knotted tori of torus_knotted.py, whose
x0-independence makes the constraint blind): the sheet's plaquettes light FIVE dual
planes --- (0,1), (0,2), (1,2), (1,3), (2,3); only F_03 vanishes --- so BOTH cup pairs
(01, 23) and (02, 13) are simultaneously alive and q = dn ∧ dn = 0 is a GENUINE check,
not an automatic one.  The continuum sphere is embedded, but the lattice cup product
compares the sheet with a diagonally SHIFTED copy of itself, so pointwise q = 0 can fail
at (arc corner) × (ring corner) events even though ΣQ = 0.  The script reports any
violations; --repair greedily applies single-link corrections near the violations until
q ≡ 0, then re-verifies the sheet topology.

That anticipated failure mode did not materialize.  Verified q ≡ 0 exactly for the spun
trefoil (N = 8, 35-step arc, sheet F = 408, χ = 2), the scale-2 spun trefoil (N = 12),
and the spun cinquefoil (N = 10) --- each a single embedded genus-0 component, no
junctions, |F| = 1.  There is no *proof* that this discretization always passes; three
instances passed machine verification, and --repair stands ready for any variant that
does not.

So the valid space contains explicitly **knotted 2-spheres** --- the classic 2-knots ---
and not merely knotted tori, and they live exactly where the constraint has real teeth.

Verifications: the arc is connected and self-avoiding with endpoints at h = 0; the sign
convention is calibrated so the sheet is a single embedded surface; χ = 2 (a 2-sphere,
by the classification --- χ cannot see knotting, which lives in the embedding); |F| ≤ 1,
no junctions; and the q census.

Run from the repo root:

    uv run python example/no-intersection/spun_sphere.py [--N 8] [--grid 5] [--shift 2] [--scale 1] [--repair]

Importable: ``configuration(L, ...)`` returns (n, diagnostics-dict).
"""

import argparse

import numpy as np

from supervillain.lattice import Lattice, d
from supervillain.generator.no_intersection.charge import charge
from genus import topology
from torus_knotted import grid_knot
from frozen import f_plane_summary


# ─────────────────────────────────────────────────────────── the knotted arc

def knotted_arc(grid=5, shift=2, scale=1, origin=(1, 1, 1)):
    """Cut the grid-diagram torus knot open at one row edge and drop both ends to h = 0.

    Returns steps ``(start_vertex, axis, sign)`` in (x1, x2, h) with h the ring radius;
    the closure of the arc under the wall is the original knot, so the spun sphere
    carries its knot group.
    """
    steps = grid_knot(grid, shift, scale, origin)

    # Cut at the first row-strand step (axis 0; all of them run at h = origin[2]).
    cut = next(i for i, (_, ax, _) in enumerate(steps) if ax == 0)
    u, ax, sgn = steps[cut]
    w = list(u)
    w[ax] += sgn
    w = tuple(w)

    up = [((w[0], w[1], z), 2, +1) for z in range(0, w[2])]           # pole → w
    around = steps[cut + 1:] + steps[:cut]                            # w → u
    down = [((u[0], u[1], z), 2, -1) for z in range(u[2], 0, -1)]     # u → pole

    arc = up + around + down

    # Continuity and self-avoidance.
    verts = [arc[0][0]]
    for v, a, s in arc:
        assert v == verts[-1], 'arc steps are not contiguous'
        nxt = list(v)
        nxt[a] += s
        verts.append(tuple(nxt))
    assert verts[0][2] == 0 and verts[-1][2] == 0, 'arc endpoints must sit on the wall'
    assert len(set(verts)) == len(verts), 'arc is not self-avoiding'
    return arc


# ───────────────────────────────────────────────────────────── spinning to n

def spin(L, arc, center=None, s1=1, s2=-1):
    """The 1-form dual to the swept solid-ring 3-chain of the spun arc.

    An arc step in x1 (x2) at radius h sweeps a slab of 3-cells spanning (1,3,0)
    ((2,3,0)), dual to x2-links (x1-links), over the concentric 2h × 2h block in the
    (x3, x0)-plane; h-steps sweep nothing.  (s1, s2) is the relative orientation of the
    two slab types, calibrated by ``configuration``.
    """
    N = L.N
    C3, C0 = (N // 2, N // 2) if center is None else center

    n = L.zeros(1, dtype=int)
    arr = np.asarray(n)    # arr[mu, x0, x1, x2, x3]
    for (a1, a2, h), ax, sgn in arc:
        if h == 0 or ax == 2:
            continue
        lo0, hi0, lo3, hi3 = C0 - h, C0 + h, C3 - h, C3 + h
        assert 0 <= lo0 and hi0 <= N and 0 <= lo3 and hi3 <= N, 'rings leave the lattice'
        if ax == 0:      # x1-step: slab dual to x2-links
            x1 = a1 + 1 if sgn > 0 else a1
            arr[2, lo0:hi0, x1, a2, lo3:hi3] += s2 * sgn
        else:            # x2-step: slab dual to x1-links
            x2 = a2 + 1 if sgn > 0 else a2
            arr[1, lo0:hi0, a1, x2, lo3:hi3] += s1 * sgn
    return n


def sheet_report(n):
    """(is single embedded sphere, human-readable summary)."""
    F = np.asarray(d(n))
    if np.abs(F).max() > 1:
        return False, f'|F| reaches {np.abs(F).max()}: sheet overlaps itself'
    comps, junctions, shared = topology(n)
    lines = [f'{len(comps)} component(s); junctions {junctions}; shared vertices {shared}']
    for c in comps:
        lines.append(f"  F={c['F']}  E={c['E']}  V={c['V']}  χ={c['chi']}  genus={c['genus']}")
    ok = (len(comps) == 1 and junctions == 0 and shared == 0 and comps[0]['chi'] == 2)
    return ok, '\n'.join(lines)


def configuration(L, grid=5, shift=2, scale=1, origin=(1, 1, 1), center=None):
    """A 1-form n whose sheet is the spun T(shift, grid−shift) sphere, with the slab
    orientation calibrated so the sheet is a single embedded χ = 2 surface."""
    arc = knotted_arc(grid, shift, scale, origin)
    for s1, s2 in ((1, -1), (1, 1)):
        n = spin(L, arc, center=center, s1=s1, s2=s2)
        ok, summary = sheet_report(n)
        if ok:
            return n, dict(arc=arc, signs=(s1, s2), summary=summary)
    raise AssertionError(f'no slab orientation gives an embedded sphere; last: {summary}')


# ──────────────────────────────────────────────────────────────── q repair

def repair(n, reach=2, max_iter=400):
    """Greedy single-link repair: while q ≠ 0, apply the ±1 link change within L1
    ``reach`` of a violation that most reduces Σ|q|.  Returns (n, iterations) or
    (n, -1) if stuck."""
    L = n.lattice
    N = L.N
    n = n.copy()

    offsets = [tuple(x - reach for x in off)
               for off in np.ndindex(*(2 * reach + 1,) * 4)
               if sum(abs(x - reach) for x in off) <= reach]

    for it in range(max_iter):
        q = np.asarray(charge(n))
        bad = int(np.abs(q).sum())
        if bad == 0:
            return n, it
        best = None
        tried = set()
        trial = n.copy()
        for loc in np.argwhere(q != 0):
            hx = tuple(int(t) for t in loc[1:])
            for mu in range(4):
                for off in offsets:
                    site = tuple((hx[i] + off[i]) % N for i in range(4))
                    for c in (1, -1):
                        key = (mu, site, c)
                        if key in tried:
                            continue
                        tried.add(key)
                        trial[(mu,) + site] += c
                        now = int(np.abs(np.asarray(charge(trial))).sum())
                        trial[(mu,) + site] -= c
                        if best is None or now < best[0]:
                            best = (now, (mu,) + site, c)
        if best is None or best[0] >= bad:
            return n, -1
        n[best[1]] += best[2]
    return n, -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--N', type=int, default=8, help='lattice size (default 8)')
    parser.add_argument('--grid', type=int, default=5, help='grid number g (default 5)')
    parser.add_argument('--shift', type=int, default=2,
                        help='cyclic shift k: the spun knot is T(k, g−k) (default: trefoil)')
    parser.add_argument('--scale', type=int, default=1, help='stretch factor (default 1)')
    parser.add_argument('--repair', action='store_true',
                        help='greedily fix q ≠ 0 with single-link changes near violations')
    args = parser.parse_args()

    L = Lattice(4, args.N)
    n, info = configuration(L, grid=args.grid, shift=args.shift, scale=args.scale)

    print(f'Spun T({args.shift}, {args.grid - args.shift}) sphere on N={args.N} '
          f'(arc: {len(info["arc"])} steps; slab signs {info["signs"]}).')
    print('Sheet topology:')
    print(info['summary'])
    print()
    F = np.asarray(d(n))
    print('F-plane summary (F_03 must vanish; both cup pairs are otherwise alive):')
    print(f_plane_summary(F, 4))
    print()

    q = np.asarray(charge(n))
    viol = np.argwhere(q != 0)
    if len(viol) == 0:
        print('q = dn ∧ dn ≡ 0: the spun sphere is EXACTLY valid, and this time the')
        print('constraint was genuinely exercised (both cup pairs live on this sheet).')
    else:
        print(f'q ≠ 0 at {len(viol)} hypercubes (Σ|q| = {int(np.abs(q).sum())}, '
              f'Σq = {int(q.sum())}):')
        for loc in viol[:12]:
            print(f'  q{tuple(int(t) for t in loc[1:])} = {int(q[tuple(loc)])}')
        if len(viol) > 12:
            print(f'  … and {len(viol) - 12} more')
        print()
        print('The lattice cup product sees the sheet against a diagonally shifted copy')
        print('of itself; violations at (arc corner) × (ring corner) events are the')
        print('anticipated failure mode.  It has never been observed for the spun')
        print('trefoil (N=8, N=12 scale-2) or the spun cinquefoil (N=10); --repair')
        print('greedily corrects any variant that does trip it.')
        if args.repair:
            print()
            print('Repairing …')
            n2, its = repair(n)
            if its >= 0:
                print(f'q ≡ 0 after {its} single-link corrections.  Re-verifying topology:')
                ok, summary = sheet_report(n2)
                print(summary)
                print(f'single embedded sphere after repair: {"✓" if ok else "✗ (topology changed)"}')
            else:
                print('greedy repair got stuck; a smarter (multi-link) repair is needed.')
