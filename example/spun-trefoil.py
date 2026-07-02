#!/usr/bin/env python
r'''
Construct an explicit Villain configuration n (an integer 1-form) on a small
D=4 lattice whose vorticity dn is the worldsheet of a SPUN TREFOIL --- a
knotted, embedded 2-sphere on the dual lattice --- and verify its properties
numerically.

Construction
------------

Directions are (0, 1, 2, 3) = (x, y, ζ, w).  The Artin spin construction takes
a knotted arc α in the half-space {(x, y, z) : z ≥ 0} with endpoints on the
plane z = 0 and sweeps it around that plane in a fourth dimension,

    (x, y, z) → (x, y, z cos θ, z sin θ),

so each arc point at height z traces a circle of radius z in the (ζ, w) plane.
The result is an embedded 2-sphere whose complement has the knot group of α:
for a trefoil arc, π₁ = ⟨a, b | aba = bab⟩, non-abelian, so the sphere is
knotted [Artin 1925; see also Hosokawa & Kawauchi, Osaka J. Math. 16 (1979)].

On the lattice we replace the circle of radius z by the square (Chebyshev
circle) of radius z, which is PL-isotopic to it.  The arc is the standard
5×5 grid diagram of the trefoil (all vertical strands cross over all
horizontal strands), cut open on a crossing-free segment, with heights

    horizontal (x-running) strands at z = Z_UNDER,
    vertical  (y-running) strands at z = Z_OVER,

so every crossing is automatically correct, and every change of z happens at
a corner column that no other strand visits.

The trick that makes n trivial to write down: instead of building the
worldsheet and solving dn = F, take n to be the SOLID the spinning sweeps out.
The solid is the arc × (square disk of radius z), a 3-chain on the dual
lattice; dual 3-cells are in bijection with primal links, so this 3-chain IS
an integer 1-form n.  The chain boundary of the solid is the spun surface,
and the boundary of a dual 3-chain is exactly dn.  Vertical (z-direction) runs
of the arc sweep zero 3-volume and contribute nothing to n; the annuli and end
caps they generate appear in dn automatically through the telescoping of the
boundary.  No linear solve, no orientation bookkeeping beyond a per-direction
permutation sign.

Verification (all numerical, using the package's own d and wedge):
  1. dn takes values in {-1, 0, +1};
  2. every 3-cell of the lattice has 0 or 2 occupied faces  (=> the dual
     worldsheet is an embedded surface, no branching, no self-touching);
  3. the dual surface is connected with Euler characteristic V - E + F = 2
     (=> it is a single 2-sphere);
  4. wedge(dn, dn) = 0 on every hypercube  (=> the configuration satisfies
     the no-self-intersection constraint dn ∧ dn = 0).

Usage:
    python example/spun-trefoil.py [--output spun-trefoil-n.npz]
'''

import argparse
import numpy as np

import supervillain
from supervillain.lattice import Lattice, Form, d, wedge


####
#### The trefoil arc: 5x5 grid diagram, cut open, ends dropped to z=0.
####

# Grid diagram of the trefoil (0-indexed, grid number 5):
#   X markers at (i, i+3 mod 5), O markers at (i, i).
# Column i's vertical segment joins its X and O; row j's horizontal segment
# joins its X and O; vertical strands always cross OVER horizontal ones.
# Crossings: (col, row) = (1,3), (2,1), (3,2); tracing the strand the pattern
# is over, under, over, under, over, under --- a reduced alternating
# 3-crossing diagram, i.e. the trefoil.
#
# The closed loop is cut on row 4 between columns 2 and 3 (that segment
# carries no crossing, and no vertical strand visits columns 2 or 3 at row 4),
# giving a knotted arc whose closure through z <= 0 is the trefoil.
#
# The arc, as straight runs in grid units, each at constant height z:
#   each entry is (start, end, z) with start/end = (col, row).

Z_UNDER = 2   # height of x-running (horizontal) strands
Z_OVER  = 4   # height of y-running (vertical)  strands; Z_OVER - Z_UNDER >= 2
              # keeps the nested square tubes at crossings 2 apart, which the
              # lattice wedge needs in order not to see a spurious corner-touch.

RUNS = [
    ((2, 4), (1, 4), Z_UNDER),
    ((1, 4), (1, 1), Z_OVER),
    ((1, 1), (3, 1), Z_UNDER),
    ((3, 1), (3, 3), Z_OVER),
    ((3, 3), (0, 3), Z_UNDER),
    ((0, 3), (0, 0), Z_OVER),
    ((0, 0), (2, 0), Z_UNDER),
    ((2, 0), (2, 2), Z_OVER),
    ((2, 2), (4, 2), Z_UNDER),
    ((4, 2), (4, 4), Z_OVER),
    ((4, 4), (3, 4), Z_UNDER),
]

SPACING = 3   # lattice units per grid unit; keeps parallel tubes' plaquettes
              # more than one lattice unit apart so the wedge cannot pair them.
XY_OFFSET = 2 # margin between the diagram and the torus seam
ZW_OFFSET = 9 # (ζ, w) = (0, 0), the axis plane of the spinning, sits here
N = 18        # lattice size; must fit x,y in [0, SPACING*4 + 2*XY_OFFSET] and
              # ζ, w in ZW_OFFSET ± (Z_OVER + margin).


def arc_edges():
    '''
    Expand the runs into unit lattice steps.

    Yields (a, tau, direction, z): a = base lattice point (x, y) (the lesser
    end of the step in direction tau), tau in {0, 1}, direction = ±1 as the
    arc traverses the step, z = the height of the run.
    '''
    for (c0, r0), (c1, r1), z in RUNS:
        x0, y0 = XY_OFFSET + SPACING * c0, XY_OFFSET + SPACING * r0
        x1, y1 = XY_OFFSET + SPACING * c1, XY_OFFSET + SPACING * r1
        if y0 == y1:    # x-run
            tau, sign, lo, hi, fixed = 0, (1 if x1 > x0 else -1), min(x0, x1), max(x0, x1), y0
        elif x0 == x1:  # y-run
            tau, sign, lo, hi, fixed = 1, (1 if y1 > y0 else -1), min(y0, y1), max(y0, y1), x0
        else:
            raise ValueError('runs must be axis-aligned')
        for s in range(lo, hi):
            a = (s, fixed) if tau == 0 else (fixed, s)
            yield a, tau, sign, z


####
#### n = the swept solid, as an integer 1-form.
####

def build_n(L):
    r'''
    Each unit arc step at height z contributes (edge) × (square disk of
    radius z in the (ζ,w) plane) worth of dual 3-cells, all with coefficient
    ±1 (the arc direction).  A dual 3-cell spanning directions {μ,ν,ρ} at dual
    base Y corresponds to the primal link in the complementary direction σ at
    x = Y + ê_μ + ê_ν + ê_ρ, weighted by the permutation sign of (σ, μ, ν, ρ):

        x-step (spans {0,2,3}): σ = 1, sign(1,0,2,3) = -1;
        y-step (spans {1,2,3}): σ = 0, sign(0,1,2,3) = +1.
    '''
    n = L.form(1, dtype=int)
    for (ax, ay), tau, direction, z in arc_edges():
        cells = slice(ZW_OFFSET - z + 1, ZW_OFFSET + z + 1)   # ζ0, w0 ∈ [-z, z-1], shifted
        if tau == 0:
            n[1, ax + 1, ay, cells, cells] += -direction
        else:
            n[0, ax, ay + 1, cells, cells] += +direction
    return n


####
#### Verification.
####

def cube_face_degrees(L, occupied):
    '''
    For every 3-cell (a,b,c)@x, count how many of its 6 plaquette faces are
    occupied: (p,q)@x and (p,q)@(x+ê_r) for each pair {p,q} ⊂ {a,b,c}.
    Returns an array of shape (4, N, N, N, N).
    '''
    degrees = np.zeros((len(L.components[3]),) + L.dims, dtype=int)
    for i, cube in enumerate(L.components[3]):
        for r in cube:
            p, q = (k for k in cube if k != r)
            face = occupied[L.comp_index[2][(p, q)]]
            degrees[i] += face + np.roll(face, -1, axis=r)
    return degrees


def dual_surface_topology(L, F):
    '''
    Treat the occupied plaquettes of F as 2-cells of the dual lattice and
    compute (V, E, faces, connected components) of that 2-complex.

    A primal plaquette with components (ρ,σ) at x is the dual plaquette
    spanning the complementary directions (μ,ν) based at Y = x - ê_μ - ê_ν;
    its corners are Y, Y+ê_μ, Y+ê_ν, Y+ê_μ+ê_ν and its edges (Y,μ),
    (Y+ê_ν,μ), (Y,ν), (Y+ê_μ,ν)  (all mod N).
    '''
    def shifted(Y, *dirs):
        Y = list(Y)
        for k in dirs:
            Y[k] = (Y[k] + 1) % L.N
        return tuple(Y)

    faces = list(zip(*np.nonzero(F)))
    vertices, edge_faces = set(), {}
    face_edges = []
    for f, (ci, *x) in enumerate(faces):
        rho, sigma = L.components[2][ci]
        mu, nu = (k for k in range(L.D) if k not in (rho, sigma))
        Y = tuple((xi - (k == mu) - (k == nu)) % L.N for k, xi in enumerate(x))
        vertices.update((Y, shifted(Y, mu), shifted(Y, nu), shifted(Y, mu, nu)))
        edges = ((Y, mu), (shifted(Y, nu), mu), (Y, nu), (shifted(Y, mu), nu))
        face_edges.append(edges)
        for e in edges:
            edge_faces.setdefault(e, []).append(f)

    # Union-find over faces sharing an edge.
    parent = list(range(len(faces)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for fs in edge_faces.values():
        for other in fs[1:]:
            parent[find(other)] = find(fs[0])
    components = len({find(i) for i in range(len(faces))})

    edge_degrees = {len(fs) for fs in edge_faces.values()}
    return len(vertices), len(edge_faces), len(faces), components, edge_degrees


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--output', default='spun-trefoil-n.npz')
    args = parser.parse_args()

    L = Lattice(4, N)
    n = build_n(L)
    F = d(n)

    print(f'{L}')
    print(f'links with n != 0:        {int(np.count_nonzero(n))}  (values in {sorted(np.unique(n[n != 0]).tolist()) if np.any(n) else []})')

    # 1. Unit vorticity everywhere on the sheet.
    values = sorted(int(v) for v in np.unique(F))
    assert set(values) <= {-1, 0, 1}, f'dn takes values {values}'
    print(f'plaquettes with dn != 0:  {int(np.count_nonzero(F))}  (values in {values})')

    # 2. Embedded closed surface: every 3-cell has 0 or 2 occupied faces.
    degrees = cube_face_degrees(L, (F != 0).astype(int))
    degset = set(np.unique(degrees).tolist())
    assert degset <= {0, 2}, f'cube face-degrees {degset}: branched or self-touching'
    print(f'cube face-degrees:        {sorted(degset)}  (embedded: no branching)')

    # 3. One 2-sphere: connected, chi = 2, every edge on exactly 2 faces.
    V, E, faces, components, edge_degrees = dual_surface_topology(L, F)
    chi = V - E + faces
    assert edge_degrees == {2}, f'dual edge degrees {edge_degrees}'
    assert components == 1, f'{components} components'
    assert chi == 2, f'Euler characteristic {chi}'
    print(f'dual worldsheet:          V={V} E={E} F={faces}  chi={chi}  components={components}')

    # 4. The no-intersection constraint.
    Q = wedge(F, F)
    assert np.all(Q == 0), f'dn ∧ dn != 0 on {int(np.count_nonzero(Q))} hypercubes'
    print(f'wedge(dn, dn):            identically 0 on all {L.sites} hypercubes')

    # A movie: occupied worldsheet plaquettes per w-slice.  The sheet is
    # symmetric under w -> -w about the axis plane at w = ZW_OFFSET.
    counts = [(w, int(np.count_nonzero(F[:, :, :, :, w]))) for w in range(L.N)]
    print('\nw-slice movie (w: plaquettes):')
    print('  ' + '  '.join(f'{w}:{c}' for w, c in counts if c))

    np.savez_compressed(
        args.output,
        n=np.asarray(n, dtype=np.int8),
        dn=np.asarray(F, dtype=np.int8),
        D=L.D, N=L.N,
        directions=np.array(['x', 'y', 'zeta', 'w']),
        description=(
            'Spun trefoil vortex worldsheet for the D=4 Villain model. '
            'n is an integer 1-form, shape (4, N, N, N, N), component index = direction, '
            'matching supervillain.lattice.Lattice(4, N) conventions on branch four-dimensional. '
            'dn = d(n) is the worldsheet: an embedded 2-sphere on the dual lattice, '
            'knotted (pi_1 of the complement is the trefoil group, by the Artin spin '
            'construction), with wedge(dn, dn) = 0 everywhere. '
            f'Arc heights {Z_UNDER}/{Z_OVER}, grid spacing {SPACING}, spin axis plane at '
            f'(zeta, w) = ({ZW_OFFSET}, {ZW_OFFSET}).'
        ),
    )
    print(f'\nSaved n and dn to {args.output}')


if __name__ == '__main__':
    main()
