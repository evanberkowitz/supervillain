#!/usr/bin/env python
r"""
Topology bookkeeping for the vortex sheets of the NoIntersections model.

The field strength F = dn is Poincaré-dual to a closed 2-cycle on the dual
lattice: the *vortex sheet*.  This script builds explicit valid
(q = dn ∧ dn = 0) configurations, computes the topology of their sheets
(connected components; Euler characteristic χ = V - E + F; genus g = (2 - χ)/2
per closed orientable component), and demonstrates that **legal single-link
moves --- the ConstrainedLinkUpdate's own proposals --- change that topology**:

  grow / shrink a sphere           (an elementary isotopy step)
  merge two spheres into one       (component number is not conserved)
  sphere <-> torus                 (genus is not conserved: this single-link
                                    move is an *internal stabilization*, the
                                    attachment/removal of a trivial 1-handle)

Every configuration here is built from links in a SINGLE direction, for which
dn ∧ dn = 0 identically, so every configuration is manifestly valid, and each
single-link move between two valid configurations is a legal
ConstrainedLinkUpdate proposal (legality *is* endpoint validity).

Verified at N = 6:

  add one link between two small spheres
      before: 2 components, χ = 2 each      after: 1 component, χ = 2
      => component number is not conserved

  remove the center link of a 3×3 patch
      before: sphere  F=30, E=60, V=32, χ=2, g=0
      after:  torus   F=32, E=64, V=32, χ=0, g=1
      => genus is not conserved

Why we care: 2-knot invariants are invariants of *isotopy of embedded surfaces
of fixed topology*.  Because the dynamics changes component number and genus
with legal elementary moves, and because internally-stabilized homologous
surfaces in a compact 4-manifold are isotopic (Hosokawa–Kawauchi 1979 for S⁴;
Baykur–Sunukjian 2016 in general), 2-knot type cannot label
dynamically-disconnected sectors --- *unless* the lattice constraint blocks the
stabilization/isotopy path on some background, which is the remaining finite,
checkable question.  The sphere <-> torus move above is precisely the
attachment/removal of a trivial 1-handle, so the elementary moves of the
existing library *are* stabilization moves and elementary isotopy steps, at
least on backgrounds where they are clean.

Terminology: "sphere" and "torus" name the *intrinsic* topology of a component.
Once a component is verified to be a closed 2-manifold (every dual edge carries
exactly two faces, no junction lines) and orientable --- which the integer signs
of F guarantee --- the classification of closed connected orientable surfaces
makes χ decisive: χ = 2 ⇔ S², χ = 0 ⇔ T².  Intrinsic topology says nothing about
the embedding: a *knotted* 2-sphere also has χ = 2.  In these demos the sheets
also arise as boundaries, since duality exchanges d on n with ∂ on the 3-chain of
dual 3-cubes of the occupied links: the 3×3 patch gives ∂(slab) = S², and the
punctured patch gives ∂(solid torus) = T².

The χ computation uses the dual-complex cell counts:

  faces    F : plaquettes where dn ≠ 0            (dual plaquettes of the sheet)
  edges    E : cubes containing sheet plaquettes  (dual edges; exactly 2 sheet
               faces per cube for a manifold sheet, 4+ marks a junction line)
  vertices V : hypercubes containing sheet plaquettes (dual vertices)

and requires |F| ≤ 1 (no multiplicity), which holds for all examples here.
"""

from itertools import combinations

import numpy as np

from supervillain.lattice import Lattice, d
from supervillain.generator.no_intersection.charge import charge

# The 2-form components in the library's lexicographic (μ<ν) order.
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
K = {p: k for k, p in enumerate(PAIRS)}


def faces_of(n):
    """The sheet's faces: plaquettes (k, x0, x1, x2, x3) where F = dn ≠ 0.

    Requires a multiplicity-free sheet (|F| ≤ 1 everywhere)."""
    F = np.asarray(d(n))
    if np.abs(F).max() > 1:
        raise ValueError('|F| > 1 somewhere: χ-counting here assumes a multiplicity-free sheet')
    return {tuple(int(i) for i in idx) for idx in np.argwhere(F != 0)}


def cube_faces(cube, x, N):
    """The 6 plaquettes on the boundary of the cube with directions ``cube`` (a<b<c) at site x."""
    a, b, c = cube
    for pair, r in (((a, b), c), ((a, c), b), ((b, c), a)):
        yield (K[pair],) + tuple(x)
        yield (K[pair],) + tuple((x[i] + (i == r)) % N for i in range(4))


def hypercube_faces(x, N):
    """The 24 plaquettes on the boundary of the hypercube at site x."""
    for (mu, nu), k in K.items():
        rho, sigma = (i for i in range(4) if i not in (mu, nu))
        for a in (0, 1):
            for b in (0, 1):
                yield (k,) + tuple((x[i] + a * (i == rho) + b * (i == sigma)) % N for i in range(4))


def topology(n):
    """Per-component (F, E, V, χ, genus) of the dual sheet, plus junction/shared-vertex counts."""
    N = n.lattice.N
    fs = faces_of(n)

    # Union-find over faces; faces sharing a cube (a dual edge) are connected.
    parent = {f: f for f in fs}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    edges = []          # one entry per cube that carries sheet plaquettes
    junctions = 0       # cubes with 4+ sheet faces: non-manifold junction lines
    for cube in combinations(range(4), 3):
        for x in np.ndindex(*(N,) * 4):
            hit = [f for f in cube_faces(cube, x, N) if f in fs]
            if not hit:
                continue
            if len(hit) == 1:
                raise AssertionError('dangling sheet face: is dF ≠ 0?')
            if len(hit) > 2:
                junctions += 1
            edges.append(hit)
            for f in hit[1:]:
                parent[find(f)] = find(hit[0])

    vertices = []       # one entry per hypercube that carries sheet plaquettes
    for x in np.ndindex(*(N,) * 4):
        hit = [f for f in hypercube_faces(x, N) if f in fs]
        if hit:
            vertices.append(hit)

    components = {}
    for f in fs:
        components.setdefault(find(f), []).append(f)

    shared_vertices = 0
    report = []
    for root, comp in components.items():
        E = sum(1 for hit in edges if find(hit[0]) == root)
        V = 0
        for hit in vertices:
            roots = {find(f) for f in hit}
            if len(roots) > 1:
                shared_vertices += 1
            if root in roots:
                V += 1
        chi = V - E + len(comp)
        report.append(dict(F=len(comp), E=E, V=V, chi=chi, genus=(2 - chi) // 2))
    return report, junctions, shared_vertices


def build(L, links):
    """An integer 1-form with n = +1 on each given link (direction, x0, x1, x2, x3)."""
    n = L.zeros(1, dtype=int)
    for link in links:
        n[link] += 1
    return n


def show(name, n):
    valid = bool(np.all(charge(n) == 0))
    comps, junctions, shared = topology(n)
    comps = sorted(comps, key=lambda c: -c['F'])
    print(f'{name}')
    print(f'  valid (q = dn∧dn = 0 everywhere): {valid}')
    print(f'  components: {len(comps)};  junction cubes: {junctions};  shared vertices: {shared}')
    for i, c in enumerate(comps):
        print(f"    component {i}: F={c['F']:3d}  E={c['E']:3d}  V={c['V']:3d}  "
              f"χ={c['chi']:2d}  genus={c['genus']}")
    print()
    return comps


if __name__ == '__main__':
    L = Lattice(4, 6)
    print(f'Lattice D=4, N={L.N}.  All configurations use direction-0 links only,')
    print('so dn ∧ dn = 0 identically: every configuration below is valid, and each')
    print('single-link move between two of them is a legal ConstrainedLinkUpdate proposal.')
    print()

    one = (0, 1, 1, 1, 1)
    far = (0, 1, 3, 1, 1)
    mid = (0, 1, 2, 1, 1)

    show('A: one link → boundary of one dual cube', build(L, [one]))
    show('B: two links, two apart in x1', build(L, [one, far]))
    show('C = B + one link between them (single legal move from B)', build(L, [one, mid, far]))

    patch = [(0, 1, i, j, 1) for i in (1, 2, 3) for j in (1, 2, 3)]
    center = (0, 1, 2, 2, 1)
    annulus = [link for link in patch if link != center]

    show('D: 3×3 patch of links → boundary of a 3×3×1 dual slab', build(L, patch))
    show('E = D − center link (single legal move from D)', build(L, annulus))

    print('Punchlines:')
    print('  B → C : one legal single-link move MERGES two sphere components into one.')
    print('  D → E : one legal single-link move turns a SPHERE (χ=2, g=0) into a TORUS')
    print('          (χ=0, g=1) --- the attachment of a trivial 1-handle, i.e. an')
    print('          internal stabilization.  E → D removes it.')
    print('  Component number and genus of the vortex sheet are NOT conserved by the')
    print('  update scheme, so no 2-knot invariant (defined for fixed topology) can be')
    print('  a conserved quantity of the dynamics.')
