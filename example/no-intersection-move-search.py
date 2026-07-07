#!/usr/bin/env python
r'''
Exhaustively enumerate the elementary clean moves of the IntersectionWorm.

A clean move is a change $\Delta n$ supported on at most 3 links, with
coefficients $\pm 1$, whose charge change on an empty background is exactly a
unit dipole $\Delta q = \{+1 \text{ at } P,\ -1 \text{ at } M\}$.  Two dipole
displacements $P - M$ are enumerated:

* the **face** displacement $+\hat e_3$ (one bucket suffices: axis permutations
  and full parity are exact lattice symmetries and generate every other face
  from this one), and
* the **opposite-sign diagonal** ("elbow") displacement $+\hat e_2 - \hat e_3$,
  which likewise generates all 24 signed 2-plane diagonals.

The same census for the **same-sign diagonal** $+\hat e_2 + \hat e_3$ finds
*nothing* with $\le 3$ links: that neighbour genuinely requires four links, and
the library's same-sign moves come from the hand-found 4-link seed
(:data:`EXTRA_SEEDS`; origin commit 522d88b) whose completeness is therefore
NOT certified by this search.

Each search is complete for moves in which every link influences the charge of
one of the two defect hypercubes; the certified library class is that census
together with its transform-orbit closure (the transforms are a *filtered*
action, so an orbit can add clean moves carrying one link that touches neither
defect --- its cross terms cancel in context).  Found moves are
grouped into orbits of the worm's transform group (384 signed axis permutations
x global negation, filtered for cleanliness), with every transformed template
re-anchored on its +1 defect before comparison --- the same dipole/re-anchor
step the worm's library builder applies --- so that translations do not split
classes.  One representative per orbit is written to the auto-generated
moves.py with --write.

Usage:
    python example/no-intersection-move-search.py [--write]
'''

from itertools import combinations, permutations, product
import numpy as np

from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.charge import dF_entries, local_dq, wedge_pairs
from supervillain.generator.no_intersection.worm import IntersectionWorm

L = Lattice(4, 8)
A = (4, 4, 4, 4)                       # the +1 defect
PAIRS = wedge_pairs(L)
F0 = np.zeros((6,) + L.dims, dtype=int)

# The hand-found 4-link same-sign-diagonal move (origin commit 522d88b), converted to
# head-relative form (its +1 defect at the origin; dipole separation +e_0 + e_1).  The
# <=3-link census below finds nothing for the same-sign displacement, so four links is
# minimal there --- and this seed's completeness is NOT certified by the search.
EXTRA_SEEDS = (
    tuple(sorted((
        (0, (-1, -1, 1, 0), +1),
        (0, (-1,  0, 1, 0), +1),
        (1, ( 0,  0, 1, 1), +1),
        (3, ( 0,  0, 1, 0), -1),
    ))),
)


def hypercube_plaquettes(x):
    '''The (component_index, site) plaquettes entering q(x).'''
    out = set()
    for A_idx, A_dirs, B_idx, sign in PAIRS:
        out.add((A_idx, x))
        ahead = tuple((x[k] + (k in A_dirs)) % L.N for k in range(4))
        out.add((B_idx, ahead))
    return out


def links_touching(plaquettes):
    '''All links whose dF hits any of the given plaquettes.'''
    box = range(-3, 3)
    pool = []
    for mu in range(4):
        for rel in product(box, repeat=4):
            s = tuple((A[k] + rel[k]) % L.N for k in range(4))
            link = (mu,) + s
            if set(dF_entries(L, {link: 1})) & plaquettes:
                pool.append(link)
    return pool


def dq_of(changes):
    merged = {}
    for link, c in changes:
        merged[link] = merged.get(link, 0) + c
    merged = {l: c for l, c in merged.items() if c != 0}
    return local_dq(L, F0, merged, pairs=PAIRS), merged


def head_relative(merged):
    return tuple(sorted(
        (mu, tuple(s[k] - A[k] for k in range(4)), c)
        for (mu, *s), c in merged.items()
    ))


def anchored(template, sep):
    '''
    Re-anchor a head-relative template on its +1 defect, exactly as the worm's
    library builder does: place it on the empty scratch lattice, demand a clean
    unit dipole with separation ``sep``, and measure the links from the +1 site.
    Returns the rebased template, or None if the dipole is unclean or points
    elsewhere (such members belong to other displacement buckets).
    '''
    changes = [((mu,) + tuple((A[k] + rs[k]) % L.N for k in range(4)), c)
               for mu, rs, c in template]
    dq, merged = dq_of(changes)
    if len(dq) != 2:
        return None
    (a, va), (b, vb) = sorted(dq.items())
    if {va, vb} != {1, -1}:
        return None
    plus, minus = ((a, b) if va == 1 else (b, a))
    if tuple(int(p - m) for p, m in zip(plus, minus)) != sep:
        return None
    shift = tuple(p - a0 for p, a0 in zip(plus, A))
    return tuple(sorted(
        (mu, tuple(rs[k] - shift[k] for k in range(4)), c)
        for mu, rs, c in template
    ))


def orbit(template, sep):
    '''All re-anchored ``sep``-bucket members of the template's transform orbit.'''
    out = set()
    for perm in permutations(range(4)):
        for flips in product((1, -1), repeat=4):
            for negate in (False, True):
                rebased = anchored(IntersectionWorm._transformed(template, perm, flips, negate), sep)
                if rebased is not None:
                    out.add(rebased)
    return out


def census(sep, label):
    '''
    Exhaustively enumerate the clean <=3-link moves whose dipole is +1 at A and
    -1 at A - sep, then group them into re-anchored orbit classes.
    '''
    M = tuple((A[k] - sep[k]) % L.N for k in range(4))
    WANT = {A: 1, M: -1}

    target_plaquettes = hypercube_plaquettes(A) | hypercube_plaquettes(M)
    T = links_touching(target_plaquettes)
    print(f'[{label}] candidate links touching the defect pair: {len(T)}')

    found = []

    # 1 and 2 links, and 3 links all touching the targets.
    for k in (1, 2, 3):
        count = 0
        for links in combinations(T, k):
            for signs in product((1, -1), repeat=k):
                dq, merged = dq_of(list(zip(links, signs)))
                if dq == WANT:
                    found.append(merged)
                    count += 1
        print(f'[{label}] {k}-link moves with all links touching the defect pair: {count}')

    unique = {tuple(sorted(m.items())): m for m in found}
    print(f'[{label}] distinct on-target clean moves with separation {sep}: {len(unique)}')

    # Close under the transform orbits (with re-anchoring).  The transforms are only a
    # *filtered* action --- single-axis reflections are not exact symmetries of q ---
    # so the orbit of an all-on-target move can contain clean moves with a link that
    # touches neither defect hypercube (its cross terms cancel in context).  The
    # library builder expands seeds by exactly these orbits, so the certified class is
    # the on-target census PLUS its orbit closure; distinct orbits partition it.
    remaining = {head_relative(m) for m in unique.values()}
    closure = set()
    classes = []
    while remaining:
        rep = sorted(remaining)[0]
        cls = orbit(rep, sep)
        assert not (cls & closure), 'orbits must partition the closure'
        closure |= cls
        remaining -= cls
        classes.append((rep, len(cls)))
    grown = len(closure) - len(unique)
    print(f'[{label}] orbit closure: {len(closure)} moves ({grown} beyond the on-target census)')

    print(f'[{label}] orbit classes: {len(classes)}')
    for rep, size in classes:
        print(f'  class of size {size:3d}: seed = {rep}')
    return len(closure), classes


if __name__ == '__main__':
    import sys

    unit_total, unit_classes = census((0, 0, 0, 1), 'face +e3')
    print()
    elbow_total, elbow_classes = census((0, 0, 1, -1), 'elbow +e2-e3')
    print()
    samediag_total, samediag_classes = census((0, 0, 1, 1), 'same-sign +e2+e3')
    assert samediag_total == 0, 'a <=3-link same-sign-diagonal move exists after all!'

    if '--write' in sys.argv:
        import pathlib
        target = pathlib.Path(__file__).parent.parent / 'supervillain' / 'generator' / 'no_intersection' / 'moves.py'
        with open(target, 'w') as f:
            f.write('#!/usr/bin/env python\n')
            f.write('r"""\n')
            f.write('AUTO-GENERATED by example/no-intersection-move-search.py --write; do not edit by hand.\n\n')
            f.write('One representative per symmetry class of the elementary clean IntersectionWorm\n')
            f.write('moves: changes of at most 3 links with coefficients +-1 whose charge change on an\n')
            f.write('empty background is a clean unit dipole, for the face displacement +e_3 and the\n')
            f.write('opposite-sign ("elbow") diagonal displacement +e_2-e_3 (both censuses exhaustive),\n')
            f.write('plus the hand-found 4-link same-sign-diagonal seed (+e_0+e_1; completeness NOT\n')
            f.write('certified --- the search is capped at 3 links and finds nothing for that\n')
            f.write('displacement).  Each seed is a tuple of (direction, site relative to the +1\n')
            f.write('defect, coefficient) triples.  Classes count the transforms (384 signed axis\n')
            f.write('permutations x global negation) together with the re-anchoring translation onto\n')
            f.write('the +1 defect; the worm expands the seeds under the same transforms (filtered\n')
            f.write('for cleanliness, re-anchored on the +1 defect) to fill every displacement bucket.\n')
            f.write('"""\n\n')
            f.write(f'# {unit_total} distinct face (+e_3) moves in {len(unit_classes)} symmetry classes.\n')
            f.write('SEEDS = (\n')
            for rep, size in unit_classes:
                f.write(f'    {rep!r},\n')
            f.write(f'    # {elbow_total} distinct elbow (+e_2-e_3) moves in {len(elbow_classes)} symmetry classes.\n')
            for rep, size in elbow_classes:
                f.write(f'    {rep!r},\n')
            f.write('    # Hand-found 4-link same-sign-diagonal seed (origin 522d88b); not certified complete.\n')
            for rep in EXTRA_SEEDS:
                f.write(f'    {rep!r},\n')
            f.write(')\n')
        print(f'\nwrote {len(unit_classes)} + {len(elbow_classes)} + {len(EXTRA_SEEDS)} seeds to {target}')
