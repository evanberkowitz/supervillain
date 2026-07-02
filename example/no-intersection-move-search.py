#!/usr/bin/env python
r'''
Exhaustively enumerate the elementary clean moves of the IntersectionWorm.

A clean move is a change $\Delta n$ supported on at most 3 links, with
coefficients $\pm 1$, whose charge change on an empty background is exactly a
unit dipole: $\Delta q = \{+1 \text{ at } P,\ -1 \text{ at } P - \hat e_3\}$
(one bucket suffices: axis permutations and full parity are exact lattice
symmetries and generate every other separation from this one).

The search is complete for moves in which every link influences the charge of
one of the two defect hypercubes, plus three-link moves in which two such
links leave a residue that the third link (anywhere) cancels.  Found moves are
grouped into orbits of the worm's transform group (384 signed axis
permutations x global negation, filtered for cleanliness) and one
representative per orbit is printed in the head-relative _SEEDS format of
supervillain.generator.no_intersection.worm.

Usage:
    python example/no-intersection-move-search.py
'''

from itertools import combinations, product
import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.charge import dF_entries, local_dq, wedge_pairs
from supervillain.generator.no_intersection.worm import IntersectionWorm

L = Lattice(4, 8)
A = (4, 4, 4, 4)                       # the +1 defect
M = (4, 4, 4, 3)                       # the -1 defect: separation +e_3
WANT = {A: 1, M: -1}
PAIRS = wedge_pairs(L)
F0 = np.zeros((6,) + L.dims, dtype=int)


def hypercube_plaquettes(x):
    '''The (component_index, site) plaquettes entering q(x).'''
    out = set()
    for A_idx, A_dirs, B_idx, sign in PAIRS:
        out.add((A_idx, x))
        ahead = tuple((x[k] + (k in A_dirs)) % L.N for k in range(4))
        out.add((B_idx, ahead))
    return out


def influenced_hypercubes(link):
    '''Hypercubes whose q the link can change.'''
    out = set()
    for (idx, site), v in dF_entries(L, {link: 1}).items():
        for A_idx, A_dirs, B_idx, sign in PAIRS:
            if idx == A_idx:
                out.add(site)
            if idx == B_idx:
                out.add(tuple((site[k] - (k in A_dirs)) % L.N for k in range(4)))
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


target_plaquettes = hypercube_plaquettes(A) | hypercube_plaquettes(M)
T = links_touching(target_plaquettes)
print(f'candidate links touching the defect pair: {len(T)}')

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
    print(f'{k}-link moves with all links touching the defect pair: {count}')

# 3 links where the third only cancels the residue of the first two.
count = 0
for l1, l2 in combinations(T, 2):
    for s1, s2 in product((1, -1), repeat=2):
        dq2, merged2 = dq_of([(l1, s1), (l2, s2)])
        if dq2 == WANT or not dq2:
            continue
        discrepancy = {x for x in set(dq2) | set(WANT) if dq2.get(x, 0) != WANT.get(x, 0)}
        if not discrepancy:
            continue
        # candidate third links: must influence every discrepancy hypercube's charge
        disc_plaquettes = set().union(*(hypercube_plaquettes(x) for x in discrepancy))
        for l3 in links_touching(disc_plaquettes):
            if l3 in (l1, l2) or l3 in T:
                continue   # (l3 in T) triples were enumerated above
            for s3 in (1, -1):
                dq3, merged3 = dq_of([(l1, s1), (l2, s2), (l3, s3)])
                if dq3 == WANT:
                    found.append(merged3)
                    count += 1
print(f'3-link moves with an off-target cancelling link: {count}')

# Deduplicate (the same merged change can arise from different orderings).
unique = {tuple(sorted(m.items())): m for m in found}
print(f'distinct clean moves with separation +e_3: {len(unique)}')

# Group into orbits of the worm transform group and print representatives.
S = supervillain.action.NoIntersections(L, kappa=0.1)
worm = IntersectionWorm(S)

def head_relative(merged):
    return tuple(sorted(
        (mu, tuple(s[k] - A[k] for k in range(4)), c)
        for (mu, *s), c in merged.items()
    ))

def orbit(template):
    out = set()
    for perm in __import__('itertools').permutations(range(4)):
        for flips in product((1, -1), repeat=4):
            for negate in (False, True):
                out.add(worm._transformed(template, perm, flips, negate))
    return out

remaining = {head_relative(m) for m in unique.values()}
classes = []
while remaining:
    rep = sorted(remaining)[0]
    cls = orbit(rep) & remaining
    remaining -= cls
    # also remove orbit members that re-anchor onto other found moves
    classes.append((rep, len(cls)))

print(f'\norbit classes among the +e_3 moves: {len(classes)}')
for rep, size in classes:
    print(f'  class of size {size:3d}: seed = {rep}')

import sys
if '--write' in sys.argv:
    import pathlib
    target = pathlib.Path(__file__).parent.parent / 'supervillain' / 'generator' / 'no_intersection' / 'moves.py'
    with open(target, 'w') as f:
        f.write('#!/usr/bin/env python\n')
        f.write('r"""\n')
        f.write('AUTO-GENERATED by example/no-intersection-move-search.py --write; do not edit by hand.\n\n')
        f.write('One representative per orbit class of the elementary clean IntersectionWorm moves:\n')
        f.write('changes of at most 3 links with coefficients ±1 whose charge change on an empty\n')
        f.write('background is a clean unit dipole.  Each seed is a tuple of\n')
        f.write('(direction, site relative to the +1 defect, coefficient) triples with dipole\n')
        f.write('separation +e_3; the worm expands these under all 384 signed axis permutations\n')
        f.write('and global negation (filtered for cleanliness) to fill every direction bucket.\n')
        f.write('"""\n\n')
        f.write(f'# {len(unique)} distinct moves with separation +e_3 in {len(classes)} orbit classes.\n')
        f.write('SEEDS = (\n')
        for rep, size in classes:
            f.write(f'    {rep!r},\n')
        f.write(')\n')
    print(f'\nwrote {len(classes)} seeds to {target}')
