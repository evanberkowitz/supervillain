#!/usr/bin/env python
r"""
Move census on valid backgrounds: corridor-checker v0.

A Metropolis update is an *atomic jump* n → n + Δn; there are no intermediate states,
so a proposal is constraint-legal if and only if its ENDPOINT is valid.  Ergodicity
questions therefore live on the graph whose nodes are valid configurations and whose
edges are the atomic jumps the library proposes.  This script measures the local
edge structure of that graph around interesting backgrounds:

  k=1 : single-link moves n_ℓ += c, c ∈ {±1, ±2}
  k=2 : two-link moves (all four sign pairs), second link within L1 distance --reach

Each move is classified by the charge pattern of its endpoint:

  clean  : q = 0 everywhere              (a legal Z-space jump)
  dipole : exactly one +1 and one -1     (a legal worm/G-space step; separation recorded)
  dirty  : anything else

Backgrounds: the vacuum, and the single-pair and six-plane frozen families imported
from frozen.py.  The frozen families are the known worst case --- zero clean k=1
moves by construction.  The *new* information here:

  * k=1, |c| = 2   : blocked too?  (Yes, provably: Δq = c·(F∧dδ + dδ∧F) is linear in c
                     for a single link, so if c = 1 is dirty every c ≠ 0 is dirty ---
                     the census confirms the sign structure numerically.)
  * k=1 dipoles    : can a worm carrying single-link templates OPEN and STEP on a
                     frozen background?  (If dipole moves with orthogonal/diagonal
                     separations exist, the answer is yes.)
  * k=2            : are frozen configurations still isolated for atomic two-link
                     jumps, or does the second link's cross term unblock them?

Run from example/no-intersection/ (imports frozen.py from the same directory):

    uv run python corridors.py [--N 4] [--reach 1]
"""

import argparse
import time
from collections import Counter

import numpy as np

from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.charge import charge
from frozen import build_single_pair, build_six_plane


def classify(dq, N):
    """('clean'|'dipole'|'dirty', canonical separation for dipoles).

    ``dq`` is the endpoint charge (background charge is 0), shape (1,) + dims.
    The dipole separation is (plus − minus) mod N, centered to (−N/2, N/2], and
    sign-canonicalized (s ~ −s) so opposite orientations pool in the histogram.
    """
    nz = np.argwhere(dq != 0)
    if len(nz) == 0:
        return 'clean', None
    if len(nz) == 2:
        (a, b) = nz
        va, vb = int(dq[tuple(a)]), int(dq[tuple(b)])
        if {va, vb} == {1, -1}:
            plus, minus = (a, b) if va == 1 else (b, a)

            def center(raw):
                return tuple(x - N if x > N // 2 else x for x in (r % N for r in raw))

            sep = center(int(p) - int(m) for p, m in zip(plus[1:], minus[1:]))
            return 'dipole', max(sep, center(-x for x in sep))
    return 'dirty', None


def census(n, reach=1):
    """Classify every k=1 (c ∈ ±1, ±2) and k=2 (links within L1 ``reach``) move on ``n``.

    Returns (k1 Counter, k1 dipole-separation Counter, k2 Counter, k2 dipole-separation
    Counter, first few clean-k2 examples).  Mutates a scratch copy in place, undoing
    each trial, exactly like frozen.exhaustive_check.
    """
    L = n.lattice
    N, D = L.N, L.D

    q0 = np.asarray(charge(n))
    assert not q0.any(), 'census requires a valid background'

    trial = n.copy()
    arr = np.asarray(trial)

    sites = list(np.ndindex(*(N,) * D))
    links = [(mu,) + s for mu in range(D) for s in sites]

    def probe(changes):
        for link, c in changes:
            arr[link] += c
        dq = np.asarray(charge(trial))
        for link, c in changes:
            arr[link] -= c
        return classify(dq, N)

    k1, k1_seps = Counter(), Counter()
    for link in links:
        for c in (1, -1, 2, -2):
            kind, sep = probe([(link, c)])
            k1[(abs(c), kind)] += 1
            if kind == 'dipole' and abs(c) == 1:
                k1_seps[sep] += 1

    # Second-link offsets: L1 distance ≤ reach from the first link's site.
    offsets = [d for d in np.ndindex(*(2 * reach + 1,) * D)
               if sum(abs(x - reach) for x in d) <= reach]
    offsets = [tuple(x - reach for x in d) for d in offsets]

    k2, k2_seps = Counter(), Counter()
    clean2 = []
    for link in links:
        s = link[1:]
        for nu in range(D):
            for off in offsets:
                partner = (nu,) + tuple((s[i] + off[i]) % N for i in range(D))
                if partner <= link:        # unordered pairs, distinct links
                    continue
                for cs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    kind, sep = probe([(link, cs[0]), (partner, cs[1])])
                    k2[kind] += 1
                    if kind == 'dipole':
                        k2_seps[sep] += 1
                    elif kind == 'clean' and len(clean2) < 5:
                        clean2.append((link, cs[0], partner, cs[1]))
    return k1, k1_seps, k2, k2_seps, clean2


def worm_census(n):
    """Try every IntersectionWorm library template at every head position and sign.

    A worm inserted at ``head`` may take its first step with shape S, displacement d,
    sign s iff the endpoint charge is exactly {head+s·d: +1, head: -1} (what
    _sheet_segment verifies).  We count those legal first steps, and separately any
    ±-dipole endpoints (a dipole in the 'wrong' place is still an immersed-sector
    configuration, just not the one the worm asked for)."""
    import supervillain.action
    from supervillain.generator.no_intersection.worm import IntersectionWorm

    L = n.lattice
    N = L.N
    S = supervillain.action.NoIntersections(L, kappa=1.0)
    G = IntersectionWorm(S)

    q0 = np.asarray(charge(n))
    assert not q0.any(), 'worm census requires a valid background'
    trial = n.copy()

    legal, any_dipole, total = 0, 0, 0
    for dvec, shapes in G._library.items():
        for shape in shapes:
            for sign in (1, -1):
                for head in np.ndindex(*(N,) * 4):
                    change = G._change_from_shape(head, dvec, sign, shape)
                    for link, c in change.items():
                        trial[link] += c
                    dq = np.asarray(charge(trial))
                    for link, c in change.items():
                        trial[link] -= c
                    total += 1
                    kind, _ = classify(dq, N)
                    if kind == 'dipole':
                        any_dipole += 1
                        target = tuple((head[k] + sign * dvec[k]) % N for k in range(4))
                        defects = {tuple(int(x) for x in h[1:]): int(dq[tuple(h)])
                                   for h in np.argwhere(dq != 0)}
                        if defects == {target: 1, tuple(head): -1}:
                            legal += 1
    return legal, any_dipole, total


def report(name, n, reach, do_census=True, do_worm=False):
    print(f'== {name} ==')
    start = time.time()
    if do_worm:
        legal, any_dipole, total = worm_census(n)
        print(f'  worm library: {legal} legal first steps of {total} (template, sign, head) '
              f'placements; {any_dipole} produce a ± dipole somewhere')
    if not do_census:
        print(f'  ({time.time() - start:.1f}s)')
        print()
        return
    k1, k1_seps, k2, k2_seps, clean2 = census(n, reach=reach)
    for absc in (1, 2):
        tot = sum(v for (a, _), v in k1.items() if a == absc)
        row = {kind: k1.get((absc, kind), 0) for kind in ('clean', 'dipole', 'dirty')}
        print(f'  k=1 |c|={absc}: clean {row["clean"]:5d}   dipole {row["dipole"]:5d}   '
              f'dirty {row["dirty"]:5d}   (of {tot})')
    if k1_seps:
        print(f'  k=1 dipole separations (canonical): '
              + ', '.join(f'{s}×{c}' for s, c in sorted(k1_seps.items(), key=lambda kv: -kv[1])))
    tot2 = sum(k2.values())
    print(f'  k=2 (reach L1≤{reach}): clean {k2.get("clean", 0):6d}   '
          f'dipole {k2.get("dipole", 0):6d}   dirty {k2.get("dirty", 0):6d}   (of {tot2})')
    if k2_seps:
        top = sorted(k2_seps.items(), key=lambda kv: -kv[1])[:6]
        print(f'  k=2 dipole separations (top): ' + ', '.join(f'{s}×{c}' for s, c in top))
    if clean2:
        print(f'  example clean k=2 moves:')
        for l1, c1, l2, c2 in clean2:
            print(f'    n{l1} += {c1:+d},  n{l2} += {c2:+d}')
    print(f'  ({time.time() - start:.1f}s)')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--N', type=int, default=4, help='lattice size (even, default 4)')
    parser.add_argument('--reach', type=int, default=1,
                        help='L1 distance allowed between the two links of a k=2 move (default 1)')
    parser.add_argument('--worm', action='store_true',
                        help='also try every IntersectionWorm library template everywhere')
    parser.add_argument('--no-census', action='store_true',
                        help='skip the k=1/k=2 census (e.g. to run --worm alone)')
    args = parser.parse_args()

    L = Lattice(4, args.N)
    print(f'Lattice D=4, N={args.N}; {4 * args.N ** 4} links.')
    print()

    kwargs = dict(reach=args.reach, do_census=not args.no_census, do_worm=args.worm)
    report('vacuum (n = 0)', L.zeros(1, dtype=int), **kwargs)
    report('single-pair frozen (a=b=1, pair 01-23)',
           build_single_pair(L, a=1, b=1, pair='01-23'), **kwargs)
    A = {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 3): 1}   # Pf(A) = 0
    report('six-plane frozen (Pf(A)=0 defaults)', build_six_plane(L, A), **kwargs)
