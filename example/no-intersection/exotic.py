#!/usr/bin/env python
r"""
Decide whether the worm needs moves beyond its committed library: mixed-magnitude
("Diophantine") 2-link moves, background-activated (±1, ±1) 2-link moves, and
taxicab-3 one-link dipoles.

Method: exact bilinear assembly.  For Δn = Σ_i c_i δ_{ℓ_i} the charge change is

    Δq = Σ_i c_i L_i  +  Σ_{i<j} c_i c_j M_{ij},

exactly (all self-wedges vanish link by link), with L_i = charge(n + δ_i) − charge(n)
the background-linear response (one charge() per link, computed once per background)
and M_ij = charge(δ_i + δ_j) the background-INDEPENDENT cross term (one charge() per
canonical relative geometry, cached).  Every candidate move is then classified by
sparse arithmetic without further charge() calls, and every reported discovery — and a
random sample of ordinary candidates — is re-verified with a direct charge() call.

Move tiers:

  CUR : the committed IntersectionWorm library (multi-link templates + 1-link ±1)
  T1  : all 2-link (±1, ±1) moves, partner within L1 ≤ --window   (background-
        activated ±1 pairs; NOT in the library, but no Diophantine content)
  T2  : 2-link mixed magnitudes, |c_i| ≤ --cmax, not both |c| = 1  (the Diophantine
        tower of §4.8b in ergodicity.md; the (2,2)-only example lives here)
  TX3 : 1-link dipoles with taxicab-3 head displacement            (would require new
        worm buckets, changing 2M)

Two provable scope restrictions keep the scan honest and finite:

  * Distant pairs add nothing: if the two links' responses have disjoint supports and
    no cross term, cleanliness forces each link to be clean alone.  So only partners
    within a small window matter.
  * On the vacuum T2 adds nothing: L ≡ 0 there, so a dipole needs c_1 c_2 M = ±1
    pattern, forcing |c_1 c_2| = 1 — the same divisibility argument that killed
    single-link magnitudes.  (The vacuum is scanned for CUR exits only.)

Decision data, per background:

  * clean-move census: does T1 or T2 provide clean Z-moves where the (±1, ±1) census
    of corridors.py provided none?  The frozen families are the known worst case ---
    corridors.py found ZERO clean and ZERO dipole moves there at (±1, ±1), but mixed
    magnitudes were never tested.
  * per-head exit census: heads with no CUR worm exit ("trapped"), and which tiers
    open them.
  * TX3 census: do taxicab-3 one-link dipoles occur at all, and is any head's ONLY
    exit taxicab-3?

Backgrounds (all mod-2 periodic, so first links are restricted to the 2^4 canonical
sites without loss): vacuum, single staggered plane (a=1, b=0), a multiplicity-2 plane
(a=2, b=0), the mixed pair (a=2, b=1), the single-pair frozen family (a=b=1), and the
six-plane frozen family.  With --knots, also the (aperiodic) trefoil torus and spun
trefoil at N=8, with first links restricted to the sheet's neighbourhood.

Run from example/no-intersection/:

    uv run python exotic.py [--N 4] [--cmax 3] [--window 3] [--knots]
"""

import argparse
import itertools
import time
from collections import Counter

import numpy as np

import supervillain.action
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection.worm import IntersectionWorm
from frozen import build_single_pair, build_six_plane


def sparse(q):
    """A charge array as a {hypercube: value} dict of its nonzero entries."""
    arr = np.asarray(q)
    return {tuple(int(x) for x in h[1:]): int(arr[tuple(h)]) for h in np.argwhere(arr != 0)}


def center(sep, N):
    return tuple(x - N if x > N // 2 else x for x in (s % N for s in sep))


def classify(dq, N):
    """('clean'|'dipole'|'dirty', plus, minus, taxicab separation) for sparse dq."""
    if not dq:
        return 'clean', None, None, None
    if len(dq) == 2:
        (a, va), (b, vb) = dq.items()
        if {va, vb} == {1, -1}:
            plus, minus = (a, b) if va == 1 else (b, a)
            sep = center((p - m for p, m in zip(plus, minus)), N)
            return 'dipole', plus, minus, sum(abs(x) for x in sep)
    return 'dirty', None, None, None


class Assembler:
    """Exact sparse Δq for arbitrary multi-link moves on a fixed background."""

    def __init__(self, n):
        self.L = n.lattice
        self.N = self.L.N
        self.n = n
        self.arr = np.asarray(n)
        q0 = np.asarray(charge(n))
        assert not q0.any(), 'background must be valid'
        # Background-linear response of every link, one charge() each.
        self.resp = {}
        scratch = n.copy()
        sarr = np.asarray(scratch)
        for link in ((mu,) + s for mu in range(4) for s in np.ndindex(*(self.N,) * 4)):
            sarr[link] += 1
            self.resp[link] = sparse(charge(scratch))
            sarr[link] -= 1
        # Background-independent cross terms, cached by relative geometry.
        self._cross_cache = {}
        self._empty = self.L.zeros(1, dtype=int)

    def cross(self, l1, l2):
        """M_12 = charge(δ_1 + δ_2), translated to l1's position."""
        (m1, *s1), (m2, *s2) = l1, l2
        rel = tuple((b - a) % self.N for a, b in zip(s1, s2))
        key = (m1, m2, rel)
        if key not in self._cross_cache:
            arr = np.asarray(self._empty)
            arr[(m1,) + (0,) * 4] += 1
            arr[(m2,) + rel] += 1
            self._cross_cache[key] = sparse(charge(self._empty))
            arr[(m1,) + (0,) * 4] -= 1
            arr[(m2,) + rel] -= 1
        return {tuple((h + a) % self.N for h, a in zip(hyper, s1)): v
                for hyper, v in self._cross_cache[key].items()}

    def assemble(self, changes):
        """Sparse Δq of the atomic move ``changes`` = [(link, coefficient), ...]."""
        out = {}
        for link, c in changes:
            for h, v in self.resp[link].items():
                out[h] = out.get(h, 0) + c * v
        for (l1, c1), (l2, c2) in itertools.combinations(changes, 2):
            for h, v in self.cross(l1, l2).items():
                out[h] = out.get(h, 0) + c1 * c2 * v
        return {h: v for h, v in out.items() if v}

    def direct(self, changes):
        """The same Δq by brute force, for verification."""
        for link, c in changes:
            self.arr[link] += c
        dq = sparse(charge(self.n))
        for link, c in changes:
            self.arr[link] -= c
        return dq

    def verify(self, changes):
        assert self.assemble(changes) == self.direct(changes), changes


def cur_exits(A, worm, heads):
    """{head: exit count} under the committed worm library, via exact assembly."""
    N = A.N
    exits = {h: 0 for h in heads}
    for head in heads:
        for d in worm._directions:
            for sign in (1, -1):
                target = tuple((head[k] + sign * d[k]) % N for k in range(4))
                want = {target: 1, head: -1}
                for shape in worm._library[d]:
                    change = worm._change_from_shape(head, d, sign, shape)
                    if A.assemble(list(change.items())) == want:
                        exits[head] += 1
                        break
    return exits


def one_link_exits(A):
    """All 1-link dipole moves, bucketed by taxicab class; exits keyed by head (= minus end)."""
    N = A.N
    by_taxi, exits = Counter(), {}
    examples = []
    for link, resp in A.resp.items():
        for c in (1, -1):
            dq = {h: c * v for h, v in resp.items()}
            kind, plus, minus, taxi = classify(dq, N)
            if kind == 'dipole':
                by_taxi[taxi] += 1
                exits.setdefault(minus, Counter())[taxi] += 1
                if taxi >= 3 and len(examples) < 3:
                    examples.append(((link, c), plus, minus))
    return by_taxi, exits, examples


def decomposes(A, l1, c1, l2, c2):
    """True if the 2-link move splits into unit steps through valid intermediates.

    Search monotone unit-step paths (0,0) → (c1,c2) in coefficient space, requiring
    every intermediate node's Δq to vanish (a valid configuration).  A decomposable
    clean move is a sequence of legal single-link constrained moves; a decomposable
    dipole move is a sequence of legal 1-link *idle* worm moves followed by 1-link
    head moves --- both already in the committed repertoire, so only moves this test
    fails to decompose can possibly be *needed*.  (Monotone-clean is a sufficient
    decomposition, so the irreducible counts are upper bounds on genuinely-new moves.)
    """
    M = A.cross(l1, l2)
    r1, r2 = A.resp[l1], A.resp[l2]

    def clean(a, b):
        out = {}
        for h, v in r1.items():
            out[h] = out.get(h, 0) + a * v
        for h, v in r2.items():
            out[h] = out.get(h, 0) + b * v
        for h, v in M.items():
            out[h] = out.get(h, 0) + a * b * v
        return not any(out.values())

    s1, s2 = (1 if c1 > 0 else -1), (1 if c2 > 0 else -1)
    stack, seen = [(0, 0)], {(0, 0)}
    while stack:
        a, b = stack.pop()
        for na, nb in ((a + s1, b), (a, b + s2)):
            if abs(na) > abs(c1) or abs(nb) > abs(c2) or (na, nb) in seen:
                continue
            if (na, nb) == (c1, c2):
                return True
            if clean(na, nb):
                seen.add((na, nb))
                stack.append((na, nb))
    return False


def pair_scan(A, first_links, window, cmax):
    """Classify every 2-link move (first link in ``first_links``, partner within L1
    ``window``) at tier T1 ((±1,±1)) and T2 (mixed magnitudes ≤ cmax).

    Returns per-tier Counters of clean/dipole/dirty (with ``(kind, 'irreducible')``
    subcounts for non-decomposable discoveries), per-tier {head: irreducible dipole
    exits}, and verified examples of irreducible discoveries."""
    N = A.N
    offsets = [tuple(x - window for x in d) for d in np.ndindex(*(2 * window + 1,) * 4)
               if sum(abs(x - window) for x in d) <= window]
    T1 = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    T2 = [(c1, c2) for c1 in range(-cmax, cmax + 1) for c2 in range(-cmax, cmax + 1)
          if c1 and c2 and max(abs(c1), abs(c2)) > 1]
    counts = {'T1': Counter(), 'T2': Counter()}
    exits = {'T1': {}, 'T2': {}}
    examples = {('T1', 'clean'): [], ('T1', 'dipole'): [],
                ('T2', 'clean'): [], ('T2', 'dipole'): []}
    for l1 in first_links:
        s = l1[1:]
        for nu in range(4):
            for off in offsets:
                l2 = (nu,) + tuple((s[i] + off[i]) % N for i in range(4))
                if l2 <= l1:
                    continue
                r1, r2, M = A.resp[l1], A.resp[l2], A.cross(l1, l2)
                if not (r1 or r2 or M):
                    continue
                for tier, coeffs in (('T1', T1), ('T2', T2)):
                    for c1, c2 in coeffs:
                        out = {}
                        for h, v in r1.items():
                            out[h] = out.get(h, 0) + c1 * v
                        for h, v in r2.items():
                            out[h] = out.get(h, 0) + c2 * v
                        for h, v in M.items():
                            out[h] = out.get(h, 0) + c1 * c2 * v
                        dq = {h: v for h, v in out.items() if v}
                        kind, plus, minus, taxi = classify(dq, N)
                        counts[tier][kind] += 1
                        if kind in ('clean', 'dipole'):
                            if decomposes(A, l1, c1, l2, c2):
                                continue
                            counts[tier][(kind, 'irreducible')] += 1
                            if kind == 'dipole':
                                exits[tier].setdefault(minus, Counter())[taxi] += 1
                            if len(examples[(tier, kind)]) < 3:
                                move = [(l1, c1), (l2, c2)]
                                A.verify(move)
                                examples[(tier, kind)].append((move, kind, plus, minus))
    return counts, exits, examples


def local_scan(A, head, window, cmax):
    """Exhaustive exit search at one head: every 1-link and 2-link move among links
    within L1 ``window`` of the head, all coefficient pairs up to ``cmax``.  Dipole
    exits at this head are tallied as reducible (unit-step decomposable, i.e. already
    executable by the committed idle + 1-link repertoire when the steps are
    head-local) or IRREDUCIBLE."""
    N = A.N
    offs = [tuple(x - window for x in d) for d in np.ndindex(*(2 * window + 1,) * 4)
            if sum(abs(x - window) for x in d) <= window]
    sites = [tuple((head[k] + o[k]) % N for k in range(4)) for o in offs]
    links = sorted({(mu,) + s for mu in range(4) for s in sites})
    found = Counter()
    example = None
    for l in links:
        for c in (1, -1):
            kind, plus, minus, taxi = classify({h: c * v for h, v in A.resp[l].items()}, N)
            if kind == 'dipole' and minus == head:
                found['1-link'] += 1
    coeffs = [(c1, c2) for c1 in range(-cmax, cmax + 1) for c2 in range(-cmax, cmax + 1)
              if c1 and c2]
    for i, l1 in enumerate(links):
        r1 = A.resp[l1]
        for l2 in links[i + 1:]:
            r2, M = A.resp[l2], A.cross(l1, l2)
            if not (head in r1 or head in r2 or head in M):
                continue   # a dipole with its -1 at head needs head in the support
            for c1, c2 in coeffs:
                out = {}
                for h, v in r1.items():
                    out[h] = out.get(h, 0) + c1 * v
                for h, v in r2.items():
                    out[h] = out.get(h, 0) + c2 * v
                for h, v in M.items():
                    out[h] = out.get(h, 0) + c1 * c2 * v
                dq = {h: v for h, v in out.items() if v}
                kind, plus, minus, taxi = classify(dq, N)
                if kind == 'dipole' and minus == head:
                    tier = 'T1' if abs(c1) == 1 and abs(c2) == 1 else 'T2'
                    red = 'reducible' if decomposes(A, l1, c1, l2, c2) else 'IRREDUCIBLE'
                    found[(tier, red)] += 1
                    if example is None and red == 'IRREDUCIBLE':
                        example = [(l1, c1), (l2, c2)]
                        A.verify(example)
    return found, example


def idle_escape(n, worm, head):
    """At a CUR-trapped head, can the committed idle + 1-link repertoire escape in two
    proposals?  Counts the worm's library idle proposals at ``head`` and how many of
    them, once applied, unlock at least one head move.  Uses direct charge() calls (the
    background changes under the idle), so this is the slow-but-decisive check."""
    N = n.lattice.N

    def proposals(m, first_move_only=False):
        q0 = np.asarray(charge(m))
        trial = m.copy()
        arr = np.asarray(trial)
        moves, idles = 0, []
        for d in worm._directions:
            for sign in (1, -1):
                target = tuple((head[k] + sign * d[k]) % N for k in range(4))
                want = {target: 1, head: -1}
                for shape in worm._library[d]:
                    change = worm._change_from_shape(head, d, sign, shape)
                    for link, c in change.items():
                        arr[link] += c
                    dq = sparse(np.asarray(charge(trial)) - q0)
                    for link, c in change.items():
                        arr[link] -= c
                    if dq == want:
                        if first_move_only:
                            return 1, idles
                        moves += 1
                    elif not dq and len(shape) == 1:
                        idles.append(change)
        return moves, idles

    moves, idles = proposals(n)
    assert moves == 0, 'idle_escape expects a trapped head'
    unlocked = 0
    for idle in idles:
        m = n.copy()
        arr = np.asarray(m)
        for link, c in idle.items():
            arr[link] += c
        if proposals(m, first_move_only=True)[0]:
            unlocked += 1
    return len(idles), unlocked


def report(name, n, args, first_sites=None):
    print(f'== {name} ==')
    t0 = time.time()
    L = n.lattice
    N = L.N
    A = Assembler(n)

    # Spot-check the bilinear assembly against brute force.
    rng = np.random.default_rng(20260704)
    links = list(A.resp)
    for _ in range(50):
        k = int(rng.integers(1, 4))
        picks = rng.choice(len(links), size=k, replace=False)
        A.verify([(links[int(i)], int(rng.integers(-3, 4)) or 1) for i in picks])

    S = supervillain.action.NoIntersections(L, kappa=1.0)
    worm = IntersectionWorm(S)

    heads = list(np.ndindex(*(N,) * 4))
    cur = cur_exits(A, worm, heads)
    trapped = sorted(h for h, e in cur.items() if e == 0)
    print(f'  CUR: {sum(cur.values())} worm exits over {len(heads)} heads; '
          f'{len(trapped)} heads trapped (no CUR exit)')

    taxi1, exits1, ex3 = one_link_exits(A)
    print(f'  1-link dipoles by taxicab class: '
          + (', '.join(f'{t}: {c}' for t, c in sorted(taxi1.items())) or 'none'))
    for (link, c), plus, minus in ex3:
        print(f'    TX3 example: n{link} += {c:+d}  moves a head {minus} -> {plus}')

    if first_sites is None:
        first_sites = [s for s in np.ndindex(*(min(2, N),) * 4)]
    first_links = [(mu,) + s for mu in range(4) for s in first_sites]
    counts, exits2, examples = pair_scan(A, first_links, args.window, args.cmax)
    for tier in ('T1', 'T2'):
        c = counts[tier]
        print(f'  {tier}: clean {c["clean"]:6d} ({c[("clean", "irreducible")]:d} irreducible)   '
              f'dipole {c["dipole"]:6d} ({c[("dipole", "irreducible")]:d} irreducible)   '
              f'dirty {c["dirty"]:8d}')
    for (tier, kind), exs in examples.items():
        for move, _, plus, minus in exs:
            desc = ',  '.join(f'n{l} += {c:+d}' for l, c in move)
            where = '' if kind == 'clean' else f'  (head {minus} -> {plus})'
            print(f'    {tier} irreducible {kind}: {desc}{where}')

    # Which trapped heads do the exotic tiers open with IRREDUCIBLE exits?  (Coverage
    # caveat: pair-scan exits only reach heads near the scanned first links; for the
    # mod-2 periodic backgrounds the canonical first sites cover everything up to
    # translation.)
    if trapped:
        opened = {t: sum(1 for h in trapped if h in exits2[t]) for t in ('T1', 'T2')}
        opened['TX3'] = sum(1 for h in trapped if 3 in exits1.get(h, ()))
        none = sum(1 for h in trapped
                   if h not in exits2['T1'] and h not in exits2['T2']
                   and h not in exits1)
        print(f'  trapped heads opened by irreducible exits: T1 {opened["T1"]}, '
              f'T2 {opened["T2"]}, TX3 {opened["TX3"]}; by nothing scanned: {none} '
              f'(of {len(trapped)})')
        for head in trapped[:2]:
            found, example = local_scan(A, head, args.window, args.cmax)
            desc = ', '.join(f'{k}: {v}' for k, v in sorted(found.items(), key=str)) or 'none'
            print(f'  exhaustive local scan at trapped head {head}: {desc}')
            if example:
                print('    IRREDUCIBLE exit: '
                      + ',  '.join(f'n{l} += {c:+d}' for l, c in example))
            n_idles, unlocked = idle_escape(n, worm, head)
            verdict = ('ESCAPES via committed idle + 1-link' if unlocked
                       else 'genuinely stuck locally; global moves only')
            print(f'  idle-escape check at {head}: {unlocked} of {n_idles} library idles '
                  f'unlock a head move — {verdict}')
    only3 = [h for h, t in exits1.items()
             if set(t) == {3} and cur.get(h, 0) == 0
             and h not in exits2['T1'] and h not in exits2['T2']]
    print(f'  heads whose ONLY scanned exit is taxicab-3: {len(only3)}')
    print(f'  ({time.time() - t0:.1f}s)')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--cmax', type=int, default=3,
                        help='max |coefficient| in the Diophantine tier (default 3)')
    parser.add_argument('--window', type=int, default=3,
                        help='L1 window for the second link of a pair (default 3)')
    parser.add_argument('--knots', action='store_true',
                        help='also scan the trefoil torus and spun trefoil at N=8 (slow)')
    args = parser.parse_args()

    L = Lattice(4, args.N)
    print(f'Lattice D=4, N={args.N}; tiers: T1 = (±1,±1) pairs, '
          f'T2 = mixed |c| ≤ {args.cmax}, TX3 = taxicab-3 one-link dipoles.')
    print()

    report('vacuum (CUR exits only; T2 provably empty, T1 = templates)',
           L.zeros(1, dtype=int), args)
    report('single staggered plane (a=1, b=0)', build_single_pair(L, a=1, b=0), args)
    report('multiplicity-2 plane (a=2, b=0)', build_single_pair(L, a=2, b=0), args)
    report('mixed pair (a=2, b=1)', build_single_pair(L, a=2, b=1), args)
    report('single-pair frozen (a=b=1)', build_single_pair(L, a=1, b=1), args)
    A6 = {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 3): 1}   # Pf = 0
    report('six-plane frozen (Pf(A)=0)', build_six_plane(L, A6), args)

    if args.knots:
        import torus_knotted
        import spun_sphere
        L8 = Lattice(4, 8)
        rng = np.random.default_rng(0)

        def sheet_sites(n, cap=48):
            F = np.asarray(supervillain.lattice.d(n))
            sites = sorted({tuple(int(x) for x in h[1:]) for h in np.argwhere(F != 0)})
            if len(sites) > cap:
                sites = [sites[int(i)] for i in rng.choice(len(sites), cap, replace=False)]
            return sites

        n8, _, _ = torus_knotted.configuration(L8, grid=5, shift=2)
        report('trefoil torus (N=8, first links near the sheet)', n8, args,
               first_sites=sheet_sites(n8))
        n8s, _ = spun_sphere.configuration(L8, grid=5, shift=2)
        report('spun trefoil (N=8, first links near the sheet)', n8s, args,
               first_sites=sheet_sites(n8s))
