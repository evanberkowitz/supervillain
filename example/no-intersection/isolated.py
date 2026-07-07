#!/usr/bin/env python
r"""
Is any valid configuration ISOLATED under the full structured library?

This is the decision tool for the kinetic-trap question (ergodicity.md §7.4/§8): the
frozen configurations prove that single-link sweeps alone are not ergodic, and the
global generators (WrappingLoopUpdate, PlanarFluxUpdate) were built to escape them
--- but "escapes the known examples" is weaker than "escapes the known trap
*classes*".  This script closes that gap two ways:

1. ``census(n)``: for ANY configuration, exhaustively enumerate the legal moves of
   every structured-generator class --- all single-link ±1 moves, ALL wrapping loops
   (axis and diagonal rings, both signs, every position), and ALL planar fluxes
   (every distinct decomposable A = u ∧ v with every anchor parity; on the even
   lattice the anchor enters legality only mod 2).  A configuration with zero legal
   moves in every class would be F-isolated under the structured library (Exact and
   Cohomology updates move only within the fixed-F fiber).

2. A COMPLETE sweep of the known trap families:

   * the single-pair family (all three complementary plane pairs, all nonzero
     coefficients a, b ∈ [-3, 3]²), and
   * the six-plane family (all A ∈ [-2, 2]⁶ on the Pfaffian quadric Pf(A) = 0 ---
     the exact validity condition, frozen.py).

   For every valid member the script decides: is it frozen (every single-link move
   blocked)?  If frozen, does at least one global move escape it?

Everything is early-exited (a single clean link disqualifies "frozen"; a single
legal global move settles "escapable"), so the sweep is fast; ``--full`` forces
complete counts for the flagship configurations.

Run from example/no-intersection/:

    uv run python isolated.py [--N 4] [--full]
"""

import argparse
import itertools

import numpy as np

from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.charge import charge
from frozen import build_single_pair, build_six_plane, pfaffian


# ─────────────────────────────────────────────── exhaustive proposal catalogs

def wrapping_loops(N):
    """Every distinct WrappingLoopUpdate proposal: axis rings and diagonal rings,
    both signs, every transverse position.  Yields dicts link -> coefficient."""
    seen = set()
    for mu in range(4):
        transverse = [a for a in range(4) if a != mu]
        for s in (1, -1):
            # axis rings
            for nu in transverse:
                others = [a for a in range(4) if a != nu]
                for fixed in itertools.product(range(N), repeat=3):
                    coords = dict(zip(others, fixed))
                    change = {}
                    for t in range(N):
                        coords[nu] = t
                        link = (mu,) + tuple(coords[k] for k in range(4))
                        change[link] = change.get(link, 0) + s
                    key = frozenset(change.items())
                    if key not in seen:
                        seen.add(key)
                        yield change
            # diagonal rings
            for nu, rho in itertools.permutations(transverse, 2):
                if nu > rho:
                    continue
                for delta in (1, -1):
                    for offset in range(N):
                        third = [a for a in transverse if a not in (nu, rho)][0]
                        for fx in range(N):
                            for fmu in range(N):
                                coords = {third: fx, mu: fmu}
                                change = {}
                                for t in range(N):
                                    coords[nu] = t
                                    coords[rho] = (delta * t + offset) % N
                                    link = (mu,) + tuple(coords[k] for k in range(4))
                                    change[link] = change.get(link, 0) + s
                                key = frozenset(change.items())
                                if key not in seen:
                                    seen.add(key)
                                    yield change


def planar_sheets(L):
    """Every distinct PlanarFluxUpdate proposal that matters for legality: the
    deposited field strength depends on (A = u ∧ v, anchor mod 2) only."""
    N = L.N
    seen = set()
    for u in itertools.product((-1, 0, 1), repeat=4):
        for v in itertools.product((-1, 0, 1), repeat=4):
            A = {(mu, nu): u[mu] * v[nu] - u[nu] * v[mu]
                 for mu in range(4) for nu in range(mu + 1, 4)}
            if not any(A.values()):
                continue
            for t in itertools.product((0, 1), repeat=4):
                key = (tuple(sorted(A.items())), t)
                if key in seen:
                    continue
                seen.add(key)
                dn = L.zeros(1, dtype=int)
                arr = np.asarray(dn)
                g = np.meshgrid(*(range(N),) * 4, indexing='ij')
                step = [(g[i] - t[i]) % 2 for i in range(4)]
                sign = [1 - 2 * step[i] for i in range(4)]
                arr[1] = A[(0, 1)] * sign[1] * step[0]
                arr[2] = (A[(0, 2)] * step[0] + A[(1, 2)] * step[1]) * sign[2]
                arr[3] = (A[(0, 3)] * step[0] + A[(1, 3)] * step[1]
                          + A[(2, 3)] * step[2]) * sign[3]
                yield dn


def census(n, full=False, periodic=False):
    """(frozen?, escapes) for configuration ``n`` under the structured library.

    ``escapes`` maps generator class -> number of legal moves found (stopping at the
    first one per class unless ``full``)."""
    L = n.lattice
    N = L.N
    q0 = np.asarray(charge(n))
    assert not q0.any(), 'census requires a valid configuration'

    trial = n.copy()
    arr = np.asarray(trial)

    def legal_change(change):
        for link, c in change.items():
            arr[link] += c
        ok = not np.asarray(charge(trial)).any()
        for link, c in change.items():
            arr[link] -= c
        return ok

    def legal_form(dn):
        a = np.asarray(dn)
        arr_all = np.asarray(trial)
        arr_all += a
        ok = not np.asarray(charge(trial)).any()
        arr_all -= a
        return ok

    # Single links: frozen ⟺ zero clean moves here.  For mod-2 periodic
    # configurations (both trap families) translation symmetry makes the 2⁴
    # representative sites exhaustive.
    sites = itertools.product(range(2 if periodic else N), repeat=4)
    clean_links = 0
    for site in sites:
        for mu in range(4):
            for c in (1, -1):
                if legal_change({(mu,) + site: c}):
                    clean_links += 1
                    if not full:
                        break
            if clean_links and not full:
                break
        if clean_links and not full:
            break
    frozen = clean_links == 0

    escapes = {'single-link': clean_links}
    for name, proposals, kind in (('wrapping', wrapping_loops(N), 'change'),
                                  ('planar', planar_sheets(L), 'form')):
        count = 0
        for p in proposals:
            ok = legal_change(p) if kind == 'change' else legal_form(p)
            if ok:
                count += 1
                if not full:
                    break
        escapes[name] = count
    return frozen, escapes


# ─────────────────────────────────────────────────────────── family sweeps

def sweep(L):
    frozen_configs, escapable, stuck = 0, 0, []
    handled = [0]

    def handle(label, n):
        nonlocal frozen_configs, escapable
        if np.asarray(charge(n)).any():
            return                     # not valid: outside the constraint surface
        frozen, esc = census(n, periodic=True)
        if frozen:
            frozen_configs += 1
            if esc['wrapping'] or esc['planar']:
                escapable += 1
            else:
                stuck.append(label)
                print(f'  {label}: FROZEN with *** NO ESCAPE ***')
        handled[0] += 1
        if handled[0] % 250 == 0:
            print(f'  ... {handled[0]} valid configurations checked '
                  f'({frozen_configs} frozen, all but {len(stuck)} escapable)', flush=True)

    print('== single-pair family: 3 plane pairs × a, b ∈ [-3, 3] nonzero ==')
    for pair in ('01-23', '02-13', '03-12'):
        for a in range(-3, 4):
            for b in range(-3, 4):
                if a == 0 or b == 0:
                    continue
                handle(f'pair {pair}, a={a:+d}, b={b:+d}',
                       build_single_pair(L, a=a, b=b, pair=pair))

    print()
    print('== six-plane family: A ∈ [-2, 2]⁶ with Pf(A) = 0, A ≠ 0 ==')
    PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    count = 0
    for entries in itertools.product(range(-2, 3), repeat=6):
        A = dict(zip(PAIRS, entries))
        if not any(entries) or pfaffian(A) != 0:
            continue
        count += 1
        handle(f'A={entries}', build_six_plane(L, A))
    print(f'  ({count} valid six-plane members swept)')

    print()
    print(f'SUMMARY: {frozen_configs} frozen configurations found; '
          f'{escapable} escapable by global moves; {len(stuck)} stuck.')
    if stuck:
        print('STUCK (F-isolated under the structured library!):')
        for s in stuck:
            print('  ', s)
    else:
        print('Every frozen configuration in the complete families has a global escape:')
        print('the known kinetic-trap classes are fully covered by the global generators.')
    return stuck


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--N', type=int, default=4, help='lattice size (default 4)')
    parser.add_argument('--full', action='store_true',
                        help='complete counts (no early exit) for the flagship traps')
    args = parser.parse_args()

    L = Lattice(4, args.N)

    if args.full:
        print('== flagship traps, complete counts ==')
        for label, n in (('single-pair a=b=1', build_single_pair(L, 1, 1)),
                         ('mixed pair a=2, b=1', build_single_pair(L, 2, 1)),
                         ('six-plane Pf=0 default',
                          build_six_plane(L, {(0, 1): 1, (0, 2): 2, (0, 3): 1,
                                              (1, 2): 1, (1, 3): 1, (2, 3): 1}))):
            frozen, esc = census(n, full=True)
            print(f'  {label}: frozen={frozen}; legal moves: {esc}')
        print()

    sweep(L)
