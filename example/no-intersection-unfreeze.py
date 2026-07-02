#!/usr/bin/env python
r"""
Which moves unfreeze the frozen configurations of the No-Intersection model?

``example/no-intersection-frozen.py`` builds valid configurations (q = dn ∧ dn = 0)
with zero legal single-link ±1 moves.  This script scans progressively larger move
classes on both constructions and reports which are legal:

  1. single links n_ℓ → n_ℓ ± 1                        (frozen: 0 legal, by construction)
  2. all 2-link templates with offsets within ±1        (0 legal)
  3. straight open segments, any direction and length   (0 legal)
  4. the IntersectionWorm's move library — exhaustive
     over ≤3-link unit-dipole moves                     (0 clean proposals)
  5. single-direction transverse rings: n_μ ± 1 on all
     N links of a ν-ring, ν ≠ μ                         (LEGAL — these unfreeze)

The pattern has a reason.  The frozen backgrounds are space-filling and staggered,
so any Δn of *bounded* support has an extremal ΔF plaquette; if that plaquette lies
in a plane whose complement is lit, its wedge cross term against the
everywhere-nonzero background deposits charge at the extremal corner where nothing
can cancel it.  A legal move must therefore have ΔF ≡ 0 in every plane complementary
to a lit one — e.g. for the single-pair {01, 23} construction, ΔF_01 = ΔF_23 = 0,
i.e. Δn_0 constant along x_1 — and constancy along a direction of the torus forces
the support to wrap it.  The minimal legal moves are the length-N rings, which is
exactly the axis-ring branch of
:class:`~supervillain.generator.no_intersection.WrappingLoopUpdate`; because only
one link direction changes, ΔF ∧ ΔF ≡ 0 and Δq is exactly linear, and the legal
rings are the kernel vectors that avoid the lit complementary planes.

The scans exploit that both constructions are periodic with period 2, so template
base points only range over one 2⁴ unit cell.

Usage:
    python example/no-intersection-unfreeze.py [--N 4]
"""

import argparse
import importlib.util
import pathlib
from itertools import product

import numpy as np

import supervillain
from supervillain.lattice import Lattice, d
from supervillain.generator.no_intersection import IntersectionWorm, WrappingLoopUpdate
from supervillain.generator.no_intersection.charge import dF_entries, local_dq, wedge_pairs

_here = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location('frozen', _here / 'no-intersection-frozen.py')
frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen)


def legal(L, F, change, pairs):
    """A change is a legal unfreezer if it alters F yet leaves q = 0 everywhere."""
    merged = {l: c for l, c in change.items() if c != 0}
    if not merged or not dF_entries(L, merged):
        return False
    return not local_dq(L, F, merged, pairs=pairs)


def ring(L, mu, nu, base, c):
    """n_μ += c on all N links of a ν-ring through base."""
    N = L.N
    return {(mu,) + tuple((base[k] + j * (k == nu)) % N for k in range(4)): c
            for j in range(N)}


def scan(L, n, worm, label):
    N = L.N
    pairs = wedge_pairs(L)
    F = np.asarray(d(n)).astype(int)
    print(f'=== {label} ===')

    single = sum(legal(L, F, {(mu,) + s: c}, pairs)
                 for mu in range(4) for s in product(range(N), repeat=4) for c in (1, -1))
    print(f'  single links:               {single} legal of {2 * 4 * N ** 4}')

    two = 0
    for mu, base in product(range(4), product(range(2), repeat=4)):
        for nu, off in product(range(4), product((-1, 0, 1), repeat=4)):
            s2 = tuple((base[k] + off[k]) % N for k in range(4))
            if (nu,) + s2 == (mu,) + base:
                continue
            two += sum(legal(L, F, {(mu,) + base: c1, (nu,) + s2: c2}, pairs)
                       for c1, c2 in product((1, -1), repeat=2))
    print(f'  2-link templates (unit cell, offsets ±1): {two} legal')

    segments = sum(legal(L, F, {(mu,) + tuple((base[k] + j * (k == mu)) % N
                                              for k in range(4)): c for j in range(ell)}, pairs)
                   for mu in range(4) for base in product(range(2), repeat=4)
                   for ell in range(2, N) for c in (1, -1))
    print(f'  straight open segments (unit cell):       {segments} legal')

    rng = np.random.default_rng(1)
    worm.rng = rng
    clean = sum(worm._sheet_segment(F,
                                    tuple(int(x) for x in rng.integers(0, N, size=4)),
                                    int(rng.integers(0, 4)),
                                    int(rng.choice([1, -1])))[0] is not None
                for _ in range(3000))
    print(f'  worm proposals (exhaustive ≤3-link library): {clean}/3000 clean')

    hits = {}
    for mu, nu in product(range(4), repeat=2):
        if nu == mu:
            continue
        for base in product(range(2), repeat=4):
            for c in (1, -1):
                if legal(L, F, ring(L, mu, nu, base, c), pairs):
                    hits[(mu, nu)] = hits.get((mu, nu), 0) + 1
    print(f'  transverse rings, legal (link dir, ring dir) classes: {hits or "NONE"}')
    print()
    return hits


def demonstrate_escape(L, n, label):
    """Run the actual WrappingLoopUpdate at κ = 0 until it accepts a clean loop."""
    S = supervillain.action.NoIntersections(L, kappa=0.0)
    G = WrappingLoopUpdate(S)
    cfg = {'phi': L.zeros(0), 'n': n.copy()}
    for attempt in range(1, 2001):
        cfg = G.step(cfg)
        if G.accepted:
            moved = int(np.abs(np.asarray(cfg['n']) - np.asarray(n)).sum())
            ok = S.valid(cfg)
            print(f'{label}: WrappingLoopUpdate escaped after {attempt} proposals '
                  f'(|Δn| = {moved} links, valid = {ok})')
            return
    print(f'{label}: no clean loop accepted in 2000 proposals')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--N', type=int, default=4)
    args = parser.parse_args()

    L = Lattice(4, args.N)
    S = supervillain.action.NoIntersections(L, kappa=0.1)
    worm = IntersectionWorm(S)

    n1 = frozen.build_single_pair(L, a=1, b=1, pair='01-23')
    scan(L, n1, worm, 'single-pair {01, 23}')
    demonstrate_escape(L, n1, 'single-pair')

    print()
    A = {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 3): 1}
    n2 = frozen.build_six_plane(L, A)
    scan(L, n2, worm, 'six-plane Pf(A) = 0')
    demonstrate_escape(L, n2, 'six-plane')
