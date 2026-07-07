#!/usr/bin/env python
r"""
Leaving the frozen sector of the :class:`~supervillain.action.NoIntersections` model.

A *frozen* configuration (see ``frozen.py``) is a dead end for the
:class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate` alone: every
single-link $\pm 1$ move violates $q = dn\wedge dn = 0$, so it is an isolated point of
the single-link move graph and a single-link-only algorithm is not ergodic.  The
coordinated :class:`~supervillain.generator.no_intersection.WrappingLoopUpdate`
connects it to the mobile bulk.  This script demonstrates that: starting *on* a frozen
configuration it interleaves

* :class:`~supervillain.generator.villain.SiteUpdate` (relaxes $\phi$; does not touch $F$),
* :class:`~supervillain.generator.no_intersection.WrappingLoopUpdate` (coordinated,
  $F$-changing wrapping loops), and
* :class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate` (a single-link sweep),

and watches single-link mobility reappear and cascade --- i.e. the combined chain
*leaves* the frozen sector.

Ergodicity is about the *connectivity* of the moves, not how often they are accepted.
It is cleanest to see at $\kappa = 0$ (the default), where the action is flat so every
constraint-preserving move is accepted and the trajectory traces exactly which
configurations the moves connect.  At physical $\kappa$ the same moves keep nonzero
acceptance, so the chain is still ergodic; the escape is merely rarer --- an
autocorrelation cost, not a correctness one.

  uv run python unfreeze.py [--N 4] [--kappa 0.0] [--rounds 60] [--wlu-per-round 100]
"""

import argparse

import numpy as np

import supervillain
from supervillain.lattice import Lattice, d

# frozen.py (this directory) is both a CLI and a small library of frozen-configuration
# constructions; we reuse its single-pair builder and its exhaustive single-link check.
from frozen import build_single_pair, exhaustive_check


def unfreeze(L, kappa, rounds, wlu_per_round):
    """Interleave SiteUpdate + WrappingLoopUpdate + ConstrainedLinkUpdate on a frozen
    configuration and report when single-link mobility returns."""
    S = supervillain.action.NoIntersections(L, kappa=kappa)
    flux = lambda m: int((np.asarray(d(m)) ** 2).sum())

    site = supervillain.generator.villain.SiteUpdate(S)
    wlu = supervillain.generator.no_intersection.WrappingLoopUpdate(S)
    clu = supervillain.generator.no_intersection.ConstrainedLinkUpdate(S)

    cfg = {'phi': L.zeros(0), 'n': build_single_pair(L)}      # start on a frozen config
    nv, nt, _ = exhaustive_check(cfg['n'])
    print(f"start:      flux ΣF²={flux(cfg['n'])}, single-link moves {nv}/{nt}  (frozen)\n")

    escaped_at = None
    for r in range(1, rounds + 1):
        cfg = site.step(cfg)                          # φ relaxes; F (hence q) untouched
        for _ in range(wlu_per_round):                # coordinated loops can change F
            cfg = wlu.step(cfg)
        cfg = clu.step(cfg)                           # single-link sweep exploits any opening
        nv, nt, _ = exhaustive_check(cfg['n'])
        print(f"round {r:4d}: flux {flux(cfg['n']):6d}, single-link moves {nv:4d}/{nt}, "
              f"WLU {wlu.accepted} acc, CLU {clu.accepted} acc")
        if nv > 0 and escaped_at is None:
            escaped_at = r
        # once mobility returns, run a few more rounds to show the cascade, then stop
        if escaped_at is not None and r >= escaped_at + 5:
            break

    print()
    if escaped_at is not None:
        print(f"LEFT THE FROZEN SECTOR at round {escaped_at}: single-link moves reappeared and "
              "ConstrainedLinkUpdate began to fire.\nWrappingLoopUpdate + ConstrainedLinkUpdate "
              "connect the frozen configuration to the mobile bulk, so the combination is ergodic.")
    else:
        print(f"Still frozen after {rounds} rounds.  The escape moves exist but have low acceptance "
              f"at κ={kappa}\n(a mixing cost, not an ergodicity failure); try --kappa 0 or more --rounds.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--N', type=int, default=4, metavar='N',
                   help='lattice size (default: 4; must be ≥ 2)')
    p.add_argument('--kappa', type=float, default=0.0, metavar='K',
                   help='Villain coupling κ (default: 0.0, the cleanest ergodicity probe)')
    p.add_argument('--rounds', type=int, default=60, metavar='R',
                   help='maximum SiteUpdate+WLU+CLU rounds (default: 60)')
    p.add_argument('--wlu-per-round', type=int, default=100, metavar='W',
                   help='WrappingLoopUpdate steps per round (default: 100)')
    args = p.parse_args()
    if args.N < 2:
        p.error('N must be ≥ 2')

    L = Lattice(4, args.N)
    print(f"Lattice D=4, N={args.N}  ({4 * args.N ** 4} links, {args.N ** 4} hypercubes)")
    print(f"start on a frozen config; per round: SiteUpdate + "
          f"{args.wlu_per_round}×WrappingLoopUpdate + ConstrainedLinkUpdate, κ={args.kappa}\n")
    unfreeze(L, args.kappa, args.rounds, args.wlu_per_round)
