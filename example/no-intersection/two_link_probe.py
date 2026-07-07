#!/usr/bin/env python
r"""
Does the two-link enrichment actually move the head where the base adaptive worm stalls?

On a Hammer-thermalized (hot F) background, run each worm alone for a fixed number of
worms and compare mean worm length and how far the head travels.  This is a mixing
sanity check, not a proof; the exactness gate is the test suite.

Note: TwoLinkAdaptiveWorm enumerates its ~15k-shape family on every proposal, so each
worm is ~15-30 s (vs ~ms for the base adaptive worm).  Keep --worms modest.

    uv run python two_link_probe.py --N 5 --kappa 0.1 --worms 20
"""

import argparse

import numpy as np

import supervillain
import supervillain.generator.no_intersection as gen
from supervillain.lattice import Lattice, d

from benchmark import _thermalize


def _run(worm, cfg, worms):
    moved = 0.0
    for _ in range(worms):
        before = np.asarray(cfg['n']).copy()
        cfg = worm.step(cfg)
        moved += float((np.asarray(cfg['n']) != before).any())
    return worm, cfg, moved / worms


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--N', type=int, default=5)
    p.add_argument('--kappa', type=float, default=0.1)
    p.add_argument('--thermalize', type=int, default=10)
    p.add_argument('--worms', type=int, default=20)
    p.add_argument('--seed', type=int, default=12345)
    args = p.parse_args()

    L = Lattice(4, args.N)
    S = supervillain.action.NoIntersections(L, kappa=args.kappa)
    cfg = _thermalize(S, args.thermalize, args.seed)
    flux = int(np.abs(np.asarray(d(cfg['n']))).sum())
    print(f'N={args.N} kappa={args.kappa} thermalized sheet area Sigma|F| = {flux}\n')

    for name, cls in (('adaptive', gen.AdaptiveIntersectionWorm),
                      ('two-link', gen.TwoLinkAdaptiveWorm)):
        w = cls(S)
        w.rng = np.random.default_rng(args.seed)
        w, _out, frac = _run(w, dict(cfg), args.worms)
        lengths = np.array(w.worm_lengths)
        print(f'{name:>9}: mean worm length {lengths.mean():8.2f}   '
              f'fraction of worms that changed n {frac:.3f}')


if __name__ == '__main__':
    main()
