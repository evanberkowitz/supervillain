#!/usr/bin/env python
r"""
Census of the IntersectionWorm's move families on thermalized backgrounds.

For each move family (2- and 3-link orthogonal, the two diagonal families, and the
background-activated single links) tally how often it is drawn, how often the draw
is clean (or idle), and how often it is accepted, over many worms on a
Hammer-thermalized configuration.  This is the data that should inform any
reweighting of the shape draw (``IntersectionWorm(S, class_weights=...)``) --- or
any decision to whittle the library.

    uv run python worm_tallies.py --N 6 --kappa 0.1 --worms 200
"""

import argparse

import numpy as np

import supervillain
from supervillain.lattice import Lattice, d

from benchmark import _thermalize


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--N', type=int, default=6)
    p.add_argument('--kappa', type=float, default=0.1)
    p.add_argument('--thermalize', type=int, default=10)
    p.add_argument('--worms', type=int, default=200)
    p.add_argument('--seed', type=int, default=12345)
    args = p.parse_args()

    L = Lattice(4, args.N)
    S = supervillain.action.NoIntersections(L, kappa=args.kappa)
    cfg = _thermalize(S, args.thermalize, args.seed)
    flux = int(np.abs(np.asarray(d(cfg['n']))).sum())
    print(f'N={args.N} kappa={args.kappa} thermalized sheet area Sigma|F| = {flux}\n')

    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    worm.rng = np.random.default_rng(args.seed)
    for _ in range(args.worms):
        cfg = worm.step(cfg)

    print(worm.report())
    print()
    for family, t in worm.tallies.items():
        if t['drawn'] == 0:
            continue
        usable = (t['clean'] + t['idle']) / t['drawn']
        accepted = (t['accepted'] + t['accepted_idle']) / t['drawn']
        print(f'{family:>8}: clean-or-idle fraction {usable:.4f}   '
              f'accepted fraction {accepted:.4f}')


if __name__ == '__main__':
    main()
