#!/usr/bin/env python
r"""
Physics cross-check for the FreeTargetWorm.

Same physics, different sampler: at moderate kappa the Intersection_Intersection
correlator (normalized at the origin) from the FreeTargetWorm must agree with the
TwoLinkAdaptiveWorm within errors.  Run by hand; eyeball the printed comparison.

    uv run python dev/free_target_crosscheck.py --N 5 --kappa 0.3 --steps 400
"""

import argparse

import numpy as np

import supervillain
import supervillain.generator.no_intersection as gen
from supervillain.lattice import Lattice

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=5)
parser.add_argument('--kappa', type=float, default=0.3)
parser.add_argument('--steps', type=int, default=400)
parser.add_argument('--burn', type=int, default=100)
parser.add_argument('--seed', type=int, default=137)
args = parser.parse_args()


def correlator(worm_cls, seed):
    L = Lattice(4, args.N)
    S = supervillain.action.NoIntersections(L, kappa=args.kappa)
    w = worm_cls(S)
    w.rng = np.random.default_rng(seed)
    e = supervillain.Ensemble(S).generate(args.steps, w, start='cold')
    theta = np.asarray(e.Intersection_Intersection)[args.burn:]
    mean = theta.mean(axis=0)
    err = theta.std(axis=0) / np.sqrt(len(theta))
    norm = mean[L.origin]
    return mean / norm, err / norm, w


for name, cls in (('TwoLinkAdaptiveWorm', gen.TwoLinkAdaptiveWorm),
                  ('FreeTargetWorm', gen.FreeTargetWorm)):
    g, dg, w = correlator(cls, args.seed)
    print(f'--- {name}')
    print(w.report())
    # the on-axis slice is enough to eyeball agreement
    axis = tuple([slice(None)] + [0] * 3)
    for r, (v, e_) in enumerate(zip(g[axis], dg[axis])):
        print(f'  r={r}: {v:+.4f} +- {e_:.4f}')
