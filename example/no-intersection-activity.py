#!/usr/bin/env python
r"""
The activity map: a diagnostic for frozen regions.

``supervillain.analysis.link_activity`` differences consecutive stored
configurations, so it sees the combined effect of every generator in the stream
without instrumenting any of them.  A healthy chain moves every link; a frozen
region --- a locally blocked texture that no local update can touch --- shows up as
a persistent spatial hole of zeros.

This script runs the same update stack on two starts at the same coupling:

  * a cold start, which thermalizes and moves everywhere;
  * the frozen staggered texture of ``example/no-intersection-frozen.py``,
    a local action minimum whose every legal exit move is uphill.

and prints the fraction of dead links and the activity quartiles for each.

Usage:
    python example/no-intersection-activity.py [--N 4] [--kappa 0.15] [--steps 120]
"""

import argparse
import importlib.util
import pathlib

import numpy as np

import supervillain
import supervillain.generator.villain as V
from supervillain.lattice import Lattice
from supervillain.analysis import link_activity
from supervillain.generator.no_intersection import (
    ConstrainedLinkUpdate, StringWorm, WrappingLoopUpdate)
from supervillain.generator.combining import Sequentially

_here = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location('frozen', _here / 'no-intersection-frozen.py')
frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen)


def stream(S, start, steps):
    H = Sequentially((
        V.SiteUpdate(S), V.ExactUpdate(S),
        ConstrainedLinkUpdate(S), WrappingLoopUpdate(S), StringWorm(S),
    ))
    cfg = dict(start)
    configurations = [cfg]
    for _ in range(steps):
        cfg = H.step(cfg)
        configurations.append(cfg)
    return configurations


def summarize(label, configurations):
    activity = link_activity(configurations)
    dead = float((activity == 0).mean())
    q = np.percentile(activity, (25, 50, 75))
    print(f'{label:16s} dead links {dead:6.1%}   activity quartiles '
          f'{q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--kappa', type=float, default=0.15)
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()

    L = Lattice(4, args.N)
    S = supervillain.action.NoIntersections(L, kappa=args.kappa)

    cold = {'phi': L.zeros(0), 'n': L.zeros(1, dtype=int)}
    summarize('cold start', stream(S, cold, args.steps))

    texture = {'phi': L.zeros(0),
               'n': frozen.build_single_pair(L, a=1, b=1, pair='01-23')}
    summarize('frozen texture', stream(S, texture, args.steps))
