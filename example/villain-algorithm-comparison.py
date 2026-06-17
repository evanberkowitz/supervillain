#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain
from supervillain.analysis import Uncertain
import supervillain.analysis.comparison_plot as comparison_plot
supervillain.observable.progress=tqdm

parser = supervillain.cli.ArgumentParser(description = '''
    The goal is to compute the same observables using the Villain action with two different
    generator suites and to check that they agree on every observable.
    In D=2 the second suite includes the worm; in D>2 the worm is omitted (it is genuinely 2D-specific).
    Since without the worm we have no W>1 algorithm, we restrict our attention to W=1.
    ''')
parser.add_argument('--D', type=int, default=2, help='Number of spacetime dimensions.  Defaults to 2.')
parser.add_argument('--N', type=int, default=5, help='Sites on a side.  Defaults to 5.')
parser.add_argument('--kappa', type=float, default=0.25, help='κ.  Defaults to 0.25.')
parser.add_argument('--configurations', type=int, default=100000, help='Defaults to 100000.  You need a good deal of configurations with κ=0.5 because of autocorrelations in the Villain sampling.')
parser.add_argument('--figure', default=False, type=str)
parser.add_argument('--observables', nargs='*', help='Names of observables to compare.  Defaults to a list of 7 observables.',
                    default=('ActionDensity',
                             'InternalEnergyDensity', 'InternalEnergyDensitySquared',
                             'WindingSquared',
                             ))

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)

logger.info(f'Running with observables: {args.observables}')

# First create the lattice and the action.
L = supervillain.lattice.Lattice(D=args.D, N=args.N)
S = supervillain.action.Villain(L, args.kappa, W=1)

with logging_redirect_tqdm():
    g = supervillain.generator.combining.Sequentially((
            supervillain.generator.villain.NeighborhoodUpdate(S),
        ))
    n = supervillain.Ensemble(S).generate(
            args.configurations,
            g,
            start='cold',
            progress=tqdm)

    print(g.report())

    local_generators = [
            supervillain.generator.villain.LinkUpdate(S),
            supervillain.generator.villain.SiteUpdate(S),
            supervillain.generator.villain.ExactUpdate(S),
            supervillain.generator.villain.HolonomyUpdate(S),
    ]
    if args.D == 2:
        local_generators.append(supervillain.generator.villain.Worm(S))

    G = supervillain.generator.combining.Sequentially(local_generators)
    w = supervillain.Ensemble(S).generate(
            args.configurations,
            G,
            progress=tqdm)

    print(G.report())


# A first computation of the autocorrelation time will have effects from thermalization.
n_autocorrelation = n.autocorrelation_time(args.observables)
w_autocorrelation = w.autocorrelation_time(args.observables)

# We aggressively cut to ensure thermalization.
n_thermalized = n.cut(10*n_autocorrelation)
w_thermalized = w.cut(10*w_autocorrelation)

# Now we can get a fair computation of the autocorrelation time.
n_autocorrelation = n_thermalized.autocorrelation_time(args.observables)
w_autocorrelation = w_thermalized.autocorrelation_time(args.observables)

label_n = 'NeighborhoodUpdate'
label_w = 'local suite + worm' if args.D == 2 else 'local suite'

print(f'Autocorrelation time')
print(f'--------------------')
print(f'{label_n:<20} {n_autocorrelation}')
print(f'{label_w:<20} {w_autocorrelation}')

n_decorrelated = n_thermalized.every(n_autocorrelation)
w_decorrelated = w_thermalized.every(w_autocorrelation)

# We can easily get a bootstrap object, which we will use for estimates.
n_bootstrap = supervillain.analysis.Bootstrap(n_decorrelated)
w_bootstrap = supervillain.analysis.Bootstrap(w_decorrelated)

# The rest is show business!
fig, ax = comparison_plot.setup(args.observables)
comparison_plot.bootstraps(ax,
        (n_bootstrap, w_bootstrap),
        (label_n, label_w),
        observables=args.observables
        )
comparison_plot.histories(ax,
        (n, w),
        (label_n, label_w),
        observables=args.observables
        )

fig.suptitle(f'Villain D={args.D} N={args.N} κ={args.kappa} W=1')
fig.tight_layout()

if args.figure:
    fig.savefig(args.figure)
else:
    plt.show()
