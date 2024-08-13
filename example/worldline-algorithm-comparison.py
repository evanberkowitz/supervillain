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

parser = supervillain.cli.ArgumentParser(description = 'The goal is to compute the same observables using both the Villain and Worldline actions and to check that they agree.')
parser.add_argument('--N', type=int, default=5, help='Sites on a side.')
parser.add_argument('--kappa', type=float, default=0.5, help='κ.  Defaults to 0.5.')
parser.add_argument('--W', type=int, default=1, help='Constraint integer W.  Defaults to 1')
parser.add_argument('--configurations', type=int, default=10000, help='Defaults to 10000.  You need a good deal of configurations with κ=0.5 because of autocorrelations with simple sampling.')
parser.add_argument('--figure', default=False, type=str)
parser.add_argument('--observables', nargs='*', help='Names of observables to compare.  Defaults to a list of 7 observables.',
                    default=('ActionDensity',
                             'InternalEnergyDensity', 'InternalEnergyDensitySquared',
                             'WindingSquared',
                             'TWrapping', 'XWrapping',
                             ))

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)

# First create the lattices and the action.
L = supervillain.lattice.Lattice2D(args.N)
S = supervillain.action.Worldline(L, args.kappa, W=args.W)

with logging_redirect_tqdm():
    g = supervillain.generator.combining.Sequentially((
            supervillain.generator.worldline.PlaquetteUpdate(S),
            supervillain.generator.worldline.VortexUpdate(S),
            supervillain.generator.worldline.CoexactUpdate(S),
            supervillain.generator.worldline.WrappingUpdate(S),
        ))
    n = supervillain.Ensemble(S).generate(args.configurations, g, start='cold', progress=tqdm)
    n.measure()

    W = supervillain.generator.combining.Sequentially((
            supervillain.generator.worldline.PlaquetteUpdate(S),
            supervillain.generator.worldline.VortexUpdate(S),
            supervillain.generator.worldline.CoexactUpdate(S),
            supervillain.generator.worldline.WrappingUpdate(S),
            supervillain.generator.worldline.worm.Classic(S),
        ))
    w = supervillain.Ensemble(S).generate(args.configurations, W, start='cold', progress=tqdm)
    w.measure()

# A first computation of the autocorrelation time will have effects from thermalization.
n_autocorrelation = n.autocorrelation_time()
w_autocorrelation = w.autocorrelation_time()

# We aggressively cut to ensure thermalization.
n_thermalized = n.cut(10*n_autocorrelation)
w_thermalized = w.cut(10*w_autocorrelation)

# Now we can get a fair computation of the autocorrelation time.
n_autocorrelation = n_thermalized.autocorrelation_time()
w_autocorrelation = w_thermalized.autocorrelation_time()

print(f'Autocorrelation time')
print(f'--------------------')
print(f'Updates         {n_autocorrelation}')
print(f' + Worm         {w_autocorrelation}')

n_decorrelated = n_thermalized.every(n_autocorrelation)
w_decorrelated = w_thermalized.every(w_autocorrelation)

# We can easily get a bootstrap object, which we will use for estimates.
n_bootstrap = supervillain.analysis.Bootstrap(n_decorrelated)
w_bootstrap = supervillain.analysis.Bootstrap(w_decorrelated)

# The rest is show business!
fig, ax = comparison_plot.setup(args.observables)
comparison_plot.bootstraps(ax,
        (n_bootstrap, w_bootstrap),
        ('Updates', ' + Worm'),
        observables=args.observables
        )
comparison_plot.histories(ax,
        (n, w),
        ('Updates', ' + Worm'),
        observables=args.observables
        )


fig.suptitle(f'Worldline N={args.N} κ={args.kappa} W={args.W}')
fig.tight_layout()

if args.figure:
    fig.savefig(args.figure)
else:
    plt.show()
