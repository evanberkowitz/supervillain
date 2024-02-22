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

parser = supervillain.cli.ArgumentParser(description = 'The goal is to compute the same observables using both the Villain and Worldline actions and to check that they agree.  When W>1 the Villain action is sampled with a combination of NeighborhoodUpdates and the Geometric worm.')
parser.add_argument('--N', type=int, default=5, help='Sites on a side.')
parser.add_argument('--kappa', type=float, default=0.5, help='κ.  Defaults to 0.5.')
parser.add_argument('--W', type=int, default=1, help='Constraint integer W.  Defaults to 1')
parser.add_argument('--configurations', type=int, default=100000, help='Defaults to 100000.  You need a good deal of configurations with κ=0.5 because of autocorrelations in the Villain sampling.')
parser.add_argument('--figure', default=False, type=str)
parser.add_argument('--observables', nargs='*', help='Names of observables to compare.  Defaults to a list of 5 observables.',
                    default=('ActionDensity', 'InternalEnergyDensity', 'InternalEnergyDensitySquared', 'SpinSusceptibility', 'WindingSquared'))

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)

# First create the lattices and the two dual actions.
L = supervillain.lattice.Lattice2D(args.N)

V = supervillain.action.Villain(L, args.kappa, W=args.W)
W = supervillain.action.Worldline(L, args.kappa, W=args.W)

# Now sample each action.
with logging_redirect_tqdm():
    if args.W == 1:
        g = supervillain.generator.villain.NeighborhoodUpdate(V)
    else:
        g = supervillain.generator.combining.Sequentially((
                # When W>1 the neighborhood update proposes large constraint-preserving changes (which is any change to n when W=1).
                # These large changes are often rejected, so it can be beneficial to propose updates to phi
                supervillain.generator.villain.NeighborhoodUpdate(V, interval_n = 0),
                # separately from the large updates.
                supervillain.generator.villain.NeighborhoodUpdate(V, interval_n = 1, interval_phi=0.001),
                # Δn=±1 changes are made by the worm in a dn=0 way.
                supervillain.generator.villain.worm.Geometric(V),
            ))
    v = supervillain.Ensemble(V).generate(args.configurations, g, start='cold', progress=tqdm)
    print(g.report())

with logging_redirect_tqdm():
    g = supervillain.generator.combining.Sequentially((
            supervillain.generator.worldline.PlaquetteUpdate(W),
            supervillain.generator.worldline.WrappingUpdate(W)
        ))
    w = supervillain.Ensemble(W).generate(args.configurations, g, start='cold', progress=tqdm)
    print(g.report())

# A first computation of the autocorrelation time will have effects from thermalization.
v_autocorrelation = v.autocorrelation_time()
w_autocorrelation = w.autocorrelation_time()

# We aggressively cut to ensure thermalization.
v_thermalized = v.cut(10*v_autocorrelation)
w_thermalized = w.cut(10*w_autocorrelation)

# Now we can get a fair computation of the autocorrelation time.
v_autocorrelation = v_thermalized.autocorrelation_time()
w_autocorrelation = w_thermalized.autocorrelation_time()

print(f'Autocorrelation time')
print(f'--------------------')
print(f'Villain   {v_autocorrelation}')
print(f'Worldline {w_autocorrelation}')

v_decorrelated = v_thermalized.every(v_autocorrelation)
w_decorrelated = w_thermalized.every(w_autocorrelation)

# We can easily get a bootstrap object, which we will use for estimates.
v_bootstrap = supervillain.analysis.Bootstrap(v_decorrelated)
w_bootstrap = supervillain.analysis.Bootstrap(w_decorrelated)

# The rest is show business!
fig, ax = comparison_plot.setup(args.observables)
comparison_plot.bootstraps(ax,
        (v_bootstrap, w_bootstrap),
        ('Villain', 'Worldline'),
        observables=args.observables
        )
comparison_plot.histories(ax,
        (v, w),
        ('Villain', 'Worldline'),
        observables=args.observables
        )

fig.suptitle(f'N={args.N} κ={args.kappa} W={args.W}')
fig.tight_layout()

if args.figure:
    fig.savefig(args.figure)
else:
    plt.show()
