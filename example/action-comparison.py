#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain
from supervillain.analysis import Uncertain

parser = supervillain.cli.ArgumentParser(description = 'The goal is to compute the same observables using both the Villain and Worldline actions and to check that they agree.')
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

V = supervillain.action.Villain(L, args.kappa)
W = supervillain.action.Worldline(L, args.kappa)

# Now sample each action.
with logging_redirect_tqdm():
    g = supervillain.generator.villain.NeighborhoodUpdate(V)
    v = supervillain.Ensemble(V).generate(args.configurations, g, start='cold', progress=tqdm)
    v.measure()

with logging_redirect_tqdm():
    g = supervillain.generator.combining.Sequentially((
            supervillain.generator.worldline.PlaquetteUpdate(W),
            supervillain.generator.worldline.WrappingUpdate(W)
        ))
    w = supervillain.Ensemble(W).generate(args.configurations, g, start='cold', progress=tqdm)
    w.measure()

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

fig, ax = plt.subplots(len(args.observables), 2,
    figsize=(10, 2.5*len(args.observables)),
    gridspec_kw={'width_ratios': [4, 1], 'wspace': 0, 'hspace': 0},
    sharey='row',
    squeeze=False
)

fig.suptitle(f'N={args.N} κ={args.kappa} W={args.W}')

for a, o in zip(ax, args.observables):
    # The worldline tends to be much more decorrelated, so plot it behind the Villain for visual clarity.

    w.plot_history(a, o, label='Worldline', alpha=0.5)
    w_decorrelated.plot_history(a, o, label='Worldline decorrelated', alpha=0.5, histogram_label=f'Worldline {Uncertain(*w_bootstrap.estimate(o))}')
    w_bootstrap.plot_band(a[0], o)

    v.plot_history(a, o, label='Villain')
    v_decorrelated.plot_history(a, o, label='Villain decorrelated', histogram_label=f'Villain {Uncertain(*v_bootstrap.estimate(o))}')
    v_bootstrap.plot_band(a[0], o)

    a[0].set_ylabel(o)
    a[1].legend()

ax[0,0].legend()
ax[-1,0].set_xlabel('Monte Carlo time')
ax[-1,1].set_xticks([])
ax[-1,1].set_xlabel('Density')

fig.tight_layout()

if args.figure:
    fig.savefig(args.figure)
else:
    plt.show()