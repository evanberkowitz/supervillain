#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain
from supervillain.analysis import Uncertain

parser = supervillain.cli.ArgumentParser()
parser.add_argument('--N', type=int, default=5, help='Sites on a side.')
parser.add_argument('--kappa', type=float, default=0.5, help='κ.')
parser.add_argument('--W', type=supervillain.cli.W, default=1, help='The constraint integer W.')
parser.add_argument('--configurations', type=int, default=10000)
parser.add_argument('--action', type=str, default='villain', choices=['villain', 'worldline'])
parser.add_argument('--bootstraps', default=100, type=int, help='Number of bootstrap resamplings.')

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)


####
#### Observables
####

# Each observable whose history and histogram you want to see can be put into this list.
observables = tuple(o for o, c in supervillain.observables.items()
                    if  issubclass(c, supervillain.observable.Scalar)
                    and (args.W != 1 or 'Vortex' not in o) # Vortex observabes are trivial when W=1.
                    and ('Scaled' not in o) # The scaled observables are directly proportional to the non-scaled ones, a waste to show.
                    )

####
#### Monte Carlo generation
####

L = supervillain.Lattice2D(args.N)

if args.action == 'villain':
    S = supervillain.action.Villain(L, args.kappa, args.W)
    G = supervillain.generator.villain.NeighborhoodUpdate(S)
elif args.action == 'worldline':
    S = supervillain.action.Worldline(L, args.kappa, args.W)
    p = supervillain.generator.worldline.PlaquetteUpdate(S)
    h = supervillain.generator.worldline.WrappingUpdate(S)
    G = supervillain.generator.combining.Sequentially((p, h))

with logging_redirect_tqdm():
    E = supervillain.Ensemble(S).generate(args.configurations, G, start='cold', progress=tqdm)

####
#### Analysis + Visualization
####

# First we'll construct a figure with one row per scalar observable, with room for the history and histogram.
histories, ax = plt.subplots(len(observables),2,
    figsize=(12, 2*len(observables)),
    gridspec_kw={'width_ratios': [4, 1], 'wspace': 0},
    sharey='row'
)

histories.suptitle(f'{S}', fontsize=16)

# To get good error estimates we should make a thermalization cut and then decorrelate.
# We calculate the autocorrelation time for each observable.
E.measure()
autocorrelation = E.autocorrelation_time()

# Now let's cut and decorrelate
print(f'Autocorrelation time = {autocorrelation}')
e = E.cut(10*autocorrelation).every(2*autocorrelation)
print(f'After decorrelation    {e.autocorrelation_time()}')
bootstrap = supervillain.analysis.Bootstrap(e, args.bootstraps)

for a, O in zip(ax, observables):
    a[0].set_ylabel(O)
    # We can plot the raw history,
    E.plot_history(a, O)
    # estimate the central value from the bootstrap of the decorrelated ensemble,
    o = bootstrap.estimate(O)
    print(f'{O:40s} {Uncertain(*o)}')
    # and plot the decorrelated ensemble with the estimate.
    e.plot_history(a, O, label=Uncertain(*o))
    bootstrap.plot_band(a[0], O)
    a[1].legend()

ax[-1,0].set_xlabel('Monte Carlo time')
histories.tight_layout()

plt.show()
