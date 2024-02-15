#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain
from supervillain.analysis import Uncertain
supervillain.observable.progress=tqdm

parser = supervillain.cli.ArgumentParser(description = '''
    The goal is to compute the same observables using the Villain action with and without the worm update and to check that the worm does not change any observable meaningfully.
    Since without the worm we have no W>1 algorithm, we restrict our attention to W=1.
    ''')
parser.add_argument('--N', type=int, default=5, help='Sites on a side.  Defaults to 5.')
parser.add_argument('--kappa', type=float, default=0.25, help='κ.  Defaults to 0.25.')
parser.add_argument('--configurations', type=int, default=100000, help='Defaults to 100000.  You need a good deal of configurations with κ=0.5 because of autocorrelations in the Villain sampling.')
parser.add_argument('--figure', default=False, type=str)
parser.add_argument('--observables', nargs='*', help='Names of observables to compare.  Defaults to a list of 7 observables.',
                    default=('ActionDensity',
                             'InternalEnergyDensity', 'InternalEnergyDensitySquared',
                             'SpinSusceptibility',
                             'WindingSquared',
                             'TWrapping', 'XWrapping',
                             ))

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)

# First create the lattices and the action.
L = supervillain.lattice.Lattice2D(args.N)
S = supervillain.action.Villain(L, args.kappa, W=1)

with logging_redirect_tqdm():
    g = supervillain.generator.villain.NeighborhoodUpdate(S)
    n = supervillain.Ensemble(S).generate(
            args.configurations,
            g,
            start='cold',
            progress=tqdm)

    print(g.report())

    G = supervillain.generator.combining.Sequentially((
            supervillain.generator.villain.NeighborhoodUpdate(S),
            supervillain.generator.villain.worm.Geometric(S)
    ))
    w = supervillain.Ensemble(S).generate(
            args.configurations,
            G,
            progress=tqdm)

    print(G.report())


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
print(f'Local Updates       {n_autocorrelation}')
print(f'With Villain Worm   {w_autocorrelation}')

n_decorrelated = n_thermalized.every(n_autocorrelation)
w_decorrelated = w_thermalized.every(w_autocorrelation)

# We can easily get a bootstrap object, which we will use for estimates.
n_bootstrap = supervillain.analysis.Bootstrap(n_decorrelated)
w_bootstrap = supervillain.analysis.Bootstrap(w_decorrelated)

# The rest is show business!

fig, ax = plt.subplots(len(args.observables), 3,
    figsize=(12, 2.5*len(args.observables)),
    gridspec_kw={'width_ratios': [4, 1, 1], 'wspace': 0, 'hspace': 0},
    sharey='row',
    squeeze=False
)

fig.suptitle(f'Villain N={args.N} κ={args.kappa} W=1')

for a, o in zip(ax, args.observables):
    # The worm tends to be much more decorrelated, so plot it behind the Villain for visual clarity.

    n_decorrelated.plot_history(a, o, label='No worm', alpha=0.5, history_kwargs={'zorder': 1})#, histogram_label=f'No worm {Uncertain(*n_bootstrap.estimate(o))}')
    n_bootstrap.plot_band(a[0], o)

    w_decorrelated.plot_history(a, o, label='+Worm', alpha=0.5, history_kwargs={'zorder': 1})#, histogram_label=f'+Worm {Uncertain(*w_bootstrap.estimate(o))}')
    w_bootstrap.plot_band(a[0], o)

    tau = supervillain.analysis.autocorrelation_time(getattr(n_thermalized, o))
    n.plot_history(a, o, alpha=0.5, history_kwargs={'zorder': -1, 'label': f'No worm τ={tau}',})

    tau = supervillain.analysis.autocorrelation_time(getattr(w_thermalized, o))
    w.plot_history(a, o, alpha=0.5, history_kwargs={'zorder': -1, 'label': f'+Worm τ={tau}'})

    a[2].hist((
            getattr(n_bootstrap, o),
            getattr(w_bootstrap, o),
        ),
        density=True,
        orientation='horizontal', alpha=0.5, bins=25,
        label=tuple(f'{name} {Uncertain(*boot.estimate(o))}' for name, boot in zip(('No worm', '+Worm'), (n_bootstrap, w_bootstrap,)))
    )


    a[0].set_ylabel(o)
    a[0].legend()
    a[2].legend()

ax[-1,0].set_xlabel('Monte Carlo time')
ax[-1,1].set_xticks([])
ax[-1,1].set_xlabel('Measurements')
ax[-1,2].set_xlabel('Bootstraps')

fig.tight_layout()

if args.figure:
    fig.savefig(args.figure)
else:
    plt.show()
