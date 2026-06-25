#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import supervillain
from supervillain.analysis import Uncertain
import supervillain.analysis.comparison_plot as comparison_plot
supervillain.observable.progress=tqdm

parser = supervillain.cli.ArgumentParser(description = 'The goal is to compute the same observables using both the Villain and Worldline actions and to check that they agree.  The Villain action is sampled with a combination of Site and Link Updates and the worm (D=2 only).')
parser.add_argument('--D', type=int, default=2, help='Number of spacetime dimensions.  Defaults to 2.')
parser.add_argument('--N', type=int, default=5, help='Sites on a side.  Defaults to 5.')
parser.add_argument('--kappa', type=float, default=0.5, help='κ.  Defaults to 0.5.')
parser.add_argument('--W', type=supervillain.cli.W, default=1, help='Constraint integer W.  Defaults to 1')
parser.add_argument('--configurations', type=int, default=100000, help='Defaults to 100000.  You need a good deal of configurations with κ=0.5 because of autocorrelations in the Villain sampling.')
parser.add_argument('--figure', default=False, type=str)
parser.add_argument('--observables', nargs='*', help='Names of observables to compare.  Defaults to a list of 3 observables.',
                    default=('ActionDensity', 'InternalEnergyDensity', 'InternalEnergyDensitySquared', ))

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)

# First create the lattices and the two dual actions.
L = supervillain.lattice.Lattice(D=args.D, N=args.N)

V = supervillain.action.Villain(L, args.kappa, W=args.W)
W = supervillain.action.Worldline(L, args.kappa, W=args.W)

# Now sample each action.  Hammer already picks the right generators per dimension
# (it omits the D=2-only worm for D>2) and per W, so we can just call it.
with logging_redirect_tqdm():
    g = supervillain.generator.villain.Hammer(V)
    v = supervillain.Ensemble(V).generate(args.configurations, g, start='cold', progress=tqdm)
    print(g.report())

with logging_redirect_tqdm():
    g = supervillain.generator.worldline.Hammer(W)
    w = supervillain.Ensemble(W).generate(args.configurations, g, start='cold', progress=tqdm)
    print(g.report())

# A first computation of the autocorrelation time will have effects from thermalization.
v_autocorrelation = v.autocorrelation_time(observables=args.observables)
w_autocorrelation = w.autocorrelation_time(observables=args.observables)

# We aggressively cut to ensure thermalization.
v_thermalized = v.cut(10*v_autocorrelation)
w_thermalized = w.cut(10*w_autocorrelation)

# Now we can get a fair computation of the autocorrelation time.
v_autocorrelation = v_thermalized.autocorrelation_time(observables=args.observables)
w_autocorrelation = w_thermalized.autocorrelation_time(observables=args.observables)

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
title = f'D={args.D} W={args.W} κ={args.kappa} N={args.N}'

# Figure 1: the observable comparison.
fig_obs, ax_obs = comparison_plot.setup(args.observables)
comparison_plot.bootstraps(ax_obs,
        (v_bootstrap, w_bootstrap),
        ('Villain', 'Worldline'),
        observables=args.observables
        )
comparison_plot.histories(ax_obs,
        (v, w),
        ('Villain', 'Worldline'),
        observables=args.observables
        )
fig_obs.suptitle(title)
fig_obs.tight_layout()

# Figure 2: one panel per correlator, stacked in a column sharing the Δx axis.
# Spin_Spin_Normalized is the derived quantity Spin_Spin / Spin_Spin[origin].
correlators = (
    ('Spin_Spin_Normalized', 'log'),
    ('Winding_Winding',      'linear'),
)
fig_corr, ax_corr = plt.subplots(
        nrows=len(correlators), ncols=1,
        sharex=True, squeeze=False, figsize=(6, 3 * len(correlators)),
)
ax_corr = ax_corr[:, 0]

for ax, (correlator, yscale) in zip(ax_corr, correlators):
    v_bootstrap.plot_correlator(ax, correlator, label='Villain')
    w_bootstrap.plot_correlator(ax, correlator, label='Worldline')
    ax.set_yscale(yscale)
    ax.set_ylabel(correlator)

# The shared x-axis is log; only the bottom panel keeps its tick labels and title.
ax_corr[0].legend()
ax_corr[-1].set_xscale('log')
for ax in ax_corr[:-1]:
    ax.set_xlabel('')
ax_corr[-1].set_xlabel('Δx')
fig_corr.suptitle(title)
fig_corr.tight_layout()

# Save both figures into a single multi-page PDF, or show them together.
if args.figure:
    with PdfPages(args.figure) as pdf:
        pdf.savefig(fig_obs)
        pdf.savefig(fig_corr)
else:
    plt.show()