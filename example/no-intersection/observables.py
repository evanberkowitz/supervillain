#!/usr/bin/env python

r"""
Sample the No-Intersection model with its :func:`~.no_intersection.Hammer` and measure
observables --- the single-action analogue of ``example/action-comparison.py``.

Because ``NoIntersections`` is a ``Villain`` action (with the vortex-sheet self-intersection
constrained to vanish), the observable machinery now dispatches its Villain implementations
to it, so the usual thermodynamic observables are available.  Two things are special to this
model:

* ``TopologicalChargeDensitySquared`` is **identically zero** --- the constraint
  $q = dn\wedge dn = 0$ --- and is reported as a sanity check;
* ``Intersection_Intersection`` (the two-point function of the operator that inserts a unit
  of self-intersection) is filled in inline by the worm, and its normalized correlator is
  plotted alongside the compact-boson ``Spin_Spin_Normalized``.

Run from example/no-intersection/:

    uv run python observables.py [--N 6] [--kappa 0.1] [--configurations 5000] [--figure out.pdf]
"""

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import supervillain
from supervillain.analysis import Bootstrap, Uncertain
import supervillain.analysis.comparison_plot as comparison_plot
import supervillain.generator.no_intersection as gen
supervillain.observable.progress = tqdm

parser = supervillain.cli.ArgumentParser(
    description='Sample the No-Intersection Hammer and measure its observables.')
parser.add_argument('--N', type=int, default=6, help='Sites on a side (D = 4 is fixed).  Defaults to 6.')
parser.add_argument('--kappa', type=float, default=0.1, help='κ.  Defaults to 0.1.')
parser.add_argument('--configurations', type=int, default=5000, help='Defaults to 5000.')
parser.add_argument('--observables', nargs='*',
                    default=('ActionDensity', 'InternalEnergyDensity', 'InternalEnergyDensitySquared'),
                    help='Scalar observables to estimate and plot.')
parser.add_argument('--figure', default=False, type=str, help='Path to a PDF; otherwise the figures are shown.')
args = parser.parse_args()

L = supervillain.lattice.Lattice(4, args.N)
S = supervillain.action.NoIntersections(L, kappa=args.kappa)
H = gen.Hammer(S)

with logging_redirect_tqdm():
    e = supervillain.Ensemble(S).generate(args.configurations, H, start='cold', progress=tqdm)
print(H.report())

# Thermalize and decorrelate, exactly as action-comparison.py does: a first autocorrelation
# time (contaminated by thermalization) sets an aggressive cut, then a fair one sets the
# decorrelation stride.
auto = e.autocorrelation_time(observables=args.observables)
thermalized = e.cut(10 * auto)
auto = thermalized.autocorrelation_time(observables=args.observables)
decorrelated = thermalized.every(auto)
b = Bootstrap(decorrelated)

print(f'\nautocorrelation time: {auto}    decorrelated configurations: {len(decorrelated)}')

print(f'\nObservable estimates')
print(f'--------------------')
for o in args.observables:
    samples = np.asarray(getattr(b, o))
    print(f'{o:32s} {Uncertain(float(samples.mean()), float(samples.std()))}')

# The constraint q = dn∧dn = 0 makes the topological charge density vanish identically;
# report it (Uncertain cannot format an exact zero) as a machine-checkable sanity test.
q2 = np.asarray(e.TopologicalChargeDensitySquared)
print(f'\nconstraint check: <q^2> = {q2.mean():.3e} (identically 0)   '
      f'max |q^2| = {np.abs(q2).max():.3e}')

# ---------------------------------------------------------------- figures
title = f'NoIntersections  D=4  N={args.N}  κ={args.kappa}  ({len(decorrelated)} decorrelated)'

# Figure 1: scalar observables --- bootstrap estimates over the Monte Carlo histories.
fig_obs, ax_obs = comparison_plot.setup(args.observables)
comparison_plot.bootstraps(ax_obs, (b,), ('NoIntersections',), observables=args.observables)
comparison_plot.histories(ax_obs, (e,), ('NoIntersections',), observables=args.observables)
fig_obs.suptitle(title)
fig_obs.tight_layout()

# Figure 2: the correlators --- the compact-boson Spin_Spin and the model's signature
# self-intersection Intersection_Intersection, both normalized to 1 at the origin.
correlators = (
    ('Spin_Spin_Normalized', 'log'),
    ('Intersection_Intersection_Normalized', 'log'),
)
fig_corr, ax_corr = plt.subplots(nrows=len(correlators), ncols=1, sharex=True,
                                 squeeze=False, figsize=(6, 3 * len(correlators)))
ax_corr = ax_corr[:, 0]
for ax, (correlator, yscale) in zip(ax_corr, correlators):
    b.plot_correlator(ax, correlator, label='NoIntersections')
    ax.set_yscale(yscale)
    ax.set_ylabel(correlator)
ax_corr[0].legend()
ax_corr[-1].set_xlabel('Δx')
fig_corr.suptitle(title)
fig_corr.tight_layout()

if args.figure:
    with PdfPages(args.figure) as pdf:
        pdf.savefig(fig_obs)
        pdf.savefig(fig_corr)
    print(f'\nfigures written to {args.figure}')
else:
    plt.show()
