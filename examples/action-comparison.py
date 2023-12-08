#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain

parser = supervillain.cli.ArgumentParser(description = 'The goal is to compute the same observables using both the Villain and Worldline actions and to check that they agree.')
parser.add_argument('--N', type=int, default=5, help='Sites on a side.')
parser.add_argument('--kappa', type=float, default=0.5, help='κ.  Defaults to 0.5.')
parser.add_argument('--configurations', type=int, default=10000, help='Defaults to 10000.  You need a good deal of configurations with κ=0.5 because of autocorrelations in the Villain sampling.')
parser.add_argument('--figure', default=False, type=str)

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)

# First create the lattices and the two dual actions.
L = supervillain.lattice.Lattice2D(args.N)

V = supervillain.action.Villain(L, args.kappa)
W = supervillain.action.Worldline(L, args.kappa)

# Now sample each action.
with logging_redirect_tqdm():
    g = supervillain.generator.NeighborhoodUpdate(V)
    v = supervillain.Ensemble(V).generate(args.configurations, g, start='cold', progress=tqdm)

with logging_redirect_tqdm():
    g = supervillain.generator.combining.Sequentially((
            supervillain.generator.worldline.PlaquetteUpdate(W),
            supervillain.generator.worldline.HolonomyUpdate(W)
        ))
    w = supervillain.Ensemble(W).generate(args.configurations, g, start='cold', progress=tqdm)

# Some bare-bones statistical estimators from the bootstrap
def boot(observable):
    # (mean, uncertainty) if the observable is bootstrapped.
    return (np.mean(observable), np.std(observable))

# Since we have bootstrapped we can make bootstrap uncertainty estimates.
estimate = boot

def error_format(estimate):
    mean = estimate[0]
    err  = estimate[1]
    return f'{mean:.3f} ± {err:.3f}'

def band(ax, estimate, color=None):
    mean = estimate[0]
    err  = estimate[1]
    if color is None:
        color = ax.get_lines()[-1].get_color()
    ax.axhspan(mean-err, mean+err, color=color, alpha=0.5, linestyle='none')


# The rest is show business!
def plot_history(
        axs, index, data,
        label=None,
        bins=31, density=True,
        alpha=0.5, color=None
        ):
    axs[0].plot(index, data, color=color, alpha=alpha)
    axs[1].hist(data, label=label,
            orientation='horizontal',
            bins=bins, density=density,
            color=color, alpha=alpha,)

# Until the winding number is implemented for the Worldline formulation, just show 1 history.
fig, ax = plt.subplots(1,2,
    figsize=(10, 3),
    gridspec_kw={'width_ratios': [4, 1], 'wspace': 0},
    sharey='row',
    squeeze=False
)

# Plot the whole history
plot_history(ax[0], v.index, v.InternalEnergyDensity, label='Villain')
#plot_history(ax[1], v.index, v.WindingSquared, bins=101)

plot_history(ax[0], w.index, w.InternalEnergyDensity, label='Worldline')
#plot_history(ax[1], w.index, w.WindingSquared, bins=101)


# Now let's cut and decorrelate
v_autocorrelation = max([supervillain.analysis.autocorrelation_time(o) for o in (v.InternalEnergyDensity, v.WindingSquared)])
w_autocorrelation = max([supervillain.analysis.autocorrelation_time(o) for o in (w.InternalEnergyDensity,
                                                                                 # Holonomies:
                                                                                 w.configurations.m[:,0].sum(axis=2).sum(axis=1),
                                                                                 w.configurations.m[:,1].sum(axis=2).sum(axis=1),
                                                                                 )])
print(f'Autocorrelation time')
print(f'--------------------')
print(f'Villain   {v_autocorrelation}')
print(f'Worldline {w_autocorrelation}')

v = v.cut(10*v_autocorrelation).every(5*v_autocorrelation)
w = w.cut(10*w_autocorrelation).every(5*w_autocorrelation)

for a in ax[:, 0]:
    a.axvline(10*v_autocorrelation, color='black')
    a.axvline(10*w_autocorrelation, color='black')

b = supervillain.analysis.Bootstrap(v, len(v))
c = supervillain.analysis.Bootstrap(w, len(w))

# Plot the whole history
plot_history(ax[0], v.index, v.InternalEnergyDensity, label=f'Villain {error_format(estimate(b.InternalEnergyDensity))}')
#plot_history(ax[1], v.index, v.WindingSquared, bins=101, label=f'Villain {error_format(estimate(b.WindingSquared))}')

plot_history(ax[0], w.index, w.InternalEnergyDensity, label=f'Worldline {error_format(estimate(c.InternalEnergyDensity))}')
#plot_history(ax[1], w.index, w.WindingSquared, bins=101, label=f'Worldline {error_format(estimate(c.WindingSquared))}')

ax[0,0].set_ylabel('U / Λ')
#ax[1,0].set_ylabel('w^2')
ax[-1,0].set_xlabel('Monte Carlo time')

for a in ax[:,1]:
    a.legend()

fig.tight_layout()

plt.show()
