#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain

parser = supervillain.cli.ArgumentParser()
parser.add_argument('--N', type=int, default=5, help='Sites on a side.')
parser.add_argument('--kappa', type=float, default=0.1, help='κ.')
parser.add_argument('--configurations', type=int, default=1000)
parser.add_argument('--cut', type=int, default=None, help='Thermalization time.  Defaults to 10*the autocorrelation time.')
parser.add_argument('--stride', type=int, default=None, help='Stride for decorrelation.  Defaults to 2*the autocorrelation time.')
parser.add_argument('--figure', default=False, type=str)
parser.add_argument('--h5', default=False, type=str)

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)

L = supervillain.Lattice2D(args.N)
S = supervillain.action.Worldline(L, args.kappa)
#G = supervillain.generator.NeighborhoodUpdate(S)
p = supervillain.generator.constraint.PlaquetteUpdate(S)
h = supervillain.generator.constraint.HolonomyUpdate(S)
G = supervillain.generator.combining.Sequentially((p, h))


with logging_redirect_tqdm():
    e = supervillain.Ensemble(S).generate(args.configurations, G, start='cold', progress=tqdm)

if args.h5:
    with h5.File(args.h5, 'w') as h:
        e.to_h5(h.create_group('/ensemble'))

def naive_estimate(observable):
    # (mean, uncertainty)
    return (np.mean(observable), np.std(observable) / np.sqrt(len(observable)))

def boot(observable):
    # (mean, uncertainty) if the observable is bootstrapped.
    return (np.mean(observable), np.std(observable))

def error_format(estimate):
    mean = estimate[0]
    err  = estimate[1]
    return f'{mean:.5f} ± {err:.5f}'

def band(ax, estimate, color=None):
    mean = estimate[0]
    err  = estimate[1]
    if color is None:
        color = ax.get_lines()[-1].get_color()
    ax.axhspan(mean-err, mean+err, color=color, alpha=0.5, linestyle='none')

def plot_history(
        axs, index, data,
        label=None,
        bins=31, density=True,
        alpha=0.5, color=None
        ):
    axs[0].plot(index, data, color=color)
    axs[1].hist(data, label=label,
            orientation='horizontal',
            bins=bins, density=density,
            color=color, alpha=alpha,)

fig, ax = plt.subplots(2,2,
    figsize=(10, 6),
    gridspec_kw={'width_ratios': [4, 1], 'wspace': 0},
    sharey='row'
)

fig.suptitle(f'{S}', fontsize=16)

# Plot the whole history
plot_history(ax[0], e.index, e.InternalEnergyDensity  )
plot_history(ax[1], e.index, e.WindingSquared, bins=101)

autocorrelation = max([supervillain.analysis.autocorrelation_time(o) for o in (e.InternalEnergyDensity, e.WindingSquared)])

# Now let's cut and decorrelate
print(f'Autocorrelation time = {autocorrelation}')
e = e.cut(10*autocorrelation if args.cut is None else args.cut).every(2*autocorrelation if args.stride is None else args.stride)
b = supervillain.analysis.Bootstrap(e, len(e))

# Since we have bootstrapped we can make bootstrap uncertainty estimates.
estimate = boot

# in order to make uncertainty estimates
s   = estimate(b.InternalEnergyDensity)
dn2 = estimate(b.WindingSquared)

print(f'Internal Energy Density     {error_format(s)}')
print(f'Winding Squared             {error_format(dn2)}')

plot_history(ax[0], e.index, e.InternalEnergyDensity,   label=error_format(s))
plot_history(ax[1], e.index, e.WindingSquared,          label=error_format(dn2), bins=101)

band(ax[0,0], s  )
band(ax[1,0], dn2)

ax[0,0].set_ylabel('U / Λ')
ax[1,0].set_ylabel('w^2')
ax[-1,0].set_xlabel('Monte Carlo time')

for a in ax[:,1]:
    a.legend()

fig.tight_layout()

if args.figure:
    plt.savefig(args.figure)
else:
    plt.show()
