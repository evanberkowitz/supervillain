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
Why measure inline?  What is the advantage?  Here we give an example of measuring the Spin_Spin and Vortex_Vortex correlators.

In the Villain formulation we have access to phi, and can measure the spin correlator directly.
In the Worldline formulation we can either
 (1) measure during the evolution of the worm, or
 (2) measure using a post-generation taxicab observable.

In the Worldline formulation we have access to v, and can measure the vortex correlator directly.
In the Villain formulation we can either
 (1) measure during the evolution of the worm, or
 (2) measure using a post-generation taxicab observable.

The expected result is that the worm measurement comes out much closer the taxicab observable,
especially in the long-distance tails.  To see this effect clearly we need a great deal of configurations.
    ''')
parser.add_argument('--N', type=int, default=9, help='Sites on a side.  Defaults to 9.')
parser.add_argument('--W', type=supervillain.cli.W, default=2, help='Winding constraint.  Defaults to 2.')
parser.add_argument('--kappa', type=float, default=0.10, help='κ.  Defaults to 0.10.')
parser.add_argument('--configurations', type=int, default=100000, help='Defaults to 100000; a large number is required to see a pronounced difference.')
parser.add_argument('--figure', default=False, type=str)

args = parser.parse_args(
#[  # Any of these lines are decent examples:
    #'--N', '9', '--W', '1', '--kappa', '0.7', '--configurations', '40000',
    #'--N', '11',  '--W', '2', '--kappa', '0.15', '--configurations', '40000',
    #'--N', '21', '--W', '3', '--kappa', f'{0.8*0.74/3**2:3.3f}', '--configurations', '400000',
#]
)

import logging
logger = logging.getLogger(__name__)

# First create the lattices and the action.
L = supervillain.lattice.Lattice2D(args.N)

e = dict()
b = dict()

for (a, A) in (
    ('villain', supervillain.action.Villain),
    ('worldline', supervillain.action.Worldline),
):
    S = A(L, args.kappa, args.W)
    
    if isinstance(S, supervillain.action.Villain):
        G = supervillain.generator.villain.Hammer(S, worms=L.plaquettes)
    elif isinstance(S, supervillain.action.Worldline):
        G = supervillain.generator.worldline.Hammer(S, worms=L.sites)

    e[a] = supervillain.Ensemble(S).generate(args.configurations, G, start='cold', progress=tqdm)
    print(G.report())

    # We want to compare the built-in observables to the inline ones.
    # Therefore we need to move the inline ones to a different name, allowing the built-ins to be measured.
    if a == 'villain':
        e[a].Vortex_Vortex_Worm = e[a].configuration.fields['Vortex_Vortex']
        del e[a].configuration.fields['Vortex_Vortex']
    elif a == 'worldline':
        e[a].Spin_Spin_Worm = e[a].configuration.fields['Spin_Spin']
        del e[a].configuration.fields['Spin_Spin']

    try:
        tau = e[a].autocorrelation_time()
    except:
        tau = 10
    thermalized = e[a].cut(5*tau)

    # Fields added manually as above are not automatically groked when post-processed.
    # We cut them ourselves and add them to the thermalized ensemble, to be blocked and bootstrapped below.
    if a == 'villain':
        thermalized.Vortex_Vortex_Worm = e[a].Vortex_Vortex_Worm[5*tau:]
    elif a == 'worldline':
        thermalized.Spin_Spin_Worm = e[a].Spin_Spin_Worm[5*tau:]
    
    try:
        tau = thermalized.autocorrelation_time()
    except:
        tau = 50
    b[a] = supervillain.analysis.Bootstrap(
        supervillain.analysis.Blocking(thermalized, width=tau))


# Showtime!

fig, ax = plt.subplots(2,1, figsize=(12,12), sharex='col')

b['worldline'].Vortex_Vortex = L.irrep(b['worldline'].Vortex_Vortex)
b['worldline'].plot_correlator(ax[0], 'Vortex_Vortex', offset=0.0, label='Worldline')

b['villain'].V_V = L.irrep(b['villain'].Vortex_Vortex_Worm / b['villain'].Vortex_Vortex_Worm[:,0,0][:,None,None])
b['villain'].plot_correlator(ax[0], 'V_V', offset=0.05, label='Villain worm inline')

b['villain'].Vortex_Vortex = L.irrep(b['villain'].Vortex_Vortex)
b['villain'].plot_correlator(ax[0], 'Vortex_Vortex', offset=0.00, label='Villain taxicab')

ax[0].legend()
ax[0].set_ylabel('Vortex_Vortex')

b['villain'].Spin_Spin = L.irrep(b['villain'].Spin_Spin)
b['villain'].plot_correlator(ax[1], 'Spin_Spin', offset=0.00, label='Villain')

b['worldline'].S_S = L.irrep(b['worldline'].Spin_Spin_Worm / b['worldline'].Spin_Spin_Worm[:,0,0][:,None,None])
b['worldline'].plot_correlator(ax[1], 'S_S', offset=0.05, label='Worldline worm inline', multiplier=1)#/1.4)

b['worldline'].Spin_Spin = L.irrep(b['worldline'].Spin_Spin)
b['worldline'].plot_correlator(ax[1], 'Spin_Spin', offset=0.10, label='Worldline taxicab')

ax[1].axhline(0, color='black')

ax[1].legend()
ax[1].set_ylabel('Spin_Spin')

# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax.set_xlim((8,15))
ax[0].set_ylim((0, 1.1))
# ax[0].axhline(0, color='black')

# ax[1].set_xscale('log')
ax[1].set_yscale('log')
#ax[1].set_ylim((0, 1.1))

fig.suptitle(f'L={L.nx} κ={args.kappa} W={args.W}')
fig.tight_layout()


if args.figure:
    plt.save(args.figure)
else:
    plt.show()
