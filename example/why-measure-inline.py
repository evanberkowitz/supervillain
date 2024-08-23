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
Why measure inline?  What is the advantage?  Here we give an example of measuring the Vortex_Vortex correlator.

In the Worldline formulation we have access to v, and can measure it directly.
In the Villain formulation we can either
 (1) measure during the evolution of the worm, or
 (2) measure using a post-generation taxicab observable.

The expected result is that the worm measurement comes out much closer to the worldline measurement than the taxicab observable,
especially in the long-distance tails.  To see this effect clearly we need a great deal of configurations.
    ''')
parser.add_argument('--N', type=int, default=11, help='Sites on a side.  Defaults to 11.')
parser.add_argument('--W', type=supervillain.cli.W, default=2, help='Winding constraint.  Defaults to 2.')
parser.add_argument('--kappa', type=float, default=0.12025, help='κ.  Defaults to 0.12025.')
parser.add_argument('--configurations', type=int, default=100000, help='Defaults to 100000; a large number is required to see a pronounced difference.')
parser.add_argument('--figure', default=False, type=str)

args = parser.parse_args()

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
        if S.W == 1:
            G = supervillain.generator.combining.Sequentially((
                    supervillain.generator.villain.SiteUpdate(S),
                    supervillain.generator.villain.LinkUpdate(S),
                    supervillain.generator.villain.ExactUpdate(S),
                    supervillain.generator.villain.HolonomyUpdate(S),
                ))
        else:
            G = supervillain.generator.combining.Sequentially((
                    supervillain.generator.villain.SiteUpdate(S),
                    supervillain.generator.villain.LinkUpdate(S),
                    supervillain.generator.villain.ExactUpdate(S),
                    supervillain.generator.villain.HolonomyUpdate(S),
                    # Can also make Δn=±1 changes by the worm in a dn=0 way.
                    supervillain.generator.villain.worm.Geometric(S),
                ))
    
    elif isinstance(S, supervillain.action.Worldline):
        G = supervillain.generator.combining.Sequentially((
            supervillain.generator.worldline.VortexUpdate(S),
            supervillain.generator.worldline.CoexactUpdate(S),
            supervillain.generator.worldline.WrappingUpdate(S, 1),
            supervillain.generator.worldline.worm.Geometric(S),
            ))

    e[a] = supervillain.Ensemble(S).generate(args.configurations, G, start='cold', progress=tqdm)
    print(G.report())

    try:
        tau = e[a].autocorrelation_time()
    except:
        tau = 10
    thermalized = e[a].cut(5*tau)
    try:
        tau = thermalized.autocorrelation_time()
    except:
        tau = 50
    b[a] = supervillain.analysis.Bootstrap(thermalized.every(tau))

fig, ax = plt.subplots(1,1, figsize=(12,8))

b['villain'].Ensemble.configuration.fields['Vortex_Vortex_Worm'] = b['villain'].Ensemble.configuration.fields['Vortex_Vortex']
del b['villain'].Ensemble.configuration.fields['Vortex_Vortex']

b['villain'].Vortex_Vortex = L.irrep(b['villain'].Vortex_Vortex)
b['villain'].plot_correlator(ax, 'Vortex_Vortex', offset=0.00, label='Villain taxicab')

b['villain'].V_V = L.irrep(b['villain'].Vortex_Vortex_Worm / b['villain'].Vortex_Vortex_Worm[:,0,0][:,None,None])
b['villain'].plot_correlator(ax, 'V_V', offset=0.05, label='Villain worm inline')

b['worldline'].plot_correlator(ax, 'Vortex_Vortex', offset=0.10, label='Worldline')
ax.legend()
ax.set_ylabel('Vortex_Vortex')
fig.suptitle(f'L={L.nx} κ={args.kappa} W={args.W}')
fig.tight_layout()

if args.figure:
    fig.savefig(args.figure)
else:
    plt.show()
