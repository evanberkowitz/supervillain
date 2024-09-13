#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain
import supervillain.analysis.comparison_plot as comparison_plot

parser = supervillain.cli.ArgumentParser(description='Generate two Villain ensembles the same way and compare their results.')
parser.add_argument('--N', type=int, default=5, help='Sites on a side.')
parser.add_argument('--kappa', type=float, default=0.1, help='κ.')
parser.add_argument('--configurations', type=int, default=10000)

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)

L = supervillain.Lattice2D(args.N)
S = supervillain.Villain(L, args.kappa)
G = supervillain.generator.combining.Sequentially((
        supervillain.generator.villain.SiteUpdate(S),
        supervillain.generator.villain.LinkUpdate(S),
    ))
with logging_redirect_tqdm():
    A = supervillain.Ensemble(S).generate(args.configurations, G, start='cold', progress=tqdm)
    B = supervillain.Ensemble(S).generate(args.configurations, G, start='cold', progress=tqdm)

ensembles = (A, B)
taus = tuple(e.autocorrelation_time() for e in ensembles)

thermalized = tuple(e.cut(10*tau) for e, tau in zip(ensembles, taus))
taus = tuple(e.autocorrelation_time() for e in thermalized)

decorrelated = tuple(e.every(tau) for e, tau in zip(thermalized, taus))
bootstraps   = tuple(supervillain.analysis.Bootstrap(e) for e in decorrelated)

fig, ax = comparison_plot.setup()
comparison_plot.bootstraps(ax,
        bootstraps,
        ('First run', 'Second run'),
        )
comparison_plot.histories(ax,
        ensembles,
        ('First run', 'Second run'),
        )

fig.suptitle(f'Villain N={args.N} κ={args.kappa} W=1')
fig.tight_layout()
plt.show()
