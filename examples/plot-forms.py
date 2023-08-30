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

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)

L = supervillain.Lattice2D(args.N)
S = supervillain.Villain(L, args.kappa)
G = supervillain.generator.NeighborhoodUpdate(S)
with logging_redirect_tqdm():
    e = supervillain.Ensemble(S).generate(args.configurations, G, start='cold', progress=tqdm)

fig, ax = plt.subplots(1,1,
    figsize=(6, 6),
)

fig.suptitle(f'{S}', fontsize=16)

cfg = e.configurations[-1]
phi = cfg['phi']
n   = cfg['n']

links = (L.d(0, phi) - 2*np.pi*n)
winding = L.d(1, n)

L.plot_form(0, phi,  ax)
L.plot_form(1, links,   ax)
L.plot_form(2, winding, ax)

fig.tight_layout()
plt.show()

