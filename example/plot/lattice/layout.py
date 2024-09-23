#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

import supervillain

parser = supervillain.cli.ArgumentParser()
parser.add_argument('--N', type=int, default=5, help='Sites on a side.')
parser.add_argument('--pdf', type=str, default='')
args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)

fig, ax = plt.subplots(1,1,
    figsize=(7, 6),
)

on_off = LinearSegmentedColormap.from_list(
    'binary',
    [(1,1,1), (0,0,0)],
    N=3
)

L = supervillain.Lattice2D(args.N)

site = L.form(0)      # automatically 0 everywhere on creation.
site[0,0] = 1

link = L.form(1)      # automatically 0 everywhere on creation.
link[:, 0, 0] = 1

plaquette = L.form(2) # automatically 0 everywhere on creation.
plaquette[0, 0] = 1

L.plot_form(0, site,      ax, cmap=on_off, norm=Normalize(vmin=-1, vmax=+1))
L.plot_form(1, link,      ax, cmap=on_off, norm=Normalize(vmin=-1, vmax=+1))
L.plot_form(2, plaquette, ax, cmap=on_off, norm=Normalize(vmin=-1, vmax=+1))

ax.set_xlabel('0th direction')
ax.set_ylabel('1st direction')

fig.tight_layout()

if args.pdf:
    fig.savefig(args.pdf)
else:
    plt.show()

