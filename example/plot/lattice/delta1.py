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

L = supervillain.Lattice2D(args.N)

fig, ax = plt.subplots(1,1, figsize=(6, 6))

on_off = LinearSegmentedColormap.from_list(
    'binary',
    [(1,1,1), (0,0,0)],
    N=3
)

plus_zero_minus = LinearSegmentedColormap.from_list(
    'ternary',
    [(0,0,1), (0.5,0.5,0.5), (1,0,0)],
    N=3
)

link = L.form(1)
link[0, -1, -1] = 1
link[1, +1, +1] = 1

divergence = L.delta(1, link)

L.plot_form(1, link,       ax, cmap=on_off, norm=Normalize(vmin=0, vmax=1))
L.plot_form(0, divergence, ax, cmap=plus_zero_minus, norm=Normalize(vmin=-1, vmax=+1))

ax.set_aspect('auto')
fig.tight_layout()

if args.pdf:
    fig.savefig(args.pdf)
else:
    plt.show()

