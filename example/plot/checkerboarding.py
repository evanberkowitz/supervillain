#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain

lattices = (10, 11)
fig, ax = plt.subplots(1,2,
    figsize=(12, 6),
)

for a, N in zip(ax.flatten(), lattices):
    a.set_title(f'{N=}')
    
    L = supervillain.lattice.Lattice2D(N)
    x = L.form(2)
    for i, color in enumerate(L.checkerboarding):
        x[color] = i

    L.plot_form(2, x/np.max(x), a, cmap='nipy_spectral')

plt.show()
