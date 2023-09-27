#!/usr/bin/env python

import numpy as np
import supervillain
import matplotlib.pyplot as plt

L = supervillain.lattice.Lattice2D(11)
c = np.random.random(L.form(0).shape)

fig, ax = plt.subplots(4, 4, figsize=(12, 12))
L.plot_form(0, c, ax[0,0])
ax[0,0].set_title('Correlator(Δx)')

[a.remove() for a in ax[0,1:]]

for a, irrep in zip(ax[1], ('A1', 'A2', 'B1', 'B2')):
    L.plot_form(0, L.irrep(c, irrep), a)
    a.set_title(irrep)

for re, im, irrep in zip(ax[2], ax[3], (("E", +1), ("E", -1), ("E'", +1), ("E'", -1))):
    L.plot_form(0, L.irrep(c, irrep).real, re)
    L.plot_form(0, L.irrep(c, irrep).imag, im)
    re.set_title(f'real{irrep}')
    im.set_title(f'imag{irrep}')

for a in ax.flatten():
    a.set_xlabel(None)
    a.set_ylabel(None)
    a.set_xticks([0])
    a.set_xticklabels('')
    a.set_yticks([0])
    a.set_yticklabels('')

fig.tight_layout()
plt.show()
