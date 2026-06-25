#!/usr/bin/env python


import numpy as np
import supervillain
import matplotlib.pyplot as plt

parser = supervillain.cli.ArgumentParser()
parser.add_argument('--N', type=int, default=11, help='Sites on a side.')
args = parser.parse_args()

L = supervillain.lattice.Lattice2D(args.N)

c = L.form(0)
c[0] = np.cos(np.expand_dims(L.x, axis=1))

sym = L.form(0)
sym[0] = L.symmetrize(c[0])

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title('0-form')
ax[1].set_title('symmetrized')

L.plot_form(c,   ax[0])
L.plot_form(sym, ax[1])

fig.tight_layout()
plt.show()
