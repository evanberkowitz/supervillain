#!/usr/bin/env python

from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import supervillain
import steps


import logging
logger = logging.getLogger(__name__)

_default_correlators=(
        ('Spin_Spin', r'$S(\Delta x)$'),
        ('Vortex_Vortex', r'$V(\Delta x)$'),
        #('Winding_Winding', r'$W(\Delta x)$'),
        #('Action_Action', r'$A(\Delta x)$'),
    )

def plot_correlators(ensembles,
                     correlators=_default_correlators,
                     ):

    if (ensembles['W']==1).all():
        correlators = tuple(c for c in correlators if c[0] != 'Vortex_Vortex')

    labels = tuple(c[1] for c in correlators)
    correlators = tuple(c[0] for c in correlators)

    fig, ax = plt.subplots(1, len(correlators), figsize=(12*len(correlators), 8), squeeze=False)
    ax = ax[0]

    e = len(ensembles)

    for i, (idx, row) in enumerate(ensembles.sort_values(by=['N'], ascending=True).iterrows()):
        L = supervillain.lattice.Lattice2D(row['N'])

        dx = L.linearize(L.R_squared**0.5)[1:]

        # x-dependent offset for a log scale:
        dx *= 1 + (i-e/2)/e / max(ensembles['N']) / 2

        for o, a in zip(correlators, ax):
            a.errorbar(
                    dx,
                    L.linearize(L.irrep(row[o].real))[1:], L.linearize(L.irrep(row[o+'±']))[1:],
                    linestyle='none',
                    label=f"N={row['N']}",
                    marker=('o' if (row['W']==1 and o=='Vortex_Vortex') else 'none')
                    )

    ax[0].legend(loc='upper right')

    for a, o in zip(ax, labels):
        a.set_xlabel(r'$\Delta x$', fontsize=18)
        a.set_ylabel(o, fontsize=18)
        a.set_xscale('log')
        a.set_yscale('log')

    return fig, ax


def visualize(ensembles):
    figs = deque()

    for (W, kappa), data in ensembles.groupby(['W', 'kappa']):
        fig, ax = plot_correlators(data)
        fig.suptitle(f'{W=} κ={kappa:0.6f}', fontsize=24)
        fig.tight_layout()
        figs.append(fig)

    return figs

if __name__ == '__main__':

    import supervillain
    import results

    parser = supervillain.cli.ArgumentParser()
    parser.add_argument('input_file', type=supervillain.cli.input_file('input'), default='input.py')
    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument('--pdf', default='', type=str)

    args = parser.parse_args()

    ensembles = args.input_file.ensembles
    if args.parallel:
        import parallel
        ensembles = ensembles.apply(parallel.io_prep, axis=1)
    logger.info(ensembles)

    data = results.collect(ensembles)
    figs = visualize(data)

    if args.pdf:
        results.pdf(args.pdf, figs)
    else:
        plt.show()

