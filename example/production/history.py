#!/usr/bin/env python

from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import supervillain
import steps


import logging
logger = logging.getLogger(__name__)

def plot_history(ensemble):

    scalars = set(o for o, cls in supervillain.observables.items() if issubclass(cls, supervillain.observable.Scalar))

    fig, ax = plt.subplots(len(scalars), 2,
        figsize=(10, 2.5*len(scalars)),
        gridspec_kw={'width_ratios': [4, 1], 'wspace': 0},
        sharey='row',
        squeeze=False
        )

    S = ensemble.Action
    fig.suptitle(f"W={S.W} κ={S.kappa:0.6f} N={S.Lattice.nx} {S.__class__.__name__}")

    for a, o in zip(ax, scalars):
        ensemble.plot_history(a, o)
        a[0].set_ylabel(o)
        
    ax[-1][0].set_xlabel('Monte Carlo Time')
    ax[-1][1].set_xlabel('Frequency')
    fig.tight_layout()

    return fig, ax

def visualize(ensembles):
    figs = deque()
    for E in ensembles:
        fig, ax = plot_history(E)
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

    figs = visualize(results.ensembles(ensembles))

    if args.pdf:
        results.pdf(args.pdf, figs)
    else:
        plt.show()

