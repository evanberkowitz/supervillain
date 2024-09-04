#!/usr/bin/env python

from collections import deque
import steps

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


def scaling_plot(ax, observable, data):

    for ((kappa, action), dat) in data.groupby(['kappa', 'action']):
        ax.errorbar(
            1/dat['N'], dat[observable], dat[f'{observable}±'],
            marker=('o' if action == 'Villain' else 's'), markerfacecolor='none',
            label=f'κ={kappa:0.3f} {action}',
            )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylabel(observable)
    ax.set_xlabel('1/N')

    L = data.N.unique()
    ax.set_xticks(1/L)
    ax.set_xticklabels([f'1/{l}' for l in L])
    ax.minorticks_off()
    ax.grid(True, which='both', axis='y')

    ax.legend(loc='upper right')
 

def visualize(data):

    figs=deque()

    for W, dat in data.groupby('W'):
        fig, ax = plt.subplots(1,2, figsize=(20, 8), sharex='col')
        fig.suptitle(f'{W=}', fontsize=24)

        for a, o in zip(ax, ('SpinCriticalMoment', 'VortexCriticalMoment')):
            scaling_plot(a, o, dat)

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
    print(ensembles)

    data = results.collect(ensembles)
    figs = visualize(data)

    if args.pdf:
        results.pdf(args.pdf, figs)
    else:
        plt.show()
