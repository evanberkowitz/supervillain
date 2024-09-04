#!/usr/bin/env python

from collections import deque
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


def transition(ax, data):

    for N, dat in data.groupby('N'):
        for observable in ('VortexCriticalMoment', 'SpinCriticalMoment'):
            line = ax.plot(
                dat['kappa'], dat[observable],
                marker='none', linestyle='-',
                label=f'{observable} {N=}',
                )

            for action, da in dat.groupby('action'):
                ax.errorbar(
                    da['kappa'], da[observable], da[f'{observable}±'],
                    marker=('o' if action == 'Villain' else 's'), markerfacecolor='none',
                    color=line[0].get_color(),
                    )

    ax.set_xlabel('κ')

    ax.legend(loc='upper right')
 

def visualize(data):

    figs=deque()

    for W, dat in data.groupby('W'):
        fig, ax = plt.subplots(1,1, figsize=(10, 8), sharex='col')
        fig.suptitle(f'{W=}', fontsize=24)

        transition(ax, dat)

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
