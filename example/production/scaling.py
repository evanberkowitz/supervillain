#!/usr/bin/env python

from collections import deque
import steps

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


def scaling_plot(ax, observable, data, kappas_per_column=8):

    label = observable[1]
    observable = observable[0]

    for ((kappa, action), dat) in data.groupby(['kappa', 'action']):
        ax.errorbar(
            1/dat['N'], dat[observable], dat[f'{observable}±'],
            marker=('o' if action == 'Villain' else 's'), markerfacecolor='none',
            label=f'κ={kappa:0.3f} {action}',
            )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylabel(label, fontsize=18)
    ax.set_xlabel('1/N', fontsize=18)

    N = data.N.unique()
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.set_xticks(1/N, minor=False)
    ax.set_xticks([],  minor=True)
    ax.set_xticklabels([f'1/{n}' for n in N])
    ax.grid(True, which='both', axis='y')
    ax.grid(False, which='both', axis='x')

    ax.legend(loc='lower left', ncols=(1+ (len(data['kappa'].unique()) // kappas_per_column)))


def visualize(data, all_observables=False):

    figs=deque()

    for W, dat in data.groupby('W'):

        plot_all = W!=1 or all_observables
        observables = (('SpinCriticalMoment', r'$C_S$'),
                       ('VortexCriticalMoment', r'$C_V$'),
                       )
        if not plot_all:
            observables = observables[:1]

        n = len(observables)

        fig, ax = plt.subplots(1, n, figsize=(10*n, 8), sharex='col', squeeze=False)
        ax = ax[0]
        fig.suptitle(f'{W=}', fontsize=24)

        for a, o in zip(ax, observables):
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
    parser.add_argument('--all', default=False, action='store_true')
    parser.add_argument('--pdf', default='', type=str)

    args = parser.parse_args()

    ensembles = args.input_file.ensembles
    if args.parallel:
        import parallel
        ensembles = ensembles.apply(parallel.io_prep, axis=1)
    print(ensembles)

    data = results.collect(ensembles, observables=('SpinCriticalMoment', 'VortexCriticalMoment'))
    figs = visualize(data, all_observables=args.all)

    if args.pdf:
        results.pdf(args.pdf, figs)
    else:
        plt.show()
