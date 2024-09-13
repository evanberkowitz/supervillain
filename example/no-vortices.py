#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain
from supervillain.analysis import Uncertain
import supervillain.analysis.comparison_plot as comparison_plot
supervillain.observable.progress=tqdm

import logging
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def path(args, action):
    return f'W=∞/kappa={args.kappa:0.6f}/N={args.N}/action={action}'

def generate(args, action):

    L = supervillain.lattice.Lattice2D(args.N)

    if action == 'villain':
        S = supervillain.action.Villain(L, args.kappa, W=float('inf'))
        g = supervillain.generator.villain.Hammer(S)
        g = supervillain.generator.combining.KeepEvery(3, g)
    elif action == 'worldline':
        S = supervillain.action.Worldline(L, args.kappa, W=float('inf'))
        g = supervillain.generator.worldline.Hammer(S)
        g = supervillain.generator.combining.KeepEvery(9, g)

    logging.info(f'{action=}...')
    with logging_redirect_tqdm():
        E = supervillain.Ensemble(S).generate(args.configurations, g, start='cold', progress=tqdm)
    logging.info(f'{g.report()}')
    return E

def decorrelate(E, observables):

    # A first computation of the autocorrelation time will have effects from thermalization.
    autocorrelation = E.autocorrelation_time(observables)
    # We aggressively cut to ensure thermalization.
    thermalized = E.cut(10*autocorrelation)

    # Now we can get a fair computation of the autocorrelation time.
    autocorrelation = thermalized.autocorrelation_time(observables=observables)
    logging.info(f'{action:<10} Autocorrelation time {autocorrelation}')

    return thermalized.every(autocorrelation)

def compare(bootstrap, observables):
    # The rest is show business!
    fig, ax = comparison_plot.setup(observables)
    comparison_plot.bootstraps(ax,
            tuple(bootstrap.values()),
            tuple(bootstrap.keys()),
            observables=observables
            )
    comparison_plot.histories(ax,
            tuple(B.Ensemble for B in bootstrap.values()),
            tuple(bootstrap.keys()),
            observables=observables
            )

    return fig, ax

def correlators(bootstrap):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    L = bootstrap['villain'].Ensemble.Action.Lattice

    bootstrap['villain'].plot_correlator(ax[0,0], 'Spin_Spin', label='Villain')
    bootstrap['villain'].plot_correlator(ax[0,1], 'Vortex_Vortex', label='Villain')

    bootstrap['worldline'].plot_correlator(ax[0,0], 'Spin_Spin', label='Worldline')
    bootstrap['worldline'].plot_correlator(ax[0,1], 'Vortex_Vortex', label='Worldline')

    diff = L.irrep(bootstrap['villain'].Spin_Spin - bootstrap['worldline'].Spin_Spin).real
    ax[1,0].errorbar(
        L.linearize(L.R_squared**0.5), L.linearize(diff.mean(axis=0)), L.linearize(diff.std(axis=0)),
        linestyle='none', marker='o',
    )
    ax[1,0].axhline(0, color='black', zorder=-1)

    diff = L.irrep(bootstrap['villain'].Vortex_Vortex - bootstrap['worldline'].Vortex_Vortex).real
    ax[1,1].errorbar(
        L.linearize(L.R_squared**0.5), L.linearize(diff.mean(axis=0)), L.linearize(diff.std(axis=0)),
        linestyle='none', marker='o',
    )
    ax[1,1].axhline(0, color='black', zorder=-1)

    for c in (0, 1):
        ax[0,c].legend()
        ax[0,c].set_xscale('log')
        ax[0,c].set_yscale('log')
        ax[0,c].set_ylabel(('Spin_Spin' if c == 0 else 'Vortex_Vortex'))
        ax[1,c].set_xscale('log')
        ax[1,c].set_ylabel('Difference')
        ax[1,c].set_xlabel('Δx')

    return fig, ax

def self_dual(bootstrap):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    v_bootstrap = bootstrap['villain']
    w_bootstrap = bootstrap['worldline']
    L = bootstrap['villain'].Ensemble.Action.Lattice

    v_bootstrap.plot_correlator(ax[0], 'Spin_Spin', label='Villain Spin_Spin')
    w_bootstrap.plot_correlator(ax[0], 'Vortex_Vortex', label='Worldline Vortex_Vortex')

    diff =       L.irrep(v_bootstrap.Spin_Spin - w_bootstrap.Vortex_Vortex).real
    mean = 0.5 * L.irrep(v_bootstrap.Spin_Spin + w_bootstrap.Vortex_Vortex).real
    ax[1].errorbar(
        L.linearize(L.R_squared**0.5), L.linearize((diff/mean).mean(axis=0)), L.linearize((diff/mean).std(axis=0)),
        linestyle='none', marker='o',
    )
    ax[1].axhline(0, color='black', zorder=-1)

    ax[0].legend()
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Two-Point Correlator')
    ax[1].set_xscale('log')
    ax[1].set_ylabel('Difference / Mean')
    ax[1].set_xlabel('Δx')

    return fig, ax

if __name__ == '__main__':
    parser = supervillain.cli.ArgumentParser(description = 'The goal is to compute the same observables using both the Villain and Worldline actions and to check that they agree.  The Villain action is sampled with a combination of Site and Link Updates and, when W>1, the worm.')
    parser.add_argument('--N', type=int, default=21, help='Sites on a side.')
    parser.add_argument('--kappa', type=float, default=0.5/np.pi, help='κ.  Defaults to the self-dual 1/2π.')
    parser.add_argument('--configurations', type=int, default=10000, help='Defaults to 100000.  You need a good deal of configurations with κ=0.5 because of autocorrelations in the Villain sampling.')
    parser.add_argument('--h5', default='no-vortices.h5', help='File for storage')
    parser.add_argument('--pdf', type=str, default='', help='PDF to write the figures into.')
    parser.add_argument('--reset', default=False, action='store_true')
    parser.add_argument('--observables', nargs='*', help='Names of observables to compare.  Defaults to a list of 4 observables.',
                        default=('ActionDensity', 'InternalEnergyDensity', 'InternalEnergyDensitySquared', ))

    args = parser.parse_args()
    kappa_str = ('1/2π' if args.kappa == 0.5/np.pi else str(kappa))

    bootstrap = dict()

    for action in ('villain', 'worldline'):
        p=path(args, action)
        try:
            with h5.File(args.h5, ('a' if args.reset else 'r')) as file:
                if p in file:
                    if args.reset:
                        logger.info(f'Resetting {p}')
                        del file[p]
                    else:
                        bootstrap[action] = supervillain.analysis.Bootstrap.from_h5(file[p])
                        continue
        except FileNotFoundError:
            with h5.File(args.h5, 'a') as file:
                pass

        E = generate(args, action)
        D = decorrelate(E, args.observables)
        D.measure()

        B = supervillain.analysis.Bootstrap(D, 200)
        B.Vortex_Vortex /= B.Vortex_Vortex[:,0,0][:,None,None]
        B.Spin_Spin /= B.Spin_Spin[:,0,0][:,None,None]

        bootstrap[action] = B

        with h5.File(args.h5, 'a') as file:
            bootstrap[action].to_h5(file.create_group(p))


    fig_comparison, ax = compare(bootstrap, args.observables)
    fig_comparison.suptitle(f'W=∞ κ={kappa_str} N={args.N}')
    fig_comparison.tight_layout()

    fig_correlators, ax = correlators(bootstrap)
    fig_correlators.suptitle(f'W=∞ κ={kappa_str} N={args.N}')
    fig_correlators.tight_layout()

    fig_dual, ax = self_dual(bootstrap)
    fig_dual.suptitle(f'W=∞ κ={kappa_str} N={args.N}')
    fig_dual.tight_layout()

    if args.pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(args.pdf) as pdf:
            for n in plt.get_fignums():
                fig= plt.figure(n)
                fig.savefig(pdf, format='pdf') 
    else:
        plt.show()
