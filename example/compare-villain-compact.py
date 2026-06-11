#!/usr/bin/env python
r'''
Compare 2D Villain MCMC: production ``Lattice2D`` vs compact ``Form`` stack.

Two *independent* chains (no shared RNG).  Observables are measured in a common
production layout (:mod:`supervillain.compare.villain`); compact configs are
bridged with :mod:`supervillain.layout`.

Matched update scheme: checkerboarded site :math:`\phi` + parallel link :math:`n`
(``SiteUpdate`` + ``LinkUpdate``).  Use ``--generator hammer`` for the full
production ergodic set (compact stays minimal — not comparable MC kernel).
'''

import logging

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import supervillain
from supervillain.analysis import autocorrelation_time
from supervillain.compare import (
    OBSERVABLES,
    bootstrap_mean_stderr,
    compare_independent,
    format_comparison,
    measure_ensemble,
)
from supervillain.compact import generator as compact_generator
from supervillain.compact import villain_action as compact_villain

logger = logging.getLogger(__name__)

parser = supervillain.cli.ArgumentParser(description=__doc__)
parser.add_argument('--N', type=int, default=6, help='Sites per side (square).')
parser.add_argument('--kappa', type=float, default=0.5)
parser.add_argument('--W', type=supervillain.cli.W, default=1)
parser.add_argument('--configurations', type=int, default=50_000)
parser.add_argument('--bootstrap-draws', type=int, default=500)
parser.add_argument('--sigma', type=float, default=3.0,
                    help='Flag mismatch when |z| exceeds this (default 3).')
parser.add_argument('--generator', choices=('minimal', 'hammer'), default='minimal',
                    help='minimal: site+link on both paths; hammer: production only.')
parser.add_argument('--observables', nargs='*',
                    default=tuple(OBSERVABLES.keys()),
                    help='Subset of compare.villain observables.')
parser.add_argument('--figure', default=False, type=str)
parser.add_argument('--skip-plot', action='store_true')
args = parser.parse_args()

if args.generator == 'hammer':
    logger.warning(
        'hammer uses production worm/exact/holonomy; compact chain stays minimal — '
        'statistics compare physics on bridged configs, not identical MC kernels.'
    )


def production_minimal_hammer(S):
    return supervillain.generator.combining.Sequentially((
        supervillain.generator.villain.SiteUpdate(S),
        supervillain.generator.villain.LinkUpdate(S),
    ))


def production_generator(S):
    if args.generator == 'hammer':
        return supervillain.generator.villain.Hammer(S)
    return production_minimal_hammer(S)


def generate_production(L, S):
    g = production_generator(S)
    with logging_redirect_tqdm():
        e = supervillain.Ensemble(S).generate(
            args.configurations, g, start='cold', progress=tqdm,
        )
    print(g.report())
    return e


def generate_compact(L, S):
    g = compact_generator.MinimalHammer(S)
    with logging_redirect_tqdm():
        e = supervillain.Ensemble(S).generate(
            args.configurations, g, start='cold', progress=tqdm,
        )
    print(g.report())
    return e


def thermalize_and_thin(ensemble, *, bridge_from_form=False):
    r'''Cut aggressively, then thin by autocorrelation of ActionDensity.'''
    action_density = measure_ensemble(
        ensemble, names=('ActionDensity',), bridge_from_form=bridge_from_form,
    )['ActionDensity']
    tau = autocorrelation_time(action_density)
    cut = max(1, int(10 * tau))
    thinned = ensemble.cut(cut)
    action_density = measure_ensemble(
        thinned, names=('ActionDensity',), bridge_from_form=bridge_from_form,
    )['ActionDensity']
    tau = autocorrelation_time(action_density)
    stride = max(1, int(tau))
    return thinned.every(stride), cut, stride, tau


def main():
    L = supervillain.lattice.Lattice(D=2, N=args.N)

    S_prod = supervillain.Villain(L, args.kappa, W=args.W)
    S_compact = compact_villain.Villain(L, args.kappa, W=args.W)

    print('=== Production chain ===')
    prod = generate_production(L, S_prod)
    print('=== Compact chain ===')
    compact = generate_compact(L, S_compact)

    prod, prod_cut, prod_stride, prod_tau = thermalize_and_thin(prod)
    compact, compact_cut, compact_stride, compact_tau = thermalize_and_thin(
        compact, bridge_from_form=True,
    )

    print()
    print('Thermalization / thinning')
    print('-------------------------')
    print(f'  production: cut {prod_cut}, stride {prod_stride}, τ≈{prod_tau},'
          f' remaining {len(prod)}')
    print(f'  compact:    cut {compact_cut}, stride {compact_stride}, τ≈{compact_tau},'
          f' remaining {len(compact)}')

    prod_series = measure_ensemble(prod, names=args.observables)
    compact_series = measure_ensemble(
        compact, names=args.observables, bridge_from_form=True,
    )

    print()
    print(f'Observable comparison (bootstrap {args.bootstrap_draws} draws,'
          f' {args.sigma}σ)')
    print('=' * 100)

    all_ok = True
    for name in args.observables:
        m_p, e_p = bootstrap_mean_stderr(
            prod_series[name], draws=args.bootstrap_draws,
        )
        m_c, e_c = bootstrap_mean_stderr(
            compact_series[name], draws=args.bootstrap_draws,
        )
        result = compare_independent(m_p, e_p, m_c, e_c, sigma=args.sigma)
        print(format_comparison(name, result))
        all_ok &= result['consistent']

    # Cross-check: production registered observables vs compare module (same chain).
    overlap = [o for o in ('ActionDensity', 'InternalEnergyDensity', 'WindingSquared')
               if o in args.observables]
    if overlap:
        print()
        print('Sanity: production Ensemble observables vs compare.villain (same chain)')
        print('-' * 100)
        prod.measure(observables=overlap)
        for name in overlap:
            registered = supervillain.batch.Batch.as_array(getattr(prod, name))
            compare = prod_series[name]
            max_diff = np.max(np.abs(registered - compare))
            print(f'  {name:28s}  max|Δ| per draw = {max_diff:.3e}')

    if not args.skip_plot:
        fig, axes = plt.subplots(len(args.observables), 2,
                                 figsize=(10, 2.5 * len(args.observables)),
                                 squeeze=False)
        for row, name in enumerate(args.observables):
            axes[row, 0].plot(prod_series[name], lw=0.5, alpha=0.7, label='production')
            axes[row, 0].plot(compact_series[name], lw=0.5, alpha=0.7, label='compact')
            axes[row, 0].set_ylabel(name)
            axes[row, 0].legend(loc='upper right', fontsize=8)
            axes[row, 1].hist(prod_series[name], bins=31, density=True,
                              alpha=0.5, label='production')
            axes[row, 1].hist(compact_series[name], bins=31, density=True,
                              alpha=0.5, label='compact')
            axes[row, 1].legend(loc='upper right', fontsize=8)
        axes[-1, 0].set_xlabel('thinned draw')
        axes[-1, 1].set_xlabel('value')
        fig.suptitle(f'N={args.N} κ={args.kappa} W={args.W}')
        fig.tight_layout()
        if args.figure:
            fig.savefig(args.figure)
        else:
            plt.show()

    if not all_ok:
        raise SystemExit('At least one observable differs beyond statistical precision.')
    print()
    print('All compared observables consistent within', args.sigma, 'σ.')


if __name__ == '__main__':
    main()
