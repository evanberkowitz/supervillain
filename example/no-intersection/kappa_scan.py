#!/usr/bin/env python

r"""
Scan κ in the No-Intersection model, measuring the long-distance behavior of
``Spin_Spin_Normalized`` and ``Intersection_Intersection_Normalized``.

Per κ this follows the ``observables.py`` recipe (generate from cold, autocorrelation
cut, decorrelation stride, bootstrap), symmetrizes both correlators over the
hyperoctahedral group, and writes ONE h5 file per ensemble --- the decorrelated
ensemble, the bootstrap, the symmetrized correlator means/errors, the plateau values
at on-axis $(N/2, 0, 0, 0)$ and the antipode $(N/2, N/2, N/2, N/2)$, and metadata.

The sampler is the Hammer roster with the worm slot swapped: by default the
``FreeTargetWorm`` in movers-only mode (``--idle-probability 0.0``); ``--worm two-link``
runs the ``TwoLinkAdaptiveWorm`` instead (the cross-check control).

Run from example/no-intersection/:

    uv run python kappa_scan.py --N 8 --configurations 1000 [--kappas 0.005 0.01 ...]
    uv run python kappa_scan.py --N 8 --worm two-link --kappas 0.01 0.1 0.6
"""

import os

import h5py as h5
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import supervillain
from supervillain.analysis import Bootstrap
import supervillain.generator.no_intersection as gen
import supervillain.generator.villain as villain
from supervillain.generator.combining import Sequentially
supervillain.observable.progress = tqdm

DEFAULT_KAPPAS = (0.005, 0.01, 0.05, 0.10, 0.15, 0.20,
                  0.30, 0.40, 0.60, 0.80, 1.00, 1.40)
SCALARS = ('ActionDensity', 'InternalEnergyDensity', 'InternalEnergyDensitySquared')
CORRELATORS = ('Spin_Spin_Normalized', 'Intersection_Intersection_Normalized')

parser = supervillain.cli.ArgumentParser(description='κ scan of the No-Intersection model.')
parser.add_argument('--N', type=int, default=8)
parser.add_argument('--kappas', type=float, nargs='+', default=list(DEFAULT_KAPPAS))
parser.add_argument('--configurations', type=int, default=1000)
parser.add_argument('--worm', choices=('free', 'two-link'), default='free')
parser.add_argument('--idle-probability', type=float, default=0.0,
                    help='FreeTargetWorm draw mode; ignored for --worm two-link.')
parser.add_argument('--outdir', default='scan')
parser.add_argument('--seed', type=int, default=137)
args = parser.parse_args()


def sampler(S):
    r"""The Hammer roster with the worm slot swapped for the requested worm."""
    if args.worm == 'free':
        worm = gen.FreeTargetWorm(S, idle_probability=args.idle_probability)
    else:
        worm = gen.TwoLinkAdaptiveWorm(S)
    worm.rng = np.random.default_rng(args.seed)
    return Sequentially((
        villain.SiteUpdate(S),
        villain.ExactUpdate(S),
        villain.CohomologyUpdate(S),
        gen.ConstrainedLinkUpdate(S),
        gen.WrappingLoopUpdate(S),
        gen.PlanarFluxUpdate(S),
        gen.ScattershotUpdate(S),
        worm,
    )), worm


header = (f'{"kappa":>8} {"tau":>5} {"decorr":>7} '
          f'{"spin(N/2,0,0,0)":>22} {"spin antipode":>22} '
          f'{"ii(N/2,0,0,0)":>22} {"ii antipode":>22} {"<worm len>":>10}')
print(header)
rows = []
for kappa in args.kappas:
    L = supervillain.lattice.Lattice(4, args.N)
    S = supervillain.action.NoIntersections(L, kappa=kappa)
    G, worm = sampler(S)
    with logging_redirect_tqdm():
        e = supervillain.Ensemble(S).generate(args.configurations, G, start='cold',
                                              progress=tqdm)

    q2 = np.asarray(e.TopologicalChargeDensitySquared)
    assert np.abs(q2).max() == 0, f'constraint violated at kappa={kappa}'

    auto = e.autocorrelation_time(observables=SCALARS)
    if 10 * auto >= args.configurations:
        raise SystemExit(f'kappa={kappa}: thermalization cut 10*tau = {10*auto} eats the whole '
                         f'run ({args.configurations} configurations); increase --configurations.')
    thermalized = e.cut(10 * auto)
    auto = thermalized.autocorrelation_time(observables=SCALARS)
    decorrelated = thermalized.every(auto)
    b = Bootstrap(decorrelated)

    onaxis = (args.N // 2, 0, 0, 0)
    antipode = (args.N // 2,) * 4
    results = {}
    # symmetrize() lazily caches the D!*2^D hyperoctahedral permutation as a
    # @cached_property on the Lattice instance; since the Form-typed configuration
    # fields (phi, n) carry a reference to this same L in their Batch item_kwargs,
    # calling L.symmetrize(...) before the h5 write would inflate that Batch's pickled
    # item_kwargs attribute past HDF5's per-attribute header limit.  A throwaway
    # Lattice sidesteps that without touching the shared instance.
    Lsym = supervillain.lattice.Lattice(4, args.N)
    for name in CORRELATORS:
        C = Lsym.symmetrize(np.asarray(getattr(b, name)).real)
        mean, err = C.mean(axis=0), C.std(axis=0)
        results[name] = (mean, err)

    os.makedirs(f'{args.outdir}/N{args.N}', exist_ok=True)
    path = f'{args.outdir}/N{args.N}/kappa{kappa}-{args.worm}.h5'
    # The binding failure is the configuration Batch's ``_item_kwargs`` HDF5 attribute,
    # which pickles the Lattice; cached properties accumulated during the analysis
    # (correlator machinery etc.) scale with volume and blow HDF5's ~64KB per-attribute
    # header limit at N >= 8 (datasets have no such limit -- same library limitation as
    # the workarounds above). Strip everything a fresh Lattice would not carry; cached
    # properties recompute on demand, so this is always safe.
    baseline = set(supervillain.lattice.Lattice(4, args.N).__dict__)
    for stale in [k for k in L.__dict__ if k not in baseline]:
        del L.__dict__[stale]
    with h5.File(path, 'w') as f:
        # The worm's private enumeration caches (e.g. FreeTargetWorm._library) are keyed
        # by shape tuples, which supervillain.h5.strategy.dict.Dict refuses to serialize
        # (github.com/evanberkowitz/supervillain/issues/65); the generator is also not
        # needed downstream (no continue_from in this pipeline), so it is withheld from
        # the write rather than pickled.  b.Ensemble is decorrelated, so the same fix
        # covers both the ensemble and bootstrap groups.
        generator = decorrelated.__dict__.pop('generator', None)
        try:
            decorrelated.to_h5(f.create_group('ensemble'))
            b.to_h5(f.create_group('bootstrap'))
        finally:
            if generator is not None:
                decorrelated.generator = generator
        for name, (mean, err) in results.items():
            f[f'correlators/{name}/mean'] = mean
            f[f'correlators/{name}/err'] = err
            f[f'plateaus/{name}/onaxis'] = mean[onaxis]
            f[f'plateaus/{name}/onaxis_err'] = err[onaxis]
            f[f'plateaus/{name}/antipode'] = mean[antipode]
            f[f'plateaus/{name}/antipode_err'] = err[antipode]
        f['meta/report'] = worm.report()
        f.attrs.update({'kappa': kappa, 'N': args.N,
                        'configurations': args.configurations, 'seed': args.seed,
                        'worm': args.worm,
                        'idle_probability': (np.nan if args.idle_probability is None
                                             or args.worm != 'free'
                                             else args.idle_probability),
                        'tau': auto, 'decorrelated': len(decorrelated)})

    lengths = np.array(worm.worm_lengths)
    ss, ii = results['Spin_Spin_Normalized'], results['Intersection_Intersection_Normalized']
    row = (f'{kappa:>8} {auto:>5} {len(decorrelated):>7} '
           f'{ss[0][onaxis]:>+11.5f}±{ss[1][onaxis]:<10.5f} '
           f'{ss[0][antipode]:>+11.5f}±{ss[1][antipode]:<10.5f} '
           f'{ii[0][onaxis]:>+11.5f}±{ii[1][onaxis]:<10.5f} '
           f'{ii[0][antipode]:>+11.5f}±{ii[1][antipode]:<10.5f} '
           f'{lengths.mean():>10.2f}')
    print(row)
    rows.append(row)

print('\n' + header)
for row in rows:
    print(row)
