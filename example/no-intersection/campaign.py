#!/usr/bin/env python

r"""
Overnight DefectGas campaign: for each volume tier and each κ, generate one ensemble
with the (compiled) grand-canonical defect gas, measure the θ-shift diagnostics
(``IntersectionSusceptibility``, ``ThetaBinderCumulant``, the absolutely-normalized
Θ correlator) alongside the spin diagnostics (``SpinSusceptibility``,
``Spin_Spin_Normalized``), and write ONE h5 file per ensemble (decorrelated ensemble +
bootstrap + symmetrized correlators + scalar summaries + metadata).

Per point the recipe is: deep untallied DefectGas thermalization from cold (the gas is
unjammable, so this works at every κ), an annealing cooldown that lands the chain
exactly in the vacuum sector (a hot handoff with defects still in flight poisons the
tuner's vacuum-dwell bookkeeping), ζ tuned on the hot **valid** configuration (the
tuner keeps the largest healthy ζ --- which is also what the Binder's quartic-sector
statistics want), generation with ``Sequentially((SiteUpdate, ExactUpdate,
CohomologyUpdate, DefectGas))`` (the cheap exact/holonomy moves ride along; the gas
emits the inline ``Theta_Theta`` / ``Vacuum_Ticks`` / ``Four_Defect``), then the
standard autocorrelation cut, decorrelation stride, and bootstrap.

Failures are isolated per point (logged, skipped) so an unattended run always makes it
to the end of the grid.  Run from example/no-intersection/:

    uv run python campaign.py                      # the whole overnight grid
    uv run python campaign.py --tiers 6 8          # just the small volumes
    uv run python campaign.py --tiers 12 --kappas 0.05 0.1
"""

import argparse
import os
import time
import traceback

import h5py as h5
import numpy as np

import supervillain
from supervillain.analysis import Bootstrap
from supervillain.lattice import Form, Lattice
import supervillain.generator.no_intersection as gen
import supervillain.generator.villain as villain
from supervillain.generator.combining import Sequentially

SCALARS = ('ActionDensity', 'InternalEnergyDensity', 'InternalEnergyDensitySquared')
SUMMARIES = ('SpinSusceptibility', 'IntersectionSusceptibility',
             'ThetaBinderCumulant', 'WindingSquared')

# Detail concentrates in the transition window κ ≈ 0.03-0.2 (spin LRO dies, χ_θ peaks),
# with anchors deep in the low-κ (defect-dense) and high-κ (spin-ordered) flanks.
KAPPAS_FULL = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08,
               0.10, 0.13, 0.16, 0.20, 0.30, 0.60, 1.00)
KAPPAS_FOCUS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.20, 0.60)
KAPPAS_PEAK = (0.03, 0.05, 0.10, 0.20)

# (N, configurations, thermalization sweeps, kappas).  Configuration counts scale down
# with volume to keep each point minutes-scale and --- at N=16 --- inside RAM.
TIERS = {
    6:  dict(configurations=3000, therm=1500, kappas=KAPPAS_FULL),
    8:  dict(configurations=2000, therm=1500, kappas=KAPPAS_FULL),
    12: dict(configurations=1000, therm=1000, kappas=KAPPAS_FOCUS),
    16: dict(configurations=250,  therm=600,  kappas=KAPPAS_PEAK),
}

# One rung below the default ladder, for the biggest volumes (entropy pushes D up, so
# the healthy ζ shrinks with V).
LADDER = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005)
D_MAX = 8
# Conservative: hot enough to open corridors through jams, vacuum-heavy enough that the
# chain is not defect-condensed when we hand it onward.  A too-hot therm ζ leaves a
# far-separated defect pair in flight whose one-way annihilation is ΔS-suppressed ---
# exactly the debt land_in_vacuum() would then grind on.
THERM_ZETA = 0.01

parser = argparse.ArgumentParser(description='Overnight DefectGas κ campaign.')
parser.add_argument('--tiers', type=int, nargs='+', default=(6, 8, 12, 16))
parser.add_argument('--kappas', type=float, nargs='+', default=None,
                    help='override the per-tier κ list')
parser.add_argument('--outdir', default='campaign-2026-07-08')
parser.add_argument('--seed', type=int, default=137)
parser.add_argument('--configurations', type=int, default=None,
                    help='override the per-tier configuration count')
parser.add_argument('--force', action='store_true',
                    help='regenerate points whose h5 already exists (default: skip)')
args = parser.parse_args()


def defect_count(S, n):
    r"""$D = \sum_x \left|q_x\right|$ of a configuration --- 0 iff exactly valid."""
    from supervillain.generator.no_intersection.charge import charge
    return int(np.abs(np.asarray(
        charge(Form(n, degree=1, lattice=S.Lattice)))).sum())


def land_in_vacuum(S, phi, n, seed):
    r"""
    Anneal any defects the thermalization left in flight: short runs at a tiny ζ
    (annihilation rewarded by $\zeta^{-|\Delta D|}$, creation suppressed), with φ
    sweeps in between reshuffling the ΔS barriers, until the configuration is exactly
    valid.  tune() and the production chain must start from the vacuum sector or the
    inherited debt poisons the vacuum-dwell bookkeeping.
    """
    for zc in (0.002, 0.0005):
        g = gen.DefectGas(S, zeta=zc, D_max=D_MAX, rng=np.random.default_rng(seed))
        for _ in range(400):
            if defect_count(S, n) == 0:
                return phi, n
            phi, n = g.run(phi, n, 5, tally=False)
    raise RuntimeError(f'cooldown failed: D = {defect_count(S, n)} persists')


def point(N, kappa, configurations, therm, seed):
    r"""Generate, measure, and write one (N, κ) ensemble; returns a summary string."""
    start_time = time.time()
    L = Lattice(4, N)
    S = supervillain.action.NoIntersections(L, kappa=kappa)

    # Deep thermalization from cold with the gas itself (compiled, unjammable at low κ,
    # φ sweeps included), then land exactly on the constraint surface: tune() and the
    # production start must be valid.  The production run's 10τ cut mops up residue.
    g0 = gen.DefectGas(S, zeta=THERM_ZETA, D_max=D_MAX,
                       rng=np.random.default_rng(seed))
    phi = np.zeros((1,) + tuple(L.dims))
    n = np.zeros((4,) + tuple(L.dims), dtype=np.int64)
    phi, n = g0.run(phi, n, therm, tally=False)
    phi, n = land_in_vacuum(S, phi, n, seed + 3)

    # Longer probes than the tune() default: a short probe from a fresh valid start has
    # not yet built the equilibrium defect density and overestimates the vacuum dwell.
    zeta = gen.DefectGas.tune(S, D_max=D_MAX, rng=np.random.default_rng(seed + 1),
                              phi=phi, n=n, ladder=LADDER, sweeps=150)

    # Even so, a mid-run condensation (step() exhausting its horizon) is possible in
    # the transition window; retry down the ladder rather than losing the point.
    candidates = [zeta] + [z for z in LADDER if z < zeta][:3]
    for attempt, z in enumerate(candidates):
        gas = gen.DefectGas(S, zeta=z, D_max=D_MAX, max_step_sweeps=2000,
                            rng=np.random.default_rng(seed + 2 + attempt))
        chain = Sequentially((
            villain.SiteUpdate(S),
            villain.ExactUpdate(S),
            villain.CohomologyUpdate(S),
            gas,                   # last, so the emitted configuration is its vacuum tick
        ))
        try:
            e = supervillain.Ensemble(S).generate(
                configurations, chain,
                start={'phi': Form(phi, degree=0, lattice=L),
                       'n': Form(n, degree=1, lattice=L)})
            zeta = z
            break
        except RuntimeError:
            if attempt == len(candidates) - 1:
                raise
            print(f'  N={N} kappa={kappa:g}: zeta={z:g} condensed mid-run; '
                  f'retrying at zeta={candidates[attempt + 1]:g}', flush=True)

    q2 = np.asarray(e.TopologicalChargeDensitySquared)
    assert np.abs(q2).max() == 0, f'constraint violated at N={N} kappa={kappa}'

    # Standard cut / decorrelate / bootstrap.  Unattended run: if the cut would eat
    # everything, take half and flag it for morning review instead of aborting.
    auto = e.autocorrelation_time(observables=SCALARS)
    suspect = bool(10 * auto >= configurations)
    thermalized = e.cut(configurations // 2 if suspect else 10 * auto)
    auto = thermalized.autocorrelation_time(observables=SCALARS)
    decorrelated = thermalized.every(auto)
    b = Bootstrap(decorrelated)

    # Scalar summaries from the bootstrap distributions.
    summaries = {}
    for name in SUMMARIES:
        dist = np.asarray(getattr(b, name)).real
        summaries[name] = (dist.mean(), dist.std())

    # Correlators, hyperoctahedrally symmetrized on a throwaway Lattice (symmetrize
    # caches would otherwise ride into the pickled Batch item_kwargs and blow HDF5's
    # per-attribute limit; see kappa_scan.py).
    Lsym = Lattice(4, N)
    T = np.asarray(b.Theta_Theta).real
    VT = np.asarray(b.Vacuum_Ticks).real
    Theta = Lsym.symmetrize(T / VT[:, None, None, None, None])
    SS = Lsym.symmetrize(np.asarray(b.Spin_Spin_Normalized).real)
    correlators = {'Theta': (Theta.mean(axis=0), Theta.std(axis=0)),
                   'Spin_Spin_Normalized': (SS.mean(axis=0), SS.std(axis=0))}

    os.makedirs(f'{args.outdir}/N{N}', exist_ok=True)
    path = f'{args.outdir}/N{N}/kappa{kappa:g}.h5'
    # Strip analysis-accumulated cached properties from the Lattice (they ride into the
    # pickled Batch item_kwargs and blow HDF5's ~64KB attribute limit at N >= 8).
    baseline = set(Lattice(4, N).__dict__)
    for stale in [k for k in L.__dict__ if k not in baseline]:
        del L.__dict__[stale]
    with h5.File(path, 'w') as f:
        # The generator carries tuple-keyed caches h5 refuses (issue #65) and is not
        # needed downstream; withhold it from the write.
        generator = decorrelated.__dict__.pop('generator', None)
        try:
            decorrelated.to_h5(f.create_group('ensemble'))
            b.to_h5(f.create_group('bootstrap'))
        finally:
            if generator is not None:
                decorrelated.generator = generator
        for name, (mean, err) in correlators.items():
            f[f'correlators/{name}/mean'] = mean
            f[f'correlators/{name}/err'] = err
        for name, (mean, err) in summaries.items():
            f[f'scalars/{name}/mean'] = mean
            f[f'scalars/{name}/err'] = err
        f['meta/report'] = chain.report()
        f.attrs.update({'kappa': kappa, 'N': N, 'configurations': configurations,
                        'therm': therm, 'zeta': zeta, 'D_max': D_MAX,
                        'seed': seed, 'tau': auto,
                        'decorrelated': len(decorrelated),
                        'thermalization_suspect': suspect})

    # The per-ensemble diagnostic PDF (histories, bootstrap distributions, correlator
    # plots) rides along next to the h5.
    try:
        import campaign_diagnostics
        campaign_diagnostics.diagnose(path)
    except Exception:
        traceback.print_exc()

    chi_S = summaries['SpinSusceptibility']
    chi_T = summaries['IntersectionSusceptibility']
    U = summaries['ThetaBinderCumulant']
    return (f'N={N:<3} kappa={kappa:<6g} zeta={zeta:<7g} tau={auto:<3} '
            f'decorr={len(decorrelated):<5} '
            f'chi_S={chi_S[0]:9.3f}({chi_S[1]:.3f}) '
            f'chi_theta-1={chi_T[0]-1:+11.4e}({chi_T[1]:.1e}) '
            f'U={U[0]:6.3f}({U[1]:.3f}) '
            f'{"SUSPECT " if suspect else ""}[{time.time()-start_time:.0f}s]')


os.makedirs(args.outdir, exist_ok=True)
for N in args.tiers:
    tier = TIERS[N]
    for kappa in (args.kappas if args.kappas is not None else tier['kappas']):
        if not args.force and os.path.exists(f'{args.outdir}/N{N}/kappa{kappa:g}.h5'):
            print(f'N={N:<3} kappa={kappa:<6g} exists, skipping', flush=True)
            continue
        seed = args.seed + 1000 * N + int(round(kappa * 10000))
        configurations = (args.configurations if args.configurations is not None
                          else tier['configurations'])
        try:
            row = point(N, kappa, configurations, tier['therm'], seed)
        except Exception:
            row = f'N={N:<3} kappa={kappa:<6g} FAILED'
            traceback.print_exc()
        print(row, flush=True)
    # Refresh the figures after every completed tier so partial results are plottable.
    try:
        import subprocess
        subprocess.run(['uv', 'run', 'python', 'campaign_plot.py',
                        '--outdir', args.outdir], check=False)
    except Exception:
        traceback.print_exc()
print('campaign complete', flush=True)
