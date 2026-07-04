#!/usr/bin/env python
r"""
Relaxation test: can the pre-existing n-moves unknot a knotted vortex sheet?

Protocol: start from an explicitly knotted valid configuration (torus_knotted.py or
spun_sphere.py) or its unknotted control (--start *-unknot: the same construction fed
the trivial knot T(1, g−1)), run the constraint-preserving generators at moderate κ,
and watch the sheet area Σ|F|.  The Villain action drives n toward dφ/2π ≈ 0, so
sheet-shrinking moves are downhill and readily accepted.  An unknotted sheet can shrink
to the vacuum through embedded intermediates; a knot is a topological obstruction to
shrinking through embedded states.  Therefore:

  both melt to F = 0        → the move set demonstrably unknots (via stabilization,
                              non-manifold intermediates, worm excursions, or the global
                              moves) --- a definitive positive for mixing across knot types.
  control melts, knot stalls → a candidate practical obstruction; the plateau remnant is
                              a minimal knotted sheet --- census it (corridors.py)!
  neither melts             → κ badly tuned (too small: thermal sheets regrow; too large:
                              dynamics freezes); retune before concluding anything.

**What the monitors do and do not show.**  Σ|F| (the literal sheet area: dual plaquettes
counted with multiplicity) and ΣF² (which overweights multiplicity; identical on |F| ≤ 1
sheets) are MELT DETECTORS, not knot detectors, and the logic is one-sided:

  * F ≡ 0 reached from a knotted start is rigorous --- the recorded trajectory is a
    constructive unknotting path through valid configurations.
  * a stall is NOT evidence of knotting by itself: greedy quenches jam for boring
    reasons, and one run is one sample of a stochastic descent.  Evidence requires an
    ensemble: controls that reliably melt vs. knotted starts that reliably floor.
  * with --no-phi at κ ≳ 1 the descent variable is Σn² and F-fluctuations are
    e^{-κ·2π²·Δ(Σn²)}-suppressed (acceptance ~3·10⁻⁹ per minimal uphill proposal at
    κ = 1), so trajectories are near-deterministic quenches and area wiggles reflect the
    sheet rearranging at fixed descent, not thermal noise.  With dynamical φ or small κ,
    F fluctuates thermally, area says nothing about knot class, and the right tool is
    instead start-independence statistics between knotted and control starts (§7.3 of
    ergodicity.md).

Every report checks q ≡ 0 (the generators must preserve it) and, when |F| ≤ 1, the sheet
topology (components, χ, genus) --- so stabilization events (genus changes) and
component splits are visible along the trajectory.

Run from example/no-intersection/:

    uv run python unknot.py --start torus-trefoil [--kappa 1.0] [--sweeps 60] [--no-worm]
    uv run python unknot.py --start torus-unknot   # the control

Starts: torus-{trefoil,cinquefoil,unknot}, spun-{trefoil,cinquefoil,unknot}, vacuum.
"""

import argparse
import time

import numpy as np

import supervillain
from supervillain.lattice import Lattice, d
from supervillain.generator.no_intersection.charge import charge
from genus import topology
import torus_knotted
import spun_sphere

KNOTS = {'trefoil': (5, 2), 'cinquefoil': (7, 2), 'unknot': (5, 1)}


def start_configuration(L, name):
    if name == 'vacuum':
        return L.zeros(1, dtype=int)
    kind, _, knot = name.partition('-')
    g, k = KNOTS[knot]
    if kind == 'torus':
        n, _, _ = torus_knotted.configuration(L, grid=g, shift=k)
    elif kind == 'spun':
        n, _ = spun_sphere.configuration(L, grid=g, shift=k)
    else:
        raise ValueError(f'unknown start {name!r}')
    return n


def generators(S, worm=True, phi=True):
    gen = supervillain.generator
    gs = [
        gen.villain.ExactUpdate(S),
        gen.villain.CohomologyUpdate(S),
        gen.no_intersection.ConstrainedLinkUpdate(S),
        gen.no_intersection.WrappingLoopUpdate(S),
        gen.no_intersection.PlanarFluxUpdate(S),
        gen.no_intersection.ScattershotUpdate(S),
    ]
    if phi:
        gs.insert(0, gen.villain.SiteUpdate(S))
    if worm:
        gs.append(gen.no_intersection.IntersectionWorm(S))
    return gs


def snapshot(n):
    # Σ|F| is the literal sheet area (dual plaquettes counted with multiplicity); ΣF²
    # overweights multiplicity quadratically.  They agree exactly on |F| ≤ 1 sheets and
    # both are melt detectors only: F ≡ 0 ⟺ either vanishes.  Neither is a knot
    # detector --- see the docstring.
    F = np.asarray(d(n))
    area = int(np.abs(F).sum())
    f2 = int((F ** 2).sum())
    fmax = int(np.abs(F).max()) if area else 0
    if area == 0:
        return area, f2, 'vacuum'
    if fmax > 1:
        return area, f2, f'|F|max={fmax} (multiplicity; topology skipped)'
    # |F| ≤ 1: the sheet is multiplicity-free and its cell topology is meaningful.
    comps, junctions, shared = topology(n)
    chis = ','.join(str(c['chi']) for c in sorted(comps, key=lambda c: -c['F']))
    extra = (f', junctions {junctions}' if junctions else '') + \
            (f', shared vertices {shared}' if shared else '')
    return area, f2, f'{len(comps)} comp (χ: {chis}){extra}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', default='torus-trefoil',
                        choices=['vacuum'] + [f'{kind}-{k}' for kind in ('torus', 'spun')
                                              for k in KNOTS])
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--sweeps', type=int, default=60)
    parser.add_argument('--every', type=int, default=5, help='report interval (default 5)')
    parser.add_argument('--no-worm', action='store_true')
    parser.add_argument('--no-phi', action='store_true',
                        help='freeze φ ≡ 0: S = κ/2 Σ(2πn)², so shrinking is strictly '
                             'downhill and growth is e^{-κ2π²}-suppressed — the cleanest '
                             'purely-topological relaxation (a different, φ-frozen ensemble)')
    parser.add_argument('--anneal', default=None, metavar='K1,K2,...',
                        help="κ ladder cycled per sweep, built as generators with 'wrong' κ "
                             "(e.g. '0.3,1.0': one hot sweep, one cold, repeat).  Legitimate "
                             "without any detailed-balance care because a melt certificate is "
                             "just a sequence of valid moves — the annealed chain samples "
                             "nothing we use.")
    args = parser.parse_args()

    L = Lattice(4, args.N)
    kappas = ([float(t) for t in args.anneal.split(',')]
              if args.anneal else [args.kappa])
    ladder = [generators(supervillain.action.NoIntersections(L, kappa=k),
                         worm=not args.no_worm, phi=not args.no_phi)
              for k in kappas]
    gs = ladder[0]

    n0 = start_configuration(L, args.start)
    cfg = {'phi': L.zeros(0), 'n': n0}

    def nsq(cfg):
        # The φ-frozen descent variable: S = κ/2 (2π)² Σ n²  (with φ, a monitor only).
        return int((np.asarray(cfg['n']) ** 2).sum())

    area, f2, topo = snapshot(cfg['n'])
    schedule = ('κ=' + str(args.kappa) if len(kappas) == 1
                else 'κ ladder ' + ','.join(str(k) for k in kappas) + ' (cycled per sweep)')
    print(f'start {args.start}  N={args.N}  {schedule}  '
          f'generators: {", ".join(str(g) for g in gs)}', flush=True)
    print(f'sweep    0: n² {nsq(cfg):5d}  area {area:5d}  F² {f2:5d}  {topo}', flush=True)

    melted_at = None
    t0 = time.time()
    for sweep in range(1, args.sweeps + 1):
        for g in ladder[(sweep - 1) % len(ladder)]:
            cfg = g.step(cfg)
        area = int(np.abs(np.asarray(d(cfg['n']))).sum())
        if sweep % args.every == 0 or area == 0:
            assert not np.asarray(charge(cfg['n'])).any(), 'constraint violated: generator bug!'
            area, f2, topo = snapshot(cfg['n'])
            print(f'sweep {sweep:4d}: n² {nsq(cfg):5d}  area {area:5d}  F² {f2:5d}  {topo}   '
                  f'({(time.time() - t0) / sweep:.1f}s/sweep)', flush=True)
        if area == 0:
            melted_at = sweep
            break

    print(flush=True)
    if melted_at is not None:
        print(f'MELTED to the vacuum at sweep {melted_at}.', flush=True)
        if 'unknot' not in args.start and args.start != 'vacuum':
            print('A KNOTTED start reached F = 0: the trajectory itself is a constructive',
                  flush=True)
            print('unknotting path --- the move set demonstrably unknots (definitive).',
                  flush=True)
    else:
        area, f2, topo = snapshot(cfg['n'])
        print(f'Did NOT melt within {args.sweeps} sweeps: area {area}, F² {f2}, {topo}.',
              flush=True)
        print('A stall alone is NOT evidence of knotting: a greedy quench can jam for',
              flush=True)
        print('non-topological reasons.  Compare an ENSEMBLE of runs against the *-unknot',
              flush=True)
        print('control; if controls reliably melt and knotted starts reliably stall at an',
              flush=True)
        print('area floor, that is (statistical) evidence of a topological obstruction ---',
              flush=True)
        print('then inspect/census the remnant (corridors.py, genus.topology).', flush=True)
