#!/usr/bin/env python
r"""
Repeatable benchmark for sampling the No-Intersection model with its Hammer.

The point of this script is to give us a *durable, comparable* number for "how
long does it take to sample the No-Intersection action?" so that as we optimize
the generators we can measure how much each change actually mattered.

Two modes:

  * ``timing`` (default) --- times every generator in the Hammer, and the whole
    Hammer, at a few lattice sizes, with deterministic RNG seeding so the timing
    reflects *code* changes rather than proposal luck.  Results (plus git commit,
    platform, numpy version) are written to ``benchmarks/<commit>-<stamp>.json``
    and compared against the most recent previous run so you immediately see the
    speedup (or regression).

  * ``profile`` --- cProfile of a short Hammer run at one size, dumping the
    hottest functions by cumulative and total time.  This is where you read off
    that the global ``charge`` recompute (``_topological_charge`` -> ``wedge`` /
    ``d``) dominates.

Run from example/no-intersection/:

    uv run python benchmark.py                    # timing sweep, saves JSON
    uv run python benchmark.py --sizes 4 6        # pick lattice sizes
    uv run python benchmark.py profile --N 6      # find hotspots
"""

import argparse
import cProfile
import io
import json
import platform
import pstats
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import supervillain
from supervillain.lattice import Lattice
import supervillain.generator.villain as villain
import supervillain.generator.no_intersection as gen
from supervillain.generator.no_intersection.charge import charge

BENCH_DIR = Path(__file__).resolve().parent / 'benchmarks'

# The generators that make up the Hammer, constructed individually so we can time
# each one as it is actually used (one ``.step()`` call apiece).  Order and
# membership mirror supervillain.generator.no_intersection.Hammer; keep in sync.
def _build_generators(S):
    return [
        ('SiteUpdate',            villain.SiteUpdate(S)),
        ('ExactUpdate',           villain.ExactUpdate(S)),
        ('CohomologyUpdate',      villain.CohomologyUpdate(S)),
        ('ConstrainedLinkUpdate', gen.ConstrainedLinkUpdate(S)),
        ('WrappingLoopUpdate',    gen.WrappingLoopUpdate(S)),
        ('PlanarFluxUpdate',      gen.PlanarFluxUpdate(S)),
        ('ScattershotUpdate',     gen.ScattershotUpdate(S)),
        ('IntersectionWorm',      gen.IntersectionWorm(S)),
    ]


def _reseed(generators, seed=12345):
    r"""Pin every generator's RNG so proposal patterns are fixed across runs."""
    for i, (_, g) in enumerate(generators):
        g.rng = np.random.default_rng(seed + i)


def _cold(L):
    return {'phi': L.zeros(0), 'n': L.zeros(1)}


def _thermalize(S, steps, seed):
    r"""
    Run the full Hammer for ``steps`` steps from cold to build up a realistic
    fluxful background.  At low kappa this matters: the worm and the constrained
    link-update behave very differently on F != 0 than on the cold F = 0 vacuum,
    so timing them on a thermalized state is the honest measurement.
    """
    L = S.Lattice
    gens = _build_generators(S)
    _reseed(gens, seed)
    combined = gen.Hammer(S)
    combined.generators = tuple(g for _, g in gens)
    cfg = _cold(L)
    for _ in range(steps):
        cfg = combined.step(cfg)
    # Drop the inline-observable keys the worm adds so each generator starts clean.
    return {'phi': cfg['phi'], 'n': cfg['n']}


def _time_step(step_fn, cfg, steps, warmup):
    r"""Return (seconds_per_step, cfg) after ``warmup`` untimed then ``steps`` timed calls."""
    for _ in range(warmup):
        cfg = step_fn(cfg)
    t0 = time.perf_counter()
    for _ in range(steps):
        cfg = step_fn(cfg)
    elapsed = time.perf_counter() - t0
    return elapsed / steps, cfg


def _charge_microbench(L, repeats=20):
    r"""Cost of a single global ``charge`` recompute --- the per-proposal unit cost."""
    n = L.zeros(1)
    charge(n)  # warm
    t0 = time.perf_counter()
    for _ in range(repeats):
        charge(n)
    return (time.perf_counter() - t0) / repeats


def _metadata():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        commit = 'unknown'
    try:
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True).strip())
    except Exception:
        dirty = None
    return {
        'commit': commit,
        'dirty': dirty,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
    }


def timing(sizes, kappa, steps, warmup, seed, thermalize):
    meta = _metadata()
    print(f"== No-Intersection Hammer benchmark ==")
    print(f"commit {meta['commit']}{' (dirty)' if meta['dirty'] else ''}  "
          f"numpy {meta['numpy']}  python {meta['python']}")
    print(f"kappa={kappa}  steps/gen={steps}  warmup={warmup}  "
          f"thermalize={thermalize}  seed={seed}\n")

    results = {'meta': meta, 'kappa': kappa, 'steps': steps, 'warmup': warmup,
               'thermalize': thermalize, 'seed': seed, 'sizes': {}}

    for N in sizes:
        L = Lattice(4, N)
        volume = N ** 4
        S = supervillain.action.NoIntersections(L, kappa=kappa)

        q_cost = _charge_microbench(L)
        start = _thermalize(S, thermalize, seed) if thermalize else _cold(L)
        flux = int(np.abs(np.asarray(__import__('supervillain').lattice.d(start['n']))).sum())
        print(f"-- N={N}  (volume {volume} sites, {4 * volume} links) --")
        print(f"   charge() recompute: {q_cost * 1e3:.3f} ms/call   "
              f"thermalized sheet area Sigma|F| = {flux}")

        per_gen = {}
        # Time each generator individually from the (shared) thermalized start.
        for name, g in _build_generators(S):
            gens = [(name, g)]
            _reseed(gens, seed)
            cfg0 = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in start.items()}
            sps, _ = _time_step(g.step, cfg0, steps, warmup)
            per_gen[name] = sps
            print(f"   {name:24s} {sps * 1e3:9.3f} ms/step")

        # The whole Hammer as one combined step (this is what a sweep costs).
        hammer_gens = _build_generators(S)
        _reseed(hammer_gens, seed)
        combined = gen.Hammer(S)
        combined.generators = tuple(g for _, g in hammer_gens)
        hammer_steps = max(1, steps // 4)
        cfg0 = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in start.items()}
        sph, _ = _time_step(combined.step, cfg0, hammer_steps, max(1, warmup // 2))
        print(f"   {'Hammer (all)':24s} {sph * 1e3:9.3f} ms/step  "
              f"({1.0 / sph:.2f} steps/s)\n")

        results['sizes'][str(N)] = {
            'volume': volume,
            'charge_ms': q_cost * 1e3,
            'per_generator_ms': {k: v * 1e3 for k, v in per_gen.items()},
            'hammer_ms': sph * 1e3,
        }

    _save_and_compare(results)
    return results


def _save_and_compare(results):
    BENCH_DIR.mkdir(exist_ok=True)
    stamp = results['meta']['timestamp'].replace(':', '').replace('-', '')[:15]
    fn = BENCH_DIR / f"{results['meta']['commit']}-{stamp}.json"
    fn.write_text(json.dumps(results, indent=2))
    print(f"saved -> {fn.relative_to(Path.cwd()) if fn.is_relative_to(Path.cwd()) else fn}")

    previous = sorted(p for p in BENCH_DIR.glob('*.json') if p != fn)
    if not previous:
        print("(no previous benchmark to compare against)")
        return
    prev = json.loads(previous[-1].read_text())
    print(f"\ncompared to {previous[-1].name} (commit {prev['meta']['commit']}):")
    for N, cur in results['sizes'].items():
        if N not in prev.get('sizes', {}):
            continue
        old = prev['sizes'][N]['hammer_ms']
        new = cur['hammer_ms']
        speedup = old / new if new else float('nan')
        arrow = 'faster' if speedup >= 1 else 'SLOWER'
        print(f"   N={N}: Hammer {old:.2f} -> {new:.2f} ms/step  "
              f"({speedup:.2f}x {arrow})")


def profile(N, kappa, steps, seed):
    L = Lattice(4, N)
    S = supervillain.action.NoIntersections(L, kappa=kappa)
    hammer_gens = _build_generators(S)
    _reseed(hammer_gens, seed)
    combined = gen.Hammer(S)
    combined.generators = tuple(g for _, g in hammer_gens)
    cfg = _cold(L)

    # warm
    cfg = combined.step(cfg)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(steps):
        cfg = combined.step(cfg)
    pr.disable()

    print(f"== cProfile: {steps} Hammer steps at N={N}, kappa={kappa} ==\n")
    for sort_key, label in (('cumulative', 'by CUMULATIVE time'),
                            ('tottime', 'by TOTAL (self) time')):
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sort_key)
        ps.print_stats(15)
        print(f"--- {label} ---")
        print(s.getvalue())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode')

    pt = sub.add_parser('timing', help='timing sweep (default)')
    pt.add_argument('--sizes', type=int, nargs='+', default=[4, 6, 8])
    pt.add_argument('--kappa', type=float, default=0.1)
    pt.add_argument('--steps', type=int, default=8)
    pt.add_argument('--warmup', type=int, default=2)
    pt.add_argument('--thermalize', type=int, default=10,
                    help='Hammer steps from cold to build flux before timing (0 = cold)')
    pt.add_argument('--seed', type=int, default=12345)

    pp = sub.add_parser('profile', help='cProfile the Hammer to find hotspots')
    pp.add_argument('--N', type=int, default=6)
    pp.add_argument('--kappa', type=float, default=0.1)
    pp.add_argument('--steps', type=int, default=4)
    pp.add_argument('--seed', type=int, default=12345)

    args = p.parse_args()
    if args.mode == 'profile':
        profile(args.N, args.kappa, args.steps, args.seed)
    else:  # timing is the default
        if args.mode is None:
            args = pt.parse_args([])
        timing(args.sizes, args.kappa, args.steps, args.warmup, args.seed,
               args.thermalize)


if __name__ == '__main__':
    main()
