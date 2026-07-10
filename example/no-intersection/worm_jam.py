#!/usr/bin/env python
r"""
Why every no-intersection worm was retired.

kappa is a temperature knob: SMALL kappa is the warm, dense, JAMMED phase, which lies
below kappa ~ 0.02.  A census that only samples kappa >= 0.1 measures outside the phase of
interest and draws the wrong conclusion.

Three measurements, on the same thermalized backgrounds:

  1. Clean sets.  The retired AdaptiveIntersectionWorm drew a DIRECTION first, then asked
     which of a fixed library advanced the head that way.  We reconstruct its clean set
     exactly -- it was never more than `_library[dd]` filtered by `_local_dq` -- and
     compare against FreeTargetWorm, which lets the transport go wherever the background
     permits.

  2. Composition.  FreeTargetWorm's clean union is ~90% IDLES, and its movers are
     overwhelmingly local two-link exact repairs.

  3. Transport.  The robust jam statistic for IntersectionWorm is the CLEAN-MOVER DRAW
     RATE, clean/drawn, read directly off its per-family `tallies` -- together with
     unclean/drawn for the multi-link families (ortho3, ortho2, same4), which sit at
     1.0000 at every kappa measured (those families never draw a usable shape at all).
     Measured: clean/drawn is <=0.08% at kappa=0.01, 0.50% at kappa=0.03, and ~1.00% at
     kappa=0.1.  This statistic integrates NO dwell, so it does not inherit the variance
     problem below.

     We also report the OFF-ORIGIN WEIGHT of the inline displacement histogram (the
     correlator itself) as a SECONDARY transport measurement, but for IntersectionWorm it
     is HIGH VARIANCE: that worm uses the Prokof'ev-Svistunov open/close branch, does NOT
     auto-close, and tallies head-tail dwell on EVERY iteration including rejected
     stay-puts (worm.py:730).  A single accepted mover therefore pins the head off-origin,
     and every subsequent rejected iteration inflates the off-origin numerator until the
     worm happens to close.  Measured at kappa=0.01 on four independent backgrounds,
     off-origin dwell read 0.00000, 0.00000, 0.00000, 0.21303 -- the outlier caused by
     exactly two accepted movers out of 2502 draws, which then contributed 533 stalled
     ticks.

     Worm_Length inherits the same defect for IntersectionWorm (it counts stalling, not
     walking), so it is not printed for that worm.  FreeTargetWorm is different: it DOES
     auto-close when the head returns to the tail, so for it ALONE frac(Worm_Length == 1)
     -- the fraction of worms that never moved -- is a LOW-VARIANCE transport statistic,
     and is the one we print.

Seeding.  --seed reproducibly drives the census heads (`census()`) and both worms' walks
in `transport()`.  It does NOT reach the thermalizing Hammer: ConstrainedLinkUpdate,
WrappingLoopUpdate, PlanarFluxUpdate, ScattershotUpdate, and DefectGas each construct
their own unseeded `np.random.default_rng()` in `__init__`, so the thermalized background
varies run to run even at fixed --seed.  This is an accepted limitation and is NOT fixed
here: the qualitative jam signature (clean/drawn near zero at small kappa, unclean/drawn
== 1.0000 for the multi-link families) is stable across every seed measured.

Background is not a confound.  Thermalizing with a Hammer roster that includes DefectGas
versus one that omits it gives the same sheet density at kappa=0.01 (|F|_1 = 7594-7748,
nnz(F) = 3298-3325) and the same clean-mover starvation.

Run:  .venv/bin/python example/no-intersection/worm_jam.py --kappa 0.01 0.03 0.1
"""
import argparse
import collections

import numpy as np

import supervillain
import supervillain.generator.no_intersection as gen
from supervillain.lattice import Lattice, d as _d


def adaptive_clean_set(worm, F, head, dd, sign):
    r"""
    The retired AdaptiveIntersectionWorm's clean set, reconstructed from the surviving
    IntersectionWorm.  Its `clean_set_local` was exactly this loop: place each library
    shape for the drawn direction, keep those whose local Delta q is precisely the head
    dipole {head: -1, target: +1}.
    """
    N = worm.Lattice.N
    target = tuple((head[k] + sign * dd[k]) % N for k in range(4))
    want = {head: -1, target: 1}
    anchor = tuple((head[k] + dd[k]) % N for k in range(4)) if sign > 0 else head
    seen, out = set(), []
    for shape in worm._library[dd]:
        change = worm._change_from_shape(head, dd, sign, shape)
        key = frozenset((lnk, c) for lnk, c in change.items() if c != 0)
        if key in seen:
            continue
        if worm._local_dq(F, change, anchor, shape) == want:
            seen.add(key)
            out.append(change)
    return out


def thermalize(kappa, N, steps, zeta):
    L = Lattice(4, N)
    S = supervillain.action.NoIntersections(L, kappa=kappa)
    H = gen.Hammer(S, zeta=zeta)
    return S, supervillain.Ensemble(S).generate(steps, H, start='cold')


def census(S, e, burn, stride, n_heads, rng):
    N = S.Lattice.N
    naive = gen.IntersectionWorm(S)
    free = gen.FreeTargetWorm(S)
    ortho = [dd for dd in naive._directions if sum(abs(x) for x in dd) == 1]

    adaptive, freemov = [], []
    idles = movers = 0
    links = collections.Counter()
    dist = collections.Counter()

    for i in range(burn, len(e.configuration), stride):
        F = np.asarray(_d(e.configuration[i]['n'])).astype(np.int64, copy=False)
        for _ in range(n_heads):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))

            mov = free.classified_set_local(F, head, classes='movers')
            freemov.append(len(mov))
            idles += len(free.classified_set_local(F, head, classes='idles'))
            movers += len(mov)
            for change, target in mov:
                links[len([c for c in change.values() if c])] += 1
                dist[sum(min(abs(target[k] - head[k]), N - abs(target[k] - head[k]))
                         for k in range(4))] += 1

            for dd in ortho:
                for sign in (+1, -1):
                    adaptive.append(len(adaptive_clean_set(naive, F, head, dd, sign)))

    return np.array(adaptive), np.array(freemov), idles, movers, links, dist


def transport(S, e, worms, seed):
    r"""Off-origin weight of the inline displacement histogram: the actual transport."""
    origin = S.Lattice.origin
    cfg = e.configuration[len(e.configuration) - 1]
    out = {}
    for i, (name, w) in enumerate((('IntersectionWorm', gen.IntersectionWorm(S)),
                                   ('FreeTargetWorm', gen.FreeTargetWorm(S)))):
        # Both worms' __init__ sets self.rng = np.random.default_rng() (unseeded); override
        # here so the headline transport measurement is reproducible.  Distinct seeds per
        # worm so the two do not walk in lockstep.
        w.rng = np.random.default_rng(seed + i)
        c, off, tot = cfg, 0.0, 0.0
        for _ in range(worms):
            c = w.step(c)
            h = np.asarray(c['IntersectionTwoPoint'])
            tot += h.sum()
            off += h.sum() - h[origin]
        out[name] = (off, tot, w.tallies, w)
    return out


def report(kappa, args, rng):
    S, e = thermalize(kappa, args.N, args.steps, args.zeta)
    print(f'=== kappa = {kappa}   (jammed phase is kappa < ~0.02) ===', flush=True)

    a, f, idles, movers, links, dist = census(
        S, e, args.burn, args.stride, args.heads, rng)
    for name, s in (('adaptive (retired)', a), ('free-target', f)):
        print(f'  |C| {name:20s} median={np.median(s):8.1f} mean={s.mean():9.2f} '
              f'frac(|C|=0)={np.mean(s == 0):.4f} max={s.max()}', flush=True)
    total = idles + movers
    if total:
        print(f'  free-target clean union: {idles} idles + {movers} movers '
              f'= {total}  ({100 * idles / total:.1f}% idles)', flush=True)
        print(f'    movers by link count:      {sorted(links.items())}', flush=True)
        print(f'    movers by L1 target dist:  {sorted(dist.items())}', flush=True)

    for name, (off, tot, tallies, w) in transport(S, e, args.worms, args.seed).items():
        # PRIMARY jam statistic: the clean-mover draw rate, straight off the per-family
        # tallies.  It integrates no dwell, so unlike off-origin dwell (below) it is not
        # inflated by a single accepted mover pinning the head away from the tail.
        drawn = sum(t['drawn'] for t in tallies.values())
        if drawn:
            unclean = sum(t['unclean'] for t in tallies.values())
            clean = sum(t['clean'] for t in tallies.values())
            accepted = sum(t['accepted'] for t in tallies.values())
            clean_rate = clean / drawn
            print(f'  {name:17s} clean/drawn (PRIMARY jam statistic) = {clean:.0f} / '
                  f'{drawn:.0f} = {clean_rate:.5f}', flush=True)
            print(f'    {"family":>8} {"drawn":>8} {"unclean":>8} {"clean":>8} '
                  f'{"accepted":>9} {"unclean/drawn":>15} {"clean/drawn":>13}', flush=True)
            for fam, t in tallies.items():
                u_over_d = t['unclean'] / t['drawn'] if t['drawn'] else float('nan')
                c_over_d = t['clean'] / t['drawn'] if t['drawn'] else float('nan')
                print(f'    {fam:>8} {t["drawn"]:>8} {t["unclean"]:>8} {t["clean"]:>8} '
                      f'{t["accepted"]:>9} {u_over_d:>15.4f} {c_over_d:>13.4f}', flush=True)
            u_rate = unclean / drawn
            print(f'    {"TOTAL":>8} {drawn:>8} {unclean:>8} {clean:>8} {accepted:>9} '
                  f'{u_rate:>15.4f} {clean_rate:>13.4f}', flush=True)

        # SECONDARY: off-origin weight of the inline displacement histogram (the
        # correlator itself).  For IntersectionWorm this is HIGH VARIANCE -- it does not
        # auto-close and tallies head-tail dwell on every iteration including rejected
        # stay-puts (worm.py:730), so one accepted mover pins the head off-origin for
        # many subsequent stalls.
        frac = off / tot if tot else float('nan')
        print(f'  {name:17s} transport (off-origin dwell; HIGH VARIANCE -- one accepted '
              f'mover pins the head off-origin for many stalls) = {off:.0f} / {tot:.0f} '
              f'= {frac:.5f}', flush=True)

        # Worm lengths are meaningful ONLY for FreeTargetWorm, which auto-closes: length
        # == 1 means zero transport (the pre-populated pivot dwell alone).  IntersectionWorm
        # does not auto-close and tallies head-tail dwell on every iteration including
        # rejected stay-puts (worm.py:730), so its Worm_Length counts stalling, not walking
        # -- printing it here would be meaningless, so we don't.
        if name == 'FreeTargetWorm':
            l = np.array(w.worm_lengths)
            print(f'    worm lengths: median={np.median(l):.1f}  mean={l.mean():.2f}  '
                  f'frac(len==1)={np.mean(l == 1):.4f}  max={int(l.max())}  worms={len(l)}',
                  flush=True)
    print(flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kappa', type=float, nargs='+', default=[0.01, 0.03, 0.1],
                   help='couplings to sweep; the jammed phase is below ~0.02')
    p.add_argument('--N', type=int, default=6, help='lattice extent (default 6)')
    p.add_argument('--zeta', type=float, default=0.025, help="the Hammer's fugacity")
    p.add_argument('--steps', type=int, default=100, help='thermalization steps')
    p.add_argument('--burn', type=int, default=70)
    p.add_argument('--stride', type=int, default=15)
    p.add_argument('--heads', type=int, default=40, help='random heads per configuration')
    p.add_argument('--worms', type=int, default=60, help='worms per transport measurement')
    p.add_argument('--seed', type=int, default=5,
                   help='seeds the census heads and both worms\' walks; does NOT reach the '
                        'thermalizing Hammer (ConstrainedLinkUpdate, WrappingLoopUpdate, '
                        'PlanarFluxUpdate, ScattershotUpdate, and DefectGas each construct '
                        'their own unseeded np.random.default_rng() in __init__), so the '
                        'thermalized background varies run to run')
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    for kappa in args.kappa:
        report(kappa, args, rng)
