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

  3. Transport.  Measured as the OFF-ORIGIN WEIGHT of the inline displacement histogram --
     which is the correlator itself.  NOT Worm_Length: the IntersectionWorm tallies dwell
     on every iteration including rejected stay-puts (worm.py:730) and does not auto-close,
     so its "length" counts stalling.  Only FreeTargetWorm auto-closes, so for it alone
     Worm_Length == 1 means zero transport.

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


def transport(S, e, worms):
    r"""Off-origin weight of the inline displacement histogram: the actual transport."""
    origin = S.Lattice.origin
    cfg = e.configuration[len(e.configuration) - 1]
    out = {}
    for name, w in (('IntersectionWorm', gen.IntersectionWorm(S)),
                    ('FreeTargetWorm', gen.FreeTargetWorm(S))):
        c, off, tot = cfg, 0.0, 0.0
        for _ in range(worms):
            c = w.step(c)
            h = np.asarray(c['IntersectionTwoPoint'])
            tot += h.sum()
            off += h.sum() - h[origin]
        out[name] = (off, tot, w.tallies)
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

    for name, (off, tot, tallies) in transport(S, e, args.worms).items():
        frac = off / tot if tot else float('nan')
        print(f'  transport {name:17s} off-origin dwell = {off:.0f} / {tot:.0f} '
              f'= {frac:.5f}', flush=True)
        drawn = sum(t['drawn'] for t in tallies.values())
        if drawn:
            print(f'    {"family":>8} {"drawn":>8} {"unclean":>8} {"clean":>8} {"accepted":>9}',
                  flush=True)
            for fam, t in tallies.items():
                u_over_d = t['unclean'] / t['drawn'] if t['drawn'] else float('nan')
                print(f'    {fam:>8} {t["drawn"]:>8} {t["unclean"]:>8} {t["clean"]:>8} '
                      f'{t["accepted"]:>9}   unclean/drawn={u_over_d:.4f}', flush=True)
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
    p.add_argument('--seed', type=int, default=5)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    for kappa in args.kappa:
        report(kappa, args, rng)
