#!/usr/bin/env python
r"""
A machine-verified certificate that the EXISTING WrappingLoopUpdate unknots the knotted tori.

Theorem (one line, from the constraint-blindness of the x0-independent sector): if n is
x0-independent with only spatial components, then
q = dn ∧ dn ≡ 0 --- for ANY such n.  Any two such configurations differ by a sum of
**x0-wrapping axis rings** (add ±1 to n_i at one spatial position, at every x0) --- exactly
the axis-ring proposals of the existing WrappingLoopUpdate, with μ spatial and the wrapped
axis ν = x0 --- and applying the rings in ANY order keeps every intermediate configuration
x0-independent and purely spatial, hence exactly valid.  So every knotted torus of
torus_knotted.py is connected to the vacuum by Σ|m| such rings, each individually clean.
The knotted tori therefore lie in the existing library's reachability graph: for them
ergodicity is settled outright, and only the rate remains open.

Machine-verified: the trefoil torus (N = 8) dismantles to n ≡ 0 in Σ|m| = 51 rings, the
cinquefoil (N = 10) in 95, applied in random order with q ≡ 0 asserted after every single
move, and the generator's own tallies reporting proposed = clean = accepted = Σ|m|.

**Is it obvious that WrappingLoopUpdate performs these rings?**  Not by fiat --- so this
script does not ask you to trust a by-eye correspondence.  Each ring is applied by
calling ``WrappingLoopUpdate.step()`` itself, with the generator's ``rng`` replaced by a
``Rigged`` stand-in that feeds it one prescribed sequence of draws (see ``rig`` for the
draw-by-draw mapping onto ``_propose_loop``).  Everything else --- how the loop of links
is constructed from the draws, the Δq = 0 cleanliness verification, and the Metropolis
accept/reject --- is the generator's ordinary code path, executed at κ = 0 where a clean
proposal is accepted with probability exactly 1 (at κ > 0 the same proposals remain
clean and accept with probability e^{-ΔS} > 0, which is all reachability requires).  At
the end the generator's own tallies must show proposed = clean = accepted = Σ|m|.

The rings are applied in a RANDOM order (--seed) to demonstrate order-independence, the
constraint is asserted after every single step, and the final configuration is n ≡ 0:
the knot is undone without ever using knot theory, because the x0-independent sector is
a constraint-blind superhighway.

The script also quantifies why plain Monte Carlo does not find this quickly: an
UNRIGGED WrappingLoopUpdate draws a specific signed axis ring with probability
1/(48 N³) per step-call, so blindly stumbling through the certificate takes
~ 48 N³ · H(Σ|m|) proposals (~10⁵ at N = 8): a RATE (mixing) problem, never a
reachability problem.  Certificates prove reachability; Monte Carlo measures rates.

Limitation: the certificate exists because the tori live in the x0-independent sector.
The spun spheres of spun_sphere.py do not --- their rings span (x3, x0) --- so no such
free pass exists for them.  A by-hand certificate there is a genuine but *structured*
project.  Hosokawa–Kawauchi stabilization, plus the fact that a crossing change on the arc
lifts to a single 1-handle stabilization of the spun sphere, suggests the spun trefoil
unknots after attaching ONE tube in the right place (the classical trefoil has unknotting
number 1).  The plan: (1) attach a tube at the lift of the crossing-change site --- a few
links, validity machine-checked; (2) isotope and shrink the now-unknotted genus-1 surface
by cube moves, the laborious part, best done by guided search (best-first on Σn² with
bounded uphill) using the stabilization as a hand-placed waypoint; (3) dismantle the
standard remnant.  That is the sharp remaining unknotting target.

Run from example/no-intersection/:

    uv run python torus_dismantle.py [--N 8] [--grid 5] [--shift 2] [--scale 1] [--seed 0]
"""

import argparse

import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection.wrapping import WrappingLoopUpdate
import torus_knotted


class Rigged:
    """A stand-in for the generator's rng that feeds .step() one prescribed proposal.

    ``integers`` pops the queued draws in order (checking each is in the requested
    range); ``uniform`` returns its lower bound, so the generator's Metropolis
    comparison  uniform(0, 1) < min(1, e^{-ΔS})  accepts exactly when the acceptance
    probability is positive --- and at κ = 0 it is exactly 1.  Only the randomness is
    rigged; every line of move construction, verification, and acceptance that runs is
    WrappingLoopUpdate's own.
    """

    def __init__(self, draws):
        self.draws = list(draws)

    def integers(self, low, high=None):
        lo, hi = (0, low) if high is None else (low, high)
        v = self.draws.pop(0)
        assert lo <= v < hi, f'rigged draw {v} outside [{lo}, {hi})'
        return v

    def uniform(self, low=0.0, high=1.0):
        return low


def rig(mu, x1, x2, x3, s):
    """The draw sequence under which WrappingLoopUpdate._propose_loop constructs exactly
    the certificate ring  n_mu(x0, x1, x2, x3) += s  at every x0.

    Follow _propose_loop's rng calls in the order it makes them:

      mu = rng.integers(0, D)                      1st draw: mu --- one of our SPATIAL
                                                   directions 1, 2, 3.
      s = 1 if rng.integers(0, 2) == 0 else -1     2nd draw: 0 for s = +1, 1 for s = -1.
      ... rng.integers(0, 2) == 0: axis branch     3rd draw: 0 selects the AXIS-ring
                                                   branch (not the diagonal ring).
      nu = transverse[rng.integers(0, 3)]          4th draw: transverse = [a ≠ mu] =
                                                   [0, ...] because mu ≥ 1, so drawing 0
                                                   picks ν = x0: the ring WRAPS THE x0
                                                   AXIS.
      fixed = {a: rng.integers(0, N) for a ≠ ν}    5th-7th draws: a runs over 1, 2, 3 in
                                                   order (ν = 0 is excluded), fixing the
                                                   ring's spatial position (x1, x2, x3).
                                                   Note ``fixed`` includes a = mu: the
                                                   site coordinate along the link's own
                                                   direction is fixed too --- only x0
                                                   varies.

    _propose_loop then executes

        for t in range(N):
            site = (t if k == ν else fixed[k] ...)   = (t, x1, x2, x3)
            change[(mu,) + site] += s

    i.e. one +s on the direction-mu link at every x0 --- precisely the certificate ring
    arr[mu, :, x1, x2, x3] += s of the §4.9 theorem.
    """
    return [mu, 0 if s > 0 else 1, 0, 0, x1, x2, x3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--grid', type=int, default=5)
    parser.add_argument('--shift', type=int, default=2)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0, help='ring-order shuffle seed')
    args = parser.parse_args()

    L = Lattice(4, args.N)
    N = args.N
    n, steps, _ = torus_knotted.configuration(L, grid=args.grid, shift=args.shift,
                                              scale=args.scale)
    arr = np.asarray(n)

    # The spatial, x0-independent profile m (this is what makes the certificate work).
    assert (arr[0] == 0).all(), 'n must have no x0 component'
    m = arr[1:, 0]                       # m[i-1, x1, x2, x3]
    assert all((arr[i] == m[i - 1]).all() for i in (1, 2, 3)), 'n must be x0-independent'

    # One ring per unit of |m| on each occupied spatial link, in random order.  Each
    # entry (mu, x1, x2, x3, s) is the ring that undoes one unit of m there.
    rings = []
    for idx in np.argwhere(m != 0):
        i, x1, x2, x3 = (int(t) for t in idx)
        value = int(m[i, x1, x2, x3])
        rings += [(i + 1, x1, x2, x3, -int(np.sign(value)))] * abs(value)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rings)

    print(f'Knotted torus T({args.shift}, {args.grid - args.shift}) on N={N}: '
          f'Σ|m| = {len(rings)} wrapping rings to remove, in a random order (seed {args.seed}).')

    # κ = 0 makes the Metropolis factor exactly 1 for clean proposals, so the rigged rng
    # is the ONLY intervention.  (At κ > 0 the identical proposals stay clean and accept
    # with probability e^{-ΔS} > 0 --- reachability only needs positivity.)
    S = supervillain.action.NoIntersections(L, kappa=0.0)
    G = WrappingLoopUpdate(S)
    cfg = {'phi': L.zeros(0), 'n': n}

    for count, (mu, x1, x2, x3, s) in enumerate(rings, start=1):
        # Hand the generator the prescribed draws and let ITS OWN step() do the rest:
        # build the loop from the draws, recompute the global charge, verify the
        # proposal is clean on the current background, Metropolis-test, and apply.
        G.rng = Rigged(rig(mu, x1, x2, x3, s))
        cfg = G.step(cfg)
        # step() applies the move only if its own cleanliness check passed AND
        # Metropolis accepted, and it tallies both; the tally is the proof that it did.
        assert G.accepted == count, f'ring {count} was not accepted by the generator'
        assert not np.asarray(charge(cfg['n'])).any(), (
            f'intermediate {count} violated q = 0 (impossible: every intermediate is '
            f'x0-independent and purely spatial)')

    assert (np.asarray(cfg['n']) == 0).all(), 'did not reach the vacuum'
    assert G.proposed == G.clean == G.accepted == len(rings)
    print(f'DISMANTLED by WrappingLoopUpdate.step() itself: n ≡ 0, q ≡ 0 verified after')
    print(f'every move, and the generator tallies proposed = clean = accepted = {len(rings)}:')
    print(f'  {G.report()}')
    print()

    # Rate context for the UNRIGGED generator: one step-call draws one specific signed
    # axis ring with probability (1/4 directions) × (1/2 axis-vs-diagonal coin) ×
    # (1/3 transverse axes) × (1/N³ fixed coordinates) × (1/2 sign) = 1/(48 N³).
    p = 1 / (4 * 2 * 3 * N ** 3 * 2)
    H = float(np.sum(1 / np.arange(1, len(rings) + 1)))
    print(f'Rate context: unrigged, WrappingLoopUpdate draws a specific signed axis ring')
    print(f'with probability 1/(48 N³) = {p:.2e} per step-call, so blind Monte Carlo needs')
    print(f'~ 48 N³ · H({len(rings)}) ≈ {int(H / p):,} proposals to reproduce this certificate:')
    print(f'a mixing-rate problem, never a reachability problem.')
