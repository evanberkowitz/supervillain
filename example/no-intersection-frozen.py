#!/usr/bin/env python
r"""
Frozen configurations of the :class:`~supervillain.action.NoIntersections` model.

A *frozen* configuration is a valid configuration --- it obeys the no-intersection
constraint

    q = dn ∧ dn = 0   on every hypercube (D = 4)

--- yet has **zero** legal single-link ±1 moves: every ``n_ℓ → n_ℓ ± 1`` violates
q = 0 somewhere.  Such a configuration is a dead end for a naive
:class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate`, which only
proposes single-link changes; escaping it requires the *coordinated* moves of the
:class:`~supervillain.generator.no_intersection.WrappingLoopUpdate` or
:class:`~supervillain.generator.no_intersection.IntersectionWorm`.  This script builds
frozen configurations analytically and verifies both properties by brute force.

Because q is quadratic in n, a single link n_ℓ is "blocked" whenever the field
strength F = dn is nonzero in the planes *complementary* to the link's direction: the
cross term dΔn ∧ F + F ∧ dΔn then lights up a defect.  A configuration is frozen when
every link is blocked in this way, which needs F nonzero (with no gaps) in enough
planes at once.  Two constructions achieve that:

single-pair  (``--construction single-pair``)
    Put the flux in one complementary pair of planes, e.g. {01, 23}:

        F_01(x) = a (-1)^{x_0}                 (depends on x_0 only)
        F_23(x) = b (-1)^{x_0+x_2+x_3}         (anti-periodic under the e_0+e_2 shift)

    q vanishes *pair-by-pair*: the shift e_μ+e_ν that relates the two wedge terms
    flips (-1)^{x_0}, so the two contributions are equal and opposite everywhere,
    for any a, b.  ``--pair`` selects which pair {01,23}, {02,13}, or {03,12}.

six-plane  (``--construction six-plane``)
    Put flux in *all six* planes with a constant antisymmetric matrix A:

        F_{μν}(x) = A_{μν} (-1)^{x_μ + x_ν}    ⇒   q(x) = 2 (-1)^{Σx} Pf(A),

    where Pf(A) = A_01 A_23 - A_02 A_13 + A_03 A_12.  Now q = 0 everywhere iff
    Pf(A) = 0, and the cancellation is genuinely *inter-pair*: each pair contributes
    ±2 A_{μν}A_{ρσ}(-1)^{Σx} ≠ 0 and the three contributions sum to zero.  This is not
    a superposition of single-pair configs --- here even F_01 depends on two
    coordinates (x_0, x_1), not one.

Both families are infinite (scale a, b, or move along the Pf(A) = 0 quadric in Z^6)
and neither exhausts the frozen set; they are the simplest closed forms we know.

search  (``--construction search``)
    Rather than *construct* a frozen configuration, *look* for one dynamically.  From a
    cold start (n = 0) at κ = 0 --- where the action is flat so every constraint-preserving
    single-link move is accepted --- run the library
    :class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate` for ``--sweeps``
    sweeps and watch for it to get stuck (a whole sweep accepting nothing), confirming any
    candidate with the exhaustive check.  In practice it never freezes: the cold-start
    single-link walk does not reach these configurations --- exactly the observation noted
    in :class:`~supervillain.generator.no_intersection.WrappingLoopUpdate`.
"""

import argparse
import time

import numpy as np

import supervillain
from supervillain.lattice import Lattice, d
# q = dn ∧ dn, the topological-charge density; the same quantity the library
# measures as TopologicalChargeDensity and that NoIntersections.valid() checks.
from supervillain.generator.no_intersection.charge import charge


# ───────────────────────────────────────────────────────── shared verification

def f_plane_summary(F_arr, D):
    """One line per 2-form component F_{μν}: the set of values it takes and max|F|.

    A frozen construction wants every relevant plane nonzero *with no gaps*; this
    summary lets you eyeball whether that holds (for odd N a few zeros creep in from
    the wrap-around plaquette, which is why odd N behaves slightly differently).
    """
    rows = []
    k = 0
    # The 2-form components are stored in lexicographic (μ<ν) order: 01,02,03,12,13,23.
    for mu in range(D):
        for nu in range(mu + 1, D):
            mx = int(np.abs(F_arr[k]).max())
            rows.append(f"  F_{mu}{nu}: values in {sorted(set(F_arr[k].flat))}, max|F|={mx}")
            k += 1
    return "\n".join(rows)


def exhaustive_check(n, shifts=(1, -1)):
    """Try every (link, ±1) move and count how many keep q = 0.

    Returns ``(n_valid, n_total, blocked_per_dir)``.  A link is "blocked" when
    *neither* +1 nor -1 preserves the constraint; the configuration is frozen iff
    ``n_valid == 0``.  We mutate a single scratch copy in place and undo each trial
    rather than allocating a fresh array per move.
    """
    L = n.lattice
    D, N = L.D, L.N

    trial = n.copy()
    arr = np.asarray(trial)          # a mutable view of the scratch 1-form

    n_valid = 0
    n_total = 0
    blocked_per_dir = [0] * D

    for mu in range(D):
        for site in np.ndindex(*((N,) * D)):
            link = (mu,) + site      # (direction, x0, x1, x2, x3)
            link_blocked = True
            for c in shifts:
                arr[link] += c                       # propose n_ℓ → n_ℓ + c
                if np.allclose(charge(trial), 0):    # does it still satisfy q = 0?
                    n_valid += 1
                    link_blocked = False
                arr[link] -= c                       # undo, leaving `trial` == `n`
                n_total += 1
            if link_blocked:
                blocked_per_dir[mu] += 1

    return n_valid, n_total, blocked_per_dir


# ───────────────────────────────────────────────── construction 1: single pair

def build_single_pair(L, a=1, b=1, pair='01-23'):
    r"""Integer 1-form n whose flux lives in one complementary plane pair.

      '01-23':  F_01 = a(-1)^{x_0},  F_23 = b(-1)^{x_0+x_2+x_3}
      '02-13':  F_02 = a(-1)^{x_0},  F_13 = b(-1)^{x_0+x_1+x_3}
      '03-12':  F_03 = a(-1)^{x_0},  F_12 = b(-1)^{x_0+x_1+x_2}

    q vanishes pair-by-pair for any nonzero integers a, b (see the module docstring).
    """
    N = L.N
    n = L.zeros(1, dtype=int)
    arr = np.asarray(n)              # shape (D, N, N, N, N): arr[mu, x0, x1, x2, x3]

    # For each pair we set exactly two link components.  The first gives the "simple"
    # F_μν = a(-1)^{x_0}: an integer 1-form whose only nonzero links alternate with
    # x_0 does the job, because n_ν = a·(x_0 mod 2) has dn in the (0,ν) plane equal to
    # a(-1)^{x_0}.  The second, staggered, component supplies the anti-periodic partner
    # in the complementary plane so that the two wedge terms cancel.
    if pair == '01-23':
        for x0 in range(N):
            arr[1, x0, :, :, :] = a * (x0 % 2)                 # → F_01 = a(-1)^{x_0}
        for x0 in range(N):
            for x3 in range(N):
                sign = (-1) ** (x0 + x3)
                for x2 in range(N):
                    arr[3, x0, :, x2, x3] = b * sign * (x2 % 2)  # → F_23 = b(-1)^{x_0+x_2+x_3}

    elif pair == '02-13':
        for x0 in range(N):
            arr[2, x0, :, :, :] = a * (x0 % 2)                 # → F_02 = a(-1)^{x_0}
        for x0 in range(N):
            for x3 in range(N):
                sign = (-1) ** (x0 + x3)
                for x1 in range(N):
                    arr[3, x0, x1, :, x3] = b * sign * (x1 % 2)  # → F_13 = b(-1)^{x_0+x_1+x_3}

    elif pair == '03-12':
        for x0 in range(N):
            arr[3, x0, :, :, :] = a * (x0 % 2)                 # → F_03 = a(-1)^{x_0}
        for x0 in range(N):
            for x2 in range(N):
                sign = (-1) ** (x0 + x2)
                for x1 in range(N):
                    arr[2, x0, x1, x2, :] = b * sign * (x1 % 2)  # → F_12 = b(-1)^{x_0+x_1+x_2}

    else:
        raise ValueError(f"Unknown pair {pair!r}; choose '01-23', '02-13', or '03-12'")

    return n


# ────────────────────────────────────────────── construction 2: six-plane / Pf(A)

def pfaffian(A):
    """Pf(A) = A_01 A_23 - A_02 A_13 + A_03 A_12 for the antisymmetric matrix A.

    q(x) = 2(-1)^{Σx} Pf(A) for the six-plane ansatz, so Pf(A) = 0 ⇔ q ≡ 0.
    """
    return A[(0, 1)] * A[(2, 3)] - A[(0, 2)] * A[(1, 3)] + A[(0, 3)] * A[(1, 2)]


def build_six_plane(L, A):
    r"""Integer 1-form n realising F_{μν}(x) = A_{μν} (-1)^{x_μ + x_ν}.

    ``A`` is a dict ``{(μ, ν): int}`` for μ < ν in {0,1,2,3}.  Requires Pf(A) = 0
    for q = 0.  The 1-form is the unique integral of F in the "triangular" gauge
    n_0 = 0, obtained by peeling off one plane at a time (n_ν absorbs the A_{μν} for
    all μ < ν).
    """
    N = L.N
    n = L.zeros(1, dtype=int)
    arr = np.asarray(n)              # shape (4, N, N, N, N): arr[mu, x0, x1, x2, x3]

    # Coordinate grids x0..x3, each of shape (N,N,N,N), so the assignments below are
    # fully vectorised.  (x_i % 2) is the "staggered step" whose lattice derivative in
    # direction i is (-1)^{x_i}; the (-1)^{x_ν} prefactor orients the plane.
    x0, x1, x2, x3 = np.meshgrid(range(N), range(N), range(N), range(N), indexing='ij')

    # n_0 = 0 (gauge).  Then integrate F plane by plane:
    arr[1] = A[(0, 1)] * (-1) ** x1 * (x0 % 2)                       # carries F_01
    arr[2] = (A[(0, 2)] * (x0 % 2) + A[(1, 2)] * (x1 % 2)) * (-1) ** x2   # F_02, F_12
    arr[3] = (A[(0, 3)] * (x0 % 2) + A[(1, 3)] * (x1 % 2)
              + A[(2, 3)] * (x2 % 2)) * (-1) ** x3                   # F_03, F_13, F_23
    return n


def pair_contributions(F_arr):
    """Each complementary pair's signed contribution to q, with q = C_0123 + C_0213 + C_0312.

    For the six-plane construction all three are nonzero everywhere yet sum to zero:
    the "delicate", genuinely inter-pair cancellation.  ``F_arr`` has shape
    (6, N, N, N, N) with planes ordered 01,02,03,12,13,23; its spatial axes are
    0=x_0 … 3=x_3.
    """
    F01, F02, F03, F12, F13, F23 = (F_arr[k] for k in range(6))

    def s(f, *axes):
        # Shift a component forward by one site along each named axis: f(x + Σ e_axis).
        for ax in axes:
            f = np.roll(f, -1, ax)
        return f

    # The lattice (F ∧ F)_{0123} pairs each plane with its complement, with the sign
    # of the ε_{0123} permutation: +1 for (01)(23) and (03)(12), −1 for (02)(13).
    C_0123 = F01 * s(F23, 0, 1) + F23 * s(F01, 2, 3)
    C_0213 = -(F02 * s(F13, 0, 2) + F13 * s(F02, 1, 3))
    C_0312 = F03 * s(F12, 0, 3) + F12 * s(F03, 1, 2)
    return C_0123, C_0213, C_0312


# ─────────────────────────────────────────────────────────────────────── report

def report(n, extra=None):
    """Print the F-plane summary, verify q = 0, and run the exhaustive frozen check.

    ``extra`` is an optional callback (used by the six-plane construction) to print
    the pair-by-pair decomposition once q has been confirmed zero.
    """
    L = n.lattice
    D, N = L.D, L.N

    F_arr = np.asarray(d(n))
    print("F-plane summary:")
    print(f_plane_summary(F_arr, D))
    print()

    Q_arr = np.asarray(charge(n))
    q_is_zero = np.allclose(Q_arr, 0)
    print(f"q = dn∧dn:  min={Q_arr.min():.0f}  max={Q_arr.max():.0f}  →  q=0: {q_is_zero}")
    print()
    if not q_is_zero:
        print("ERROR: q ≠ 0.  The construction is invalid for these parameters.")
        return

    if extra is not None:
        extra(F_arr)

    # The payoff: does *any* single-link ±1 move preserve the constraint?
    print("Exhaustive single-link ±1 check …")
    t0 = time.time()
    n_valid, n_total, blocked = exhaustive_check(n)
    dt = time.time() - t0
    for mu in range(D):
        print(f"  dir {mu}: {blocked[mu]}/{N ** D} links fully blocked")
    print()
    print(f"{n_valid}/{n_total} valid moves  ({dt:.1f}s)")
    print()
    if n_valid == 0:
        print("FROZEN CONFIGURATION CONFIRMED: no single-link ±1 move preserves q = 0.")
    else:
        print(f"NOT frozen: {n_valid} valid single-link moves remain.")


# ─────────────────────────────────────────────── construction 3: dynamical search

# The six 2-form planes in lexicographic (μ<ν) order, matching d(n)'s components.
_PLANES = [(mu, nu) for mu in range(4) for nu in range(mu + 1, 4)]


def search(L, sweeps, probe_every=200):
    """Look for a frozen configuration by letting ConstrainedLinkUpdate wander.

    At κ = 0 the action is flat, so :class:`~.ConstrainedLinkUpdate` accepts *every*
    constraint-preserving single-link move and random-walks freely over the surface
    q = 0.  Should it ever reach a frozen configuration it must stall --- an entire
    sweep accepts nothing --- so we run the exhaustive check whenever a sweep stalls,
    and also periodically (every ``probe_every`` sweeps) once at least two F-planes are
    lit, since a frozen config needs flux in complementary planes.
    """
    S = supervillain.action.NoIntersections(L, kappa=0.0)   # κ=0 ⇒ accept every valid move
    G = supervillain.generator.no_intersection.ConstrainedLinkUpdate(S)

    # Cold start: n = 0 is trivially valid (F = dn = 0, so q = 0).
    cfg = {'phi': L.zeros(0), 'n': L.zeros(1, dtype=int)}

    prev_accepted = G.accepted          # the generator tallies accepted moves cumulatively
    for sweep in range(1, sweeps + 1):
        cfg = G.step(cfg)
        this_accepted = G.accepted - prev_accepted
        prev_accepted = G.accepted

        # Which F-planes currently carry flux?
        F_arr = np.asarray(d(cfg['n']))
        lit = [f'{mu}{nu}' for k, (mu, nu) in enumerate(_PLANES)
               if np.abs(F_arr[k]).max() > 0]

        # Only pay for the exhaustive check when it might matter.
        stalled = (this_accepted == 0)
        if not (stalled or (len(lit) >= 2 and sweep % probe_every == 0)):
            continue

        n_valid, n_total, _ = exhaustive_check(cfg['n'])
        if n_valid == 0:
            print(f"sweep {sweep:5d}: *** FROZEN *** — 0/{n_total} valid single-link moves")
            print(f"  F-planes lit: {lit}")
            return cfg['n']

        # A stalled sweep with valid moves still available just means the random
        # proposals happened to miss them this pass --- not frozen.
        if stalled:
            print(f"sweep {sweep:5d}: sweep accepted 0, but {n_valid}/{n_total} valid moves "
                  f"exist (proposals missed them); planes {set(lit)}")
        elif sweep % probe_every == 0:
            print(f"sweep {sweep:5d}: {len(lit)} F-planes {set(lit)}, "
                  f"{n_valid}/{n_total} valid moves, {this_accepted} accepted this sweep")

    print(f"\nNo frozen configuration found in {sweeps} sweeps "
          "(the cold-start single-link walk does not reach one).")
    return None


# ────────────────────────────────────────────────────────────────────────── main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--construction', choices=['single-pair', 'six-plane', 'search'],
                        default='single-pair',
                        help='build a single-pair or six-plane frozen config, or dynamically '
                             'search for one (default: single-pair)')
    parser.add_argument('--N', type=int, default=4, metavar='N',
                        help='lattice size (default: 4; must be ≥ 2; even N gives zero-free F)')
    parser.add_argument('--sweeps', type=int, default=5000, metavar='S',
                        help='[search] number of ConstrainedLinkUpdate sweeps (default: 5000)')

    # single-pair parameters
    parser.add_argument('--a', type=int, default=1, metavar='A',
                        help='[single-pair] nonzero integer scale of the first F-plane (default: 1)')
    parser.add_argument('--b', type=int, default=1, metavar='B',
                        help='[single-pair] nonzero integer scale of the second F-plane (default: 1)')
    parser.add_argument('--pair', default='01-23', choices=['01-23', '02-13', '03-12'],
                        help='[single-pair] complementary plane pair (default: 01-23)')

    # six-plane parameters (defaults give Pf(A) = 1·1 - 2·1 + 1·1 = 0 with all A_{μν} ≠ 0)
    parser.add_argument('--a01', type=int, default=1, metavar='INT', help='[six-plane] A_01')
    parser.add_argument('--a02', type=int, default=2, metavar='INT', help='[six-plane] A_02')
    parser.add_argument('--a03', type=int, default=1, metavar='INT', help='[six-plane] A_03')
    parser.add_argument('--a12', type=int, default=1, metavar='INT', help='[six-plane] A_12')
    parser.add_argument('--a13', type=int, default=1, metavar='INT', help='[six-plane] A_13')
    parser.add_argument('--a23', type=int, default=1, metavar='INT', help='[six-plane] A_23')

    args = parser.parse_args()
    if args.N < 2:
        parser.error('N must be ≥ 2')

    D, N = 4, args.N                 # the model (and q = dn∧dn) only makes sense in D = 4
    L = Lattice(D, N)
    print(f"Lattice D={D}, N={N}  ({D * N ** D} links, {N ** D} hypercubes)")

    if args.construction == 'search':
        print(f"search:  cold start, κ=0, {args.sweeps} ConstrainedLinkUpdate sweeps\n")
        search(L, args.sweeps)

    elif args.construction == 'single-pair':
        if args.a == 0 or args.b == 0:
            parser.error('a and b must be nonzero')
        print(f"single-pair:  a={args.a}, b={args.b}, pair={args.pair}\n")
        n = build_single_pair(L, a=args.a, b=args.b, pair=args.pair)
        report(n)

    else:  # six-plane
        # Assemble the antisymmetric matrix A and require Pf(A) = 0 up front.
        A = {
            (0, 1): args.a01, (0, 2): args.a02, (0, 3): args.a03,
            (1, 2): args.a12, (1, 3): args.a13, (2, 3): args.a23,
        }
        pf = pfaffian(A)
        print("six-plane:  " + "  ".join(f"A_{mu}{nu}={v}" for (mu, nu), v in A.items()))
        print(f"Pf(A) = A01·A23 - A02·A13 + A03·A12 = "
              f"{A[(0,1)]}·{A[(2,3)]} - {A[(0,2)]}·{A[(1,3)]} + {A[(0,3)]}·{A[(1,2)]} = {pf}")
        print()
        if pf != 0:
            parser.error(f'Pf(A) = {pf} ≠ 0; the six-plane construction requires Pf(A) = 0')

        # Genuine "delicate cancellation" needs every plane nonzero; warn otherwise.
        zero_entries = [f'A_{mu}{nu}' for (mu, nu), v in A.items() if v == 0]
        if zero_entries:
            print(f"WARNING: zero entries {zero_entries} — those F-planes vanish, so the "
                  "cancellation is not genuinely six-plane.\n")

        n = build_six_plane(L, A)

        def show_pairs(F_arr):
            # Confirm each pair contributes nonzero q yet the three sum to zero.
            C = pair_contributions(F_arr)
            print("Pair-by-pair q contributions:")
            for name, Ck in zip(('{01,23}', '{02,13}', '{03,12}'), C):
                print(f"  pair {name}: nonzero at {np.count_nonzero(Ck)}/{N ** 4} sites, "
                      f"values {sorted(set(Ck.flat))}")
            print(f"  sum (= q): max|residual| = {np.abs(sum(C)).max():.0f}")
            if all(np.any(Ck != 0) for Ck in C) and not zero_entries:
                print("  → delicate cancellation: every pair nonzero, cancelling globally "
                      "(not pair-by-pair).")
            print()

        report(n, extra=show_pairs)
