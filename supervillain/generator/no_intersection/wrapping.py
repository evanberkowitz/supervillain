#!/usr/bin/env python

import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import charge

import logging
logger = logging.getLogger(__name__)


class WrappingLoopUpdate(ReadWriteable, Generator):
    r"""
    A *global*, coordinated change of $F = dn$ that preserves the
    $q = dn\wedge dn = 0$ constraint by adding a **closed, torus-wrapping loop of
    single-direction links** to $n$ — proposed and accepted/rejected **atomically**
    (one Metropolis test on the whole loop), not built up one cell at a time.

    Why a single link direction.
    -----------------------------
    Confine the change to one link direction $\mu$ (only $n_\mu$ changes).  Then
    $d\Delta n$ has nonzero components only in the planes $\{\mu\nu\}$, so the self-term
    $d\Delta n\wedge d\Delta n$ vanishes *identically* — no two complementary planes are
    ever both present (the same fact that makes a single link's self-term vanish, lifted
    to an arbitrary single-direction $\Delta n$).  Hence on **any** background $F$,

    .. math::
        \Delta q(\Delta n) = F\wedge d\Delta n + d\Delta n\wedge F
        \qquad\text{(exactly linear in }\Delta n\text{)} .

    A single-direction $\Delta n$ that preserves the constraint is therefore exactly a
    vector in the kernel of this linear map; the frozen configurations are *not*
    degree-zero vertices once such coordinated moves are allowed.

    Why a closed, wrapping loop.
    ----------------------------
    The endpoint of a string of $n_\mu$ links is a place where the attached $F$-sheet has
    a free edge — a $q\ne 0$ source (this is precisely the worm head/tail).  A move with
    $\Delta q = 0$ everywhere can therefore have **no endpoints**: it must be a closed
    cycle.  On a generic $F\ne 0$ background (e.g. a frozen configuration) a *contractible*
    loop still leaks $\Delta q\ne 0$ where its interior meets the background $F$, so only
    **non-contractible** (torus-wrapping) loops survive.  On an $F = 0$ background the
    cross term vanishes and *every* single-direction loop is clean, so there this update
    freely deposits wrapping $F$-sheets — the genuine $F\ne 0$, $F\wedge F = 0$ moves that
    the worm alone must otherwise supply.

    Why atomic rather than incremental.
    -----------------------------------
    One might hope to walk the loop down cell by cell, accepting each segment in the
    constraint-lifted ($q\ne 0$) ensemble — i.e. as a worm.  But on a frozen background
    the clean continuation at each step is **forced** (the cross term with $F$ pins the
    loop's shape: a $(1,1)$ diagonal is clean, a turn or the anti-diagonal relights
    defects).  With no branching, the head's walk is a one-dimensional forced path, and
    the probability of completing the loop incrementally is governed by the *same*
    Boltzmann factor $e^{-\Delta S}$ as accepting the whole loop at once.  The incremental
    dressing buys nothing where there is no choice to make, so this update proposes the
    entire loop and tests it once.  (Where the background *does* offer branching — generic
    configurations — the incremental worm :class:`IntersectionWorm` is the appropriate tool; the
    two are complementary.)

    Proposal and detailed balance.
    ------------------------------
    Each step draws, from a **state-independent** distribution, a closed single-direction
    loop: a direction $\mu$, a loop type (a thin ring wrapping one transverse axis, or a
    thin diagonal ring wrapping two), random offsets locating it, and a sign $s = \pm 1$.
    The constraint is then **verified** ($\Delta q = 0$ on the current $n$); clean
    proposals are Metropolis-tested against the Villain action, unclean ones are null
    moves (the configuration is unchanged).

    The proposal is symmetric: drawing loop $L$ and drawing $-L$ have equal probability
    (the sign is uniform), and — because $d\Delta n\wedge d\Delta n = 0$ for a
    single-direction $\Delta n$ —

    .. math::
        \Delta q_{n'}(-L) = -\bigl(F\wedge dL + dL\wedge F + 2\,dL\wedge dL\bigr) = -\Delta q_n(L),
        \qquad n' = n + L,

    so $L$ is clean on $n$ **iff** $-L$ is clean on $n' = n + L$, on any background.  The
    forward and reverse proposals are thus drawn with equal probability and are
    simultaneously available, and the plain acceptance $\min(1, e^{-\Delta S})$ — with
    $\Delta S$ the change in the Villain action $\frac{\kappa}{2}(d\phi - 2\pi n)^2$ —
    satisfies detailed balance.  (This is the same symmetry argument as the one
    uniformly-chosen shape in :class:`IntersectionWorm`.)

    .. warning::

        Restricted to $D = 4$.  Updates $n$ only; combine with a $\phi$-update such as
        :class:`~.villain.SiteUpdate`.  The single-direction trick is dimension-general,
        but the $q = dn\wedge dn$ constraint is specific to $D = 4$.

    .. note::

        The constraint is verified by a global ``charge`` recompute per proposal
        ($O(\text{volume})$); only the touched links contribute to $\Delta S$.  A local
        $\Delta q$ check would be cheaper but is left for a production version, matching
        the reference-implementation choices in :class:`IntersectionWorm` and
        :class:`ConstrainedLinkUpdate`.

    .. note::

        **Scope of validity vs. efficiency.**  From a cold $F = 0$ start this update
        builds valid wrapping $F$-sheets ($F\ne 0$, $F\wedge F = 0$) while keeping
        $q = 0$ at every step — its intended job — and on an (injected) frozen
        configuration it does accept coordinated loops that single-link and worm moves
        cannot, with $q = 0$ throughout.  It is *not*, however, an efficient un-freezer on
        its own: the **uniform-random** loop proposal mostly finds *sideways* clean loops
        (translates of sheets already present) and only rarely the specific
        flux-*cancelling* loop, so the chain mixes poorly on the frozen sublattice and the
        total flux $\sum F^2$ does not reliably come down.  Deterministically selecting
        flux-reducing kernel moves (a greedy descent) *does* walk a frozen configuration
        back to the mobile sector; reproducing that with a symmetric, state-independent
        proposal — or carrying the Hastings ratio for a state-dependent one — is the open
        efficiency item, exactly analogous to enriching :class:`IntersectionWorm`'s move library.
        (In a cold-start physics run the frozen configurations are unreachable anyway, so
        this update's correctness on the $F = 0$ sector is what matters in practice.)
    """

    def __init__(self, S, diagonal=True):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('WrappingLoopUpdate requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('WrappingLoopUpdate is only implemented for D = 4.')

        self.Action = S
        self.Lattice = S.Lattice
        self.kappa = S.kappa
        self.rng = np.random.default_rng()

        # Whether to also propose two-axis diagonal rings (needed to escape some
        # frozen configurations) in addition to single-axis rings.
        self.diagonal = diagonal

        self.proposed = 0       # all proposals
        self.clean = 0          # proposals that preserved q (Δq = 0)
        self.accepted = 0       # clean proposals that passed Metropolis

    def __str__(self):
        return 'WrappingLoopUpdate'

    def inline_observables(self, steps):
        return {}

    def _propose_loop(self):
        r"""
        A closed, single-direction, torus-wrapping loop drawn from a state-independent
        distribution.  Returns a dict ``link -> coefficient`` ($\Delta n$).

        Two loop types, both thin (so $d\Delta n \ne 0$) and closed (so they carry no
        endpoint/defect):

        * **axis ring** — $n_\mu \mathrel{+}= s$ along a full ring in one transverse axis
          $\nu$, at random fixed values of the other three site coordinates;
        * **diagonal ring** — $n_\mu \mathrel{+}= s$ along a $(1,\pm1)$ diagonal that
          wraps two transverse axes simultaneously, at a random offset and random fixed
          value of the third transverse axis.
        """
        D = self.Lattice.D
        N = self.Lattice.N
        rng = self.rng

        mu = int(rng.integers(0, D))
        transverse = [a for a in range(D) if a != mu]
        s = 1 if rng.integers(0, 2) == 0 else -1

        change = {}
        if not self.diagonal or rng.integers(0, 2) == 0:
            # axis ring wrapping a single transverse axis
            nu = transverse[int(rng.integers(0, len(transverse)))]
            fixed = {a: int(rng.integers(0, N)) for a in range(D) if a != nu}
            for t in range(N):
                site = tuple(t if k == nu else fixed[k] for k in range(D))
                link = (mu,) + site
                change[link] = change.get(link, 0) + s
        else:
            # diagonal ring wrapping two transverse axes
            perm = rng.permutation(len(transverse))
            nu, rho = transverse[int(perm[0])], transverse[int(perm[1])]
            delta = 1 if rng.integers(0, 2) == 0 else -1
            offset = int(rng.integers(0, N))
            fixed = {a: int(rng.integers(0, N))
                     for a in range(D) if a not in (nu, rho)}
            for t in range(N):
                coords = dict(fixed)
                coords[nu] = t
                coords[rho] = (delta * t + offset) % N
                site = tuple(coords[k] for k in range(D))
                link = (mu,) + site
                change[link] = change.get(link, 0) + s
        return change

    def _delta_S(self, dphi, n, change):
        r"""Change in the Villain action from adding ``change`` to $n$ (touched links only)."""
        total = 0.0
        for link, c in change.items():
            A = dphi[link] - 2 * np.pi * n[link]
            total += (self.kappa / 2) * ((A - 2 * np.pi * c) ** 2 - A ** 2)
        return total

    def step(self, configuration):
        r"""
        Propose one closed single-direction wrapping loop, verify it preserves
        $q = dn\wedge dn = 0$, and Metropolis-test it against the Villain action.
        """
        L = self.Lattice
        n = configuration['n'].copy()
        dphi = d(configuration['phi'])
        q_now = charge(n)

        self.proposed += 1
        change = self._propose_loop()

        trial = n.copy()
        for link, c in change.items():
            trial[link] += c

        # Verify the loop is clean (Δq = 0) on THIS background; else a null move.
        if not np.array_equal(charge(trial), q_now):
            return configuration | {'n': configuration['n']}
        self.clean += 1

        dS = self._delta_S(dphi, n, change)
        if self.rng.uniform(0, 1) < min(1.0, np.exp(-dS)):
            for link, c in change.items():
                n[link] += c
            self.accepted += 1
            return configuration | {'n': Form(n, degree=1, lattice=L)}

        return configuration | {'n': configuration['n']}

    def report(self):
        if self.proposed == 0:
            return 'WrappingLoopUpdate: no proposals.'
        return (f'WrappingLoopUpdate: {self.accepted} / {self.proposed} wrapping loops '
                f'accepted ({self.accepted / self.proposed:.6f}); '
                f'{self.clean} / {self.proposed} were constraint-preserving '
                f'({self.clean / self.proposed:.6f}).')
