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
    single-direction links** to $n$, proposed and accepted or rejected **atomically**
    (one Metropolis test on the whole loop).  It is the coordinated move that unfreezes
    the frozen configurations described above.

    Confining $\Delta n$ to a single link direction makes $\Delta q$ *linear* in
    $\Delta n$, so a clean move is one in the kernel of that linear map --- reachable
    even on a frozen background where every single-link move fails.  The move must be a
    *closed* loop because an open string of $n_\mu$ links leaves a free $F$-sheet edge
    (a $q\ne 0$ defect) at each end; and on an $F\ne 0$ background only *non-contractible*
    (torus-wrapping) loops stay clean, since a contractible loop leaks $\Delta q\ne 0$
    where its interior meets the background.  On an $F = 0$ background every
    single-direction loop is clean, so there this update freely deposits the wrapping
    $F$-sheets of the $F\ne 0$, $F\wedge F = 0$ sector.

    Each step draws a closed single-direction loop --- a direction $\mu$, a loop type (a
    thin ring wrapping one transverse axis, or a diagonal ring wrapping two), a random
    location, and a sign $s = \pm 1$ --- verifies it preserves $q = 0$ on the current $n$
    (unclean proposals are null moves), and Metropolis-tests the clean ones against the
    Villain action.  Because $d\Delta n\wedge d\Delta n = 0$ for a single-direction
    $\Delta n$, a loop $L$ is clean on $n$ iff $-L$ is clean on $n + L$, and $L$, $-L$ are
    drawn with equal probability, so the plain acceptance $\min(1, e^{-\Delta S})$
    satisfies detailed balance.

    .. warning::

        Restricted to $D = 4$.  Updates $n$ only, and is not ergodic on its own.

    .. note::

        This reference implementation verifies the constraint with a global ``charge``
        recompute per proposal ($O(\text{volume})$); a local $\Delta q$ check would be
        cheaper.  From a cold $F = 0$ start it correctly builds wrapping $F$-sheets, but
        it is not an efficient *un-freezer*: the uniform-random proposal only rarely hits
        the specific flux-cancelling loop, so it mixes poorly on the frozen sublattice.
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
        self.acceptance = 0.    # summed Metropolis acceptance probability over clean proposals

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
        prob = min(1.0, np.exp(-dS))
        self.acceptance += prob
        if self.rng.uniform(0, 1) < prob:
            for link, c in change.items():
                n[link] += c
            self.accepted += 1
            return configuration | {'n': Form(n, degree=1, lattice=L)}

        return configuration | {'n': configuration['n']}

    def report(self):
        if self.proposed == 0:
            return 'There were 0 proposed wrapping loops.'
        return (
            f'There were {self.accepted} wrapping loops accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.clean / self.proposed:.6f} constraint-preserving fraction'
            +'\n'+
            f'    {self.accepted / self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.proposed:.6f} expected Metropolis acceptance'
        )
