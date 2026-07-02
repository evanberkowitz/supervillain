#!/usr/bin/env python

import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import charge

import logging
logger = logging.getLogger(__name__)


class PlanarFluxUpdate(ReadWriteable, Generator):
    r"""
    A *global* tunneling move that deposits a single decomposable ("planar") vortex sheet
    on top of the current configuration.  It proposes a $\Delta n$ whose field strength is
    the staggered constant-$A$ sheet

    .. math::
        d(\Delta n)_{\mu\nu}(x) = A_{\mu\nu}\,(-1)^{x_\mu + x_\nu},
        \qquad A = u \wedge v,\quad u, v \in \{-1, 0, 1\}^4 .

    Because $A = u\wedge v$ is *decomposable* its Pfaffian vanishes, so the sheet is a
    simple 2-form --- a single vortex plane $\mathrm{span}\{u, v\}$ that does not
    self-intersect --- and on the $F = 0$ vacuum it is automatically clean ($q = 0$).  On a
    nonzero background the cross term $A \wedge F$ can relight $q$, so every proposal is
    verified to keep $q = dn\wedge dn = 0$ and Metropolis-tested against the Villain action.

    Unlike the small loops of :class:`~.WrappingLoopUpdate`, each accepted move changes $F$
    over the whole lattice, so it makes large jumps: a *tunneling* move well matched to the
    sheet-like frozen sector (a frozen configuration is itself such a sheet), complementary
    to the local loop and worm moves.  See :ref:`the frozen-configuration discussion
    <no_intersection>`.

    The proposal is symmetric --- $A$ and $-A = (-u)\wedge v$ are drawn with equal
    probability --- so the reverse move is always available and the plain acceptance
    $\min(1, e^{-\Delta S})$ satisfies detailed balance.

    .. warning::

        Restricted to $D = 4$.  Updates $n$ only; combine with a $\phi$-update such as
        :class:`~.villain.SiteUpdate`.

    .. note::

        The sheet spans the whole lattice, so at physical $\kappa$ its $\Delta S$ is large
        and acceptance is low (from the $F = 0$ vacuum it is effectively zero).  It is
        useful as a rare, decisive tunneling move *within* the sheet sector rather than a
        from-cold flux builder.  The constraint is verified by a global ``charge`` recompute
        per proposal; $u \parallel v$ (or a zero vector) gives $A = 0$, a null move.
    """

    def __init__(self, S):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('PlanarFluxUpdate requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('PlanarFluxUpdate is only implemented for D = 4.')

        self.Action = S
        self.Lattice = S.Lattice
        self.kappa = S.kappa
        self.rng = np.random.default_rng()

        self.proposed = 0
        self.clean = 0          # proposals that preserved q
        self.accepted = 0       # clean proposals that passed Metropolis
        self.degenerate = 0     # u ∧ v = 0 (u ∥ v or a zero vector)

    def __str__(self):
        return 'PlanarFluxUpdate'

    def inline_observables(self, steps):
        return {}

    def _propose_sheet(self):
        r"""
        A $\Delta n$ (as a 1-form) whose curl is the staggered sheet
        $A_{\mu\nu}(-1)^{(x-t)_\mu + (x-t)_\nu}$ with $A = u \wedge v$,
        $u, v \in \{-1, 0, 1\}^4$, anchored at a random site $t$ (the *translation
        offset*, drawn fresh each proposal).

        The $\Delta n$ integrates $A$ in the triangular gauge $\Delta n_0 = 0$, so
        $\Delta n_\nu$ absorbs the $A_{\mu\nu}$ for all $\mu < \nu$.  On an even lattice
        the offset acts on the field strength only through the sign $(-1)^{t_\mu + t_\nu}$
        (already reachable by flipping $u, v$) but shifts the torus-wrapping holonomy of
        the deposited $\Delta n$; on an odd lattice it also relocates the wrap-around
        seam, giving genuinely distinct sheets.
        """
        L = self.Lattice
        N = L.N
        u = self.rng.integers(-1, 2, size=4)
        v = self.rng.integers(-1, 2, size=4)
        A = {(mu, nu): int(u[mu] * v[nu] - u[nu] * v[mu])
             for mu in range(4) for nu in range(mu + 1, 4)}
        t = self.rng.integers(0, N, size=4)

        dn = L.zeros(1, dtype=int)
        arr = np.asarray(dn)
        g = np.meshgrid(range(N), range(N), range(N), range(N), indexing='ij')
        step = [(g[i] - t[i]) % 2 for i in range(4)]         # staggered step at offset t
        sign = [1 - 2 * step[i] for i in range(4)]           # its orientation, (-1)^{x-t}
        arr[1] = A[(0, 1)] * sign[1] * step[0]
        arr[2] = (A[(0, 2)] * step[0] + A[(1, 2)] * step[1]) * sign[2]
        arr[3] = (A[(0, 3)] * step[0] + A[(1, 3)] * step[1]
                  + A[(2, 3)] * step[2]) * sign[3]
        return dn

    def step(self, configuration):
        r"""
        Propose one planar flux sheet, verify it preserves $q = dn\wedge dn = 0$, and
        Metropolis-test it against the Villain action.
        """
        L = self.Lattice
        n = configuration['n']
        dphi = d(configuration['phi'])
        q_now = charge(n)

        self.proposed += 1
        c = np.asarray(self._propose_sheet())
        if not c.any():                          # A = 0: a null move
            self.degenerate += 1
            return configuration | {'n': n}

        trial = Form(np.asarray(n) + c, degree=1, lattice=L)

        # Verify the sheet keeps q = 0 on THIS background; else a null move.
        if not np.array_equal(charge(trial), q_now):
            return configuration | {'n': n}
        self.clean += 1

        # ΔS in the Villain action over every touched link (vectorised).
        A = np.asarray(dphi) - 2 * np.pi * np.asarray(n)
        dS = (self.kappa / 2) * ((A - 2 * np.pi * c) ** 2 - A ** 2).sum()
        if self.rng.uniform(0, 1) < min(1.0, np.exp(-dS)):
            self.accepted += 1
            return configuration | {'n': trial}
        return configuration | {'n': n}

    def report(self):
        if self.proposed == 0:
            return 'PlanarFluxUpdate: no proposals.'
        return (f'PlanarFluxUpdate: {self.accepted} / {self.proposed} sheets accepted '
                f'({self.accepted / self.proposed:.6f}); '
                f'{self.clean} / {self.proposed} were constraint-preserving; '
                f'{self.degenerate} / {self.proposed} were degenerate (A = 0).')
