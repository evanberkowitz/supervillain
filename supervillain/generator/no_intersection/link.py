#!/usr/bin/env python

import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import charge

import logging
logger = logging.getLogger(__name__)


class ConstrainedLinkUpdate(ReadWriteable, Generator):
    r"""
    A *local* fluctuation of $F = dn$ that preserves the $Q = dn\wedge dn = 0$
    constraint — the $\theta$-theory analog of the worldline
    :class:`~.worldline.CoexactUpdate`.

    In the worldline formulation `CoexactUpdate` adds $\delta(\text{one plaquette})$ to
    $m$; because the constraint $\delta m = 0$ is *linear* and $\delta\delta = 0$, that
    move is **automatically** constraint-preserving (it is a "one-plaquette worm").

    Here the minimal local move that changes $F$ is a **single-link** change
    $n_{\ell} \to n_{\ell} \pm 1$.  On an $F = 0$ region it creates no charge at all: its
    self-term $d\Delta n\wedge d\Delta n$ vanishes because all six plaquettes of
    $d\Delta n$ share the link's direction, so no two are complementary.  But the
    constraint $F\wedge F = 0$ is **quadratic**, so on a background that already carries
    $F \ne 0$ the cross term $d\Delta n\wedge F + F\wedge d\Delta n$ can produce charge.
    Unlike the worldline coexact move, therefore, this update is **not** automatically
    legal: every proposal is *verified* to keep $Q = 0$ and rejected otherwise.

    Where the worm makes large, coordinated sheet moves (and carries a head/tail dipole
    through intermediate $Q\ne0$ states), this update makes small, purely local
    $F$-fluctuations that never leave the constraint surface — letting $dn$ breathe
    without a worm.

    .. warning::

        Restricted to $D = 4$.  Updates $n$ only; combine with a $\phi$-update such as
        :class:`~.villain.SiteUpdate`.

    .. note::

        The reference check recomputes the global charge for each proposal, which is
        $O(\text{volume})$ per link.  Since only the $\sim$ hypercubes adjacent to the
        link can change, a local $\Delta Q$ check would be far cheaper; that
        optimization is left for a production version.
    """

    def __init__(self, S, interval_n=1):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('ConstrainedLinkUpdate requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('ConstrainedLinkUpdate is only implemented for D = 4.')

        self.Action = S
        self.Lattice = S.Lattice
        self.kappa = S.kappa
        self.rng = np.random.default_rng()

        # Proposed nonzero integer shifts of a single link.
        self.shifts = tuple(c for c in range(-interval_n, interval_n + 1) if c != 0)

        self.accepted = 0
        self.proposed = 0
        self.sweeps = 0

    def __str__(self):
        return 'ConstrainedLinkUpdate'

    def inline_observables(self, steps):
        return {}

    def step(self, cfg):
        r"""
        One sweep: visit every link in random order and offer it a single-link change
        $n_{\ell} \to n_{\ell} + c$ that (i) preserves $Q = dn\wedge dn = 0$ everywhere and
        (ii) passes a Metropolis test against the Villain action.
        """
        L = self.Lattice
        N = L.N
        D = L.D

        n = cfg['n'].copy()
        dphi = d(cfg['phi'])

        self.sweeps += 1
        accepted = 0

        # All links, visited in random order.  We go one at a time because the
        # constraint couples links: two simultaneous changes could interact in F∧F.
        links = [(mu,) + tuple(int(x) for x in site)
                 for mu in range(D)
                 for site in np.ndindex(*((N,) * D))]
        self.rng.shuffle(links)

        for link in links:
            self.proposed += 1
            c = self.shifts[self.rng.integers(0, len(self.shifts))]

            # Reject anything that would violate Q = dn ∧ dn = 0.
            trial = n.copy()
            trial[link] += c
            if not np.allclose(charge(trial), 0):
                continue

            # Metropolis on the Villain action; only this link's term changes.
            #   ΔS = κ/2 [ (A − 2π c)² − A² ],  A = (dφ − 2π n)_ℓ.
            A = dphi[link] - 2 * np.pi * n[link]
            dS = (self.kappa / 2) * ((A - 2 * np.pi * c) ** 2 - A ** 2)
            if self.rng.uniform(0, 1) < min(1.0, np.exp(-dS)):
                n[link] += c
                accepted += 1

        self.accepted += accepted
        new_n = Form(n, degree=1, lattice=L)
        return cfg | {'n': new_n}

    def report(self):
        if self.proposed == 0:
            return 'ConstrainedLinkUpdate: no proposals.'
        return (f'ConstrainedLinkUpdate: {self.accepted} / {self.proposed} '
                f'single-link changes accepted '
                f'({self.accepted / self.proposed:.6f}).')
