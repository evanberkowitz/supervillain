#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d, delta

import logging
logger = logging.getLogger(__name__)


class SiteOverrelaxation(ReadWriteable, Generator):
    r'''
    The microcanonical overrelaxation partner of :class:`~.SiteHeatbath`: instead of
    resampling $\phi$ from its conditional it *reflects* $\phi$ about the exact
    conditional mean, an action-preserving move that takes no step size, uses no
    randomness in the move itself, and never rejects.

    At fixed $n$ the :class:`~.Villain` action
    $S=\frac{\kappa}{2}\left\|d\phi - 2\pi n\right\|^2$ is quadratic in $\phi$, so the
    conditional of a single site given its neighbors is a one-dimensional Gaussian with
    mean $\mu_x = \phi_x - g_x/2D$, where $g=\delta(d\phi-2\pi n)$ is the local force
    0-form and $2D$ is the coordination number in dimension $D$.  Reflecting $\phi_x$
    about $\mu_x$,

    .. math ::

        \phi_x \;\to\; 2\mu_x - \phi_x \;=\; \phi_x - \frac{g_x}{D},

    leaves the conditional --- and hence the total action --- exactly invariant.
    Sweeping the colors of the :attr:`~.Lattice.checkerboarding` freezes every neighbor
    within a color, so each color reflects simultaneously and exactly.  The update leaves
    $n$ untouched and is dimension-independent.

    Each single-color reflection is an involution, but a full checkerboard *sweep*
    composes non-commuting per-color reflections and is **not** an involution --- it is a
    rotation on the constant-action shell.  So ``applications`` full sweeps per step walk
    ``applications`` genuine steps around the shell; this is the standard $N_{OR}>1$
    overrelaxation practice.

    .. note ::

        This move is **not ergodic on its own** --- it never leaves the action shell.
        Run it interleaved with an ergodic sampler such as :class:`~.SiteHeatbath`.

    .. seealso ::

        :class:`~.SiteHeatbath` for the exact-conditional draw and :class:`~.SiteUpdate`
        for the Metropolis version.

    Parameters
    ----------
    action: supervillain.action.Villain
        The Villain action whose $\phi$ is reflected.
    applications: int
        How many full reflection sweeps to perform per :func:`step`.  A single sweep is
        action-neutral; more sweeps decorrelate further around the action shell.
    rng: numpy.random.Generator
        A source of randomness used only to randomize the color order each sweep (for a
        reversible composite); if omitted a fresh default generator is used.
    '''

    def __init__(self, action, applications=3, rng=None):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The SiteOverrelaxation requires the Villain action.')
        if applications < 1:
            raise ValueError(f'applications must be a positive integer, got {applications}.')

        self.Action = action
        self.Lattice = action.Lattice
        self.kappa = action.kappa
        self.applications = applications

        self.rng = rng if rng is not None else np.random.default_rng()

        self.sweeps = 0

    def __str__(self):
        return f'SiteOverrelaxation(applications={self.applications})'

    def _sweep(self, phi, n):
        r'''One checkerboard reflection sweep, $\phi_x \to \phi_x - g_x/D$.'''
        L = self.Lattice
        D = L.D

        # Residual r = dφ − 2πn; updated incrementally as each color is reflected.
        r = d(phi) - 2 * np.pi * n

        colors = L.checkerboarding
        for i in self.rng.permutation(len(colors)):
            color = colors[i]

            # Local force g = δr; on a color's sites the neighbors are frozen, so g there
            # is the exact conditional gradient.  Reflecting about the conditional mean
            # μ = φ − g/2D gives φ → 2μ − φ = φ − g/D, which preserves the action.
            g = delta(r)
            reflected = phi[0, *color] - g[0, *color] / D

            change_phi = L.zeros(0)
            change_phi[0, *color] = reflected - phi[0, *color]

            phi = phi + change_phi
            r = r + d(change_phi)

        return phi

    def step(self, cfg):
        r'''
        Reflect $\phi$ everywhere with ``applications`` action-preserving checkerboard
        sweeps, leaving $n$ untouched.

        Parameters
        ----------
        cfg: dict
            A dictionary with phi and n as Forms.

        Returns
        -------
        dict
            Updated configuration (only phi changes).
        '''
        phi = cfg['phi'].copy()
        n = cfg['n']

        for _ in range(self.applications):
            phi = self._sweep(phi, n)
            self.sweeps += 1

        return cfg | {'phi': phi}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'{self.sweeps} overrelaxation sweeps of φ.'
        )
