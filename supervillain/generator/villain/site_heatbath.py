#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d, delta

import logging
logger = logging.getLogger(__name__)


class SiteHeatbath(ReadWriteable, Generator):
    r'''
    The exact heatbath (Gibbs) counterpart of :class:`~.SiteUpdate`: it resamples
    $\phi$ from its exact conditional rather than proposing a change and
    accepting or rejecting, so it takes no step size and never rejects.

    At fixed $n$ the :class:`~.Villain` action
    $S=\frac{\kappa}{2}\left\|d\phi - 2\pi n\right\|^2$ is quadratic in $\phi$, so
    the conditional of a single site given its neighbors is a one-dimensional
    Gaussian.  Writing $g = \delta(d\phi - 2\pi n)$ for the local force 0-form (so
    that $\kappa g = \partial S / \partial \phi$) and using that the scalar
    Laplacian $\delta d$ has diagonal equal to the coordination number $2D$ in dimension $D$,

    .. math ::

        \phi_x \sim \mathcal{N}\left(\phi_x - \frac{g_x}{2D},\; \frac{1}{2D\kappa}\right).

    Sweeping the colors of the :attr:`~.Lattice.checkerboarding` freezes every
    neighbor within a color, so each color's draws are simultaneously exact; this
    is red--black Gibbs / stochastic Gauss--Seidel on $\Delta\phi = 2\pi\,\delta n$.
    The update leaves $n$ untouched and is dimension-independent.

    .. seealso ::
        :class:`~.SiteUpdate` for the Metropolis version, which offers the same
        $\phi$ update but with a tunable proposal width and accept/reject step.

    Parameters
    ----------
    action: supervillain.action.Villain
        The Villain action whose $\phi$ is resampled.
    rng: numpy.random.Generator
        A source of randomness; if omitted a fresh default generator is used.
    '''

    def __init__(self, action, rng=None):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The SiteHeatbath requires the Villain action.')

        self.Action = action
        self.Lattice = action.Lattice
        self.kappa = action.kappa

        self.rng = rng if rng is not None else np.random.default_rng()

        self.sweeps = 0

    def __str__(self):
        return 'SiteHeatbath'

    def step(self, cfg):
        r'''
        Resample $\phi$ everywhere with one exact checkerboard heatbath sweep,
        leaving $n$ untouched.

        Parameters
        ----------
        cfg: dict
            A dictionary with phi and n as Forms.

        Returns
        -------
        dict
            Updated configuration (only phi changes).
        '''
        L = self.Lattice
        coordination = 2 * L.D
        sigma = 1.0 / np.sqrt(coordination * self.kappa)

        phi = cfg['phi'].copy()
        n = cfg['n']

        self.sweeps += 1

        # Residual r = dφ − 2πn; updated incrementally as each color is redrawn.
        r = d(phi) - 2 * np.pi * n

        colors = L.checkerboarding
        # The color order is randomized every sweep.  This is not needed for
        # correctness --- each color's draw is an exact conditional, so any fixed
        # order already samples $P(\phi\mid n)$ --- but it makes the composite sweep
        # kernel reversible rather than merely stationary.
        for i in self.rng.permutation(len(colors)):
            color = colors[i]

            # Local force g = δr; on a color's sites the neighbors are frozen, so g
            # there is the exact conditional gradient.  The exact Gibbs mean is
            # φ − g/2D (the φ_x cancels: it depends only on the neighbors), with
            # variance 1/(2D κ).
            g = delta(r)
            draw = phi[0, *color] - g[0, *color] / coordination \
                + sigma * self.rng.normal(size=len(color[0]))

            change_phi = L.zeros(0)
            change_phi[0, *color] = draw - phi[0, *color]

            phi = phi + change_phi
            r = r + d(change_phi)

        return cfg | {'phi': phi}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'{self.sweeps} exact heatbath sweeps of φ.'
        )
