#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d

import logging
logger = logging.getLogger(__name__)


class CohomologyHeatbath(ReadWriteable, Generator):
    r'''
    The exact heatbath (Gibbs) counterpart of :class:`~.CohomologyUpdate`: for each
    direction it resamples the global winding holonomy from its exact conditional rather
    than proposing an integer shift and accepting or rejecting.

    Like :class:`~.CohomologyUpdate` it adds a constant integer $h_\mu$ to $n_\mu$ on the
    single slice $x_\mu = 0$; because $\Delta {n}_\mu$ is constant on that slice,
    $d(\Delta n) = 0$ exactly, so the constraint $dn \equiv 0\ (\bmod W)$ is preserved for
    any $W$ and the winding $w_\mu$ changes by $h_\mu$.  With everything else fixed the
    :class:`~.Villain` action is quadratic in the integer $h_\mu$, so its conditional is a
    one-dimensional discrete Gaussian.  Writing $r = d\phi - 2\pi n$ and
    ${R}_\mu = \sum_{\ell\in\text{slice }\mu} {r}_\ell$, and letting the slice hold
    $N^{D-1}$ links,

    .. math ::

        \Delta S(h) = 2\pi\kappa\left[-h\,R_\mu + \pi N^{D-1} h^2\right]
        \;=\; \tfrac12 a\,(h - h^\ast)^2 - \tfrac12 a\,(h^\ast)^2,

    a discrete Gaussian with curvature $a = 4\pi^2 \kappa N^{D-1}$ (width
    $\sigma = 1/\sqrt a$) centered on $h^\ast = {R}_\mu/(2\pi N^{D-1})$.  The trailing
    $-\tfrac12 a\,(h^\ast)^2$ is the vertex value that keeps $\Delta S(0) = 0$; being
    independent of $h$ it cancels from the normalized conditional and never enters the
    draw.  The $D$ directions
    are independent and processed sequentially, updating the residual between them.  The
    integers are drawn by truncated enumeration with a Gumbel-max pick, exactly as in
    :class:`~.LinkHeatbath`.  The update leaves $\phi$ and $dn$ untouched.

    .. note ::

        The action barrier scales as $O(\kappa N^{D-1})$ --- the genuine physical cost of
        tunneling between winding sectors --- so $\sigma$ shrinks as $N^{-(D-1)/2}$ and the
        enumeration window stays tiny.  Unlike a Metropolis proposal the heatbath still
        draws the exact conditional, but the sectors themselves are exponentially
        separated at large $\kappa N^{D-1}$; that is physics, not an algorithmic defect.

    .. note ::

        Being discrete-Gaussian rather than continuous, this move has no clean
        microcanonical overrelaxation partner.

    .. seealso ::
        :class:`~.CohomologyUpdate` for the Metropolis version.

    Parameters
    ----------
    action: supervillain.action.Villain
        The Villain action whose winding holonomy is resampled.
    rng: numpy.random.Generator
        A source of randomness; if omitted a fresh default generator is used.
    coverage_sigmas: float
        How many conditional widths the integer-enumeration window covers before the
        discrete-Gaussian sum is truncated.
    '''

    def __init__(self, action, rng=None, coverage_sigmas=6.0):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The CohomologyHeatbath requires the Villain action.')

        self.Action = action
        self.Lattice = action.Lattice
        self.kappa = action.kappa
        self.coverage_sigmas = coverage_sigmas

        self.rng = rng if rng is not None else np.random.default_rng()

        # A slice x_mu = 0 holds N^{D-1} links; curvature a = 4 π² κ N^{D-1}.
        L = self.Lattice
        self.slice_links = L.N ** (L.D - 1)
        self.curvature = 4 * np.pi**2 * self.kappa * self.slice_links
        sigma = 1.0 / np.sqrt(self.curvature)
        self.K = int(np.ceil(self.coverage_sigmas * sigma)) + 2

        self.sweeps = 0

    def __str__(self):
        return 'CohomologyHeatbath'

    def _draw(self, hstar):
        r'''Sample integers from the discrete Gaussian $\propto e^{-\frac12 a (h-h^\ast)^2}$
        around the real means ``hstar`` by truncated enumeration + Gumbel-max.'''
        hstar = np.asarray(hstar)
        hround = np.round(hstar).astype(int)
        offsets = np.arange(-self.K, self.K + 1)
        cand = hround[..., None] + offsets
        logw = -0.5 * self.curvature * (cand - hstar[..., None])**2
        gumbel = -np.log(-np.log(self.rng.uniform(size=logw.shape)))
        return hround + (np.argmax(logw + gumbel, axis=-1) - self.K)

    def step(self, configuration):
        r'''
        Resample the winding holonomy in each direction from its exact conditional,
        leaving $\phi$ and $dn$ untouched.

        Parameters
        ----------
        configuration: dict
            A dictionary with phi and n as Forms.

        Returns
        -------
        dict
            Updated configuration (only n changes, by a closed constant-on-a-slice form).
        '''
        L = self.Lattice

        n = configuration['n'].copy()
        r = d(configuration['phi']) - 2 * np.pi * n

        self.sweeps += 1

        for mu in range(L.D):
            slice_idx = (mu,) + tuple(0 if i == mu else slice(None) for i in range(L.D))

            R_mu = float(r[slice_idx].sum())
            hstar = R_mu / (2 * np.pi * self.slice_links)
            h = int(self._draw(hstar))

            n[slice_idx] += h
            r[slice_idx] += -2 * np.pi * h

        return configuration | {'n': n}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'{self.sweeps} exact heatbath sweeps of the winding holonomy.'
        )
