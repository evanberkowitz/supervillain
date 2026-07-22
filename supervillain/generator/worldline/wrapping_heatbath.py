#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import delta

import logging
logger = logging.getLogger(__name__)


class WrappingHeatbath(ReadWriteable, Generator):
    r'''
    The exact heatbath (Gibbs) counterpart of :class:`~.WrappingUpdate`: for every
    torus-wrapping cycle it resamples the winding shift from its exact conditional rather than
    proposing an integer shift and accepting or rejecting, so it takes no step size and never
    rejects.

    Like :class:`~.WrappingUpdate` it changes $m$ by a constant integer $\Delta$ on all the
    links of a single cycle around the torus; because that change is a closed loop current it
    leaves $\delta m = 0$, and it never touches $v$.  For a $\mu$-direction cycle of $N$ links,
    writing $f = m - \delta v / \bar{W}$ for the residual one-form (with $\bar{W} = W$ finite or
    $2\pi$ when $W=\infty$) and ${A} = \sum_{\ell\in\text{cycle}} {f}_\ell$, the
    :class:`~.Worldline` action changes by

    .. math ::

        \Delta S(\Delta) = \frac{1}{2\kappa}\sum_{\ell\in\text{cycle}} \Delta\,(2 {f}_\ell + \Delta)
        = \frac{N}{2\kappa}\Delta^2 + \frac{{A}}{\kappa}\Delta
        \;=\; \tfrac12 a\,(\Delta - \Delta^\ast)^2 - \tfrac12 a\,(\Delta^\ast)^2,

    a discrete Gaussian with curvature $a = N/\kappa$ (width $\sigma = \sqrt{\kappa/N}$) centered
    on the real mean shift $\Delta^\ast = -{A}/N$.  The trailing $-\tfrac12 a\,(\Delta^\ast)^2$
    is the vertex value that keeps $\Delta S(0) = 0$; being independent of $\Delta$ it cancels
    from the normalized conditional and never enters the draw.  Every one of the $D\,N^{D-1}$
    cycles touches a disjoint set of links, so the shifts are mutually independent and drawn
    simultaneously by truncated enumeration with a Gumbel-max pick, exactly as in
    :class:`~.CoexactHeatbath`.  The update leaves $v$ untouched, leaves $\delta m = 0$, and
    works for every $W$.

    .. note ::

        Being discrete-Gaussian rather than continuous, this move has no clean microcanonical
        overrelaxation partner (a reflection about the real mean $\Delta^\ast$ leaves the
        integer lattice).

    .. warning::

        Like :class:`~.WrappingUpdate` this move is not ergodic on its own --- it cannot
        generate coexact changes.  It is a partner to the vortex and coexact moves, not a
        standalone sampler.

    .. seealso ::
        :class:`~.WrappingUpdate` for the Metropolis version, which offers the same
        cycle-winding move but with a tunable proposal width and an accept/reject step.

    Parameters
    ----------
    action: supervillain.action.Worldline
        The Worldline action whose torus-wrapping cycles are resampled.
    rng: numpy.random.Generator
        A source of randomness; if omitted a fresh default generator is used.
    coverage_sigmas: float
        How many conditional widths the integer-enumeration window covers before the
        discrete-Gaussian sum is truncated; the integer half-window is derived from it.
    '''

    def __init__(self, action, rng=None, coverage_sigmas=6.0):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The WrappingHeatbath requires the Worldline action.')

        self.Action = action
        self.Lattice = action.Lattice
        self.kappa = action.kappa
        self.coverage_sigmas = coverage_sigmas

        self.rng = rng if rng is not None else np.random.default_rng()

        # Each cycle has N links, so curvature a = N/κ and width σ = √(κ/N).
        L = self.Lattice
        self.cycle_links = L.N
        self.curvature = self.cycle_links / self.kappa
        sigma = 1.0 / np.sqrt(self.curvature)
        # Enumeration half-window; the + 2 floors K >= 3 so a nonzero shift is always a
        # candidate even when σ << 1, with its (exponentially small) weight then set by the
        # conditional rather than by the window.
        self.K = int(np.ceil(self.coverage_sigmas * sigma)) + 2

        self.sweeps = 0

    def __str__(self):
        return 'WrappingHeatbath'

    def _draw(self, dstar):
        r'''Sample integers from the discrete Gaussian $\propto e^{-\frac12 a (\Delta-\Delta^\ast)^2}$
        around the real means ``dstar`` by truncated enumeration + Gumbel-max.'''
        dstar = np.asarray(dstar)
        dround = np.round(dstar).astype(int)
        offsets = np.arange(-self.K, self.K + 1)
        cand = dround[..., None] + offsets
        logw = -0.5 * self.curvature * (cand - dstar[..., None])**2
        gumbel = -np.log(-np.log(self.rng.uniform(size=logw.shape)))
        return dround + (np.argmax(logw + gumbel, axis=-1) - self.K)

    def step(self, cfg):
        r'''
        Resample the winding shift on every torus cycle from its exact conditional, leaving
        $v$ and $\delta m$ untouched.

        Parameters
        ----------
        cfg: dict
            A dictionary with m and v as Forms.

        Returns
        -------
        dict
            Updated configuration (only m changes, by a sum of closed cycle currents).
        '''
        L = self.Lattice

        m = cfg['m'].copy()
        v = cfg['v']

        self.sweeps += 1

        # Residual f = m - δv/W̄, frozen over the sweep (v untouched).  Each μ-cycle's conditional
        # sees only the μ-component of f summed along μ, and the cycles are mutually independent,
        # so all directions are drawn from f at the current m in a single pass.
        f = np.asarray(m) - np.asarray(delta(v)) / self.Action._W

        change_m = L.form(1, dtype=int)
        for mu in range(L.D):
            # A[perp] = sum of f over the N links of each μ-cycle; center Δ* = -A/N.
            A = f[mu].sum(axis=mu)
            dstar = -A / self.cycle_links
            change_m[mu] = np.expand_dims(self._draw(dstar), axis=mu)

        return cfg | {'m': m + change_m}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'{self.sweeps} exact heatbath sweeps of the torus-wrapping cycles.'
        )
