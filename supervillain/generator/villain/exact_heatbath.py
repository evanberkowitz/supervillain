#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d, delta

import logging
logger = logging.getLogger(__name__)


class ExactHeatbath(ReadWriteable, Generator):
    r'''
    The exact heatbath counterpart of :class:`~.ExactUpdate` resamples the
    exact part of $n$ from its exact conditional rather than proposing $\Delta n = dz$ and
    accepting or rejecting, so it takes no step size and never rejects.

    :class:`~.ExactUpdate` shifts $n$ by $dz$ for an integer 0-form $z$; because
    $d^2 z = 0$ this leaves $dn$ (and therefore the winding constraint $dn \equiv 0
    \bmod W$) untouched.  At fixed $\phi$ and fixed complement the :class:`~.Villain`
    action $S=\frac{\kappa}{2}\left\|d\phi - 2\pi n\right\|^2$ is quadratic in a single
    site's shift $z_x$, so its conditional is a one-dimensional *discrete* Gaussian on the
    integers.  Writing $r = d\phi - 2\pi n$ for the residual 1-form and using that the
    scalar Laplacian $\delta d$ has diagonal equal to the coordination number $2D$ in
    dimension $D$, shifting $z_x$ by an integer $k$ changes the action by

    .. math ::

        \Delta S(k) = 2\pi\kappa\left[-k\,(\delta r)_x + \pi (2D)\, k^2\right]
        \;=\; \tfrac{1}{2} a\,(k - k^\ast)^2 - \tfrac{1}{2} a\,(k^\ast)^2,

    a discrete Gaussian with curvature $a = 8\pi^2\kappa D$ (width $\sigma = 1/\sqrt a$)
    centered on $k^\ast = (\delta r)_x/(4\pi D)$.  The trailing $-\tfrac{1}{2} a\,(k^\ast)^2$
    is the vertex value that keeps $\Delta S(0) = 0$; being independent of $k$ it cancels
    from the normalized conditional and never enters the draw.  Sweeping the colors of the
    :attr:`~.Lattice.checkerboarding` freezes every neighbor within a color, so each
    color's draws are simultaneously exact.  The integers are drawn by truncated
    enumeration (a window of ``coverage_sigmas`` widths) with a Gumbel-max pick, exactly
    as in :class:`~.LinkHeatbath`.  The update leaves $\phi$ untouched, leaves $dn$
    untouched, and is dimension-independent.

    .. note ::

        Being discrete-Gaussian rather than continuous, this move has no clean
        microcanonical overrelaxation partner (a reflection about the real mean $k^\ast$
        leaves the integer lattice).

    .. seealso ::
        :class:`~.ExactUpdate` for the Metropolis version, which offers the same
        $\Delta n = dz$ move but with a tunable proposal width and an accept/reject step.

    Parameters
    ----------
    action: supervillain.action.Villain
        The Villain action whose exact part of $n$ is resampled.
    rng: numpy.random.Generator
        A source of randomness; if omitted a fresh default generator is used.
    coverage_sigmas: float
        How many conditional widths the integer-enumeration window covers before the
        discrete-Gaussian sum is truncated; the integer half-window is derived from it.
    '''

    def __init__(self, action, rng=None, coverage_sigmas=6.0):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The ExactHeatbath requires the Villain action.')

        self.Action = action
        self.Lattice = action.Lattice
        self.kappa = action.kappa
        self.coverage_sigmas = coverage_sigmas

        self.rng = rng if rng is not None else np.random.default_rng()

        # Curvature a = 8 π² κ D and width σ = 1/√a of the per-site z conditional.
        D = self.Lattice.D
        self.curvature = 8 * np.pi**2 * self.kappa * D
        sigma = 1.0 / np.sqrt(self.curvature)
        # The + 2 floors K >= 3 (ceil of any positive is >= 1), so _draw always enumerates
        # round(k*) +/- 1, 2, 3: a nonzero shift is always a candidate even when sigma << 1.
        self.K = int(np.ceil(self.coverage_sigmas * sigma)) + 2

        self.sweeps = 0

    def __str__(self):
        return 'ExactHeatbath'

    def _draw(self, kstar):
        r'''Sample integers from the discrete Gaussian $\propto e^{-\frac12 a (k-k^\ast)^2}$
        around the real means ``kstar`` by truncated enumeration + Gumbel-max.'''
        kround = np.round(kstar).astype(int)
        offsets = np.arange(-self.K, self.K + 1)
        cand = kround[..., None] + offsets
        logw = -0.5 * self.curvature * (cand - kstar[..., None])**2
        gumbel = -np.log(-np.log(self.rng.uniform(size=logw.shape)))
        return kround + (np.argmax(logw + gumbel, axis=-1) - self.K)

    def step(self, cfg):
        r'''
        Resample the exact part of $n$ with one exact checkerboard heatbath sweep,
        leaving $\phi$ and $dn$ untouched.

        Parameters
        ----------
        cfg: dict
            A dictionary with phi and n as Forms.

        Returns
        -------
        dict
            Updated configuration (only n changes, by a closed form $dz$).
        '''
        L = self.Lattice
        D = L.D

        phi = cfg['phi']
        n = cfg['n'].copy()

        self.sweeps += 1

        # Residual r = dφ − 2πn; updated incrementally as each color is redrawn.
        r = d(phi) - 2 * np.pi * n

        colors = L.checkerboarding
        for i in self.rng.permutation(len(colors)):
            color = colors[i]

            # Local coefficient (δr)_x; on a color's sites the neighbors are frozen, so
            # this is the exact conditional linear term.  The Gibbs mean shift is
            # k* = (δr)_x / (4πD).
            g = delta(r)
            kstar = g[0, *color] / (4 * np.pi * D)
            ksel = self._draw(kstar)

            # z must carry n's integer dtype: L.zeros defaults to float, and since d
            # faithfully preserves the dtype it is handed, a float z makes dn float and
            # silently widens the *integer* field n to float64 on `n = n + dn`.  The
            # drawn values are integral either way (ksel comes from _draw as an int),
            # so nothing numerical changes -- but everything downstream that treats n
            # as exactly integral (h5 dtype, equality, mod-W arithmetic) would be
            # working on floats after a single Hammer sweep.
            change_z = L.zeros(0, dtype=n.dtype)
            change_z[0, *color] = ksel

            dn = d(change_z)
            n = n + dn
            # r is the float residual dφ − 2πn, so this addition is float by design
            r = r - 2 * np.pi * dn

        return cfg | {'n': n}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'{self.sweeps} exact heatbath sweeps of the exact part of n.'
        )
