#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d, delta, delta_sparse, Form

import logging
logger = logging.getLogger(__name__)


class CoexactHeatbath(ReadWriteable, Generator):
    r'''
    The exact heatbath (Gibbs) counterpart of :class:`~.CoexactUpdate`: it resamples the
    coexact part of $m$ from its exact conditional rather than proposing $\Delta m = \delta t$
    and accepting or rejecting, so it takes no step size and never rejects.

    :class:`~.CoexactUpdate` shifts $m$ by $\delta t$ for an integer two-form $t$; because
    $\delta^2 t = 0$ this leaves $\delta m$ (and therefore the constraint $\delta m = 0$)
    untouched, and it never changes $v$.  Writing $f = m - \delta v / \bar{W}$ for the
    residual one-form (with $\bar{W} = W$ finite or $2\pi$ when $W=\infty$), the
    :class:`~.Worldline` action $S = \frac{1}{2\kappa}\sum_\ell {f}_\ell^2$ is quadratic in a
    single plaquette's integer shift $t_p$.  Shifting $t_p$ by an integer $x$ sends
    ${f}_\ell \to {f}_\ell + x\, c_{\ell p}$ on the four boundary links of $p$ (with signed
    incidence $c_{\ell p}$, independent of $W$ since $t$ is not divided by $\bar{W}$), and
    changes the action by

    .. math ::

        \Delta S(x) = \frac{1}{\kappa}\left[x\,(df)_p + 2 x^2\right]
        \;=\; \tfrac{1}{2} a\,(x - \mu)^2 + \text{const},

    a discrete Gaussian with curvature $a = 4/\kappa$ (width $\sigma = \sqrt{\kappa}/2$)
    centered on the real mean shift $\mu = -(df)_p / 4$, where
    $(df)_p = \sum_{\ell \in \partial p} c_{\ell p} {f}_\ell$ is the exterior derivative of
    $f$ at the plaquette $p$.  Sweeping the colors of the
    :attr:`~.Lattice.checkerboarding` (and processing the $\binom{D}{2}$ two-form
    components sequentially) freezes every boundary link within a color, so each color's
    draws are simultaneously exact.  The integers are drawn by truncated enumeration (a
    window of ``coverage_sigmas`` widths) with a Gumbel-max pick, exactly as in
    :class:`~.ExactHeatbath`.  The update leaves $v$ untouched, leaves $\delta m$
    untouched, and works for every $W$.

    .. note ::

        Because $t_p$ is always an integer (for every $W$), this move is always a
        discrete Gaussian and has no clean microcanonical overrelaxation partner (a
        reflection about the real mean $\mu$ leaves the integer lattice).

    .. seealso ::
        :class:`~.CoexactUpdate` for the Metropolis version, which offers the same
        $\Delta m = \delta t$ move but with a tunable proposal width and an accept/reject
        step.

    Parameters
    ----------
    action: supervillain.action.Worldline
        The Worldline action whose coexact part of $m$ is resampled.
    rng: numpy.random.Generator
        A source of randomness; if omitted a fresh default generator is used.
    coverage_sigmas: float
        How many conditional widths the integer-enumeration window covers before the
        discrete-Gaussian sum is truncated; the integer half-window is derived from it.
    '''

    def __init__(self, action, rng=None, coverage_sigmas=6.0):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The CoexactHeatbath requires the Worldline action.')

        self.Action = action
        self.Lattice = action.Lattice
        self.kappa = action.kappa
        self.coverage_sigmas = coverage_sigmas

        self.rng = rng if rng is not None else np.random.default_rng()

        # Curvature a = 4/κ and width σ = √κ/2 of the per-plaquette t conditional.
        self.curvature = 4.0 / self.kappa
        sigma = 1.0 / np.sqrt(self.curvature)
        self.K = int(np.ceil(self.coverage_sigmas * sigma)) + 2

        self.sweeps = 0

    def __str__(self):
        return 'CoexactHeatbath'

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
        Resample the coexact part of $m$ with one exact checkerboard heatbath sweep,
        leaving $v$ and $\delta m$ untouched.

        Parameters
        ----------
        cfg: dict
            A dictionary with m and v as Forms.

        Returns
        -------
        dict
            Updated configuration (only m changes, by a coexact form $\delta t$).
        '''
        L = self.Lattice

        v = cfg['v']
        delta_v_by_W = np.asarray(delta(v)) / self.Action._W   # frozen over the sweep (v untouched)

        m = cfg['m'].copy()
        m_raw = np.asarray(m)                                  # patched in place below

        self.sweeps += 1

        n_comps = len(L.components[2])

        # The change in m is a coexact form δt (t a 2-form) so that δm stays 0 (δ²t = 0).  Two
        # plaquettes of the same component at same-color sites never share boundary links, so a
        # color's draws are simultaneously exact; we process components sequentially to avoid
        # conflicts between different components at the same site.  The residual f = m - δv/W̄ is
        # updated incrementally through m_raw, and the exact conditional's mean shift on each
        # plaquette p is k* = μ = -(df)_p/4 with (df)_p = d(f) at p.
        colors = L.checkerboarding
        for i in self.rng.permutation(len(colors)):
            color = colors[i]
            for comp_idx in range(n_comps):

                f = Form(m_raw - delta_v_by_W, degree=1, lattice=L)
                g = np.asarray(d(f))

                kstar = -g[comp_idx][color] / 4.0
                ksel = self._draw(kstar)

                # Apply m += δ(t restricted to this component and color), patched in place.
                delta_sparse(L, 2, comp_idx, color, ksel, out=m_raw)

        return cfg | {'m': m}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'{self.sweeps} exact heatbath sweeps of the coexact part of m.'
        )
