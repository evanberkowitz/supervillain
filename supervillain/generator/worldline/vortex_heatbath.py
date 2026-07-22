#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import delta, delta_sparse, coface_sum_at, Form

import logging
logger = logging.getLogger(__name__)

class VortexHeatbath(ReadWriteable, Generator):
    r'''
    An exact heatbath for the vortex two-form $v$ of the :class:`~.Worldline` action, leaving $m$ untouched.

    Writing the gauge-invariant link one-form $f = m - \delta v / \bar{W}$ (with $\bar{W}=W$ for finite $W$ and $\bar{W}=2\pi$ for $W=\infty$), the action is

    .. math ::

        S[m, v] = \frac{1}{2\kappa} \sum_\ell f_\ell^2.

    Changing a single plaquette $v_p \rightarrow v_p + \Delta$ shifts $f$ on the four boundary links of $p$, so the conditional distribution of $\Delta$ is Gaussian,

    .. math ::

        P(\Delta) \propto \exp\left(-\tfrac{1}{2} a (\Delta - \mu)^2\right),
        \qquad
        \mu = \frac{\bar{W}}{4} (df)_p,
        \qquad
        a = \frac{4}{\kappa \bar{W}^2},

    where $(df)_p = \sum_{\ell \in \partial p} c_{\ell p} f_\ell$ is the signed boundary sum of $f$ over the plaquette.

    When $W < \infty$ the plaquette value is an integer, so $\Delta$ is drawn from the **discrete** Gaussian on the integers (truncated enumeration with a Gumbel-max draw).  When $W=\infty$ the plaquette value is real and $\Delta$ is a continuous normal draw with variance $1/a = \pi^2 \kappa$.

    Because $m$ is never touched the constraint $\delta m = 0$ is preserved trivially.

    .. note ::
        The plaquettes are processed one two-form component and checkerboard color at a time so that a color's plaquettes have frozen boundary links; $f$ is patched incrementally after each color with :func:`~.delta_sparse`.

    .. warning ::
        Like the :class:`~.worldline.VortexUpdate` this is not ergodic on its own since it does not change $m$.
    '''

    # Truncate the discrete-Gaussian enumeration this many standard deviations
    # on either side of the mean; 8σ makes the truncation error utterly negligible.
    _sigma_window = 8

    def __init__(self, action, rng=None):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The VortexHeatbath requires the Worldline action.')

        self.Action = action
        self.finite = (action.W < float('inf'))

        self.rng = rng if rng is not None else np.random.default_rng()

        self.sweeps = 0

    def __str__(self):
        return 'VortexHeatbath'

    def _draw(self, mu, a):
        r'''
        Draw the per-plaquette shift $\Delta$ from $\propto \exp(-\tfrac12 a (\Delta-\mu)^2)$.

        For finite $W$ the shift is an integer drawn from the discrete Gaussian by
        truncated enumeration and a Gumbel-max argmax; for $W=\infty$ it is a
        continuous normal draw with variance $1/a$.
        '''

        if not self.finite:
            return mu + self.rng.normal(size=mu.shape) * np.sqrt(1.0 / a)

        sigma = 1.0 / np.sqrt(a)
        half = int(np.ceil(self._sigma_window * sigma)) + 1
        offsets = np.arange(-half, half + 1)

        # Enumerate integer candidates in a window centered on the (rounded) mean.
        base = np.rint(mu).astype(int)
        candidates = base[:, None] + offsets[None, :]

        # Gumbel-max: argmax over (log-weight + Gumbel noise) samples the categorical
        # distribution with the enumerated (unnormalized) log-weights, exactly.
        logw = -0.5 * a * (candidates - mu[:, None])**2
        gumbel = -np.log(-np.log(self.rng.uniform(0, 1, candidates.shape)))
        pick = np.argmax(logw + gumbel, axis=1)
        return candidates[np.arange(len(mu)), pick]

    def step(self, cfg):
        r'''
        Resample a volume's worth of $v$ from its exact Gaussian conditional, leaving $m$ fixed.

        Parameters
        ----------
        cfg: dict
            A dictionary with m and v field variables.

        Returns
        -------
        dict
            Another configuration of fields.
        '''

        self.sweeps += 1

        m = cfg['m']            # untouched: δm=0 preserved trivially
        v = cfg['v'].copy()

        L = self.Action.Lattice
        W = self.Action._W
        kappa = self.Action.kappa

        a = 4.0 / (kappa * W**2)

        n_comps = len(L.components[2])

        # f = m − δv/W̄, the gauge-invariant link one-form; patched in place below.
        f = Form(np.asarray(m, dtype=float) - np.asarray(delta(v)) / W, degree=1, lattice=L)
        f_raw = np.asarray(f)

        # Each v only talks to f on the four surrounding links.  Freezing everything but one
        # (2-form component, checkerboard color) makes that color's plaquettes independent, so
        # they can be resampled from their Gaussian conditionals simultaneously.  Components are
        # processed sequentially since different components at the same site share links; f is
        # patched after each so the next conditional sees the update.
        for color in L.checkerboarding:
            for comp_idx in range(n_comps):

                n = len(color[0])

                # The signed incidence of this color's plaquettes onto their (disjoint) boundary
                # links.  Multiplying f by it and doing the unsigned coface_sum recovers the
                # signed boundary sum (df)_p = Σ_{ℓ∈∂p} c_{ℓp} f_ℓ, exactly as VortexUpdate carries
                # orientation inside its dS_link before the coface reduction.
                signed_incidence = delta_sparse(L, 2, comp_idx, color, np.ones(n))
                df = coface_sum_at(Form(signed_incidence * f_raw, degree=1, lattice=L), comp_idx, color)

                mu = (W / 4.0) * df
                shift = self._draw(mu, a)

                v[comp_idx][color] += shift
                # Patch f = m − δv/W̄ to reflect the accepted change on the boundary links.
                f_raw -= delta_sparse(L, 2, comp_idx, color, shift) / W

        return cfg | {'v': v}

    def report(self):
        return f'The VortexHeatbath resampled v over {self.sweeps} sweeps.'
