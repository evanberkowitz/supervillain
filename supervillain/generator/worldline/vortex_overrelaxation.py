#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import delta, delta_sparse, coface_sum_at, Form

import logging
logger = logging.getLogger(__name__)

class VortexOverrelaxation(ReadWriteable, Generator):
    r'''
    A microcanonical overrelaxation for the vortex two-form $v$ of the :class:`~.Worldline` action, leaving $m$ untouched.

    With the gauge-invariant link one-form $f = m - \delta v / 2\pi$, the conditional of a single plaquette $v_p$ is Gaussian with mean $\mu_p = v_p + \tfrac{\pi}{2}(df)_p$, where $(df)_p = \sum_{\ell \in \partial p} c_{\ell p} f_\ell$.  Reflecting the plaquette about that mean,

    .. math ::

        v_p \rightarrow 2\mu_p - v_p,

    leaves the action unchanged (it reflects a quadratic about its minimum), which is a valid microcanonical move that decorrelates without a Metropolis accept/reject.

    A reflection about a generally non-integer mean would leave the integer lattice, so this move is **only well-defined for** $W=\infty$, where $v$ is real; the constructor raises a :class:`ValueError` otherwise.

    Each :meth:`step` performs ``applications`` reflection sweeps.  The reflections themselves are deterministic; the random number generator only randomizes the order in which the checkerboard colors are visited.

    .. warning ::
        Being microcanonical this update is not ergodic on its own and, like the :class:`~.worldline.VortexUpdate`, never changes $m$.
    '''

    def __init__(self, action, applications=3, rng=None):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The VortexOverrelaxation requires the Worldline action.')
        if action.W < float('inf'):
            raise ValueError('The VortexOverrelaxation is only well-defined for W=∞; a reflection about the non-integer mean leaves the integer lattice for finite W.')

        self.Action = action
        self.applications = applications

        self.rng = rng if rng is not None else np.random.default_rng()

        self.sweeps = 0

    def __str__(self):
        return 'VortexOverrelaxation'

    def step(self, cfg):
        r'''
        Reflect $v$ about its conditional mean, ``applications`` times, leaving $m$ fixed.

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
        W = self.Action._W       # 2π, since W=∞ here

        n_comps = len(L.components[2])

        # f = m − δv/2π, the gauge-invariant link one-form; patched in place below.
        f = Form(np.asarray(m, dtype=float) - np.asarray(delta(v)) / W, degree=1, lattice=L)
        f_raw = np.asarray(f)

        for _ in range(self.applications):
            colors = list(L.checkerboarding)
            self.rng.shuffle(colors)
            for color in colors:
                for comp_idx in range(n_comps):

                    n = len(color[0])

                    # Signed boundary sum (df)_p, via the unsigned coface_sum of a sign-carrying operand.
                    signed_incidence = delta_sparse(L, 2, comp_idx, color, np.ones(n))
                    df = coface_sum_at(Form(signed_incidence * f_raw, degree=1, lattice=L), comp_idx, color)

                    # The conditional mean is v_p + (π/2)(df)_p, so the reflection shift is 2·(π/2)(df)_p.
                    shift = 2.0 * (W / 4.0) * df

                    v[comp_idx][color] += shift
                    f_raw -= delta_sparse(L, 2, comp_idx, color, shift) / W

        return cfg | {'v': v}

    def report(self):
        return f'The VortexOverrelaxation reflected v over {self.sweeps} sweeps of {self.applications} applications each.'
