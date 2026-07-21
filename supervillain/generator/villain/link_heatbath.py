#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d

import logging
logger = logging.getLogger(__name__)

_TWO_PI = 2 * np.pi


class LinkHeatbath(ReadWriteable, Generator):
    r'''
    The exact heatbath counterpart of :class:`~.LinkUpdate`: it resamples the
    integer link field $n$ from its exact conditional rather than proposing a
    change of $\pm W$ and accepting or rejecting, so it takes no step size and
    never rejects.
    The update leaves $\phi$ untouched.

    At fixed $\phi$ each $n_\ell$ appears in a single term of the
    :class:`~.Villain` action, so the links are conditionally independent and every
    link can be resampled simultaneously.  With $d\phi{}_{\ell}$ a
    fixed background 1-form, the exact conditional on the constraint-preserving
    coset $n_\ell^0 + W\mathbb{Z}$ is a discrete Gaussian

    .. math ::

        P(n{}_\ell = n{}_\ell^0 + W m) \propto \exp\left[-\frac{\kappa}{2}\left(d\phi{}_\ell - 2\pi n{}_\ell^0 - 2\pi W m\right)^2\right],
        \qquad m \in \mathbb{Z},

    centered on $m^* = (d\phi{}_\ell / 2\pi - n{}_\ell^0)/W$ with width
    $\sigma_m = 1/(2\pi W \sqrt{\kappa})$.  We draw it by truncated enumeration and
    Gumbel-max sampling, vectorized across all links: the candidates run over a
    window of $\pm K$ integers about $m^*$, where the integer half-window $K$ is
    set to span ``coverage_sigmas`` standard deviations $\sigma_m$
    of the conditional (so the truncated tail $\sim e^{-\frac{1}{2}\kappa(2\pi W)^2
    K^2}$ is negligible).  $K$ is not a user parameter; it adapts to $\kappa$ and
    $W$ through $\sigma_m$.

    Because the moves are per-link shifts by multiples of $W$, they preserve the
    constraint $dn \equiv 0 \pmod W$ automatically and independently on every link,
    for any $W$.  Like :class:`~.LinkUpdate`, the heatbath therefore holds
    $n \bmod W$ (the winding sector) fixed; for $W=1$ that is vacuous and the
    update is complete, while for $W>1$ the sector must be moved separately (see
    :class:`~.ExactUpdate`, :class:`~.CohomologyUpdate`, and the :class:`~villain.worm.ClassicWorm`).

    .. seealso ::
        :class:`~.LinkUpdate` is the Metropolis version, whose stationary
        distribution on the coset is the same discrete Gaussian sampled here.

    Parameters
    ----------
    action: supervillain.action.Villain
        The Villain action whose $n$ is resampled.
    rng: numpy.random.Generator
        A source of randomness; if omitted a fresh default generator is used.
    coverage_sigmas: float
        How many standard deviations $\sigma_m$ of the conditional the enumeration
        window covers before the (in-principle infinite) discrete-Gaussian sum is
        truncated.  The integer half-window $K$ is derived from this and the width;
        the default 6 leaves a negligible tail and rarely needs changing.
    '''

    def __init__(self, action, rng=None, coverage_sigmas=6.0):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The LinkHeatbath requires the Villain action.')

        self.Action = action
        self.Lattice = action.Lattice
        self.kappa = action.kappa
        self.W = action.W

        self.rng = rng if rng is not None else np.random.default_rng()
        self.coverage_sigmas = coverage_sigmas

        # Derived integer half-window K, from the coset width σ_m = 1/(2π W √κ);
        # the user knob is coverage_sigmas, not K.
        sigma_m = 1.0 / (_TWO_PI * self.W * np.sqrt(self.kappa))
        self.K = int(np.ceil(self.coverage_sigmas * sigma_m)) + 2

        self.sweeps = 0

    def __str__(self):
        return 'LinkHeatbath'

    def step(self, cfg):
        r'''
        Resample every link's $n$ from its exact coset conditional in one parallel
        sweep, leaving $\phi$ untouched.

        Parameters
        ----------
        cfg: dict
            A dictionary with phi and n as Forms.

        Returns
        -------
        dict
            Updated configuration (only n changes).
        '''
        W, kappa, K = self.W, self.kappa, self.K

        n = cfg['n']
        A = np.asarray(d(cfg['phi']))          # fixed background 1-form
        n0 = np.asarray(n)

        self.sweeps += 1

        # Candidate integers m near the coset center m* = (A/2π − n0)/W.
        m_round = np.round((A / _TWO_PI - n0) / W).astype(int)
        offsets = np.arange(-K, K + 1)
        m_cand = m_round[..., None] + offsets                       # (...links..., 2K+1)

        # Unnormalized log-weights of each candidate; Gumbel-max samples one per
        # link exactly ∝ exp(logw) with no explicit normalization.
        resid = A[..., None] - _TWO_PI * (n0[..., None] + W * m_cand)
        logw = -0.5 * kappa * resid**2
        gumbel = -np.log(-np.log(self.rng.uniform(size=logw.shape)))
        choice = np.argmax(logw + gumbel, axis=-1)                  # index into offsets

        m_sel = m_round + (choice - K)
        return cfg | {'n': n + (W * m_sel)}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'LinkHeatbath: {self.sweeps} exact heatbath sweeps of n '
            f'(coset step W={self.W}, window ±{self.K}, no accept/reject).'
        )
