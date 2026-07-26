#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d, delta, laplacian

import logging
logger = logging.getLogger(__name__)


class FourierSiteHeatbath(ReadWriteable, Generator):
    r'''
    Draws the *whole* $\phi$ field from its exact joint conditional in one shot,
    where the :class:`~.SiteHeatbath` draws one site at a time and sweeps.

    At fixed $n$ the :class:`~.Villain` action is quadratic in $\phi$, so the
    conditional $P(\phi\mid n)$ is a multivariate Gaussian --- not merely each
    site's conditional, but the *joint* one.  Splitting it about its mean,

    .. math ::

        \phi = \bar\phi + \eta,
        \qquad
        \Delta \bar\phi = 2\pi\, \delta n,
        \qquad
        \eta \sim \exp\left[-\frac{\kappa}{2}\left(\eta, \Delta \eta\right)\right],

    with $\Delta = \delta d$ the scalar Laplacian (the $d\delta$ term annihilates the 0-form $\phi$).  Both
    pieces are available in closed form because $\Delta$ is diagonal in Fourier
    space with symbol
    $\hat{k}^2 = \sum_\mu \left|e^{i{k}_\mu}-1\right|^2$: the mean is one solve of a
    Poisson equation, and the fluctuation's Fourier modes are *independent*
    Gaussians of variance $1/\kappa \hat k^2$.  The whole update is three fast
    Fourier transforms and costs $O(V \log V)$.

    In practice the mean is computed from the same local force
    $g = \delta(d\phi - 2\pi n)$ that the :class:`~.SiteHeatbath` uses: since
    $g = \Delta\phi - 2\pi\delta n$, the conditional mean is
    $\bar\phi = \phi - \Delta^{-1} g$ exactly.

    .. note ::

        The successive $\phi$ produced by this generator are **statistically
        independent** --- it is a direct draw from $P(\phi\mid n)$, not a Markov
        step towards it --- so the $\phi$ sector has no autocorrelation at all.
        A checkerboard sweep is instead stochastic Gauss--Seidel, which relaxes
        the long-wavelength modes of the massless field only diffusively.

    .. note ::

        The constant mode of $\phi$ is an exact flat direction of the action
        ($d$ annihilates it), so its conditional is improper and "resampling it"
        is not defined.  This generator leaves it where it is, which is
        stationary.  Every observable built from $d\phi$ or from differences of
        $\phi$ --- which is every observable in the library --- is blind to it.

    .. warning ::

        Like the :class:`~.SiteHeatbath` this updates $\phi$ only, and is not
        ergodic by itself; combine it with updates of $n$.

    .. seealso ::

        :class:`~.SiteHeatbath` for the site-by-site version, which is the same
        conditional sampled one coordinate at a time, and
        :class:`~.SiteOverrelaxation` for its microcanonical partner.  Neither
        overrelaxation nor sweeping helps here: an independent draw cannot be
        decorrelated further.

    Parameters
    ----------
    action: supervillain.action.Villain
        The Villain action whose $\phi$ is resampled.
    rng: numpy.random.Generator
        A source of randomness; if omitted a fresh default generator is used.
    '''

    def __init__(self, action, rng=None):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The FourierSiteHeatbath requires the Villain action.')

        self.Action = action
        self.Lattice = action.Lattice
        self.kappa = action.kappa

        self.rng = rng if rng is not None else np.random.default_rng()

        self.sweeps = 0

        # The symbol of Δ₀, read off the lattice's own operator rather than
        # rewritten by hand: Δ₀ is translation invariant, so its kernel is what it
        # does to a point source and its symbol is that kernel's transform.  This
        # cannot disagree with the operators the update actually uses, whatever
        # conventions they carry.  (It comes out as the familiar
        # $\hat k^2 = 4\sum_\mu \sin^2 k_\mu/2$.)
        L = self.Lattice
        probe = L.zeros(0)
        probe[(0,) * (L.D + 1)] = 1.
        symbol = np.fft.fftn(np.asarray(laplacian(probe))[0])
        if np.abs(symbol.imag).max() > 1e-9 * max(1., np.abs(symbol.real).max()):
            raise ValueError('the Laplacian is not symmetric; its symbol should be real.')
        symbol = symbol.real
        if symbol.min() < -1e-9 * symbol.max():
            raise ValueError('the Laplacian is not positive semi-definite.')

        # The zero mode is the constant flat direction; inverting there is
        # meaningless, and zeroing it is what leaves that mode alone.
        tiny = 1e-9 * symbol.max()
        self._inverse_laplacian = np.where(
            symbol > tiny, 1. / np.where(symbol > tiny, symbol, 1.), 0.)

    def __str__(self):
        return 'FourierSiteHeatbath'

    def conditional_mean(self, cfg):
        r'''
        The exact conditional mean $\bar\phi$ of $\phi$ at fixed $n$: the unique
        (up to the constant mode) minimizer of the action, $\Delta\bar\phi =
        2\pi\delta n$.

        It is a useful quantity in its own right, since the marginal action after
        integrating $\phi$ out is $S(\bar\phi, n)$ up to an $n$-independent
        constant.

        Parameters
        ----------
        cfg: dict
            A dictionary with phi and n as Forms.

        Returns
        -------
        supervillain.lattice.Form
            A 0-form; the constant mode is inherited from ``cfg['phi']``.
        '''
        L = self.Lattice
        force = delta(d(cfg['phi']) - 2 * np.pi * cfg['n'])
        shift = L.ifft(L.fft(force) * self._inverse_laplacian).real

        mean = L.zeros(0)
        mean[...] = np.asarray(cfg['phi']) - shift
        return mean

    def step(self, cfg):
        r'''
        Replace $\phi$ by an independent draw from $P(\phi\mid n)$, leaving $n$
        untouched.

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

        self.sweeps += 1

        mean = self.conditional_mean(cfg)

        # η has independent Fourier modes of variance 1/κ k̂².  The lattice's fft
        # is unitary, so white noise stays white under it and the fluctuation is
        # just a reweighting of its modes.  Reality is automatic: the weight is
        # even in k.
        noise = self.rng.normal(size=mean.shape)
        eta = L.ifft(L.fft(noise) * np.sqrt(self._inverse_laplacian / self.kappa)).real

        phi = L.zeros(0)
        phi[...] = np.asarray(mean) + eta
        return cfg | {'phi': phi}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'{self.sweeps} exact Fourier draws of φ.'
        )
