#!/usr/bin/env python

from itertools import product

import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge

import logging
logger = logging.getLogger(__name__)

_TWO_PI = 2 * np.pi


class ConstrainedLinkHeatbath(ReadWriteable, Generator):
    r"""
    The exact-heatbath counterpart of :class:`~.ConstrainedLinkUpdate`: rather than
    proposing ${n}_\ell \to {n}_\ell \pm 1$ and accepting or rejecting, it resamples each
    link's ${n}_\ell$ from its exact conditional in one shot.  It takes no step size and
    never rejects, and it leaves $\phi$ untouched.

    The move rests on the single-link linearity of the constraint.  For a shift confined
    to one link the self-wedge $d\delta_\ell\wedge d\delta_\ell \equiv 0$, so the charge
    change

    .. math ::

        \Delta q = c\,\bigl(F\wedge d\delta_\ell + d\delta_\ell\wedge F\bigr) \equiv c\, L_\ell(F)

    is *exactly linear* in the shift $c$, and $L_\ell(F)$ does not depend on $n_\ell$
    itself (it reads $F$ only on the planes complementary to $\ell$'s direction, which
    $n_\ell$ never touches).  Cleanliness is therefore a property of the background alone,
    independent of $c$, and it splits every link into exactly two cases with nothing in
    between:

    * **clean** ($L_\ell(F) = 0$): *every* integer shift preserves $q = 0$, so the
      constraint is completely inert and the exact conditional is the unconstrained
      Villain discrete Gaussian

      .. math ::

          P(n_\ell = n_\ell^0 + m) \propto \exp\!\left[-\tfrac{\kappa}{2}\left(d\phi_\ell - 2\pi n_\ell^0 - 2\pi m\right)^2\right],
          \qquad m \in \mathbb{Z},

      centered on ${m}^* = d\phi_\ell/2\pi - {n}_\ell^0$ with width
      $\sigma_m = 1/(2\pi\sqrt{\kappa})$.  We draw it exactly by Gumbel-max over a window
      of $\pm K$ integers about $m^*$, as in :class:`~.villain.LinkHeatbath`.

    * **frozen** ($L_\ell(F) \ne 0$): the conditional is a point mass at the current
      value; the link is left where it is.

    Because a clean link stays clean after any resample (its $L_\ell$ reads a background
    the resample does not change), the move is reversible link by link.  Links are swept
    in the same non-interacting **colours** as :class:`~.ConstrainedLinkUpdate` --- two
    links interact only within Chebyshev distance $1$ --- so a whole colour is resampled
    together and the field strength $F = dn$ is patched after each colour.

    .. note ::

        Restricted to $D = 4$.  Updates $n$ only; combine with a $\phi$-update such as
        :class:`~.villain.SiteHeatbath`.

    .. seealso ::

        :class:`~.ConstrainedLinkUpdate` is the Metropolis version, whose stationary
        distribution on the clean links is the same discrete Gaussian sampled here.
        :class:`~.villain.LinkHeatbath` is the unconstrained Villain heatbath this reduces
        to on the clean subset.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The No-Intersection action whose $n$ is resampled.
    rng: numpy.random.Generator
        A source of randomness; if omitted a fresh default generator is used.
    coverage_sigmas: float
        How many standard deviations $\sigma_m = 1/(2\pi\sqrt{\kappa})$ of the clean
        conditional the enumeration window covers before the discrete-Gaussian sum is
        truncated.  The integer half-window $K$ is derived from this and the width; the
        default 6 leaves a negligible tail and rarely needs changing.
    """

    def __init__(self, S, rng=None, coverage_sigmas=6.0):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('ConstrainedLinkHeatbath requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('ConstrainedLinkHeatbath is only implemented for D = 4.')

        self.Action = S
        self.Lattice = S.Lattice
        self.kappa = S.kappa

        self.rng = rng if rng is not None else np.random.default_rng()
        self.coverage_sigmas = coverage_sigmas

        # Derived integer half-window K, from the clean conditional's width
        # σ_m = 1/(2π √κ) (the W=1 special case of the Villain LinkHeatbath window);
        # the user knob is coverage_sigmas, not K.
        sigma_m = 1.0 / (_TWO_PI * np.sqrt(self.kappa))
        self.K = int(np.ceil(self.coverage_sigmas * sigma_m)) + 2

        self.clean = 0          # links found clean (resampleable)
        self.resampled = 0      # clean links whose value actually changed
        self.proposed = 0       # links visited
        self.sweeps = 0

    def __str__(self):
        return 'ConstrainedLinkHeatbath'

    def _resample_shift(self, n0, Aell, kappa, K):
        r"""
        The exact discrete-Gaussian shift $m$ for a whole colour grid of clean links,
        drawn by Gumbel-max over a $\pm K$ window about the coset centre.  ``n0`` is the
        current $n_\ell$ on the colour, ``Aell`` the background $d\phi_\ell$; the returned
        integer array is the shift $m$ such that the resampled value is ``n0 + m``.
        """
        m_round = np.round(Aell / _TWO_PI - n0).astype(int)
        offsets = np.arange(-K, K + 1)
        m_cand = m_round[..., None] + offsets                 # (...links..., 2K+1)
        resid = Aell[..., None] - _TWO_PI * (n0[..., None] + m_cand)
        logw = -0.5 * kappa * resid**2
        gumbel = -np.log(-np.log(self.rng.uniform(size=logw.shape)))
        return m_round + (np.argmax(logw + gumbel, axis=-1) - K)

    def step(self, cfg):
        r"""
        One sweep resampling every *clean* link's $n_\ell$ from its exact discrete-Gaussian
        conditional and leaving every *frozen* link fixed, then patching $F = dn$.

        The links are partitioned into the same non-interacting **colours** as
        :meth:`~.ConstrainedLinkUpdate.step`: a single-link change touches $q$ only within
        one lattice step, so same-colour links are conditionally independent and resampling
        them together equals resampling them one at a time.  Cleanliness is read off the
        carried $F$ with the local charge stencil (:func:`~.clean_mask_for_color`); the
        discrete-Gaussian draw is the $W = 1$ Villain conditional restricted to those links.

        :meth:`step_reference` is the plain, obviously-correct version this is validated
        against.
        """
        L = self.Lattice
        N = L.N
        kappa, K = self.kappa, self.K

        n = np.asarray(cfg['n']).astype(np.int64)
        A = np.asarray(d(cfg['phi']))                       # fixed background dφ 1-form
        F = np.asarray(d(cfg['n'])).astype(np.int64)        # maintained across the sweep

        self.sweeps += 1
        axis = local_charge.axis_colors(N)

        # Visit the colours in a fresh random order each sweep.  Any order is stationary ---
        # each colour's resample preserves the target on its own --- but when the constraint
        # bites, mobility is scarce and whichever direction is always offered first gets
        # first claim on the clean links, which reads as an anisotropy in directional
        # quantities (the winding, TorusWrapping) though the physics is isotropic.  It also
        # restores reversibility of the sweep as a whole, which a fixed order loses even
        # when every colour is individually detailed-balanced.
        colours = [(mu, choice) for mu in range(4)
                   for choice in product(range(len(axis)), repeat=4)]
        for index in self.rng.permutation(len(colours)):
            mu, choice = colours[index]
            idx = [axis[choice[a]] for a in range(4)]
            sub = np.ix_(*idx)

            clean = local_charge.clean_mask_for_color(F, mu, idx, N)
            self.proposed += clean.size
            self.clean += int(clean.sum())

            m = self._resample_shift(n[mu][sub], A[mu][sub], kappa, K)
            flip = np.where(clean, m, 0)                # resample clean links, freeze the rest
            local_charge.apply_color(n, F, mu, idx, N, flip)
            self.resampled += int(np.count_nonzero(flip))

        return cfg | {'n': Form(n, degree=1, lattice=L)}

    def step_reference(self, cfg):
        r"""
        Reference sweep: the plain, obviously-correct implementation.

        Visits every link in random order; a link is *clean* iff a global ``charge``
        recompute of the trial $n_\ell \to n_\ell + 1$ still vanishes everywhere (cleanliness
        is independent of the shift, so $+1$ decides it).  A clean link is resampled from its
        discrete-Gaussian conditional; a frozen link is left untouched.  Kept as the
        correctness oracle the vectorized :meth:`step` is validated against.
        """
        L = self.Lattice
        N = L.N
        D = L.D
        kappa, K = self.kappa, self.K

        n = cfg['n'].copy()
        dphi = d(cfg['phi'])

        self.sweeps += 1

        links = [(mu,) + tuple(int(x) for x in site)
                 for mu in range(D)
                 for site in np.ndindex(*((N,) * D))]
        self.rng.shuffle(links)

        for link in links:
            self.proposed += 1

            trial = n.copy()
            trial[link] += 1
            if not np.allclose(charge(trial), 0):           # frozen: leave it
                continue
            self.clean += 1

            n0 = int(n[link])
            m = int(self._resample_shift(np.array(n0), np.array(dphi[link]), kappa, K))
            if m != 0:
                n[link] = n0 + m
                self.resampled += 1

        return cfg | {'n': Form(np.asarray(n), degree=1, lattice=L)}

    def inline_observables(self, steps):
        return {}

    def report(self):
        if self.proposed == 0:
            return 'There were 0 links visited by the ConstrainedLinkHeatbath.'
        return (
            f'ConstrainedLinkHeatbath: {self.sweeps} exact heatbath sweeps of n '
            f'(window ±{self.K}, no accept/reject).'
            + '\n' +
            f'    {self.clean / self.proposed:.6f} clean (resampleable) fraction'
            + '\n' +
            f'    {self.resampled / self.proposed:.6f} fraction of links moved'
        )
