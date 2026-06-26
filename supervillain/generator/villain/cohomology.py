#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d

import logging
logger = logging.getLogger(__name__)

class CohomologyUpdate(ReadWriteable, Generator):
    r'''
    Local updates generate only exact changes $\Delta n = dk$, which have zero cohomology class.
    No combination of them can change the global winding numbers

    .. math::

        w_\mu = \sum_{x_\mu} n_\mu(x_\mu, x_\perp)

    which are $x_\perp$-independent whenever $dn = 0$.
    These winding numbers label sectors of $H^1(\mathbb{T}^D, \mathbb{Z}) = \mathbb{Z}^D$.

    This update proposes, for each direction $\mu$, adding a constant $h_\mu$ to $n_\mu$ on the
    single slice $x_\mu = 0$ (all perpendicular positions).  Because $\Delta{}n_\mu$ is constant
    on that slice and zero elsewhere, $d(\Delta n) = 0$ exactly, so the constraint
    $dn \equiv 0\; (\text{mod}\; W)$ is preserved automatically for any $W$.
    The winding number $w_\mu$ changes by $h_\mu$.

    Proposals are drawn from

    .. math::

        h_\mu \sim [-\texttt{interval\_h},\, +\texttt{interval\_h}] \setminus \{0\}

    One independent scalar $h_\mu$ is drawn and Metropolized per direction, changing $N^{D-1}$
    links at once.  Acceptance is typically low at large $\kappa$ because the action barrier
    scales as $O(\kappa N^{D-1})$ — this is the genuine physical cost of tunneling between
    winding sectors.

    '''

    def __init__(self, action, interval_h=1):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('Need a Villain action')

        self.Action       = action
        self.Lattice      = action.Lattice
        self.kappa        = action.kappa

        self.interval_h = interval_h
        self.h = tuple(h for h in range(-interval_h, 0)) + tuple(h for h in range(1, interval_h + 1))

        self.rng = np.random.default_rng()

        self.accepted   = 0
        self.proposed   = 0
        self.acceptance = 0.
        self.sweeps     = 0

    def __str__(self):
        return 'CohomologyUpdate'

    def step(self, configuration):
        r'''
        Offer one winding-sector update per direction.

        For each $\mu$, draws a scalar $h_\mu$, computes $\Delta S$ on the $N^{D-1}$ links of
        the slice $x_\mu = 0$, and Metropolizes.  The D directions are independent and processed
        sequentially; accepted changes are applied incrementally to the residual
        $r = d\phi - 2\pi n$ so that later directions see the updated residuals.

        Parameters
        ----------
        configuration: dict
            A dictionary with :math:`\phi` and :math:`n` as Forms.

        Returns
        -------
        dict
            Updated configuration.
        '''
        S = self.Action
        L = S.Lattice

        n = configuration['n'].copy()
        r = d(configuration['phi']) - 2 * np.pi * n

        self.sweeps += 1
        total_acceptance = 0
        total_accepted   = 0

        for mu in range(L.D):
            h = self.rng.choice(self.h)

            # Select n[mu] and r[mu] on the slice x_mu = 0 (all perpendicular positions).
            slice_idx = (mu,) + tuple(0 if i == mu else slice(None) for i in range(L.D))

            change_r  = -2 * np.pi * h
            r_slice   = r[slice_idx]

            dS = float(((S.kappa / 2) * change_r * (2 * r_slice + change_r)).sum())

            acceptance = float(np.clip(np.exp(-dS), 0, 1))
            if self.rng.uniform(0, 1) < acceptance:
                n[slice_idx] += h
                r[slice_idx] += change_r
                total_accepted += 1

            total_acceptance += acceptance

        self.proposed   += L.D
        self.acceptance += total_acceptance / L.D
        self.accepted   += total_accepted
        logger.debug(f'Accepted {total_accepted}/{L.D} cohomology proposals')

        return configuration | {'n': n}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'There were {self.accepted} cohomology proposals accepted of {self.proposed} proposed updates.'
            + '\n' +
            f'    {self.accepted / self.proposed:.6f} acceptance rate'
            + '\n' +
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )
