#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable

import logging
logger = logging.getLogger(__name__)

class HolonomyUpdate(ReadWriteable, Generator):
    r'''
    The :class:`~.villain.ExactUpdate` can change $n$ by ±1 (even when $W>1$), but it does it in a coordinated way---the changes offered are exact, d(a zero form).
    No combination of exact updates, however, can create a net winding around the torus.

    This update offers a $dn$-preserving update by changing all of the links in a parallel strip around the lattice (see Fig. 2 of Ref. :cite:`Berkowitz:2023pnz`).

    Proposals change a whole strip of links simultaneously by

    .. math ::

        \begin{align}
        h   &\sim [-\texttt{interval_h}, +\texttt{interval_h}] \setminus \{0\}
        \end{align}

    Because the proposal touches a (linearly) extensive number of variables, this update may frequently be rejected.
    '''

    def __init__(self, action, interval_h = 1):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('Need a Villain action')

        self.Action       = action
        self.Lattice      = action.Lattice
        self.kappa        = action.kappa

        self.interval_h = interval_h
        self.h = tuple(h for h in range(-interval_h, 0)) + tuple(h for h in range(1, interval_h+1))

        self.rng = np.random.default_rng()

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def __str__(self):
        return 'HolonomyUpdate'

    def step(self, cfg):
        r'''
        Offer independent wrapping updates across the whole torus.

        Parameters
        ----------
        cfg: dict
            A dictionary with phi and n field variables.

        Returns
        -------
        dict
            Another configuration of fields.
        '''

        self.sweeps += 1
        total_acceptance = 0
        accepted = 0

        phi = cfg['phi'].copy()
        dphi = self.Lattice.d(0, phi)

        n   = cfg['n'].copy()

        L = self.Lattice


        # Each strip of parallel links we can change the holonomy in a different way.
        change_n = L.form(1, dtype=int)
        change_n[0] = self.rng.choice(self.h, L.nt)[:,None]
        change_n[1] = self.rng.choice(self.h, L.nx)[None,:]
        # assert self.Action.valid(change_n)

        # The change in action on every link is simply
        #dS_link = 0.5 * self.Action.kappa * (-2*np.pi*change_n) * (2*(dphi - 2*np.pi*n) - 2*np.pi*change_n)
        dS_link = -2*np.pi * self.Action.kappa * change_n * ((dphi - 2*np.pi*n) - np.pi*change_n)

        # We need to Metropolis-accept or -reject the whole strip at once.
        # So, we sum the changes in action across the strips; first the temporal links.
        dS = dS_link[0].sum(axis=1)
        # Now we Metropolize
        acceptance = np.clip( np.exp(-dS), a_min=0, a_max=1)
        metropolis = self.rng.uniform(0, 1, acceptance.shape)
        accepted = metropolis < acceptance
        # and zero out the rejected changes.
        change_n[0] *= accepted[:,None]

        total_acceptance = acceptance.sum()
        total_accepted   = accepted.sum()

        # Then, the spatial links.
        dS = dS_link[1].sum(axis=0)
        acceptance = np.clip( np.exp(-dS), a_min=0, a_max=1)
        metropolis = self.rng.uniform(0, 1, acceptance.shape)
        accepted = metropolis < acceptance
        change_n[1] *= accepted[None, :]

        total_acceptance += acceptance.sum()
        total_accepted   += accepted.sum()

        # Now change_n only contains Metropolized changes, and we can add it to n.
        n += change_n

        self.proposed += L.nx + L.nt
        self.acceptance += total_acceptance / (L.nx + L.nt)
        self.accepted += total_accepted

        logger.debug(f'Average proposal acceptance {total_acceptance / (L.nx + L.nt):.6f}; Actually accepted {total_accepted} / {(L.nx + L.nt)} = {total_accepted / (L.nx + L.nt)}')

        return {'phi': phi, 'n': n}



    def report(self):
        return (
            f'There were {self.accepted} holonomy proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )

