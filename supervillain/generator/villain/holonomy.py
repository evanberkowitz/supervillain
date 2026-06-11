#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice.compact import d

import logging
logger = logging.getLogger(__name__)

class HolonomyUpdate(ReadWriteable, Generator):
    r'''
    The :class:`~.villain.ExactUpdate` can change $n$ by ±1 (even when $W>1$), but it does it in a coordinated way---the changes offered are exact, d(a zero form).
    No combination of exact updates, however, can create a net winding around the torus.

    This update offers a $dn$-preserving update by changing all of the links in a parallel strip around the lattice (see Fig. 2 of Ref. :cite:`Berkowitz:2023pnz`).

    Proposals change a whole strip of links simultaneously by

    .. math ::

        \begin{aligned}
        h   &\sim [-\texttt{interval\_h}, +\texttt{interval\_h}] \setminus \{0\}
        \end{aligned}

    Because the proposal touches a (linearly) extensive number of variables, this update may frequently be rejected.

    .. todo::
        Generalize to D>2. In D dimensions there are D independent holonomy directions, each requiring a strip of
        co-dimension 1. The current implementation is hardcoded for D=2 and raises :exc:`NotImplementedError` otherwise.
    '''

    def __init__(self, action, interval_h = 1):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('Need a Villain action')
        if action.Lattice.D != 2:
            raise NotImplementedError('HolonomyUpdate is only implemented for D=2')

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
            A dictionary with phi and n as Forms.

        Returns
        -------
        dict
            Updated configuration.
        '''
        S = self.Action
        L = S.Lattice

        n = cfg['n'].copy()

        self.sweeps += 1
        total_acceptance = 0
        total_accepted = 0

        dphi = d(cfg['phi'])

        # Each strip of parallel links we can change the holonomy in a different way.
        # For D=2: direction 0 = t, direction 1 = x.
        # n[0] has shape (N, N): t-direction links, one strip per t-row.
        # n[1] has shape (N, N): x-direction links, one strip per x-column.
        change_n = L.zeros(1, dtype=int)
        change_n[0] = self.rng.choice(self.h, L.N)[:, None]
        change_n[1] = self.rng.choice(self.h, L.N)[None, :]

        links = dphi - 2 * np.pi * n   # gauge-invariant link variable

        dS_link = -2 * np.pi * S.kappa * change_n * (links - np.pi * change_n)

        # We need to Metropolis-accept or -reject the whole strip at once.
        # So, we sum the changes in action across the strips; first the temporal links.
        dS = dS_link[0].sum(axis=1)   # shape (N,) — one per t-row
        acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)
        metropolis = self.rng.uniform(0, 1, acceptance.shape)
        accepted = metropolis < acceptance
        # Zero out the rejected changes before accumulating.
        change_n[0] *= accepted[:, None]

        total_acceptance += acceptance.sum()
        total_accepted += accepted.sum()

        # Then, the spatial links.
        dS = dS_link[1].sum(axis=0)   # shape (N,) — one per x-column
        acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)
        metropolis = self.rng.uniform(0, 1, acceptance.shape)
        accepted = metropolis < acceptance
        change_n[1] *= accepted[None, :]

        total_acceptance += acceptance.sum()
        total_accepted += accepted.sum()

        # Now change_n only contains Metropolized changes, and we can add it to n.
        n = n + change_n

        nx, nt = self.Lattice.nx, self.Lattice.nt
        self.proposed += nx + nt
        self.acceptance += total_acceptance / (nx + nt)
        self.accepted += total_accepted

        logger.debug(f'Average proposal acceptance {total_acceptance / (nx + nt):.6f}; Actually accepted {total_accepted} / {(nx + nt)} = {total_accepted / (nx + nt)}')

        return cfg | {'n': n}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'There were {self.accepted} holonomy proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )
