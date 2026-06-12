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

    This update offers a $dn$-preserving update by changing all of the links in a cycle around the torus in each direction (see Fig. 2 of Ref. :cite:`Berkowitz:2023pnz`).
    In each direction $\mu$ there are $N^{D-1}$ independent cycles (one per perpendicular position), each proposed simultaneously.

    Proposals change a whole cycle of links simultaneously by

    .. math ::

        \begin{aligned}
        h   &\sim [-\texttt{interval\_h}, +\texttt{interval\_h}] \setminus \{0\}
        \end{aligned}

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
        links = dphi - 2 * np.pi * n   # gauge-invariant link variable

        # In each direction mu there are N^(D-1) independent mu-cycles, one per perpendicular position.
        # Each cycle gets an independent proposed change h, constant along the mu-direction.
        # The perpendicular shape has size 1 in direction mu (broadcast) and N in all other directions.
        change_n = L.zeros(1, dtype=int)
        for mu in range(L.D):
            perp_shape = tuple(1 if i == mu else L.N for i in range(L.D))
            change_n[mu] = self.rng.choice(self.h, L.N**(L.D-1)).reshape(perp_shape)

        dS_link = -2 * np.pi * S.kappa * change_n * (links - np.pi * change_n)

        # Accept or reject each cycle independently: sum the per-link action change along the mu-direction
        # to get one dS per cycle, then Metropolize and zero out rejected changes.
        for mu in range(L.D):
            dS = dS_link[mu].sum(axis=mu)   # one value per perpendicular position
            acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)
            metropolis = self.rng.uniform(0, 1, acceptance.shape)
            accepted = metropolis < acceptance
            change_n[mu] *= np.expand_dims(accepted, axis=mu)

            total_acceptance += acceptance.sum()
            total_accepted += accepted.sum()

        # Now change_n only contains Metropolized changes, and we can add it to n.
        n = n + change_n

        n_cycles = L.D * L.N**(L.D-1)
        self.proposed += n_cycles
        self.acceptance += total_acceptance / n_cycles
        self.accepted += total_accepted

        logger.debug(f'Average proposal acceptance {total_acceptance / n_cycles:.6f}; Actually accepted {total_accepted} / {n_cycles} = {total_accepted / n_cycles}')

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
