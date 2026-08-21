#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d

import logging
logger = logging.getLogger(__name__)

class LinkUpdate(ReadWriteable, Generator):
    r'''
    This performs the same update to $n$ as :class:`NeighborhoodUpdate <supervillain.generator.villain.NeighborhoodUpdate>` but leaves $\phi$ untouched.

    Proposals are drawn according to

    .. math ::

        \begin{aligned}
        \Delta n_\ell   &\sim W \times [-\texttt{interval\_n}, +\texttt{interval\_n}] \setminus \{0\}
        \end{aligned}

    We pick :math:`\Delta n_\ell` to be a multiple of the constraint integer $W$ so that if the adjacent plaquettes satisfy the :ref:`winding constraint <winding constraint>` $dn \equiv 0 \text{ mod }W$
    before the update they satisfy it after as well.

    .. note ::
        You can run ``python supervillain/generator/villain/link.py`` to compare a pure $W=1$ :class:`~.NeighborhoodUpdate` ensemble against an ensemble which also does :class:`LinkUpdates <LinkUpdate>`.
        Note that adding the :class:`LinkUpdate` costs essentially 0 time because all the links are done in parallel and there are no python-level for loops.
    '''

    def __init__(self, action, interval_n=1):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The LinkUpdate requires the Villain action.')

        self.Action       = action
        self.Lattice      = action.Lattice
        self.kappa        = action.kappa

        self.interval_n   = interval_n

        self.rng = np.random.default_rng()
        self.n_changes = tuple(n for n in range(-interval_n, 0)) + tuple(n for n in range(1, interval_n+1))

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def __str__(self):
        return 'LinkUpdate'

    def step(self, cfg):
        r'''
        Make volume's worth of random single-link updates to $n$.

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

        n = cfg['n'].copy()

        self.sweeps += 1

        # The n variables are all independent, in the sense that the action S doesn't couple them directly.
        # We can therefore offer updates independently, holding dphi a fixed background.
        dphi = d(cfg['phi'])
        # That lets us do a whole 1-form worth of updates simultaneously.
        change_n = S.W * self.rng.choice(self.n_changes, size=n.shape)

        # Use numpy's broadcasting to evaluate the change in S independently for each link.
        # What lets us do this so simply is that this generator does not update phi.
        # So the change in action from changing n just depends on a fixed background dphi,
        # and on n itself---no n from any other link is involved.
        dS = (
            -2 * np.pi * S.kappa * change_n
            * (dphi - 2 * np.pi * n - np.pi * change_n)
        )
        # The point is, dS can really be evaluated link-by-link if we freeze phi;
        # we're not missing any pieces that come from changing n on two nearby links at once.

        # Because the links don't talk to one another, we can accept or reject them simultaneously.
        acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)
        metropolis = self.rng.uniform(0, 1, size=acceptance.shape)
        accepted = metropolis < acceptance

        self.acceptance += acceptance.mean()
        self.accepted += accepted.sum()
        self.proposed += int(np.prod(n.shape))

        n = n + np.where(accepted, change_n, 0).astype(int)

        return cfg | {'n': n}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'There were {self.accepted} single-link proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )
