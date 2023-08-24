#!/usr/bin/env python

import numpy as np

import logging
logger = logging.getLogger(__name__)

class NeighborhoodUpdate:
    r'''
    A neighborhood update changes only fields in some small area of the lattice.

    In particular, this updating scheme changes the $\phi$ and $n$ fields in the :class:`~.Villain` formulation.

    It works by picking a site $x$ at random, proposing a change 

    .. math ::
        
        \begin{align}
        \Delta\phi_x    &\sim \text{uniform}(-\texttt{interval_phi}, +\texttt{interval_phi})
        \\
        \Delta n_\ell   &\sim [-\texttt{interval_n}, +\texttt{interval_n}]
        \end{align}

    for the $\phi$ on $x$ and $n$ on links $\ell$ which touch $x$.
    '''

    def __init__(self, action, interval_phi=np.pi, interval_n=1):
        self.Action       = action
        self.interval_phi = interval_phi
        self.interval_n   = interval_n

        self.rng = np.random.default_rng()
        self.n_changes = np.arange(-interval_n, 1+interval_n)

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def proposal(self, cfg, dx):
        r'''
        Parameters
        ----------
        cfg: dict
            A dictionary with $\phi$ and $n$ to update.
        dx: Lattice coordinates
            Which site to move to the origin and update.

        Returns
        -------
        dict:
            A new configuration with updated $\phi$ and $n$.
        '''
        L = self.Action.Lattice

        phi = L.roll(cfg['phi'], dx)
        n   = L.roll(cfg['n'],   dx)

        phi[0,0] += self.rng.uniform(-self.interval_phi,+self.interval_phi,None)

        n_shift = self.rng.choice(self.n_changes,4)
        n[0][+0,+0] += n_shift[0]
        n[0][-1,+0] += n_shift[1]
        n[1][+0,+0] += n_shift[2]
        n[1][+0,-1] += n_shift[3]

        return {'phi': phi, 'n': n}

    def site(self, cfg, dx):
        r'''
        Rather than accepting every :func:`~proposal` we perform *importance sampling* by doing a Metropolis accept/reject step :cite:`Metropolis` on every single-site proposal.

        Parameters
        ----------
        cfg: dict
            A dictionary with $\phi$ and $n$ to update.
        dx: Lattice coordinates
            Which site to move to the origin and update.

        Returns
        -------
        dict:
            A configuration; either the provided one a new one changed by a proposal.
        float:
            The Metropolis-Hastings acceptance probability.
        int:
            1 if the proposal was accepted, 0 otherwise.
        '''

        S_start    = self.Action(cfg['phi'], cfg['n'])
        proposal   = self.proposal(cfg, dx)
        S_proposal = self.Action(proposal['phi'], proposal['n'])

        dS = S_proposal - S_start

        acceptance = np.clip( np.exp(-dS), a_min=0, a_max=1)
        metropolis = np.random.default_rng().uniform(0,1,None)
        if metropolis < acceptance:
            logger.debug(f'Proposal accepted; ∆S = {dS:f}; acceptance probability = {acceptance:f}')
            return proposal, acceptance, 1
        else:
            logger.debug(f'Proposal rejected; ∆S = {dS:f}; acceptance probability = {acceptance:f}')
            return cfg, acceptance, 0

    def step(self, cfg):
        r'''
        Make volume's worth of random single-site updates.

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
        current = cfg
        acceptance = 0
        accepted = 0

        L = self.Action.Lattice

        shifts = np.stack((
            np.random.randint(L.dims[0], size=L.sites),
            np.random.randint(L.dims[1], size=L.sites)
        )).transpose()

        for dx in shifts:
            subsequent, probability, acc = self.site(current, dx)
            current = subsequent
            acceptance += probability
            accepted   += acc

        self.accepted += accepted
        self.proposed += len(shifts)

        acceptance /= len(shifts)
        self.acceptance += acceptance
        logger.info(f'Average proposal {acceptance=:.6f}; Actually {accepted = } / {self.Action.Lattice.sites} = {accepted / self.Action.Lattice.sites}')

        return current

    def report(self):
        r'''
        Returns a string with some summarizing statistics.
        '''
        return (
            f'There were {self.accepted} single-site proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate' 
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )
