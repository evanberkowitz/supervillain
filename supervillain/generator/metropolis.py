#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.h5 import H5able

import logging
logger = logging.getLogger(__name__)

class SlowNeighborhoodUpdate(H5able):
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

    .. warning ::
        Because we currently restrict to $W=1$ for the Villain formulation we do not update $v$.

    Parameters
    ----------
    action: Villain
        The action from which we sample.
    interval_phi: float
        A single float used to construct the uniform distribution for $\phi$.
    interval_n: int
        A single integer that gives the biggest allowed changes to $n$.
    '''

    def __init__(self, action, interval_phi=np.pi, interval_n=1):

        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The Neighborhood Metropolis update requires the Villain action.')
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

        # We move the lattice around (which is fine by translational symmetry)
        # so that we update a different site with each proposal.
        # The advantage of thinking this way is that we only have to reckon from the origin.
        phi = L.roll(cfg['phi'].copy(), dx)
        n   = L.roll(cfg['n'].copy(),   dx)

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


class NeighborhoodUpdate(H5able):
    r'''
    This performs the same update as :class:`SlowNeighborhoodUpdate <supervillain.generator.metropolis.SlowNeighborhoodUpdate>` but is streamlined to eliminate calls, to calculate the change in action directly, and to avoid data movement.

    .. note ::
       On a small 5×5 example this generator yields about three times as many updates per second than :class:`SlowNeighborhoodUpdate <supervillain.generator.metropolis.SlowNeighborhoodUpdate>` on my machine!
       This ratio should *improve* for larger lattices because the change in action is computed directly and is of fixed cost, rather than scaling with the volume.

    .. warning ::
        Because we currently restrict to $W=1$ for the Villain formulation we do not update $v$.
    '''

    def __init__(self, action, interval_phi=np.pi, interval_n=1):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The Neighborhood Metropolis update requires the Villain action.')
        self.Action       = action
        self.Lattice      = action.Lattice
        self.kappa        = action.kappa

        self.interval_phi = interval_phi
        self.interval_n   = interval_n

        self.rng = np.random.default_rng()
        self.n_changes = np.arange(-interval_n, 1+interval_n)

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

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
        total_acceptance = 0
        accepted = 0

        phi = cfg['phi'].copy()
        n   = cfg['n'].copy()

        # Rather than sweeping the lattice in a particular order, we randomly update sites.
        sites = np.stack((
            np.random.randint(self.Lattice.dims[0], size=self.Lattice.sites),
            np.random.randint(self.Lattice.dims[1], size=self.Lattice.sites)
        )).transpose()

        for here, metropolis in zip(sites, self.rng.uniform(0,1,len(sites))):
            # Rather than leveraging translational symmetry and reckoning from the origin,
            # it is faster to do a little bit of index arithmetic and avoid all the data movement.
            # This is particularly noticable on large lattices.
            north, south, east, west = self.Lattice.mod(here + np.array([[+1,0],[-1,0],[0,-1],[0,+1]]))
                # Since time is the zeroeth axis, *west* is the positive space direction.

            change_phi = self.rng.uniform(-self.interval_phi,+self.interval_phi,None)
            change_n = self.rng.choice(self.n_changes,4)

            # We don't even construct a new field until we know whether we know we'll accept or reject.
            # We can calculate dS directly from just the previous values and the proposed changes.
            # This formula is the application of the difference of two squares for each changed link.
            dS = 0.5*self.kappa*(
                +(-change_phi-2*np.pi*change_n[0])*(2*(phi[north[0],north[1]]-phi[here [0],here [1]]-2*np.pi*n[0][here [0],here [1]])-change_phi-2*np.pi*change_n[0])
                +(+change_phi-2*np.pi*change_n[1])*(2*(phi[here [0],here [1]]-phi[south[0],south[1]]-2*np.pi*n[0][south[0],south[1]])+change_phi-2*np.pi*change_n[1])
                +(-change_phi-2*np.pi*change_n[2])*(2*(phi[west [0],west [1]]-phi[here [0],here [1]]-2*np.pi*n[1][here [0],here [1]])-change_phi-2*np.pi*change_n[2])
                +(+change_phi-2*np.pi*change_n[3])*(2*(phi[here [0],here [1]]-phi[east [0],east [1]]-2*np.pi*n[1][east [0],east [1]])+change_phi-2*np.pi*change_n[3])
            )

            # Now we Metropolize
            acceptance = np.clip( np.exp(-dS), a_min=0, a_max=1)
            total_acceptance += acceptance
            if metropolis < acceptance:
                logger.debug(f'Proposal accepted; ∆S = {dS:f}; acceptance probability = {acceptance:f}')
                accepted += 1
                # and conditionally update the configuration.
                phi [here [0],here [1]] += change_phi
                # These assignments are picked to match the unrolled dS calculation.
                n[0][here [0],here [1]] += change_n[0]
                n[0][south[0],south[1]] += change_n[1]
                n[1][here [0],here [1]] += change_n[2]
                n[1][east [0],east [1]] += change_n[3]
            else:
                logger.debug(f'Proposal rejected; ∆S = {dS:f}; acceptance probability = {acceptance:f}')

        self.accepted += accepted
        self.proposed += len(sites)

        total_acceptance /= len(sites)
        self.acceptance += total_acceptance
        logger.info(f'Average proposal {acceptance=:.6f}; Actually {accepted = } / {self.Action.Lattice.sites} = {accepted / self.Action.Lattice.sites}')

        return {'phi': phi, 'n': n}

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

