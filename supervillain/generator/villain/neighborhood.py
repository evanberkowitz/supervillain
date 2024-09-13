#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.h5 import ReadWriteable
from supervillain.generator import Generator

import logging
logger = logging.getLogger(__name__)

class NeighborhoodUpdate(ReadWriteable, Generator):
    r'''
    This performs the same update as :class:`NeighborhoodUpdateSlow <supervillain.generator.reference_implementation.villain.NeighborhoodUpdateSlow>` but is streamlined to eliminate calls, to calculate the change in action directly, and to avoid data movement.

    Proposals are drawn according to

    .. math ::
    
        \begin{align}
        \Delta\phi_x    &\sim \text{uniform}(-\texttt{interval_phi}, +\texttt{interval_phi})
        \\
        \Delta n_\ell   &\sim W \times [-\texttt{interval_n}, +\texttt{interval_n}]
        \end{align}

    We pick :math:`\Delta n_\ell` to be a multiple of the constraint integer $W$ so that if the adjacent plaquettes satisfy the :ref:`winding constraint <winding constraint>` $dn \equiv 0 \text{ mod }W$
    before the update they satisfy it after as well.

    .. seealso ::
       On a small 5×5 example this generator yields about three times as many updates per second than :class:`NeighborhoodUpdateSlow <supervillain.generator.reference_implementation.villain.NeighborhoodUpdateSlow>` on my machine.
       This ratio should *improve* for larger lattices because the change in action is computed directly and is of fixed cost, rather than scaling with the volume.

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

    def __str__(self):
        return 'NeighborhoodUpdate'

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

            # TODO: consider drawing the change in phi from W * the below interval
            change_phi =                 self.rng.uniform(-self.interval_phi,+self.interval_phi,None)
            change_n   = self.Action.W * self.rng.choice(self.n_changes,4)

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
        logger.debug(f'Average proposal {acceptance=:.6f}; Actually {accepted = } / {self.Action.Lattice.sites} = {accepted / self.Action.Lattice.sites}')

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

