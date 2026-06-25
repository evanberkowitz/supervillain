#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d

import logging
logger = logging.getLogger(__name__)

class SiteUpdate(ReadWriteable, Generator):
    r'''
    This performs the same update to $\phi$ as :class:`NeighborhoodUpdate <supervillain.generator.villain.NeighborhoodUpdate>` but leaves $n$ untouched.

    Proposals are drawn according to

    .. math ::

        \Delta \phi_x   \sim \text{uniform}(-\texttt{interval\_phi}, +\texttt{interval\_phi})
    '''

    def __init__(self, action, interval_phi = np.pi):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('Need a Villain action')

        self.Action       = action
        self.Lattice      = action.Lattice
        self.kappa        = action.kappa

        self.interval_phi = interval_phi

        self.rng = np.random.default_rng()

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def __str__(self):
        return 'SiteUpdate'

    def step(self, cfg):
        r'''
        Make volume's worth of random single-site updates to $\phi$.

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

        phi = cfg['phi'].copy()
        n   = cfg['n']

        self.sweeps += 1
        total_accepted = 0
        total_acceptance = 0

        metropolis = self.rng.uniform(0, 1, (L.N,) * L.D)

        # The idea is to make the same sort of update to phi as the Neighborhood update gives.
        # However, rather than a python-level for loop over space, we can accomplish a lot more at the numpy level,
        # as in the villain.LinkUpdate.

        # However, in the LinkUpdate we change n, and each n contributes to an independent term in the action.
        # In contrast, what enters the action (per link) is dphi, which knows about phi on two sites.
        #
        # That poses a small problem because if we change the action by changing dphi, we want to be able to track
        # that change in dphi back to a change in phi on ONE particular site.
        # Therefore, we use checkerboarding.

        # Precompute dphi once; we update it incrementally as accepted changes accumulate.
        dphi = d(phi)

        for color in L.checkerboarding:

            # We only offer changes to phi on a single color at once.  The benefit is that the surrounding sites
            # do not have updates.  So we know where any change in the action on any link came from: it came from
            # the site in the partition (color) we are updating.
            change_phi = L.zeros(0)
            change_phi[0, *color] = self.rng.uniform(-self.interval_phi, +self.interval_phi, len(color[0]))

            # Expanding S.local(phi+Δφ, n) − S.local(phi, n) algebraically avoids two full d(phi) calls and additional arithmetic operations:
            #   κ/2 · (dφ + dΔφ − 2πn)² − κ/2 · (dφ − 2πn)²  =  κ/2 · dΔφ · (2(dφ−2πn) + dΔφ)
            change_dphi = d(change_phi)
            dS_link = (S.kappa / 2) * change_dphi * (2 * (dphi - 2 * np.pi * n) + change_dphi)

            # The change in action originating from the change in phi on the color under consideration
            # is just the sum of all the changes from the adjacent links.  face_sum collects them.
            dS = dS_link.face_sum()

            # dS is not 0 on the off-color sites---those sites still have links that land on the current color.
            # We only want to accept/reject updates on the current color.
            acceptance = np.clip(np.exp(-dS[0, *color]), a_min=0, a_max=1)
            accepted = metropolis[color] < acceptance

            total_accepted += accepted.sum()
            total_acceptance += acceptance.sum()

            # Update phi and dphi where the change is accepted.
            change_phi[0, *color] *= accepted
            phi  = phi  + change_phi
            dphi = dphi + d(change_phi)

        sites = self.Lattice.cells_of_degree[0]
        self.proposed += sites
        self.acceptance += total_acceptance / sites
        self.accepted += total_accepted

        logger.debug(f'Average proposal acceptance {total_acceptance / sites:.6f}; Actually accepted {total_accepted} / {sites} = {total_accepted / sites}')

        return cfg | {'phi': phi}

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'There were {self.accepted} single-phi proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )
