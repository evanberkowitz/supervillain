#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice.compact import d

import logging
logger = logging.getLogger(__name__)

class SiteUpdate(ReadWriteable, Generator):
    r'''
    This performs the same update to $\phi$ as :class:`NeighborhoodUpdate <supervillain.generator.villain.NeighborhoodUpdate>` but leaves $n$ untouched.

    Proposals are drawn according to

    .. math ::

        \Delta \phi_x   \sim \text{uniform}(-\texttt{interval_phi}, +\texttt{interval_phi})
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
            A dictionary with phi and n as compact Forms.

        Returns
        -------
        dict
            Updated phi field only (to be merged by the caller).
        '''
        S = self.Action
        L = S.Lattice

        phi = cfg['phi'].copy()

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
        for color in L.checkerboarding:

            # We only offer changes to phi on a single color at once.  The benefit is that the surrounding sites
            # do not have updates.  So we know where any change in the action on any link came from: it came from
            # the site in the partition (color) we are updating.
            change_phi = L.zeros(0)
            change_phi[0, *color] = self.rng.uniform(-self.interval_phi, +self.interval_phi, len(color[0]))

            # dphi changes in the obvious way, and then dphi changes the action on every link.
            # The change in action originating from the change in phi on the color under consideration
            # is just the sum of all the changes from the adjacent links.  face_sum collects them.
            change_S_local = (
                S.local(phi + change_phi, cfg['n'])
                - S.local(phi, cfg['n'])
            ).face_sum()

            # Now dS is a 0-form encoding the changes in action from change_phi.  But we should be careful:
            # dS is not 0 on the off-color sites---those sites still have links that land us on the current color.
            # We only want to accept/reject updates on the current color, so we restrict our attention when computing the acceptance.
            acceptance = np.clip(np.exp(-change_S_local[(slice(None), *color)]), a_min=0, a_max=1)
            accepted = metropolis[color] < acceptance[0]

            total_accepted += accepted.sum()
            total_acceptance += acceptance[0].sum()

            # Finally, we update the phi where the change is accepted.
            change_phi[0, *color] *= accepted
            phi = phi + change_phi

        sites = self.Lattice.sites
        self.proposed += sites
        self.acceptance += total_acceptance / sites
        self.accepted += total_accepted

        logger.debug(f'Average proposal acceptance {total_acceptance / sites:.6f}; Actually accepted {total_accepted} / {sites} = {total_accepted / sites}')

        return {'phi': phi}

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
