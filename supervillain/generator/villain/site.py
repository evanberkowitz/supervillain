#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable

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

        L = self.Lattice

        metropolis = self.rng.uniform(0, 1, phi.shape)
        total_accepted = 0
        total_acceptance = 0

        # The idea is to make the same sort of update to phi as the Neighborhood update gives.
        # However, rather than a python-level for loop over space, we can accomplish a lot more at the numpy level,
        # as in the villain.LinkUpdate.

        # However, in the LinkUpdate we change n, and each n contributes to an independent term in the action.
        # In constrast, what enters the action (per link) is dphi, which knows about phi on two sites.
        #
        # That poses a small problem because if we change the action by changing dphi, we want to be able to track
        # that change in dphi back to a change in phi on ONE particular site.
        # Therefore, we use checkerboarding.
        for color in L.checkerboarding:

            dphi = self.Lattice.d(0, phi)

            # We only offer changes to phi on a single color at once.  The benefit is that the surrounding sites
            # do not have updates.  So we know where any change in the action on any link came from: it came from
            # the site in the partition (color) we are updating.
            change_phi = L.form(0)
            change_phi[color] = self.rng.uniform(-self.interval_phi,+self.interval_phi, len(color[0]))

            # dphi changes in the obvious way, and then dphi changes the action on every link.
            change_dphi = L.d(0, change_phi)
            dS_link = 0.5 * self.Action.kappa * change_dphi * (2*(dphi - 2*np.pi*n) + change_dphi)

            # The change in action originating from the change in phi on the color under consideration
            # is just the sum of all the changes from the adjacent links.  So we sum them up.
            dS = dS_link[0] + dS_link[1] + L.roll(dS_link[0], (+1, 0)) + L.roll(dS_link[1], (0, +1))

            # Now dS is a 0-form encoding the changes in action from change_phi.  But we should be careful:
            # dS is not 0 on the off-color sites---those sites still have links that land us on the current color.
            # We only want to accept/reject updates on the current color, so we restrict our attention when computing the acceptance.
            acceptance = np.clip( np.exp(-dS[color]), a_min=0, a_max=1)
            accepted = (metropolis[color] < acceptance)

            total_accepted += accepted.sum()
            total_acceptance += acceptance.sum()

            # Finally, we update the phi where the change is accepted.
            phi[color] += np.where(accepted, change_phi[color], 0)

        self.proposed += L.sites
        self.acceptance += total_acceptance / L.sites
        self.accepted += total_accepted

        logger.debug(f'Average proposal acceptance {total_acceptance / L.sites:.6f}; Actually accepted {total_accepted} / {L.sites} = {total_accepted / L.sites}')

        return {'phi': phi, 'n': n}



    def report(self):
        return (
            f'There were {self.accepted} single-phi proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )

