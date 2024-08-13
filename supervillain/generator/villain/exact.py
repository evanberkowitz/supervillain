#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable

import logging
logger = logging.getLogger(__name__)

class ExactUpdate(ReadWriteable, Generator):
    r'''
    The :class:`~.villain.LinkUpdate` only updates n by multiples of $W$ in order to preserve the constraint $dn = 0\; (\text{mod } W$).
    Another way to preserve the constraint is to update n around a given site in a coordinated way so that $dn$ is not changed on any of the
    neighboring plaquettes.

    One way to accomplish this coordinated change is to start with a zero-form $z$ and update $n$ by $dz$.  Then, $dn = d^2z = 0$.

    Proposals are drawn according to

    .. math ::

        \begin{align}
        z_x   &\sim [-\texttt{interval_z}, +\texttt{interval_z}] \setminus \{0\}
        \end{align}
    '''

    def __init__(self, action, interval_z = 1):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('Need a Villain action')

        self.Action       = action
        self.Lattice      = action.Lattice
        self.kappa        = action.kappa

        self.interval_z = interval_z
        self.zs = tuple(z for z in range(-interval_z, 0)) + tuple(z for z in range(1, interval_z+1))

        self.rng = np.random.default_rng()

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def __str__(self):
        return 'SiteUpdate'

    def step(self, cfg):
        r'''
        Make a volume's worth of locally-exact updates to n.

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

        metropolis = self.rng.uniform(0, 1, phi.shape)
        total_accepted = 0
        total_acceptance = 0

        # The idea is to make coordinated changes to n that keep dn=0.  We can do that by letting the change in n
        # be an exact form dz with z a zero form so that the change in dn is d^2z = 0.
        # However, rather than a python-level for loop over space, we can accomplish a lot more at the numpy level,
        # as in the villain.LinkUpdate.

        # Since the coordinated updates of n are derived from z we can think like we think in the SiteUpdate:
        # What enters the change in action (per link) is dz, which knows about z on two sites.
        #
        # That poses a small problem because if we change the action by changing dz, we want to be able to track
        # that change in dz back to a change in z on ONE particular site, and to accept or reject that change independently
        # from other changes in z. Therefore, we use checkerboarding.
        for color in L.checkerboarding:


            # We only offer changes to z on a single color at once.  The benefit is that the surrounding sites
            # do not have updates.  So we know where any change in dz and therefore any change in the action on any link came from:
            # it came from the site in the partition (color) we are updating.
            z = L.form(0, dtype=int)
            z[color] = self.rng.choice(self.zs, len(color[0]))

            # To keep dn=0 we let the change in n be given by d(z), so that d(change_n) = 0, since it is d^2(z).
            change_n = L.d(0, z)
            dS_link = 0.5 * self.Action.kappa * (-2*np.pi*change_n) * (2*(dphi - 2*np.pi*n) - 2*np.pi*change_n)

            # The change in action originating from the zero form on the color under consideration
            # is just the sum of all the changes from the adjacent links.  So we sum them up.
            dS = dS_link[0] + dS_link[1] + L.roll(dS_link[0], (+1, 0)) + L.roll(dS_link[1], (0, +1))

            # Now dS is a 0-form encoding the changes in action from n = d(the zero form z).  But we should be careful:
            # dS is not 0 on the off-color sites---those sites still have links that land us on the current color.
            # We only want to accept/reject updates on the current color, so we restrict our attention when computing the acceptance.
            acceptance = np.clip( np.exp(-dS[color]), a_min=0, a_max=1)
            accepted = (metropolis[color] < acceptance)

            total_accepted += accepted.sum()
            total_acceptance += acceptance.sum()

            # Finally, we update the n where the change is accepted.
            z[color] *= accepted
            n += L.d(0, z)

        self.proposed += L.sites
        self.acceptance += total_acceptance / L.sites
        self.accepted += total_accepted

        logger.info(f'Average proposal acceptance {total_acceptance / L.sites:.6f}; Actually accepted {total_accepted} / {L.sites} = {total_accepted / L.sites}')

        return {'phi': phi, 'n': n}



    def report(self):
        return (
            f'There were {self.accepted} exact proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )

