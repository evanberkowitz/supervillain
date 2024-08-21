#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable

import logging
logger = logging.getLogger(__name__)

class VortexUpdate(ReadWriteable, Generator):
    r'''
    This performs the same update to $v$ as :class:`PlaquetteUpdate <supervillain.generator.worldline.PlaquetteUpdate>` but leaves $m$ untouched.

    Proposals are drawn according to

    .. math ::

        \begin{align}
            \Delta v_p   &\sim [-\texttt{interval_v}, +\texttt{interval_v}] \setminus \{0\} &&(W<\infty)
            \\
            \Delta v_p   &\sim \text{uniform}(-\texttt{interval_v}, +\texttt{interval_v}) &&(W=\infty)
        \end{align}

    on each plaquette $p$ independently.

    .. warning ::
        This update is not ergodic on its own, since it does not change $m$ at all.
    '''

    def __init__(self, action, interval_v = 1):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('Need a Worldline action')

        self.Action       = action

        self.interval_v = interval_v
        self.vs = tuple(v for v in range(-interval_v, 0)) + tuple(v for v in range(1, interval_v+1))

        self.rng = np.random.default_rng()

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def __str__(self):
        return 'VortexUpdate'

    def step(self, cfg):
        r'''
        Make a volume's worth of changes to v.

        Parameters
        ----------
        cfg: dict
            A dictionary with m and v field variables.

        Returns
        -------
        dict
            Another configuration of fields.
        '''

        self.sweeps += 1
        total_acceptance = 0
        accepted = 0

        m = cfg['m'].copy()
        v = cfg['v'].copy()

        L = self.Action.Lattice
        W = self.Action._W

        metropolis = self.rng.uniform(0, 1, v.shape)
        total_accepted = 0
        total_acceptance = 0

        # Each v only talks to the m on the immediately surrounding links (through δv).  So if we freeze m
        # and only change v one checkerboarding color at a time then the change in action on each link
        # comes from the v of that color.
        for color in L.checkerboarding:

            # We need to compute delta_v every time through the loop because v will get updated on each pass.
            delta_v = L.delta(2, v)

            # Randomly bump v
            if self.Action.W < float('inf'):
                change_v = L.form(0, dtype=int)
                change_v[color] = self.rng.choice(self.vs, len(color[0]))
            else:
                change_v = L.form(0, dtype=float)
                change_v[color] = self.rng.uniform(-self.interval_v, +self.interval_v, len(color[0]))

            # and compute the change of action on each link.
            change_delta_v = L.delta(2, change_v)
            dS_link = 0.5 / self.Action.kappa * (-change_delta_v / W) * (2*(m - delta_v / W) - change_delta_v / W)

            # The change in action originating from the plaquette on the color under consideration
            # is just the sum of all the changes from the boundary links.  So we sum them up.
            dS = dS_link[0] + dS_link[1] + L.roll(dS_link[0], (0, -1)) + L.roll(dS_link[1], (-1, 0))

            # Now dS is a 2-form encoding the change in action from the changes in v.  But we should be careful:
            # dS is not 0 on the off-color plaquettes---those plaquettes still have links touching the current color.
            # We only want to accept/reject updates on the current color, so we restrict our attention when computing the acceptance.
            acceptance = np.clip( np.exp(-dS[color]), a_min=0, a_max=1)
            accepted = (metropolis[color] < acceptance)

            total_accepted += accepted.sum()
            total_acceptance += acceptance.sum()

            # Finally, we update the v where the change is accepted.
            v[color] += change_v[color] * accepted

        self.proposed += L.plaquettes
        self.acceptance += total_acceptance / L.plaquettes
        self.accepted += total_accepted

        logger.info(f'Average proposal acceptance {total_acceptance / L.plaquettes:.6f}; Actually accepted {total_accepted} / {L.plaquettes} = {total_accepted / L.plaquettes}')

        return {'m': m, 'v': v}



    def report(self):
        return (
            f'There were {self.accepted} vortex proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )
