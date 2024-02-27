#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.h5 import ReadWriteable

import logging
logger = logging.getLogger(__name__)

class CoexactUpdate(ReadWriteable):
    r'''
    One way to guarantee that $\delta m = 0$ is to change $m$ by $\delta t$ where $t$ is an integer-valued two-form.

    Proposals are drawn according to

    .. math ::

        \begin{align}
            t_p   &\sim [-\texttt{interval_t}, +\texttt{interval_t}] \setminus \{0\}
            &
            \Delta m_\ell &= (\delta t)_\ell
        \end{align}

    .. warning ::
        This algorithm is not ergodic on its own.  It does not change $v$ (see the :class:`~.worldline.VortexUpdate`)
        nor can it produce all coclosed changes---only coexact changes.
        For a coïnexact coclosed update consider the :class:`~.worldline.WrappingUpdate` or the :class:`Geometric worm <supervillain.generator.worldline.worm.Geometric>`.
    '''

    def __init__(self, action, interval_t = 1):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('Need a Worldline action')

        self.Action       = action
        self.Lattice      = action.Lattice
        self.kappa        = action.kappa

        self.interval_t = interval_t
        self.ts = tuple(t for t in range(-interval_t, 0)) + tuple(t for t in range(1, interval_t+1))

        self.rng = np.random.default_rng()

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def __str__(self):
        return 'SiteUpdate'

    def step(self, cfg):
        r'''
        Make a volume's worth of locally-exact updates to m.

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

        v = cfg['v'].copy()
        delta_v_by_W = self.Lattice.delta(2, v)/self.Action.W

        m = cfg['m'].copy()

        L = self.Lattice

        metropolis = self.rng.uniform(0, 1, v.shape)
        total_accepted = 0
        total_acceptance = 0

        # The idea is to make coordinated changes to m that keep δm=0.  We can do that by letting the change in m
        # be a coexact form δt with t a two-form so that the change in δm is δ^2t = 0.
        # However, rather than a python-level for loop over space, we can accomplish a lot more at the numpy level,
        # as in the villain.ExactUpdate.

        # Since the coordinated updates of m are derived from t we can think like we think in the ExactUpdate:
        # What enters the change in action (per link) is δt, which knows about t on two plaquettes.
        #
        # That poses a small problem because if we change the action by changing m, we want to be able to track
        # that change back to a change in t on ONE particular plaquette, and to accept or reject that change independently
        # from other changes in t. Therefore, we use checkerboarding.
        for color in L.checkerboarding:


            # We only offer changes to t on a single color at once.  The benefit is that the surrounding plaquettes
            # do not have updates.  So we know where any change in m=δt and therefore any change in the action on any link came from:
            # it came from the plaquette in the partition (color) we are updating.
            t = L.form(2, dtype=int)
            t[color] = self.rng.choice(self.ts, len(color[0]))

            # To keep δm=0 we let the change in m be given by δt, so that δ(change_m) = δ^2(t) = 0.
            change_m = L.delta(2, t)
            dS_link = 0.5 / self.Action.kappa * change_m * (2*(m - delta_v_by_W) + change_m)

            # The change in action originating from the two form on the color under consideration
            # is just the sum of all the changes from the adjacent links.  So we sum them up.
            dS = dS_link[0] + dS_link[1] + L.roll(dS_link[0], (0, -1)) + L.roll(dS_link[1], (-1, 0))

            # Now dS is a 2-form encoding the changes in action from n = d(the zero form z).  But we should be careful:
            # dS is not 0 on the off-color sites---those sites still have links that land us on the current color.
            # We only want to accept/reject updates on the current color, so we restrict our attention when computing the acceptance.
            acceptance = np.clip( np.exp(-dS[color]), a_min=0, a_max=1)
            accepted = (metropolis[color] < acceptance)

            total_accepted += accepted.sum()
            total_acceptance += acceptance.sum()

            # Finally, we update the n where the change is accepted.
            t[color] *= accepted
            m += L.delta(2, t)

        self.proposed += L.sites
        self.acceptance += total_acceptance / L.plaquettes
        self.accepted += total_accepted

        logger.info(f'Average proposal acceptance {total_acceptance / L.plaquettes:.6f}; Actually accepted {total_accepted} / {L.plaquettes} = {total_accepted / L.plaquettes}')

        return {'m': m, 'v': v}



    def report(self):
        return (
            f'There were {self.accepted} coexact proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )

