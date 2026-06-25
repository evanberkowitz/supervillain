#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import delta

import logging
logger = logging.getLogger(__name__)

class VortexUpdate(ReadWriteable, Generator):
    r'''
    This performs the same update to $v$ as :class:`PlaquetteUpdate <supervillain.generator.worldline.PlaquetteUpdate>` but leaves $m$ untouched.

    Proposals are drawn according to

    .. math ::

        \begin{aligned}
            \Delta v_p   &\sim [-\texttt{interval\_v}, +\texttt{interval\_v}] \setminus \{0\} &&(W<\infty)
            \\
            \Delta v_p   &\sim \text{uniform}(-\texttt{interval\_v}, +\texttt{interval\_v}) &&(W=\infty)
        \end{aligned}

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
        total_accepted = 0
        total_acceptance = 0

        m = cfg['m'].copy()
        v = cfg['v'].copy()

        L = self.Action.Lattice
        W = self.Action._W

        n_comps = len(L.components[2])

        # One independent Metropolis draw per (2-form component, site).
        metropolis = self.rng.uniform(0, 1, (n_comps,) + L.dims)

        # Each v only talks to the m on the immediately surrounding links (through δv).  So if we freeze m
        # and only change v one checkerboarding color at a time then the change in action on each link
        # comes from the v of that color.
        #
        # In D>2 there are C(D,2) independent 2-form components. We process each independently per color,
        # using coface_sum() to aggregate per-plaquette ΔS from their boundary links.
        for color in L.checkerboarding:
            for comp_idx in range(n_comps):

                # We need to compute delta_v each time because v is updated on each pass.
                delta_v = delta(v)

                # Randomly bump v at this component and color.
                if self.Action.W < float('inf'):
                    change_v = L.form(2, dtype=int)
                    change_v[comp_idx][color] = self.rng.choice(self.vs, len(color[0]))
                else:
                    change_v = L.form(2, dtype=float)
                    change_v[comp_idx][color] = self.rng.uniform(-self.interval_v, +self.interval_v, len(color[0]))

                # Compute the change of action on each link.
                change_delta_v = delta(change_v)
                dS_link = 0.5 / self.Action.kappa * (-change_delta_v / W) * (2*(m - delta_v / W) - change_delta_v / W)

                # The change in action from this plaquette is the sum of changes on its boundary links.
                # coface_sum() accumulates those, giving dS[comp_idx][x] for the plaquette at x.
                dS = dS_link.coface_sum()

                # dS is not 0 on off-color plaquettes. Only accept/reject on the current color.
                acceptance = np.clip(np.exp(-dS[comp_idx][color]), a_min=0, a_max=1)
                accepted = (metropolis[comp_idx][color] < acceptance)

                total_accepted += accepted.sum()
                total_acceptance += acceptance.sum()

                # Update v where the change is accepted.
                v[comp_idx][color] += change_v[comp_idx][color] * accepted

        self.proposed += L.cells_of_degree[2]
        self.acceptance += total_acceptance / L.cells_of_degree[2]
        self.accepted += total_accepted

        logger.debug(f'Average proposal acceptance {total_acceptance / L.cells_of_degree[2]:.6f}; Actually accepted {total_accepted} / {L.cells_of_degree[2]} = {total_accepted / L.cells_of_degree[2]}')

        return cfg | {'v': v}



    def report(self):
        return (
            f'There were {self.accepted} vortex proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )
