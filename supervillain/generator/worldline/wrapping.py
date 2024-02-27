#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.h5 import ReadWriteable

class WrappingUpdate(ReadWriteable):
    r'''
    Because :class:`~.CoexactUpdate` fails to change the wrapping, we should separately offer wrapping-changing proposals.

    We propose coordinated changes on all the x-direction links on a single timeslice and coordinated changes on all the t-direction links on a single spatial slice.

    That is, on all the $m$ on a cycle around the torus we propose to change $m$ according to

    .. math ::
        \Delta m \sim [-\texttt{interval_w}, +\texttt{interval_w}] \setminus \{0\}

    .. warning::
        This algorithm is not ergodic on its own.
        The issue is that no proposal can generate coexact changes.

    '''

    def __init__(self, action, interval_w):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The WrappingUpdate requires the Worldline action.')
        self.Action = action

        self.interval_w = interval_w
        self.w = tuple(h for h in range(-interval_w, 0)) + tuple(h for h in range(1, interval_w+1))

        self.accepted = 0
        self.proposed = 0
        self.sweeps = 0
        self.acceptance = 0.
        self.rng = np.random.default_rng()

    def __str__(self):
        return f'WrappingUpdate'

    def step(self, cfg):
        '''
        Propose independent updates of $m$ on every timeslice and on every spatial slice.

        In principle all the proposals may be made in parallel but we just do them sequentially.
        '''

        L = self.Action.Lattice

        m = cfg['m'].copy()
        v = cfg['v'].copy()

        # The main idea is that each straight-shot cycle around the torus changes the action only on the links it traverses.
        # The cycles are independent.  So we can accept or reject them independently---changes on one cycle don't talk to changes on another.
        # We can fill a whole 1-form with changes to m that satisfy δm=0 along every cycle.
        change_m = L.form(1, dtype=int)
        change_m[0] = self.rng.choice(self.w, L.nx)[None,:]
        change_m[1] = self.rng.choice(self.w, L.nt)[:,None]
        #assert self.Action.valid(change_m)

        # Now we compute the change in action on every link, which we will reduce along different directions
        # to get the change in action from each cycle.
        dS_link = 0.5 / self.Action.kappa * change_m * (2*(m - L.δ(2, v) / self.Action.W) + change_m)

        # First the temporal cycles.
        dS = dS_link[0].sum(axis=0)
        # Now we Metropolize
        acceptance = np.clip( np.exp(-dS), a_min=0, a_max=1)
        metropolis = self.rng.uniform(0, 1, acceptance.shape)
        accepted = metropolis < acceptance
        # and zero out the rejected changes.
        change_m[0] *= accepted[None,:]

        total_acceptance = acceptance.sum()
        total_accepted   = accepted.sum()

        # Now the spatial links
        dS = dS_link[1].sum(axis=1)
        # Now we Metropolize
        acceptance = np.clip( np.exp(-dS), a_min=0, a_max=1)
        metropolis = self.rng.uniform(0, 1, acceptance.shape)
        accepted = metropolis < acceptance
        # and zero out the rejected changes.
        change_m[1] *= accepted[:,None]

        total_acceptance += acceptance.sum()
        total_accepted   += accepted.sum()

        self.proposed += L.nx + L.nt
        self.acceptance += total_acceptance / (L.nx + L.nt)
        self.accepted += total_accepted
        self.sweeps += 1

        #assert self.Action.valid(change_m)
        #assert self.Action.valid(m+change_m)

        return {'m': m + change_m, 'v': v}

    def report(self):
        return (
                f'There were {self.accepted} single-wrapping proposals accepted of {self.proposed} proposed updates.'
                +'\n'+
                f'    {self.accepted   / self.proposed :.6f} acceptance rate' 
                +'\n'+
                f'    {self.acceptance / self.sweeps :.6f} average Metropolis acceptance probability.'
            )
