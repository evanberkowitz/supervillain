#!/usr/bin/env python

import numpy as np
from supervillain.h5 import H5able

class PlaquetteUpdate(H5able):
    r'''
    Ref. :cite:`Gattringer:2018dlw` suggests a simple update scheme where the links surrounding a single plaquette are updated in concert so that the :class:`~.Worldline` constraint is maintained.

    The links on a plaquette are updated in the same (oriented) way, which guarantees that $\delta m$ is invariant before and after the update, so that if it started 0 everywhere so it remains.
    The coordinated change is randomly chosen from ±1.

    .. warning::
        HOWEVER this algorithm is not ergodic on its own.
        The issue is that no proposal can change the holonomies.
        Instead, if you start cold with $m=0$, which has global winding of (0,0) you stay in the (0,0) sector.

    '''
    
    def __init__(self, Action):
        self.Action = Action
        self.accepted = 0
        self.proposed = 0
        self.rng = np.random.default_rng()
        self.acceptance = 0.

    def step(self, cfg):
        r'''
        Performs a sweep of the plaquettes in a randomized order.
        '''
        
        kappa = self.Action.kappa
        L = self.Action.Lattice

        m = cfg['m'].copy()
        S_start = self.Action(m)
        
        for here, change_m, metropolis in zip(np.random.permutation(L.coordinates), self.rng.choice([-1, +1], L.sites), self.rng.uniform(0,1,L.sites)):
            
            north = L.mod(here + np.array([1,0]))
            west  = L.mod(here + np.array([0,1]))
            
            dS = change_m / kappa * (
                + m[0][here [0], here [1]]
                - m[1][here [0], here [1]]
                + m[1][north[0], north[1]]
                - m[0][west [0], west [1]]
                + 2 * change_m
            )
            acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)

            
            self.acceptance += acceptance
            if metropolis < acceptance:
                # Accept :)
                m[0][here [0], here [1]] += change_m
                m[1][here [0], here [1]] -= change_m
                m[1][north[0], north[1]] += change_m
                m[0][west [0], west [1]] -= change_m
                self.accepted+=1

            else:
                # Reject :(
                pass
        
        self.proposed += L.sites
        return {'m': m}


class HolonomyUpdate(H5able):
    r'''
    Because :class:`~.PlaquetteUpdate` fails to change the holonomies, we should separately offer holonomy-changing proposals.

    We propose coordinated changes on all the x-direction links on a single timeslice and coordinated changes on all the t-direction links on a single spatial slice.

    The coordinated change is randomly chosen from ±1 (the same on each link).

    .. warning::
        HOWEVER this algorithm is not ergodic on its own.
        The issue is that no proposal can generate holonomy-preserving changes.

    '''

    def __init__(self, action):
        self.Action = action

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.rng = np.random.default_rng()

    def step(self, cfg):
        '''
        Propose independent updates on every timeslice and on every spatial slice.

        In principle all the proposals may be made in parallel but we just do them sequentially.
        '''
        kappa = self.Action.kappa
        L = self.Action.Lattice

        m = cfg['m'].copy()

        # First try updating all the x-direction links on a timeslice t.
        for t, change_m, metropolis, in zip(L.t, np.random.choice([-1,+1], L.nt), self.rng.uniform(0,1,L.nt)):
            
            # Directly evaluate ∆S = S_proposal - S_current, which is the difference of squares on every link.
            # That difference simplifies dramatically.
            dS = change_m / kappa * ( m[1][t,:].sum() + L.nt * change_m / 2)

            acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)
            self.acceptance += acceptance

            if metropolis < acceptance:
                # Accept :)
                m[1][t,:] += change_m
                self.accepted+=1

        # Then try updating all the t-direction links on a spatial slice x.
        for x, change_m, metropolis, in zip(L.x, np.random.choice([-1,+1], L.nx), self.rng.uniform(0,1,L.nx)):

            # Directly evaluate ∆S = S_proposal - S_current, which is the difference of squares on every link.
            # That difference simplifies dramatically.
            dS = change_m / kappa * ( m[0][:,x].sum() + L.nx * change_m / 2)

            acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)
            self.acceptance += acceptance

            if metropolis < acceptance:
                # Accept :)
                m[0][:,x] += change_m
                self.accepted+=1


        self.proposed += L.nt + L.nx
        return {'m': m}
