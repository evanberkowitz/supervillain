#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.h5 import H5able

class PlaquetteUpdate(H5able):
    r'''
    Ref. :cite:`Gattringer:2018dlw` suggests a simple update scheme where the links surrounding a single plaquette are updated in concert so that the :class:`~.Worldline` constraint is maintained.

    The links on a plaquette are updated in the same (oriented) way, which guarantees that $\delta m$ is invariant before and after the update, so that if it started 0 everywhere so it remains.
    The coordinated change is randomly chosen from ±1.

    .. warning::
        HOWEVER this algorithm is not ergodic on its own.
        The issue is that no proposal can change the worldline :class:`~.TorusWrapping`.
        Instead, if you start cold with $m=0$, which has global wrapping of (0,0) you stay in the (0,0) sector.

    '''
    
    def __init__(self, action):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The PlaquetteUpdate requires the Worldline action.')
        self.Action = action
        self.accepted = 0
        self.proposed = 0
        self.rng = np.random.default_rng()
        self.acceptance = 0.

    def __str__(self):
        return f'PlaquetteUpdate'

    def step(self, cfg):
        r'''
        Performs a sweep of the plaquettes in a randomized order.
        '''
        
        kappa = self.Action.kappa
        W     = self.Action.W
        L = self.Action.Lattice

        m = cfg['m'].copy()
        v = cfg['v'].copy()
        
        for here, change_m, change_v, metropolis in zip(
                np.random.permutation(L.coordinates),
                self.rng.choice([-1, +1], L.sites),
                self.rng.choice([-1, 0, +1], L.sites),
                self.rng.uniform(0,1,L.sites)
                ):
            
            north, west, south, east = L.mod(here + np.array([[+1,0], [0,+1], [-1,0], [0,-1]]))
            
            dS = (change_m - change_v/W) / kappa * (
                + (m[0][here [0], here [1]] - (v[here [0], here [1]] - v[east [0], east [1]])/W)
                - (m[1][here [0], here [1]] - (v[south[0], south[1]] - v[here [0], here [1]])/W)
                + (m[1][north[0], north[1]] - (v[here [0], here [1]] - v[north[0], north[1]])/W)
                - (m[0][west [0], west [1]] - (v[west [0], west [1]] - v[here [0], here [1]])/W)
                + 2 * (change_m - change_v/W)
            )
            acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)

            
            self.acceptance += acceptance
            if metropolis < acceptance:
                # Accept :)
                m[0][here [0], here [1]] += change_m
                m[1][here [0], here [1]] -= change_m
                m[1][north[0], north[1]] += change_m
                m[0][west [0], west [1]] -= change_m
                v[here[0], here[1]]      += change_v
                self.accepted+=1

            else:
                # Reject :(
                pass
        
        self.proposed += L.sites
        return {'m': m, 'v': v}

    def report(self):
        return (
                f'There were {self.accepted} single-plaquette proposals accepted of {self.proposed} proposed updates.'
                +'\n'+
                f'    {self.accepted/self.proposed:.6f} acceptance rate'
                +'\n'+
                f'    {self.acceptance / self.proposed :.6f} average Metropolis acceptance probability.'
            )

class WrappingUpdate(H5able):
    r'''
    Because :class:`~.PlaquetteUpdate` fails to change the wrapping, we should separately offer wrapping-changing proposals.

    We propose coordinated changes on all the x-direction links on a single timeslice and coordinated changes on all the t-direction links on a single spatial slice.

    The coordinated change of $m$ is randomly chosen from ±1 (the same on each link).
    
    The 2-form constraint field $v$ contributes to the action as $\delta v$ and has no nontrivial winding around the torus, so it is not changed by this update.

    .. warning::
        HOWEVER this algorithm is not ergodic on its own.
        The issue is that no proposal can generate wrapping-preserving changes.

    '''

    def __init__(self, action):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The WrappingUpdate requires the Worldline action.')
        self.Action = action

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.rng = np.random.default_rng()

    def __str__(self):
        return f'WrappingUpdate'

    def step(self, cfg):
        '''
        Propose independent updates of $m$ on every timeslice and on every spatial slice.

        In principle all the proposals may be made in parallel but we just do them sequentially.
        '''
        kappa = self.Action.kappa
        W     = self.Action.W
        L = self.Action.Lattice

        m = cfg['m'].copy()
        v = cfg['v'].copy()

        # One might worry that we really need to recompute some elements of this inside the loop,
        # since m gets updated in the loops. However, the changes do not influence one another;
        # we could parallelize the update on each torus wrapping.
        #
        # Therefore we can get a speedup by vectorizing the needed differences.
        #
        # TODO: in fact, it may be possible to completely vectorize this update.
        link = m - L.δ(2, v) / W

        # First try updating all the x-direction links on a timeslice t.
        for t, change_m, metropolis, in zip(L.t, np.random.choice([-1,+1], L.nt), self.rng.uniform(0,1,L.nt)):
            
            # Directly evaluate ∆S = S_proposal - S_current, which is the difference of squares on every link.
            # That difference simplifies dramatically.
            dS = change_m / kappa * ( link[1][t,:].sum() + L.nt * change_m / 2)

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
            dS = change_m / kappa * ( link[0][:,x].sum() + L.nx * change_m / 2)

            acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)
            self.acceptance += acceptance

            if metropolis < acceptance:
                # Accept :)
                m[0][:,x] += change_m
                self.accepted+=1


        self.proposed += L.nt + L.nx
        return {'m': m, 'v': v}

    def report(self):
        return (
                f'There were {self.accepted} single-wrapping proposals accepted of {self.proposed} proposed updates.'
                +'\n'+
                f'    {self.accepted   / self.proposed :.6f} acceptance rate' 
                +'\n'+
                f'    {self.acceptance / self.proposed :.6f} average Metropolis acceptance probability.'
            )
