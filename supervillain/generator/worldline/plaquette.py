#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable

class PlaquetteUpdate(ReadWriteable, Generator):
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
        W     = self.Action._W
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

