#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.h5 import H5able

import logging
logger = logging.getLogger(__name__)

class UndirectedWorm(H5able):
    r'''
    '''

    def __init__(self, action, interval_m):

        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The (undirected) worm algorithm update requires the Worldline action.')
        self.Action     = action
        self.kappa      = action.kappa
        self.Lattice    = action.Lattice
        self.interval_m = interval_m

        self.rng = np.random.default_rng()

        self.avg_length = 0.
        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def burrow(self, cfg, currentsite):
        # Takes in the current position of the worm, returns the new configuration after the worm has moved and the coordinate to which it has burrowed.
        pT, mT, mX, pX = self.Lattice.mod(currentsite + np.array([[+1,0],[-1,0],[0,-1],[0,+1]]))
        # +T
        proposedconfigpT = cfg.copy()
        currentlink = proposedconfigpT[0,currentsite[0],currentsite[1]]
        proposedlink = currentlink + 1
        deltaEpT = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        AEpT = min(1,np.exp(-deltaEpT/self.Action.kappa))

        proposedconfigmT = cfg.copy()
        currentlink = proposedconfigmT[0,mT[0],mT[1]]
        proposedlink = currentlink + 1
        deltaEmT = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        AEmT = min(1,np.exp(-deltaEmT/self.Action.kappa))

        proposedconfigpX = cfg.copy()
        currentlink = proposedconfigpX[0,currentsite[0],currentsite[1]]
        proposedlink = currentlink + 1
        deltaEpX = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        AEpX = min(1,np.exp(-deltaEpX/self.Action.kappa))

        proposedconfigmX = cfg.copy()
        currentlink = proposedconfigmX[0,mX[0],mX[1]]
        proposedlink = currentlink + 1
        deltaEmX = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        AEmX = min(1,np.exp(-deltaEmX/self.Action.kappa))

        N = np.sum(np.array([AEpT,AEmT,AEpX,AEmX]))
        PpT = AEpT/N
        PmT = AEmT/N
        PpX = AEpX/N
        PmX = AEmX/N

        chosenDirection = self.rng.choice([1,2,3,4],p=[PpT,PmT,PpX,PmX])
        newconfig = cfg.copy()

        if chosenDirection==1:
            linksite = currentsite.copy()
            newconfig[0,linksite[0],linksite[1]] += 1
            newsite = pT
        if chosenDirection==2:
            linksite = mT.copy()
            newconfig[0,linksite[0],linksite[1]] -= 1
            newsite = mT
        if chosenDirection==3:
            linksite = currentsite.copy()
            newconfig[1,linksite[0],linksite[1]] += 1
            newsite = pX
        if chosenDirection==4:
            linksite = mX.copy()
            newconfig[1,linksite[0],linksite[1]] -= 1
            newsite = mX

        return [newconfig,newsite]

    def step(self, cfg):
        # Step encapsulates one iteration of a worm to completion, either with acceptance or erasure
        # Structure: 
        #   -Starting site chosen
        #       ? Eventually may want to store starting sites
        #       *No subroutine just initialized at start of each step
        #   -Choose direction to move, as per metropolis weighting
        #       -Involves calculating weights for each of 4 links
        #       *This will repeat until the worm has returned to start. Looping call to subroutine 'burrow'. 
        #       *Weightings can be calculated within burrow to start, may be relegated to inner subroutine later.
        #       -End of burrow subroutine returns starting config with one link changed
        #           -May include records of directional biases later, so will store output in dict
        #       -Final check for worm erasure, 
        #       -After burrow concludes loop, add to avg_length
        #       !!!!! Ask about whether erased worms should contribute to the length records. Will not to begin with
        #

        L = self.Action.Lattice
        startingcoordt,startingcoordx = self.rng.choice(L.coordinates)

