#!/usr/bin/env python

import numpy as np

import logging
logger = logging.getLogger(__name__)

class NeighborhoodUpdate:

    def __init__(self, action, interval_phi=2*np.pi, interval_n=1):
        self.Action       = action
        self.interval_phi = interval_phi
        self.interval_n   = interval_n

        self.accepted = 0
        self.proposed = 0

    def proposal(self, cfg, dx):

        L = self.Action.Lattice

        phi = L.roll(cfg['phi'], dx)
        n   = L.roll(cfg['n'],   dx)

        phi[0,0] += np.random.default_rng().uniform(-self.interval_phi,+self.interval_phi,None)

        n_shift = np.random.randint(-self.interval_n,1+self.interval_n,4)
        n[0][+0,+0] += n_shift[0]
        n[0][-1,+0] += n_shift[1]
        n[1][+0,+0] += n_shift[2]
        n[1][+0,-1] += n_shift[3]

        return {'phi': phi, 'n': n}

    def site(self, cfg, dx):

        S_start    = self.Action(cfg['phi'], cfg['n'])
        proposal   = self.proposal(cfg, dx)
        S_proposal = self.Action(proposal['phi'], proposal['n'])

        dS = S_proposal - S_start

        acceptance = np.clip( np.exp(-dS), a_min=0, a_max=1)
        metropolis = np.random.default_rng().uniform(0,1,None)
        if metropolis < acceptance:
            logger.debug(f'Proposal accepted; ∆S = {dS:f}; acceptance probability = {acceptance:f}')
            return proposal, acceptance, 1
        else:
            logger.debug(f'Proposal rejected; ∆S = {dS:f}; acceptance probability = {acceptance:f}')
            return cfg, acceptance, 0

    def step(self, cfg):

        current = cfg
        acceptance = 0
        accepted = 0

        L = self.Action.Lattice

        shifts = np.stack((
            np.random.randint(L.dims[0], size=L.sites),
            np.random.randint(L.dims[1], size=L.sites)
        )).transpose()

        for dx in shifts:
            subsequent, acceptance, accepted = self.site(current, dx)
            current = subsequent
            acceptance += acceptance
            accepted   += accepted

        self.accepted += accepted
        self.proposed += len(shifts)

        acceptance /= self.Action.Lattice.sites
        logger.info(f'Average proposal {acceptance=:.6f}; Actually {accepted = } / {self.Action.Lattice.sites} = {accepted / self.Action.Lattice.sites}')

        return current
