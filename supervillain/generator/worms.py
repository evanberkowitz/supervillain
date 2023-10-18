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

        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The (undirected) worm algorithm update requires the Worldline action.')
        self.Action     = action
        self.interval_m = interval_m

        self.rng = np.random.default_rng()

        self.avg_length = 0.
        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0


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
        coordt,coordx = self.rng.choice(L.coordinates)





    # def proposal(self, cfg, dx):
    #     r'''
    #     Parameters
    #     ----------
    #     cfg: dict
    #         A dictionary with $\phi$ and $n$ to update.
    #     dx: Lattice coordinates
    #         Which site to move to the origin and update.

    #     Returns
    #     -------
    #     dict:
    #         A new configuration with updated $\phi$ and $n$.
    #     '''
    #     L = self.Action.Lattice

    #     # We move the lattice around (which is fine by translational symmetry)
    #     # so that we update a different site with each proposal.
    #     # The advantage of thinking this way is that we only have to reckon from the origin.
    #     phi = L.roll(cfg['phi'].copy(), dx)
    #     n   = L.roll(cfg['n'].copy(),   dx)

    #     phi[0,0] += self.rng.uniform(-self.interval_phi,+self.interval_phi,None)

    #     n_shift = self.rng.choice(self.n_changes,4)
    #     n[0][+0,+0] += n_shift[0]
    #     n[0][-1,+0] += n_shift[1]
    #     n[1][+0,+0] += n_shift[2]
    #     n[1][+0,-1] += n_shift[3]

    #     return {'phi': phi, 'n': n}

    # def site(self, cfg, dx):
    #     r'''
    #     Rather than accepting every :func:`~proposal` we perform *importance sampling* by doing a Metropolis accept/reject step :cite:`Metropolis` on every single-site proposal.

    #     Parameters
    #     ----------
    #     cfg: dict
    #         A dictionary with $\phi$ and $n$ to update.
    #     dx: Lattice coordinates
    #         Which site to move to the origin and update.

    #     Returns
    #     -------
    #     dict:
    #         A configuration; either the provided one a new one changed by a proposal.
    #     float:
    #         The Metropolis-Hastings acceptance probability.
    #     int:
    #         1 if the proposal was accepted, 0 otherwise.
    #     '''

    #     S_start    = self.Action(cfg['phi'], cfg['n'])
    #     proposal   = self.proposal(cfg, dx)
    #     S_proposal = self.Action(proposal['phi'], proposal['n'])

    #     dS = S_proposal - S_start

    #     acceptance = np.clip( np.exp(-dS), a_min=0, a_max=1)
    #     metropolis = np.random.default_rng().uniform(0,1,None)
    #     if metropolis < acceptance:
    #         logger.debug(f'Proposal accepted; ∆S = {dS:f}; acceptance probability = {acceptance:f}')
    #         return proposal, acceptance, 1
    #     else:
    #         logger.debug(f'Proposal rejected; ∆S = {dS:f}; acceptance probability = {acceptance:f}')
    #         return cfg, acceptance, 0

    # def step(self, cfg):
    #     r'''
    #     Make volume's worth of random single-site updates.

    #     Parameters
    #     ----------
    #     cfg: dict
    #         A dictionary with phi and n field variables.

    #     Returns
    #     -------
    #     dict
    #         Another configuration of fields.
    #     '''

    #     self.sweeps += 1
    #     current = cfg
    #     acceptance = 0
    #     accepted = 0

    #     L = self.Action.Lattice

    #     shifts = np.stack((
    #         np.random.randint(L.dims[0], size=L.sites),
    #         np.random.randint(L.dims[1], size=L.sites)
    #     )).transpose()

    #     for dx in shifts:
    #         subsequent, probability, acc = self.site(current, dx)
    #         current = subsequent
    #         acceptance += probability
    #         accepted   += acc

    #     self.accepted += accepted
    #     self.proposed += len(shifts)

    #     acceptance /= len(shifts)
    #     self.acceptance += acceptance
    #     logger.info(f'Average proposal {acceptance=:.6f}; Actually {accepted = } / {self.Action.Lattice.sites} = {accepted / self.Action.Lattice.sites}')

    #     return current

    # def report(self):
    #     r'''
    #     Returns a string with some summarizing statistics.
    #     '''
    #     return (
    #         f'There were {self.accepted} single-site proposals accepted of {self.proposed} proposed updates.'
    #         +'\n'+
    #         f'    {self.accepted/self.proposed:.6f} acceptance rate' 
    #         +'\n'+
    #         f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
    #     )

