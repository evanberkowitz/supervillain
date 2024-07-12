#!/usr/bin/env python

from collections import deque
import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable

import logging
logger = logging.getLogger(__name__)

class Geometric(ReadWriteable, Generator):
    r'''

    We adapt the undirected worm algorithm of Alet and Sørensen :cite:`PhysRevE.68.026702`.

    We start with a constraint-satisfying $z$ configuration with no worm on it.

    We randomly pick a plaquette to place the head and tail, not changing any of the physical fields.

    From the current plaquette, there are four or five allowable moves:

    The head can cross any of the four boundary links, changing that link push the defect over one plaquette.
    These moves change $n$ by ±1 and therefore changes the action.

    If the head and tail are in the same place the constraint is satisfied and both the head and tail can be taken away,
    moving from $g$ to $z$ without changing the action.  But if the head and tail are in different places and we try to
    stop evolving the worm, the configuration will have two defects, violating the constraint, and costing infinite action.
    In other words, the Saint Patrick move only works against ouroborous configurations.

    Let $\Delta S$ represent these five changes in action---four for single-plaquette motions and one for the Saint Patrick worm-elimination.

    We Metropolis-normalize these to transition rates

    .. math ::

        A_i = \min(1, \exp(-\Delta S_i))

    for each move $i$ and select the next configuration according to the probabilities

    .. math ::
        
        P_i = A_i / \sum A

    When the head and tail are not coincident the probability of transitioning to a $z$ configuration is $P_z=0$,
    and the $A$s are summed exactly as in Alet and Sørensen; when the head and tail are coincident the Saint Patrick move has $A_z = 1$ and is not unlikely to be accepted.

    This algorithm differs slightly from Alet and Sørensen, who, once the worm closes, Metropolis test accepting the worm.
    In their case, if the worm is rejected, the previous configuration must be repeated in the ensemble.
    By framing the acceptance of the worm as just another transition (from a diagonal $g$ configuration to a $z$ configuration),
    we are never faced with the prospect of throwing away a whole update.  If it doesn't transition to a $z$ configuration
    the worm keeps on doing its wormy digging, and it will ultimately return for another bite at the apple.
    Eventually the transition from $g$ to $z$ will be accepted and we will receive that configuration in our Markov chain for $Z$.


    .. warning ::
        
        This update algorithm is not ergodic on its own.  It doesn't change $\phi$ at all and even leaves $dn$ alone (while changing $n$ itself).
        It can be used, for example, :class:`~.Sequentially` with the :class:`~.NeighborhoodUpdate` for an ergodic method.

    .. note ::
        
        **Because** it doesn't change $dn$ at all, this algorithm can play an important role in sampling the $W=\infty$ sector, where all vortices are completely killed, though updates to $\phi$ would still be needed.
    '''

    def __init__(self, S):
        self.Action = S
        self.rng = np.random.default_rng()

        self.worm_lengths = deque()

    def _neighboring_plaquettes(self, here):
        # east, north, west, south
        return self.Action.Lattice.mod(here + np.array([[0,-1], [+1,0], [0,+1], [-1,0]]))

    def _surrounding_links(self, here):
        # These are the directions we'd like to move the head of the defect.
        _, north, west, _ = self._neighboring_plaquettes(here)

        return ((0, here [0], here [1]), # t link to the east
                (1, north[0], north[1]), # x link to the north
                (0, west [0], west [1]), # t link to the west
                (1, here [0], here [1])) # x link to the south

    def inline_observables(self, steps):
        r'''
        The worm algorithm can measure the ``Vortex_Vortex`` correlator.
        We also store the ``Worm_Length`` for each step.
        '''

        return {
            'Vortex_Vortex': self.Action.Lattice.form(0, steps),
            'Worm_Length':   np.zeros(steps),
        }

    def step(self, configuration):
        r'''
        Given a constraint-satisfying configuration, returns another constraint-satisfying configuration udpated via worm as described above.
        '''

        S = self.Action
        L = S.Lattice

        displacements = L.form(0)

        # This algorithm will not update phi; but it is useful to precompute dphi
        # which is used in the evaluation of the changes in action.
        phi = configuration['phi'].copy()
        dphi = L.d(0, phi)

        # The contributions to the plaquette tell you how an n contributes to dn.
        # Opposite directions contribute oppositely, which is exactly what you want.
        # That way, if the worm moves north, you increase n by 1, but if the worm then
        # immediately moves south it would cross the same link but decrease n by 1,
        # so that the constraint on this cul-de-sac would be restored.
        plaquette = np.array([+1, +1, -1, -1]) # east, north, west, south
        # The documentation gives a definitive statement about moving the head only.
        # But we could equally well move the tail, making the opposite moves in the opposite worm evolution.
        # This can be accomplished simply by multiplying the offered changes to the links by -1.
        # We can randomly decide this orientation of the worm
        orientation = self.rng.choice([-1, +1])
        # and then simply multiply it into the constraint-restoring proposals.
        change_n = orientation * plaquette

        # We start with a constraint-satisfying configuration of n that is in the z sector.
        n = configuration['n'].copy()
        # and insert both the head and tail onto any random plaquette---because the head and the tail are
        # coincident, they don't change the action and so any choice should be equally weighted.
        tail = self.rng.choice(L.coordinates)
        head = tail.copy()
        worm_length = 0
        # by placing the head and tail down we have moved to the g sector!
        # Now we are ready to start evolving in z union g.

        while True:
            x, y = L.mod(head-tail)
            displacements[x, y] += 1

            # There are 4 or 5 possible moves that we may make.
            # We may move the head to 1 of 4 neighboring plaquettes
            next = self._neighboring_plaquettes(head)
            # in which case we will cross the corresponding links.
            link = self._surrounding_links(head)

            # Crossing the link changes n and therefore the action.
            change_link = np.array([dphi[l] - 2*np.pi*n[l] for l in link])
            change_S = (
                (S.kappa / 2) *
                (-2*np.pi*change_n) *
                (2*change_link - 2*np.pi*change_n)
            )

            # There is also the possibility to move from the g sector to the z sector, which we might add to the 4 worm-movement moves.
            change_S = np.concatenate(([
            # That transition costs 0 action if the head and the tail are coincident and the winding constraint is satisfied everywhere.
                0 if (head==tail).all()
            # But, that transition should be impossible if the head and the tail are not coincident, because the winding constraint would be violated.
                else float('inf')
                ], change_S))

            # Now we must compute the Metropolis amplitudes
            #
            #   A = min(1, exp(-ΔS) )
            #
            A = np.clip(np.exp(-change_S), a_min=0, a_max=1)
            # and normalize them to get the probabilities.
            P = A/A.sum()

            # With those probabilities in hand we can choose the update.
            choice = self.rng.choice([-1,0,1,2,3], p=P)

            # We might transition to the z sector, in which case we have produced a configuration that can go into our Markov chain.
            if choice == -1:
                self.worm_lengths.append(worm_length)
                return {'n': n, 'phi': phi, 'Vortex_Vortex': displacements, 'Worm_Length': worm_length}
                # Note!  We don't need to Metrpolis accept/reject.  That was built in when we included the g --> z update
                # in the A amplitudes that went into the update probabilities.
                # That is why we went through this whole story rigamarole of distinguishing z configurations from diagonal g configurations:
                # to unify the treatment of the acceptance of the closed worm with the movement of the worm's head.

            # Otherwise we need to cross the link,
            n[link[choice]] += change_n[choice]
            worm_length += 1
            # move the head
            head = next[choice]
            # and consider our next move.

    def report(self):
        l = np.array(self.worm_lengths)
        return f'There were {len(l)} worms.\nWorms lengths:\n    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}'
