#!/usr/bin/env python

from collections import deque
import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
import supervillain.h5.extendable as extendable

import logging
logger = logging.getLogger(__name__)

class Classic(ReadWriteable, Generator):
    r'''
    This implements the classic worm of Prokof'ev and Svistunov :cite:`PhysRevLett.87.160601` for the Villain links $n\in\mathbb{Z}$ which satisfy $dn \equiv 0 $ (mod W) on every plaquette.

    On top of a constraint-satisfying configuration we put down a worm and let the head move, changing the crossed links.
    We uniformly propose a move in all 4 directions and Metropolize the change.

    Additionally, when the head and tail coincide, we allow a fifth possible move, where we remove the worm and emit the updated $z$ configuration into the Markov chain.
    
    As we evolve the worm we tally the histogram that yields the :class:`~.Vortex_Vortex` correlation function.

    .. warning ::
        
        This update algorithm is not ergodic on its own.  It doesn't change $\phi$ at all and even leaves $dn$ alone (while changing $n$ itself).
        It can be used, for example, :class:`~.Sequentially` with the :class:`~.SiteUpdate` and :class:`~.LinkUpdate` for an ergodic method.

    .. note ::
        
        **Because** it doesn't change $dn$ at all, this algorithm can play an important role in sampling the $W=\infty$ sector, where all vortices are completely killed, though updates to $\phi$ would still be needed.
    '''

    def __init__(self, S):
        self.Action = S
        self.rng = np.random.default_rng()

        self.worm_lengths = deque()
        # The contributions to the plaquette tell you how an n contributes to dn.
        # Opposite directions contribute oppositely, which is exactly what you want.
        # That way, if the worm moves north, you increase n by 1, but if the worm then
        # immediately moves south it would cross the same link but decrease n by 1,
        # so that the constraint on this cul-de-sac would be restored.
        self.plaquette = np.array([+1, +1, -1, -1]) # east, north, west, south

    def __str__(self):
        return 'ClassicWorm'

    def _neighboring_plaquettes(self, here):
        # east, north, west, south
        return self.Action.Lattice.mod(here + np.array([[0,-1], [+1,0], [0,+1], [-1,0]]))

    def _surrounding_links(self, here):
        # These are the directions we'd like to move the head of the defect.
        east, north, west, south = self._neighboring_plaquettes(here)

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
            'Vortex_Vortex': extendable.array(self.Action.Lattice.form(0, steps)),
            'Worm_Length':   extendable.array(np.zeros(steps)),
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

        # The documentation gives a definitive statement about moving the head only.
        # But we could equally well move the tail, making the opposite moves in the opposite worm evolution.
        # This can be accomplished simply by multiplying the offered changes to the links by -1.
        # We can randomly decide this orientation of the worm
        orientation = self.rng.choice([-1, +1])
        # and then simply multiply it into the constraint-restoring proposals.
        change_n = orientation * self.plaquette

        # We start with a constraint-satisfying configuration of n that is in the z sector.
        n = configuration['n'].copy()
        # and insert both the head and tail onto any random plaquette---because the head and the tail are
        # coincident, they don't change the action and so any choice should be equally weighted.
        tail = self.rng.choice(L.coordinates)
        #   The only exception is that when W=1 the unmodified Z constraint is (mod 1) which is always satisfied, even with an open worm.
        #   Therefore, we can put down an open worm from the start.  When W>1 the head and the tail have to be coincident
        #   in order to satisfy the unmodified constraint.
        head = (tail.copy() if S.W != 1 else self.rng.choice(L.coordinates))
        # by placing the head and tail down we have moved to the g sector!
        # Now we are ready to start evolving in z union g.

        while True:

            # In the general case we will uniformly choose between 4 moves,
            # but if the head and tail are together, we add the g--> z transition.
            # This has likelihood of 20%, conditioned on the worm being closed.
            # If it is proposed, however, the change in action is 0 and it is automatically accepted as a z configuration.
            if ((head == tail).all() or (S.W==1)) and (self.rng.uniform(0, 1) >= 0.8):
                # Just as we could put down an open worm when W=1, we can also remove an open worm in that case.
                # In other words, if W=1 we always have the possibility of return a Z configuration from the existing G configuration.
                wl = displacements.sum()
                self.worm_lengths.append(wl)
                return {'n': n, 'phi': phi, 'Vortex_Vortex': displacements, 'Worm_Length': wl}
            
            # Conditioned on not transitioning to z, we make a uniform choice of the 4 possible directions.
            choice = self.rng.choice([0,1,2,3])

            # We may move the head to 1 of 4 neighboring plaquettes
            the_next = self._neighboring_plaquettes(head)[choice]
            # in which case we will cross the corresponding links.
            link = self._surrounding_links(head)[choice]

            # Crossing the link changes n and therefore the action.
            change_link = dphi[link] - 2*np.pi*n[link]
            delta_n = change_n[choice]
            change_S = (
                (S.kappa / 2) *
                (-2*np.pi*delta_n) *
                (2*change_link - 2*np.pi*delta_n)
            )

            # Now we must compute the Metropolis amplitude
            #
            #   A = min(1, exp(-ΔS) )
            #
            A = np.clip(np.exp(-change_S), a_min=0, a_max=1)

            # and Metropolis-test the update.
            if self.rng.uniform(0, 1) < A:
                # If it accepted we move the head
                head = the_next
                # and cross the link.
                n[link] += delta_n

            # Finally, we tally the worm,
            x, y = L.mod(head-tail)
            displacements[x, y] += 1
            # and consider our next move.

    def report(self):
        l = np.array(self.worm_lengths)
        return f'There were {len(l)} worms.\nWorms lengths:\n    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}'

