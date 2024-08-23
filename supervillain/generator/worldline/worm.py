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
    This implements the classic worm of Prokof'ev and Svistunov :cite:`PhysRevLett.87.160601` for the worldline links $m\in\mathbb{Z}$ which satisfy $\delta m = 0$ on every site.

    On top of a constraint-satisfying configuration we put down a worm and let the head move, changing the crossed links.
    We uniformly propose a move in all 4 directions and Metropolize the change.

    Additionally, when the head and tail coincide, we allow a fifth possible move, where we remove the worm and emit the updated $z$ configuration into the Markov chain.
    
    As we evolve the worm we tally the histogram that yields the :class:`~.Spin_Spin` correlation function.

    .. warning ::
        
        When $W>1$ this update algorithm is not ergodic on its own.  It doesn't change $v$ at all.
        However, when $W=1$ we can always pick $v=0$ (any other choice may be absorbed into $m$), and this generator can stand alone.

    '''

    def __init__(self, S):
        if not isinstance(S, supervillain.action.Worldline):
            raise ValueError('The classic worm algorithm update requires the Worldline action.')

        self.Action = S
        self.rng = np.random.default_rng()

        self.worm_lengths = deque()

        # The contributions to the divergence tell you how an m contributes to δm.
        # Opposite directions contribute oppositely, which is exactly what you want.
        # That way, if the worm moves north, you increase n by 1, but if the worm then
        # immediately moves south it would cross the same link but decrease m by 1,
        # so that the constraint on this cul-de-sac would be restored.
        self.divergence = np.array([+1, +1, -1, -1]) # east, north, west, south

    def _neighboring_sites(self, here):
        # east, north, west, south
        return self.Action.Lattice.mod(here + np.array([[+1,0], [0,+1], [-1,0], [0,-1]]))

    def _adjacent_links(self, here):
        # These are the directions we'd like to move the head of the defect.
        east, north, west, south = self._neighboring_sites(here)

        return ((0, here [0], here [1]), # t link to the east
                (1, here [0], here [1]), # x link to the north
                (0, west [0], west [1]), # t link to the west
                (1, south[0], south[1])) # x link to the south

    def inline_observables(self, steps):
        r'''
        The worm algorithm can measure the ``Spin_Spin`` correlator.
        We also store the ``Worm_Length`` for each step.
        '''

        return {
            'Spin_Spin':    extendable.array(self.Action.Lattice.form(0, steps)),
            'Worm_Length':  extendable.array(np.zeros(steps)),
        }

    def step(self, configuration):
        r'''
        Given a constraint-satisfying configuration, returns another constraint-satisfying configuration udpated via worm as described above.
        '''

        S = self.Action
        L = S.Lattice

        displacements = L.form(0)

        m = configuration['m'].copy()

        # This algorithm will not update v; but it is useful to precompute δv
        # which is used in the evaluation of the changes in action.
        v = configuration['v'].copy()
        delta_v_by_W = L.delta(2, v) / S._W

        # The documentation gives a definitive statement about moving the head only.
        # But we could equally well move the tail, making the opposite moves in the opposite worm evolution.
        # This can be accomplished simply by multiplying the offered changes to the links by -1.
        # We can randomly decide this orientation of the worm
        orientation = self.rng.choice([-1, +1])
        # and then simply multiply it into the constraint-restoring proposals.
        change_m = orientation * self.divergence

        # We start with a constraint-satisfying configuration of n that is in the z sector.
        # and insert both the head and tail onto any random site---because the head and the tail are
        # coincident, they don't change the action and so any choice should be equally weighted.
        tail = self.rng.choice(L.coordinates)
        head = tail.copy()
        # by placing the head and tail down we have moved to the g sector!
        # Now we are ready to start evolving in z union g.

        while True:
            # In the general case we will uniformly choose between 4 moves,
            # but if the head and tail are together, we add the g--> z transition.
            # This has likelihood of 20%, conditioned on the worm being closed.
            # If it is proposed, however, the change in action is 0 and it is automatically accepted as a z configuration.
            if (head == tail).all() and (self.rng.uniform(0, 1) >= 0.8):
                wl = displacements.sum()
                self.worm_lengths.append(wl)
                return {'m': m, 'v': v, 'Spin_Spin': displacements, 'Worm_Length': wl}

            # Conditioned on not transitioning to z, we make a uniform choice of the 4 possible directions.
            choice = self.rng.choice([0,1,2,3])

            # Now we propose a move to the next position.
            next = self._neighboring_sites(head)[choice]
            # in which case we will cross the corresponding link.
            link = self._adjacent_links(head)[choice]

            # Crossing the link changes m and therefore the action.
            change_link = m[link] - delta_v_by_W[link]
            delta_m = change_m[choice]
            change_S = (
                (1 / (2*S.kappa)) *
                delta_m *
                (2*change_link + delta_m)
            )

            # Now we must compute the Metropolis amplitude
            #
            #   A = min(1, exp(-ΔS) )
            #
            A = np.clip(np.exp(-change_S), a_min=0, a_max=1)

            # and Metropolis-test the update.
            if self.rng.uniform(0, 1) < A:
                # If it accepted we move the head
                head = next
                # and cross the link.
                m[link] += delta_m

            # Finally, we tally the worm,
            x, y = L.mod(head-tail)
            displacements[x, y] +=1
            # and consider our next move.

    def report(self):
        l = np.array(self.worm_lengths)
        return f'There were {len(l)} worms.\nWorms lengths:\n    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}'


