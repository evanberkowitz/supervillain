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
        delta_v_by_W = L.delta(2, v) / S.W

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

    def report(self):
        l = np.array(self.worm_lengths)
        return f'There were {len(l)} worms.\nWorms lengths:\n    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}'


class Geometric(ReadWriteable, Generator):
    r'''
    Unlike in :py:class:`the Villain case <supervillain.generator.villain.worm.Geometric>`, the constraint in the Worldline formulation is $\delta m = 0$ on every site,
    where $m$ is a link-valued integer field.

    However, we can use the same conceptual algorithm, again adapting the undirected worm algorithm of Alet and Sørensen :cite:`PhysRevE.68.026702`.

    We randomly pick a site on which to place the head and tail of the worm, not changing any physical fields.

    From the current site, there are four or five allowable moves:

    The head can traverse any adjacent link, changing that link and pushing the defect over by one site.
    These moves change $m$ by ±1 and therefore change the action.

    If the head and tail are in the same place the constraint is satisfied and both the head and tail can be taken away,
    moving from $g$ to $z$ without changing the action.  But if the head and tail are in different places and we try to stop evolving the worm,
    the configuration will have two defects, violating the constraint, and costing infinite action.
    In other words, the Saint Patrick move only works against ouroborous (closed worldline) configurations.

    Let $\Delta S$ represent these five changes in action---four for single-link traversals and one for the Saint Patrick worm-elimination.

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

    The major advantage of this algorithm is that oftehn the :class:`~.WrappingUpdate` is rejected---it touches a macroscopic number of variables and tends to require big changes in action.
    The worm, in constrast, makes updates to a dynamically-generated selection of links.
    Some straightforward tests show that the worm reduces the explosion of the autocorrelation time of the :class:`~.TorusWrapping` dramatically, particularly with larger lattices.

    .. warning ::
        
        When $W>1$ this update algorithm is not ergodic on its own.  It doesn't change $v$ at all.
        However, when $W=1$ we can always pick $v=0$ (any other choice may be absorbed into $m$), and this generator can stand alone.
    '''

    def __init__(self, S):
        if not isinstance(S, supervillain.action.Worldline):
            raise ValueError('The (undirected) worm algorithm update requires the Worldline action.')

        self.Action = S
        self.rng = np.random.default_rng()

        self.worm_lengths = deque()

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
        delta_v_by_W = L.delta(2, v) / S.W

        # The contributions to the plaquette tell you how an n contributes to dn.
        # Opposite directions contribute oppositely, which is exactly what you want.
        # That way, if the worm moves north, you increase n by 1, but if the worm then
        # immediately moves south it would cross the same link but decrease n by 1,
        # so that the constraint on this cul-de-sac would be restored.
        divergence = np.array([+1, +1, -1, -1]) # east, north, west, south
        # The documentation gives a definitive statement about moving the head only.
        # But we could equally well move the tail, making the opposite moves in the opposite worm evolution.
        # This can be accomplished simply by multiplying the offered changes to the links by -1.
        # We can randomly decide this orientation of the worm
        orientation = self.rng.choice([-1, +1])
        # and then simply multiply it into the constraint-restoring proposals.
        change_m = orientation * divergence

        # We start with a constraint-satisfying configuration of n that is in the z sector.
        # m = configuration['m'].copy()
        # and insert both the head and tail onto any random plaquette---because the head and the tail are
        # coincident, they don't change the action and so any choice should be equally weighted.
        tail = self.rng.choice(L.coordinates)
        head = tail.copy()
        worm_length = 0
        # by placing the head and tail down we have moved to the g sector!
        # Now we are ready to start evolving in z union g.

        while True:
            # There are 4 or 5 possible moves that we may make.
            # We may move the head to 1 of 4 neighboring plaquettes
            next = self._neighboring_sites(head)
            # in which case we will cross the corresponding links.
            link = self._adjacent_links(head)

            # Crossing the link changes m and therefore the action.
            change_link = np.array([m[l] - delta_v_by_W[l] for l in link])
            change_S = (
                (1 / (2*S.kappa)) *
                change_m *
                (2*change_link + change_m)
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
                return {'m': m, 'v': v, 'Spin_Spin': displacements, 'Worm_Length': worm_length}
                # Note!  We don't need to Metrpolis accept/reject.  That was built in when we included the g --> z update
                # in the A amplitudes that went into the update probabilities.
                # That is why we went through this whole story rigamarole of distinguishing z configurations from diagonal g configurations:
                # to unify the treatment of the acceptance of the closed worm with the movement of the worm's head.

            # Otherwise we need to cross the link,
            m[link[choice]] += change_m[choice]
            worm_length += 1
            # move the head,
            head = next[choice]
            # tally the worm,
            x, y = L.mod(head-tail)
            displacements[x, y] +=1
            # and consider our next move.

    def report(self):
        l = np.array(self.worm_lengths)
        return f'There were {len(l)} worms.\nWorms lengths:\n    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}'

