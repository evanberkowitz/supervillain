#!/usr/bin/env python

from collections import deque
import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
import supervillain.h5.extendable as extendable

from supervillain.lattice import _Lattice2D
import numba

import logging
logger = logging.getLogger(__name__)

class ClassicWorm(ReadWriteable, Generator):
    r'''
    This implements the classic worm of Prokof'ev and Svistunov :cite:`PhysRevLett.87.160601` for the worldline links $m\in\mathbb{Z}$ which satisfy $\delta m = 0$ on every site.

    On top of a constraint-satisfying configuration we put down a worm and let the head move, changing the crossed links.
    We uniformly propose a move in all 4 directions and Metropolize the change.

    Additionally, when the head and tail coincide, we allow a fifth possible move, where we remove the worm and emit the updated $z$ configuration into the Markov chain.
    
    As we evolve the worm we tally the histogram that yields the :class:`~.Spin_Spin` correlation function.

    .. warning ::
        
        When $W>1$ this update algorithm is not ergodic on its own.  It doesn't change $v$ at all.
        However, when $W=1$ we can always pick $v=0$ (any other choice may be absorbed into $m$), and this generator can stand alone.

    .. note ::

       This class contains kernels accelerated using numba.

    .. seealso::

       There is :class:`a reference implementation without any numba acceleration <supervillain.generator.reference_implementation.worldline.ClassicWorm>`.

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

    def __str__(self):
        return 'ClassicWorm'

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

        new_m, spin_spin = worm_kernel(self.rng,
            _Lattice2D(self.Action.Lattice.dims),
            S.kappa,
            head, tail,
            m, delta_v_by_W, change_m
        )

        wl = spin_spin.sum()
        self.worm_lengths.append(wl)
        return {'m': new_m, 'v': v, 'Spin_Spin': spin_spin, 'Worm_Length': wl}


    def report(self):
        l = np.array(self.worm_lengths)
        return f'There were {len(l)} worms.\nWorms lengths:\n    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}'

@numba.jit
def worm_kernel(rng, _Lattice, kappa, head, tail, m, delta_v_by_W, change_m):

    displacements = np.zeros((_Lattice.nt, _Lattice.nx), dtype=np.int64)

    L = _Lattice

    while True:
            # In the general case we will uniformly choose between 4 moves,
            # but if the head and tail are together, we add the g--> z transition.
            # This has likelihood of 20%, conditioned on the worm being closed.
            # If it is proposed, however, the change in action is 0 and it is automatically accepted as a z configuration.
            if (head == tail).all() and (np.random.uniform(0., 1.) >= 0.8):
                return m, displacements

            # Conditioned on not transitioning to z, we make a uniform choice of the 4 possible directions.
            choice = np.random.choice(np.arange(4))

            # Now we propose a move to the next position.
            t, x = L.neighboring_sites(head)
            nxt = np.array([t[choice], x[choice]], dtype=np.int64)
            # in which case we will cross the corresponding link.
            link = L.adjacent_links(0, head)[choice]

            # Crossing the link changes m and therefore the action.
            change_link = m[link] - delta_v_by_W[link]
            delta_m = change_m[choice]
            change_S = (
                (1 / (2*kappa)) *
                delta_m *
                (2*change_link + delta_m)
            )

            # Now we must compute the Metropolis amplitude
            #
            #   A = min(1, exp(-ΔS) )
            #
            A = np.min(np.array([1., np.exp(-change_S)])) # numba doesn't support clip?

            # and Metropolis-test the update.
            if np.random.uniform(0., 1.) < A:
                # If it accepted we move the head
                head = nxt
                # and cross the link.
                m[link] += delta_m

            # Finally, we tally the worm,
            x, y = L.mod(head-tail)
            displacements[x, y] +=1
            # and consider our next move.


