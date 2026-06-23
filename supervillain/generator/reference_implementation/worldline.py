#!/usr/bin/env python

from collections import deque
import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.batch import Batch
from supervillain.lattice import delta

import logging
logger = logging.getLogger(__name__)

class ClassicWorm(ReadWriteable, Generator):
    r'''
    This implements the classic worm of Prokof'ev and Svistunov :cite:`PhysRevLett.87.160601` for the worldline links $m\in\mathbb{Z}$ which satisfy $\delta m = 0$ on every site.

    On top of a constraint-satisfying configuration we put down a worm and let the head move, changing the crossed links.
    We uniformly propose a move in all $2D$ directions and Metropolize the change.

    Additionally, when the head and tail coincide, we allow a $(2D+1)$-th possible move, where we remove the worm and emit the updated configuration into the Markov chain.

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

        D = S.Lattice.D
        # Moving the head in direction +ê_k crosses the link at the current site,
        # shifting (δm) by −1 at the current site and +1 at the next site.
        # Moving in −ê_k crosses the link behind the head with the opposite sign.
        # The orientation (±1) drawn once per worm flips all contributions together;
        # change_m = orientation * divergence is the actual per-direction Δm offered.
        self.divergence = np.array([+1]*D + [-1]*D)

    def __str__(self):
        return 'ClassicWorm'

    def _neighboring_sites(self, here):
        r"""
        The $2D$ sites adjacent to ``here``, in order $+\hat{e}_0, \ldots, +\hat{e}_{D-1},
        -\hat{e}_0, \ldots, -\hat{e}_{D-1}$.
        """
        L = self.Action.Lattice
        D = L.D
        eye = np.eye(D, dtype=int)
        return [L.mod(here + disp) for disp in np.vstack([eye, -eye])]

    def _adjacent_links(self, here):
        r"""
        The link crossed when the head moves in each of the $2D$ directions.

        Moving $+\hat{e}_k$ crosses link $(k, \texttt{here})$; moving $-\hat{e}_k$ crosses
        link $(k, \texttt{here} - \hat{e}_k)$.
        """
        L = self.Action.Lattice
        D = L.D
        neighbors = self._neighboring_sites(here)
        links = []
        for k in range(D):
            links.append((k,) + tuple(here))             # +ê_k: link at here
        for k in range(D):
            links.append((k,) + tuple(neighbors[D + k])) # −ê_k: link at here − ê_k
        return links

    def inline_observables(self, steps):
        r'''
        The worm algorithm can measure the ``Spin_Spin`` correlator.
        We also store the ``Worm_Length`` for each step.
        '''

        L = self.Action.Lattice
        return {
            'Spin_Spin':    Batch(steps, shape=L.dims),
            'Worm_Length':  Batch(steps, shape=(), dtype=float),
        }

    def step(self, configuration):
        r'''
        Given a constraint-satisfying configuration, returns another constraint-satisfying configuration updated via worm as described above.
        '''

        S = self.Action
        L = S.Lattice
        D = L.D
        n_dirs = 2 * D

        displacements = np.zeros(L.dims)

        m = configuration['m'].copy()

        # This algorithm will not update v; but it is useful to precompute δv
        # which is used in the evaluation of the changes in action.
        v = configuration['v'].copy()
        delta_v_by_W = delta(v) / S._W

        # The documentation gives a definitive statement about moving the head only.
        # But we could equally well move the tail, making the opposite moves in the opposite worm evolution.
        # This can be accomplished simply by multiplying the offered changes to the links by -1.
        # We can randomly decide this orientation of the worm
        orientation = self.rng.choice([-1, +1])
        # and then simply multiply it into the constraint-restoring proposals.
        change_m = orientation * self.divergence

        # We start with a constraint-satisfying configuration of m that is in the z sector,
        # and insert both the head and tail onto any random site---because the head and the tail are
        # coincident, they don't change the action and so any choice should be equally weighted.
        tail = self.rng.choice(L.coordinates)
        head = tail.copy()
        # By placing the head and tail down we have moved to the g sector!
        # Now we are ready to start evolving in z union g.

        while True:
            # In the general case we uniformly choose between 2D moves,
            # but if the head and tail are together, we add the g --> z transition.
            # This has probability 1/(2D+1), making all 2D+1 options equally likely.
            # If it is proposed, the change in action is 0 and it is automatically accepted.
            if (head == tail).all() and (self.rng.uniform(0, 1) < 1 / (n_dirs + 1)):
                wl = displacements.sum()
                self.worm_lengths.append(wl)
                return configuration | {'m': m, 'Spin_Spin': displacements, 'Worm_Length': wl}

            # Conditioned on not transitioning to z, we make a uniform choice of the 2D directions.
            choice = self.rng.integers(0, n_dirs)

            # Now we propose a move to the next position.
            next_site = self._neighboring_sites(head)[choice]
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
                # If accepted, move the head
                head = next_site
                # and cross the link.
                m[link] += delta_m

            # Finally, tally the head−tail displacement for the Spin_Spin correlator
            disp = L.mod(head - tail)
            displacements[tuple(disp)] += 1
            # and consider our next move.

    def report(self):
        l = np.array(self.worm_lengths)
        return f'There were {len(l)} worms.\nWorms lengths:\n    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}'
