#!/usr/bin/env python

from collections import deque
import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.batch import Batch
from supervillain.lattice import delta, Form
import numba

import logging
logger = logging.getLogger(__name__)


@numba.njit
def _flat_index(coords, N, D):
    r"""Flat ``0 … N^D−1`` index for a *D*-component coordinate array, each component taken mod *N*."""
    idx = np.int64(0)
    for i in range(D):
        idx = idx * N + (coords[i] % N)
    return idx


@numba.njit
def worm_kernel(rng, D, N, coord_1d, kappa, head, tail, m_2d, dvW_2d, change_m):
    r"""
    D-general worm kernel (numba-accelerated).

    *m_2d* and *dvW_2d* have shape ``(D, N**D)``; the spatial axes are collapsed so that
    ``_flat_index(site, N, D)`` selects the correct element.

    Returns the updated *m_2d* and a flat displacement histogram of length ``N**D``.
    """
    V = np.int64(1)
    for _ in range(D):
        V *= N
    n_dirs = 2 * D
    displacements = np.zeros(V, dtype=np.int64)

    while True:
            # In the general case we uniformly choose between 2D moves,
            # but if the head and tail are together, we add the g --> z transition.
            # This has probability 1/(2D+1), making all 2D+1 options equally likely.
            # If it is proposed, the change in action is 0 and it is automatically accepted as a z configuration.
            same = True
            for i in range(D):
                if head[i] != tail[i]:
                    same = False
                    break
            if same and rng.uniform(0., 1.) < 1.0 / (n_dirs + 1):
                return m_2d, displacements

            # Conditioned on not transitioning to z, we make a uniform choice of the 2D directions.
            choice = rng.integers(0, n_dirs)
            k = choice % D        # which axis
            forward = choice < D  # True for +ê_k, False for −ê_k

            # Now we propose a move to the next position.
            # Advance (or retreat) by one step in direction k using FFT-convention coordinates.
            next_head = head.copy()
            arr_idx = head[k] % N
            if forward:
                next_head[k] = coord_1d[(arr_idx + 1) % N]
            else:
                next_head[k] = coord_1d[(arr_idx - 1 + N) % N]

            # in which case we will cross the corresponding link.
            # Moving +ê_k crosses the link at head; moving −ê_k crosses the link at head−ê_k (= next_head).
            if forward:
                link_flat = _flat_index(head, N, D)
            else:
                link_flat = _flat_index(next_head, N, D)

            # Crossing the link changes m and therefore the action.
            change_link = m_2d[k, link_flat] - dvW_2d[k, link_flat]
            delta_m = change_m[choice]
            change_S = (1.0 / (2.0 * kappa)) * delta_m * (2.0 * change_link + delta_m)

            # Now we must compute the Metropolis amplitude
            #
            #   A = min(1, exp(-ΔS) )
            #
            A = min(1.0, np.exp(-change_S))

            # and Metropolis-test the update.
            if rng.uniform(0., 1.) < A:
                # If accepted, move the head
                head = next_head
                # and cross the link.
                m_2d[k, link_flat] += delta_m

            # Finally, tally the head−tail displacement for the Spin_Spin correlator,
            displacements[_flat_index(head - tail, N, D)] += 1
            # and consider our next move.


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

        D = S.Lattice.D
        # Moving the head in direction +ê_k crosses the link at the current site,
        # shifting (δm) by −1 at the current site and +1 at the next site.
        # Moving in −ê_k crosses the link behind the head with the opposite sign.
        # The orientation (±1) drawn once per worm flips all contributions together;
        # change_m = orientation * divergence is the actual per-direction Δm offered.
        self.divergence = np.array([+1]*D + [-1]*D, dtype=int)

    def __str__(self):
        return 'ClassicWorm'

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
        N = L.N
        V = N**D

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
        tail = self.rng.choice(L.coordinates).astype(np.int64)
        head = tail.copy()
        # By placing the head and tail down we have moved to the g sector!
        # Now we are ready to start evolving in z union g.

        # Collapse spatial axes so the kernel can use a flat site index.
        m_2d  = np.ascontiguousarray(m.reshape(D, V))
        dvW_2d = np.ascontiguousarray(delta_v_by_W.reshape(D, V))

        new_m_2d, disp_flat = worm_kernel(
            self.rng,
            np.int64(D), np.int64(N),
            L._coord_1d.astype(np.int64),
            float(S.kappa),
            head, tail,
            m_2d, dvW_2d,
            change_m,
        )

        new_m    = Form(new_m_2d.reshape(m.shape), degree=1, lattice=L)
        # if not S.valid({'m': new_m}):
        #     raise ValueError('The new configuration does not satisfy the constraint δm = 0 everywhere.')
        spin_spin = disp_flat.reshape(L.dims)

        wl = int(spin_spin.sum())
        self.worm_lengths.append(wl)
        return configuration | {'m': new_m, 'Spin_Spin': spin_spin, 'Worm_Length': wl}

    def report(self):
        l = np.array(self.worm_lengths)
        return f'There were {len(l)} worms.\nWorms lengths:\n    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}'
