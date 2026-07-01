#!/usr/bin/env python

from collections import deque
from itertools import permutations
import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.batch import Batch
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import charge

import logging
logger = logging.getLogger(__name__)


# One known clean elementary move, expressed as (direction, site, coefficient) with the
# +1 (head) defect landing at ``_SEED_HEAD``.  It shifts the head by +ê_3.  Every other
# clean move we use is generated from this one by relabelling the axes.
_SEED_HEAD = (1, 1, 0, 2)
_SEED = (
    (0, (1, 1, 1, 1), +1),
    (0, (1, 1, 1, 2), +1),
    (1, (2, 1, 1, 2), +1),
)


class ThetaWorm(ReadWriteable, Generator):
    r"""
    Prokof'ev–Svistunov worm for the $q = dn\wedge dn = 0$ constraint in 4D.

    The head and tail live on hypercubes (4-cells; there is one per site in 4D).
    Moving the head by one hypercube extends the dragged sheet of $F = dn$ by a clean,
    coordinated three-link change; when the head returns to the tail the constraint is
    restored everywhere and the configuration is emitted into the Markov chain.

    As the head moves we tally the head$-$tail displacement histogram that yields the
    :class:`Theta_Theta` correlator $\langle e^{i\theta_h} e^{-i\theta_t}\rangle$.

    .. warning::

        Restricted to $D = 4$.  This generator updates $n$ only, so it is not ergodic
        on its own; combine it with a $\phi$-update such as
        :class:`~.villain.SiteUpdate`.

    .. note::

        The move library here holds only the orbit of a single clean move shape, which
        does not always offer a clean step on every trail.  Stalled proposals are
        simply rejected (the head stays put), which is detailed-balance safe; enriching
        the library improves efficiency and ergodicity.  See :ref:`the No-Intersection
        model <no_intersection>`.
    """

    def __init__(self, S):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('ThetaWorm requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('The θ worm is only implemented for D = 4.')

        self.Action = S
        self.Lattice = S.Lattice
        self.kappa = S.kappa
        self.rng = np.random.default_rng()

        self.worm_lengths = deque()

        # Build the move library: for each positive direction μ, the clean 3-link
        # shapes that shift the +1 head by +ê_μ, expressed RELATIVE to the head.
        self._library = self._build_library()

    def __str__(self):
        return 'ThetaWorm'

    # ------------------------------------------------------------------ library

    def _build_library(self):
        r"""
        The orbit of :data:`_SEED` under the 24 axis permutations, bucketed by the
        unit direction the move shifts the head.  Each entry is a tuple of
        ``(direction, relative_site, coefficient)`` triples, with the relative site
        measured from the head (the +1 defect).
        """
        L = self.Lattice
        N = L.N

        # Relative form of the seed (links measured from the seed's head).
        seed_rel = tuple(
            (mu, tuple(s[k] - _SEED_HEAD[k] for k in range(4)), c)
            for mu, s, c in _SEED
        )

        def permuted(perm):
            out = []
            for mu, rs, c in seed_rel:
                nrs = [0, 0, 0, 0]
                for k in range(4):
                    nrs[perm[k]] = rs[k]
                out.append((perm[mu], tuple(nrs), c))
            return tuple(out)

        # Place a relative template with its head at ``head`` and read off the dipole.
        base = charge(L.zeros(1, dtype=int))

        def separation(template, head):
            dn = L.zeros(1, dtype=int)
            for mu, rs, c in template:
                site = tuple((head[k] + rs[k]) % N for k in range(4))
                dn[(mu,) + site] += c
            dq = charge(dn) - base
            nz = np.argwhere(dq != 0)
            if len(nz) != 2:
                return None
            defects = {tuple(int(x) for x in h[1:]): int(dq[tuple(h)]) for h in nz}
            (a, va), (b, vb) = sorted(defects.items())
            if {va, vb} != {1, -1}:
                return None
            plus = np.array(a if va == 1 else b)
            minus = np.array(b if va == 1 else a)
            if tuple(int(x) % N for x in plus) != tuple(int(x) % N for x in head):
                return None  # require the +1 defect to sit on the head
            sep = tuple(int(x) % N for x in (plus - minus))
            return tuple(x if x <= N // 2 else x - N for x in sep)

        anchor = (N // 2,) * 4
        library = {}
        for perm in permutations(range(4)):
            template = permuted(perm)
            sep = separation(template, anchor)
            if sep is not None and sum(abs(x) for x in sep) == 1:
                library.setdefault(sep, []).append(template)
        return library

    # ------------------------------------------------------------------ helpers

    def _change_from_shape(self, head, mu, sign, shape):
        r"""
        The $\Delta n$ (as a dict ``link -> coefficient``) for moving the head by
        ``sign``$\,\hat e_\mu$ using library ``shape``.

        A forward step ($+\hat e_\mu$) places the template with its head at
        ``head``$+\hat e_\mu$.  A backward step is the *negated* template anchored at
        ``head`` — exactly the inverse of the forward step that would have arrived
        here, so backward$\circ$forward $= -\Delta n + \Delta n = 0$.
        """
        N = self.Lattice.N
        positive = tuple(1 if k == mu else 0 for k in range(4))
        if sign > 0:
            anchor = tuple(head[k] + positive[k] for k in range(4))
            factor = +1
        else:
            anchor = tuple(head[k] for k in range(4))
            factor = -1
        change = {}
        for direction, rs, c in shape:
            site = tuple((anchor[k] + rs[k]) % N for k in range(4))
            link = (direction,) + site
            change[link] = change.get(link, 0) + factor * c
        return change

    def _sheet_segment(self, n, q_now, head, mu, sign):
        r"""
        Propose a sheet-extending $\Delta n$ that moves the head by ``sign``$\,\hat
        e_\mu$, choosing **one** library shape uniformly at random and attempting only
        it.  Returns ``(change, target)`` if that shape gives a clean dipole shift on
        the current ``n``, else ``(None, None)``.

        Selecting a single, uniformly-chosen shape makes the proposal **symmetric**:
        the reverse move is the same shape with the opposite sign, drawn with the same
        probability $\tfrac{1}{2D}\cdot\tfrac{1}{K}$, and it is guaranteed clean on the
        proposed state.  Detailed balance then holds with the plain Metropolis
        acceptance $\min(1, e^{-\Delta S})$.  (Trying several shapes and taking the
        first clean one would make $q$ asymmetric and break this.)
        """
        N = self.Lattice.N
        positive = tuple(1 if k == mu else 0 for k in range(4))
        shapes = self._library[positive]
        shape = shapes[self.rng.integers(0, len(shapes))]

        target = tuple((head[k] + (sign if k == mu else 0)) % N for k in range(4))
        want = {} if target == head else {target: 1, head: -1}

        change = self._change_from_shape(head, mu, sign, shape)
        trial = n.copy()
        for link, c in change.items():
            trial[link] += c
        dq = charge(trial) - q_now
        nz = np.argwhere(dq != 0)
        defects = {tuple(int(x) for x in h[1:]): int(dq[tuple(h)]) for h in nz}
        if defects == want:
            return change, target
        return None, None

    def _delta_S(self, dphi, n, change):
        r"""
        Change in the Villain action $\frac{\kappa}{2}\sum_\ell (d\phi - 2\pi n)_\ell^2$
        from adding ``change`` to $n$.  Only the touched links contribute:

        .. math::
            \Delta S = \sum_\ell \frac{\kappa}{2}\big[(A_\ell - 2\pi\,\Delta n_\ell)^2 - A_\ell^2\big],
            \quad A_\ell = (d\phi - 2\pi n)_\ell .
        """
        total = 0.0
        for link, c in change.items():
            A = dphi[link] - 2 * np.pi * n[link]
            total += (self.kappa / 2) * ((A - 2 * np.pi * c) ** 2 - A ** 2)
        return total

    # ------------------------------------------------------------------ observables

    def inline_observables(self, steps):
        r"""Storage for the inline ``Theta_Theta`` histogram and ``Worm_Length``."""
        L = self.Lattice
        return {
            'Theta_Theta': Batch(steps, shape=L.dims),
            'Worm_Length': Batch(steps, shape=(), dtype=float),
        }

    # ------------------------------------------------------------------ step

    def step(self, configuration):
        r"""
        Lay down a worm on a valid configuration, evolve the head until it returns to
        the tail, and emit the resulting valid configuration together with the inline
        head$-$tail displacement histogram.
        """
        L = self.Lattice
        N = L.N
        D = L.D
        n_dirs = 2 * D

        n = configuration['n'].copy()
        dphi = d(configuration['phi'])
        q_now = charge(n)

        displacements = np.zeros(L.dims)

        # Lay down head and tail on the same random hypercube; ΔS = 0, so this g-sector
        # entry is automatically accepted.
        tail = tuple(int(x) for x in self.rng.integers(0, N, size=D))
        head = tail

        while True:
            # When the head and tail coincide, offer the (2D+1)-th move: close the worm
            # and emit the (valid) configuration.  All 2D+1 options are equally likely.
            if head == tail and self.rng.uniform(0, 1) < 1.0 / (n_dirs + 1):
                wl = displacements.sum()
                self.worm_lengths.append(wl)
                new_n = Form(n, degree=1, lattice=L)
                return configuration | {'n': new_n, 'Theta_Theta': displacements, 'Worm_Length': wl}

            # Otherwise propose a uniformly random one of the 2D head moves.
            mu = int(self.rng.integers(0, D))
            sign = 1 if self.rng.integers(0, 2) == 0 else -1

            change, target = self._sheet_segment(n, q_now, head, mu, sign)
            if change is not None:
                # Metropolis-test the change in the Villain action.
                dS = self._delta_S(dphi, n, change)
                if self.rng.uniform(0, 1) < min(1.0, np.exp(-dS)):
                    for link, c in change.items():
                        n[link] += c
                    q_now = charge(n)
                    head = target
            # If no clean library move exists this step, the proposal is simply
            # rejected and the head stays put.

            # Tally the head−tail displacement for the Theta_Theta correlator.
            disp = tuple((head[k] - tail[k]) % N for k in range(D))
            displacements[disp] += 1

    def report(self):
        l = np.array(self.worm_lengths)
        if len(l) == 0:
            return 'There were 0 worms.'
        return (f'There were {len(l)} worms.\nWorms lengths:\n'
                f'    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}')
