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

# A second, *shorter* clean elementary move: a two-link "elbow" of two links sharing a
# corner in one 2-plane.  Its charge dipole shifts the head *diagonally*, by ê_μ - ê_ν
# (here +ê_2 - ê_3), into a non-orthogonal hypercube neighbour.  Two links is the minimum
# that can move charge at all (a single link never changes q), so this is the leanest
# possible sheet-extending step.  Its orbit under the axis permutations supplies all six
# canonical diagonal directions, and it interleaves freely with the 3-link :data:`_SEED`
# moves.
_DIAG_SEED_HEAD = (1, 0, 1, 0)
_DIAG_SEED = (
    (0, (1, 1, 1, 1), +1),
    (1, (1, 0, 1, 1), +1),
)


class IntersectionWorm(ReadWriteable, Generator):
    r"""
    Prokof'ev–Svistunov worm for the $q = dn\wedge dn = 0$ constraint in 4D.

    The head and tail live on hypercubes (4-cells; there is one per site in 4D).
    Moving the head by one hypercube extends the dragged sheet of $F = dn$ by a clean,
    coordinated change of $n$; when the head returns to the tail the constraint is restored
    everywhere and the configuration is emitted into the Markov chain.

    Two move families extend the sheet, each leaving the charge changed only by a $\pm 1$
    dipole that advances the head:

    - an **orthogonal** step: a coordinated *three*-link change that shifts
      the head by $\pm\hat e_{\mu}$ to a face neighbour --- the two hypercubes share a 3-cell;
    - a **diagonal** step: a *two*-link "elbow" --- two links meeting
      at a corner in one 2-plane --- that shifts the head by $\hat e_{\mu} - \hat e_{\nu}$ to
      a diagonal neighbour, where the two hypercubes share only a 2-cell.

    Two links is the *minimum* that can move charge at all: a single-link change has
    $q = d\Delta n \wedge d\Delta n \equiv 0$ and moves nothing.  The library stores the
    axis-permutation orbits of both seeds.  Counting *oriented* moves (a displacement and
    its reverse count separately) there are

    - $2 \times 4 = 8$ **orthogonal** moves, one along each signed axis $\pm\hat e_{\mu}$
      --- the eight face neighbours, one per dual-lattice-link direction; and
    - $4 \times 3 = 12$ **diagonal** moves $\hat e_{\mu} - \hat e_{\nu}$ with $\mu \neq \nu$,
      equivalently the $\binom{4}{2} = 6$ coordinate 2-planes each supplying only its
      *opposite-sign* diagonal $\pm(\hat e_{\mu} - \hat e_{\nu})$.  The same-sign diagonal
      $\hat e_{\mu} + \hat e_{\nu}$ is *not* reachable by any two-link elbow, so a plane
      contributes two oriented moves, not four.

    That is $8 + 12 = 20 = 2M$ oriented moves ($M = 10$ canonical displacements --- those
    whose first nonzero component is $+1$), and each step draws one uniformly.  The families
    interleave freely: a diagonal step is cheaper (fewer links, hence smaller $\Delta S$ and
    higher acceptance) and preserves the parity of $\sum_{k} x_{k}$, while an orthogonal step
    flips it, so together they mix the head's walk more efficiently than either alone.

    As the head moves we tally the head$-$tail displacement histogram that yields the
    :class:`~.Intersection_Intersection` correlator $\langle e^{i\theta_h} e^{-i\theta_t}\rangle$ ---
    the two-point function of the operator $e^{i\theta}$ that inserts a unit of
    vortex-sheet self-intersection $q = dn\wedge dn$.

    .. warning::

        Restricted to $D = 4$.  This generator updates $n$ only, so it is not ergodic
        on its own; at least combine it with a $\varphi$-update such as
        :class:`~.villain.SiteUpdate`.

    .. note::

        The move library holds the axis-permutation orbits of two clean move shapes ---
        the 3-link orthogonal :data:`_SEED` and the 2-link diagonal :data:`_DIAG_SEED` ---
        which still do not always offer a clean step on every trail.  When the drawn shape
        is not clean the proposal is rejected and the head stays put.  Like a Metropolis
        rejection of a clean shape, this is a genuine *self-loop* of the chain, and it is
        **counted** in the head$-$tail histogram: self-loops leave detailed balance between
        distinct states untouched, so the histogram still samples the stationary marginal
        $\propto G(r)$.  Dropping these stay-puts would instead reweight $G(r)$ by the
        configuration-dependent fraction of clean proposals and bias the correlator.  (One
        *could* propose only among the clean shapes and count each as a step, but that
        asymmetric proposal would then require a Metropolis--Hastings correction; treating
        an unclean draw as an ordinary rejection avoids it.)  Further enriching the library
        improves efficiency and ergodicity.  See :ref:`the No-Intersection model
        <no_intersection>`.

    .. danger::

        It's not clear to us whether this worm is an ergodic update to $n$ even with the combination of the :class:`~supervillain.generator.villain.ExactUpdate`.
        In particular, we've had a hard time understanding whether it creates 2-knots.
    """

    def __init__(self, S):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('IntersectionWorm requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('IntersectionWorm is only implemented for D = 4.')

        self.Action = S
        self.Lattice = S.Lattice
        self.kappa = S.kappa
        self.rng = np.random.default_rng()

        self.worm_lengths = deque()

        # Build the move library: for each canonical displacement d (the four positive
        # unit directions and the six diagonals ê_μ - ê_ν), the clean shapes that shift
        # the +1 head by +d, expressed RELATIVE to the head.  The opposite displacement
        # -d is generated on the fly by negating a shape, so we store only canonical d.
        self._library = self._build_library()
        self._directions = sorted(self._library)

    def __str__(self):
        return 'IntersectionWorm'

    # ------------------------------------------------------------------ library

    def _build_library(self):
        r"""
        The axis-permutation orbits of the seed moves, bucketed by the displacement the
        move gives the head.  Each entry is a tuple of ``(direction, relative_site,
        coefficient)`` triples, with the relative site measured from the head (the +1
        defect).  Only *canonical* displacements (first nonzero component positive) are
        stored; the opposite direction is recovered by negating a shape at step time.
        """
        L = self.Lattice
        N = L.N

        def orbit(seed, seed_head):
            # Relative form of the seed (links measured from its head), permuted over axes.
            seed_rel = tuple(
                (mu, tuple(s[k] - seed_head[k] for k in range(4)), c)
                for mu, s, c in seed
            )
            for perm in permutations(range(4)):
                out = []
                for mu, rs, c in seed_rel:
                    nrs = [0, 0, 0, 0]
                    for k in range(4):
                        nrs[perm[k]] = rs[k]
                    out.append((perm[mu], tuple(nrs), c))
                yield tuple(out)

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
        # ``steps`` is the number of unit hops the displacement makes: 1 for the
        # orthogonal 3-link seed, 2 for the diagonal 2-link elbow.
        for seed, seed_head, steps in ((_SEED, _SEED_HEAD, 1), (_DIAG_SEED, _DIAG_SEED_HEAD, 2)):
            for template in orbit(seed, seed_head):
                sep = separation(template, anchor)
                if sep is None or sum(abs(x) for x in sep) != steps:
                    continue
                if next((x for x in sep if x != 0), 0) <= 0:
                    continue  # keep only canonical directions; -d is made by negation
                library.setdefault(sep, []).append(template)
        return library

    # ------------------------------------------------------------------ helpers

    def _change_from_shape(self, head, d, sign, shape):
        r"""
        The $\Delta n$ (as a dict ``link -> coefficient``) for moving the head by the
        displacement ``sign``$\,d$ using library ``shape``.

        A forward step ($+d$) places the template with its head at ``head``$+d$.  A
        backward step is the *negated* template anchored at ``head`` — exactly the
        inverse of the forward step that would have arrived here, so
        backward$\circ$forward $= -\Delta n + \Delta n = 0$.
        """
        N = self.Lattice.N
        if sign > 0:
            anchor = tuple(head[k] + d[k] for k in range(4))
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

    def _sheet_segment(self, n, q_now, head, d, sign):
        r"""
        Propose a sheet-extending $\Delta n$ that moves the head by the displacement
        ``sign``$\,d$ (a unit hop for the orthogonal shapes, a diagonal $\hat e_\mu -
        \hat e_\nu$ hop for the elbow shapes), choosing **one** library shape uniformly
        at random and attempting only it.  Returns ``(change, target)`` if that shape
        gives a clean dipole shift on the current ``n``, else ``(None, None)``.

        Selecting a single, uniformly-chosen shape makes the proposal **symmetric**:
        the reverse move is the same shape with the opposite sign, drawn with the same
        probability $\tfrac{1}{2M}\cdot\tfrac{1}{K}$ ($M$ canonical displacements, $K$
        shapes for this one), and it is guaranteed clean on the proposed state.  Detailed
        balance then holds with the plain Metropolis acceptance $\min(1, e^{-\Delta S})$.
        (Trying several shapes and taking the first clean one would make $q$ asymmetric
        and break this.)
        """
        N = self.Lattice.N
        shapes = self._library[d]
        shape = shapes[self.rng.integers(0, len(shapes))]

        target = tuple((head[k] + sign * d[k]) % N for k in range(4))
        want = {} if target == head else {target: 1, head: -1}

        change = self._change_from_shape(head, d, sign, shape)
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
        r"""Storage for the inline ``Intersection_Intersection`` histogram and ``Worm_Length``."""
        L = self.Lattice
        return {
            'Intersection_Intersection': Batch(steps, shape=L.dims),
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
        # Every canonical displacement d contributes two head moves (+d and -d); together
        # with the "close" option this gives 2M+1 equally likely choices when head==tail.
        n_moves = 2 * len(self._directions)

        n = configuration['n'].copy()
        dphi = d(configuration['phi'])
        q_now = charge(n)

        displacements = np.zeros(L.dims)

        # Lay down head and tail on the same random hypercube; ΔS = 0, so this g-sector
        # entry is automatically accepted.
        tail = tuple(int(x) for x in self.rng.integers(0, N, size=D))
        head = tail

        while True:
            # When the head and tail coincide, offer the (2M+1)-th move: close the worm
            # and emit the (valid) configuration.  All 2M+1 options are equally likely.
            if head == tail and self.rng.uniform(0, 1) < 1.0 / (n_moves + 1):
                wl = displacements.sum()
                self.worm_lengths.append(wl)
                new_n = Form(n, degree=1, lattice=L)
                return configuration | {'n': new_n, 'Intersection_Intersection': displacements, 'Worm_Length': wl}

            # Otherwise propose a uniformly random one of the 2M head moves: a canonical
            # displacement (orthogonal or diagonal) and a sign for its orientation.
            hop = self._directions[self.rng.integers(0, len(self._directions))]
            sign = 1 if self.rng.integers(0, 2) == 0 else -1

            change, target = self._sheet_segment(n, q_now, head, hop, sign)
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

            # Tally the head−tail displacement for the Intersection_Intersection correlator.
            disp = tuple((head[k] - tail[k]) % N for k in range(D))
            displacements[disp] += 1

    def report(self):
        l = np.array(self.worm_lengths)
        if len(l) == 0:
            return 'There were 0 worms.'
        return (f'There were {len(l)} worms.\nWorms lengths:\n'
                f'    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}')
