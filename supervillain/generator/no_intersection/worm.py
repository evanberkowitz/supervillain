#!/usr/bin/env python

from collections import deque
from itertools import permutations, product
import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.batch import Batch
from supervillain.lattice import Lattice, Form, d
from supervillain.generator.no_intersection.charge import dF_entries, local_dq, wedge_pairs

import logging
logger = logging.getLogger(__name__)


# The clean elementary moves: one representative per orbit class of every change of
# at most 3 links with coefficients ±1 whose Δq on an empty background is a clean unit
# dipole, found by the exhaustive search in example/no-intersection-move-search.py and
# stored head-relative (the +1 defect at the origin) in the auto-generated moves.py.
# The full move library is the orbit of these seeds under the hyperoctahedral group
# (all 384 signed axis permutations) and global negation, re-anchored so the +1 defect
# defines the head.
from supervillain.generator.no_intersection.moves import SEEDS as _SEEDS

# The library is a pure geometric object (head-relative templates, independent of the
# lattice size and the action), so it is built once per process and shared.
_LIBRARY_CACHE = None


class IntersectionWorm(ReadWriteable, Generator):
    r"""
    Prokof'ev–Svistunov worm for the $q = dn\wedge dn = 0$ constraint in 4D.

    The head and tail live on hypercubes (4-cells; there is one per site in 4D).
    Moving the head by one hypercube extends the dragged sheet of $F = dn$ by a clean,
    coordinated three-link change; when the head returns to the tail the constraint is
    restored everywhere and the configuration is emitted into the Markov chain.

    As the head moves we tally the head$-$tail displacement histogram that yields the
    ``Intersection_Intersection`` correlator $\langle e^{i\theta_h} e^{-i\theta_t}\rangle$ ---
    the two-point function of the operator $e^{i\theta}$ that inserts a unit of
    vortex-sheet self-intersection $q = dn\wedge dn$.

    .. warning::

        Restricted to $D = 4$.  This generator updates $n$ only, so it is not ergodic
        on its own; at least combine it with a $\varphi$-update such as
        :class:`~.villain.SiteUpdate`.

    .. note::

        The move library contains **every** elementary clean move: all changes of at
        most 3 links with coefficients $\pm 1$ whose $\Delta q$ on an empty
        background is a clean unit dipole, as enumerated exhaustively by
        ``example/no-intersection-move-search.py`` (828 shapes per direction, in 41
        symmetry classes once transforms *and* re-anchoring translations are counted;
        ``moves.py`` stores 93 translation-blind representatives, a harmless
        redundancy the builder de-duplicates) and expanded from those seeds
        under all 384 signed axis permutations and global negation, re-anchored so
        that the +1 defect sits on the head.  Not every shape offers a clean step on
        every trail; stalled proposals are simply rejected (the head stays put),
        which is detailed-balance safe.  See :ref:`the No-Intersection model
        <no_intersection>`.

    .. note::

        The change $\Delta q$ of a proposal is computed *locally* from the linearized
        wedge $\Delta q = \Delta F\wedge F + F\wedge\Delta F + \Delta F\wedge\Delta F$
        with $\Delta F = d(\Delta n)$ supported on a handful of plaquettes, so each
        head move costs $O(1)$ rather than $O(\text{volume})$.

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

        # Build the move library: for each of the 8 unit directions ±ê_μ, the clean
        # 3-link shapes whose dipole separation (+1 site minus −1 site) is that unit
        # vector, expressed RELATIVE to the +1 defect (the head).
        self._library = self._build_library()

        # Complementary plaquette pairs (A, B) and the sign σ(A⌢B) entering the
        # 4-form wedge (a∧b)_{(0,1,2,3)}[x] = Σ σ(A⌢B) a_A[x] b_B[x+ê_A].
        self._wedge_pairs = wedge_pairs(self.Lattice)

    def __str__(self):
        return 'IntersectionWorm'

    # ------------------------------------------------------------------ library

    @staticmethod
    def _transformed(template, perm, flips, negate):
        r"""
        Apply a signed axis permutation to a template of ``(direction, site,
        coefficient)`` triples: first send axis $k$ to ``perm[k]``, then reflect the
        axes with ``flips[k] == -1``.  Reflecting the axis a link points along maps
        the link $[s, s+\hat e_k]$ to $[-s-\hat e_k, -s]$: the base shifts by
        $-\hat e_k$ and the coefficient flips.  ``negate`` flips all coefficients
        (allowed because $q$ is quadratic in $n$: $-\Delta n$ makes the same dipole
        on an empty background but is a genuinely different move).
        """
        out = []
        for mu, rs, c in template:
            site = [0, 0, 0, 0]
            for k in range(4):
                site[perm[k]] = rs[k]
            nmu = perm[mu]
            coeff = c
            for k in range(4):
                if flips[k] == -1:
                    site[k] = -site[k]
                    if k == nmu:
                        site[k] -= 1
                        coeff = -coeff
            if negate:
                coeff = -coeff
            out.append((nmu, tuple(site), coeff))
        return tuple(sorted(out))

    def _build_library(self):
        r"""
        The orbit of the ``moves.SEEDS`` under the 384 signed axis permutations and
        global negation, bucketed by the dipole separation (+1 site minus −1 site, a
        unit vector) and re-anchored so the +1 defect sits at the origin of the
        template's relative coordinates.  Each entry is a tuple of ``(direction,
        relative_site, coefficient)`` triples measured from the head.

        The candidates are validated with the local $\Delta q$ on an empty scratch
        lattice: not every transform is an exact lattice symmetry of the wedge
        (single-axis reflections pick up shifts, like ★★), so only the candidates
        whose dipole stays clean are kept.  The result is cached per process.
        """
        global _LIBRARY_CACHE
        if _LIBRARY_CACHE is not None:
            return _LIBRARY_CACHE

        # A scratch lattice comfortably larger than any template, so that placing a
        # template near the middle cannot wrap around the torus.
        scratch = Lattice(4, 8)
        anchor = (4, 4, 4, 4)
        pairs = wedge_pairs(scratch)
        empty = np.zeros((6,) + scratch.dims, dtype=int)

        def dipole(template):
            '''The (+1 site, -1 site) of the template placed at ``anchor`` on an empty lattice.'''
            change = {}
            for mu, rs, c in template:
                link = (mu,) + tuple((anchor[k] + rs[k]) % scratch.N for k in range(4))
                change[link] = change.get(link, 0) + c
            dq = local_dq(scratch, empty, {l: c for l, c in change.items() if c != 0}, pairs=pairs)
            if len(dq) != 2:
                return None
            (a, va), (b, vb) = sorted(dq.items())
            if {va, vb} != {1, -1}:
                return None
            return (a, b) if va == 1 else (b, a)

        library = {}
        seen = set()
        for seed in _SEEDS:
            for perm in permutations(range(4)):
                for flips in product((1, -1), repeat=4):
                    for negate in (False, True):
                        template = self._transformed(seed, perm, flips, negate)
                        if template in seen:
                            continue
                        seen.add(template)
                        pm = dipole(template)
                        if pm is None:
                            continue
                        plus, minus = pm
                        sep = tuple(int(p - m) for p, m in zip(plus, minus))
                        if sum(abs(x) for x in sep) != 1:
                            continue
                        # Re-anchor: measure the links from the +1 defect.
                        shift = tuple(p - a for p, a in zip(plus, anchor))
                        rebased = tuple(sorted(
                            (mu, tuple(rs[k] - shift[k] for k in range(4)), c)
                            for mu, rs, c in template
                        ))
                        library.setdefault(sep, set()).add(rebased)

        _LIBRARY_CACHE = {sep: tuple(sorted(shapes)) for sep, shapes in library.items()}
        return _LIBRARY_CACHE

    # ------------------------------------------------------------------ helpers

    def _place(self, shape, anchor, factor):
        r"""
        The $\Delta n$ (as a dict ``link -> coefficient``) for library ``shape``
        anchored (its +1 defect) at ``anchor``, scaled by ``factor``$= \pm 1$.
        """
        N = self.Lattice.N
        change = {}
        for direction, rs, c in shape:
            site = tuple((anchor[k] + rs[k]) % N for k in range(4))
            link = (direction,) + site
            change[link] = change.get(link, 0) + factor * c
        return {link: c for link, c in change.items() if c != 0}

    def _dF_entries(self, change):
        r"""
        The plaquette changes $\Delta F = d(\Delta n)$ of a sparse link change;
        see :func:`supervillain.generator.no_intersection.charge.dF_entries`.
        """
        return dF_entries(self.Lattice, change)

    def _dq(self, F, change):
        r"""
        The change of the charge density $q = F\wedge F$ from a sparse link change,
        computed locally in $O(1)$;
        see :func:`supervillain.generator.no_intersection.charge.local_dq`.
        """
        return local_dq(self.Lattice, F, change, pairs=self._wedge_pairs)

    def _sheet_segment(self, F, head, mu, sign):
        r"""
        Propose a sheet-extending $\Delta n$ that moves the head by ``sign``$\,\hat
        e_\mu$, choosing **one** move uniformly at random from the proposals for that
        step and attempting only it.  The proposals are the shapes in the
        ``sign``$\,\hat e_\mu$ bucket anchored at the target (their +1 defect lands
        on the target) together with the *negated* shapes of the $-$``sign``$\,\hat
        e_\mu$ bucket anchored at the head (each the exact inverse of a forward step
        that could have arrived here).  Returns ``(change, target)`` if the chosen
        move gives a clean dipole shift on the current configuration, else
        ``(None, None)``.

        Selecting a single, uniformly-chosen move makes the proposal **symmetric**:
        the exact inverse of every option is one of the reverse step's options, drawn
        with the same probability $\tfrac{1}{2D}\cdot\tfrac{1}{K}$, and its
        cleanliness on the proposed state is automatic.  Detailed balance then holds
        with the plain Metropolis acceptance $\min(1, e^{-\Delta S})$.  (Trying
        several shapes and taking the first clean one would break this.)
        """
        N = self.Lattice.N
        step = tuple(sign if k == mu else 0 for k in range(4))
        back = tuple(-x for x in step)
        direct = self._library.get(step, ())
        negated = self._library.get(back, ())
        K = len(direct) + len(negated)
        if K == 0:
            return None, None

        target = tuple((head[k] + step[k]) % N for k in range(4))

        i = int(self.rng.integers(0, K))
        if i < len(direct):
            change = self._place(direct[i], target, +1)
        else:
            change = self._place(negated[i - len(direct)], head, -1)

        want = {target: 1, head: -1}
        if self._dq(F, change) == want:
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
        n_dirs = 2 * D

        n = configuration['n'].copy()
        dphi = d(configuration['phi'])
        F = np.asarray(d(n)).astype(int)   # maintained incrementally as the head moves

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
                return configuration | {'n': new_n, 'Intersection_Intersection': displacements, 'Worm_Length': wl}

            # Otherwise propose a uniformly random one of the 2D head moves.
            mu = int(self.rng.integers(0, D))
            sign = 1 if self.rng.integers(0, 2) == 0 else -1

            change, target = self._sheet_segment(F, head, mu, sign)
            if change is not None:
                # Metropolis-test the change in the Villain action.
                dS = self._delta_S(dphi, n, change)
                if self.rng.uniform(0, 1) < min(1.0, np.exp(-dS)):
                    for link, c in change.items():
                        n[link] += c
                    for (idx, site), v in self._dF_entries(change).items():
                        F[(idx,) + site] += v
                    head = target
            # If the chosen move is not clean on the current trail, the proposal is
            # simply rejected and the head stays put.

            # Tally the head−tail displacement for the Intersection_Intersection correlator.
            disp = tuple((head[k] - tail[k]) % N for k in range(D))
            displacements[disp] += 1

    def report(self):
        l = np.array(self.worm_lengths)
        if len(l) == 0:
            return 'There were 0 worms.'
        return (f'There were {len(l)} worms.\nWorms lengths:\n'
                f'    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}')
