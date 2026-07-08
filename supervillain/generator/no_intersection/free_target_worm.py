#!/usr/bin/env python

from itertools import product

import numpy as np

from supervillain.lattice import Form, d as _d
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge, two_link_kernel
from supervillain.generator.no_intersection.adaptive_worm import AdaptiveIntersectionWorm, _ravel


class FreeTargetWorm(AdaptiveIntersectionWorm):
    r"""
    $\Delta q$-classified Metropolis--Hastings worm for the $q = dn \wedge dn = 0$
    constraint in 4D.

    Where the adaptive worms draw a direction first and keep only the templates whose
    charge transport matches it, this worm enumerates one head-anchored family, computes
    each placement's $\Delta q$ on the current $F = dn$, and lets the transport go where
    it wants: $\Delta q \equiv 0$ is an idle, a clean dipole
    $\{\mathrm{head}{:}\,{-1},\ y{:}\,{+1}\}$ is a mover to $y$ --- orthogonal, diagonal,
    or farther --- and anything else is discarded.  One uniform draw over the clean union
    $C$, accepted with $\min\!\big(1, (\left|C\right|/\left|C'\right|)\,e^{-\Delta S}\big)$.
    The draw is uniform over $C$ unconditionally --- there is no per-template class
    weighting here (see ``class_weights`` below).

    The worm auto-closes the instant the head returns to the tail; the emitted histogram
    is pre-populated with $1$ at the origin (the pivot dwell), and a worm whose opening
    proposal rejects is a legitimate zero-length worm (one that still emits
    $\texttt{Worm\_Length} = 1$, the pre-populated pivot dwell alone).

    .. warning::

        Restricted to $D = 4$.  Opt-in --- not part of the default
        :func:`~supervillain.generator.no_intersection.Hammer`.
    """

    # Coefficient box for the two-link pairs; sign-symmetric so the family is
    # negation-closed (closure fact (2)); |c| = 2 carries the mixed-magnitude
    # Diophantine solutions.
    COEFF_BOX = (-2, -1, 1, 2)

    # Two slots may pair iff their linear-reach supports come within this taxicab
    # distance.  1 provably contains every pair the TwoLinkAdaptiveWorm can use: its
    # mover pairs each touch one of two ADJACENT cells (so their supports come within
    # distance 1) and its idle pairs both touch the head (distance 0).  This bound is
    # placement-intrinsic -- it never references head or target -- so it cannot break
    # closure.  Raising it admits farther-flung pairs (possible distant transport) at
    # roughly (ball volume) growth in family size and per-iteration enumeration cost.
    PAIR_PROXIMITY = 1

    def __init__(self, S, class_weights=None):
        if class_weights is not None:
            raise ValueError(
                'FreeTargetWorm draws uniformly from the classified clean union; '
                'the parent per-template class_weights do not apply.  (A future '
                'idle-vs-mover draw weighting is a different, deferred knob.)')
        super().__init__(S)
        self._reach = self._link_reach()     # derive once; _link_reach probes a scratch lattice
        self._candidate_family = self._build_family()
        # Flatten the family once for the compiled classifier.
        self._dq_stencil = two_link_kernel.dq_stencil_arrays()
        self._candidate_flat = two_link_kernel.flatten_family(self._candidate_family,
                                                               self._self_charge)

    def __str__(self):
        return 'FreeTargetWorm'

    def report(self):
        r"""A short summary of the worm lengths sampled so far."""
        l = np.array(self.worm_lengths)
        if len(l) == 0:
            return 'There were 0 free-target worms.'
        return (f'There were {len(l)} free-target worms.\nWorm lengths:\n'
                f'    mean {l.mean()}\n    std  {l.std()}\n    max  {int(max(l))}')

    # ------------------------------------------------------------------ family

    def _slot_support(self, mu, r):
        r"""The linear-reach support of link slot ``(mu, r)``: the hypercube cells (mod
        $N$) where its background-linear charge response can be nonzero."""
        N = self.Lattice.N
        return frozenset(tuple((r[k] + off[k]) % N for k in range(4))
                         for off in self._reach[mu])

    def _build_family(self):
        r"""
        The head-anchored candidate family: every shape, relative to the head, whose
        charge-reach support contains the head.  Deduped and negation-closed --- the two
        properties the closure fact (2) rests on.
        """
        # Support-anchoring gives closure by construction: Δq ≠ 0 at y forces y into the
        # union of the links' supports (the cross term dδ_1∧dδ_2 lives in the
        # INTERSECTION of the two supports, so the union is exactly the possible-response
        # region).  The negated shape re-anchored at y then satisfies the same membership
        # rule, and negation-closure of the coefficient sets puts it in the family.
        N = self.Lattice.N
        origin = (0, 0, 0, 0)
        seen = set()
        family = []

        def register(shape, sc):
            # sc: the shape's self-charge pattern (offsets from the head), precomputed by
            # the caller -- one unit_M full-lattice derivation per link pair / library
            # shape, scaled or shifted per variant, never one per coefficient combo.
            shape = tuple(sorted(shape))
            key = frozenset(shape)
            if key in seen:
                return
            seen.add(key)
            if shape not in self._self_charge:
                self._self_charge[shape] = sc
            family.append(shape)

        # (a) Single links touching the head, both signs.  These are the same slots
        # _idle_links enumerates; c = ±1 only (a mover's unit dipole demands c | 1, and
        # a ±1 idle exists wherever a ±2 one would).  The single-link self-wedge vanishes
        # identically.
        A = self._reach_touching_slots((origin,))
        for mu, r in A:
            for c in (+1, -1):
                register(((mu, r, c),), ())

        # (b) COEFF_BOX pairs: at least one slot touches the head, the partner's support
        # within PAIR_PROXIMITY of the first slot's.  Partner candidates are the slots
        # touching the ball of radius PAIR_PROXIMITY around the anchored slot's support.
        # The cross term M_12 is derived ONCE per pair and scaled by c1*c2.
        ball_offsets = [off for off in product(range(-self.PAIR_PROXIMITY,
                                                     self.PAIR_PROXIMITY + 1), repeat=4)
                        if sum(abs(x) for x in off) <= self.PAIR_PROXIMITY]
        done_pairs = set()
        for mu1, r1 in A:
            supp = self._slot_support(mu1, r1)
            ball = sorted({tuple((cell[k] + off[k]) % N for k in range(4))
                           for cell in supp for off in ball_offsets})
            for mu2, r2 in self._reach_touching_slots(tuple(ball)):
                if (mu2, r2) == (mu1, r1):
                    continue
                pair = tuple(sorted(((mu1, r1), (mu2, r2))))
                if pair in done_pairs:
                    continue
                done_pairs.add(pair)
                (m1, s1), (m2, s2) = pair
                unit_M = self._shape_self_charge(((m1, s1, 1), (m2, s2, 1)))
                for c1, c2 in product(self.COEFF_BOX, repeat=2):
                    register(((m1, s1, c1), (m2, s2, c2)),
                             self._scaled_self_charge(unit_M, c1 * c2))

        # (c) ± the parent library's multi-link templates (3-link orthogonal, 4-link
        # same-sign diagonal, and the 2-link shapes, which mostly dedup into (b)),
        # support-anchored: one relative placement per support cell, so the head can sit
        # anywhere the template's response reaches.  The self-charge is quadratic in
        # Delta n -- identical for the negated shape -- and translating the shape by -u
        # shifts its pattern offsets by -u.
        for shapes in self._library.values():
            for shape in shapes:
                if len(shape) == 1:
                    continue                     # singles already covered by (a)
                base_sc = self._shape_self_charge(shape)
                support = set()
                for mu, rs, _c in shape:
                    for off in self._reach[mu]:
                        support.add(tuple((rs[k] + off[k]) % N for k in range(4)))
                for off, _v in base_sc:
                    support.add(off)
                for u in sorted(support):
                    shifted_sc = tuple((tuple((off[k] - u[k]) % N for k in range(4)), v)
                                       for off, v in base_sc)
                    for sgn in (+1, -1):
                        register(tuple((mu, tuple((rs[k] - u[k]) % N for k in range(4)),
                                        sgn * c) for mu, rs, c in shape),
                                 shifted_sc)
        return family

    # ------------------------------------------------------------------ enumeration (oracle)

    def _placed(self, head, shapes=None):
        r"""The family (or the subset ``shapes``) placed at ``head``: ``(change, shape)``
        pairs, in the given order."""
        N = self.Lattice.N
        out = []
        for shape in (self._candidate_family if shapes is None else shapes):
            change = {}
            for mu, rs, c in shape:
                link = (mu,) + tuple((head[k] + rs[k]) % N for k in range(4))
                change[link] = change.get(link, 0) + c
            out.append((change, shape))
        return out

    def classified_set_reference(self, n_arr, q0, head, shapes=None):
        r"""
        The clean union $C$ at ``head`` on background ``n_arr`` (charge ``q0``), by a
        global ``charge`` recompute --- the readable oracle the local enumerations are
        validated against.  Returns ``[(change, target), ...]`` with ``target == head``
        for idles and duplicate $\Delta n$ collapsed, so the length is $\left|C\right|$.

        A global recompute per shape is slow, so full-family calls are hand-run only;
        tests validate on subsets via ``shapes``.
        """
        L = self.Lattice
        seen = set()
        out = []
        for change, _shape in self._placed(head, shapes):
            key = frozenset((lnk, c) for lnk, c in change.items() if c != 0)
            if not key or key in seen:
                continue
            trial = n_arr.copy()
            for lnk, c in change.items():
                trial[lnk] += c
            dq = charge(Form(trial, degree=1, lattice=L)) - q0
            defects = {tuple(int(x) for x in z[1:]): int(dq[tuple(z)])
                       for z in np.argwhere(dq != 0)}
            if defects == {}:
                seen.add(key)
                out.append((change, head))
            elif (len(defects) == 2 and defects.get(head) == -1
                  and sorted(defects.values()) == [-1, 1]):
                target = next(cell for cell, v in defects.items() if v == 1)
                seen.add(key)
                out.append((change, target))
        return out

    # ------------------------------------------------------------------ enumeration (local)

    def classified_set_local_py(self, F, head, shapes=None):
        r"""
        Pure-Python twin of :meth:`classified_set_reference`: same result, same order,
        but $\Delta q$ comes from :meth:`_local_dq` on the maintained ``F`` $= dn$
        instead of a global recompute.  :meth:`step_reference` drives this (full
        family); tests compare subsets against the oracle via ``shapes``.
        """
        seen = set()
        out = []
        for change, shape in self._placed(head, shapes):
            key = frozenset((lnk, c) for lnk, c in change.items() if c != 0)
            if not key or key in seen:
                continue
            defects = self._local_dq(F, change, head, shape)
            if defects == {}:
                seen.add(key)
                out.append((change, head))
            elif (len(defects) == 2 and defects.get(head) == -1
                  and sorted(defects.values()) == [-1, 1]):
                target = next(cell for cell, v in defects.items() if v == 1)
                seen.add(key)
                out.append((change, target))
        return out

    def classified_set_local(self, F, head):
        r"""
        Compiled twin of :meth:`classified_set_local_py`: the whole family's $\Delta q$
        is classified against ``F`` by :func:`.two_link_kernel.classify_mask`; the clean
        shapes become ``(change, target)`` pairs in Python (deduped, family order).
        Bit-for-bit with the pure-Python version; used by :meth:`step`.
        """
        N = self.Lattice.N
        Farr = np.asarray(F)
        F2 = np.ascontiguousarray(Farr.reshape(Farr.shape[0], -1))
        status = two_link_kernel.classify_mask(
            F2, N, np.array(head, dtype=np.int64), _ravel(head, N),
            *self._candidate_flat, *self._dq_stencil)
        seen = set()
        out = []
        for si in range(len(self._candidate_family)):
            if status[si] == -2:
                continue
            shape = self._candidate_family[si]
            change = {}
            for mu, rs, c in shape:
                link = (mu,) + tuple((head[k] + rs[k]) % N for k in range(4))
                change[link] = change.get(link, 0) + c
            key = frozenset((lnk, c) for lnk, c in change.items() if c != 0)
            if not key or key in seen:
                continue
            seen.add(key)
            if status[si] == -1:
                out.append((change, head))
            else:
                t = int(status[si])
                target = ((t // (N * N * N)) % N, (t // (N * N)) % N,
                          (t // N) % N, t % N)
                out.append((change, target))
        return out

    # ------------------------------------------------------------------ step

    def step(self, configuration):
        r"""
        One worm update: drop a coincident head/tail on a random hypercube, evolve the
        head with the $\Delta q$-classified move until it returns to the tail, and emit
        the resulting valid configuration and the inline head$-$tail histogram
        (pre-populated at the origin with the pivot dwell).  The fast path --- compiled
        classification, and the reverse enumeration carried into the next iteration;
        :meth:`step_reference` is the fresh-enumeration oracle it matches bit-for-bit.
        """
        return self._run_free_worm(configuration, self.classified_set_local, reuse=True)

    def step_reference(self, configuration):
        r"""
        Readable worm walk driving the pure-Python enumeration, recomputed fresh every
        iteration --- the oracle :meth:`step` is validated bit-for-bit against (which is
        also what validates :meth:`step`'s enumeration reuse).
        """
        return self._run_free_worm(configuration, self.classified_set_local_py,
                                   reuse=False)

    def _run_free_worm(self, configuration, classify, reuse):
        r"""
        The auto-close worm walk, parameterized by the enumeration
        ``classify(F, head) -> [(change, target), ...]`` and by whether the reverse
        enumeration is carried into the next iteration (``reuse``).
        """
        # --- Why this Metropolis--Hastings worm is exact (the load-bearing facts) ---
        # (1) Reversal identity.  Writing F' = F + dΔn,
        #         Δq(-Δn on F') = -Δq(Δn on F)
        #     (self-wedge even, cross terms odd under Δn -> -Δn): the reverse of a clean
        #     idle is a clean idle, the reverse of a clean mover is the reversed dipole.
        # (2) Closure.  The family is support-anchored (a shape is enumerable at h iff h
        #     lies in its charge-reach support) and negation-closed, so a mover to y --
        #     whose Δq ≠ 0 at y forces y into the support -- has its negation enumerable
        #     at y.  (1)+(2): the reverse of every accepted move is IN the reverse clean
        #     union, so |C'| >= 1 (no zero division) and 1/|C'| is the true reverse
        #     proposal probability.  Breaking closure (e.g. "optimizing" the family) biases
        #     the chain with every constraint-validity test still green.
        # (3) Uniform draw over DISTINCT Δn (dedup by frozenset): forward probability
        #     exactly 1/|C|.  Δq is a function of (F, Δn), so Δn determines the target --
        #     each element of C lands on a distinct (s', head'), and every transition is
        #     proposable in exactly one way.  There is no direction menu and no idle slot,
        #     hence no cross-slot argument and no menu factor in the balance.
        # (4) Auto-close.  Every state (n, head, tail) carries plain weight w(n): no
        #     proposal probability is diverted to a close branch, so pivot states carry NO
        #     excess weight.  Closing and reopening at a uniformly-drawn tail is a
        #     tail-relabel on the pivot class {(n, x, x) : x}, a free symmetry move (w(n)
        #     does not care where a coincident head/tail sits); quotienting by it, the 1/V
        #     open factor cancels the pivot class's V-fold multiplicity and the balance
        #     collapses to w(n)·(1/|C|)·A = w(n')·(1/|C'|)·A'.  The pivot's single dwell
        #     slot is tallied by PRE-POPULATING the origin bin with 1 at open; arrival at
        #     the pivot closes WITHOUT tallying (the next worm's pre-population is that
        #     slot).  A rejected -- or idle-accepted -- opening proposal leaves head ==
        #     tail and therefore closes: a legitimate zero-length worm (Worm_Length = 1,
        #     the pre-populated pivot dwell alone), emitted, never retried (retrying would
        #     under-count pivot dwell).
        #
        # Termination: mid-flight |C| >= 1 always -- after an acceptance C' contains the
        # reverse (facts (1)+(2)); after a rejection C is unchanged -- so a positive-
        # probability path back to the tail always exists and the walk halts almost
        # surely.  An empty C can only occur at open (before any move): close immediately.
        L = self.Lattice
        N = L.N
        D = L.D
        n = np.asarray(configuration['n']).astype(np.int64)
        dphi = np.asarray(_d(configuration['phi']))
        F = np.asarray(_d(configuration['n'])).astype(np.int64, copy=False)
        displacements = np.zeros(L.dims)

        tail = tuple(int(x) for x in self.rng.integers(0, N, size=D))
        head = tail
        displacements[L.origin] += 1     # the pivot dwell slot (fact (4))

        def emit():
            wl = displacements.sum()
            self.worm_lengths.append(wl)
            return configuration | {'n': Form(n, degree=1, lattice=L),
                                    'Intersection_Intersection': displacements,
                                    'Worm_Length': wl}

        def touch(change, s):
            # add s*change to both n and the maintained F (s = +1 apply, -1 revert);
            # integer arithmetic, so a revert is bit-exact -- which is what keeps the
            # kept-on-rejection enumeration valid.
            for link, c in change.items():
                n[link] += s * c
                local_charge.apply_link_to_F(F, link[0], link[1:], s * c, N)

        C = classify(F, head)
        while True:
            if not C:
                return emit()                                # only possible at open
            change, target = C[int(self.rng.integers(0, len(C)))]
            dS = self._delta_S(dphi, n, change)              # pre-move n
            touch(change, +1)
            Cp = classify(F, target)
            if self.rng.uniform(0, 1) < min(1.0, (len(C) / len(Cp)) * np.exp(-dS)):
                head = target
                C_next = Cp
            else:
                touch(change, -1)
                C_next = C
            if head == tail:
                return emit()                                # auto-close (fact (4))
            disp = tuple((head[k] - tail[k]) % N for k in range(D))
            displacements[disp] += 1
            C = C_next if reuse else classify(F, head)
