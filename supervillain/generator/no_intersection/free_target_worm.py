#!/usr/bin/env python

from itertools import product

import numpy as np

from supervillain.lattice import Form, Lattice, d as _d
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge, two_link_kernel
from supervillain.generator.no_intersection.adaptive_worm import AdaptiveIntersectionWorm


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

    The worm auto-closes the instant the head returns to the tail; the emitted histogram
    is pre-populated with $1$ at the origin (the pivot dwell), and a worm whose opening
    proposal rejects is a legitimate zero-length worm.

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
        super().__init__(S, class_weights=class_weights)
        self._reach = self._link_reach()     # derive once; _link_reach probes a scratch lattice
        self._family = self._build_family()

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
        for shape in (self._family if shapes is None else shapes):
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
