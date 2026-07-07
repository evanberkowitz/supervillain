#!/usr/bin/env python

from itertools import product

import numpy as np

from supervillain.lattice import Form, Lattice
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection.adaptive_worm import AdaptiveIntersectionWorm


class TwoLinkAdaptiveWorm(AdaptiveIntersectionWorm):
    r"""
    Two-link enrichment of the $F$-adaptive Metropolis--Hastings worm.

    On a hot vortex-sheet background the fixed library rarely offers a *clean* single- or
    templated-multi-link dipole, so the head stalls.  This worm additionally enumerates,
    for the drawn direction, every **two-link** coordinated move on the current background
    $F = dn$ — pairs of distinct links whose combined charge change
    $\Delta q = c_{1} L_{1}(F) + c_{2} L_{2}(F) + c_{1} c_{2} M_{12}$ is the exact head
    dipole — and the head-fixed **two-link idle isotopies** ($\Delta q \equiv 0$).  These
    ride in the *same* menu slots as the existing moves (one per orthogonal direction, plus
    the idle slot): they enlarge each clean set $C$, and the inherited acceptance
    $\min(1, (|C|/|C'|)\,e^{-\Delta S})$ corrects the state-dependence exactly.

    The candidate families are fixed, state-independent (geometry $+$ the coefficient box
    :data:`COEFF_BOX`), and stored canonically, so the inherited ``_change_from_shape``
    sign mechanism makes the reverse of every forward move the same shape negated — the
    head$\leftrightarrow$target symmetry the $|C|/|C'|$ ratio requires (see the exactness
    facts documented in :meth:`~.AdaptiveIntersectionWorm.step_reference`).  The coefficient
    box includes $|c| = 2$ to carry the mixed-magnitude Diophantine solutions (pairs whose
    only clean move needs $|c| = 2$, e.g. $(c_{1}-1)(c_{2}-1) = 1$).

    Only enumeration changes: the $\Delta q$ computation (:meth:`_local_dq`, per-link
    stencils $+$ the precomputed self-charge $c_{1} c_{2} M_{12}$), the acceptance ratio,
    the menu, and ``step``/``step_reference`` are all inherited unchanged.

    .. warning::

        Restricted to $D = 4$, updates $n$ only, orthogonal transport $\pm\hat e_{\mu}$
        (diagonals deferred, as in the base worm).  Not wired into the default ``Hammer``;
        construct it explicitly to experiment.
    """

    # Coefficient box for two-link moves; |c| = 2 carries the mixed-magnitude solutions.
    COEFF_BOX = (-2, -1, 1, 2)

    def __init__(self, S, class_weights=None):
        super().__init__(S, class_weights=class_weights)
        self._two_movers = self._build_two_link_movers()
        self._two_idles = self._build_two_link_idles()

    def __str__(self):
        return 'TwoLinkAdaptiveWorm'

    # ------------------------------------------------------------------ family construction

    def _shape_self_charge(self, shape):
        r"""
        The background-independent self-charge $d\Delta n \wedge d\Delta n$ of ``shape``,
        as ``((offset, value), ...)`` with offsets measured (mod $N$) from the placement
        anchor.  Computed on a scratch lattice of the *same* extent $N$ as the target, so
        any small-$N$ periodic-image cross terms are exact.  For a two-link shape this is
        exactly $c_{1} c_{2} M_{12}$; the self-wedges vanish.
        """
        N = self.Lattice.N
        L0 = Lattice(4, N)
        anchor = (N // 2,) * 4
        dn = L0.zeros(1, dtype=int)   # Lattice.zeros returns a Form; charge(dn) needs no re-wrap
        for mu, rs, c in shape:
            dn[(mu,) + tuple((anchor[k] + rs[k]) % N for k in range(4))] += c
        q = np.asarray(charge(dn))
        return tuple(
            (tuple((int(h[1 + k]) - anchor[k]) % N for k in range(4)), int(q[tuple(h)]))
            for h in np.argwhere(q != 0)
        )

    def _reach_touching_slots(self, cells):
        r"""
        Link slots ``(mu, r)`` whose single-link charge reach touches any cell in ``cells``
        (each a 4-tuple offset from the anchor).  A ``mu``-link at ``r`` responds on
        ``r + reach[mu]``, so it touches ``x`` iff ``r == x - off`` for some ``off`` in the
        reach.  Offsets are reduced mod $N$ so the box is correct on small lattices.
        """
        N = self.Lattice.N
        reach = self._link_reach()
        slots = set()
        for mu, offsets in reach.items():
            for x in cells:
                for off in offsets:
                    r = tuple((x[k] - off[k]) % N for k in range(4))
                    slots.add((mu, r))
        return sorted(slots)

    def _build_two_link_movers(self):
        r"""
        For each canonical orthogonal direction ``dd``, the fixed family of two-link mover
        shapes: distinct link pairs (both reach-touching the dipole endpoints) with every
        coefficient pair in :data:`COEFF_BOX`, kept when the union of the two links' reach
        and the shape's self-charge support covers *both* defect cells (a necessary,
        $F$-independent condition — a shape whose support already misses an endpoint can
        never be that dipole for any background).  Self-charges are registered into
        ``self._self_charge`` so :meth:`_local_dq` can score them.

        Shapes are stored relative to the anchor ``target`` (where the ``+1`` defect lands),
        with the ``-1`` defect at offset ``-dd`` — the same convention as the library, so
        the inherited sign mechanism yields the reverse move for free.
        """
        N = self.Lattice.N
        reach = self._link_reach()
        target_off = (0, 0, 0, 0)
        movers = {}
        for dd in self._ortho:
            head_off = tuple((-dd[k]) % N for k in range(4))
            slots = self._reach_touching_slots((target_off, head_off))
            shapes = []
            for i in range(len(slots)):
                mu1, r1 = slots[i]
                for j in range(i + 1, len(slots)):
                    mu2, r2 = slots[j]
                    # Linear-reach support of this link pair (F-independent).
                    support = set()
                    for mu, r in ((mu1, r1), (mu2, r2)):
                        for off in reach[mu]:
                            support.add(tuple((r[k] + off[k]) % N for k in range(4)))
                    for c1, c2 in product(self.COEFF_BOX, repeat=2):
                        shape = ((mu1, r1, c1), (mu2, r2, c2))
                        sc = self._shape_self_charge(shape)
                        cover = support | {off for off, _v in sc}
                        if target_off not in cover or head_off not in cover:
                            continue
                        self._self_charge[shape] = sc
                        shapes.append(shape)
            movers[dd] = shapes
        return movers

    # ------------------------------------------------------------------ enumeration seam

    def _mover_shapes(self, dd):
        r"""The library family plus the two-link movers for ``dd`` (one combined slot)."""
        return self._library[dd] + self._two_movers[dd]

    # ------------------------------------------------------------------ two-link idles

    def _build_two_link_idles(self):
        r"""
        The fixed family of two-link idle shapes: distinct link pairs both reach-touching
        the head, coefficients in :data:`COEFF_BOX`, stored relative to the head.  The box
        is sign-symmetric, so the coefficient-negated partner of every shape is also in the
        family — the reverse isotopy is always enumerated (``|I'| >= 1``).  Self-charges are
        registered for :meth:`_local_dq`.  Cleanliness ($\Delta q \equiv 0$) is decided per
        background at enumeration time, not here.
        """
        head = (0, 0, 0, 0)
        slots = self._reach_touching_slots((head,))
        shapes = []
        for i in range(len(slots)):
            mu1, r1 = slots[i]
            for j in range(i + 1, len(slots)):
                mu2, r2 = slots[j]
                for c1, c2 in product(self.COEFF_BOX, repeat=2):
                    shape = ((mu1, r1, c1), (mu2, r2, c2))
                    self._self_charge[shape] = self._shape_self_charge(shape)
                    shapes.append(shape)
        return shapes

    def _two_idle_changes(self, head):
        r"""The two-link idle shapes placed at ``head``, as ``(change, shape)`` pairs."""
        N = self.Lattice.N
        out = []
        for shape in self._two_idles:
            change = {}
            for mu, rs, c in shape:
                link = (mu,) + tuple((head[k] + rs[k]) % N for k in range(4))
                change[link] = change.get(link, 0) + c
            out.append((change, shape))
        return out

    def clean_idle_local(self, F, head):
        r"""
        Base single-link idles, then the clean two-link idles scored locally on ``F``.
        Deduped by the change's frozenset; single- and two-link changes never collide
        (distinct link counts), so the base order is preserved and the two-link tail shares
        its order with :meth:`clean_idle_reference` (both iterate ``self._two_idles``).
        """
        out = super().clean_idle_local(F, head)
        seen = {frozenset((lnk, c) for lnk, c in ch.items() if c != 0) for ch in out}
        for change, shape in self._two_idle_changes(head):
            key = frozenset((lnk, c) for lnk, c in change.items() if c != 0)
            if key in seen:
                continue
            if self._local_dq(F, change, head, shape) == {}:
                seen.add(key)
                out.append(change)
        return out

    def clean_idle_reference(self, n_arr, q0, head):
        r"""Global-recompute twin of :meth:`clean_idle_local` (same order)."""
        L = self.Lattice
        out = super().clean_idle_reference(n_arr, q0, head)
        seen = {frozenset((lnk, c) for lnk, c in ch.items() if c != 0) for ch in out}
        for change, _shape in self._two_idle_changes(head):
            key = frozenset((lnk, c) for lnk, c in change.items() if c != 0)
            if key in seen:
                continue
            trial = n_arr.copy()
            for lnk, c in change.items():
                trial[lnk] += c
            dq = charge(Form(trial, degree=1, lattice=L)) - q0
            if np.abs(dq).max() == 0:
                seen.add(key)
                out.append(change)
        return out
