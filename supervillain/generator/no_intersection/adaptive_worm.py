#!/usr/bin/env python

from itertools import product

import numpy as np

from supervillain.lattice import Form, Lattice, d as _d
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge, two_link_kernel
from supervillain.generator.no_intersection.worm import IntersectionWorm


def _ravel(site, N):
    # C-order flat index of a 4D hypercube site (matches F.reshape(n_planes, -1)).
    return ((int(site[0]) * N + int(site[1])) * N + int(site[2])) * N + int(site[3])


class AdaptiveIntersectionWorm(IntersectionWorm):
    r"""
    $F$-adaptive Metropolis--Hastings worm for the $q = dn\wedge dn = 0$ constraint in 4D.

    Where :class:`~.IntersectionWorm` draws one stencil and usually finds it
    constraint-violating on a fluxful background, this worm *enumerates* the clean set $C$
    of moves that advance the head in the drawn direction on the current $F = dn$, draws
    one uniformly, and accepts with the exact Metropolis--Hastings ratio
    $\min\!\big(1, (\left|C\right|/\left|C'\right|)\,e^{-\Delta S}\big)$.  Because $F$ is a
    function of the current state and not of the chain's history, this is ordinary
    state-dependent MH.

    Covers orthogonal $\pm\hat e_{\mu}$ transport and single-link idle isotopies.
    """

    def __init__(self, S, class_weights=None):
        super().__init__(S, class_weights=class_weights)
        # v1 menu: canonical orthogonal displacements only (diagonals deferred).
        self._ortho = [dd for dd in self._directions
                       if sum(abs(x) for x in dd) == 1]
        # One scratch lattice, reused for every self-charge derivation (same extent N as
        # the target, so small-N periodic-image cross terms are exact).
        self._scratch = Lattice(4, self.Lattice.N)

    def __str__(self):
        return 'AdaptiveIntersectionWorm'

    def report(self):
        r"""A short summary of the worm lengths sampled so far."""
        # The parent's per-family tallies are not populated: the adaptive proposal
        # enumerates the clean set rather than drawing-and-classifying a single shape.
        l = np.array(self.worm_lengths)
        if len(l) == 0:
            return 'There were 0 adaptive worms.'
        return (f'There were {len(l)} adaptive worms.\nWorm lengths:\n'
                f'    mean {l.mean()}\n    std  {l.std()}\n    max  {int(max(l))}')

    # ------------------------------------------------------------------ family construction

    def _shape_self_charge(self, shape):
        r"""
        The background-independent self-charge $d\Delta n \wedge d\Delta n$ of ``shape``
        (for a two-link shape, exactly $c_{1} c_{2} M_{12}$), as ``((offset, value), ...)``
        with offsets measured (mod $N$) from the anchor.
        """
        N = self.Lattice.N
        L0 = self._scratch
        anchor = (N // 2,) * 4
        dn = L0.zeros(1, dtype=int)   # Lattice.zeros returns a Form; charge(dn) needs no re-wrap
        for mu, rs, c in shape:
            dn[(mu,) + tuple((anchor[k] + rs[k]) % N for k in range(4))] += c
        q = np.asarray(charge(dn))
        return tuple(
            (tuple((int(h[1 + k]) - anchor[k]) % N for k in range(4)), int(q[tuple(h)]))
            for h in np.argwhere(q != 0)
        )

    @staticmethod
    def _scaled_self_charge(unit_M, factor):
        r"""Scale a unit cross-charge pattern by ``factor`` $= c_{1} c_{2}$, pruning zeros."""
        return tuple((off, factor * v) for off, v in unit_M if factor * v)

    def _reach_touching_slots(self, cells):
        r"""Link slots ``(mu, r)`` whose single-link charge reach touches any cell in ``cells``."""
        # A mu-link at r responds on r + reach[mu], so it touches x iff r == x - off for
        # some off in the reach.  Offsets are reduced mod N so the box is right at small N.
        N = self.Lattice.N
        reach = self._link_reach()
        slots = set()
        for mu, offsets in reach.items():
            for x in cells:
                for off in offsets:
                    r = tuple((x[k] - off[k]) % N for k in range(4))
                    slots.add((mu, r))
        return sorted(slots)

    # ------------------------------------------------------------------ clean sets (oracle)

    def _mover_shapes(self, dd):
        r"""The candidate mover family for direction ``dd`` (subclasses may enrich it)."""
        return self._library[dd]

    def clean_set_reference(self, n_arr, q0, head, dd, sign):
        r"""
        The clean movers for ``(dd, sign)`` at ``head`` on background ``n_arr`` (charge
        ``q0``), by a global ``charge`` recompute --- the readable oracle
        :meth:`clean_set_local` is validated against.  Returns ``[(change, target), ...]``
        with duplicate $\Delta n$ collapsed, so the length is $\left|C\right|$.
        """
        N = self.Lattice.N
        L = self.Lattice
        target = tuple((head[k] + sign * dd[k]) % N for k in range(4))
        want = {head: -1, target: 1}
        seen = set()
        out = []
        for shape in self._mover_shapes(dd):
            change = self._change_from_shape(head, dd, sign, shape)
            key = frozenset((lnk, c) for lnk, c in change.items() if c != 0)
            if key in seen:
                continue
            trial = n_arr.copy()
            for lnk, c in change.items():
                trial[lnk] += c
            dq = charge(Form(trial, degree=1, lattice=L)) - q0
            defects = {tuple(int(x) for x in z[1:]): int(dq[tuple(z)])
                       for z in np.argwhere(dq != 0)}
            if defects == want:
                seen.add(key)
                out.append((change, target))
        return out

    # ------------------------------------------------------------------ idle (oracle)

    def _idle_links(self, head):
        r"""The single links whose charge reach touches ``head`` --- idle-isotopy candidates."""
        # A link (mu, site) affects hypercube x iff (x - site) in reach[mu], so the
        # candidates are site = head - offset over the reach.
        N = self.Lattice.N
        reach = self._link_reach()
        links = set()
        for mu, offsets in reach.items():
            for off in offsets:
                site = tuple((head[k] - off[k]) % N for k in range(4))
                links.add((mu, site))
        return sorted(links)

    def clean_idle_reference(self, n_arr, q0, head):
        r"""
        The head-fixed single-link idles ($\Delta q \equiv 0$) at ``head``, both signs, by
        a global ``charge`` recompute.
        """
        L = self.Lattice
        out = []
        for mu, site in self._idle_links(head):
            for c in (+1, -1):
                trial = n_arr.copy()
                trial[(mu,) + site] += c
                dq = charge(Form(trial, degree=1, lattice=L)) - q0
                if np.abs(dq).max() == 0:
                    out.append({(mu,) + site: c})
        return out

    # ------------------------------------------------------------------ acceptance

    @staticmethod
    def _accept(lenC, lenCp, dS):
        r"""Metropolis--Hastings acceptance $\min(1, (\left|C\right|/\left|C'\right|)\,e^{-\Delta S})$."""
        return min(1.0, (lenC / lenCp) * np.exp(-dS))

    # ------------------------------------------------------------------ step (oracle)

    def step_reference(self, configuration):
        r"""
        Readable global-recompute worm walk --- the oracle :meth:`step` is validated
        against.  Open a worm, evolve the head (enumerate the clean set, draw uniformly,
        accept with $\left|C\right|/\left|C'\right|$), close when the head returns to the tail, and emit the
        valid configuration and its inline head$-$tail histogram.
        """
        L = self.Lattice
        N = L.N
        D = L.D
        n_moves = 2 * len(self._ortho)       # signed orthogonal movers
        menu = n_moves + 1                     # + one idle slot

        # --- Why this Metropolis--Hastings worm is exact (the load-bearing facts) ---
        # For a drawn direction we ENUMERATE the clean moves on the current background,
        # draw one uniformly, and correct the state-dependence of that enumeration with
        # the Hastings ratio |C|/|C'| in _accept.  Three facts make it an exact sampler;
        # an edit that quietly breaks any one biases the chain with no test failure:
        #
        # (1) The reverse move is ALWAYS in the reverse clean set.  Writing F' = F + dΔn,
        #     the reversal identity
        #         Δq(-Δn on F') = F'∧d(-Δn) + d(-Δn)∧F' + d(-Δn)∧d(-Δn) = -Δq(Δn on F)
        #     (the self-wedge dΔn∧dΔn is even under Δn -> -Δn, the two cross terms are odd)
        #     turns a clean forward dipole {head:-1, target:+1} into exactly the reverse
        #     dipole {target:-1, head:+1}.  So |C'| >= 1 always (never a zero division) and
        #     q(s'->s) = 1/(menu·|C'|) is exactly what |C|/|C'| corrects.  The idle case is
        #     the same with the self term absent: L_ℓ(F + c·dδ_ℓ) = L_ℓ(F), so the negated
        #     link is idle on the arrived state iff it was idle on the departed one.
        # (2) The uniform draw is over DISTINCT Δn (clean_set_* dedup by frozenset), so the
        #     per-transition proposal probability is exactly 1/|C|, with each element of C
        #     landing on a distinct s'.  A weighted draw, or removing the dedup, would need
        #     multiplicity/weight factors in _accept that are NOT |C|/|C'|.  Cross-slot: a
        #     transition is proposable through exactly one menu slot -- the displacement
        #     pins (dd, sign) for movers, and no change is a clean mover and a clean idle at
        #     once -- so 1/(menu·|C|) is the TOTAL forward probability of the transition.
        # (3) The pivot factor menu/(menu+1) is left uncorrected (inherited from the parent
        #     worm).  It is self-consistent: it is equivalent to pivot states (head==tail)
        #     carrying extra weight (menu+1)/menu, and detailed balance then holds
        #     transition-by-transition.  Idles never cross the pivot boundary, so the factor
        #     cancels for them identically; movers are the only pivot-crossing transitions,
        #     and the standard open/close counting applies to them.  The r=0 histogram bin
        #     stays unbiased because the close branch returns BEFORE the tally below, so
        #     pivots are tallied with probability menu/(menu+1) -- exactly cancelling their
        #     (menu+1)/menu weight excess.
        #
        # Termination: an enumeration worm could otherwise deadlock if every clean set were
        # empty mid-flight (close is unavailable off the pivot).  Fact (1) forbids it -- the
        # reverse of the last accepted move is always available -- so a positive-probability
        # path back to the pivot always exists and step_reference() halts almost surely.
        n = np.asarray(configuration['n']).astype(np.int64)
        dphi = np.asarray(_d(configuration['phi']))
        q_now = charge(configuration['n'])
        displacements = np.zeros(L.dims)

        tail = tuple(int(x) for x in self.rng.integers(0, N, size=D))
        head = tail

        while True:
            if head == tail and self.rng.uniform(0, 1) < 1.0 / (menu + 1):
                wl = displacements.sum()
                self.worm_lengths.append(wl)
                return configuration | {'n': Form(n, degree=1, lattice=L),
                                        'Intersection_Intersection': displacements,
                                        'Worm_Length': wl}

            pick = int(self.rng.integers(0, menu))
            if pick == n_moves:
                # Idle isotopy (head fixed, Δq ≡ 0): enumerate the clean idles on the
                # departed state (I) and, after tentatively applying, on the arrived state
                # (Ip); accept with |I|/|Ip|.  Both signs are enumerated so the family is
                # closed under inversion -- by fact (1) the reverse idle is in Ip, |Ip| >= 1.
                I = self.clean_idle_reference(n, q_now, head)
                if I:
                    change = I[int(self.rng.integers(0, len(I)))]
                    trial = n.copy()
                    for link, c in change.items():
                        trial[link] += c
                    q1 = charge(Form(trial, degree=1, lattice=L))
                    Ip = self.clean_idle_reference(trial, q1, head)
                    dS = self._delta_S(dphi, n, change)
                    if self.rng.uniform(0, 1) < self._accept(len(I), len(Ip), dS):
                        n = trial
                        q_now = q1
                        # head unchanged (idle)
            else:
                # Orthogonal head transport.  C = clean movers for (dd, sign) on the
                # departed state; the reverse set Cp is enumerated for the OPPOSITE
                # displacement (target, -sign) on the arrived state and, by fact (1),
                # contains the exact reverse of every accepted move, so |Cp| >= 1.  dS is
                # taken on the pre-move n and _delta_S is antisymmetric under reversal, so
                # w(s)/w(s') = e^{-dS} closes the balance with the |C|/|Cp| Hastings factor.
                dd = self._ortho[pick // 2]
                sign = +1 if pick % 2 == 0 else -1
                C = self.clean_set_reference(n, q_now, head, dd, sign)
                if C:
                    change, target = C[int(self.rng.integers(0, len(C)))]
                    trial = n.copy()
                    for link, c in change.items():
                        trial[link] += c
                    q1 = charge(Form(trial, degree=1, lattice=L))
                    Cp = self.clean_set_reference(trial, q1, target, dd, -sign)
                    dS = self._delta_S(dphi, n, change)
                    if self.rng.uniform(0, 1) < self._accept(len(C), len(Cp), dS):
                        n = trial
                        q_now = q1
                        head = target

            # Tally AFTER the close test (which already returned for closed worms).  This
            # ordering is what makes the pivot weighting of fact (3) cancel in the r=0 bin.
            disp = tuple((head[k] - tail[k]) % N for k in range(D))
            displacements[disp] += 1

    # ------------------------------------------------------------------ clean sets (fast)

    def clean_set_local(self, F, head, dd, sign):
        r"""
        Fast twin of :meth:`clean_set_reference`: same result, but $\Delta q$ comes from
        :meth:`_local_dq` on the maintained ``F`` $= dn$ instead of a global recompute.
        """
        N = self.Lattice.N
        target = tuple((head[k] + sign * dd[k]) % N for k in range(4))
        want = {head: -1, target: 1}
        anchor = tuple((head[k] + dd[k]) % N for k in range(4)) if sign > 0 else head
        seen = set()
        out = []
        for shape in self._mover_shapes(dd):
            change = self._change_from_shape(head, dd, sign, shape)
            key = frozenset((lnk, c) for lnk, c in change.items() if c != 0)
            if key in seen:
                continue
            if self._local_dq(F, change, anchor, shape) == want:
                seen.add(key)
                out.append((change, target))
        return out

    def clean_idle_local(self, F, head):
        r"""Fast twin of :meth:`clean_idle_reference` (local stencil, not global recompute)."""
        # A single link is idle iff its local charge response is empty; the response is
        # linear in the coefficient, so c=+1 deciding implies c=-1 too.
        N = self.Lattice.N
        out = []
        for mu, site in self._idle_links(head):
            if not local_charge.charge_change_from_link(F, mu, site, +1, N):
                for c in (+1, -1):
                    out.append({(mu,) + site: c})
        return out

    # ------------------------------------------------------------------ step (fast)

    def step(self, configuration):
        r"""
        One worm update: drop a coincident head/tail, evolve the head with the adaptive
        move until it returns to the tail, and emit the resulting valid configuration and
        the inline head$-$tail histogram (:class:`~.Intersection_Intersection`).  The fast
        path --- local stencils on an incrementally maintained $F = dn$;
        :meth:`step_reference` is the readable global-recompute version it is validated
        bit-for-bit against.
        """
        return self._run_worm_local(configuration, self.clean_set_local, self.clean_idle_local)

    def _run_worm_local(self, configuration, clean_movers, clean_idles):
        r"""
        The local-stencil worm walk, parameterized by the mover and idle clean-set
        functions ``clean_movers(F, head, dd, sign)`` and ``clean_idles(F, head)`` so a
        subclass can drive the identical walk with a compiled or a reference enumeration
        (:class:`TwoLinkAdaptiveWorm` uses this for its numba ``step`` and pure-Python
        ``step_reference``).
        """
        L = self.Lattice
        N = L.N
        D = L.D
        n_moves = 2 * len(self._ortho)
        menu = n_moves + 1

        # Exact for the same three facts documented in step_reference; the only difference
        # is that Δq comes from local stencils on the maintained F = dn rather than a global
        # recompute.  F is patched on every TENTATIVE change and reverted on rejection
        # (touch(change, -1)), and integer touch/revert is exact, so a rejected proposal
        # leaves n and F bit-identical to their pre-proposal values.  That is what lets the
        # reverse enumeration (Cp / Ip) see the true arrived background and lets this step
        # match step_reference move-for-move on a shared seed: fact (1)'s |Cp|,|Ip| >= 1
        # guarantee and fact (2)'s dedup are properties of the enumeration, not of how Δq is
        # computed, so switching to stencils cannot change which moves are clean.
        n = np.asarray(configuration['n']).astype(np.int64)
        dphi = np.asarray(_d(configuration['phi']))
        F = np.asarray(_d(configuration['n'])).astype(np.int64, copy=False)
        displacements = np.zeros(L.dims)

        tail = tuple(int(x) for x in self.rng.integers(0, N, size=D))
        head = tail

        def touch(change, s):
            # add s*change to both n and the maintained F (s = +1 apply, -1 revert)
            for link, c in change.items():
                n[link] += s * c
                local_charge.apply_link_to_F(F, link[0], link[1:], s * c, N)

        while True:
            if head == tail and self.rng.uniform(0, 1) < 1.0 / (menu + 1):
                wl = displacements.sum()
                self.worm_lengths.append(wl)
                return configuration | {'n': Form(n, degree=1, lattice=L),
                                        'Intersection_Intersection': displacements,
                                        'Worm_Length': wl}

            pick = int(self.rng.integers(0, menu))
            if pick == n_moves:
                # Idle isotopy; enumerate Ip on the tentatively-applied F, accept |I|/|Ip|.
                I = clean_idles(F, head)
                if I:
                    change = I[int(self.rng.integers(0, len(I)))]
                    dS = self._delta_S(dphi, n, change)          # pre-move n
                    touch(change, +1)
                    Ip = clean_idles(F, head)
                    if self.rng.uniform(0, 1) < self._accept(len(I), len(Ip), dS):
                        pass                                      # keep; head unchanged
                    else:
                        touch(change, -1)                         # revert n and F
            else:
                # Head transport; Cp is enumerated for (target, -sign) on the tentatively-
                # applied F, so it holds this move's reverse (fact (1)) -- accept |C|/|Cp|.
                dd = self._ortho[pick // 2]
                sign = +1 if pick % 2 == 0 else -1
                C = clean_movers(F, head, dd, sign)
                if C:
                    change, target = C[int(self.rng.integers(0, len(C)))]
                    dS = self._delta_S(dphi, n, change)          # pre-move n
                    touch(change, +1)
                    Cp = clean_movers(F, target, dd, -sign)
                    if self.rng.uniform(0, 1) < self._accept(len(C), len(Cp), dS):
                        head = target
                    else:
                        touch(change, -1)

            # Tally after the close test, exactly as in step_reference: the ordering makes
            # the fact (3) pivot weighting cancel in the r=0 bin.
            disp = tuple((head[k] - tail[k]) % N for k in range(D))
            displacements[disp] += 1


class TwoLinkAdaptiveWorm(AdaptiveIntersectionWorm):
    r"""
    Two-link enrichment of :class:`AdaptiveIntersectionWorm`.

    On a hot background even the fixed library rarely offers a clean move.  This worm
    additionally *live-enumerates*, for the drawn direction, every clean **two-link**
    coordinated move on the current $F = dn$ (and the head-fixed two-link idle isotopies),
    so the head can advance where no fixed template is clean.  It is a superset of the
    adaptive worm's moves and uses the same fixed-family-filtered-by-$F$ construction, so
    the Metropolis--Hastings exactness is unchanged.

    The family is large (thousands of shapes per direction), so :meth:`step` evaluates it
    in a compiled ``njit`` kernel; :meth:`step_reference` is the equivalent pure-Python
    walk it is validated bit-for-bit against.

    .. warning::

        Restricted to $D = 4$; orthogonal transport only.  Opt-in --- not part of the
        default :func:`~supervillain.generator.no_intersection.Hammer`.
    """

    # Coefficient box for two-link moves; |c| = 2 carries the mixed-magnitude Diophantine
    # solutions (pairs whose only clean move needs |c| = 2).
    COEFF_BOX = (-2, -1, 1, 2)

    def __init__(self, S, class_weights=None):
        super().__init__(S, class_weights=class_weights)
        self._two_movers = self._build_two_link_movers()
        self._two_idles = self._build_two_link_idles()
        # Flatten the families to integer arrays for the compiled clean-set kernel.
        self._dq_stencil = two_link_kernel.dq_stencil_arrays()
        self._mover_flat = {dd: two_link_kernel.flatten_family(self._mover_shapes(dd),
                                                               self._self_charge)
                            for dd in self._ortho}
        self._idle_flat = two_link_kernel.flatten_family(self._two_idles, self._self_charge)

    def __str__(self):
        return 'TwoLinkAdaptiveWorm'

    # ------------------------------------------------------------------ family construction

    def _build_two_link_movers(self):
        r"""
        For each orthogonal direction, the fixed family of two-link mover shapes: distinct
        link pairs (coefficients in :data:`COEFF_BOX`) whose reach can cover both dipole
        endpoints.  Self-charges are registered into ``self._self_charge`` for scoring.
        """
        # Shapes are stored relative to the anchor `target` (the +1 defect), with the -1
        # defect at -dd -- the library convention, so the inherited sign mechanism yields
        # the reverse move for free.  The cross term M12 is computed ONCE per link-pair and
        # scaled by c1*c2 (self-charge = c1*c2*M12 exactly); the coverage prune is
        # coefficient-independent (a nonzero c1*c2 never changes which cells are nonzero),
        # so it is decided once per pair.
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
                    support = set()
                    for mu, r in ((mu1, r1), (mu2, r2)):
                        for off in reach[mu]:
                            support.add(tuple((r[k] + off[k]) % N for k in range(4)))
                    unit_M = self._shape_self_charge(((mu1, r1, 1), (mu2, r2, 1)))
                    cover = support | {off for off, _v in unit_M}
                    if target_off not in cover or head_off not in cover:
                        continue
                    for c1, c2 in product(self.COEFF_BOX, repeat=2):
                        shape = ((mu1, r1, c1), (mu2, r2, c2))
                        self._self_charge[shape] = self._scaled_self_charge(unit_M, c1 * c2)
                        shapes.append(shape)
            movers[dd] = shapes
        return movers

    def _mover_shapes(self, dd):
        r"""The library family plus the two-link movers for ``dd`` (one combined slot)."""
        return self._library[dd] + self._two_movers[dd]

    # ------------------------------------------------------------------ compiled clean sets

    def clean_set_local(self, F, head, dd, sign):
        r"""
        Compiled twin of :meth:`_clean_set_local_py`: the whole family's $\Delta q$ is
        evaluated against ``F`` by :func:`.two_link_kernel.clean_mask`, and only the clean
        shapes become change dicts in Python (deduped, family order).  Bit-for-bit with the
        pure-Python version; used by :meth:`step`.
        """
        N = self.Lattice.N
        target = tuple((head[k] + sign * dd[k]) % N for k in range(4))
        anchor = tuple((head[k] + dd[k]) % N for k in range(4)) if sign > 0 else head
        factor = 1 if sign > 0 else -1
        Farr = np.asarray(F)
        F2 = np.ascontiguousarray(Farr.reshape(Farr.shape[0], -1))
        clean = two_link_kernel.clean_mask(
            F2, N, factor, np.array(anchor, dtype=np.int64),
            _ravel(head, N), _ravel(target, N), 1,
            *self._mover_flat[dd], *self._dq_stencil)
        shapes = self._mover_shapes(dd)
        seen, out = set(), []
        for si in range(len(shapes)):
            if not clean[si]:
                continue
            change = self._change_from_shape(head, dd, sign, shapes[si])
            key = frozenset((lnk, c) for lnk, c in change.items() if c != 0)
            if key in seen:
                continue
            seen.add(key)
            out.append((change, target))
        return out

    def _clean_set_local_py(self, F, head, dd, sign):
        r"""Pure-Python local clean set (the base enumeration over the enriched family) that
        the compiled :meth:`clean_set_local` is validated against and :meth:`step_reference`
        drives."""
        return AdaptiveIntersectionWorm.clean_set_local(self, F, head, dd, sign)

    # ------------------------------------------------------------------ two-link idles

    def _build_two_link_idles(self):
        r"""
        The fixed family of two-link idle shapes: distinct link pairs both reach-touching
        the head, coefficients in :data:`COEFF_BOX`, stored relative to the head.
        Self-charges are registered for scoring; cleanliness is decided per background.
        """
        # The sign-symmetric COEFF_BOX makes the coefficient-negated partner of every shape
        # a family member too, so the reverse isotopy is always enumerated (|I'| >= 1).
        head = (0, 0, 0, 0)
        slots = self._reach_touching_slots((head,))
        shapes = []
        for i in range(len(slots)):
            mu1, r1 = slots[i]
            for j in range(i + 1, len(slots)):
                mu2, r2 = slots[j]
                unit_M = self._shape_self_charge(((mu1, r1, 1), (mu2, r2, 1)))
                for c1, c2 in product(self.COEFF_BOX, repeat=2):
                    shape = ((mu1, r1, c1), (mu2, r2, c2))
                    self._self_charge[shape] = self._scaled_self_charge(unit_M, c1 * c2)
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
        Compiled twin of :meth:`_clean_idle_local_py`: base single-link idles, then the
        two-link idle family scored by :func:`.two_link_kernel.clean_mask`.  Deduped and
        ordered exactly as the pure-Python version; used by :meth:`step`.
        """
        N = self.Lattice.N
        out = AdaptiveIntersectionWorm.clean_idle_local(self, F, head)
        seen = {frozenset((lnk, c) for lnk, c in ch.items() if c != 0) for ch in out}
        Farr = np.asarray(F)
        F2 = np.ascontiguousarray(Farr.reshape(Farr.shape[0], -1))
        clean = two_link_kernel.clean_mask(
            F2, N, 1, np.array(head, dtype=np.int64), 0, 0, 0,
            *self._idle_flat, *self._dq_stencil)
        for si in range(len(self._two_idles)):
            if not clean[si]:
                continue
            shape = self._two_idles[si]
            change = {}
            for mu, rs, c in shape:
                link = (mu,) + tuple((head[k] + rs[k]) % N for k in range(4))
                change[link] = change.get(link, 0) + c
            key = frozenset((lnk, c) for lnk, c in change.items() if c != 0)
            if key in seen:
                continue
            seen.add(key)
            out.append(change)
        return out

    def _clean_idle_local_py(self, F, head):
        r"""Pure-Python local idle clean set (base single-link idles + the clean two-link
        idles) that :meth:`clean_idle_local` is validated against and :meth:`step_reference`
        drives."""
        # single- and two-link changes never collide (distinct link counts), so the base
        # order is preserved and the two-link tail shares its order with the compiled path.
        out = AdaptiveIntersectionWorm.clean_idle_local(self, F, head)
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

    # ------------------------------------------------------------------ step reference

    def step_reference(self, configuration):
        r"""
        The pure-Python worm walk that the compiled :meth:`step` is validated bit-for-bit
        against --- same update and acceptance, only the clean-set enumeration differs.
        """
        return self._run_worm_local(configuration, self._clean_set_local_py,
                                    self._clean_idle_local_py)
