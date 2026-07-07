#!/usr/bin/env python

import numpy as np

import supervillain.action
from supervillain.lattice import Form, d as _d
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge
from supervillain.generator.no_intersection.worm import IntersectionWorm


class AdaptiveIntersectionWorm(IntersectionWorm):
    r"""
    $F$-adaptive Metropolis--Hastings worm for the $q = dn\wedge dn = 0$ constraint in 4D.

    Where :class:`~.IntersectionWorm` draws one library shape per step and tests it --- so
    on a hot vortex-sheet background almost every draw is unclean and the head barely moves
    --- this worm *enumerates* the clean moves for the drawn direction on the current
    background $F = dn$, draws one uniformly, and corrects the state-dependence with the
    exact ratio $\min\!\big(1, (|C|/|C'|)\,e^{-\Delta S}\big)$.  Because $F$ is a function
    of the current state (not the chain's history), this is ordinary state-dependent MH:
    exact, no diminishing-adaptation.

    The candidate family is the parent worm's locality-bounded library: a link can serve
    the dipole $\{h, t\}$ only if both ends lie in its charge reach, so the family is
    fixed, state-independent, and closed under inversion.  Idle isotopies ($\Delta q\equiv
    0$, head fixed) are handled as a null-displacement "direction''.

    v1 covers orthogonal $\pm\hat e_\mu$ transport and single-link idles; diagonal buckets,
    multi-link idle isotopies, and the absolute-normalization / susceptibility validation
    are deferred (see
    ``docs/superpowers/specs/2026-07-06-adaptive-intersection-worm-design.md``).

    .. warning::

        Restricted to $D = 4$ and updates $n$ only, so combine it with a $\phi$-update.
    """

    def __init__(self, S, class_weights=None):
        super().__init__(S, class_weights=class_weights)
        # v1 menu: canonical orthogonal displacements only (diagonals deferred).
        self._ortho = [dd for dd in self._directions
                       if sum(abs(x) for x in dd) == 1]

    def __str__(self):
        return 'AdaptiveIntersectionWorm'

    def report(self):
        r"""Worm-length summary.  (The parent's per-family tallies are not populated by
        the adaptive proposal, which enumerates rather than draws-and-classifies.)"""
        l = np.array(self.worm_lengths)
        if len(l) == 0:
            return 'There were 0 adaptive worms.'
        return (f'There were {len(l)} adaptive worms.\nWorm lengths:\n'
                f'    mean {l.mean()}\n    std  {l.std()}\n    max  {int(max(l))}')

    # ------------------------------------------------------------------ clean sets (oracle)

    def _mover_shapes(self, dd):
        r"""
        The candidate mover shapes for canonical direction ``dd`` — the fixed,
        state-independent family that :meth:`clean_set_local` and
        :meth:`clean_set_reference` filter against the current background.  The base
        family is the parent worm's locality-bounded library; subclasses may enrich it
        (e.g. with live-enumerated two-link coordinated moves) by overriding this method.
        """
        return self._library[dd]

    def clean_set_reference(self, n_arr, q0, head, dd, sign):
        r"""
        Every distinct library shape in bucket ``dd`` whose global $\Delta q$ is the exact
        dipole $\{head:-1,\ target:+1\}$ on background ``n_arr`` (with ``q0`` its charge).
        Returns ``[(change, target), ...]`` with duplicate $\Delta n$ collapsed, so the
        length is $|C|$.  Global recompute --- the readable oracle.
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
        r"""
        The single links whose charge reach touches ``head`` --- candidates for a
        head-fixed isotopy.  A link $(\mu, \text{site})$ affects hypercube $x$ iff
        $(x - \text{site}) \in$ ``reach[mu]``, so ``site = head - offset`` over the reach.
        (v1 idle family is single-link; multi-link isotopies are a deferred enrichment.)
        """
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
        The head-fixed isotopies at ``head``: single-link changes ($c = \pm1$) whose
        global $\Delta q \equiv 0$.  Returns ``[change, ...]`` (each ``{link: coeff}``);
        both signs are enumerated, so the family is closed under inversion.
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
        r"""Metropolis--Hastings acceptance $\min(1, (|C|/|C'|)\,e^{-\Delta S})$."""
        return min(1.0, (lenC / lenCp) * np.exp(-dS))

    # ------------------------------------------------------------------ step (oracle)

    def step_reference(self, configuration):
        r"""
        Readable oracle: open a worm, walk the head with the adaptive orthogonal move
        (enumerate the clean set, draw uniformly, accept with $|C|/|C'|$), close when the
        head returns to the tail, and emit the valid configuration.  Cleanliness is a
        global ``charge`` recompute on a trial copy of $n$.

        v1 keeps :class:`~.IntersectionWorm`'s open/close structure; the close constant
        only shifts the overall amplitude, deferred with the susceptibility.
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
        Local-stencil twin of :meth:`clean_set_reference`: identical iteration order and
        return contract, but $\Delta q$ comes from :meth:`_local_dq` on the maintained
        field strength ``F`` $= dn$ instead of a global recompute.
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
        r"""
        Local-stencil twin of :meth:`clean_idle_reference`: a single link is idle iff its
        local charge response is empty (``charge_change_from_link`` returns ``{}``); the
        response is linear in the coefficient, so $c=+1$ deciding implies $c=-1$ too.
        """
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
        Accelerated :meth:`step_reference`: $\Delta q$ from local stencils on the
        maintained $F = dn$ (patched on every accepted *and* every tentative change, then
        reverted on rejection), never a global recompute.  Validated bit-for-bit against
        :meth:`step_reference` on a shared seed.
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
                I = self.clean_idle_local(F, head)
                if I:
                    change = I[int(self.rng.integers(0, len(I)))]
                    dS = self._delta_S(dphi, n, change)          # pre-move n
                    touch(change, +1)
                    Ip = self.clean_idle_local(F, head)
                    if self.rng.uniform(0, 1) < self._accept(len(I), len(Ip), dS):
                        pass                                      # keep; head unchanged
                    else:
                        touch(change, -1)                         # revert n and F
            else:
                # Head transport; Cp is enumerated for (target, -sign) on the tentatively-
                # applied F, so it holds this move's reverse (fact (1)) -- accept |C|/|Cp|.
                dd = self._ortho[pick // 2]
                sign = +1 if pick % 2 == 0 else -1
                C = self.clean_set_local(F, head, dd, sign)
                if C:
                    change, target = C[int(self.rng.integers(0, len(C)))]
                    dS = self._delta_S(dphi, n, change)          # pre-move n
                    touch(change, +1)
                    Cp = self.clean_set_local(F, target, dd, -sign)
                    if self.rng.uniform(0, 1) < self._accept(len(C), len(Cp), dS):
                        head = target
                    else:
                        touch(change, -1)

            # Tally after the close test, exactly as in step_reference: the ordering makes
            # the fact (3) pivot weighting cancel in the r=0 bin.
            disp = tuple((head[k] - tail[k]) % N for k in range(D))
            displacements[disp] += 1
