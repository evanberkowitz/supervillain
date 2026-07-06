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
        for shape in self._library[dd]:
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
                # idle: head-fixed isotopy, enumerate + |C|/|C'|
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
        for shape in self._library[dd]:
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

            disp = tuple((head[k] - tail[k]) % N for k in range(D))
            displacements[disp] += 1
