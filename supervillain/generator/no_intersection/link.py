#!/usr/bin/env python

from itertools import product

import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge

import logging
logger = logging.getLogger(__name__)


class ConstrainedLinkUpdate(ReadWriteable, Generator):
    r"""
    In the generic Villain model the :class:`~.villain.LinkUpdate` is a local fluctuation of $n$.
    This update scheme is exactly the same algorithm---except that it automatically rejects as
    invalid any update that would violate the no-intersection constraint.

    The minimal local move that changes $F$ is a **single-link** change $n_{\ell} \to n_{\ell} \pm 1$.
    In a $dn = F = 0$ region it creates no charge at all because the other field strengths in $dn\wedge dn$ vanish.
    But the constraint $F\wedge F = 0$ is quadratic, so on a background that already carries
    $F \ne 0$ the cross term $d\Delta n\wedge F + F\wedge d\Delta n$ can produce charge.
    Therefore, this proposal is **not** automatically legal: every proposal must be *verified* to keep $q = 0$ and rejected otherwise.

    Only $n_{\ell} \to n_{\ell} \pm 1$ is proposed.  Because the single-link charge change
    is exactly linear in the magnitude of the change, larger single-link changes offer no
    new constraint-preserving moves and no new connectivity; the proof is spelled out in
    the constructor's inline comments.

    In a phase with a lot of vortices, most of the proposals will be rejected and this generator will be very inefficient.

    .. note ::

        Restricted to $D = 4$.  Updates $n$ only; combine with a $\phi$-update such as
        :class:`~.villain.SiteUpdate`.

    .. warning ::

        This update is **not** ergodic by itself.  There exist valid :ref:`frozen
        configurations <no_intersection>` on which every single-link $\pm 1$ move lights up a
        defect, so the chain can neither escape nor reach one --- it is an isolated point of
        the single-link move graph.  Leaving a frozen configuration needs a *coordinated*
        move (and typically one that wraps the lattice).
    """

    def __init__(self, S):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('ConstrainedLinkUpdate requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('ConstrainedLinkUpdate is only implemented for D = 4.')

        self.Action = S
        self.Lattice = S.Lattice
        self.kappa = S.kappa
        self.rng = np.random.default_rng()

        # Only c = ±1 is proposed, because larger single-link magnitudes provably add
        # nothing.  Every plaquette of dδ_ℓ contains ℓ's own direction, so the wedge
        # dδ_ℓ ∧ dδ_ℓ has no complementary plane pair and vanishes identically; hence
        # for Δn = c δ_ℓ the charge change
        #
        #     Δq = c (F ∧ dδ_ℓ + dδ_ℓ ∧ F) ≡ c L_ℓ(F)
        #
        # is exactly linear in c.  Cleanliness (Δq = 0 ⇔ L_ℓ(F) = 0) is therefore
        # independent of c: no magnitude ever unlocks a blocked link.  Nor does |c| ≥ 2
        # add connectivity: the cross term in L_ℓ(F + dδ_ℓ) is again the vanishing
        # self-wedge, so L_ℓ(F + dδ_ℓ) = L_ℓ(F), the midpoint of a clean two-unit jump
        # is itself valid, and inductively any clean c-jump decomposes into |c| clean
        # unit steps: the reachability graph has the same connected components for any
        # magnitude range.  Metropolis additionally suppresses the redundant jumps by
        # e^{-κ 2π² c²}.  (Multi-link moves are different: mixed magnitudes can achieve
        # cancellations Σ_i c_i L_i = 0 unavailable at ±1; see ScattershotUpdate.)
        self.shifts = (+1, -1)

        self.accepted = 0
        self.proposed = 0
        self.sweeps = 0

    def __str__(self):
        return 'ConstrainedLinkUpdate'

    def step(self, cfg):
        r"""
        One sweep offering every link a single-link change $n_{\ell} \to n_{\ell} + c$ that
        (i) preserves $q = dn\wedge dn = 0$ everywhere and (ii) passes a Metropolis test
        against the Villain action.

        The links are partitioned into **colours** of mutually non-interacting links.  A
        single-link flip changes $q$ only in the immediate neighbourhood of the link, so two
        flips affect one another only when their links sit within one lattice step; links
        farther apart are independent.  Each colour is updated together, which --- because its
        links do not interact --- is identical to visiting them one at a time, so the sweep
        preserves the constraint and satisfies detailed balance colour by colour.  The field
        strength $F = dn$ is carried across the sweep and patched after each colour.

        :meth:`step_reference` is the plain, obviously-correct version this is validated
        against; :meth:`step_reference_broadcast` is a readable, un-compiled statement of the
        same checkerboard sweep.
        """
        L = self.Lattice
        N = L.N
        twopi = 2 * np.pi

        n = np.asarray(cfg['n']).astype(np.int64)
        dphi = np.asarray(d(cfg['phi']))
        F = np.asarray(d(cfg['n'])).astype(np.int64, copy=False)
        F2 = F.reshape(F.shape[0], -1)         # (n_planes, N**4); maintained across the sweep

        self.sweeps += 1
        accepted = 0

        for mu, base, block, coords in local_charge.numba_plan(N):
            stencil = local_charge._numba_stencil(mu)

            # The RNG draws and the float metropolis test stay in numpy -- same order, same
            # shapes, same exp as step_reference_broadcast -- so this reproduces that sweep
            # bit-for-bit; only the integer work below is compiled.
            c = 2 * self.rng.integers(0, 2, size=block) - 1
            A = dphi[mu][base] - twopi * n[mu][base]
            dS = (self.kappa / 2) * ((A - twopi * c) ** 2 - A ** 2)
            coin = self.rng.uniform(0, 1, size=block)
            metro = coin < np.exp(np.minimum(-dS, 0.0))
            self.proposed += c.size

            # The integer clean check and apply for the whole colour, in one compiled kernel
            # (no per-term Python/numpy dispatch, which was the sweep's entire cost).  It
            # visits the colour's links one at a time; because they are non-interacting this
            # equals updating them simultaneously.  c/metro are raveled in the colour's
            # C-order, matching coords.
            accepted += local_charge._color_kernel(
                F2, n[mu].reshape(-1), np.ascontiguousarray(metro.reshape(-1)),
                np.ascontiguousarray(c.reshape(-1)), coords, *stencil, N)

        self.accepted += accepted
        return cfg | {'n': Form(n, degree=1, lattice=L)}

    def step_reference_broadcast(self, cfg):
        r"""
        The checkerboard sweep of :meth:`step`, written with plain whole-array numpy
        operations instead of a compiled kernel --- the readable reference that :meth:`step`
        is validated against.
        """
        L = self.Lattice
        N = L.N
        twopi = 2 * np.pi

        n = np.asarray(cfg['n']).astype(int)
        dphi = np.asarray(d(cfg['phi']))
        F = np.asarray(d(cfg['n']))

        self.sweeps += 1
        accepted = 0
        axis = local_charge.axis_colors(N)

        # Same colours and draws as step, but each colour rebuilds its index grids with
        # np.ix_ on the fly (step precomputes them) and does the whole-colour clean check and
        # apply with numpy fancy-indexing (clean_mask_for_color / apply_color).  It samples
        # the same distribution as step_reference, but draws the RNG in a different
        # (per-colour) order, so it is not bit-for-bit with that global-recompute reference.
        for mu in range(4):
            for choice in product(range(len(axis)), repeat=4):
                idx = [axis[choice[a]] for a in range(4)]
                sub = np.ix_(*idx)

                clean = local_charge.clean_mask_for_color(F, mu, idx, N)
                block = clean.shape
                self.proposed += clean.size

                c = 2 * self.rng.integers(0, 2, size=block) - 1
                A = dphi[mu][sub] - twopi * n[mu][sub]
                dS = (self.kappa / 2) * ((A - twopi * c) ** 2 - A ** 2)
                coin = self.rng.uniform(0, 1, size=block)
                accept = clean & (coin < np.exp(np.minimum(-dS, 0.0)))

                flip = np.where(accept, c, 0)
                local_charge.apply_color(n, F, mu, idx, N, flip)
                accepted += int(accept.sum())

        self.accepted += accepted
        return cfg | {'n': Form(n, degree=1, lattice=L)}

    def step_reference(self, cfg):
        r"""
        Reference sweep: the plain, obviously-correct implementation.

        Identical to :meth:`step` except the constraint is verified by a **global**
        ``charge`` recompute on a trial copy of $n$ ($O(N^4)$ per link, $O(N^8)$ per
        sweep).  Kept as the correctness oracle the accelerated :meth:`step` is tested
        against, and as the readable statement of what the move is.
        """
        L = self.Lattice
        N = L.N
        D = L.D

        n = cfg['n'].copy()
        dphi = d(cfg['phi'])

        self.sweeps += 1
        accepted = 0

        # All links, visited in random order.  We go one at a time because the
        # constraint couples links: two simultaneous changes could interact in F∧F.
        links = [(mu,) + tuple(int(x) for x in site)
                 for mu in range(D)
                 for site in np.ndindex(*((N,) * D))]
        self.rng.shuffle(links)

        for link in links:
            self.proposed += 1
            c = self.shifts[self.rng.integers(0, len(self.shifts))]

            # Reject anything that would violate q = dn ∧ dn = 0.
            trial = n.copy()
            trial[link] += c
            if not np.allclose(charge(trial), 0):
                continue

            # Metropolis on the Villain action; only this link's term changes.
            #   ΔS = κ/2 [ (A − 2π c)² − A² ],  A = (dφ − 2π n)_ℓ.
            A = dphi[link] - 2 * np.pi * n[link]
            dS = (self.kappa / 2) * ((A - 2 * np.pi * c) ** 2 - A ** 2)
            if self.rng.uniform(0, 1) < min(1.0, np.exp(-dS)):
                n[link] += c
                accepted += 1

        self.accepted += accepted
        new_n = Form(n, degree=1, lattice=L)
        return cfg | {'n': new_n}

    def report(self):
        if self.proposed == 0:
            return 'ConstrainedLinkUpdate: no proposals.'
        return (f'ConstrainedLinkUpdate: {self.accepted} / {self.proposed} '
                f'single-link changes accepted '
                f'({self.accepted / self.proposed:.6f}).')
