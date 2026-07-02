#!/usr/bin/env python

from collections import deque
import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import dF_entries, local_dq, wedge_pairs

import logging
logger = logging.getLogger(__name__)


class StringWorm(ReadWriteable, Generator):
    r"""
    An **incremental** version of the coordinated single-direction moves: a
    Prokof'ev–Svistunov worm that builds strings of $n_\mu$-links one link at a time,
    with a Metropolis test per link, instead of proposing an entire torus-wrapping
    loop atomically like :class:`WrappingLoopUpdate`.

    Why.
    ----
    The atomic wrapping proposals change $O(N)$ links at once, so their acceptance
    dies like $e^{-c\kappa N}$ on generic backgrounds; and near frozen textures the
    legal coordinated moves have background-dependent shapes that no fixed catalog
    can enumerate.  Growing the string incrementally lets the path follow the local
    energy landscape and, crucially, lets the worm emit **contractible closed
    strings** — local, coordinated multi-link moves of arbitrary shape — in addition
    to the wrapped ones.

    Ensemble.
    ---------
    An episode fixes a link direction $\mu$ and a sign $s$, and lays head and tail
    on the same random site.  Each move steps the head by one unit in a uniformly
    random direction $\pm\hat e_\nu$ (any $\nu$, including $\mu$), depositing
    $\pm s$ on the $\mu$-link it crosses — exactly the classic-worm rule, so
    stepping back undoes the deposit and every move's exact inverse is proposed
    with the same probability $\tfrac18$.  Because only $n_\mu$ changes,
    $d\Delta n\wedge d\Delta n \equiv 0$ and the accumulated charge violations are
    the *endpoint* structure of the open string.  The extended (g-sector) states
    are weighted by $e^{-S}$ times the indicator that all $q\ne 0$ sites lie within
    Chebyshev distance ``radius`` of the head or the tail; moves that would leak
    charge elsewhere are rejected as null (a zero-weight target), which is
    detailed-balance safe.  Each move's $\Delta q$ is computed locally
    (:func:`~supervillain.generator.no_intersection.charge.local_dq`, $O(1)$) and
    the violations are tracked incrementally, so the constraint is never
    recomputed globally.

    When the head sits on the tail *and* no violations remain, the configuration is
    valid and a $(2D+1)$-th equally-likely option closes the worm and emits it —
    the same bookkeeping as :class:`IntersectionWorm`, and the same path-reversal
    argument gives detailed balance (episode endpoints are always closable states,
    so the diagonal factors match between a path and its reverse).

    The g-sector weight carries a **defect fugacity** $e^{-\lambda\sum_x |q_x|}$
    (``fugacity``) and a hard cap $\sum_x|q_x| \le$ ``ceiling``.  Without them the
    violations cost nothing, and the worm entropically stews in a charged blob
    around its endpoints instead of returning to the constraint surface.  Both
    only reweight or restrict the auxiliary g-sector, so the emitted z-sector
    distribution is untouched.

    .. warning::

        Restricted to $D = 4$.  Updates $n$ only; combine with a $\varphi$-update.

    .. warning::

        The Villain action gives the string a tension $\sim 2\pi^2\kappa$ per
        deposited link, which is what keeps the head confined near the tail.  The
        head's excursions are a branching walk that goes *critical* when
        $(2D-1)\,e^{-2\pi^2\kappa} = 1$, i.e. at $\kappa^* = \ln 7/(2\pi^2) \approx
        0.099$: below (and near) that coupling, episode lengths acquire heavy tails
        and the mean diverges — use the atomic :class:`WrappingLoopUpdate` in that
        regime instead.  Safely subcritical couplings ($\kappa \gtrsim 0.12$) give
        short episodes.

    .. note::

        Not yet part of the default :func:`~supervillain.generator.no_intersection.Hammer`;
        combine it in explicitly while it is being evaluated.
    """

    def __init__(self, S, radius=2, fugacity=0.5, ceiling=12, reach=None, cap=5000):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('StringWorm requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('StringWorm is only implemented for D = 4.')

        self.Action = S
        self.Lattice = S.Lattice
        self.kappa = S.kappa
        self.rng = np.random.default_rng()

        # The g-sector weight is e^{-S - fugacity * Σ|q|} restricted to states whose
        # violations lie within Chebyshev distance ``radius`` of the head or tail and
        # carry total magnitude Σ|q| <= ceiling.  Without the fugacity the violations
        # are free and the worm entropically stews in a charged endpoint blob instead
        # of returning to q = 0; without the cap the confined state space is huge.
        self.radius = radius
        self.fugacity = fugacity
        self.ceiling = ceiling

        # Optionally, the head may not stray farther than ``reach`` (Chebyshev) from
        # the tail --- a pure state-based restriction (zero weight outside), so
        # detailed balance is untouched.  ``None`` leaves the head free.
        self.reach = reach

        # Episodes that exceed ``cap`` iterations are aborted and the WHOLE episode
        # is discarded (the pre-episode configuration is returned).  This is
        # detailed-balance safe: the bijection between a path and its reverse
        # preserves the iteration count (per-stall probabilities depend only on the
        # state), so truncation removes forward/reverse pairs together and only
        # feeds the diagonal.  It bounds the cost of the rare episodes that wander
        # into flat (zero-action-cost) regions of the landscape, whose entropic
        # return times are otherwise heavy-tailed.
        self.cap = cap
        self.aborted = 0

        self._wedge_pairs = wedge_pairs(self.Lattice)

        self.worm_lengths = deque()
        self.emitted_changed = 0    # emissions whose n differs from the episode start
        self.emitted = 0

    def __str__(self):
        return 'StringWorm'

    def inline_observables(self, steps):
        return {}

    def _distance(self, a, b):
        """Chebyshev distance on the periodic lattice."""
        N = self.Lattice.N
        return max(min(abs(x - y), N - abs(x - y)) for x, y in zip(a, b))

    def _confined(self, defects, head, tail):
        """Are all violation sites within ``radius`` of the head or the tail?"""
        N = self.Lattice.N
        r = self.radius

        def near(site, anchor):
            for a, b in zip(site, anchor):
                delta = abs(a - b)
                if min(delta, N - delta) > r:
                    return False
            return True

        return all(near(site, head) or near(site, tail) for site in defects)

    def step(self, configuration):
        r"""
        Run one worm episode: insert head and tail on a random site, evolve the head
        (depositing $\pm s$ on the crossed $\mu$-links) until it closes on the tail
        with no charge violations, and emit the resulting valid configuration.
        """
        L = self.Lattice
        N = L.N
        D = L.D
        n_dirs = 2 * D

        n = configuration['n'].copy()
        changed = False
        dphi = d(configuration['phi'])
        F = np.asarray(d(n)).astype(int)

        tail = tuple(int(x) for x in self.rng.integers(0, N, size=D))
        head = tail
        mu = int(self.rng.integers(0, D))
        s = 1 if self.rng.integers(0, 2) == 0 else -1

        defects = {}
        length = 0

        while True:
            # When the head sits on the tail and the configuration is valid, offer
            # the (2D+1)-th move: close the worm and emit.  All 2D+1 options are
            # equally likely.
            if head == tail and not defects and self.rng.uniform(0, 1) < 1.0 / (n_dirs + 1):
                self.worm_lengths.append(length)
                self.emitted += 1
                self.emitted_changed += changed and not np.array_equal(
                    np.asarray(n), np.asarray(configuration['n']))
                return configuration | {'n': Form(n, degree=1, lattice=L)}

            length += 1
            if self.cap is not None and length > self.cap:
                self.aborted += 1
                return configuration | {'n': configuration['n']}

            # Step the head in a uniformly random direction, crossing one μ-link.
            nu = int(self.rng.integers(0, D))
            sigma = 1 if self.rng.integers(0, 2) == 0 else -1
            site = tuple((head[k] - (sigma < 0) * (k == nu)) % N for k in range(4))
            link = (mu,) + site
            c = sigma * s
            target = tuple((head[k] + sigma * (k == nu)) % N for k in range(4))

            # Where does the charge move?  Merge the local Δq into the violations
            # and require them to remain confined to the endpoints and bounded.
            dq = local_dq(L, F, {link: c}, pairs=self._wedge_pairs)
            merged = dict(defects)
            for x, v in dq.items():
                merged[x] = merged.get(x, 0) + v
                if merged[x] == 0:
                    del merged[x]
            violation = sum(abs(v) for v in merged.values())
            if (violation > self.ceiling
                    or (self.reach is not None and self._distance(target, tail) > self.reach)
                    or not self._confined(merged, target, tail)):
                continue

            # Metropolis on the Villain action of the single crossed link, plus the
            # defect fugacity on the change of the total violation.
            A = dphi[link] - 2 * np.pi * n[link]
            dS = (self.kappa / 2) * ((A - 2 * np.pi * c) ** 2 - A ** 2)
            dS += self.fugacity * (violation - sum(abs(v) for v in defects.values()))
            if self.rng.uniform(0, 1) < min(1.0, np.exp(-dS)):
                n[link] += c
                for (idx, x), v in dF_entries(L, {link: c}).items():
                    F[(idx,) + x] += v
                defects = merged
                head = target
                changed = True

    def report(self):
        l = np.array(self.worm_lengths)
        if len(l) == 0:
            return 'There were 0 string worms.'
        return (f'There were {len(l)} string worms ({self.aborted} aborted at the cap).\n'
                f'    lengths: mean {l.mean():.1f}, max {int(l.max())}\n'
                f'    emissions with a changed configuration: '
                f'{self.emitted_changed} / {self.emitted}')
