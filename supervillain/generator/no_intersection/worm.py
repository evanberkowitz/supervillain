#!/usr/bin/env python

from collections import deque
from itertools import permutations, product
import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.batch import Batch
from supervillain.lattice import Form, Lattice, d, wedge
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge

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

# A *two*-link move that shifts the head *orthogonally*, by a single face step +ê_μ (here
# +ê_2).  Unlike the 3-link :data:`_SEED` its two links do not share a corner; their charge
# dipole nonetheless lands one face away.  It reaches exactly the same four face neighbours
# as :data:`_SEED`, but with one fewer link (smaller $\Delta S$, higher acceptance), and its
# shapes share the orthogonal buckets with the 3-link shapes: a step just draws uniformly
# among all of them, so more shapes means a clean orthogonal step is more often available.
_ORTHO2_SEED_HEAD = (1, 1, 1, 1)
_ORTHO2_SEED = (
    (0, (1, 1, 1, 1), +1),
    (1, (2, 1, 1, 2), +1),
)

# A *four*-link move that shifts the head by the *same-sign* diagonal ê_μ + ê_ν (here
# +(ê_0 + ê_1)).  This is the sign-partner of the elbow's ê_μ - ê_ν, and it is *not* reachable
# by any two-link move: four links is the minimum (see the class docstring).  Its dipole must
# already point in a *canonical* (positive) direction, because axis permutations cannot flip
# the two +1's of a same-sign diagonal into -1's the way they can reorder the +1/-1 of the
# opposite-sign elbow.  Like the elbow it preserves the parity of Σ_k x_k, and together the
# two diagonal families reach all four ±(ê_μ ± ê_ν) neighbours in every 2-plane.
_SAMEDIAG_SEED_HEAD = (2, 2, 0, 1)
_SAMEDIAG_SEED = (
    (0, (1, 1, 1, 1), +1),
    (0, (1, 2, 1, 1), +1),
    (1, (2, 2, 1, 2), +1),
    (3, (2, 2, 1, 1), -1),
)


class IntersectionWorm(ReadWriteable, Generator):
    r"""
    Prokof'ev–Svistunov worm for the $q = dn\wedge dn = 0$ constraint in 4D.

    The head and tail live on hypercubes (4-cells; there is one per site in 4D).
    Moving the head by one hypercube extends the dragged sheet of $F = dn$ by a clean,
    coordinated change of $n$; when the head returns to the tail the constraint is restored
    everywhere and the configuration is emitted into the Markov chain.

    Several kinds of template extend the sheet, each leaving the charge changed only by a
    $\pm 1$ dipole that advances the head.  Counting each template and its reverse separately,
    the library carries

    - **8 three-link orthogonal** moves $\pm\hat e_{\mu}$ --- the intuition of crossing a
      3-cube to a face neighbour, where the two hypercubes share a 3-cell;
    - **8 two-link orthogonal** moves $\pm\hat e_{\mu}$ to those same face neighbours, one link
      cheaper --- the surprising part being that the two links do *not* share a corner;
    - **12 two-link diagonal** moves $\pm(\hat e_{\mu} - \hat e_{\nu})$ --- a corner-sharing
      "elbow" in one 2-plane, where the two hypercubes share only a 2-cell;
    - **12 four-link diagonal** moves $\pm(\hat e_{\mu} + \hat e_{\nu})$ --- the same-sign
      partner of the elbow, so the two diagonal families together reach all four corners
      $\pm(\hat e_{\mu} \pm \hat e_{\nu})$ of every 2-plane;
    - **background-activated one-link** moves in every bucket, with coefficients
      $c = \pm 1$ only: on a flux background the *linear* response
      $\Delta q = c\,(F \wedge d\delta_{\ell} + d\delta_{\ell} \wedge F)$ of a single link
      can itself be the unit dipole, transporting the head where no coordinated template
      is clean.  Restricting to $c = \pm 1$ loses nothing, provably --- the response is
      exactly linear in $c$ and a unit dipole demands $c \mid 1$ (the full proof is inline
      in :meth:`_build_library`).  When the response instead vanishes identically
      ($\Delta q \equiv 0$, e.g.\ wherever the background is locally flat), the drawn
      1-link shape becomes an **idle** move: the $n$-change is Metropolis-tested and, if
      accepted, applied *without moving the head* --- a rearrangement of the sheet at
      fixed defect positions, the isotopy that head transport alone cannot supply.  (See
      :meth:`_sheet_segment` for why idle acceptance is symmetric for 1-link shapes and
      *only* for them.)

    To propose a step the worm picks one of these signed neighbour directions uniformly and
    then, uniformly, one of the templates that realises it.  Drawing the direction
    first is chosen so the number of proposal slots is the number of *neighbours*, not of
    templates: enriching a bucket with extra shapes --- the 2-link orthogonal rides in the
    same $\pm\hat e_{\mu}$ bucket as the 3-link one --- then costs nothing in the rate at which
    the worm closes, whereas a flat template draw would let every added shape lengthen the
    worm.

    Two links is the *minimum* whose *self*-charge moves anything: a single-link change has
    $d\Delta n \wedge d\Delta n \equiv 0$, so on the flux-free cold background it shifts
    nothing.  On a background with flux $F = dn$, though, the *linear* term
    $\Delta q = F \wedge d\Delta n + d\Delta n \wedge F$ is generically nonzero, and a single
    link can cleanly transport the head --- the one-link shapes above exist precisely to
    exploit that background-dependent effect (see also the discussion of frozen
    configurations in :ref:`the No-Intersection model <no_intersection>`; on frozen
    backgrounds even these have no clean first step).  The same-sign diagonal, by contrast,
    is the one short neighbour that *cannot* be built from two links --- four is its
    minimum --- which is why it carries a heavier shape than its opposite-sign partner.

    The families interleave freely: a diagonal step preserves the parity of $\sum_{k} x_{k}$
    while an orthogonal step flips it, so together they mix the head's walk more efficiently
    than either alone.  On the cold background the orthogonal $\pm\hat e_{\mu}$ steps alone
    already connect every hypercube, so the diagonals and heavier shapes are not needed for
    *reachability* there.  On a nontrivial background they earn their keep differently: a
    coordinated move deposits its net dipole directly, never placing a defect on the
    intermediate hypercube a stepwise decomposition would have to pass a clean hop through, so
    it can be clean exactly where that chain of smaller hops is blocked --- the same
    coordinated-move logic that escapes a frozen configuration.

    **The worm walks the Freedman--Quinn corridor.**  Poincaré-dually, a valid configuration
    is an *embedded* vortex sheet (no transverse self-intersections: $q \equiv 0$) and a $G$
    configuration is an *immersed* sheet carrying one $+/-$ pair of double points --- the
    head and the tail.  A classical fact of 4-manifold topology (Whitney's disk construction
    :cite:`Whitney1944`, Casson's finger moves :cite:`Casson`, and general position; stated
    systematically by Freedman and Quinn :cite:`FreedmanQuinn`, whose chapter 1 is the
    standard reference and lends the corridor its name here) connects any two homotopic
    embedded surfaces in a 4-manifold by exactly three elementary processes --- ambient
    isotopies, finger moves (double-point pair creation), and Whitney moves (pair
    annihilation) --- and the worm's three aspects implement them one-to-one:

    - **opening and the first step $=$ finger move.**  Dropping head $=$ tail on one
      hypercube ($\Delta S = 0$, automatically accepted) is a double-point pair at zero
      separation; the first accepted head move separates the pair --- a patch of sheet
      piercing another, creating the $+1$ and $-1$ transversally.
    - **transport and closing $=$ Whitney move.**  Each subsequent head move drags the $+1$
      double point, extending the dragged sheet of $F = dn$ (the finger); when the head
      rejoins the tail and the worm closes, the pair annihilates --- the Whitney move, with
      the dragged sheet playing the role of the Whitney disk's neighbourhood.
    - **idle moves $=$ ambient isotopy.**  Accepted 1-link draws with $\Delta q \equiv 0$
      rearrange the sheet while both double points stay put.  Without this leg the corridor
      would be incomplete: head transport alone can never move the sheet out of its own way
      at fixed defects.

    The known gaps between the theorem and this algorithm: Freedman--Quinn homotopies may
    require several pairs in flight simultaneously (Casson's obstruction :cite:`Casson` ---
    Whitney disks can themselves intersect things, and repairing that creates more pairs)
    while this worm carries exactly one, and the lattice $q$ is a cup-product density on a
    possibly non-manifold sheet --- so the smooth theorem is the structural reason for
    optimism, not a proof of lattice ergodicity.

    As the head moves we tally the head$-$tail displacement histogram that yields the
    :class:`~.Intersection_Intersection` correlator $\langle e^{i\theta_h} e^{-i\theta_t}\rangle$ ---
    the two-point function of the operator $e^{i\theta}$ that inserts a unit of
    vortex-sheet self-intersection $q = dn\wedge dn$.

    .. warning::

        Restricted to $D = 4$.  This generator updates $n$ only, so it is not ergodic
        on its own; at least combine it with a $\phi$-update such as
        :class:`~.villain.SiteUpdate`.

    .. danger::

        Whether this worm is an ergodic update to $n$, even combined with the
        :class:`~supervillain.generator.villain.ExactUpdate`, is not proven.  The
        Freedman--Quinn corridor above is the structural reason to expect mixing even
        across 2-knot classes --- knotted and unknotted sheets in the same homotopy class
        are connected through the immersed configurations the worm samples --- but the
        Casson multi-pair caveat and the lattice cup-product caveat keep this an
        empirical question, not a theorem.
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
        # unit directions ê_μ and the twelve diagonals ê_μ ± ê_ν with a positive first
        # component), the clean shapes that shift the +1 head by +d, expressed RELATIVE to
        # the head.  The opposite displacement -d is generated on the fly by negating a
        # shape, so we store only canonical d.
        self._library = self._build_library()
        self._directions = sorted(self._library)

        # family[d][k] names the move family of shape k in bucket d, so per-draw
        # bookkeeping is a lookup, not a re-classification.
        self._family = {
            dd: tuple(self._classify(dd, shape) for shape in self._library[dd])
            for dd in self._directions
        }
        self.tallies = {
            family: {outcome: 0 for outcome in
                     ('drawn', 'unclean', 'clean', 'idle', 'accepted', 'accepted_idle')}
            for family in ('ortho3', 'ortho2', 'elbow2', 'same4', '1link')
        }
        self._last_family = None
        self._self_charge = self._self_charges()

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
        # ``steps`` is the taxicab length of the displacement (number of unit hops): 1 for
        # the orthogonal seeds (±ê_μ), 2 for the diagonal seeds (ê_μ ± ê_ν).  It filters an
        # orbit down to just the templates whose self-charge dipole is the intended
        # neighbour; the orthogonal shapes (3-link and 2-link) share the ±ê_μ buckets.
        for seed, seed_head, steps in (
            (_SEED, _SEED_HEAD, 1),
            (_ORTHO2_SEED, _ORTHO2_SEED_HEAD, 1),
            (_DIAG_SEED, _DIAG_SEED_HEAD, 2),
            (_SAMEDIAG_SEED, _SAMEDIAG_SEED_HEAD, 2),
        ):
            for template in orbit(seed, seed_head):
                sep = separation(template, anchor)
                if sep is None or sum(abs(x) for x in sep) != steps:
                    continue
                if next((x for x in sep if x != 0), 0) <= 0:
                    continue  # keep only canonical directions; -d is made by negation
                library.setdefault(sep, []).append(template)

        # -------------------------------------------------- background-activated 1-link moves
        #
        # Single-link shapes carry coefficients c = ±1 ONLY, and that is provably complete —
        # no magnitude ladder is missing:
        #
        # (1) Linearity.  For Δn = c δ_ℓ, every plaquette of dδ_ℓ contains the link's
        #     direction, and the cup product pairs only complementary planes — which share
        #     no direction — so the self term vanishes identically,
        #
        #         dΔn ∧ dΔn ≡ 0,
        #
        #     and the charge response is EXACTLY linear in c:
        #
        #         Δq = F ∧ dΔn + dΔn ∧ F = c · L_ℓ(F),      F = dn the current background.
        #
        #     (This is also why a single link can never move the head on the cold
        #     background: F = 0 forces Δq = 0 for every c.)
        #
        # (2) Divisibility ⇒ c = ±1.  A head move requires Δq to be the unit dipole
        #     {target: +1, head: -1}.  Every entry of c·L_ℓ(F) is divisible by c, and the
        #     dipole's entries are ±1, so c | 1: only c = ±1 can ever transport this worm's
        #     unit-charge head.  A |c| ≥ 2 single link could only move defects of charge
        #     divisible by c — nothing here.  (Contrast ≥ 2-link templates: the bilinear
        #     cross term c_i c_j (dδ_i ∧ dδ_j + dδ_j ∧ dδ_i) survives, coefficient choices
        #     become a genuine Diophantine question, and mixed magnitudes can be the only
        #     clean solution — e.g. L_1 = L_2 = 1, M_12 = -1 forces (c_1-1)(c_2-1) = 1,
        #     i.e. (2, 2).  That enrichment is a separate, future extension.)
        #
        # (3) Reversibility (why plain Metropolis still suffices for these background-
        #     dependent shapes).  The linearity in (1) also gives L_ℓ(F + c dδ_ℓ) = L_ℓ(F)
        #     — the cross term is the vanishing self-wedge — so the negated shape applied
        #     at the arrived configuration has Δq exactly negated: the reverse move is
        #     clean precisely when the forward one was.  Since the reverse is the same
        #     shape with the opposite sign, drawn from the same bucket with the same
        #     (1/2M)·(1/K) probability, the proposal stays symmetric and the closing
        #     balance in step() — which never sees the per-bucket shape count K — is
        #     untouched.
        #
        # (4) Registration.  L_ℓ(F)_x can be nonzero only for x in a fixed neighbourhood
        #     S(ℓ) of the link, probed in _link_reach from the code's own wedge (so no
        #     cup-shift convention is hardcoded here).  A link can serve bucket d only if
        #     both dipole ends sit in its reach: target = link + u and head = link + v
        #     with u, v ∈ S(ℓ) and u - v = d.  Anchored at the target (the convention of
        #     _change_from_shape), the shape's relative site is r = -u.  Which registered
        #     links are ACTUALLY clean is background-dependent and is decided per proposal
        #     in _sheet_segment exactly as for every other shape; a registered-but-dirty
        #     draw is an ordinary stay-put rejection.  We register only into the existing
        #     canonical buckets (ê_μ and ê_μ ± ê_ν): reach-pair differences also hit
        #     taxicab-3 neighbours, but new buckets would grow 2M and lengthen every worm
        #     on the cold background — where 1-link shapes can never fire — so those wait
        #     until coverage demands them.
        reach = self._link_reach()
        for sep in library:
            for mu, S in reach.items():
                for u in S:
                    v = tuple(u[k] - sep[k] for k in range(4))
                    if v not in S:
                        continue
                    r = tuple(-x for x in u)
                    for c in (+1, -1):
                        shape = ((mu, r, c),)
                        if shape not in library[sep]:
                            library[sep].append(shape)
        return library

    def _link_reach(self):
        r"""
        For each link direction $\mu$, the set of hypercube offsets, relative to the
        link's site, where the background-linear charge response
        $L_{\ell}(F) = F \wedge d\delta_{\ell} + d\delta_{\ell} \wedge F$ of a
        single-link change can be nonzero.

        Probed on a fixed small scratch lattice with a *generic* 2-form background ---
        a distinct power of 4 on every nearby plaquette, in exact integer arithmetic.
        Entries of $L_{\ell}$ have magnitude at most $2 < 4$, so contributions from
        distinct plaquettes occupy distinct base-4 digits and can never cancel: the
        computed support is exact, and the cup product's shift conventions are inherited
        from :func:`~supervillain.lattice.wedge` itself rather than duplicated here.
        """
        L0 = Lattice(4, 8)
        anchor = (4, 4, 4, 4)
        probe = L0.zeros(2, dtype=object)
        weight = 1
        for idx in range(len(L0.components[2])):
            for offset in product(range(-2, 3), repeat=4):
                site = tuple((anchor[k] + offset[k]) % 8 for k in range(4))
                probe[(idx,) + site] = weight
                weight *= 4
        reach = {}
        for mu in range(4):
            delta = L0.zeros(1, dtype=int)
            delta[(mu,) + anchor] = 1
            e = d(delta)
            response = np.asarray(wedge(probe, e)) + np.asarray(wedge(e, probe))
            reach[mu] = frozenset(
                tuple(int(h[k + 1]) - anchor[k] for k in range(4))
                for h in np.argwhere(response != 0)
            )
        return reach

    @staticmethod
    def _classify(d, shape):
        r"""
        The move family of ``shape`` in bucket ``d``: ``'1link'`` for the
        background-activated single links, ``'ortho2'``/``'ortho3'`` for the
        $\pm\hat e_{\mu}$ shapes, ``'elbow2'`` for the opposite-sign diagonal
        $\hat e_{\mu} - \hat e_{\nu}$, and ``'same4'`` for the same-sign diagonal
        $\hat e_{\mu} + \hat e_{\nu}$.
        """
        if len(shape) == 1:
            return '1link'
        if sum(abs(x) for x in d) == 1:
            return 'ortho2' if len(shape) == 2 else 'ortho3'
        return 'elbow2' if sum(d) == 0 else 'same4'

    def _self_charges(self):
        r"""
        For every shape in the library, the background-independent **self-charge**
        $d\Delta n \wedge d\Delta n$ of the placed template, as a tuple of
        ``(offset, value)`` pairs with the offset measured from the template's anchor
        (the placement origin of :meth:`_change_from_shape`).

        Together with the per-link background-linear stencils this reconstructs the
        full charge change of any template:
        $\Delta q = F\wedge d\Delta n + d\Delta n\wedge F + d\Delta n\wedge d\Delta n$.

        Derived on a scratch lattice of the *same extent* as the target lattice, so any
        wrap-around cross terms between the template and its periodic images at small
        $N$ are captured exactly.  Quadratic in $\Delta n$, hence identical for the
        negated (backward) placement --- one pattern serves both signs.
        """
        N = self.Lattice.N
        L0 = Lattice(4, N)
        anchor = (N // 2,) * 4
        patterns = {}
        for shapes in self._library.values():
            for shape in shapes:
                if shape in patterns:
                    continue
                if len(shape) == 1:
                    # The single-link self-wedge vanishes identically.
                    patterns[shape] = ()
                    continue
                dn = L0.zeros(1, dtype=int)
                for mu, rs, c in shape:
                    site = tuple((anchor[k] + rs[k]) % N for k in range(4))
                    dn[(mu,) + site] += c
                q = np.asarray(charge(dn))
                patterns[shape] = tuple(
                    (tuple((int(h[1 + k]) - anchor[k]) % N for k in range(4)),
                     int(q[tuple(h)]))
                    for h in np.argwhere(q != 0)
                )
        return patterns

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
        gives a clean dipole shift on the current ``n``; ``(change, head)`` if the shape
        is a single link whose $\Delta q$ vanishes identically (an **idle** move: the
        sheet changes, the head does not); else ``(None, None)``.

        Selecting a single, uniformly-chosen shape makes the proposal **symmetric**:
        the reverse of a *head-moving* step is the same shape with the opposite sign,
        drawn with the same probability $\tfrac{1}{2M}\cdot\tfrac{1}{K}$ ($M$ canonical
        displacements, $K$ shapes for this one), and it is guaranteed clean on the
        proposed state.  The reverse of an *idle* step is instead the coefficient-negated
        shape from the **same** bucket at the **same** sign --- it anchors at the same
        absolute links and exactly undoes $\Delta n$ --- and it exists with the same
        probability because 1-link shapes are registered with both $c = \pm 1$.  Detailed
        balance then holds case by case with the plain Metropolis acceptance
        $\min(1, e^{-\Delta S})$.  (Trying several shapes and taking the first clean one
        would make $q$ asymmetric and break all of this.)

        Note that "idle" is an *outcome*, not a menu item: $M$ counts neighbour
        displacements only, with no don't-move category, and a given shape is a mover or
        an idle deterministically per background, never both at once.  The close option's
        only effect on the balance is the asymmetric prefactor $2M/(2M+1)$ between pivot
        (head $=$ tail) and non-pivot proposals --- and an idle step never changes whether
        head $=$ tail, so its forward and reverse proposals always share the same
        prefactor (both compete with the close option, or neither does) and the factor
        cancels identically.  Head-moving steps are the only transitions that cross the
        pivot boundary, and for those the standard counting applies unchanged.

        Idle acceptance is restricted to 1-link shapes: a *multi-link* template can also
        produce $\Delta q \equiv 0$ (its background-linear response cancelling its
        self-charge), but its coefficient-negated partner is **not** registered in the
        bucket --- negations only arise through the sign draw, which anchors elsewhere ---
        so idle-accepting it would break proposal symmetry.  Those draws remain ordinary
        rejections.

        An accepted idle move is a rearrangement of the sheet at fixed defect positions
        --- the *isotopy* leg of the isotopy + finger + Whitney corridor, which head
        moves alone cannot supply: the open worm can move the sheet out of its own way
        while the dipole is in flight, at zero marginal cost (the classifying charge
        recompute is already paid).
        """
        N = self.Lattice.N
        shapes = self._library[d]
        k = int(self.rng.integers(0, len(shapes)))
        shape = shapes[k]
        family = self._family[d][k]
        self._last_family = family
        self.tallies[family]['drawn'] += 1

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
            self.tallies[family]['clean'] += 1
            return change, target
        if not defects and len(shape) == 1:
            self.tallies[family]['idle'] += 1
            return change, head
        self.tallies[family]['unclean'] += 1
        return None, None

    def _local_dq(self, F, change, anchor, shape):
        r"""
        The change $\Delta q$ in the charge density from applying ``change``, computed
        locally: the background-linear per-link responses read off the current field
        strength ``F`` $= dn$, plus the precomputed self-charge of ``shape`` placed at
        ``anchor``.  Returns ``{hypercube: value}`` with zeros pruned --- the same
        dictionary a global ``charge(n + \Delta n) - charge(n)`` recompute yields, at
        $O(1)$ cost.
        """
        N = self.Lattice.N
        dq = {}
        for link, c in change.items():
            local = local_charge.charge_change_from_link(F, link[0], link[1:], c, N)
            for cell, v in local.items():
                dq[cell] = dq.get(cell, 0) + v
        for off, v in self._self_charge[shape]:
            cell = tuple((anchor[k] + off[k]) % N for k in range(4))
            dq[cell] = dq.get(cell, 0) + v
        return {cell: v for cell, v in dq.items() if v}

    def _sheet_segment_local(self, F, head, d, sign):
        r"""
        The accelerated :meth:`_sheet_segment`: identical draw, classification, and
        return contract, but $\Delta q$ comes from :meth:`_local_dq` on the maintained
        field strength ``F`` $= dn$ instead of a global recompute on a trial copy of
        $n$.  The proposal-symmetry and detailed-balance discussion lives on
        :meth:`_sheet_segment` and applies verbatim.
        """
        N = self.Lattice.N
        shapes = self._library[d]
        k = int(self.rng.integers(0, len(shapes)))
        shape = shapes[k]
        family = self._family[d][k]
        self._last_family = family
        self.tallies[family]['drawn'] += 1

        target = tuple((head[j] + sign * d[j]) % N for j in range(4))
        want = {} if target == head else {target: 1, head: -1}

        change = self._change_from_shape(head, d, sign, shape)
        # The anchor mirrors _change_from_shape: forward templates place their head at
        # head + d, backward (negated) templates anchor at head itself.
        anchor = tuple((head[j] + d[j]) % N for j in range(4)) if sign > 0 else head
        defects = self._local_dq(F, change, anchor, shape)
        if defects == want:
            self.tallies[family]['clean'] += 1
            return change, target
        if not defects and len(shape) == 1:
            self.tallies[family]['idle'] += 1
            return change, head
        self.tallies[family]['unclean'] += 1
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

        Cleanliness of each proposed template is decided **locally**: the charge
        change is the background-linear per-link stencil response read off the
        maintained field strength $F = dn$ plus the template's precomputed
        self-charge (:meth:`_local_dq`), so no proposal ever recomputes $q$ globally.
        :meth:`step_reference` is the global-recompute oracle this is validated
        against, bit-for-bit on a shared seed.
        """
        L = self.Lattice
        N = L.N
        D = L.D
        # Every canonical displacement d contributes two head moves (+d and -d); together
        # with the "close" option this gives 2M+1 equally likely choices when head==tail, so
        # the worm closes with probability 1/(2M+1).  That count is the number of oriented
        # NEIGHBOUR moves, 2M --- NOT the number of templates: the per-bucket shape count
        # cancels out of the g<->z (open/close) balance.  A specific move (direction d, a
        # sign, and a shape S drawn in _sheet_segment) is proposed at the pivot (head==tail)
        # with probability
        #     [2M/(2M+1)]·[1/(2M)]·[1/K]  =  1/[(2M+1)·K]        (K = shapes in d's bucket),
        # while its reverse, offered from the neighbour it lands on --- a non-pivot state with
        # no close option --- is proposed with
        #     [1/(2M)]·[1/K]              =  1/[2M·K].
        # The 1/K cancels in the forward/reverse ratio, leaving 2M/(2M+1); the closing balance
        # only ever sees the neighbour count.  So enriching a bucket with extra shapes (e.g.
        # the 2-link orthogonal sharing the ±ê_μ bucket) leaves the worm's closing rate --- and
        # detailed balance --- untouched.  (We don't re-derive that 1/(2M+1) is the value that
        # makes the plain head−tail histogram unbiased at the origin; it is the standard
        # Prokof'ev–Svistunov prescription, shared with the worldline and villain ClassicWorms.)
        n_moves = 2 * len(self._directions)

        n = np.asarray(configuration['n']).astype(np.int64)
        dphi = np.asarray(d(configuration['phi']))
        # F = dn is maintained incrementally across the whole worm (patched on every
        # accepted change), so each proposal costs a handful of stencil reads instead
        # of a global recompute; q itself is never needed, only its local change.
        F = np.asarray(d(configuration['n'])).astype(np.int64, copy=False)

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
                return configuration | {'n': new_n,
                                        'Intersection_Intersection': displacements,
                                        'Worm_Length': wl}

            # Otherwise propose a uniformly random one of the 2M head moves: a canonical
            # displacement (orthogonal or diagonal) and a sign for its orientation.
            hop = self._directions[self.rng.integers(0, len(self._directions))]
            sign = 1 if self.rng.integers(0, 2) == 0 else -1

            change, target = self._sheet_segment_local(F, head, hop, sign)
            if change is not None:
                # Metropolis-test the change in the Villain action.  For a head-moving
                # step target is the neighbour; for an accepted idle step (a 1-link shape
                # with Δq ≡ 0 -- see _sheet_segment) target == head and the sheet changes
                # under a stationary head.
                dS = self._delta_S(dphi, n, change)
                if self.rng.uniform(0, 1) < min(1.0, np.exp(-dS)):
                    for link, c in change.items():
                        n[link] += c
                        local_charge.apply_link_to_F(F, link[0], link[1:], c, N)
                    self.tallies[self._last_family]['accepted_idle' if target == head else 'accepted'] += 1
                    head = target
            # The library does not always offer a clean step on every trail, so the drawn
            # shape may be unclean: its Δn would put charge outside the valid G-space (a
            # dipole in the wrong place, a quadrupole, or a Δq ≡ 0 multi-link draw, whose
            # idle acceptance would break proposal symmetry) instead of shifting the
            # head's +1/-1 dipole.  That is not a special "malformed, never-happened"
            # event -- it is a proposal into a zero-probability region, i.e. an ordinary
            # Metropolis rejection with acceptance min(1, 0) = 0.  So, exactly like a
            # clean-but-rejected shape, the head stays put and we fall through to the
            # tally below.

            # Tally the head−tail displacement for the Intersection_Intersection correlator.
            # We tally on EVERY step, including these stay-puts.  A rejection is a genuine
            # self-loop of the chain, and self-loops leave detailed balance between distinct
            # states untouched (only clean, symmetric draws move between distinct states), so
            # the histogram still samples the stationary marginal ∝ G(r).  The estimator is a
            # time-average whose numerator and denominator share one clock; dropping stay-puts
            # would reweight G(r) by the configuration- and position-dependent fraction of
            # clean proposals and bias the correlator.
            disp = tuple((head[k] - tail[k]) % N for k in range(D))
            displacements[disp] += 1

    def step_reference(self, configuration):
        r"""
        Reference worm: the plain, obviously-correct implementation.  Identical to
        :meth:`step` except each proposal's cleanliness is verified by a **global**
        ``charge`` recompute on a trial copy of $n$ (inside :meth:`_sheet_segment`).
        Kept as the correctness oracle :meth:`step` is tested against, bit-for-bit on
        a shared seed, and as the readable statement of the algorithm.
        """
        L = self.Lattice
        N = L.N
        D = L.D
        # Every canonical displacement d contributes two head moves (+d and -d); together
        # with the "close" option this gives 2M+1 equally likely choices when head==tail, so
        # the worm closes with probability 1/(2M+1).  That count is the number of oriented
        # NEIGHBOUR moves, 2M --- NOT the number of templates: the per-bucket shape count
        # cancels out of the g<->z (open/close) balance.  A specific move (direction d, a
        # sign, and a shape S drawn in _sheet_segment) is proposed at the pivot (head==tail)
        # with probability
        #     [2M/(2M+1)]·[1/(2M)]·[1/K]  =  1/[(2M+1)·K]        (K = shapes in d's bucket),
        # while its reverse, offered from the neighbour it lands on --- a non-pivot state with
        # no close option --- is proposed with
        #     [1/(2M)]·[1/K]              =  1/[2M·K].
        # The 1/K cancels in the forward/reverse ratio, leaving 2M/(2M+1); the closing balance
        # only ever sees the neighbour count.  So enriching a bucket with extra shapes (e.g.
        # the 2-link orthogonal sharing the ±ê_μ bucket) leaves the worm's closing rate --- and
        # detailed balance --- untouched.  (We don't re-derive that 1/(2M+1) is the value that
        # makes the plain head−tail histogram unbiased at the origin; it is the standard
        # Prokof'ev–Svistunov prescription, shared with the worldline and villain ClassicWorms.)
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
                # Metropolis-test the change in the Villain action.  For a head-moving
                # step target is the neighbour; for an accepted idle step (a 1-link shape
                # with Δq ≡ 0 -- see _sheet_segment) target == head and the sheet changes
                # under a stationary head.
                dS = self._delta_S(dphi, n, change)
                if self.rng.uniform(0, 1) < min(1.0, np.exp(-dS)):
                    for link, c in change.items():
                        n[link] += c
                    q_now = charge(n)
                    self.tallies[self._last_family]['accepted_idle' if target == head else 'accepted'] += 1
                    head = target
            # The library does not always offer a clean step on every trail, so the drawn
            # shape may be unclean: its Δn would put charge outside the valid G-space (a
            # dipole in the wrong place, a quadrupole, or a Δq ≡ 0 multi-link draw, whose
            # idle acceptance would break proposal symmetry) instead of shifting the
            # head's +1/-1 dipole.  That is not a special "malformed, never-happened"
            # event -- it is a proposal into a zero-probability region, i.e. an ordinary
            # Metropolis rejection with acceptance min(1, 0) = 0.  So, exactly like a
            # clean-but-rejected shape, the head stays put and we fall through to the
            # tally below.

            # Tally the head−tail displacement for the Intersection_Intersection correlator.
            # We tally on EVERY step, including these stay-puts.  A rejection is a genuine
            # self-loop of the chain, and self-loops leave detailed balance between distinct
            # states untouched (only clean, symmetric draws move between distinct states), so
            # the histogram still samples the stationary marginal ∝ G(r).  The estimator is a
            # time-average whose numerator and denominator share one clock; dropping stay-puts
            # would reweight G(r) by the configuration- and position-dependent fraction of
            # clean proposals and bias the correlator.  (One could instead propose only among
            # the clean shapes and count each as a step, but that proposal is asymmetric and
            # would then require a Metropolis--Hastings |C(s)|/|C(s')| correction; the single
            # uniform draw here avoids it.)
            disp = tuple((head[k] - tail[k]) % N for k in range(D))
            displacements[disp] += 1

    def report(self):
        l = np.array(self.worm_lengths)
        if len(l) == 0:
            lines = ['There were 0 worms.']
        else:
            lines = [f'There were {len(l)} worms.\nWorms lengths:\n'
                     f'    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}']
        lines.append(f'{"family":>8} {"drawn":>10} {"unclean":>10} {"clean":>10} '
                     f'{"idle":>10} {"accepted":>10} {"acc.idle":>10}')
        for family, t in self.tallies.items():
            lines.append(f'{family:>8} {t["drawn"]:>10} {t["unclean"]:>10} {t["clean"]:>10} '
                         f'{t["idle"]:>10} {t["accepted"]:>10} {t["accepted_idle"]:>10}')
        return '\n'.join(lines)
