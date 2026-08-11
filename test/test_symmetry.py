#!/usr/bin/env python

r"""Lattice symmetry transformations and generators.

The hypercubic torus's space group $G = \mathbb{Z}_N^D \rtimes (\{\pm 1\}^D
\rtimes S_D)$ acts on every :class:`~supervillain.lattice.Form` by
:func:`~supervillain.lattice.translate`, :func:`~supervillain.lattice.reflect`,
and :func:`~supervillain.lattice.permute`; :mod:`supervillain.generator.symmetry`
draws elements of it (or of the subgroup an action declares admissible via its
own ``admissible_symmetries``) as always-accepted Markov moves.

An always-accepted move has no acceptance rate to reveal a bug, so these tests
are exhaustive over group elements rather than sampled, mirroring the gates
this file was migrated from
(``no-intersections/kernel-relax-2026-08-04/test_symmetry.py``).

.. note::
    Every quantity here transforms according to its tensorial type: $q = dn
    \wedge dn$ is a top-degree form and so transforms COVARIANTLY, scalar
    observables are invariant, ``TorusWrapping`` is a vector.  Constraint
    preservation ($q = 0$ survives) is a COROLLARY of $q$'s equivariance ---
    ``apply`` is linear, so $g(q) = g(0) = 0$ whenever $q = 0$ to start ---
    not a separate fact to verify from scratch.

.. warning::
    NO DATA DEPENDENCY.  Every fixture here is built in a few lines from a
    small ``Lattice`` and a seeded RNG; nothing reads an ensemble file.  The
    notebook (``no-intersections/kernel-relax-2026-08-04/test_symmetry.py``)
    keeps a companion integration test for the one thing a cheap fixture
    cannot honestly stand in for: a configuration where $q = dn \wedge dn = 0$
    NON-trivially, by cancellation among several nonzero components of $dn$
    rather than because $dn$ has only one nonzero component to begin with
    (see the warning on :func:`_degenerate_zero_charge_n` below). Only the
    sampler produces those.
"""

import itertools
from math import factorial

import numpy as np
import pytest

from supervillain.lattice import (
    Lattice, Form, d, delta, star, wedge, laplacian, translate, reflect, permute,
)
import supervillain.lattice.interlaced as interlaced
from supervillain.generator.symmetry import (
    Symmetry, all_flip_sets, LatticeSymmetry, Translation, Reflection,
    AxisPermutation, SpaceGroup, Conjugation,
)
from supervillain.action import Action, Villain, Worldline
from supervillain.observable.wrapping import TorusWrapping
from supervillain.observable.action import ActionDensity
from supervillain.observable.energy import InternalEnergyDensity
from supervillain.observable.winding import WindingSquared


class WedgeConstrained(Villain):
    r"""A formulation whose Boltzmann weight contains a WEDGE, and which
    therefore admits only $\{I, -I\} \times S_D$ rather than the whole point
    group.

    The wedge is a cup product, natural only under order-preserving cubical
    maps: a reflection carries it into the opposite cup product, differing by
    a coboundary.  A cup *square* survives the full inversion even though a
    general wedge does not, which is what leaves $\{I, -I\}$ --- see
    :func:`test_wedge_equivariance`, which measures both halves of that claim.

    .. note::
        This exists so the tests below gate the CONTRACT --- an action
        declares a restricted group, and the generators honour it --- rather
        than one particular client of it.  The No-Intersection model, whose
        constraint $q = dn \wedge dn = 0$ is exactly such a weight, is the
        motivating case; its own declaration is gated with that model.
    """

    def admissible_symmetries(self):
        D = self.Lattice.D
        return dict(translations=True, flip_sets=[(), tuple(range(D))],
                    perms=True)


def random_form(lattice, degree, seed, dtype=np.int64, lo=-999, hi=1000):
    r"""A form with generic entries.

    .. warning::
        Tests that count orbits or exercise the excluded-direction half of an
        equivariance check REQUIRE a generic form.  A structured form has a
        nontrivial stabilizer --- a constant 1-form has orbit size 1 under
        $S_3$, not 6 --- so replacing this with a constant silently turns a
        real check into a vacuous one.
    """
    f = lattice.form(degree, dtype=dtype)
    rng = np.random.default_rng(seed)
    a = np.asarray(f)
    if np.issubdtype(np.dtype(dtype), np.integer):
        a[...] = rng.integers(lo, hi, size=a.shape)
    else:
        a[...] = rng.uniform(lo, hi, size=a.shape)
    return f


def _charge(n):
    r"""$q = dn \wedge dn$, the no-intersection topological-charge density,
    as a top-degree :class:`Form`."""
    F = d(n)
    return wedge(F, F)


def _degenerate_zero_charge_n(lattice, seed):
    r"""A 1-form with $q = dn \wedge dn \equiv 0$ EXACTLY, built cheaply
    (no sampler) by giving $dn$ only one nonzero component.

    Component $(1,)$ (direction 1) is a function of $x_0$ alone; every other
    component is zero.  Then $(dn)_{01} = \partial_0 n_1$ is the only nonzero
    component of $dn$ --- $(dn)_{12} = -\partial_2 n_1 = 0$ since $n_1$ does
    not depend on $x_2$, and likewise for $(dn)_{02}, (dn)_{03}, (dn)_{13},
    (dn)_{23}$.  In $D = 4$ the wedge $dn \wedge dn$ needs two COMPLEMENTARY
    nonzero components of $dn$ (e.g. $(01)$ and $(23)$) to contribute to the
    single top component; with only $(01)$ nonzero there is no partner, so
    $q \equiv 0$ for the degenerate reason that a 2-form with one nonzero
    component always has zero cup square, not because of any interesting
    cancellation.

    .. warning::
        Use this ONLY to demonstrate that $q = 0$ survives the ADMISSIBLE
        subgroup, never the excluded one.  On this configuration ALL 14
        excluded flip sets also happen to preserve $q = 0$ (verified), so an
        excluded-direction assertion built on it would silently become
        vacuous --- exactly the trap the real (non-degenerate) equivariance
        test below is built to avoid.
    """
    D = lattice.D
    n = lattice.form(1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    vals = rng.integers(-9, 10, size=lattice.N)
    shape = (lattice.N,) + (1,) * (D - 1)
    np.asarray(n)[1] = vals.reshape(shape)
    return n


def _via_interlaced(fn, f, *args):
    r"""Route ``f`` through the interlaced reference implementation
    (:mod:`supervillain.lattice.interlaced`, which operates on raw $(2N)^D$
    arrays) and back, for comparison against the dense production path."""
    xi = np.asarray(f.to_interlaced())
    return Form.from_interlaced(f.degree, fn(xi, *args), lattice=f.lattice)


def _scalars(S, phi, n):
    r"""The four scalar observables, read directly from their per-formulation
    static methods (e.g. ``ActionDensity.Villain``) rather than by building a
    :class:`~supervillain.Ensemble` around a single configuration.

    .. note::
        Numerically identical to going through ``Ensemble`` (verified) ---
        that machinery exists to batch many configurations and is needless
        overhead for one.  Calling the observable directly is the same
        pattern already used for :class:`TorusWrapping` above, and it is
        what keeps a 384-element sweep cheap.
    """
    return {
        'ActionDensity': float(ActionDensity.Villain(S, phi, n)),
        'InternalEnergyDensity': float(InternalEnergyDensity.Villain(S, phi, n)),
        'WindingSquared': float(WindingSquared.Villain(S, n)),
    }


# ---------------------------------------------------------------------------
# The three transformations: d-equivariance, inverses, exact orbit sizes,
# dense/interlaced agreement.  All data-free, all exact (integer arithmetic).
# ---------------------------------------------------------------------------

def _moves(D):
    r"""Representative elements of each factor of the space group, as
    ``(kind, flips, perm, shift)``.

    ``kind`` is ``'translation'``, ``'permutation'``, or ``'flip'``, so a test
    can assert a different verdict per factor --- which is the whole point,
    since :func:`~supervillain.lattice.star` and
    :func:`~supervillain.lattice.wedge` are equivariant under the first two
    and not the third.
    """
    identity = tuple(range(D))
    yield ('translation', (), identity, (1,) + (0,) * (D - 1))
    yield ('translation', (), identity, identity)
    for perm in itertools.permutations(range(D)):
        yield ('permutation', (), perm, (0,) * D)
    for flips in all_flip_sets(D):
        if flips:
            yield ('flip', flips, identity, (0,) * D)


def _apply(f, flips, perm, shift):
    r"""A space-group element applied to a form, in the same order
    :meth:`~supervillain.generator.symmetry.Symmetry.apply` uses."""
    return translate(reflect(permute(f, perm), flips), shift)


def _determinant(D, flips, perm):
    r"""$\det R$ for the signed permutation matrix of an element."""
    M = np.zeros((D, D), dtype=int)
    for mu in range(D):
        M[perm[mu], mu] = -1 if mu in flips else 1
    return int(round(np.linalg.det(M)))


def test_d_equivariance():
    r"""$d$ is equivariant under every element of the space group:
    $R(d\omega) = d(R\omega)$, at every degree, $D = 2, 3, 4, 5$.

    $d$ is the coboundary of the cell complex --- pure combinatorics of which
    cell borders which --- so it survives any relabelling of cells.
    """
    for D, N in ((2, 6), (3, 4), (4, 4), (5, 3)):
        lattice = Lattice(D, N)
        for degree in range(D):
            f = random_form(lattice, degree, 100 + D * 10 + degree)
            assert np.asarray(d(f)).any(), (
                f'D={D} degree={degree}: d of the reference form vanishes, '
                'which would make its equivariance vacuous')
            for kind, flips, perm, shift in _moves(D):
                lhs = np.asarray(_apply(d(f), flips, perm, shift))
                rhs = np.asarray(d(_apply(f, flips, perm, shift)))
                assert np.array_equal(lhs, rhs), (
                    f'D={D} degree={degree}: d is not equivariant under the '
                    f'{kind} flips={flips} perm={perm} shift={shift}')


def test_delta_equivariance():
    r"""$\delta$ is equivariant under every element of the space group, at
    every degree.

    This is what :meth:`Worldline.admissible_symmetries
    <supervillain.action.Worldline.admissible_symmetries>` rests on.  It
    follows from :func:`test_d_equivariance` together with
    ``test_compact_adjointness`` in :source:`test/test_lattice.py`: $\delta$
    is the adjoint of $d$ under the cell-wise inner product, which the space
    group leaves invariant (checked below).  Note that it does NOT follow
    from $\delta \sim {\star} d {\star}$ --- :func:`test_star_equivariance`
    shows ${\star}$ is not equivariant under a sign flip at all.
    """
    for D, N in ((2, 6), (3, 4), (4, 4)):
        lattice = Lattice(D, N)
        for degree in range(1, D + 1):
            f = random_form(lattice, degree, 200 + D * 10 + degree)
            assert np.asarray(delta(f)).any(), (
                f'D={D} degree={degree}: delta of the reference form vanishes')
            for kind, flips, perm, shift in _moves(D):
                lhs = np.asarray(_apply(delta(f), flips, perm, shift))
                rhs = np.asarray(delta(_apply(f, flips, perm, shift)))
                assert np.array_equal(lhs, rhs), (
                    f'D={D} degree={degree}: delta is not equivariant under '
                    f'the {kind} flips={flips} perm={perm} shift={shift}')


def test_inner_product_invariance():
    r"""The cell-wise inner product $\langle \omega, \eta \rangle = \sum_c
    \omega_c \eta_c$ is invariant under every element of the space group.

    Together with :func:`test_d_equivariance` this is *why*
    :func:`test_delta_equivariance` holds: an element that permutes cells and
    squares signs cannot change a sum over cells, so the adjoint of an
    equivariant operator is equivariant.
    """
    for D, N in ((2, 6), (3, 4), (4, 4)):
        lattice = Lattice(D, N)
        for degree in range(D + 1):
            a = random_form(lattice, degree, 300 + D * 10 + degree)
            b = random_form(lattice, degree, 400 + D * 10 + degree)
            reference = int((np.asarray(a) * np.asarray(b)).sum())
            assert reference != 0, (
                f'D={D} degree={degree}: vanishing reference inner product')
            for kind, flips, perm, shift in _moves(D):
                Ra = _apply(a, flips, perm, shift)
                Rb = _apply(b, flips, perm, shift)
                got = int((np.asarray(Ra) * np.asarray(Rb)).sum())
                assert got == reference, (
                    f'D={D} degree={degree}: the inner product changed from '
                    f'{reference} to {got} under the {kind} flips={flips} '
                    f'perm={perm} shift={shift}')


def test_laplacian_equivariance():
    r"""$\Delta$ is equivariant under every element of the space group.

    It is built from $d$ and $\delta$, so this follows from
    :func:`test_d_equivariance` and :func:`test_delta_equivariance`; it is
    asserted directly so the operator table in :doc:`the sampling docs
    </supervillain/generator>` is checked row by row rather than by
    inference.
    """
    for D, N in ((2, 6), (3, 4), (4, 4)):
        lattice = Lattice(D, N)
        for degree in range(D + 1):
            f = random_form(lattice, degree, 500 + D * 10 + degree)
            assert np.asarray(laplacian(f)).any(), (
                f'D={D} degree={degree}: the Laplacian of the reference '
                'form vanishes')
            for kind, flips, perm, shift in _moves(D):
                lhs = np.asarray(_apply(laplacian(f), flips, perm, shift))
                rhs = np.asarray(laplacian(_apply(f, flips, perm, shift)))
                assert np.array_equal(lhs, rhs), (
                    f'D={D} degree={degree}: the Laplacian is not equivariant '
                    f'under the {kind} flips={flips} perm={perm} shift={shift}')


def test_star_equivariance():
    r"""${\star}$ is equivariant up to $\det R$ under translations and axis
    permutations, and NOT equivariant under any sign flip.

    The $\det R$ is ${\star}$'s continuum behaviour --- it is orientation-odd
    there too --- so equivariance means $R({\star}\omega) = \det R \;
    {\star}(R\omega)$.

    .. warning::
        The failure under a sign flip is asserted in BOTH directions: not
        equal to $+{\star}(R\omega)$ and not equal to $-{\star}(R\omega)$
        either.  It is not a sign convention that could be absorbed; the base
        point shifts, and no overall sign repairs it.  Note that orientation
        is not what breaks it --- odd permutations are orientation-reversing
        and pass.
    """
    for D, N in ((2, 6), (3, 4), (4, 4)):
        lattice = Lattice(D, N)
        for degree in range(D + 1):
            f = random_form(lattice, degree, 600 + D * 10 + degree)
            assert np.asarray(star(f)).any(), (
                f'D={D} degree={degree}: the star of the reference form vanishes')
            for kind, flips, perm, shift in _moves(D):
                lhs = np.asarray(_apply(star(f), flips, perm, shift))
                rhs = np.asarray(star(_apply(f, flips, perm, shift)))
                signed = _determinant(D, set(flips), perm) * rhs
                if kind == 'flip':
                    assert not np.array_equal(lhs, signed) and \
                           not np.array_equal(lhs, -signed), (
                        f'D={D} degree={degree}: star is unexpectedly '
                        f'equivariant under the flip {flips}.  If this now '
                        'holds, the operator table in the sampling docs and '
                        'the admissible symmetries of the constrained '
                        'actions are both too conservative.')
                else:
                    assert np.array_equal(lhs, signed), (
                        f'D={D} degree={degree}: star is not equivariant up '
                        f'to det under the {kind} perm={perm} shift={shift}')


def test_wedge_equivariance():
    r"""$a \wedge b$ is equivariant under translations and axis permutations
    and NOT under any sign flip; the cup square $a \wedge a$ is additionally
    equivariant under the full inversion, up to $(-1)^p$.

    The lattice wedge is a cup product, natural only under order-preserving
    cubical maps, so a sign flip carries it into the opposite cup product ---
    which differs by a coboundary and moves the result cell by cell.

    .. note::
        A general wedge fails under a sign flip in BOTH directions --- equal
        to neither $+(Ra) \wedge (Rb)$ nor $-(Ra) \wedge (Rb)$ --- so it is
        not a sign convention that could be absorbed.  The cup square is the
        exception, and only for the FULL inversion:

        .. math ::
            R_{\mathrm{inv}}(a \wedge a) = (-1)^p\, (Ra) \wedge (Ra)

        That is what leaves $\{I, -I\}$ admissible in
        :meth:`WedgeConstrained.admissible_symmetries` rather than the
        identity alone.  The sign is irrelevant there, since
        $(-1)^p \cdot 0 = 0$ --- what matters is that the cup square is
        equivariant under the inversion at all, which a general wedge is not.
    """
    for D, N in ((2, 6), (3, 4), (4, 4)):
        lattice = Lattice(D, N)
        full_inversion = tuple(range(D))
        for degree in range(1, D // 2 + 1):
            a = random_form(lattice, degree, 700 + D * 10 + degree)
            b = random_form(lattice, degree, 800 + D * 10 + degree)
            assert np.asarray(wedge(a, b)).any(), (
                f'D={D} degree={degree}: the reference wedge vanishes')
            for kind, flips, perm, shift in _moves(D):
                lhs = np.asarray(_apply(wedge(a, b), flips, perm, shift))
                rhs = np.asarray(wedge(_apply(a, flips, perm, shift),
                                       _apply(b, flips, perm, shift)))
                if kind == 'flip':
                    assert not np.array_equal(lhs, rhs), (
                        f'D={D} degree={degree}: a general wedge is '
                        f'unexpectedly equivariant under the flip {flips}')
                else:
                    assert np.array_equal(lhs, rhs), (
                        f'D={D} degree={degree}: the wedge is not equivariant '
                        f'under the {kind} perm={perm} shift={shift}')

            # The cup square under the full inversion, up to (-1)^p.
            if not np.asarray(wedge(a, a)).any():
                continue
            identity = tuple(range(D))
            inverted = np.asarray(_apply(wedge(a, a), full_inversion,
                                         identity, (0,) * D))
            Ra = _apply(a, full_inversion, identity, (0,) * D)
            assert np.array_equal(inverted, (-1) ** degree
                                  * np.asarray(wedge(Ra, Ra))), (
                f'D={D} degree={degree}: the cup square should be equivariant '
                f'under the full inversion up to (-1)^{degree}; this is what '
                'leaves {I, -I} admissible for a wedge-constrained action')

            # ... but not under a PARTIAL flip, in either sign.
            for flips in all_flip_sets(D):
                if not flips or flips == full_inversion:
                    continue
                partial = np.asarray(_apply(wedge(a, a), flips, identity, (0,) * D))
                Pa = _apply(a, flips, identity, (0,) * D)
                square = np.asarray(wedge(Pa, Pa))
                assert not np.array_equal(partial, square) and \
                       not np.array_equal(partial, -square), (
                    f'D={D} degree={degree}: the cup square is unexpectedly '
                    f'equivariant under the partial flip {flips}, which would '
                    'widen the admissible group of a wedge-constrained action')


def _inverse_test_group(lattice):
    r"""Elements to sweep in :func:`test_inverse_undoes_the_element`: the
    FULL group at $D = 2$; at $D \ge 3$, the full point group (all flips,
    all permutations) crossed with just TWO shifts instead of all $N^D$ of
    them.

    See the docstring of the caller for why this loses no coverage of the
    part of the inverse formula that is actually delicate.
    """
    D = lattice.D
    if D < 3:
        yield from Symmetry.all(lattice)
        return
    representative_shifts = ((0,) * D, tuple(range(1, D + 1)))
    for perm in itertools.permutations(range(D)):
        for flip in all_flip_sets(D):
            for shift in representative_shifts:
                yield Symmetry(lattice, shift=shift, flips=flip, perm=perm)


def test_inverse_undoes_the_element():
    r"""Applying an element then its inverse returns the form bit for bit,
    and a pure reflection squares to the identity (it is its own inverse).

    .. note::
        The inverse formula (:meth:`~supervillain.generator.symmetry.Symmetry.inverse`)
        is

        .. math ::
            \sigma = \pi^{-1}, \quad G = \pi^{-1}(F), \quad
            b_\mu = \begin{cases}
                +a_{\pi(\mu)} & \pi(\mu) \in F \\
                -a_{\pi(\mu)} & \text{otherwise}
            \end{cases}

        Its entire subtlety --- the conjugation $\sigma = \pi^{-1}$, $G =
        \pi^{-1}(F)$, and which sign each shift component $b_\mu$ picks up
        --- lives in the point-group part $(F, \pi)$ alone.  Sweeping the
        FULL point group ($2^D D!$ elements: 48 at $D = 3$, 384 at $D = 4$)
        exhausts every one of those conjugation/sign patterns.  The
        dependence on the shift $a$ itself is exactly LINEAR
        ($b_\mu = \pm a_{\pi(\mu)}$; no $a$ enters which sign is chosen), so
        TWO shifts suffice: zero (isolates the point-group part with no
        shift arithmetic at all) and one with a DISTINCT nonzero value on
        every axis (so a bug that reads the wrong component of $a$, or
        applies the wrong sign to it, changes the result rather than hiding
        behind a repeated or zero entry).  This loses no coverage of the
        delicate part while cutting the $D = 4$ element count from
        $N^D 2^D D! = 31104$ down to $2 \cdot 2^D D! = 768$.  $D = 2$
        ($N = 5$: 200 elements) stays fully exhaustive, including every
        shift --- cheap enough that there is no reason to sample it.
    """
    for D, N in ((2, 5), (3, 4), (4, 3)):
        lattice = Lattice(D, N)
        elements = list(_inverse_test_group(lattice))
        for degree in range(D + 1):
            f = random_form(lattice, degree, 200 + D * 10 + degree)
            for g in elements:
                back = g.inverse().apply(g.apply(f))
                assert np.array_equal(np.asarray(back), np.asarray(f)), (
                    f'D={D} degree={degree}: inverse of {g} is not an inverse')
            for flips in all_flip_sets(D):
                g = Symmetry(lattice, flips=flips)
                twice = g.apply(g.apply(f))
                assert np.array_equal(np.asarray(twice), np.asarray(f)), (
                    f'D={D}: reflection {flips} does not square to the identity')


def test_orbit_sizes_are_exact():
    r"""A generic form's orbit under each factor, and their composition,
    has exactly the group's order --- no fewer (distinct elements acting
    identically) and no more (the move leaving the group).

    .. note::
        The composed-orbit ``SpaceGroup`` check at $D = 4$ uses $N = 2$
        deliberately: its orbit already has $N^D 2^D D! = 6144$ elements at
        that size, so a larger $N$ buys nothing but wall-clock.  $N = 2$ is
        fine HERE even though it is too small for
        :func:`test_inverse_undoes_the_element` (where $-1 \equiv +1 \bmod
        2$ makes the two signs in that formula indistinguishable): counting
        DISTINCT images never needs sign-distinguishability, only that a
        generic form's stabilizer is trivial, and that was checked directly
        --- all four orbit sizes below are exact at $N = 2, D = 4$,
        matching their formulas with no collisions.
    """
    import hashlib

    def digest(form):
        return hashlib.blake2b(np.ascontiguousarray(form).tobytes(),
                               digest_size=16).digest()

    for D, N in ((2, 3), (2, 5), (3, 3), (4, 2)):
        lattice = Lattice(D, N)
        f = random_form(lattice, 1, 300 + D)
        factors = {
            'Translation':  dict(translations=True,  flip_sets=[()], perms=False),
            'Reflection':   dict(translations=False, flip_sets=None, perms=False),
            'Permutation':  dict(translations=False, flip_sets=[()], perms=True),
            'SpaceGroup':   dict(translations=True,  flip_sets=None, perms=True),
        }
        expected = {
            'Translation': N ** D,
            'Reflection': 2 ** D,
            'Permutation': factorial(D),
            'SpaceGroup': N ** D * 2 ** D * factorial(D),
        }
        for name, which in factors.items():
            orbit = {digest(g.apply(f)) for g in Symmetry.all(lattice, **which)}
            assert len(orbit) == expected[name], (
                f'{name} orbit at D={D} N={N} is {len(orbit)}, expected '
                f'{expected[name]}')


def test_interlaced_round_trip_is_exact():
    r"""``to_interlaced`` then ``from_interlaced`` is the identity, exactly,
    at every degree."""
    for D, N in ((2, 6), (3, 4), (4, 4)):
        lattice = Lattice(D, N)
        for degree in range(D + 1):
            f = random_form(lattice, degree, 400 + D * 10 + degree)
            back = Form.from_interlaced(degree, np.asarray(f.to_interlaced()),
                                        lattice=lattice)
            assert np.array_equal(np.asarray(back), np.asarray(f)), (
                f'D={D} degree={degree}: interlaced round trip is lossy')


def test_dense_and_interlaced_agree_exactly():
    r"""The production (dense, :mod:`supervillain.lattice.compact`) and
    reference (:mod:`supervillain.lattice.interlaced`) implementations agree
    exactly, for every move and every group element.

    Two independently derived implementations agreeing is the strongest
    available check here, precisely because an always-accepted move has no
    acceptance rate of its own to reveal a bug.
    """
    for D, N in ((2, 6), (3, 4), (4, 4)):
        lattice = Lattice(D, N)
        for degree in range(D + 1):
            f = random_form(lattice, degree, 500 + D * 10 + degree)
            pairs = (
                [(translate(f, s), _via_interlaced(interlaced.translate, f, s))
                 for s in ((1,) + (0,) * (D - 1), tuple(range(D)))]
                + [(reflect(f, fl), _via_interlaced(interlaced.reflect, f, fl))
                   for fl in all_flip_sets(D)]
                + [(permute(f, p), _via_interlaced(interlaced.permute, f, p))
                   for p in itertools.permutations(range(D))]
            )
            for dense, ref in pairs:
                assert np.array_equal(np.asarray(dense), np.asarray(ref)), (
                    f'D={D} degree={degree}: dense and interlaced disagree')


# ---------------------------------------------------------------------------
# Equivariance: q, scalar observables, TorusWrapping, Conjugation.
# ---------------------------------------------------------------------------

def test_topological_charge_equivariant_under_admissible_not_excluded():
    r"""$q = dn \wedge dn$ is a top-degree FORM, so it transforms
    covariantly: $q(g \cdot n) = g \cdot q(n)$.  This holds for every element
    :class:`WedgeConstrained` admits, and --- because the
    whitelist must be exactly right in both directions, not merely
    conservative --- for NONE of the excluded flip sets.

    Replaces old gates 5 and 9.  $q = 0$ preservation (old gate 5's headline
    claim) is a COROLLARY: ``apply`` is linear, so if $q(n) = 0$ then
    $q(g \cdot n) = g \cdot q(n) = g \cdot 0 = 0$ for any equivariant $g$.
    See :func:`test_topological_charge_zero_survives_admissible_symmetry`
    for that corollary made concrete on an actual $q \equiv 0$ configuration.
    """
    D, N = 4, 4
    lattice = Lattice(D, N)
    S = WedgeConstrained(lattice, kappa=0.03)
    n = random_form(lattice, 1, 23)
    q = _charge(n)
    assert np.asarray(q).any(), (
        're-seed: this random configuration satisfies q == 0 everywhere, '
        'which would make the excluded-direction check vacuous')

    allowed = S.admissible_symmetries()['flip_sets']
    admissible = list(Symmetry.all(lattice, translations=False,
                                    flip_sets=allowed, perms=True))
    for shift in ((1, 0, 0, 0), (0, 2, 0, 0), (1, 1, 1, 1), (3, 1, 0, 2)):
        admissible.append(Symmetry(lattice, shift=shift))

    for g in admissible:
        lhs = np.asarray(_charge(g.apply(n)))
        rhs = np.asarray(g.apply(q))
        assert np.array_equal(lhs, rhs), (
            f'q does not transform covariantly under the admissible {g}')
    assert len(admissible) == 48 + 4

    excluded = [f for f in all_flip_sets(D) if f not in allowed]
    assert len(excluded) == 14
    for flips in excluded:
        g = Symmetry(lattice, flips=flips)
        lhs = np.asarray(_charge(g.apply(n)))
        rhs = np.asarray(g.apply(q))
        assert not np.array_equal(lhs, rhs), (
            f'q transforms covariantly under the EXCLUDED flip set {flips} '
            f'-- either it is safe and should be whitelisted, or this '
            f'assertion is broken')


def test_topological_charge_zero_survives_admissible_symmetry():
    r"""$q = 0$ survives every admissible element --- the corollary of
    :func:`test_topological_charge_equivariant_under_admissible_not_excluded`
    made concrete on an actual $q \equiv 0$ configuration, built cheaply by
    :func:`_degenerate_zero_charge_n` rather than read from an ensemble.

    Deliberately does NOT also check the excluded direction: see the warning
    on :func:`_degenerate_zero_charge_n` for why that would be vacuous here.
    """
    D, N = 4, 4
    lattice = Lattice(D, N)
    S = WedgeConstrained(lattice, kappa=0.03)
    n = _degenerate_zero_charge_n(lattice, seed=99)
    assert not np.asarray(_charge(n)).any(), 'fixture does not satisfy q == 0'

    allowed = S.admissible_symmetries()['flip_sets']
    checked = 0
    for g in Symmetry.all(lattice, translations=False, flip_sets=allowed,
                          perms=True):
        assert not np.asarray(_charge(g.apply(n))).any(), (
            f'q != 0 under the admissible {g}')
        checked += 1
    assert checked == 48


def test_action_and_scalars_invariant_under_admissible_symmetry():
    r"""The action and every scalar observable are invariant under a
    constrained action's admissible subgroup, and the Villain action alone is
    invariant under the FULL point group --- exactly what licenses
    ``SpaceGroup`` always to accept single axis flips for ``Villain``.

    Replaces old gate 6.  Compared with EXACT equality, spatial
    transformations reorder the summation so the last bits move; use a
    relative tolerance (measured worst case here $\sim 10^{-15}$, comfortably
    under the $10^{-12}$ used below).
    """
    D, N = 4, 4
    lattice = Lattice(D, N)
    n = random_form(lattice, 1, 11, lo=-5, hi=6)
    phi = random_form(lattice, 0, 13, dtype=float, lo=-np.pi, hi=np.pi)
    observables = ('ActionDensity', 'InternalEnergyDensity',
                   'WindingSquared')

    constrained = WedgeConstrained(lattice, kappa=0.03)
    reference = _scalars(constrained, phi, n)
    assert all(reference[o] != 0.0 for o in observables), (
        'a vanishing reference observable would make its invariance check vacuous')
    factors = dict(constrained.admissible_symmetries()) | {'translations': False}
    checked = 0
    for g in Symmetry.all(lattice, **factors):
        got = _scalars(constrained, g.apply(phi), g.apply(n))
        checked += 1
        for o in observables:
            relative = abs(reference[o] - got[o]) / max(abs(reference[o]), 1e-30)
            assert relative < 1e-12, (
                f'{o} changed by {relative:.2e} under the admissible {g}')
    assert checked == 48

    # The Villain action is invariant under the FULL point group, unrestricted
    # by the no-intersection constraint's flip-set whitelist.
    SV = Villain(lattice, kappa=0.03)
    reference_v = float(ActionDensity.Villain(SV, phi, n))
    checked_v = 0
    for g in Symmetry.all(lattice, translations=False):
        got_v = float(ActionDensity.Villain(SV, g.apply(phi), g.apply(n)))
        checked_v += 1
        relative = abs(reference_v - got_v) / max(abs(reference_v), 1e-30)
        assert relative < 1e-12, (
            f'Villain ActionDensity changed by {relative:.2e} under {g}')
    assert checked_v == 2 ** D * factorial(D)


def test_torus_wrapping_transforms_as_a_vector():
    r"""``TorusWrapping`` is a VECTOR: component $\mu$ moves to direction
    ``perm[mu]``, negated if $\mu$ was flipped --- not merely invariant.
    This is the check that catches a component/axis mismatch.

    The fixture's seed is fixed deliberately: ``TorusWrapping`` must have
    four DISTINCT nonzero components (verified below) so that no permutation
    or sign error can hide behind a repeated or zero entry.
    """
    D, N = 4, 4
    lattice = Lattice(D, N)
    S = WedgeConstrained(lattice, kappa=0.03)
    n = random_form(lattice, 1, 823)
    phi = random_form(lattice, 0, 1, dtype=float, lo=-np.pi, hi=np.pi)
    reference = np.asarray(TorusWrapping.Villain(S, phi, n))
    assert (reference != 0).all() and len(set(reference.tolist())) == 4, (
        're-seed: TorusWrapping needs four distinct nonzero components or '
        'this gate is weakened')

    factors = dict(S.admissible_symmetries()) | {'translations': False}
    checked = 0
    for g in Symmetry.all(lattice, **factors):
        got = np.asarray(TorusWrapping.Villain(S, phi, g.apply(n)))
        expected = np.zeros(D, dtype=np.int64)
        for mu in range(D):
            expected[g.perm[mu]] = (-1 if mu in g.flips else 1) * reference[mu]
        assert np.array_equal(got, expected), (
            f'TorusWrapping did not transform as a vector under {g}: got '
            f'{got}, expected {expected}')
        checked += 1
    assert checked == 48


def test_conjugation_leaves_action_and_charge_invariant():
    r"""Charge conjugation $C$ ($\varphi \to -\varphi$, $n \to -n$) leaves the
    Villain action invariant BIT FOR BIT, and leaves $q$ invariant (not
    merely zero-preserving) because $q$ is QUADRATIC in $F = dn$.

    .. note::
        Bit-for-bit is the right standard here, unlike the spatial
        transformations above: negation reorders no summation, and
        $(-x)^2 = x^2$ exactly, so there is no floating-point excuse for a
        mismatch.
    """
    D, N = 4, 4
    lattice = Lattice(D, N)
    S = Villain(lattice, kappa=0.03)
    n = random_form(lattice, 1, 11, lo=-5, hi=6)
    phi = random_form(lattice, 0, 13, dtype=float, lo=-np.pi, hi=np.pi)

    reference_action = S(phi, n)
    assert S(-phi, -n) == reference_action, (
        'C does not leave the Villain action bit-for-bit invariant')

    generator = Conjugation(S, rng=np.random.default_rng(0))
    generator.draw = lambda: True     # force the C branch, not the coin
    conjugated = generator.step({'phi': phi, 'n': n})
    assert S(conjugated['phi'], conjugated['n']) == reference_action, (
        'Conjugation.step does not reproduce the bit-for-bit invariant action')

    arbitrary_n = random_form(lattice, 1, 731)
    q_before = np.asarray(_charge(arbitrary_n))
    assert q_before.any(), (
        're-seed: this configuration satisfies q == 0 everywhere, which '
        'would weaken invariance to mere zero-preservation')
    q_after = np.asarray(_charge(-arbitrary_n))
    assert np.array_equal(q_before, q_after), (
        'C did not leave q invariant (only zero-preserving) on an '
        'arbitrary configuration')


# ---------------------------------------------------------------------------
# The generators.
# ---------------------------------------------------------------------------

def test_apply_never_aliases_its_input():
    r"""``Symmetry.apply`` must never alias its input's memory --- checked at
    the IDENTITY element specifically, since that is exactly where
    permute/reflect/translate each degenerate to a pass-through on a 0-form.
    A Form sharing memory with its input would let a later in-place write
    silently corrupt the previous configuration."""
    S = Villain(Lattice(3, 4), kappa=0.5)
    phi_in = S.configurations(1)[0]['phi']
    identity = Symmetry(S.Lattice)
    phi_out = identity.apply(phi_in)
    assert not np.shares_memory(np.asarray(phi_out), np.asarray(phi_in))


def test_generators_preserve_degree_dtype_and_undeclared_entries():
    r"""Each generator runs, preserves every field's degree and dtype, and
    leaves an undeclared entry (e.g. an inline observable riding along in
    the record) untouched."""
    S = Villain(Lattice(3, 4), kappa=0.5)
    start = S.configurations(1)[0]
    for cls in (Translation, Reflection, AxisPermutation, SpaceGroup):
        generator = cls(S, rng=np.random.default_rng(5))
        configuration = dict(start)
        configuration['not_a_field'] = 'passed through untouched'
        for _ in range(10):
            configuration = generator.step(configuration)
        assert configuration['not_a_field'] == 'passed through untouched', (
            f'{cls.__name__} mangled an undeclared entry')
        for key, degree in generator.degrees().items():
            assert configuration[key].degree == degree, (
                f'{cls.__name__} changed the degree of {key}')
            assert configuration[key].dtype == np.asarray(start[key]).dtype, (
                f'{cls.__name__} changed the dtype of {key}')
        assert generator.report()


def test_declared_field_with_unreadable_degree_raises():
    r"""A DECLARED field whose degree cannot be read is an error, never a
    silent pass-through."""
    S = Villain(Lattice(3, 4), kappa=0.5)
    start = S.configurations(1)[0]
    generator = SpaceGroup(S, rng=np.random.default_rng(5))
    generator._degrees = None
    generator.degrees()
    generator._degrees['phantom'] = None
    with pytest.raises((TypeError, ValueError, KeyError)):
        generator.step(dict(start) | {'phantom': np.zeros(3)})


def test_reflection_refuses_a_constrained_action():
    r"""``Reflection`` refuses a wedge-constrained action:
    a generator named for sign flips that cannot apply most of them would be
    a lie.  ``SpaceGroup`` (below) is the one that restricts honestly."""
    constrained = WedgeConstrained(Lattice(4, 3), kappa=0.03)
    with pytest.raises(ValueError):
        Reflection(constrained)


def test_reflection_does_not_refuse_worldline():
    r"""``Reflection`` does NOT raise for
    :class:`~supervillain.action.Worldline`: unlike
    :class:`WedgeConstrained`, it admits the full space group and has no restriction to refuse."""
    Wl = Worldline(Lattice(4, 3), kappa=0.5, W=3)
    Reflection(Wl)     # must not raise


def test_space_group_narrows_and_reports_the_restricted_order():
    r"""``SpaceGroup`` restricts to the admissible subgroup for a constrained
    action, and its ``report()`` names the group actually drawn from, so a
    restricted generator never reads as though it had the full one."""
    D, N = 4, 3
    full = SpaceGroup(Villain(Lattice(D, N), kappa=0.03))
    assert full.order == N ** D * 2 ** D * factorial(D)

    narrowed = SpaceGroup(WedgeConstrained(Lattice(D, N), kappa=0.03))
    assert narrowed.order == N ** D * 2 * factorial(D)
    assert str(narrowed.order) in narrowed.report()


def test_space_group_draws_only_whitelisted_flip_sets():
    r"""Over many draws, ``SpaceGroup`` of a wedge-constrained action never
    returns an element outside its admissible flip-set whitelist."""
    constrained = WedgeConstrained(Lattice(4, 4), kappa=0.03)
    allowed = set(constrained.admissible_symmetries()['flip_sets'])
    generator = SpaceGroup(constrained, rng=np.random.default_rng(11))
    for _ in range(200):
        assert generator.draw().flips in allowed, (
            'SpaceGroup drew a flip set outside the '
            'admissible whitelist')


def test_undeclared_action_fails_closed():
    r"""An action outside :class:`LatticeSymmetry`'s known dispatch --- one
    with no ``admissible_symmetries`` method --- must RAISE rather than
    silently receive the full space group.  Because a ``LatticeSymmetry``
    move is always accepted, failing open here would corrupt an ensemble
    with no acceptance rate ever available to reveal it."""
    class _NotYetEstablished:
        r"""A stand-in for an action ``LatticeSymmetry`` has never been
        taught about."""
        def __init__(self, lattice):
            self.Lattice = lattice

    with pytest.raises(NotImplementedError):
        LatticeSymmetry(_NotYetEstablished(Lattice(3, 4)))


def test_action_subclass_that_forgets_fails_closed():
    r"""The other half of the fail-closed check: an action that DOES inherit
    from :class:`~supervillain.action.Action` but never overrides
    ``admissible_symmetries`` reaches the base class's declaration, which
    raises.  Inheriting the interface must not be mistaken for having
    answered the question --- an unmeasured action is exactly the one whose
    always-accepted moves could corrupt an ensemble unnoticed.
    """
    class _Forgetful(Action):
        r"""Inherits the interface, declares nothing."""
        def __init__(self, lattice):
            self.Lattice = lattice

    with pytest.raises(NotImplementedError):
        LatticeSymmetry(_Forgetful(Lattice(3, 4)))


# ---------------------------------------------------------------------------
# Worldline: the full space group, no reflection restriction.
# ---------------------------------------------------------------------------

def test_worldline_admits_the_full_space_group():
    r"""``Worldline`` admits the FULL space group: no reflection restriction,
    unlike a wedge-constrained action.

    Builds a VALID configuration ``m = delta(w)`` for a random integer
    2-form ``w`` --- valid because $\delta \circ \delta = 0$ --- with ``v``
    an independent random integer 2-form.  ``v`` being a 2-form (not a
    1-form) exercises the permutation SORT SIGN inside $\delta$: at a 1-form
    the insertion position is trivial, so a 2-form is the lowest degree
    where a sign-placement bug could hide.

    Deliberately data-free like every other test in this file: no ensemble
    needed, so unlike the notebook's original gates 5, 6, 7, 9, 10 this one
    was never skip-guarded even there.
    """
    lattice = Lattice(4, 4)
    D, N = lattice.D, lattice.N
    Wl = Worldline(lattice, kappa=0.5, W=3)

    w = random_form(lattice, 2, 1123)
    v = random_form(lattice, 2, 1129)
    m = delta(w)
    assert np.asarray(m).any(), (
        're-seed: reference m is identically zero, which would make the '
        'constraint check vacuous')
    assert Wl.valid({'m': m}), 'fixture m = delta(w) does not satisfy delta m == 0'

    reference_action = Wl(m, v)
    nontrivial_shift = (1, 2, 3, 1)
    assert any(s % N for s in nontrivial_shift), 'the translation is trivial mod N'

    checked = 0
    for base in Symmetry.all(lattice, translations=False, flip_sets=None, perms=True):
        g = Symmetry(lattice, shift=nontrivial_shift, flips=base.flips, perm=base.perm)
        mp, vp = g.apply(m), g.apply(v)
        assert Wl.valid({'m': mp}), f'delta m == 0 broken under {g}'
        got = Wl(mp, vp)
        relative = abs(got - reference_action) / max(abs(reference_action), 1e-30)
        assert relative < 1e-12, (
            f'Worldline action changed by {relative:.2e} under {g}')
        checked += 1
    assert checked == 2 ** D * factorial(D)

    full = SpaceGroup(Wl)
    assert full.order == N ** D * 2 ** D * factorial(D)
