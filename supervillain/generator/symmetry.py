#!/usr/bin/env python

r"""One element of the lattice space group, and the generators that draw them.

The symmetry group of the periodic hypercubic lattice is

.. math ::
    G = \mathbb{Z}_N^D \rtimes \left(\{\pm 1\}^D \rtimes S_D\right),
    \qquad |G| = N^D \cdot 2^D \cdot D!

--- translations, semidirect the hyperoctahedral point group.  This is not the
Poincare group: no boosts, Euclidean, discrete.

.. note::
    $S_D$ is NOT the rotation subgroup.  An odd permutation has determinant
    $-1$; a transposition $x_\mu \leftrightarrow x_\nu$ is the mirror across
    the diagonal hyperplane $x_\mu = x_\nu$.  In $D = 4$, 12 of the 24 axis
    permutations are orientation-reversing.  The point inversion $-I$ has
    determinant $(-1)^D$, so in even $D$ it lies INSIDE the rotation subgroup
    and the familiar 3D factorization into rotations times $\{I, -I\}$ fails.

Formulation-agnostic --- this module knows about :class:`Form`\ s and space
groups, not about any particular action --- so it lives at the top level of
:mod:`supervillain.generator`, beside :mod:`~supervillain.generator.combining`
and :mod:`~supervillain.generator.monitor`, rather than under ``villain`` or
``worldline``.  What varies by action is which subgroup is *admissible*; each
action declares that itself via ``admissible_symmetries`` (see
:meth:`supervillain.action.Villain.admissible_symmetries`,
:meth:`supervillain.action.Worldline.admissible_symmetries`, and
:meth:`supervillain.action.NoIntersections.admissible_symmetries`).
"""

import itertools

import numpy as np

from supervillain.h5 import ReadWriteable
from supervillain.generator import Generator
from supervillain.lattice import Form, translate, reflect, permute
from supervillain.lattice.compact import _perm_sign


class Symmetry:
    r"""A single space-group element $g = (a, F, \pi)$.

    Acting on a form, ``g`` permutes the axes, then reflects, then translates.

    .. note::
        Enumeration (:meth:`all`) and sampling (:meth:`draw`) come from this
        one class deliberately.  If a generator owned its draw and a gate owned
        its sweep independently, a gate could be exhaustive over a group the
        sampler does not actually cover uniformly, and nothing would catch it.

    Parameters
    ----------
    lattice: supervillain.lattice.Lattice
    shift: sequence of int, optional
        The translation $a$; reduced modulo $N$.  Default no translation.
    flips: sequence of int, optional
        The axes to negate, $F$.  Default none.
    perm: sequence of int, optional
        A permutation of ``range(D)``, $\pi$.  Default the identity.
    """

    def __init__(self, lattice, shift=None, flips=(), perm=None):
        D = lattice.D
        self.lattice = lattice
        self.shift = tuple(int(s) % lattice.N
                           for s in (shift if shift is not None else (0,) * D))
        self.flips = tuple(sorted(int(m) for m in flips))
        self.perm = tuple(int(m) for m in
                          (perm if perm is not None else range(D)))

    def __str__(self):
        return (f'Symmetry(shift={self.shift}, flips={self.flips}, '
                f'perm={self.perm})')

    __repr__ = __str__

    def __eq__(self, other):
        return (isinstance(other, Symmetry)
                and self.shift == other.shift
                and self.flips == other.flips
                and self.perm == other.perm)

    def __hash__(self):
        return hash((self.shift, self.flips, self.perm))

    @property
    def determinant(self):
        r"""$\det g = \varepsilon(\pi) \cdot (-1)^{\left|F\right|}$, which is
        $+1$ for an orientation-preserving element and $-1$ otherwise."""
        return _perm_sign(self.perm) * (-1) ** len(self.flips)

    def apply(self, form):
        r"""Transform a form by this element: permute, then reflect, then
        translate.

        .. note::
            Always returns an independent, C-contiguous array --- never a
            view of ``form``'s memory.  For the identity element every one of
            the three passes degenerates to a pass-through (``permute``'s
            transpose returns a view, and an empty flip set or a zero shift
            each return their argument unchanged), so without this copy the
            result would silently ALIAS the input --- and a later generator
            that writes into a configuration in place would then corrupt the
            PREVIOUS configuration, intermittently and unreproducibly.

        Parameters
        ----------
        form: Form

        Returns
        -------
        Form
            Same degree as ``form``.  Never shares memory with ``form``.
        """
        result = translate(reflect(permute(form, self.perm), self.flips),
                           self.shift)
        return Form(np.asarray(result).copy(order='C'),
                   degree=result.degree, lattice=result.lattice)

    def inverse(self):
        r"""The inverse element.

        .. warning::
            The group is a SEMIDIRECT product, so the inverse is not
            $(-a, F, \pi^{-1})$.  Writing $g = T_a R_F P_{\pi}$ and commuting
            $g^{-1} = P_{\pi^{-1}} R_F T_{-a}$ back into that order using
            $P_s R_F = R_{s(F)} P_s$ and $R_G T_c = T_{c'} R_G$ gives

            .. math ::
                \sigma = \pi^{-1}, \quad G = \pi^{-1}(F), \quad
                b_\mu = \begin{cases}
                    +a_{\pi(\mu)} & \pi(\mu) \in F \\
                    -a_{\pi(\mu)} & \text{otherwise}
                \end{cases}

        Returns
        -------
        Symmetry
        """
        D = self.lattice.D
        F = set(self.flips)
        inverse_perm = tuple(int(i) for i in np.argsort(np.asarray(self.perm)))
        return Symmetry(
            self.lattice,
            shift=tuple(self.shift[self.perm[mu]] if self.perm[mu] in F
                        else -self.shift[self.perm[mu]] for mu in range(D)),
            flips=tuple(sorted(inverse_perm[m] for m in F)),
            perm=inverse_perm,
        )

    @classmethod
    def all(cls, lattice, translations=True, flip_sets=None, perms=True):
        r"""Enumerate a subgroup, lazily.

        .. warning::
            The full group has $N^D 2^D D!$ elements --- 497,664 at $N = 6$,
            $D = 4$.  This is a generator; do not build a list of it.

        .. note::
            ``flip_sets`` is an explicit whitelist rather than a boolean
            because the admissible flips for a constrained action are neither
            all nor none.  :class:`~supervillain.action.NoIntersections`
            admits exactly the identity and the full inversion; see
            :meth:`~supervillain.action.NoIntersections.admissible_symmetries`.

        Parameters
        ----------
        lattice: supervillain.lattice.Lattice
        translations: bool
            Include the $\mathbb{Z}_N^D$ factor.
        flip_sets: iterable of tuple, optional
            Which axis-flip sets to draw from.  ``None`` means all $2^D$ of
            them; ``[()]`` means no reflections at all.
        perms: bool
            Include the $S_D$ factor.

        Yields
        ------
        Symmetry
        """
        D, N = lattice.D, lattice.N
        # These must be LISTS, not iterators: each is re-traversed once per
        # element of the outer loops, and a one-shot itertools object would be
        # exhausted after the first pass, silently yielding too few elements.
        # Only the yielded elements need to stay lazy -- N^D shift tuples is
        # 1296 at N=6, D=4, which is nothing.
        shifts = (list(itertools.product(range(N), repeat=D)) if translations
                  else [(0,) * D])
        flips = list(all_flip_sets(D) if flip_sets is None else flip_sets)
        permutations = (list(itertools.permutations(range(D))) if perms
                        else [tuple(range(D))])
        for perm in permutations:
            for flip in flips:
                for shift in shifts:
                    yield cls(lattice, shift=shift, flips=flip, perm=perm)

    @classmethod
    def draw(cls, lattice, rng, translations=True, flip_sets=None, perms=True):
        r"""Draw one element uniformly from the same set :meth:`all`
        enumerates.

        .. note::
            Every element factors UNIQUELY as translation, sign flip,
            permutation, so drawing each factor independently and uniformly
            hits each group element exactly once.

        Parameters
        ----------
        lattice: supervillain.lattice.Lattice
        rng: numpy.random.Generator
        translations: bool
            Include the $\mathbb{Z}_N^D$ factor.
        flip_sets: iterable of tuple, optional
            Which axis-flip sets to draw from.  ``None`` means all $2^D$ of
            them; ``[()]`` means no reflections at all.
        perms: bool
            Include the $S_D$ factor.

        Returns
        -------
        Symmetry
        """
        D, N = lattice.D, lattice.N
        shift = (tuple(int(s) for s in rng.integers(0, N, size=D))
                 if translations else (0,) * D)
        choices = list(all_flip_sets(D) if flip_sets is None else flip_sets)
        flip = choices[int(rng.integers(len(choices)))]
        perm = (tuple(int(m) for m in rng.permutation(D)) if perms
                else tuple(range(D)))
        return cls(lattice, shift=shift, flips=flip, perm=perm)


def all_flip_sets(D):
    r"""Every subset of the axes, as sorted tuples: $2^D$ of them."""
    return [s for k in range(D + 1) for s in itertools.combinations(range(D), k)]


class LatticeSymmetry(ReadWriteable, Generator):
    r"""Draw a lattice symmetry and apply it to every field the action
    declares.  Always accepted.

    .. note::
        $\Delta S = 0$ identically for every action that declares a nonempty
        admissible group --- see each action's own
        ``admissible_symmetries`` for why.  Detailed balance is trivial: the
        drawn-from subgroup is closed under inverses and the draw is uniform.

    .. warning::
        Because the move is always accepted there is no acceptance rate to
        reveal a bug.  A mis-implemented symmetry silently corrupts the
        ensemble while everything downstream still looks healthy.

    Parameters
    ----------
    S: supervillain.action.Villain
        Or a subclass.  Read for its lattice, its field degrees, and (via
        ``admissible_symmetries``) the space-group factors it admits.
    rng: numpy.random.Generator, optional
        A fresh default when omitted.
    """

    #: Which factors of the space group this generator lets vary, BEFORE the
    #: action's admissibility is applied.  Subclasses override this class
    #: attribute; the RESTRICTED, instance-level set lives in the read-only
    #: :attr:`factors` property instead, so the restriction cannot be undone
    #: by reassigning it after construction.
    default_factors = dict(translations=True, flip_sets=None, perms=True)

    #: Set on subclasses that are meaningless once restricted.
    refuse_if_restricted = False

    def __init__(self, S, rng=None):
        self.S = S
        self.rng = rng if rng is not None else np.random.default_rng()
        self.applied = 0
        self._degrees = None
        self._factors = self._restrict(S)

    @property
    def factors(self):
        r"""The space-group factors this generator actually draws from,
        after the action's admissibility restriction.

        .. note::
            Read-only, and backed by a private attribute computed once in
            :meth:`__init__` by :meth:`_restrict`.  A plain writable
            attribute would let ``generator.factors = {...}`` silently
            defeat the admissibility restriction after construction --- and
            since a ``LatticeSymmetry`` move is always accepted, nothing
            downstream would ever reveal that it had.  Returns a COPY of the
            private dict so mutating the returned mapping in place cannot do
            the same thing.
        """
        return dict(self._factors)

    def _restrict(self, S):
        r"""Intersect this generator's default factors with what the action
        admits.

        .. warning::
            FAILS CLOSED.  ``S`` must supply an ``admissible_symmetries()``
            method (see :meth:`supervillain.action.Villain.admissible_symmetries`
            and its overrides) --- an action that has none RAISES rather than
            receiving the full group by default.  An action whose admissible
            symmetry group has never been measured has no business inheriting
            reflections just because nothing narrowed them for it: because a
            ``LatticeSymmetry`` move is always accepted, a wrong guess here
            would corrupt the ensemble silently, with no acceptance rate ever
            to reveal it.  The fix for a new action is to establish its
            admissible group by measurement and give it an
            ``admissible_symmetries`` method, never to loosen this check.

            An inadmissible element would silently corrupt the ensemble the
            same way: the move is always accepted, so there is no acceptance
            rate to reveal that it left the constraint surface.
        """
        declare = getattr(S, 'admissible_symmetries', None)
        if declare is None:
            raise NotImplementedError(
                f'{type(S).__name__} has not declared which factors of the '
                f'hypercubic-torus space group it admits (no '
                f'admissible_symmetries() method).  Because a '
                f'LatticeSymmetry move is always accepted, assuming the full '
                f'space group for an action that has never measured its '
                f'admissible subgroup is a fail-OPEN guess that could '
                f'corrupt the ensemble silently, with no acceptance rate '
                f'ever to reveal it.  Establish {type(S).__name__}\'s '
                f'admissible group by measurement and add an '
                f'admissible_symmetries() method (see Villain, Worldline, '
                f'and NoIntersections) -- do not loosen this check.')
        admissible = declare()
        D = S.Lattice.D
        allowed = (all_flip_sets(D) if admissible['flip_sets'] is None
                   else list(admissible['flip_sets']))
        wanted = (all_flip_sets(D) if self.default_factors['flip_sets'] is None
                  else list(self.default_factors['flip_sets']))
        kept = [f for f in wanted if f in allowed]
        if self.refuse_if_restricted and len(kept) < len(wanted):
            raise ValueError(
                f'{type(self).__name__} is not available for '
                f'{type(S).__name__}: of the {len(wanted)} axis-flip sets it '
                f'exists to apply, only {len(kept)} preserve the constraint '
                f'({kept}).  Reflections are exact symmetries of the action but '
                f'not necessarily of every constraint. '
                f'Use SpaceGroup, which restricts to the admissible subgroup.')
        return dict(self.default_factors) | {'flip_sets': kept}

    @property
    def order(self):
        r"""How many distinct elements this generator draws from, after the
        action's restriction."""
        from math import factorial
        D, N = self.S.Lattice.D, self.S.Lattice.N
        return ((N ** D if self.factors['translations'] else 1)
                * len(self.factors['flip_sets'])
                * (factorial(D) if self.factors['perms'] else 1))

    def __str__(self):
        return (f'{type(self).__name__}(D={self.S.Lattice.D}, '
                f'|group|={self.order})')

    def draw(self):
        r"""Draw one element from this generator's factors."""
        return Symmetry.draw(self.S.Lattice, self.rng, **self.factors)

    def degrees(self):
        r"""``{field: form degree}``, read from the ACTION rather than guessed.

        .. note::
            Guessing from the array shape is not an option --- in $D = 4$ a
            1-form and a 3-form both have four components --- and hard-coding
            ``n``/``phi`` would silently pass an unrotated field through for
            any other formulation (the Worldline model's ``v`` is a 2-form).
        """
        if self._degrees is None:
            probe = self.S.configurations(1)[0]
            self._degrees = {k: getattr(v, 'degree', None)
                             for k, v in probe.items()}
        return self._degrees

    def step(self, configuration):
        r"""Draw an element and relabel every declared field by it.

        An entry the action does not declare --- an inline observable riding
        along in the record --- is passed through untouched.  A DECLARED field
        whose degree could not be read raises rather than passing silently.

        Parameters
        ----------
        configuration: dict

        Returns
        -------
        dict
        """
        element = self.draw()
        self.applied += 1
        degrees = self.degrees()
        out = dict(configuration)
        for key, value in configuration.items():
            if key not in degrees or value is None:
                continue
            degree = degrees[key]
            if degree is None:
                raise TypeError(
                    f'{type(self).__name__} cannot transform the declared field '
                    f'{key!r}: the action gave it no form degree, so applying '
                    f'the symmetry would silently leave it unrotated.')
            out[key] = element.apply(
                value if hasattr(value, 'degree')
                else Form(np.asarray(value), degree=degree,
                          lattice=self.S.Lattice))
        return out

    def report(self):
        r"""Names the group ACTUALLY drawn from, so a restricted generator
        never reads as though it had the full one."""
        return (f'{type(self).__name__}: {self.applied} elements applied, '
                f'drawn uniformly from a group of order {self.order} '
                f'(flip sets {self.factors["flip_sets"]})')

    def inline_observables(self, steps):
        return {}


class Translation(LatticeSymmetry):
    r"""Uniform over $\mathbb{Z}_N^D$: $N^D$ elements.  Admissible for every
    action."""
    default_factors = dict(translations=True, flip_sets=[()], perms=False)


class Reflection(LatticeSymmetry):
    r"""Uniform over the $2^D$ axis sign flips.

    .. warning::
        RAISES for an action whose admissible flip sets are a proper subset
        of all $2^D$.  A generator named for sign flips that cannot apply
        most of them is a lie.  Use :class:`SpaceGroup`, which restricts
        honestly.

    .. note::
        A reflection about the origin composed with a translation is a
        reflection about any hyperplane, so this stays the pure sign flip and
        :class:`Translation` supplies the offset.
    """
    default_factors = dict(translations=False, flip_sets=None, perms=False)
    refuse_if_restricted = True


class AxisPermutation(LatticeSymmetry):
    r"""Uniform over $S_D$: $D!$ elements.  Admissible for every action.

    .. note::
        Named for what it is.  This is neither the full point group nor the
        rotation subgroup --- half its elements are orientation-reversing.
    """
    default_factors = dict(translations=False, flip_sets=[()], perms=True)


class SpaceGroup(LatticeSymmetry):
    r"""Uniform over the largest group the action admits.

    $N^D \cdot 2^D \cdot D!$ for actions (like
    :class:`~supervillain.action.Villain` and
    :class:`~supervillain.action.Worldline`) admitting the full space group;
    $N^D \cdot 2 \cdot D!$ for
    :class:`~supervillain.action.NoIntersections`, whose constraint admits
    only the identity and the full inversion among the axis flips.

    .. note::
        A single draw rather than a composition of the three single-factor
        generators, so the applied element is inspectable.  It RESTRICTS
        rather than refusing, and ``report`` names the group actually drawn
        from, so it never claims reach it does not have.
    """
    default_factors = dict(translations=True, flip_sets=None, perms=True)


class Conjugation(ReadWriteable, Generator):
    r"""Charge conjugation $C$: negate every declared field.  Always
    accepted.

    .. note::
        $C$ acts on the FIELDS, not the lattice coordinates --- unlike every
        other generator in this module, which draws a :class:`Symmetry` and
        relabels sites.  For the Villain formulation $C$ is
        $\varphi \to -\varphi$, $n \to -n$; it has no shift, no flip, no
        permutation to draw, so it is not built from :class:`Symmetry` and
        does not subclass :class:`LatticeSymmetry`.  The group is
        $\mathbb{Z}_2 = \{1, C\}$: each :meth:`step` is a coin flip between
        leaving the configuration alone and negating everything.

    .. note::
        Admissible for EVERY action, with no whitelist, unlike
        :class:`Reflection` and :class:`SpaceGroup`.  The no-intersection
        constraint is $q = F \wedge F \equiv 0$ with $F = dn$; $q$ is
        QUADRATIC in $F$, so $n \to -n$ sends $F \to -F$ and
        $q \to (-F) \wedge (-F) = F \wedge F = q$ --- the constraint is left
        genuinely INVARIANT, not merely zero-preserving.  Contrast a spatial
        reflection, which is exact for the ACTION but breaks the constraint
        because it acts on the lattice map underlying the cup product, not on
        the field values.

    .. warning::
        Because the move is always accepted there is no acceptance rate to
        reveal a bug; see :class:`LatticeSymmetry`'s identical warning.

    Parameters
    ----------
    S: supervillain.action.Villain
        Or a subclass.  Read for its field degrees, exactly as
        :class:`LatticeSymmetry` does.
    rng: numpy.random.Generator, optional
        A fresh default when omitted.
    """

    def __init__(self, S, rng=None):
        self.S = S
        self.rng = rng if rng is not None else np.random.default_rng()
        self.applied = 0
        self._degrees = None

    def degrees(self):
        r"""``{field: form degree}``, read from the ACTION exactly as
        :meth:`LatticeSymmetry.degrees` does."""
        if self._degrees is None:
            probe = self.S.configurations(1)[0]
            self._degrees = {k: getattr(v, 'degree', None)
                             for k, v in probe.items()}
        return self._degrees

    @property
    def order(self):
        r"""$|\mathbb{Z}_2| = 2$."""
        return 2

    def __str__(self):
        return f'{type(self).__name__}(|group|={self.order})'

    def draw(self):
        r"""Coin flip: ``True`` to negate every declared field this step,
        ``False`` to leave the configuration alone."""
        return bool(self.rng.integers(2))

    def step(self, configuration):
        r"""Draw a coin flip and, if it comes up $C$, negate every declared
        field.

        An entry the action does not declare --- an inline observable riding
        along in the record --- is passed through untouched.  A DECLARED
        field whose degree could not be read raises rather than passing
        silently, whether or not this particular draw negates it: the check
        is on the field, not on the coin.

        Parameters
        ----------
        configuration: dict

        Returns
        -------
        dict
        """
        conjugate = self.draw()
        self.applied += 1
        degrees = self.degrees()
        out = dict(configuration)
        for key, value in configuration.items():
            if key not in degrees or value is None:
                continue
            degree = degrees[key]
            if degree is None:
                raise TypeError(
                    f'{type(self).__name__} cannot conjugate the declared '
                    f'field {key!r}: the action gave it no form degree, so '
                    f'negating it would silently apply to an unrecognized '
                    f'array.')
            if not conjugate:
                continue
            out[key] = -(value if hasattr(value, 'degree')
                        else Form(np.asarray(value), degree=degree,
                                  lattice=self.S.Lattice))
        return out

    def report(self):
        r"""Names the group drawn from, matching
        :meth:`LatticeSymmetry.report`."""
        return (f'{type(self).__name__}: {self.applied} elements applied, '
                f'drawn uniformly from a group of order {self.order}')

    def inline_observables(self, steps):
        return {}
