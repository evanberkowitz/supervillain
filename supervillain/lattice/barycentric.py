#!/usr/bin/env python
# coding: utf-8

r"""
Differential cochains on the barycentric subdivision $S = \mathrm{sd}\,K$ of a
periodic $D$-dimensional hypercubic lattice $K$, stored on the same doubled
$(2N)^D$ array the :ref:`interlaced <interlaced>` representation uses.

The fine vertices of $S$ are the barycenters $b(F)$ of every face $F$ of every
hypercube.  A rank-$r$ barycenter has exactly $r$ half-integer coordinates, so
in *doubled* coordinates $\xi = 2x$ the fine vertices are exactly the sites of a
$(2N)^D$ array, and

.. math::

    \mathrm{rank}\, b(F) = \dim F = \#\{\mu : \xi_\mu \text{ odd}\},

which is precisely the interlaced "$p$-form site" condition: a rank-$p$
barycenter and an interlaced $p$-form site are the same array element.

A fine $p$-simplex is a flag $F_0 \subset \cdots \subset F_p$, and in doubled
coordinates a flag is a *staircase*.  From the simplex's unique lowest-rank
vertex --- its **anchor** --- each step releases a disjoint, nonempty, signed
set of directions,

.. math::

    v_i = v_{i-1} + \sum_{\mu \in \Delta_i} \sigma_\mu \hat\varepsilon_\mu,

flipping those coordinates from even to odd.  Skipped ranks are steps that
release several directions at once.  So a $p$-cochain is an array of shape
``(T_p,) + (2N,)*D``: the leading axis enumerates the **types**
$(\Delta_1, \ldots, \Delta_p)$ and the rest give the anchor.

``D`` is not a global constant.  A per-$D$ cache stores the tables, which depend
only on $D$; the $N$-dependent arrays hang off :class:`Lattice`.

Running this module as a script executes its self-tests.
"""

from functools import cached_property
from itertools import combinations, product, permutations
from math import comb, factorial
import numpy as np


# ---------------------------------------------------------------------------
# Per-D structure cache
# ---------------------------------------------------------------------------

class _Structures:
    r"""
    The $D$-only tables: which staircases exist, and where their vertices sit.

    A *type* is a tuple of $p$ steps; a *step* is a tuple of ``(direction,
    sign)`` pairs sorted by direction.  Steps are pairwise disjoint in their
    directions and each is nonempty, so a type of length $p$ touches between
    $p$ and $D$ directions.
    """

    def __init__(self, D):
        self.D = D
        self.step_types = tuple(self._build(p) for p in range(D + 1))
        self.type_index = tuple({t: i for i, t in enumerate(types)}
                                for types in self.step_types)
        self.type_count = tuple(len(types) for types in self.step_types)
        self.touched    = tuple(
            tuple(frozenset(mu for step in steps for mu, _ in step)
                  for steps in types)
            for types in self.step_types)
        self.offsets    = tuple(self._offsets(p) for p in range(D + 1))
        self.planes           = tuple(tuple(combinations(range(D), p))
                                       for p in range(D + 1))
        self.plane_index      = tuple({I: i for i, I in enumerate(pl)}
                                      for pl in self.planes)
        self.measure          = tuple(self._measure(p) for p in range(D + 1))
        self.pvector          = tuple(self._pvector(p) for p in range(D + 1))
        self.orientation_sign = self._orientation_sign()
        self.link_classes     = tuple(
            (r, s) for r in range(D + 1) for s in range(r + 1, D + 1))
        self.link_class_index = {c: i for i, c in enumerate(self.link_classes)}
        self.coboundary = tuple(self._coboundary(P) for P in range(D + 1))
        self._cup_cache = {}
        self._relabel_cache = {}
        self._tau_cache = {}

    def _build(self, p):
        """Ordered tuples of p disjoint nonempty signed direction sets."""
        if p == 0:
            return ((),)
        out = []
        for labels in product(range(p + 1), repeat=self.D):
            # label 0 means the direction is never released; label i means it
            # is released by step i.  Every step must release something.
            if not all(i in labels for i in range(1, p + 1)):
                continue
            used = [mu for mu in range(self.D) if labels[mu]]
            for signs in product((+1, -1), repeat=len(used)):
                sign_of = dict(zip(used, signs))
                out.append(tuple(
                    tuple((mu, sign_of[mu]) for mu in used if labels[mu] == i)
                    for i in range(1, p + 1)))
        return tuple(out)

    def _offsets(self, p):
        """(T_p, p+1, D) displacements of each vertex from the anchor, doubled."""
        table = np.zeros((self.type_count[p], p + 1, self.D), dtype=int)
        for t, steps in enumerate(self.step_types[p]):
            v = np.zeros(self.D, dtype=int)
            for i, step in enumerate(steps):
                for mu, sign in step:
                    v[mu] += sign
                table[t, i + 1] = v
        return table

    def _measure(self, p):
        r"""
        The $p$-volume of each type, $\sqrt{\det G}/p!$ with $G$ the Gram matrix
        of the edge vectors $v_k - v_0$, in original lattice units.

        .. note ::
            This depends on the type alone, not on the anchor, because the
            complex is translation invariant.
        """
        if p == 0:
            return np.ones(1)
        # offsets are doubled; halve them to get original units.
        W = self.offsets[p][:, 1:, :] / 2.0            # (T_p, p, D)
        G = np.einsum('tid,tjd->tij', W, W)
        return np.sqrt(np.maximum(np.linalg.det(G), 0.0)) / factorial(p)

    def _pvector(self, p):
        r"""
        The simplex's $p$-vector, component by coordinate $p$-plane.

        Entry ``[t, i]`` is the determinant of the edge vectors $v_k - v_0$
        restricted to ``planes[p][i]`` --- the Plücker coordinate of the
        $p$-plane the simplex spans, in doubled units.

        .. note ::
            A simplex whose every step releases a single direction spans a
            coordinate $p$-plane, and has exactly one nonzero component,
            $\pm 1$.  One whose steps release several directions at once spans
            a $p$-plane that is not coordinate-aligned, and has several nonzero
            components.  A zero entry is not a missing value: it says the
            simplex has no extent in that plane, which is why the components
            multiply correctly (see :meth:`Lattice.pvector`).
        """
        planes = self.planes[p]
        out = np.zeros((self.type_count[p], len(planes)), dtype=int)
        if p == 0:
            out[:, 0] = 1          # the empty determinant
            return out
        W = self.offsets[p][:, 1:, :]                      # (T_p, p, D)
        for i, I in enumerate(planes):
            out[:, i] = np.rint(np.linalg.det(W[:, :, list(I)])).astype(int)
        return out

    def _orientation_sign(self):
        r"""
        $\epsilon_\sigma = \mathrm{sign}\,\det[v_k - v_0]$ for every top simplex,
        comparing the rank orientation with the positive spacetime orientation.
        """
        W = self.offsets[self.D][:, 1:, :]             # (T_D, D, D)
        return np.rint(np.sign(np.linalg.det(W))).astype(int)

    def _coboundary(self, P):
        r"""
        For each degree-$P$ type, the faces the coboundary sums over.

        Returns a tuple indexed by out-type, each entry a tuple of
        ``(in_type_index, shift, sign)`` with ``shift`` in doubled units.
        """
        if P == 0:
            return tuple(() for _ in self.step_types[0])
        table = []
        for t, steps in enumerate(self.step_types[P]):
            terms = []
            for i in range(P + 1):
                if i == 0:
                    face  = steps[1:]
                    shift = tuple(self.offsets[P][t][1])
                elif i == P:
                    face  = steps[:-1]
                    shift = (0,) * self.D
                else:
                    merged = tuple(sorted(steps[i - 1] + steps[i]))
                    face   = steps[:i - 1] + (merged,) + steps[i + 1:]
                    shift  = (0,) * self.D
                terms.append((self.type_index[P - 1][face], shift, (-1)**i))
            table.append(tuple(terms))
        return tuple(table)

    def cup_table(self, p, q):
        r"""
        For each degree-$(p+q)$ type, which factors the cup product reads.

        Returns a tuple indexed by out-type of ``(a_type, c_type, shift)``:
        $a$ is read at the anchor from the first $p$ steps and $c$ at $v_p$
        from the last $q$, with ``shift`` in doubled units.
        """
        key = (p, q)
        if key not in self._cup_cache:
            table = []
            for t, steps in enumerate(self.step_types[p + q]):
                table.append((self.type_index[p][steps[:p]],
                              self.type_index[q][steps[p:]],
                              tuple(self.offsets[p + q][t][p])))
            self._cup_cache[key] = tuple(table)
        return self._cup_cache[key]

    def relabel(self, p, perm, flips):
        r"""
        How a signed coordinate permutation permutes the degree-$p$ types.

        ``perm`` sends direction $\mu$ to ``perm[mu]``; ``flips`` is the set of
        directions whose sign is reversed.

        .. note ::
            Rank is preserved, so the rank-ordering of every flag survives and
            no orientation sign is needed --- the whole transformation is this
            permutation of the type axis plus the array operation on the sites.

        Parameters
        ----------
        p : int
            The degree.
        perm : sequence of int
            A permutation of ``range(D)``.
        flips : sequence of int
            The directions whose sign is reversed.

        Returns
        -------
        np.ndarray
            ``image[t]`` is the index of the type that ``t`` becomes.
        """
        key = (p, tuple(perm), tuple(sorted(flips)))
        if key not in self._relabel_cache:
            flip = set(flips)
            image = np.empty(self.type_count[p], dtype=int)
            for t, steps in enumerate(self.step_types[p]):
                moved = tuple(
                    tuple(sorted((perm[mu], -sign if mu in flip else sign)
                                 for mu, sign in step))
                    for step in steps)
                image[t] = self.type_index[p][moved]
            self._relabel_cache[key] = image
        return self._relabel_cache[key]

    def half_diagonal(self, p):
        r"""
        How $\tau$ acts on degree-$p$ types.

        Returns ``(image, shift, sign)``: ``image[t]`` is the type $t$ becomes,
        ``shift[t]`` the doubled displacement of the anchor, and ``sign`` the
        common factor $(-1)^{p(p+1)/2}$ from reversing the vertex order.
        """
        if p not in self._tau_cache:
            image = np.empty(self.type_count[p], dtype=int)
            shift = np.empty((self.type_count[p], self.D), dtype=int)
            for t, steps in enumerate(self.step_types[p]):
                reversed_steps = tuple(
                    tuple(sorted((mu, -sign) for mu, sign in step))
                    for step in reversed(steps))
                image[t] = self.type_index[p][reversed_steps]
                shift[t] = self.offsets[p][t][p] + 1
            self._tau_cache[p] = (image, shift, (-1)**(p * (p + 1) // 2))
        return self._tau_cache[p]


_cache = {}

def _structures(D):
    r"""
    Return the cached :class:`_Structures` for dimension ``D``, building it on
    first call.

    Parameters
    ----------
    D : int
        The number of spacetime dimensions.

    Returns
    -------
    _Structures
        The $D$-only tables.
    """
    if D not in _cache:
        _cache[D] = _Structures(D)
    return _cache[D]


# ---------------------------------------------------------------------------
# Form  — a cochain that knows its own degree
# ---------------------------------------------------------------------------

class Form(np.ndarray):
    r"""
    A $p$-cochain on a :class:`Lattice`, of shape ``(T_p,) + (2N,)*D``.

    The leading axis enumerates the types --- the staircases
    $(\Delta_1, \ldots, \Delta_p)$ --- and the remaining axes give the anchor,
    the simplex's unique lowest-rank vertex, in doubled coordinates.

    .. note ::
        The degree is carried on the array, not inferred from its shape.  $T_p$
        does determine $p$, but relying on that would let a mis-shaped array be
        silently read as some other degree, so it is used only as an assertion.

    .. seealso ::
        :class:`supervillain.lattice.compact.Form`, whose ``degree`` /
        ``lattice`` attributes and ufunc behaviour this mirrors.

    Parameters
    ----------
    input_array : np.ndarray
        The data.
    degree : int
        The cochain degree $p$.
    lattice : Lattice
        The lattice it lives on.
    dtype : np.dtype, optional
        Passed to :func:`numpy.asarray`.
    """

    def __new__(cls, input_array, *, degree, lattice, dtype=None):
        obj = np.asarray(input_array, dtype=dtype).view(cls)
        obj.degree  = degree
        obj.lattice = lattice
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.degree  = getattr(obj, 'degree',  None)
        self.lattice = getattr(obj, 'lattice', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        forms   = [x for x in inputs if isinstance(x, Form)]
        degrees = {f.degree for f in forms}

        raw = tuple(np.asarray(x) for x in inputs)
        out = kwargs.get('out')
        if out is not None:
            kwargs['out'] = tuple(np.asarray(o) for o in out)

        result = getattr(ufunc, method)(*raw, **kwargs)

        # Re-wrap only when the degree is unambiguous and the operation was
        # element-wise (same shape), exactly as compact.Form does.
        if (len(degrees) == 1
                and isinstance(result, np.ndarray)
                and result.shape == forms[0].shape):
            return Form(result, degree=forms[0].degree, lattice=forms[0].lattice)
        return result

    def __repr__(self):
        return f'Form(degree={self.degree}, {self.lattice})'


# ---------------------------------------------------------------------------
# Lattice  — factory for barycentric cochains
# ---------------------------------------------------------------------------

class Lattice:
    r"""
    The barycentric subdivision of a periodic $D$-dimensional hypercubic
    lattice with $N$ sites per direction.

    Parameters
    ----------
    D : int
        Spacetime dimensions.
    N : int
        Original lattice sites per direction; the doubled array is $(2N)^D$.
    """

    def __init__(self, D, N):
        self.D = D
        self.N = N
        self._s = _structures(D)

    def __repr__(self):
        return f'Lattice(D={self.D}, N={self.N})'

    @cached_property
    def site_rank(self):
        r"""
        The rank of the barycenter at each site: a $(2N)^D$ integer array
        counting odd doubled coordinates.

        .. note ::
            ``site_rank == p`` is exactly the interlaced $p$-form filter.
        """
        return (np.indices((2 * self.N,) * self.D) % 2).sum(axis=0)

    @cached_property
    def _parity(self):
        return np.indices((2 * self.N,) * self.D) % 2

    @cached_property
    def _supports(self):
        """support(p) for every p; a type is valid where the directions it
        touches are all even at the anchor.

        Stored as ``int8``, not the default ``int``: the D + 1 masks together
        are (sum_p T_p) (2N)^D entries, which at D = 4, N = 6 is 282 MB as
        int64 against 35 MB as int8, and N <= 6 is the working range.  Every
        use is either a comparison (``== 0``, ``== 1``) or a multiplication
        against an int64/float64 array, which promotes."""
        out = []
        for p in range(self.D + 1):
            mask = np.ones((self._s.type_count[p],) + (2 * self.N,) * self.D,
                           dtype=np.int8)
            for t, dirs in enumerate(self._s.touched[p]):
                if dirs:
                    odd = sum(self._parity[mu] for mu in dirs)
                    mask[t] = (odd == 0).astype(np.int8)
            out.append(mask)
        return tuple(out)

    def support(self, p):
        r"""
        The 0/1 mask of real $p$-simplices, shape ``(T_p,) + (2N,)*D``.

        Parameters
        ----------
        p : int
            The degree.

        Returns
        -------
        np.ndarray
            ``int8``; 1 where the ``(type, anchor)`` slot is a fine
            $p$-simplex, else 0.  Multiplying a cochain by it promotes to the
            cochain's own dtype.
        """
        return self._supports[p]

    @cached_property
    def _counts(self):
        return tuple(int(m.sum()) for m in self._supports)

    def count(self, p):
        r"""
        The number of fine $p$-simplices on the lattice.

        Parameters
        ----------
        p : int
            The degree.

        Returns
        -------
        int
            How many there are.
        """
        return self._counts[p]

    @cached_property
    def _measures(self):
        return tuple(self._s.measure[p].reshape((-1,) + (1,) * self.D)
                     for p in range(self.D + 1))

    def measure(self, p):
        r"""
        The $p$-volume of each type --- length, area, volume, 4-volume as $p$
        runs --- shaped ``(T_p,) + (1,)*D`` so it broadcasts against a
        degree-$p$ cochain.

        .. note ::
            It depends only on the type, hence the broadcast shape;
            ``(lat.measure(p) * a).sum()`` integrates ``a``.

        Parameters
        ----------
        p : int
            The degree.

        Returns
        -------
        np.ndarray
            The measure, in original lattice units.
        """
        return self._measures[p]

    @property
    def link_classes(self):
        r"""The $(r,s)$ rank pairs labelling link classes, sorted."""
        return self._s.link_classes

    @cached_property
    def link_class(self):
        r"""
        The class of each link slot as an index into :attr:`link_classes`, with
        ``-1`` on slots that are not links.

        .. note ::
            Unlike :meth:`measure`, this is not type-only: the lower rank $r$ is
            the anchor's, so the array is shaped ``(T_1,) + (2N,)*D``.
        """
        out = np.full((self._s.type_count[1],) + (2 * self.N,) * self.D, -1, dtype=int)
        for t, steps in enumerate(self._s.step_types[1]):
            k = len(steps[0])
            valid = self._supports[1][t] == 1
            r = self.site_rank
            for (lo, hi), i in self._s.link_class_index.items():
                out[t] = np.where(valid & (r == lo) & (r + k == hi), i, out[t])
        return out

    def planes(self, p):
        r"""
        The coordinate $p$-planes, as sorted direction tuples, in the order
        :meth:`pvector` indexes them.

        Parameters
        ----------
        p : int
            The degree.

        Returns
        -------
        tuple
            $\binom{D}{p}$ tuples of directions.
        """
        return self._s.planes[p]

    def pvector(self, p):
        r"""
        The $p$-vector of each type, component by coordinate $p$-plane:
        shape ``(T_p, C(D,p))``, indexed by :meth:`planes`.

        Entry ``[t, i]`` is the Plücker coordinate of the $p$-plane the simplex
        spans --- the determinant of its edge vectors restricted to that plane.

        .. note ::
            The components **multiply**.  Splitting a top simplex's steps at any
            position gives a front face and a back face, and whenever each spans
            a single coordinate plane and the two planes are complementary,

            .. math::

                \epsilon_\sigma = v_{\mathrm{front}}\; v_{\mathrm{back}}\;
                    \sigma(I_{\mathrm{front}} \frown I_{\mathrm{back}}).

            :attr:`orientation_sign` is the $p = D$ case, where there is only
            one coordinate plane.

        .. note ::
            A simplex whose every step releases one direction spans a coordinate
            plane and has a single component, $\pm 1$.  One whose steps release
            several directions at once has several nonzero components --- it is
            not unoriented, it simply has extent in more than one plane.  A zero
            entry says the simplex has no extent in that plane, which is what
            makes the product law above come out right.

        Parameters
        ----------
        p : int
            The degree.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(T_p, C(D,p))``.
        """
        return self._s.pvector[p]

    @property
    def orientation_sign(self):
        r"""
        $\epsilon_\sigma$ for each top-simplex type, shaped to broadcast against
        a degree-$D$ cochain.

        .. note ::
            This is the sole component of the top :meth:`pvector`: a $D$-simplex
            spans the one coordinate $D$-plane there is, so its $p$-vector has a
            single entry, and that entry compares the simplex's rank ordering
            with the ambient orientation.  For $p < D$ no such comparison exists
            without choosing a complement, which is why only the top degree has
            a *sign* rather than a vector of components.
        """
        return self._s.pvector[self.D][:, 0].reshape((-1,) + (1,) * self.D)

    def zeros(self, p, dtype=float):
        r"""
        A zero $p$-cochain.

        Parameters
        ----------
        p : int
            The degree.
        dtype : np.dtype
            Element type.

        Returns
        -------
        Form
            Zeros of shape ``(T_p,) + (2N,)*D``.
        """
        return Form(np.zeros((self._s.type_count[p],) + (2 * self.N,) * self.D, dtype=dtype),
                    degree=p, lattice=self)

    def random(self, p):
        r"""
        A $p$-cochain uniform on $[0,1)$ at real $p$-simplices and zero elsewhere.

        Parameters
        ----------
        p : int
            The degree.

        Returns
        -------
        Form
            The random cochain.
        """
        data = self._supports[p] * np.random.random(
            (self._s.type_count[p],) + (2 * self.N,) * self.D)
        return Form(data, degree=p, lattice=self)

    def random_integer(self, p, low=-2, high=3):
        r"""
        A $p$-cochain of random integers at real $p$-simplices, zero elsewhere.

        Parameters
        ----------
        p : int
            The degree.
        low, high : int
            Passed to :func:`numpy.random.randint`; ``high`` is exclusive.

        Returns
        -------
        Form
            The random integer cochain.
        """
        data = self._supports[p] * np.random.randint(
            low, high, (self._s.type_count[p],) + (2 * self.N,) * self.D)
        return Form(data, degree=p, lattice=self)



# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
#
# A half-step roll is NOT a symmetry of this complex --- it carries a rank-1
# edge midpoint to a rank-2 plaquette center --- so it is available only as the
# private _gather, which d and cup use to read a neighbouring barycenter.  The
# public translations roll whole unit cells.

def _gather(array, shift):
    r"""
    ``result[xi] = array[xi + shift]``, with ``shift`` in raw **doubled**
    coordinates and the type axis (if any) left alone.

    .. warning ::
        This is not a lattice symmetry.  It is index arithmetic on a periodic
        array, for reading a neighbouring barycenter.  Use :func:`translate` for
        a translation.

    Parameters
    ----------
    array : np.ndarray
        Raw data whose LAST ``len(shift)`` axes are the site axes.
    shift : sequence of int
        One integer per direction, in doubled units.

    Returns
    -------
    np.ndarray
        The gathered array.
    """
    D = len(shift)
    axes = tuple(range(array.ndim - D, array.ndim))
    return np.roll(array, tuple(-int(s) for s in shift), axis=axes)


def push(data, shift):
    r"""
    Translate by whole unit cells: ``result[x + shift] = data[x]``.

    .. note ::
        ``shift`` is in **original lattice units**, so the doubled array is
        rolled by $2\,\text{shift}$.  This differs deliberately from
        :func:`supervillain.lattice.interlaced.push`, whose shift is raw; here a
        raw half-step is not a symmetry, so it is not offered.

    Parameters
    ----------
    data : Form
        A cochain.
    shift : sequence of int
        One integer per direction, in original lattice units.

    Returns
    -------
    Form
        The translated cochain, of the same degree.
    """
    D = data.lattice.D
    axes = tuple(range(1, D + 1))
    rolled = np.roll(np.asarray(data), tuple(2 * int(s) for s in shift), axis=axes)
    return Form(rolled, degree=data.degree, lattice=data.lattice)


def pull(data, shift):
    r"""
    Translate the other way: ``result[x] = data[x + shift]``.

    Parameters
    ----------
    data : Form
        A cochain.
    shift : sequence of int
        One integer per direction, in original lattice units.

    Returns
    -------
    Form
        The translated cochain.
    """
    return push(data, tuple(-int(s) for s in shift))


translate = push


# ---------------------------------------------------------------------------
# Exterior derivative  d : C^p -> C^{p+1}
# ---------------------------------------------------------------------------

def d(a):
    r"""
    The simplicial coboundary: on a fine $(p+1)$-simplex with vertices
    $u_0 \cdots u_{p+1}$ in increasing rank,

    .. math::

        (da)[u_0 \cdots u_{p+1}] = \sum_{i=0}^{p+1} (-1)^i\,
            a[u_0 \cdots \hat u_i \cdots u_{p+1}],

    the hat meaning omission.  On staircases, omitting an interior $u_i$ merges
    the steps $\Delta_i$ and $\Delta_{i+1}$; omitting $u_0$ deletes the first
    step and re-anchors at $v_1$; omitting the last vertex deletes the last step.

    .. note ::
        Every codimension-two face appears twice with opposite signs, so
        $d^2 = 0$ exactly, at finite spacing.

    Parameters
    ----------
    a : Form
        A degree-$p$ cochain.

    Returns
    -------
    Form
        The degree-$(p+1)$ cochain $da$.
    """
    lat = a.lattice
    P   = a.degree + 1
    if P > lat.D:
        raise ValueError(f'No degree-{P} cochains in D = {lat.D}.')
    raw = np.asarray(a)
    out = np.zeros((lat._s.type_count[P],) + (2 * lat.N,) * lat.D, dtype=raw.dtype)
    for t, terms in enumerate(lat._s.coboundary[P]):
        for source, shift, sign in terms:
            out[t] += sign * _gather(raw[source], shift)
    return Form(out * lat.support(P), degree=P, lattice=lat)


# ---------------------------------------------------------------------------
# Codifferential  delta : C^p -> C^{p-1}
# ---------------------------------------------------------------------------

def delta(a):
    r"""
    The formal adjoint of :func:`d` with respect to the diagonal pairing
    $\langle a, b \rangle = \sum_{\sigma} a_{\sigma} b_{\sigma}$ over fine simplices,
    so that $\sum (da)\cdot b = \sum a \cdot (\delta b)$.

    .. note ::
        Nothing else in this module needs $\delta$: neither :func:`d` nor
        :func:`cup` is built from it.  It is here because adjointness is the
        strongest available test that the coboundary table is right.

    Parameters
    ----------
    a : Form
        A degree-$p$ cochain.

    Returns
    -------
    Form
        The degree-$(p-1)$ cochain $\delta a$.
    """
    lat = a.lattice
    P   = a.degree
    if P < 1:
        raise ValueError('delta is not defined on a 0-cochain.')
    raw = np.asarray(a)
    out = np.zeros((lat._s.type_count[P - 1],) + (2 * lat.N,) * lat.D, dtype=raw.dtype)
    # The transpose of d: where d gathers from `source` with `shift`, delta
    # scatters back to `source` with the opposite roll.
    for t, terms in enumerate(lat._s.coboundary[P]):
        for source, shift, sign in terms:
            out[source] += sign * _gather(raw[t], tuple(-s for s in shift))
    return Form(out * lat.support(P - 1), degree=P - 1, lattice=lat)


δ = delta


# ---------------------------------------------------------------------------
# Alexander-Whitney cup product
# ---------------------------------------------------------------------------

def cup(a, c):
    r"""
    The Alexander--Whitney cup product: a $p$-cochain is evaluated
    on the first $p+1$ vertices and a $q$-cochain on the last $q+1$, so on a
    simplex in rank order

    .. math::

        (a \smile c)[v_0 \cdots v_{p+q}] = a[v_0 \cdots v_p]\, c[v_p \cdots v_{p+q}].

    .. note ::
        There is exactly **one** term per output simplex --- no sum over
        shuffles.  That is what distinguishes this from the hypercubic wedge:
        a product formed from it contributes a single term to each top
        simplex, so distinct contributions cannot cancel within one.

    .. seealso ::
        :attr:`Lattice.orientation_sign`, which a model combines with the
        cup-square to build an orientation-signed top-degree charge.  The
        lattice deliberately does not define such a charge itself: which
        product to form, and what it means, belongs to the model.

    Parameters
    ----------
    a : Form
        A degree-$p$ cochain.
    c : Form
        A degree-$q$ cochain on the same lattice.

    Returns
    -------
    Form
        The degree-$(p+q)$ cochain $a \smile c$.
    """
    lat = a.lattice
    p, q = a.degree, c.degree
    P = p + q
    if P > lat.D:
        raise ValueError(f'Degree {p} cup degree {q} exceeds D = {lat.D}.')
    ra, rc = np.asarray(a), np.asarray(c)
    out = np.zeros((lat._s.type_count[P],) + (2 * lat.N,) * lat.D,
                   dtype=np.result_type(ra.dtype, rc.dtype))
    for t, (ia, ic, shift) in enumerate(lat._s.cup_table(p, q)):
        out[t] = ra[ia] * _gather(rc[ic], shift)
    return Form(out * lat.support(P), degree=P, lattice=lat)


wedge = cup


# ---------------------------------------------------------------------------
# The point group  (Z_2)^D semidirect S_D
# ---------------------------------------------------------------------------

def reflect(data, flips):
    r"""
    Negate the axes in ``flips``.

    .. note ::
        The base-point shift is automatic, exactly as in
        :func:`supervillain.lattice.interlaced.reflect`: negating an odd doubled
        coordinate leaves it odd.  Unlike that function, **no sign is needed** ---
        a signed permutation preserves rank, hence the rank order of every flag.

    Parameters
    ----------
    data : Form
        A cochain.
    flips : sequence of int
        The axes to negate.

    Returns
    -------
    Form
        The reflected cochain.
    """
    lat = data.lattice
    F   = sorted(set(int(m) for m in flips))
    out = np.asarray(data)
    for mu in F:
        out = np.roll(np.flip(out, axis=mu + 1), 1, axis=mu + 1)
    image = lat._s.relabel(data.degree, tuple(range(lat.D)), F)
    out = out[np.argsort(image)]
    return Form(out * lat.support(data.degree), degree=data.degree, lattice=lat)


def permute(data, perm):
    r"""
    Relabel the axes, so axis $\mu$ becomes axis ``perm[mu]``.

    .. note ::
        Both the anchor and the step directions permute together, and rank is
        preserved, so --- unlike
        :func:`supervillain.lattice.interlaced.permute` --- there is no sort sign
        to apply.

    Parameters
    ----------
    data : Form
        A cochain.
    perm : sequence of int
        A permutation of ``range(D)``.

    Returns
    -------
    Form
        The permuted cochain.
    """
    lat     = data.lattice
    perm    = tuple(int(m) for m in perm)
    inverse = tuple(int(i) for i in np.argsort(np.asarray(perm)))
    out     = np.transpose(np.asarray(data), axes=(0,) + tuple(1 + i for i in inverse))
    image   = lat._s.relabel(data.degree, perm, ())
    out     = out[np.argsort(image)]
    return Form(out * lat.support(data.degree), degree=data.degree, lattice=lat)


# ---------------------------------------------------------------------------
# The half-diagonal translation
# ---------------------------------------------------------------------------

def tau(data, power=1):
    r"""
    The half-diagonal translation $\tau : x \mapsto x + \tfrac12(1,\ldots,1)$,
    applied ``power`` times.

    .. note ::
        $\tau$ exists in every $D$ because translating by $\tfrac12\mathbb{1}$
        carries the cubulation $K$ to the dual cubulation $K^*$, and
        $\mathrm{sd}\,K = \mathrm{sd}\,K^*$: dual cells share barycenters and
        reverse nesting.  That is why rank reverses, $r \mapsto D - r$, and why
        link class $(r,s)$ becomes $(D-s, D-r)$.  It is the *only* half-integer
        shift that works; the mixed ones permute the fine vertices but do not
        carry links to links.

    .. warning ::
        Because rank reverses, so does every simplex's vertex order: $\tau$
        re-anchors each simplex at the opposite end and carries
        $(-1)^{p(p+1)/2}$.  An even ``power`` is an ordinary translation and
        carries neither.

    Parameters
    ----------
    data : Form
        A cochain.
    power : int
        How many half-diagonal steps.

    Returns
    -------
    Form
        The translated cochain.
    """
    lat = data.lattice
    p   = data.degree
    if power % 2 == 0:
        return translate(data, (power // 2,) * lat.D)
    image, shift, sign = lat._s.half_diagonal(p)
    raw = np.asarray(data)
    out = np.zeros_like(raw)
    for t in range(lat._s.type_count[p]):
        out[image[t]] = np.roll(raw[t], tuple(int(s) for s in shift[t]),
                                axis=tuple(range(lat.D)))
    out = sign * out
    if power != 1:
        out = np.asarray(translate(Form(out, degree=p, lattice=lat),
                                   ((power - 1) // 2,) * lat.D))
    return Form(out * lat.support(p), degree=p, lattice=lat)


def translate_diagonal(data, signs, steps=1):
    r"""
    Translate by $\tfrac{\text{steps}}{2}(\pm 1, \ldots, \pm 1)$, for arbitrary $D$.

    .. note ::
        These are not independent symmetries: every sign pattern is $\tau$
        composed with an ordinary unit-cell translation, since
        $\tfrac12(1,\ldots,1,-1) = \tfrac12(1,\ldots,1) + (0,\ldots,0,-1)$.  The
        enlarged translation group is generated by the unit translations and
        $\tau$ alone.

    Parameters
    ----------
    data : Form
        A cochain.
    signs : sequence of int
        A length-$D$ vector of $\pm 1$.
    steps : int
        How many half-diagonal steps.

    Returns
    -------
    Form
        The translated cochain.
    """
    lat   = data.lattice
    signs = tuple(int(s) for s in signs)
    if len(signs) != lat.D or any(s not in (+1, -1) for s in signs):
        raise ValueError(f'signs must be {lat.D} entries of +1 or -1, got {signs}.')
    integer = tuple(steps * (s - 1) // 2 for s in signs)
    return translate(tau(data, steps), integer)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Test barycentric-subdivision differential cochains.')
    parser.add_argument('--D', type=int, default=4)
    parser.add_argument('--N', type=int, default=3)
    args = parser.parse_args()

    D = args.D
    N = args.N
    lat = Lattice(D, N)
    s   = _structures(D)
    print(f'Testing on {lat}\n')

    print('TYPE COUNTS')
    for p in range(D + 1):
        # Asserted against the closed form, not merely printed: a checkmark
        # beside an unchecked number reads as a verified claim.
        expected = sum((-1)**j * comb(p, j) * (2 * (p - j) + 1)**D
                       for j in range(p + 1))
        assert s.type_count[p] == expected, f'T_{p} = {s.type_count[p]} vs {expected}'
        print(f'  ✅ T_{p} = {s.type_count[p]} = Σ_j (-1)^j C({p},j) (2({p}-j)+1)^{D}')

    print('\nMEASURE INTEGRATES TO THE VOLUME')
    total = (lat.measure(D) * lat.support(D)).sum()
    assert np.isclose(total, N**D), f'{total} vs {N**D}'
    print(f'  ✅ Σ (top-simplex volumes) = N^D = {N**D}')

    print('\nNILPOTENCY OF d')
    for p in range(D - 1):
        a   = lat.random(p)
        da  = d(a)
        dda = d(da)
        assert not np.allclose(np.asarray(da), 0)
        assert np.allclose(np.asarray(dda), 0)
        print(f'  ✅ d² = 0 on {p}-cochains')

    print('\nADJOINTNESS  Σ (da)·c = Σ a·(δc)')
    for p in range(D):
        a, c = lat.random(p), lat.random(p + 1)
        lhs = float((np.asarray(d(a)) * np.asarray(c)).sum())
        rhs = float((np.asarray(a) * np.asarray(delta(c))).sum())
        assert not np.isclose(lhs, 0)
        assert np.isclose(lhs, rhs)
        print(f'  ✅ Σ (da)·c = Σ a·(δc) for p={p}')

    print('\nCUP ASSOCIATIVITY')
    for p, q, r in product(range(D), repeat=3):
        if p + q + r > D:
            continue
        a, c, e = lat.random(p), lat.random(q), lat.random(r)
        lhs = cup(cup(a, c), e)
        rhs = cup(a, cup(c, e))
        assert not np.allclose(np.asarray(lhs), 0)
        assert np.allclose(np.asarray(lhs), np.asarray(rhs))
        print(f'  ✅ (a⌣c)⌣e = a⌣(c⌣e) for {p},{q},{r}')

    print('\nLEIBNIZ RULE  d(a⌣c) = da⌣c + (-1)^p a⌣dc')
    for p, q in product(range(D + 1), repeat=2):
        if p + q + 1 > D:
            continue
        a, c = lat.random(p), lat.random(q)
        lhs = d(cup(a, c))
        rhs = cup(d(a), c) + (-1)**p * cup(a, d(c))
        assert not np.allclose(np.asarray(lhs), 0)
        assert np.allclose(np.asarray(lhs), np.asarray(rhs))
        print(f'  ✅ d(a⌣c) = da⌣c + (-1)^{p} a⌣dc for {p}⌣{q}')

    if D == 4:
        print('\nTHE CUP-SQUARE AT A PLAQUETTE CENTRE  Σ_64 = Φ(P)·Φ(P*)')

        def _plaquette_families(mu, nu, z):
            """The blue and orange triangle slots at the centre of plaquette
            (mu,nu) at z.  Mirrors test_barycentric.py's _plaquette_families;
            the two live in different files (the module and the test
            transplant to different library directories), so the reasoning
            that justifies the two non-obvious choices below travels with
            each copy rather than with only one of them.

            Returns (blue, orange), each a list of (type_index, anchor_tuple,
            sign2d), where sign2d is the orientation of the triangle inside
            its own square.

            Two fixes on top of the naive per-plane construction, both needed
            to make P . P* = +1 for every plaquette:

            - The **anchor** of a blue triangle is its own v0, reached from
              the centre xi_P (= its own v2) by subtracting that type's
              offset.  The orange triangle is different: xi_P is its own v0
              (the anchor and the centre coincide -- xi_P already has even
              coordinates in the complementary directions), so no offset
              subtraction applies there.  Subtracting it anyway reads the
              orange flux off a location outside its support -- always zero
              -- which is exactly the kind of vacuous 0 == 0 pass the
              non-degeneracy guard below exists to catch.
            - The dual (orange) family additionally needs the global
              orientation factor epsilon_{mu,nu,o1,o2} (o1 < o2 the
              complementary directions): the sign of the permutation
              (mu, nu, o1, o2) of (0, 1, 2, 3).  Without it, planes related
              by an odd permutation (e.g. (mu,nu) = (1,3)) come out with the
              wrong overall sign relative to (0,1) or (2,3).
            """
            others = [k for k in range(4) if k not in (mu, nu)]
            centre = tuple((2 * z[k] + (1 if k in (mu, nu) else 0)) % (2 * N)
                           for k in range(4))
            perm = (mu, nu) + tuple(others)
            epsilon = -1 if sum(1 for i in range(4) for j in range(i + 1, 4)
                                 if perm[i] > perm[j]) % 2 else 1

            def family(pair, anchor_is_v2, extra_sign):
                out = []
                for first, second in ((pair[0], pair[1]), (pair[1], pair[0])):
                    for sa, sb in product((+1, -1), repeat=2):
                        steps = (((first, sa),), ((second, sb),))
                        t = s.type_index[2][steps]
                        if anchor_is_v2:
                            offset = s.offsets[2][t][2]
                            anchor = tuple((centre[k] - offset[k]) % (2 * N)
                                           for k in range(4))
                        else:
                            anchor = centre
                        plane = sorted(pair)
                        W = s.offsets[2][t][1:][:, plane]
                        sign2d = int(round(np.sign(np.linalg.det(W)))) * extra_sign
                        out.append((t, anchor, sign2d))
                return out

            return (family((mu, nu), True, 1),
                    family(tuple(others), False, epsilon))

        n = lat.random_integer(1)
        f = d(n)
        # The orientation-signed cup-square of a 2-cochain.  The lattice
        # provides cup and orientation_sign; forming this particular product,
        # and calling it a charge, is a model's choice, so it is built here.
        q = Form(np.asarray(cup(f, f)) * lat.orientation_sign,
                 degree=4, lattice=lat)
        raw_f, raw_q = np.asarray(f), np.asarray(q)

        # Sample several planes and centres, exactly as
        # test_charge_factorizes_into_primal_and_dual_flux does, and require
        # at least one sampled centre to be non-degenerate (both factors
        # nonzero) -- otherwise a green checkmark here could just as well be
        # reporting 0 == 0.
        rng = np.random.default_rng(2026)
        found = None
        for mu, nu in ((0, 1), (1, 3), (2, 3)):
            for _ in range(3):
                z = tuple(int(k) for k in rng.integers(0, N, D))
                blue, orange = _plaquette_families(mu, nu, z)
                Phi_P  = sum(sg * raw_f[(t,) + xi] for t, xi, sg in blue)
                Phi_Ps = sum(sg * raw_f[(t,) + xi] for t, xi, sg in orange)
                total = 0
                for tb, xb, _ in blue:
                    for to, _, _ in orange:
                        steps = s.step_types[2][tb] + s.step_types[2][to]
                        t4 = s.type_index[4][steps]
                        total += raw_q[(t4,) + xb]
                assert total == Phi_P * Phi_Ps, f'({mu},{nu}) at {z}'
                if found is None and Phi_P != 0 and Phi_Ps != 0:
                    found = (mu, nu, z, Phi_P, Phi_Ps, total)
        assert found is not None, (
            'every sampled (Phi_P, Phi_Ps) pair had a zero factor -- the '
            'sample never exercised the non-trivial factorization, so this '
            'check would only show 0 == 0')
        mu, nu, z, Phi_P, Phi_Ps, total = found
        print(f'  ✅ Σ of the 64 signed cup-squares at a plaquette centre '
              f'= Φ(P)·Φ(P*) = {Phi_P}·{Phi_Ps} = {total}  '
              f'(plane ({mu},{nu}) at {z}, non-degenerate)')
    else:
        print('\nTHE PLAQUETTE-CENTRE THEOREM  (skipped: the cup-square is a '
              '4-cochain, so this is a D = 4 statement)')

    print('\nPOINT GROUP ORDER')
    a = lat.random(1)
    seen = set()
    for perm in permutations(range(D)):
        for bits in product((0, 1), repeat=D):
            flips = tuple(k for k in range(D) if bits[k])
            moved = permute(reflect(a, flips), perm)
            seen.add(np.asarray(moved).tobytes())
    expected = 2**D * factorial(D)
    assert len(seen) == expected, f'{len(seen)} vs {expected}'
    print(f'  ✅ |G| = 2^D D! = {expected}, measured by enumeration '
          'of distinct transformed arrays')

    print("\nTAU'S CLASS MAP")
    shifted = np.roll(lat.site_rank, (1,) * D, axis=tuple(range(D)))
    assert (shifted == D - lat.site_rank).all()
    print(f'  ✅ tau reverses rank: r → {D} - r')

    # composed, not short-circuited: tau(a, 2) takes an explicit even-power
    # branch that *returns* translate(a, (1,)*D), so comparing it against
    # that same call cannot fail.  Applying the odd branch twice can.
    a2 = lat.random(2) if D >= 2 else lat.random(D)
    once = tau(a2, 1)
    assert not np.allclose(np.asarray(once), 0)
    squared = np.asarray(tau(once, 1))
    assert np.allclose(squared, np.asarray(translate(a2, (1,) * D)))
    assert np.allclose(np.asarray(tau(a2, 2)), squared)
    print('  ✅ tau∘tau = translate by the unit diagonal (odd branch twice)')

    idx = {c: i for i, c in enumerate(lat.link_classes)}
    checked_any = False
    for source in lat.link_classes:
        marked = lat.zeros(1, dtype=float)
        np.asarray(marked)[lat.link_class == idx[source]] = 1.0
        moved = tau(marked)
        carried = np.asarray(moved) != 0
        if not carried.any():
            continue
        target = (D - source[1], D - source[0])
        assert (lat.link_class[carried] == idx[target]).all(), f'{source}->{target}'
        checked_any = True
    assert checked_any
    print('  ✅ every link class (r,s) is carried to (D-s,D-r) under tau')

    print('\nAll tests passed.')
