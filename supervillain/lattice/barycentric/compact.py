#!/usr/bin/env python
# coding: utf-8

r"""
The barycentric subdivision stored compactly: one entry per fine simplex.

:mod:`~supervillain.lattice.barycentric.doubled` keeps a degree-$p$ cochain in
a $(T_p,) + (2N,)^D$ array, of which only a shrinking fraction is ever used ---
$100\%, 19\%, 11\%, 8\%, 6\%$ for $p = 0 \ldots 4$ in $D = 4$.  Here a cochain
is one dense array instead.

A **component** is a realized (anchor parity, type) pair: a parity pattern
together with a type whose released directions all avoid it.  Anchors of a
given parity form an $N^D$ sublattice, so a component occurs exactly once per
original site, and a degree-$p$ cochain is an array of shape
``(S_p,) + (N,)*D`` at full occupancy.  Components are ordered lexicographically
by (parity bitmask, type index).
"""

from functools import cached_property
from itertools import product
import numpy as np

from supervillain.lattice.barycentric import doubled


class _CompactStructures:
    r"""
    The $D$-only tables of the compact layout: which (parity, type) pairs are
    realized, and in what order.
    """

    def __init__(self, D):
        self.D = D
        s = doubled._structures(D)
        comps = []
        for p in range(D + 1):
            here = []
            for bits in range(2 ** D):
                parity = tuple((bits >> mu) & 1 for mu in range(D))
                S = {mu for mu in range(D) if parity[mu]}
                for t, touched in enumerate(s.touched[p]):
                    if not (touched & S):
                        here.append((parity, t))
            comps.append(tuple(here))
        self.components      = tuple(comps)
        self.component_index = tuple({c: i for i, c in enumerate(cs)}
                                     for cs in comps)
        self.count           = tuple(len(cs) for cs in comps)
        self.coboundary = tuple(self._route(P, s.coboundary[P])
                                if P > 0 else ()
                                for P in range(D + 1))
        self._cup_cache = {}
        self._point_group_cache = {}
        self._half_diagonal_cache = {}

    def _route(self, p_out, table):
        r"""
        Re-index a doubled operator table into the compact layout.

        A component $(\pi, t)$ at site $x$ is the doubled anchor $2x + \pi$, so a
        doubled shift $\delta$ lands at parity $(\pi + \delta) \bmod 2$ and site
        $x + \lfloor (\pi + \delta)/2 \rfloor$.

        .. note ::
            The site offset lies in $\{-1, 0\}$ per direction, so the stencil
            reaches the $2^D$ neighbouring sites.  A component is realized only
            where every direction its type releases is even at the anchor, and a
            shift is nonzero only in released directions, so $\pi + \delta$ never
            reaches $2$ --- the only route to a $+1$ offset.

        Parameters
        ----------
        p_out : int
            Degree of the output.
        table : tuple
            The doubled table: per out-type, ``(in_type, doubled shift, sign)``.

        Returns
        -------
        tuple
            Per out-component, ``(in_component, site shift, sign)``.
        """
        p_in = p_out - 1
        out = []
        for parity, t in self.components[p_out]:
            terms = []
            for src, shift, sign in table[t]:
                total = [parity[mu] + shift[mu] for mu in range(self.D)]
                in_parity = tuple(v % 2 for v in total)
                site      = tuple(v // 2 for v in total)
                terms.append((self.component_index[p_in][(in_parity, src)],
                              site, sign))
            out.append(tuple(terms))
        return tuple(out)

    def cup_table(self, p, q):
        r"""
        Re-index the doubled cup table into the compact layout.

        Parameters
        ----------
        p : int
            Degree of the left factor.
        q : int
            Degree of the right factor.

        Returns
        -------
        tuple
            Per out-component, ``(a_component, c_component, site shift)``.
        """
        key = (p, q)
        if key not in self._cup_cache:
            s = doubled._structures(self.D)
            table = s.cup_table(p, q)
            out = []
            for parity, t in self.components[p + q]:
                ia, ic, shift = table[t]
                total = [parity[mu] + shift[mu] for mu in range(self.D)]
                c_parity = tuple(v % 2 for v in total)
                site     = tuple(v // 2 for v in total)
                out.append((self.component_index[p][(parity, ia)],
                            self.component_index[q][(c_parity, ic)],
                            site))
            self._cup_cache[key] = tuple(out)
        return self._cup_cache[key]

    def point_group(self, p, perm, flips):
        r"""
        Re-index a signed coordinate permutation into the compact layout.

        The element sends direction $\mu$ to ``perm[mu]``, negating it if
        $\mu \in$ ``flips``.  On anchors it is $\xi_{\text{perm}[\mu]} =
        \epsilon_\mu \xi_\mu$, and writing $\xi = 2x + \pi$,

        .. math::

            \pi'_{\text{perm}[\mu]} = \pi_\mu,
            \qquad
            y_{\text{perm}[\mu]} = \begin{cases}
                x_\mu            & \epsilon_\mu = +1 \\
                -x_\mu - \pi_\mu & \epsilon_\mu = -1.
            \end{cases}

        .. note ::
            The parity pattern is carried along unchanged --- negating an odd
            coordinate leaves it odd --- so the component moves but its parity
            only permutes.  The site action is a *flip* of each negated axis
            followed by a per-component roll of $1 - \pi_\mu$, and then the
            transpose; it is **not** a translation, so it cannot be folded into
            the roll-only routing that :meth:`_route` uses.

        .. warning ::
            ``rolls`` are the roll amounts to apply *after* flipping, and they
            depend on the component's own parity.  A single lattice-wide roll
            would be wrong.

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
        tuple
            Per source component, ``(target component, rolls)``, with ``rolls``
            a length-$D$ tuple that is $0$ in the unflipped directions.
        """
        perm = tuple(int(m) for m in perm)
        F    = tuple(sorted(set(int(m) for m in flips)))
        key  = (p, perm, F)
        if key not in self._point_group_cache:
            image = doubled._structures(self.D).relabel(p, perm, F)
            out   = []
            for parity, t in self.components[p]:
                new_parity = [0] * self.D
                for mu in range(self.D):
                    new_parity[perm[mu]] = parity[mu]
                rolls = tuple(1 - parity[mu] if mu in F else 0
                              for mu in range(self.D))
                out.append((self.component_index[p][(tuple(new_parity),
                                                     int(image[t]))], rolls))
            self._point_group_cache[key] = tuple(out)
        return self._point_group_cache[key]

    def half_diagonal(self, p):
        r"""
        Re-index $\tau$, the half-diagonal translation, into the compact layout.

        .. note ::
            Unlike the point group this *is* a translation of the sites, so
            each component carries a plain per-component roll: the doubled
            anchor moves by ``shift[t]``, and $2x + \pi + \text{shift}$ has
            parity $(\pi + \text{shift}) \bmod 2$ at site
            $x + \lfloor (\pi + \text{shift})/2 \rfloor$.

        Parameters
        ----------
        p : int
            The degree.

        Returns
        -------
        tuple
            ``(table, sign)``, where ``table[i]`` is
            ``(target component, site shift)`` and ``sign`` is the common
            $(-1)^{p(p+1)/2}$ from reversing the vertex order.
        """
        if p not in self._half_diagonal_cache:
            image, shift, sign = doubled._structures(self.D).half_diagonal(p)
            out = []
            for parity, t in self.components[p]:
                total = [parity[mu] + int(shift[t][mu]) for mu in range(self.D)]
                new_parity = tuple(v % 2 for v in total)
                offset     = tuple(v // 2 for v in total)
                out.append((self.component_index[p][(new_parity, int(image[t]))],
                            offset))
            self._half_diagonal_cache[p] = (tuple(out), int(sign))
        return self._half_diagonal_cache[p]


_cache = {}

def _compact_structures(D):
    r"""
    Return the cached :class:`_CompactStructures` for dimension ``D``.

    Parameters
    ----------
    D : int
        The number of spacetime dimensions.

    Returns
    -------
    _CompactStructures
        The $D$-only compact tables.
    """
    if D not in _cache:
        _cache[D] = _CompactStructures(D)
    return _cache[D]


# ---------------------------------------------------------------------------
# Form  — a cochain that knows its own degree and duality
# ---------------------------------------------------------------------------

class Form(np.ndarray):
    r"""
    A cochain on the compact barycentric lattice.

    .. note ::
        The component axis is always indexed by **primal** simplices.  A primal
        degree-$p$ form has $S_p$ components; a dual degree-$k$ form is indexed
        by the $(D-k)$-simplices whose dual blocks it lives on, so it has
        $S_{D-k}$ of them.

    .. warning ::
        A primal and a dual form can have *identical shapes* --- a primal
        $2$-form and a dual $2$-form in $D = 4$ both have $S_2$ components ---
        so numpy would happily add them.  Arithmetic therefore requires
        agreement on ``(degree, dual)``, and a mixed pair degrades to a plain
        ``ndarray`` rather than pretending to be a ``Form``.

    Parameters
    ----------
    input_array : np.ndarray
        The data.
    degree : int
        The dimension of the cell each number lives on.
    lattice : Lattice
        The lattice it lives on.
    dual : bool
        Whether the cells are the dual blocks rather than the simplices.
    dtype : np.dtype, optional
        Passed to :func:`numpy.asarray`.
    """

    def __new__(cls, input_array, *, degree, lattice, dual=False, dtype=None):
        obj = np.asarray(input_array, dtype=dtype).view(cls)
        obj.degree  = degree
        obj.lattice = lattice
        obj.dual    = dual
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.degree  = getattr(obj, 'degree',  None)
        self.lattice = getattr(obj, 'lattice', None)
        self.dual    = getattr(obj, 'dual',    None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        forms = [x for x in inputs if isinstance(x, Form)]
        kinds = {(f.degree, f.dual) for f in forms}

        raw = tuple(np.asarray(x) for x in inputs)
        out = kwargs.get('out')
        if out is not None:
            kwargs['out'] = tuple(np.asarray(o) for o in out)

        result = getattr(ufunc, method)(*raw, **kwargs)

        if (len(kinds) == 1
                and isinstance(result, np.ndarray)
                and result.shape == forms[0].shape):
            return Form(result, degree=forms[0].degree,
                        lattice=forms[0].lattice, dual=forms[0].dual)
        return result

    def __repr__(self):
        kind = 'dual' if self.dual else 'primal'
        return f'Form({kind}, degree={self.degree}, shape={self.shape}, {self.lattice})'

    def to_doubled(self):
        r"""
        The same cochain in the
        :mod:`~supervillain.lattice.barycentric.doubled` layout.

        .. note ::
            Component $(\pi, t)$ at site $x$ is the doubled slot of type $t$ at
            anchor $2x + \pi$, so this is a strided scatter and loses nothing.

        .. warning ::
            Primal forms only.  The doubled module has no notion of a dual
            cochain, so there is nothing to convert a dual form *to*.

        Returns
        -------
        supervillain.lattice.barycentric.doubled.Form
            The same numbers, on the doubled array.
        """
        if self.dual:
            raise ValueError(
                'to_doubled is for primal forms; the doubled module has no '
                'dual cochains to convert to.')
        lat = self.lattice
        D, N, p = lat.D, lat.N, self.degree
        out = np.zeros((lat._s.type_count[p],) + (2 * N,) * D,
                       dtype=np.asarray(self).dtype)
        raw = np.asarray(self)
        for i, (parity, t) in enumerate(lat._c.components[p]):
            index = (t,) + tuple(slice(parity[mu], None, 2) for mu in range(D))
            out[index] = raw[i]
        return doubled.Form(out, degree=p, lattice=lat._doubled)

    @classmethod
    def from_doubled(cls, f, lattice):
        r"""
        A compact form from a
        :mod:`~supervillain.lattice.barycentric.doubled` one.

        Parameters
        ----------
        f : supervillain.lattice.barycentric.doubled.Form
            The doubled cochain.
        lattice : Lattice
            The compact lattice to build on.

        Returns
        -------
        Form
            The same numbers, compactly.
        """
        D, N, p = lattice.D, lattice.N, f.degree
        raw = np.asarray(f)
        out = np.zeros((lattice._c.count[p],) + (N,) * D, dtype=raw.dtype)
        for i, (parity, t) in enumerate(lattice._c.components[p]):
            index = (t,) + tuple(slice(parity[mu], None, 2) for mu in range(D))
            out[i] = raw[index]
        return cls(out, degree=p, lattice=lattice)


# ---------------------------------------------------------------------------
# Lattice  — factory for compact cochains
# ---------------------------------------------------------------------------

class Lattice:
    r"""
    The barycentric subdivision of a periodic $D$-dimensional hypercubic
    lattice with $N$ sites per direction, stored compactly.

    .. seealso ::
        :class:`supervillain.lattice.barycentric.doubled.Lattice`, the obvious
        implementation this one is validated against.

    Parameters
    ----------
    D : int
        Spacetime dimensions.
    N : int
        Original lattice sites per direction.
    """

    def __init__(self, D, N):
        self.D = D
        self.N = N
        self._c = _compact_structures(D)
        self._s = doubled._structures(D)

    def __repr__(self):
        return f'Lattice(D={self.D}, N={self.N})'

    @cached_property
    def _doubled(self):
        r"""The matching doubled lattice, built once and reused by the converters."""
        return doubled.Lattice(self.D, self.N)

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
            $S_p N^D$.
        """
        return self._c.count[p] * self.N ** self.D

    @cached_property
    def _measures(self):
        return tuple(
            np.array([self._s.measure[p][t] for _, t in self._c.components[p]]
                     ).reshape((-1,) + (1,) * self.D)
            for p in range(self.D + 1))

    def measure(self, p):
        r"""
        The $p$-volume of each component, shaped to broadcast against a
        degree-$p$ cochain.

        Parameters
        ----------
        p : int
            The degree.

        Returns
        -------
        np.ndarray
            Shape ``(S_p,) + (1,)*D``, in original lattice units.
        """
        return self._measures[p]

    def planes(self, p):
        r"""
        The coordinate $p$-planes, as sorted direction tuples, in the order
        :meth:`pvector` indexes them.

        .. note ::
            The planes depend only on $D$ and $p$, not on the layout, so this
            is the same tuple the doubled module reports.

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

    @cached_property
    def _pvectors(self):
        return tuple(
            np.array([self._s.pvector[p][t] for _, t in self._c.components[p]])
            for p in range(self.D + 1))

    def pvector(self, p):
        r"""
        The $p$-vector of each component, component by coordinate $p$-plane:
        shape ``(S_p, C(D,p))``, indexed by :meth:`planes`.

        Entry ``[i, j]`` is the Plucker coordinate of the $p$-plane the simplex
        spans --- the determinant of its edge vectors restricted to that plane.

        .. note ::
            A component's $p$-vector is a property of its type alone: the
            anchor parity says where a simplex sits, not which way it points.
            So this is the doubled module's table gathered onto the component
            list, with no re-indexing of the site axes.

        .. note ::
            :attr:`orientation_sign` is the $p = D$ case, where there is only
            one coordinate plane, and equals ``pvector(D)[:, 0]``.

        Parameters
        ----------
        p : int
            The degree.

        Returns
        -------
        np.ndarray
            Shape ``(S_p, C(D,p))``.
        """
        return self._pvectors[p]

    @property
    def link_classes(self):
        r"""The $(r,s)$ rank pairs labelling link classes, sorted."""
        return self._s.link_classes

    @cached_property
    def link_class(self):
        r"""
        The class of each link component as an index into :attr:`link_classes`.

        .. note ::
            Unlike the doubled layout there is no ``-1``: every component is a
            link, so every entry is a real class.
        """
        out = np.empty(self._c.count[1], dtype=int)
        for i, (parity, t) in enumerate(self._c.components[1]):
            r = sum(parity)
            k = len(self._s.step_types[1][t][0])
            out[i] = self._s.link_class_index[(r, r + k)]
        return out

    @property
    def orientation_sign(self):
        r"""$\epsilon_\sigma$ for each top component, shaped to broadcast."""
        return np.array([self._s.pvector[self.D][t][0]
                         for _, t in self._c.components[self.D]]
                        ).reshape((-1,) + (1,) * self.D)

    def _shape(self, p, dual):
        index_degree = (self.D - p) if dual else p
        return (self._c.count[index_degree],) + (self.N,) * self.D

    def zeros(self, p, dtype=float, dual=False):
        r"""
        A zero cochain.

        Parameters
        ----------
        p : int
            The degree.
        dtype : np.dtype
            Element type.
        dual : bool
            Whether it lives on the dual blocks.

        Returns
        -------
        Form
            Zeros of the appropriate shape.
        """
        return Form(np.zeros(self._shape(p, dual), dtype=dtype),
                    degree=p, lattice=self, dual=dual)

    def random(self, p, dual=False):
        r"""
        A cochain uniform on $[0,1)$.

        .. note ::
            Every entry is nonzero: unlike the doubled layout there is nothing
            to mask, since every slot is a simplex.

        Parameters
        ----------
        p : int
            The degree.
        dual : bool
            Whether it lives on the dual blocks.

        Returns
        -------
        Form
            The random cochain.
        """
        return Form(np.random.random(self._shape(p, dual)),
                    degree=p, lattice=self, dual=dual)

    def random_integer(self, p, low=-2, high=3, dual=False):
        r"""
        A cochain of random integers.

        Parameters
        ----------
        p : int
            The degree.
        low, high : int
            Passed to :func:`numpy.random.randint`; ``high`` is exclusive.
        dual : bool
            Whether it lives on the dual blocks.

        Returns
        -------
        Form
            The random integer cochain.
        """
        return Form(np.random.randint(low, high, self._shape(p, dual)),
                    degree=p, lattice=self, dual=dual)


# ---------------------------------------------------------------------------
# Exterior derivative  d : C^p -> C^{p+1}
# ---------------------------------------------------------------------------

def _gather(array, shift):
    r"""
    ``result[x] = array[x + shift]`` on the site axes, periodically.

    Parameters
    ----------
    array : np.ndarray
        Site-indexed data.
    shift : sequence of int
        One integer per direction, in original lattice units.

    Returns
    -------
    np.ndarray
        The gathered array.
    """
    return np.roll(array, tuple(-int(s) for s in shift),
                   axis=tuple(range(array.ndim - len(shift), array.ndim)))


def d(a):
    r"""
    The coboundary, in the compact layout.

    .. note ::
        The table is the doubled module's, re-indexed: the mathematics is
        established there, and what is new here is the routing between
        components.

    Parameters
    ----------
    a : Form
        A primal degree-$p$ cochain.

    Returns
    -------
    Form
        The degree-$(p+1)$ cochain $da$.
    """
    if a.dual:
        raise ValueError('d on a dual form is Phase 2; this is a primal operator.')
    lat = a.lattice
    P = a.degree + 1
    if P > lat.D:
        raise ValueError(f'No degree-{P} cochains in D = {lat.D}.')
    raw = np.asarray(a)
    out = np.zeros((lat._c.count[P],) + (lat.N,) * lat.D, dtype=raw.dtype)
    for i, terms in enumerate(lat._c.coboundary[P]):
        for src, shift, sign in terms:
            out[i] += sign * _gather(raw[src], shift)
    return Form(out, degree=P, lattice=lat)


# ---------------------------------------------------------------------------
# Codifferential  delta : C^p -> C^{p-1}
# ---------------------------------------------------------------------------

def delta(a):
    r"""
    The codifferential, the transpose of :func:`d` with respect to the plain
    sum over components.

    .. warning ::
        The pairing carries no measure weighting.  Inserting
        :meth:`Lattice.measure` breaks adjointness, exactly as in the doubled
        layout.

    Parameters
    ----------
    a : Form
        A primal degree-$p$ cochain.

    Returns
    -------
    Form
        The degree-$(p-1)$ cochain $\delta a$.
    """
    if a.dual:
        raise ValueError('delta on a dual form is Phase 2; this is a primal operator.')
    lat = a.lattice
    P = a.degree
    if P < 1:
        raise ValueError('delta is not defined on a 0-cochain.')
    raw = np.asarray(a)
    out = np.zeros((lat._c.count[P - 1],) + (lat.N,) * lat.D, dtype=raw.dtype)
    for i, terms in enumerate(lat._c.coboundary[P]):
        for src, shift, sign in terms:
            out[src] += sign * _gather(raw[i], tuple(-s for s in shift))
    return Form(out, degree=P - 1, lattice=lat)


δ = delta


# ---------------------------------------------------------------------------
# Alexander-Whitney cup product
# ---------------------------------------------------------------------------

def cup(a, c):
    r"""
    The Alexander--Whitney cup product, in the compact layout.

    There is exactly one term per output simplex --- no sum over shuffles ---
    as in the doubled module.

    .. warning ::
        Refused on dual forms.  Alexander--Whitney is a simplicial
        construction and the dual block complex is not simplicial, so there is
        no dual cup to dispatch to.

    Parameters
    ----------
    a : Form
        A primal degree-$p$ cochain.
    c : Form
        A primal degree-$q$ cochain on the same lattice.

    Returns
    -------
    Form
        The degree-$(p+q)$ cochain $a \smile c$.
    """
    if a.dual or c.dual:
        raise ValueError(
            'cup is refused on dual forms: Alexander-Whitney is simplicial and '
            'the dual block complex is not, so there is no dual cup.')
    lat = a.lattice
    p, q = a.degree, c.degree
    P = p + q
    if P > lat.D:
        raise ValueError(f'Degree {p} cup degree {q} exceeds D = {lat.D}.')
    ra, rc = np.asarray(a), np.asarray(c)
    out = np.zeros((lat._c.count[P],) + (lat.N,) * lat.D,
                   dtype=np.result_type(ra.dtype, rc.dtype))
    for i, (ia, ic, shift) in enumerate(lat._c.cup_table(p, q)):
        out[i] = ra[ia] * _gather(rc[ic], shift)
    return Form(out, degree=P, lattice=lat)


wedge = cup


# ---------------------------------------------------------------------------
# Poincare duality
# ---------------------------------------------------------------------------

def poincare_star(a):
    r"""
    The Poincare duality map, exchanging a degree-$p$ cochain for a dual
    degree-$(D-p)$ one.

    .. note ::
        This **moves no data** and costs nothing.  A dual $(D-p)$-cochain
        holds one number per primal $p$-simplex, so it is indexed by the same
        component axis the input already uses; all this does is relabel the
        type.  Its job is to tell :func:`d` and :func:`delta` which complex
        they are on.

    .. note ::
        It carries **no sign**, and $\star\star = +1$ exactly --- not the
        continuum's $(-1)^{p(D-p)}$.  Each dual block is oriented so that
        $\sigma$'s orientation followed by $D(\sigma)$'s gives the ambient
        orientation, which absorbs that sign.  The metric star is where the
        continuum sign lives; see :func:`hodge_star`.

    .. note ::
        This is Poincare duality, a statement of topology, and is metric-free.
        It is emphatically not the Hodge star, which needs an inner product.

    Parameters
    ----------
    a : Form
        A cochain, primal or dual.

    Returns
    -------
    Form
        The same numbers, of degree $D - $ ``a.degree``, with ``.dual``
        toggled.
    """
    lat = a.lattice
    return Form(np.array(np.asarray(a), copy=True),
                degree=lat.D - a.degree, lattice=lat, dual=not a.dual)


# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------

def push(data, shift):
    r"""
    Translate by whole unit cells: ``result[x + shift] = data[x]``.

    .. note ::
        A whole-cell translation leaves the component alone --- it changes
        neither the anchor parity nor the type --- so this is a plain roll of
        the site axes, with no re-indexing.  It is the one symmetry the compact
        layout gets for free.

    .. warning ::
        ``shift`` is in original lattice units, as in
        :func:`supervillain.lattice.barycentric.doubled.push`; the compact site
        axes are already in those units, so it is *not* doubled here.

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
    rolled = np.roll(np.asarray(data), tuple(int(s) for s in shift),
                     axis=tuple(range(1, D + 1)))
    return Form(rolled, degree=data.degree, lattice=data.lattice, dual=data.dual)


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
# The point group  (Z_2)^D semidirect S_D
# ---------------------------------------------------------------------------

def _point_group(data, perm, flips):
    r"""
    Apply a signed coordinate permutation, using
    :meth:`_CompactStructures.point_group`.

    .. note ::
        The table is static --- it depends on $D$ and the group element, never
        on $N$ --- and it is built by re-indexing the doubled module's
        ``relabel``, exactly as :func:`d` re-indexes its coboundary.

    .. warning ::
        Unlike a translation, the site action here is a flip and a transpose,
        so each component's block is reshaped rather than merely rolled.

    .. seealso ::
        The element is $\xi_{\text{perm}[\mu]} = \epsilon_\mu \xi_\mu$, so
        a general one is :func:`permute` *after* :func:`reflect`; the two public
        entry points use only the pure-permutation and pure-flip branches.

    Parameters
    ----------
    data : Form
        A primal cochain.
    perm : sequence of int
        A permutation of ``range(D)``.
    flips : sequence of int
        The directions whose sign is reversed.

    Returns
    -------
    Form
        The transformed cochain, of the same degree.
    """
    if data.dual:
        raise ValueError(
            'The point group on a dual form is Phase 2; these are primal '
            'operators.')
    lat  = data.lattice
    D    = lat.D
    perm = tuple(int(m) for m in perm)
    F    = tuple(sorted(set(int(m) for m in flips)))
    inverse = tuple(int(i) for i in np.argsort(np.asarray(perm)))
    raw = np.asarray(data)
    out = np.zeros_like(raw)
    for i, (j, rolls) in enumerate(lat._c.point_group(data.degree, perm, F)):
        block = raw[i]
        if F:
            block = np.roll(np.flip(block, axis=F), rolls,
                            axis=tuple(range(D)))
        out[j] = np.transpose(block, axes=inverse)
    return Form(out, degree=data.degree, lattice=lat)


def reflect(data, flips):
    r"""
    Negate the axes in ``flips``.

    .. note ::
        No sign is needed, exactly as in
        :func:`supervillain.lattice.barycentric.doubled.reflect`: a signed
        permutation preserves rank, hence the rank order of every flag.

    Parameters
    ----------
    data : Form
        A primal cochain.
    flips : sequence of int
        The axes to negate.

    Returns
    -------
    Form
        The reflected cochain.
    """
    return _point_group(data, tuple(range(data.lattice.D)), flips)


def permute(data, perm):
    r"""
    Relabel the axes, so axis $\mu$ becomes axis ``perm[mu]``.

    .. note ::
        Both the anchor parity and the step directions permute together, so the
        component's parity pattern is permuted rather than changed, and the site
        block is transposed with no shift at all.

    Parameters
    ----------
    data : Form
        A primal cochain.
    perm : sequence of int
        A permutation of ``range(D)``.

    Returns
    -------
    Form
        The permuted cochain.
    """
    return _point_group(data, perm, ())


# ---------------------------------------------------------------------------
# The half-diagonal translation
# ---------------------------------------------------------------------------

def tau(data, power=1):
    r"""
    The half-diagonal translation $\tau : x \mapsto x + \tfrac12(1,\ldots,1)$,
    applied ``power`` times.

    .. note ::
        An even ``power`` is a whole-cell translation, so it short-circuits to
        :func:`push`.  An odd one moves every component: the anchor parity
        flips wherever $\tau$'s doubled shift is odd, which is precisely how a
        half-step becomes visible in a layout whose sites are whole cells.

    .. warning ::
        Because rank reverses, so does every simplex's vertex order, and an odd
        power carries $(-1)^{p(p+1)/2}$ --- see
        :func:`supervillain.lattice.barycentric.doubled.tau`.

    Parameters
    ----------
    data : Form
        A primal cochain.
    power : int
        How many half-diagonal steps.

    Returns
    -------
    Form
        The translated cochain.
    """
    if data.dual:
        raise ValueError(
            "tau on a dual form is Phase 2: the dual blocks' orientation sign "
            'under rank reversal is not settled here.')
    lat = data.lattice
    p   = data.degree
    if power % 2 == 0:
        return push(data, (power // 2,) * lat.D)
    table, sign = lat._c.half_diagonal(p)
    raw = np.asarray(data)
    out = np.zeros_like(raw)
    for i, (j, offset) in enumerate(table):
        out[j] = sign * np.roll(raw[i], offset, axis=tuple(range(lat.D)))
    return push(Form(out, degree=p, lattice=lat), ((power - 1) // 2,) * lat.D)


def translate_diagonal(data, signs, steps=1):
    r"""
    Translate by $\tfrac{\text{steps}}{2}(\pm 1, \ldots, \pm 1)$.

    .. note ::
        These are not independent symmetries: every sign pattern is $\tau$
        composed with a whole-cell translation, since
        $\tfrac12(1,\ldots,1,-1) = \tfrac12(1,\ldots,1) + (0,\ldots,0,-1)$.

    Parameters
    ----------
    data : Form
        A primal cochain.
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
    return push(tau(data, steps), integer)
