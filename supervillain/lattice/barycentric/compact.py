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
from supervillain.lattice.barycentric import whitney


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
        self._mass_cache = {}

    def _split_anchor(self, anchor):
        r"""
        Split a doubled anchor into the compact layout's ``(parity, site)``.

        A component $(\pi, t)$ at site $x$ is the doubled anchor $2x + \pi$, so
        reading a doubled anchor back gives $\pi = \xi \bmod 2$ and
        $x = \lfloor \xi/2 \rfloor$.

        .. note ::
            Floor division is the point: a doubled anchor of $-1$ is parity $1$
            at site $-1$, not parity $-1$ at site $0$.  Every table in this
            class routes through here so that convention is written once.

        Parameters
        ----------
        anchor : sequence of int
            A doubled coordinate, one entry per direction.

        Returns
        -------
        tuple
            ``(parity, site)``, each a length-$D$ tuple.
        """
        doubled_anchor = tuple(int(v) for v in anchor)
        return (tuple(v % 2 for v in doubled_anchor),
                tuple(v // 2 for v in doubled_anchor))

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
                in_parity, site = self._split_anchor(
                    parity[mu] + shift[mu] for mu in range(self.D))
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
                c_parity, site = self._split_anchor(
                    parity[mu] + shift[mu] for mu in range(self.D))
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
                new_parity, offset = self._split_anchor(
                    parity[mu] + int(shift[t][mu]) for mu in range(self.D))
                out.append((self.component_index[p][(new_parity, int(image[t]))],
                            offset))
            self._half_diagonal_cache[p] = (tuple(out), int(sign))
        return self._half_diagonal_cache[p]

    def mass_table(self, p):
        r"""
        The Whitney mass matrix at degree $p$, as a compact-layout stencil.

        Every fine top simplex is an orthoscheme congruent to the reference one
        the ``whitney`` module describes, by an isometry that permutes the axes
        and flips signs so that the $k$th step of the chain becomes the
        reference's $k$th step.  That isometry carries chain vertex $k$ to
        reference vertex $k$, so once both are indexed by the *position* of a
        vertex along the chain the local matrix is literally the same array for
        all $2^D D!$ of them, and
        :func:`~supervillain.lattice.barycentric.whitney.local_mass` is called
        once.

        A $p$-face of a top simplex picks positions $k_0 < \cdots < k_p$ along
        the chain.  It is anchored at vertex $k_0$, and its type merges the step
        sets released strictly between consecutive chosen positions --- the same
        anchor-plus-merged-steps rule the coboundary uses.  Every ordered pair
        of $p$-faces of every top simplex contributes, and a pair shared by
        several top simplices accumulates once per simplex, which is what makes
        this stencil wider than the coboundary's.

        .. note ::
            The enumeration runs over top *types*, never over sites: a top
            simplex's anchor is a rank-zero barycenter, so its doubled
            coordinate is even in every direction and the whole incidence
            pattern is a translate of the one anchored at the origin.

        .. note ::
            Entries whose coefficient vanishes are **retained**: a pair of
            faces that co-occur in a top simplex but whose Whitney forms happen
            to be orthogonal there still gets a row.  That keeps the row count
            equal to the co-occurrence count the design spec states --- $81,
            125, 49, 9, 1$ in $D = 4$ --- rather than to a numerical accident.
            Roughly a sixth of the $D = 4$ degree-1 entries are zero this way,
            and not all of them exactly: cancellation leaves some at $10^{-19}$.
            :func:`hodge_star` skips them at the point of use, on a threshold
            rather than on ``if c``, so nothing pays for them per call.

        .. warning ::
            Unlike the coboundary this cannot be assembled per out-*type* and
            then routed by :meth:`_route`.  Which top simplices contain a
            $p$-simplex depends on its anchor parity and not only on its type:
            in $D = 2$ the link type releasing $+\hat e_1$ sits at chain
            positions $(0,1)$ of its two top simplices when its anchor is even,
            but at positions $(1,2)$ when the anchor is the midpoint of an
            $\hat e_0$ edge.  So the assembly is per component from the start.

        Parameters
        ----------
        p : int
            The degree.

        Returns
        -------
        tuple
            Entries ``(out_component, in_component, site shift, coefficient)``,
            sorted, with the coefficient a float.
        """
        if p not in self._mass_cache:
            D = self.D
            s = doubled._structures(D)
            local  = whitney.local_mass(D, p)
            chains = whitney.faces(D, p)
            entries = {}
            for t, steps in enumerate(s.step_types[D]):
                anchors = s.offsets[D][t]
                here = []
                for chain in chains:
                    merged = tuple(
                        tuple(sorted(sum(steps[chain[j - 1]:chain[j]], ())))
                        for j in range(1, len(chain)))
                    parity, site = self._split_anchor(anchors[chain[0]])
                    here.append((self.component_index[p][
                        (parity, s.type_index[p][merged])], site))
                for a, (out_component, out_site) in enumerate(here):
                    for b, (in_component, in_site) in enumerate(here):
                        key = (out_component, in_component,
                               tuple(in_site[mu] - out_site[mu]
                                     for mu in range(D)))
                        entries[key] = entries.get(key, 0.0) + local[a, b]
            self._mass_cache[p] = tuple(
                key + (coefficient,)
                for key, coefficient in sorted(entries.items()))
        return self._mass_cache[p]


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


def _raise_index(lat, raw, m, dtype):
    r"""
    Gather along the coboundary table: index degree $m-1 \to m$.
    """
    out = np.zeros((lat._c.count[m],) + (lat.N,) * lat.D, dtype=dtype)
    for i, terms in enumerate(lat._c.coboundary[m]):
        for src, shift, sign in terms:
            out[i] += sign * _gather(raw[src], shift)
    return out


def _lower_index(lat, raw, m, dtype):
    r"""
    Scatter along the same table, its transpose: index degree $m \to m-1$.
    """
    out = np.zeros((lat._c.count[m - 1],) + (lat.N,) * lat.D, dtype=dtype)
    for i, terms in enumerate(lat._c.coboundary[m]):
        for src, shift, sign in terms:
            out[src] += sign * _gather(raw[i], tuple(-s for s in shift))
    return out


def d(a):
    r"""
    The coboundary, in the compact layout.

    .. note ::
        The table is the doubled module's, re-indexed: the mathematics is
        established there, and what is new here is the routing between
        components.

    .. note ::
        On a dual form this dispatches to the dual coboundary, which is the
        transpose of the primal one: a dual $k$-cochain is indexed by primal
        $(D-k)$-simplices, so raising its degree lowers the index degree.
        No separate table is needed.

    Parameters
    ----------
    a : Form
        A degree-$p$ cochain, primal or dual.

    Returns
    -------
    Form
        The degree-$(p+1)$ cochain $da$.
    """
    lat = a.lattice
    P = a.degree + 1
    if P > lat.D:
        raise ValueError(f'No degree-{P} cochains in D = {lat.D}.')
    raw = np.asarray(a)
    if a.dual:
        # On the dual complex the index degree is D - degree, so raising the
        # degree lowers the index -- the transpose of the primal coboundary.
        out = _lower_index(lat, raw, lat.D - a.degree, raw.dtype)
    else:
        out = _raise_index(lat, raw, P, raw.dtype)
    return Form(out, degree=P, lattice=lat, dual=a.dual)


# ---------------------------------------------------------------------------
# Codifferential  delta : C^p -> C^{p-1}
# ---------------------------------------------------------------------------

def delta(a):
    r"""
    The codifferential, the transpose of :func:`d` with respect to the plain
    sum over components.

    .. note ::
        On a dual form this dispatches to the dual coboundary's transpose: a
        dual $k$-cochain is indexed by primal $(D-k)$-simplices, so lowering
        its degree raises the index degree.  No separate table is needed.

    .. warning ::
        The pairing carries no measure weighting.  Inserting
        :meth:`Lattice.measure` breaks adjointness, exactly as in the doubled
        layout.

    Parameters
    ----------
    a : Form
        A degree-$p$ cochain, primal or dual.

    Returns
    -------
    Form
        The degree-$(p-1)$ cochain $\delta a$.
    """
    lat = a.lattice
    P = a.degree
    if P < 1:
        raise ValueError('delta is not defined on a 0-cochain.')
    raw = np.asarray(a)
    if a.dual:
        out = _raise_index(lat, raw, lat.D - P + 1, raw.dtype)
    else:
        out = _lower_index(lat, raw, P, raw.dtype)
    return Form(out, degree=P - 1, lattice=lat, dual=a.dual)


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
# The metric Hodge star
# ---------------------------------------------------------------------------

_NEGLIGIBLE = 1e-12
r"""Below this a mass-table coefficient is orthogonality, not signal."""


def hodge_star(a):
    r"""
    The metric Hodge star, taking a primal degree-$p$ cochain to a dual
    degree-$(D-p)$ one.

    .. note ::
        This is the Whitney/Galerkin mass matrix $M_p$, whose entries are
        $\int \langle w_\sigma, w_\tau \rangle$ over Whitney forms; the
        definitions are those of :cite:`Arnold:2006zzb`.

    .. note ::
        It is **not** the diagonal ratio $|D(\sigma)|/|\sigma|$.  That is the
        Hodge star only under primal-dual orthogonality, which needs a
        circumcentric dual on a well-centred mesh :cite:`Desbrun:2005dec`.
        Every fine simplex here is an orthoscheme whose circumcentre lies on
        its boundary, so the circumcentric dual is degenerate and the
        diagonal shortcut is unavailable.

    .. note ::
        The output has the same array shape as the input: a dual $(D-p)$-cochain
        is indexed by the primal $p$-simplices whose dual blocks it lives on.

    .. warning ::
        Unlike :func:`poincare_star` this is **not** involutive: $M \neq
        M^{-1}$.  Use :func:`hodge_star_inverse` for the other direction.

    Parameters
    ----------
    a : Form
        A primal degree-$p$ cochain.

    Returns
    -------
    Form
        The dual degree-$(D-p)$ cochain $\star a$.
    """
    if a.dual:
        raise ValueError(
            'hodge_star maps primal to dual; use hodge_star_inverse on a dual form.')
    lat = a.lattice
    p = a.degree
    raw = np.asarray(a)
    out = np.zeros_like(raw, dtype=float)
    for i, j, shift, c in lat._c.mass_table(p):
        # The table keeps orthogonal co-occurring pairs so its row counts stay
        # the spec's; skip them here rather than roll and multiply by nothing.
        # A threshold, not `if c`: cancellation leaves some at ~1e-19.
        if abs(c) <= _NEGLIGIBLE:
            continue
        out[i] += c * _gather(raw[j], shift)
    return Form(out, degree=lat.D - p, lattice=lat, dual=True)


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

    .. note ::
        On a dual form the table is keyed by the **index** degree
        $m = D - k$, since the component axis is indexed by primal
        $m$-simplices regardless of which complex the form lives on; the
        output degree stays ``data.degree``.

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
        A cochain, primal or dual.
    perm : sequence of int
        A permutation of ``range(D)``.
    flips : sequence of int
        The directions whose sign is reversed.

    Returns
    -------
    Form
        The transformed cochain, of the same degree and duality.
    """
    lat  = data.lattice
    D    = lat.D
    perm = tuple(int(m) for m in perm)
    F    = tuple(sorted(set(int(m) for m in flips)))
    inverse = tuple(int(i) for i in np.argsort(np.asarray(perm)))
    raw = np.asarray(data)
    out = np.zeros_like(raw)
    m = (D - data.degree) if data.dual else data.degree
    for i, (j, rolls) in enumerate(lat._c.point_group(m, perm, F)):
        block = raw[i]
        if F:
            block = np.roll(np.flip(block, axis=F), rolls,
                            axis=tuple(range(D)))
        out[j] = np.transpose(block, axes=inverse)
    return Form(out, degree=data.degree, lattice=lat, dual=data.dual)


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
        A cochain, primal or dual.
    flips : sequence of int
        The axes to negate.

    Returns
    -------
    Form
        The reflected cochain, of the same duality.
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
        A cochain, primal or dual.
    perm : sequence of int
        A permutation of ``range(D)``.

    Returns
    -------
    Form
        The permuted cochain, of the same duality.
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
        On a primal $p$-cochain $\tau$ carries $(-1)^{p(p+1)/2}$, from
        reversing the simplex's vertex order.  $\tau$ is a translation by half
        the unit diagonal, so it preserves the *ambient* orientation.  The
        resolved Poincare convention (:func:`poincare_star`) orients each dual
        block so that $\sigma$'s orientation followed by $D(\sigma)$'s gives
        that ambient orientation; since $\tau$ cannot flip the ambient
        orientation, $D(\sigma)$'s sign under $\tau$ must transform exactly as
        $\sigma$'s does.  So the sign follows the **index** degree
        $m = D - k$ rather than the dual degree $k$ --- a component-counting
        argument alone could fix the table *index* but never a sign; this
        orientation argument is what fixes the sign.  The test
        ``test_tau_on_a_dual_form_follows_the_index_degree``, equivalently
        $\tau$ commuting with :func:`poincare_star`, is a consistency check
        that the implementation matches this derivation, not an independent
        measurement of it.

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
        A cochain, primal or dual.
    power : int
        How many half-diagonal steps.

    Returns
    -------
    Form
        The translated cochain, of the same duality.
    """
    lat = data.lattice
    p   = data.degree
    if power % 2 == 0:
        return push(data, (power // 2,) * lat.D)
    m = (lat.D - p) if data.dual else p
    table, sign = lat._c.half_diagonal(m)
    raw = np.asarray(data)
    out = np.zeros_like(raw)
    for i, (j, offset) in enumerate(table):
        out[j] = sign * np.roll(raw[i], offset, axis=tuple(range(lat.D)))
    return push(Form(out, degree=p, lattice=lat, dual=data.dual),
                ((power - 1) // 2,) * lat.D)


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
        A cochain, primal or dual.
    signs : sequence of int
        A length-$D$ vector of $\pm 1$.
    steps : int
        How many half-diagonal steps.

    Returns
    -------
    Form
        The translated cochain, of the same duality.
    """
    lat   = data.lattice
    signs = tuple(int(s) for s in signs)
    if len(signs) != lat.D or any(s not in (+1, -1) for s in signs):
        raise ValueError(f'signs must be {lat.D} entries of +1 or -1, got {signs}.')
    integer = tuple(steps * (s - 1) // 2 for s in signs)
    return push(tau(data, steps), integer)
