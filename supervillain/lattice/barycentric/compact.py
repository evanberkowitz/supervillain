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
