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
