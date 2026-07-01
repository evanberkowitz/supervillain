#!/usr/bin/env python

import numpy as np
from supervillain.observable.topological import _topological_charge


def charge(n):
    r"""
    The topological-charge density $q = dn \wedge dn = dJ$ (with $J = n \wedge dn$)
    as a plain array carrying one integer per hypercube (shape ``(1,) + lattice.dims``).

    Delegates to :func:`supervillain.observable.topological._topological_charge`;
    ``n`` must be a :class:`~supervillain.lattice.Form` (it carries its own lattice).
    """
    return np.asarray(_topological_charge(n.lattice, n))
