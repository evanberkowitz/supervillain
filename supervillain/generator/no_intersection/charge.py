#!/usr/bin/env python

from itertools import combinations
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


def wedge_pairs(lattice):
    r"""
    For the single 4-form component $(0,1,2,3)$, the six ordered complementary
    plaquette pairs $(A, B)$ with the sign $\sigma(A\frown B)$, matching
    :func:`supervillain.lattice.wedge`,

    .. math::
        (a\wedge b)_{(0,1,2,3)}[x] = \sum \sigma(A\frown B)\, a_A[x]\, b_B[x+\hat e_A].

    Returns a tuple of ``(A_index, A_directions, B_index, sign)``.
    """
    pairs = []
    for A in combinations(range(4), 2):
        B = tuple(k for k in range(4) if k not in A)
        sign = (-1) ** sum(1 for k in A for j in B if j < k)
        pairs.append((lattice.comp_index[2][A], A, lattice.comp_index[2][B], sign))
    return tuple(pairs)


def dF_entries(lattice, change):
    r"""
    The plaquette changes $\Delta F = d(\Delta n)$ of a sparse link change (a dict
    ``(direction, *site) -> coefficient``), as a dict ``(component_index, site) ->
    coefficient``.  A link $n_\mu[s] \mathrel{+}= c$ changes the plaquettes
    $(a, \mu)$ with $a < \mu$ by $+c$ at $s - \hat e_a$ and $-c$ at $s$, and the
    plaquettes $(\mu, b)$ with $\mu < b$ by $-c$ at $s - \hat e_b$ and $+c$ at $s$,
    matching :func:`supervillain.lattice.d`.
    """
    N = lattice.N
    dF = {}

    def add(comp, site, value):
        key = (lattice.comp_index[2][comp], site)
        dF[key] = dF.get(key, 0) + value

    for (mu, *s), c in change.items():
        for nu in range(4):
            if nu == mu:
                continue
            comp = (nu, mu) if nu < mu else (mu, nu)
            sign = +1 if nu < mu else -1
            back = tuple((s[k] - (k == nu)) % N for k in range(4))
            add(comp, back, sign * c)
            add(comp, tuple(s), -sign * c)
    return {key: v for key, v in dF.items() if v != 0}


def local_dq(lattice, F, change, pairs=None):
    r"""
    The change of the charge density $q = F\wedge F$ from a sparse link change,
    computed locally:

    .. math::
        \Delta q = \Delta F\wedge F + F\wedge\Delta F + \Delta F\wedge\Delta F,
        \qquad \Delta F = d(\Delta n),

    where the wedge follows :func:`supervillain.lattice.wedge`,
    $(a\wedge b)[x] = \sum \sigma(A\frown B)\, a_A[x]\, b_B[x+\hat e_A]$.
    Returns a dict ``site -> change`` with zero entries dropped; an empty dict
    means the change preserves $q$ everywhere.

    ``F`` is the *current* plain integer array $d(n)$ of shape ``(6, N, N, N, N)``
    and ``change`` a dict ``(direction, *site) -> coefficient``.  The cost is
    $O(1)$ in the lattice volume.
    """
    N = lattice.N
    if pairs is None:
        pairs = wedge_pairs(lattice)
    dF = dF_entries(lattice, change)
    dq = {}

    def add(site, value):
        dq[site] = dq.get(site, 0) + value

    for (idx, site), v in dF.items():
        for A_idx, A_dirs, B_idx, sign in pairs:
            if idx == A_idx:
                # ΔF_A[x] (F_B + ΔF_B)[x+ê_A]: the ΔF∧F and ΔF∧ΔF terms together.
                ahead = tuple((site[k] + (k in A_dirs)) % N for k in range(4))
                add(site, sign * v * (int(F[(B_idx,) + ahead]) + dF.get((B_idx, ahead), 0)))
            if idx == B_idx:
                # F_A[x] ΔF_B[x+ê_A] at x = site - ê_A: the F∧ΔF term.
                behind = tuple((site[k] - (k in A_dirs)) % N for k in range(4))
                add(behind, sign * int(F[(A_idx,) + behind]) * v)
    return {site: v for site, v in dq.items() if v != 0}
