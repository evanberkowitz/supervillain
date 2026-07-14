#!/usr/bin/env python
r"""
The pair-separation umbrella w2(r): a learned shell table multiplying the
enlarged-ensemble weight in the D=2 (single unit pair) and D=4 (four-unit,
Wick-sum) sectors.  An empty table is the off switch, preserving all prior
paths bit-for-bit.
"""

import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection import DefectGas
from supervillain.generator.no_intersection.defect_gas import pair_shells


def _action(N=4, kappa=0.05):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def _cold(S):
    L = S.Lattice
    return (np.zeros((1,) + tuple(L.dims)),
            np.zeros((4,) + tuple(L.dims), dtype=np.int64))


def _twins(S, seed=17, **kwargs):
    return (DefectGas(S, rng=np.random.default_rng(seed), **kwargs),
            DefectGas(S, rng=np.random.default_rng(seed), **kwargs))


def test_pair_shells():
    N = 4
    lookup, values = pair_shells(N)
    # Realized min-image r^2 on N=4: components in {0, 1, 4}; r^2 = 0 excluded.
    d = np.minimum(np.arange(N), N - np.arange(N))**2
    rsq = (d[:, None, None, None] + d[None, :, None, None]
           + d[None, None, :, None] + d[None, None, None, :]).ravel()
    expected = np.unique(rsq[rsq > 0])
    assert np.array_equal(values, expected)
    assert lookup.shape == (N**2 + 1,)
    for r2 in range(N**2 + 1):
        if r2 in set(expected.tolist()):
            assert values[lookup[r2]] == r2
        else:
            assert lookup[r2] == -1


def test_w2_constructor():
    S = _action()
    _, values = pair_shells(4)
    w2 = np.linspace(1.0, 2.0, len(values))
    g = DefectGas(S, weights=(1.0, 0.04, 2.4e-3, 5e-5, 4e-6), w2=w2)
    assert np.array_equal(g.w2, w2)
    # The per-displacement field carries w2 by shell and 1.0 at the origin,
    # and is hypercubically symmetric by construction.
    L = Lattice(4, 4)
    f = g._w2_field
    assert f[0, 0, 0, 0] == 1.0
    assert f[1, 0, 0, 0] == f[0, 0, 0, 1] == f[3, 0, 0, 0]
    assert np.allclose(np.asarray(L.symmetrize(f)), f, rtol=0, atol=1e-14)


def test_w2_validation():
    S = _action()
    _, values = pair_shells(4)
    for bad in (np.ones(len(values) - 1),          # wrong length
                -np.ones(len(values)),             # not positive
                np.zeros(len(values))):            # not positive
        try:
            DefectGas(S, fugacity=0.1, w2=bad)
        except ValueError:
            pass
        else:
            assert False, 'bad w2 must raise ValueError'


def test_no_w2_is_sentinel():
    S = _action()
    g = DefectGas(S, fugacity=0.1)
    assert g.w2.size == 0 and g._w2_field is None
