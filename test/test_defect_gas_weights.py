#!/usr/bin/env python
r"""
DefectGas with a per-sector weight table w[k] (k = D/2) generalizing the geometric
zeta^D pricing.  The table path must validate its inputs, reproduce the fugacity
path when handed a geometric table, and match the pure-python reference chain
bit-for-bit; the geometric path itself is untouched.
"""

import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection import DefectGas


def _action(N=4, kappa=0.05):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def _cold(S):
    L = S.Lattice
    return (np.zeros((1,) + tuple(L.dims)),
            np.zeros((4,) + tuple(L.dims), dtype=np.int64))


def _twins(S, seed=17, **kwargs):
    return (DefectGas(S, rng=np.random.default_rng(seed), **kwargs),
            DefectGas(S, rng=np.random.default_rng(seed), **kwargs))


def test_exactly_one_pricing():
    S = _action()
    for kwargs in ({}, {'fugacity': 0.1, 'weights': (1.0, 0.01)}):
        try:
            DefectGas(S, **kwargs)
        except ValueError:
            pass
        else:
            assert False, f'DefectGas(S, **{kwargs}) must raise ValueError'


def test_weights_normalized_and_pin_D_max():
    S = _action()
    g = DefectGas(S, weights=(2.0, 1.0, 0.5))
    assert g.fugacity is None
    assert g.D_max == 4
    assert np.array_equal(g.w, [1.0, 0.5, 0.25])
    assert g._w1 == 0.5 and g._w4 == 0.25


def test_weights_short_table_has_no_four_sector():
    S = _action()
    g = DefectGas(S, weights=(1.0, 0.25))
    assert g.D_max == 2 and g._w4 == 1.0


def test_weights_D_max_mismatch():
    S = _action()
    try:
        DefectGas(S, weights=(1.0, 0.1, 0.01), D_max=8)
    except ValueError:
        pass
    else:
        assert False, 'disagreeing D_max must raise ValueError'


def test_weights_must_be_positive():
    S = _action()
    for w in ((1.0, 0.0, 0.1), (1.0, -0.1, 0.1), (1.0,)):
        try:
            DefectGas(S, weights=w)
        except ValueError:
            pass
        else:
            assert False, f'weights={w} must raise ValueError'


def test_geometric_path_unchanged():
    S = _action()
    g = DefectGas(S, fugacity=0.1)
    assert g.fugacity == 0.1 and g.w.size == 0 and g.D_max is None
    assert g._w1 == 0.1**2 and g._w4 == 0.1**4
