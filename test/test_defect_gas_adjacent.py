#!/usr/bin/env python
r"""
Defect-adjacent proposal targeting: a mixture proposal (gamma_k uniform /
charge-weighted defect-adjacent) with the full Metropolis-Hastings ratio.
gamma is a (K+1)-vector indexed by sector; the empty array is the legacy
sentinel (bit-for-bit today's sampler, same RNG stream).
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


W = (1.0, 0.04, 2.4e-3)          # a known-healthy N=4 kappa=0.05 table, D_max=4


def test_gamma_default_is_legacy():
    g = DefectGas(_action(), weights=W)
    assert g.gamma.size == 0


def test_gamma_scalar_broadcasts():
    g = DefectGas(_action(), weights=W, gamma=0.5)
    assert g.gamma.shape == (3,)               # K+1 = D_max/2 + 1 = 3
    assert np.all(g.gamma == 0.5)


def test_gamma_vector_accepted():
    g = DefectGas(_action(), weights=W, gamma=(1.0, 0.5, 0.25))
    assert np.array_equal(g.gamma, [1.0, 0.5, 0.25])


def test_gamma_needs_cap():
    try:
        DefectGas(_action(), fugacity=0.1, gamma=0.5)      # D_max None
    except ValueError:
        pass
    else:
        assert False, 'gamma without a capped gas must raise ValueError'


def test_gamma_works_with_capped_fugacity():
    g = DefectGas(_action(), fugacity=0.1, D_max=4, gamma=0.5)
    assert g.gamma.shape == (3,)


def test_gamma_validation():
    S = _action()
    for bad in (0.0, -0.5, 1.5, (0.5, 0.5), (1.0, 0.5, 0.0), (1.0, 0.5, 2.0)):
        try:
            DefectGas(S, weights=W, gamma=bad)
        except ValueError:
            pass
        else:
            assert False, f'gamma={bad} must raise ValueError'
