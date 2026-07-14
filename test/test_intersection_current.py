#!/usr/bin/env python
r"""
The theta-sector current J = n wedge dn and its consumers: dJ = q must hold
EXACTLY configuration by configuration (lattice Leibniz + d^2 = 0), the
winding slice sums must be slice-independent integers on constrained
configurations, and the charge-2 susceptibility must reduce to its defining
arithmetic on the {+2,-2} class tally.
"""

import numpy as np

import supervillain
from supervillain.lattice import Lattice, Form, d, wedge
from supervillain.observable import DoubleIntersectionSusceptibility


def _action(N=4, kappa=0.1):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def _random_n(L, rng):
    return Form(rng.integers(-2, 3, size=(4,) + tuple(L.dims)),
                degree=1, lattice=L)


def test_dJ_equals_q():
    # d(n ∧ dn) = dn ∧ dn exactly, for arbitrary (unconstrained) integer n.
    L = Lattice(4, 4)
    rng = np.random.default_rng(7)
    for _ in range(3):
        n = _random_n(L, rng)
        J = wedge(n, d(n))
        q = wedge(d(n), d(n))
        assert np.array_equal(np.asarray(d(J)), np.asarray(q))


def test_winding_topological_on_constrained_configurations():
    # On q = 0 configurations every slice of J carries the same flux, and the
    # ensemble observable equals the direct slice sum.
    S = _action()
    L = S.Lattice
    gas = supervillain.generator.no_intersection.DefectGas(
        S, sectorWeights=(1.0, 0.04, 2.4e-3), emit_every=100,
        rng=np.random.default_rng(11))
    e = supervillain.Ensemble(S).generate(
        3, supervillain.generator.combining.Sequentially(
            (supervillain.generator.villain.SiteUpdate(S), gas)),
        start='cold')
    J_all = np.asarray(e.IntersectionCurrent)
    W_all = np.asarray(e.IntersectionWinding)
    for cfg in range(len(e)):
        J = J_all[cfg]
        for mu in range(4):
            comp = tuple(k for k in range(4) if k != mu)
            field = J[L.comp_index[3][comp]]
            slice_sums = [np.take(field, c, axis=mu).sum() for c in range(L.N)]
            assert len(set(slice_sums)) == 1                 # topological
            assert W_all[cfg, mu] == slice_sums[0]           # and as measured
            assert float(W_all[cfg, mu]).is_integer()
    W2 = np.asarray(e.IntersectionWindingSquared)
    assert np.allclose(W2, (W_all**2).mean(axis=1))


def test_double_intersection_susceptibility_arithmetic():
    # chi_2 = 1 + FDD[3] / (V * VacuumTicks), from the class tally alone.
    S = _action()
    V = S.Lattice.N ** 4
    fdd = np.array([12.0, 3.0, 2.0, 8.0])
    vt = 400.0
    chi2 = DoubleIntersectionSusceptibility.NoIntersections(S, fdd, vt)
    assert np.isclose(chi2, 1 + 8.0 / (V * 400.0))
