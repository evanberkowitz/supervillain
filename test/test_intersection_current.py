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


def _zero_form(L, rng, integer=True):
    a = (rng.integers(-2, 3, size=(1,) + tuple(L.dims)) if integer
         else rng.normal(size=(1,) + tuple(L.dims)))
    return Form(a, degree=0, lattice=L)


def test_current_is_gauge_variant_but_its_periods_are_not():
    # j = n ^ dn shifts by the EXACT form d(m ^ dn) under n -> n + dm, so it is
    # not an observable pointwise -- only its fluxes through 3-cycles are.
    # Anything correlating j at separated points or at nonzero momentum is
    # measuring the storage gauge.  See the warning on IntersectionCurrent.
    L = Lattice(4, 4)
    rng = np.random.default_rng(23)

    def periods(J):
        return np.array([J[L.comp_index[3][tuple(k for k in range(4) if k != mu)]].sum()
                         for mu in range(4)])

    for _ in range(3):
        n = _random_n(L, rng)
        m = _zero_form(L, rng)
        ng = Form(np.asarray(n) + np.asarray(d(m)), degree=1, lattice=L)

        j = np.asarray(wedge(n, d(n)))
        jg = np.asarray(wedge(ng, d(ng)))

        assert not np.array_equal(j, jg)                                  # variant pointwise
        assert np.array_equal(jg - j, np.asarray(d(wedge(m, d(n)))))      # ... by an exact form
        assert np.array_equal(periods(j), periods(jg))                    # periods survive


def test_gauge_invariant_current_is_invariant_and_shares_the_periods():
    # j_gi = (dphi - 2 pi n) ^ dn is invariant POINTWISE, obeys d j_gi = -2 pi q,
    # and differs from j only by the improvement term d(phi dn) -- so it carries
    # the same periods up to -2 pi and is the representative to correlate.
    L = Lattice(4, 4)
    rng = np.random.default_rng(29)

    def j_gi(phi, n):
        A = Form(np.asarray(d(phi)) - 2 * np.pi * np.asarray(n), degree=1, lattice=L)
        return np.asarray(wedge(A, d(n)))

    for _ in range(3):
        n = _random_n(L, rng)
        phi = _zero_form(L, rng, integer=False)
        m = _zero_form(L, rng)

        phig = Form(np.asarray(phi) + 2 * np.pi * np.asarray(m), degree=0, lattice=L)
        ng = Form(np.asarray(n) + np.asarray(d(m)), degree=1, lattice=L)

        assert np.allclose(j_gi(phi, n), j_gi(phig, ng))                  # invariant pointwise

        q = np.asarray(wedge(d(n), d(n)))
        assert np.allclose(
            np.asarray(d(Form(j_gi(phi, n), degree=3, lattice=L))), -2 * np.pi * q)

        improvement = (-2 * np.pi * np.asarray(wedge(n, d(n)))
                       + np.asarray(d(wedge(phi, d(n)))))
        assert np.allclose(j_gi(phi, n), improvement)
