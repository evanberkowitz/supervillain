#!/usr/bin/env python

import numpy as np

import supervillain
from supervillain.observable import SpinStiffness, TorusWrapping, WrappingSquared


def _upsilon_direct(S, n):
    r"""The stiffness computed straight from the winding sums, the reference the
    library DerivedQuantity must reproduce: Upsilon = kappa - (2 pi kappa)^2
    <M^2>_c / V, averaged over directions, M_mu = sum_{l in mu} n_l."""
    kappa = S.kappa
    L = S.Lattice
    M = np.asarray(n).reshape(len(n), L.D, -1).sum(axis=2)   # (nconf, D)
    var = (M**2).mean(0) - M.mean(0)**2                       # connected, per direction
    return kappa - (2 * np.pi * kappa)**2 * var.mean() / L.sites


def test_spin_stiffness_matches_winding_sum_formula():
    # The DerivedQuantity is built from WrappingSquared and TorusWrapping; check it
    # reproduces the direct winding-sum formula on an explicit ensemble of n's.
    L = supervillain.lattice.Lattice(D=4, N=4)
    S = supervillain.action.Villain(L, kappa=0.3, W=1)
    rng = np.random.default_rng(0)
    n = np.stack([L.form(1, dtype=int) for _ in range(200)])
    n[:] = rng.integers(-2, 3, size=n.shape)

    tw = np.stack([TorusWrapping.Villain(S, None, n[i]) for i in range(len(n))])
    ws = np.array([WrappingSquared.default(S, tw[i]) for i in range(len(n))])

    # Feed ensemble-mean WrappingSquared / TorusWrapping through the DQ as the
    # bootstrap machinery would, and compare to the direct formula.
    upsilon = SpinStiffness.Villain(S, ws.mean(), tw.mean(0))
    assert np.isclose(upsilon, _upsilon_direct(S, n))


def test_spin_stiffness_frozen_field_is_kappa():
    # kappa -> infinity limit, realized exactly by n = 0 (no winding fluctuation):
    # a rigid field absorbs the twist and Upsilon = kappa.
    L = supervillain.lattice.Lattice(D=4, N=4)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    tw = np.zeros((10, L.D))
    ws = np.zeros(10)
    assert np.isclose(SpinStiffness.Villain(S, ws.mean(), tw.mean(0)), S.kappa)


def test_spin_stiffness_free_field_screens_to_zero():
    # In the unconstrained small-kappa limit the links are independent discrete
    # Gaussians of variance 1/(4 pi^2 kappa), so <M^2> = V/(4 pi^2 kappa) per
    # direction and Upsilon -> kappa - (2 pi kappa)^2 (1/(4 pi^2 kappa)) = 0:
    # complete screening.  Check the algebra with that analytic variance.
    L = supervillain.lattice.Lattice(D=4, N=6)
    kappa = 0.02
    S = supervillain.action.Villain(L, kappa=kappa, W=1)
    var_per_direction = L.sites / (4 * np.pi**2 * kappa)
    ws_mean = L.D * var_per_direction          # sum over directions of <M_mu^2>
    tw_mean = np.zeros(L.D)                     # <M_mu> = 0
    assert np.isclose(SpinStiffness.Villain(S, ws_mean, tw_mean), 0.0)


def test_spin_stiffness_through_bootstrap():
    # End-to-end: a short unconstrained Villain chain deep in the ordered phase
    # (kappa well above the 4D critical ~0.26) must have Upsilon/kappa near 1.
    L = supervillain.lattice.Lattice(D=4, N=4)
    S = supervillain.action.Villain(L, kappa=0.6, W=1)
    e = supervillain.Ensemble(S).generate(
        200, supervillain.generator.villain.Hammer(S), start='cold')
    b = supervillain.analysis.Bootstrap(e.cut(50))
    upsilon, err = b.estimate('SpinStiffness')
    assert 0.7 < float(np.real(upsilon)) / S.kappa <= 1.0 + 5 * float(np.real(err)) / S.kappa
