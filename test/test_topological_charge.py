#!/usr/bin/env python

import numpy as np
import pytest

import supervillain
from supervillain.batch import Batch
from supervillain.lattice import d, wedge
from supervillain.observable.topological import AbsoluteTopologicalChargeDensity


def unit_charge_dipole(L):
    r'''Return an integer 1-form whose charge consists of one +1/-1 pair.'''
    n = L.zeros(1, dtype=int)
    origin = L.origin

    n[L.comp_index[1][(0,)]][origin] = 1
    shifted = list(origin)
    shifted[0] = 1
    n[L.comp_index[1][(1,)]][tuple(shifted)] = 1

    return n


def test_absolute_topological_charge_density_vacuum_is_zero():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)

    assert AbsoluteTopologicalChargeDensity.Villain(S, L.zeros(1, dtype=int)) == 0


@pytest.mark.parametrize('kappa', [0.05, 0.5, 2.0])
def test_absolute_topological_charge_density_counts_unit_charge_dipole(kappa):
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=kappa, W=1)
    n = unit_charge_dipole(L)
    charge = wedge(d(n), d(n))

    assert np.array_equal(np.sort(np.asarray(charge)[charge != 0]), [-1, 1])
    assert charge.sum() == 0
    assert np.abs(charge).sum() == 2
    assert AbsoluteTopologicalChargeDensity.Villain(S, n) == 2 / L.sites


def test_absolute_topological_charge_density_matches_wedge_definition():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    rng = np.random.default_rng(20260629)
    n = L.zeros(1, dtype=int)
    n[...] = rng.integers(-2, 3, size=n.shape)

    charge = wedge(d(n), d(n))
    expected = np.abs(charge).sum() / L.cells_of_degree[4]
    measured = AbsoluteTopologicalChargeDensity.Villain(S, n)

    # Q is exact on the periodic lattice, while its L1 norm is nonzero.
    assert charge.sum() == 0
    assert expected > 0
    assert measured == expected


def test_absolute_topological_charge_density_is_gauge_invariant():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    n = unit_charge_dipole(L)
    phi = L.zeros(0)
    k = L.zeros(0, dtype=int)
    rng = np.random.default_rng(1234)
    k[...] = rng.integers(-3, 4, size=k.shape)

    transformed = S.gauge_transform({'phi': phi, 'n': n}, k)

    before = AbsoluteTopologicalChargeDensity.Villain(S, n)
    after = AbsoluteTopologicalChargeDensity.Villain(S, transformed['n'])
    assert after == before


def test_absolute_topological_charge_density_integrates_with_ensemble():
    L = supervillain.lattice.Lattice(D=4, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    configurations = S.configurations(2)
    configurations[0] = {'phi': L.zeros(0), 'n': L.zeros(1, dtype=int)}
    configurations[1] = {'phi': L.zeros(0), 'n': unit_charge_dipole(L)}
    ensemble = supervillain.Ensemble(S).from_configurations(configurations)

    assert (
        supervillain.observable.AbsoluteTopologicalChargeDensity
        is AbsoluteTopologicalChargeDensity
    )
    assert (
        supervillain.observables['AbsoluteTopologicalChargeDensity']
        is AbsoluteTopologicalChargeDensity
    )

    measured = ensemble.measure(['AbsoluteTopologicalChargeDensity'])
    values = measured['AbsoluteTopologicalChargeDensity']

    assert isinstance(values, Batch)
    assert values.shape == (2,)
    assert np.array_equal(Batch.as_array(values), [0, 2 / L.sites])
    assert ensemble.AbsoluteTopologicalChargeDensity is values
    assert 'AbsoluteTopologicalChargeDensity' in ensemble.measured
    assert AbsoluteTopologicalChargeDensity.autocorrelation(ensemble)


@pytest.mark.parametrize(('D', 'W'), [(3, 1), (4, 2)])
def test_absolute_topological_charge_density_rejects_unsupported_models(D, W):
    L = supervillain.lattice.Lattice(D=D, N=3)
    S = supervillain.action.Villain(L, kappa=0.5, W=W)

    with pytest.raises(NotImplementedError):
        AbsoluteTopologicalChargeDensity.Villain(S, L.zeros(1, dtype=int))
