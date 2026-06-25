#!/usr/bin/env python

import numpy as np
import pytest

import supervillain
from supervillain.observable.vortex import Vortex_Vortex


def _random_two_form(L, seed):
    rng = np.random.default_rng(seed)
    v = L.form(2, dtype=int)
    v[:] = rng.integers(-3, 4, size=v.shape)
    return v


@pytest.mark.parametrize('D', [2, 3, 4])
def test_vortex_vortex_worldline_origin_is_one(D):
    # V[origin] = mean_c (1/N^D) Σ_x |exp(2πi v_c/W)|^2 = 1, in any D.
    L = supervillain.lattice.Lattice(D=D, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=float('inf'))
    v = _random_two_form(L, seed=0)

    vv = np.asarray(Vortex_Vortex.Worldline(S, v))
    assert vv.shape == L.dims
    assert np.isclose(vv[L.origin], 1.0)


def test_vortex_vortex_worldline_averages_over_orientations_in_D3():
    # The old code used only v[0]; with C(3,2)=3 independent orientations the
    # orientation-average must differ from the single-orientation correlator.
    L = supervillain.lattice.Lattice(D=3, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=float('inf'))
    v = _random_two_form(L, seed=1)

    vortex = np.exp(2j * np.pi * np.asarray(v) / S._W)
    single = np.asarray(L.correlation(vortex[0:1], vortex[0:1])).mean(axis=0)
    averaged = np.asarray(Vortex_Vortex.Worldline(S, v))

    assert averaged.shape == L.dims
    assert not np.allclose(averaged, single)


def test_vortex_vortex_worldline_requires_D_at_least_2():
    L = supervillain.lattice.Lattice(D=1, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=float('inf'))
    v = L.form(2, dtype=int)
    with pytest.raises(NotImplementedError):
        Vortex_Vortex.Worldline(S, v)
