#!/usr/bin/env python

import numpy as np
import pytest

import supervillain
from supervillain.observable.winding import WindingSquared, Winding_Winding


def _random_one_form(L, dtype, seed):
    rng = np.random.default_rng(seed)
    f = L.form(1, dtype=dtype)
    if np.issubdtype(np.dtype(dtype), np.integer):
        f[:] = rng.integers(-2, 3, size=f.shape)
    else:
        f[:] = rng.normal(size=f.shape)
    return f


@pytest.mark.parametrize('D', [2, 3])
def test_winding_winding_origin_equals_winding_squared_villain(D):
    L = supervillain.lattice.Lattice(D=D, N=4)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    n = _random_one_form(L, int, seed=0)

    ww = np.asarray(Winding_Winding.Villain(S, n))
    wsq = WindingSquared.Villain(S, n)

    assert ww.shape == L.dims
    assert np.isclose(ww[L.origin], wsq)


@pytest.mark.parametrize('D', [2, 3])
def test_winding_winding_origin_equals_winding_squared_worldline(D):
    L = supervillain.lattice.Lattice(D=D, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=1)
    Links = _random_one_form(L, float, seed=1)

    ww = np.asarray(Winding_Winding.Worldline(S, Links))
    wsq = WindingSquared.Worldline(S, Links)

    assert ww.shape == L.dims
    assert np.isclose(ww[L.origin], wsq)


@pytest.mark.parametrize('D', [2, 3])
def test_winding_winding_formulations_share_shape(D):
    # #2: Villain and Worldline must return the same (N,)*D shape (no leading component axis).
    L = supervillain.lattice.Lattice(D=D, N=4)
    Sv = supervillain.action.Villain(L, kappa=0.5, W=1)
    Sw = supervillain.action.Worldline(L, kappa=0.5, W=1)
    n = _random_one_form(L, int, seed=2)
    Links = _random_one_form(L, float, seed=3)

    villain = np.asarray(Winding_Winding.Villain(Sv, n))
    worldline = np.asarray(Winding_Winding.Worldline(Sw, Links))
    assert villain.shape == worldline.shape == L.dims


def test_winding_requires_D_at_least_2():
    L = supervillain.lattice.Lattice(D=1, N=4)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    n = L.form(1, dtype=int)
    with pytest.raises(NotImplementedError):
        WindingSquared.Villain(S, n)
    with pytest.raises(NotImplementedError):
        Winding_Winding.Villain(S, n)


def test_winding_stencil_cache_is_dimension_safe():
    # The dδ contact-stencil cache is keyed by (D, N), so measuring at one dimension
    # must not clobber the stencil for another.
    shapes = {}
    for D in (2, 3):
        L = supervillain.lattice.Lattice(D=D, N=4)
        S = supervillain.action.Worldline(L, kappa=0.5, W=1)
        Links = _random_one_form(L, float, seed=5)
        ww = np.asarray(Winding_Winding.Worldline(S, Links))
        assert ww.shape == L.dims
        assert np.isclose(ww[L.origin], WindingSquared.Worldline(S, Links))
        shapes[D] = Winding_Winding._stencil[(D, 4)].shape
    assert shapes[2] == (4, 4)
    assert shapes[3] == (4, 4, 4)
