#!/usr/bin/env python

import numpy as np
import pytest

import supervillain
from supervillain.lattice import d, delta


@pytest.mark.parametrize('degree, dtype', [(1, int), (2, int), (0, float), (1, float)])
def test_d_and_delta_preserve_input_dtype(degree, dtype):
    # d and δ are exact ±1 integer combinations, so they must preserve the dtype.
    L = supervillain.lattice.Lattice(D=3, N=4)
    f = L.form(degree, dtype=dtype)
    assert np.asarray(d(f)).dtype == np.dtype(dtype)
    if degree > 0:
        assert np.asarray(delta(f)).dtype == np.dtype(dtype)


@pytest.mark.parametrize('D', [2, 3])
def test_villain_hammer_preserves_field_dtypes(D):
    # phi is float, n is int -- and must stay so through a full update.
    L = supervillain.lattice.Lattice(D=D, N=4)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    out = supervillain.generator.villain.Hammer(S).step(S.configurations(1)[0])
    assert np.issubdtype(np.asarray(out['phi']).dtype, np.floating)
    assert np.issubdtype(np.asarray(out['n']).dtype, np.integer)


@pytest.mark.parametrize('D', [2, 3])
def test_worldline_hammer_preserves_field_dtypes(D):
    # m is int, and v is int for finite W.
    L = supervillain.lattice.Lattice(D=D, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=1)
    out = supervillain.generator.worldline.Hammer(S).step(S.configurations(1)[0])
    assert np.issubdtype(np.asarray(out['m']).dtype, np.integer)
    assert np.issubdtype(np.asarray(out['v']).dtype, np.integer)


def test_worldline_v_is_float_when_W_infinite():
    # At W=∞ the vortex field is continuous, so v is float (and m is still int).
    L = supervillain.lattice.Lattice(D=2, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=float('inf'))
    out = supervillain.generator.worldline.Hammer(S).step(S.configurations(1)[0])
    assert np.issubdtype(np.asarray(out['m']).dtype, np.integer)
    assert np.issubdtype(np.asarray(out['v']).dtype, np.floating)
