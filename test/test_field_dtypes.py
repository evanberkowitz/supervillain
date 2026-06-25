#!/usr/bin/env python

import numpy as np
import pytest

import supervillain
from supervillain.lattice import d, delta
from supervillain.lattice.compact import star, wedge


@pytest.mark.parametrize('degree, dtype', [(1, int), (2, int), (0, float), (1, float)])
def test_d_and_delta_preserve_input_dtype(degree, dtype):
    # d and δ are exact ±1 integer combinations, so they must preserve the dtype.
    L = supervillain.lattice.Lattice(D=3, N=4)
    f = L.form(degree, dtype=dtype)
    assert np.asarray(d(f)).dtype == np.dtype(dtype)
    if degree > 0:
        assert np.asarray(delta(f)).dtype == np.dtype(dtype)


@pytest.mark.parametrize('dtype', [int, float])
def test_form_operators_preserve_dtype(dtype):
    # face_sum, coface_sum, ★ and ∧ are signed/product combinations, so they must
    # not silently promote an integer form to float.
    L = supervillain.lattice.Lattice(D=3, N=4)
    f1 = L.form(1, dtype=dtype)
    f2 = L.form(2, dtype=dtype)
    assert np.asarray(f2.face_sum()).dtype == np.dtype(dtype)
    assert np.asarray(f1.coface_sum()).dtype == np.dtype(dtype)
    assert np.asarray(star(f1)).dtype == np.dtype(dtype)
    assert np.asarray(wedge(f1, f1)).dtype == np.dtype(dtype)


@pytest.mark.parametrize('dA, dB', [
    (np.int64,   np.float64),    # int ∧ float -> float
    (np.float32, np.int64),      # -> float64 (int64 does not fit in float32), not float32
    (np.int32,   np.complex64),  # -> complex128
    (np.float64, np.complex128), # -> complex128
    (np.float32, np.float64),    # -> float64
])
def test_wedge_matches_numpy_promotion_for_mixed_dtypes(dA, dB):
    # wedge is bilinear, so its result dtype must match what the elementwise
    # product a*b actually produces under numpy's promotion rules.
    L = supervillain.lattice.Lattice(D=3, N=4)
    a = L.form(1, dtype=dA)
    b = L.form(1, dtype=dB)
    expected = (np.array(1, dA) * np.array(1, dB)).dtype
    assert np.asarray(wedge(a, b)).dtype == expected


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
