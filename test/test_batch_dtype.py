#!/usr/bin/env python

import numpy as np
import pytest

from supervillain.batch import Batch


def test_wrapping_infers_integer_dtype():
    # No requested dtype: keep the data's own dtype (integers stay integers).
    batch = Batch(np.arange(5))
    assert np.issubdtype(batch.dtype, np.integer)


def test_wrapping_infers_complex_dtype():
    batch = Batch(np.array([1 + 2j, 3 + 4j]))
    assert np.issubdtype(batch.dtype, np.complexfloating)
    assert batch[0] == 1 + 2j


def test_compatible_dtype_widening_is_allowed():
    # Integers fit losslessly into floats.
    batch = Batch(np.arange(5), dtype=float)
    assert np.issubdtype(batch.dtype, np.floating)


def test_complex_to_float_raises():
    with pytest.raises(TypeError):
        Batch(np.array([1 + 2j, 3 + 4j]), dtype=float)


def test_float_to_int_raises():
    with pytest.raises(TypeError):
        Batch(np.array([1.5, 2.5]), dtype=int)
