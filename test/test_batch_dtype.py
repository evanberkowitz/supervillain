#!/usr/bin/env python

import numpy as np
import pytest

from supervillain.batch import Batch


# --- inference: no requested dtype keeps the data's own dtype ---------------

@pytest.mark.parametrize('source', [np.int32, np.int64, np.float32, np.float64, np.complex128])
def test_no_dtype_preserves_source_dtype(source):
    data = np.ones(3, dtype=source)
    assert Batch(data).dtype == np.dtype(source)


def test_inference_does_not_drop_imaginary_part():
    data = np.array([1 + 2j, 3 + 4j])
    batch = Batch(data)
    assert batch[0] == 1 + 2j and batch[1] == 3 + 4j


# --- explicit dtype: allowed exactly when every value is preserved ----------

# (values, requested dtype) that round-trip losslessly and are therefore allowed.
VALUE_PRESERVING = [
    ([1, 2, 3],        np.int64),       # identity
    ([1, 2, 3],        np.float64),     # int -> float
    ([1.0, 2.0, 3.0],  np.int64),       # integer-valued float -> int (e.g. the worldline m)
    ([1 + 0j, 2 + 0j], np.float64),     # zero-imaginary complex -> float
    ([1, 2, 3],        np.int32),       # in-range int64 -> int32
]

# (values, requested dtype) that change a value and must therefore raise.
VALUE_CHANGING = [
    ([2.7, 3.9],       np.int64),       # fractional float -> int
    ([1 + 2j, 3 + 4j], np.float64),     # nonzero imaginary -> float
    ([1 + 2j],         np.int64),       # complex -> int
    ([2 ** 40],        np.int32),       # out-of-range int -> int32
]


@pytest.mark.parametrize('values, requested', VALUE_PRESERVING)
def test_construct_allows_value_preserving_casts(values, requested):
    batch = Batch(np.array(values), dtype=requested)
    assert batch.dtype == np.dtype(requested)
    assert np.array_equal(batch.array, np.array(values))


@pytest.mark.parametrize('values, requested', VALUE_CHANGING)
def test_construct_rejects_value_changing_casts(values, requested):
    with pytest.raises(TypeError):
        Batch(np.array(values), dtype=requested)


# --- the per-draw write path must enforce the same contract (no silent drops) ---

def test_write_path_rejects_float_to_int():
    b = Batch(3, shape=(), dtype=int)
    with pytest.raises(TypeError):
        b[0] = 2.7


def test_write_path_rejects_complex_to_float():
    c = Batch(2, shape=(), dtype=float)
    with pytest.raises(TypeError):
        c[0] = np.array(1 + 2j)


def test_write_path_allows_lossless_widening():
    c = Batch(2, shape=(), dtype=float)
    c[0] = 3                      # int -> float is lossless
    assert c[0] == 3.0

    b = Batch(2, shape=(), dtype=int)
    b[1] = np.int32(5)            # int32 -> int64 is lossless
    assert b[1] == 5
