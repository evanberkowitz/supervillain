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


# --- explicit dtype: allowed only when the data fits losslessly -------------

# (source dtype, requested dtype, fits losslessly?)
CAST_MATRIX = [
    # identity and lossless widenings are allowed
    (np.float64,    np.float64,    True),   # identity (the dtype=data.dtype path)
    (np.int32,      np.int64,      True),   # integer widening
    (np.float32,    np.float64,    True),   # float widening
    (np.int64,      np.float64,    True),   # int -> float
    (np.int64,      np.complex128, True),   # int -> complex
    (np.float64,    np.complex128, True),   # float -> complex
    # lossy casts are rejected
    (np.float64,    np.int64,      False),  # float -> int truncates
    (np.complex128, np.float64,    False),  # complex -> float drops imag
    (np.complex128, np.int64,      False),  # complex -> int
    (np.int64,      np.int32,      False),  # integer narrowing
    (np.float64,    np.float32,    False),  # float precision downcast
]


@pytest.mark.parametrize('source, requested, fits', CAST_MATRIX)
def test_explicit_dtype_cast_matrix(source, requested, fits):
    data = np.ones(4, dtype=source)
    if fits:
        assert Batch(data, dtype=requested).dtype == np.dtype(requested)
    else:
        with pytest.raises(TypeError):
            Batch(data, dtype=requested)
