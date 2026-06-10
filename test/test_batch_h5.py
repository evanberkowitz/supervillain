#!/usr/bin/env python

import h5py as h5
import numpy as np
import pytest

import supervillain.h5.strategy.batch  # noqa: F401 — register strategy
from supervillain.h5 import Data
from supervillain.batch import Batch
from supervillain.compact.compact import Form, Lattice


@pytest.fixture
def lattice():
    return Lattice(D=2, N=3)


def test_batch_h5_round_trip(tmp_path, lattice):
    original = Batch(4, cls=Form, degree=0, lattice=lattice)
    original[0] = lattice.zeros(0) + 1.0

    path = tmp_path / 'batch.h5'
    with h5.File(path, 'w') as f:
        Data.write(f, 'phi', original)

    with h5.File(path, 'r') as f:
        restored = Data.read(f['phi'], strict=True)

    assert isinstance(restored, Batch)
    assert restored.shape == original.shape
    assert restored.cls is Form
    assert restored._item_kwargs['degree'] == 0
    assert np.allclose(restored[0], original[0])


def test_batch_h5_extend(tmp_path, lattice):
    first = Batch(3, shape=(2, 2), dtype=float)
    first[:, 0, 0] = np.arange(3)

    second = Batch(2, shape=(2, 2), dtype=float)
    second[:, 0, 0] = [10, 11]

    path = tmp_path / 'extend.h5'
    with h5.File(path, 'w') as f:
        g = f.create_group('column')
        Data.write(g, 'obs', first)
        g = f['column/obs']
        second.extend_h5(g)

    with h5.File(path, 'r') as f:
        combined = Data.read(f['column/obs'], strict=True)

    assert len(combined) == 5
    assert combined[0, 0, 0] == pytest.approx(0)
    assert combined[4, 0, 0] == pytest.approx(11)
