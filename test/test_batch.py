#!/usr/bin/env python

import numpy as np
import pytest

import supervillain.h5.extendable as extendable
from supervillain.batch import Batch
from supervillain.lattice import Form, Lattice


@pytest.fixture
def lattice():
    return Lattice(D=2, N=4)


def test_array_property():
    batch = Batch(4, shape=(2, 2), dtype=float)
    assert isinstance(batch.array, extendable.array)
    assert batch.array.shape == (4, 2, 2)
    batch.array[0] = 1.0
    assert batch[0, 0, 0] == pytest.approx(1.0)


def test_scalar_batch():
    batch = Batch(5, shape=(), dtype=float)
    assert batch.shape == (5,)
    assert len(batch) == 5
    batch[2] = 3.5
    assert batch[2] == pytest.approx(3.5)


def test_spatial_batch():
    batch = Batch(3, shape=(4, 4), dtype=float)
    assert batch.shape == (3, 4, 4)
    item = batch[1]
    assert item.shape == (4, 4)
    batch[1] = np.ones((4, 4))
    assert batch[1].sum() == pytest.approx(16)


def test_form_batch(lattice):
    batch = Batch(7, cls=Form, degree=0, lattice=lattice)
    assert batch.shape == (7, 1, 4, 4)
    phi = batch[3]
    assert isinstance(phi, Form)
    assert phi.degree == 0
    assert phi.lattice is lattice
    batch[3] = lattice.zeros(0)
    assert batch[3].shape == (1, 4, 4)


def test_form_batch_int_dtype(lattice):
    batch = Batch(2, cls=Form, degree=1, lattice=lattice, dtype=int)
    assert batch.dtype == int
    n = batch[0]
    assert isinstance(n, Form)
    assert n.dtype == int


def test_batch_slice(lattice):
    batch = Batch(10, cls=Form, degree=0, lattice=lattice)
    sub = batch[2:5]
    assert isinstance(sub, Batch)
    assert sub.shape == (3, 1, 4, 4)
    assert sub.cls is Form
    assert sub._item_kwargs == batch._item_kwargs


def test_setitem_coerces_dtype(lattice):
    batch = Batch(1, cls=Form, degree=1, lattice=lattice, dtype=int)
    batch[0] = lattice.zeros(1, dtype=int)
    assert batch[0].dtype == int
