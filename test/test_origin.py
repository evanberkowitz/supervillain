#!/usr/bin/env python

import numpy as np
import pytest

import supervillain
from supervillain.observable.action import ActionTwoPoint


@pytest.mark.parametrize('D', [1, 2, 3, 4])
def test_origin_is_zero_tuple_of_length_D(D):
    L = supervillain.lattice.Lattice(D=D, N=3)
    assert L.origin == (0,) * D


@pytest.mark.parametrize('D', [1, 2, 3, 4])
def test_origin_indexes_zero_displacement(D):
    L = supervillain.lattice.Lattice(D=D, N=3)
    a = np.arange(L.sites).reshape(L.dims)
    assert a[L.origin] == 0


def test_action_two_point_contact_term_only_at_origin_in_D3():
    # Regression for blocker #1: the contact-term subtraction must hit the single
    # origin site, not the whole result[0,0] slab that [0,0] selects when D>2.
    L = supervillain.lattice.Lattice(D=3, N=4)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)

    rng = np.random.default_rng(0)
    Links = rng.normal(size=(L.D,) + L.dims)

    result = ActionTwoPoint.Villain(S, Links)

    density = 0.5 * S.kappa * (Links ** 2).sum(axis=0)
    raw = L.correlation(density, density)

    expected = np.zeros(L.dims)
    expected[L.origin] = -density.mean()

    assert np.allclose(np.asarray(result) - np.asarray(raw), expected)
