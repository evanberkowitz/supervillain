#!/usr/bin/env python

import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice, d, wedge


def _cold(S):
    # A single cold configuration dict {'phi': 0-form, 'n': 0-form}.
    return S.configurations(1)[0]


def test_no_intersections_requires_D4():
    L = Lattice(3, 4)
    with pytest.raises(ValueError):
        supervillain.action.NoIntersections(L, kappa=0.5)


def test_no_intersections_constructs_in_D4():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.5)
    assert S.W == 1
    assert S.Lattice is L
    assert 'NoIntersections' in str(S)


def test_valid_accepts_cold_config():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.5)
    assert S.valid(_cold(S))


def test_valid_rejects_intersecting_config():
    # A minimal intersecting configuration: two links whose field strength gives a
    # nonzero topological-charge density dn ∧ dn (per the library's operators).
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.5)
    n = L.zeros(1, dtype=int)
    n[0, 0, 0, 0, 0] = 1
    n[1, 1, 0, 0, 0] = 1
    assert np.any(np.asarray(wedge(d(n), d(n))) != 0)
    assert not S.valid({'phi': L.zeros(0), 'n': n})


def test_valid_and_action_are_gauge_invariant():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.7)
    cfg = _cold(S)
    k = L.zeros(0, dtype=int)
    k += np.random.default_rng(0).integers(-3, 4, size=k.shape)
    gauged = S.gauge_transform(cfg, k)
    assert S.valid(gauged) == S.valid(cfg)
    assert np.isclose(S(**gauged), S(**cfg))
