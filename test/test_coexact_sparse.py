#!/usr/bin/env python
"""The sparse CoexactUpdate.step must reproduce the dense step_reference.

m is always integer and δv is constant over the sweep, so the sparse step is
bit-identical to the dense reference for every W (including W = ∞)."""

import numpy as np
import pytest

from supervillain.lattice import Lattice, Form, delta
from supervillain.action import Worldline
from supervillain.generator.worldline import CoexactUpdate


def _config(L, W, seed):
    rng = np.random.default_rng(seed)
    v_dtype = int if W < float('inf') else float
    v_data = (rng.integers(-3, 4, (len(L.components[2]),) + L.dims) if v_dtype is int
              else rng.standard_normal((len(L.components[2]),) + L.dims))
    v = Form(v_data.astype(v_dtype), degree=2, lattice=L)
    m = Form(np.zeros((len(L.components[1]),) + L.dims, dtype=int), degree=1, lattice=L)
    return {'m': m, 'v': v}


@pytest.mark.parametrize("D", [2, 3, 4])
@pytest.mark.parametrize("W", [1, 3, float('inf')])
def test_step_matches_step_reference_bitexact(D, W):
    L = Lattice(D=D, N=5)
    S = Worldline(L, kappa=0.5, W=W)
    cfg = _config(L, W, seed=10 * D)

    sparse = CoexactUpdate(S); sparse.rng = np.random.default_rng(99)
    dense  = CoexactUpdate(S); dense.rng  = np.random.default_rng(99)

    c_sparse = c_dense = cfg
    for _ in range(5):
        c_sparse = c_sparse | sparse.step(c_sparse)
        c_dense  = c_dense  | dense.step_reference(c_dense)
        assert (np.asarray(c_sparse['m']) == np.asarray(c_dense['m'])).all()


@pytest.mark.parametrize("D", [2, 3, 4])
def test_step_preserves_constraint_and_dtype(D):
    # Coexact changes m by δt, so it must keep δm = 0 and leave m integer.
    L = Lattice(D=D, N=5)
    S = Worldline(L, kappa=0.5, W=2)
    cfg = _config(L, 2, seed=D)
    g = CoexactUpdate(S)
    out = g.step(cfg)
    assert np.issubdtype(np.asarray(out['m']).dtype, np.integer)
    assert (np.asarray(delta(out['m'])) == 0).all()
