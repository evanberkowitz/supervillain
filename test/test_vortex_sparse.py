#!/usr/bin/env python
"""The sparse/hoisted VortexUpdate.step must reproduce the dense step_reference.

For integer v (finite W) the incrementally-maintained δv is exact, so the two
implementations must produce bit-identical Markov chains given the same seed."""

import numpy as np
import pytest

from supervillain.lattice import Lattice, Form
from supervillain.action import Worldline
from supervillain.generator.worldline import VortexUpdate


def _config(L, seed):
    rng = np.random.default_rng(seed)
    m = Form(rng.integers(-3, 4, (len(L.components[1]),) + L.dims), degree=1, lattice=L)
    v = Form(rng.integers(-3, 4, (len(L.components[2]),) + L.dims), degree=2, lattice=L)
    return {'m': m, 'v': v}


@pytest.mark.parametrize("D", [2, 3, 4])
@pytest.mark.parametrize("W", [1, 3])
def test_step_matches_step_reference_bitexact(D, W):
    L = Lattice(D=D, N=5)
    S = Worldline(L, kappa=0.5, W=W)
    cfg = _config(L, seed=10 * D + W)

    # Run several steps through each implementation, feeding the output back, so
    # the incremental δv is exercised across step boundaries.
    sparse = VortexUpdate(S); sparse.rng = np.random.default_rng(99)
    dense  = VortexUpdate(S); dense.rng  = np.random.default_rng(99)

    c_sparse = c_dense = cfg
    for _ in range(5):
        c_sparse = c_sparse | sparse.step(c_sparse)
        c_dense  = c_dense  | dense.step_reference(c_dense)
        assert (np.asarray(c_sparse['v']) == np.asarray(c_dense['v'])).all()
        assert (np.asarray(c_sparse['m']) == np.asarray(c_dense['m'])).all()


@pytest.mark.parametrize("D", [2, 3, 4])
def test_step_preserves_integer_v_dtype(D):
    L = Lattice(D=D, N=5)
    S = Worldline(L, kappa=0.5, W=2)
    cfg = _config(L, seed=D)
    g = VortexUpdate(S)
    out = g.step(cfg)
    assert np.issubdtype(np.asarray(out['v']).dtype, np.integer)
