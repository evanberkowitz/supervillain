#!/usr/bin/env python

import pytest
from functools import cache

import numpy as np
import h5py as h5

import supervillain
import generate
import harness

def compare(measured, unmeasured, equality_threshold=1e-12):
    unmeasured.measure()

    for o in measured.measured:
        m = getattr(measured, o)
        u = getattr(measured, o)

        if (np.abs(m-u) > equality_threshold).any():
            return False

    return True

@pytest.mark.parametrize('action', ('villain', 'worldline'))
@harness.for_each_test_ensemble
def test_extend_and_compare(action, W, kappa, N, tmp_path, file='extend-ensemble.h5', configurations=500):
    E = generate.ensemble(configurations, action, N, kappa, W)
    
    with h5.File(tmp_path / file, 'w') as f:
        unmeasured = f.create_group('unmeasured')
        measured   = f.create_group('measured')

        E.to_h5(unmeasured)
        E.measure()
        E.to_h5(measured)

        F = supervillain.Ensemble.continue_from(unmeasured, configurations)
        F.extend_h5(unmeasured)
        F.measure()
        F.extend_h5(measured)

        combined_m = supervillain.Ensemble.from_h5(measured)
        combined_u = supervillain.Ensemble.from_h5(unmeasured)

    assert compare(combined_m, combined_u)


