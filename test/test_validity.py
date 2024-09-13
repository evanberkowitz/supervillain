
#!/usr/bin/env python

from functools import cache

import numpy as np

import supervillain
import generate
import harness

@harness.for_each_test_ensemble
def test_equivalence_class(N, kappa, W, configurations=1000):
    E = generate.cached_ensemble('worldline', configurations, N, kappa, W)

    S = E.Action

    valid = np.array([S.valid(c) for c in E.configuration])
    assert valid.all()
