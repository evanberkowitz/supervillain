#!/usr/bin/env python

from functools import cache

import numpy as np

import supervillain
import generate
import harness

equality_threshold = 1e-12

@cache
def transform(before):

    S = before.Action
    L = S.Lattice

    transformed = S.configurations(len(before))
    for i, c in enumerate(before.configuration):
        k = np.random.randint(-10,10, L.dims)
        transformed[i] = S.gauge_transform(c, k)

    return supervillain.Ensemble(S).from_configurations(transformed)


@harness.for_each_test_ensemble
@harness.for_each_observable
@harness.skip_on(NotImplementedError, 'Villain formulation has no implementation.')
def test_gauge_invariance(N, kappa, W, observable, configurations=1000):

    before = generate.cached_ensemble('villain', configurations, N, kappa, W)
    after  = transform(before)

    difference = getattr(before, observable) - getattr(after, observable)
    assert (np.abs(difference) <= equality_threshold).all().item()
