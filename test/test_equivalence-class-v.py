#!/usr/bin/env python

from functools import cache

import numpy as np

import supervillain
import generate
import harness

equality_threshold = 1e-12

@cache
def transform(E):
    S = E.Action
    transformed = E.configuration.copy()
    for i, c in enumerate(E.configuration):
        transformed[i] = S.equivalence_class_v(c)

    return supervillain.Ensemble(S).from_configurations(transformed)

@harness.for_each_test_ensemble
@harness.for_each_observable
@harness.skip_on(NotImplementedError, 'Not implemented for Worldline formulation')
def test_equivalence_class(N, kappa, W, observable, configurations=1000):
    E = generate.cached_ensemble('worldline', configurations, N, kappa, W)
    F = transform(E)

    difference = getattr(E, observable) - getattr(F, observable)
    assert (np.abs(difference) <= equality_threshold).all().item()
