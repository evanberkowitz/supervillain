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
@harness.skip_on(NotImplementedError, 'Not implemented for Villain formulation')
def test_equivalence_class(N, kappa, W, observable, configurations=1000):
    E = generate.cached_ensemble('worldline', configurations, N, kappa, W)
    F = transform(E)

    # We want to measure directly rather than rely on the inline observables.
    difference = harness.measure_without_inline(E, observable) - harness.measure_without_inline(F, observable)
    assert (np.abs(difference) <= equality_threshold).all().item()
