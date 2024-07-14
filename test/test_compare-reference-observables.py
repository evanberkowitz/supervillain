#!/usr/bin/env python

import pytest

import numpy as np
import supervillain
from supervillain.observable.reference_implementation.spin import Spin_SpinSlow
import generate
import harness

equality_threshold = 1e-12

####
#### Observables
####

pairs = (
        ('Spin_Spin', 'Spin_SpinSlow'),
        # ('Spin_Spin', 'Spin_SpinSloppy'), # fails due to sloppiness, uncomment to check the test is meaningful.
)


@pytest.mark.parametrize('pair', pairs)
@pytest.mark.parametrize('action', ('villain', 'worldline'))
@harness.for_each_test_ensemble
def test_reference_observable(action, N, kappa, W, pair, configurations=1000, ):
    ensemble = generate.cached_ensemble(action, configurations, N, kappa, W)

    values = ()
    for i, o in enumerate(pair):
        # If the observable is measured inline it may not match the reference.
        if o in ensemble.configuration.fields:
            # Temporarily store inline observables...
            tmp = ensemble.configuration.fields[o]
            del ensemble.configuration.fields[o]
            # ... measure the desired observable
            values += (getattr(ensemble, o),)
            # ... and restore the inline measurement.
            ensemble.configuration.fields[o] = tmp
        # But if the observable wasn't measured inline, measure it!
        else:
            values += (getattr(ensemble, o),)

    difference = np.abs(values[0]-values[1])

    
    assert (difference < equality_threshold).all().item()

