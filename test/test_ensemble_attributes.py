#!/usr/bin/env python
r"""An Ensemble exposes its configuration's fields as attributes, which it does
by delegating in __getattr__.  These are the edges of that delegation: an
ensemble that has no configurations yet, and a Configurations that has no fields.
"""

import copy

import numpy as np
import pytest

import supervillain
from supervillain.batch import Batch
from supervillain.configurations import Configurations

N = 4
KAPPA = 0.2


@pytest.fixture
def action():
    return supervillain.action.Villain(supervillain.lattice.Lattice2D(N), KAPPA)


@pytest.fixture
def populated(action):
    return supervillain.Ensemble(action).generate(
            8, supervillain.generator.villain.NeighborhoodUpdate(action), start='cold')


@pytest.mark.parametrize('name', ('weight', 'phi', 'anything', '__deepcopy__'))
def test_an_ensemble_without_configurations_says_so(action, name):
    r'''__getattr__ delegates to the configuration, and on an ensemble that has
    not been given one the delegate is itself a miss.  Reaching it as
    self.configuration would make this method call itself until the stack ran
    out, so every attribute --- including the ones the machinery asks for behind
    the reader's back --- would answer RecursionError instead of AttributeError.
    '''
    e = supervillain.Ensemble(action)

    with pytest.raises(AttributeError):
        getattr(e, name)


def test_the_machinery_that_asks_quietly_still_works(action):
    r'''hasattr catches AttributeError and nothing else, and copy.deepcopy probes
    for __deepcopy__ and __getstate__ before doing anything.  Neither is asking
    for something unusual; both broke.
    '''
    e = supervillain.Ensemble(action)

    assert not hasattr(e, 'anything')
    assert isinstance(copy.deepcopy(e), supervillain.Ensemble)


def test_a_populated_ensemble_still_delegates(populated):
    r'''The point of __getattr__ is the delegation; it has to survive the guard.'''
    assert np.asarray(Batch.as_array(populated.phi)).shape[0] == len(populated)

    with pytest.raises(AttributeError):
        populated.anything

    assert not hasattr(populated, 'anything')


def test_configurations_with_no_fields_are_empty():
    r'''__len__ tracks the length across fields, starting from None to mean that
    no field has reported one.  With no fields at all that sentinel was returned
    as the length, and len() refuses to be told None.
    '''
    assert len(Configurations({})) == 0


def test_configurations_still_report_and_check_their_length():
    r'''The empty case must not cost the ordinary ones.'''
    assert len(Configurations({'x': Batch(np.zeros(5)), 'y': Batch(np.zeros(5))})) == 5

    with pytest.raises(ValueError):
        len(Configurations({'x': Batch(np.zeros(5)), 'y': Batch(np.zeros(4))}))
