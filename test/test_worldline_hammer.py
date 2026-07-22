#!/usr/bin/env python

import numpy as np
import pytest
import supervillain
from supervillain.action import Worldline
from supervillain.lattice import Lattice, delta


def _action(D=2, N=4, kappa=0.5, W=1):
    return Worldline(Lattice(D, N), kappa=kappa, W=W)


def test_worldline_hammer_uses_heatbaths():
    # The default sampler uses the exact heatbaths, not the Metropolis VortexUpdate/CoexactUpdate.
    s = str(supervillain.generator.worldline.Hammer(_action(W=1)))
    assert 'VortexHeatbath' in s
    assert 'CoexactHeatbath' in s
    assert 'VortexUpdate' not in s
    assert 'CoexactUpdate' not in s


def test_worldline_hammer_overrelaxation_only_at_infinite_W():
    # VortexOverrelaxation is only defined for continuous v (W=inf).
    assert 'VortexOverrelaxation' not in str(supervillain.generator.worldline.Hammer(_action(W=1)))
    assert 'VortexOverrelaxation' not in str(supervillain.generator.worldline.Hammer(_action(W=3)))
    assert 'VortexOverrelaxation' in str(supervillain.generator.worldline.Hammer(_action(W=float('inf'))))


def test_worldline_hammer_overrelax_must_be_positive():
    with pytest.raises(ValueError):
        supervillain.generator.worldline.Hammer(_action(W=float('inf')), overrelax=0)


@pytest.mark.parametrize('W', [1, 2, float('inf')])
def test_worldline_hammer_produces_valid_configs(W):
    # The heatbath-based Hammer must maintain the constraint delta m = 0.
    S = _action(D=2, N=4, kappa=0.5, W=W)
    e = supervillain.Ensemble(S).generate(30, supervillain.generator.worldline.Hammer(S),
                                          start='cold')
    for i in range(len(e)):
        assert (np.asarray(delta(e.configuration[i]['m'])) == 0).all()
