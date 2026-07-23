#!/usr/bin/env python

import numpy as np

import supervillain
from supervillain.analysis import Bootstrap
from supervillain.observable.action import ActionSusceptibility


def test_definition_is_sum_of_action_action():
    # χ_S is definitionally the spacetime sum of the connected correlator.
    L = supervillain.lattice.Lattice(2, 4)
    S = supervillain.action.Villain(L, kappa=0.5)
    rng = np.random.default_rng(7)
    correlator = rng.normal(size=L.dims)
    assert np.isclose(ActionSusceptibility.default(S, correlator), correlator.sum())


def test_bootstrap_plumbing():
    # End to end on a tiny cheap ensemble: the DerivedQuantity resolves through the
    # Bootstrap and matches the per-sample sum of Action_Action.
    L = supervillain.lattice.Lattice(2, 4)
    S = supervillain.action.Villain(L, kappa=0.5)
    g = supervillain.generator.villain.Hammer(S)
    e = supervillain.Ensemble(S).generate(32, g, start='cold')
    b = Bootstrap(e, draws=25)

    chi = np.asarray(b.ActionSusceptibility)
    aa = np.asarray(b.Action_Action)
    axes = tuple(range(1, aa.ndim))
    assert chi.shape == (25,)
    assert np.allclose(chi, aa.real.sum(axis=axes))
