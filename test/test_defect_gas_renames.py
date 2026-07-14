#!/usr/bin/env python
r"""
The semantic renames: sectorWeights (was weights, stored as w),
pairSeparationUmbrella (was w2), and uniformProposalFraction (was gamma).  The math-symbol spellings are GONE ---
stored h5 data was migrated (datasets/attrs linked to the new names) rather
than shimmed, so the old keyword arguments must fail loudly, not silently
vanish into **kwargs someday.
"""

import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection import DefectGas, DefectGasWeightTuner
from supervillain.generator.no_intersection.defect_gas import pair_shells


def _action(N=4, kappa=0.05):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


W = (1.0, 0.04, 2.4e-3)


def test_old_spellings_are_gone():
    S = _action()
    _, values = pair_shells(S.Lattice.N)
    for kwargs in ({'w2': np.ones(len(values))}, {'gamma': 0.5},
                   {'weights': W}):
        try:
            DefectGas(S, sectorWeights=W, **kwargs)
        except TypeError:
            pass
        else:
            assert False, f'retired keyword must raise TypeError: {sorted(kwargs)}'
    try:
        DefectGasWeightTuner(S, D_max=4, gamma=0.5)
    except TypeError:
        pass
    else:
        assert False, 'retired tuner keyword gamma must raise TypeError'
    import supervillain.observable
    assert not hasattr(supervillain.observable, 'ThetaBinderCumulant'), \
        'retired observable name must be gone (use IntersectionBinderCumulant)'
    assert hasattr(supervillain.observable, 'IntersectionBinderCumulant')


def test_semantic_names_carry_the_tables():
    S = _action()
    _, values = pair_shells(S.Lattice.N)
    w2 = np.linspace(1.0, 2.0, len(values))
    g = DefectGas(S, sectorWeights=W, pairSeparationUmbrella=w2,
                  uniformProposalFraction=0.5)
    assert np.array_equal(g.pairSeparationUmbrella, w2)
    assert g.uniformProposalFraction.shape == (3,)
    assert DefectGasWeightTuner(S, D_max=4).uniformProposalFraction == 0.5
    assert DefectGasWeightTuner(S, D_max=4,
                                uniformProposalFraction=None).uniformProposalFraction is None
