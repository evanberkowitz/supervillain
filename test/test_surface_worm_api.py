#!/usr/bin/env python
r"""Public and subpackage API surface for the Surface Worm Gas: the exports the
migration promises, and the confirmation that the ``windingInSampler`` toggle
(dropped when the winding tilt became unconditionally physical) has not crept
back into the constructor.
"""


def test_public_api():
    from supervillain.generator.no_intersection import (
        SurfaceWormGas, SectorWeightTuner, PairUmbrellaTuner)


def test_subpackage_api():
    from supervillain.generator.no_intersection.surface_worm import (
        SurfaceWormGas, SectorWeights, PairUmbrella, FState,
        CorrelatorAccumulator, SectorWeightTuner, PairUmbrellaTuner)


def test_no_winding_option_remains():
    import inspect
    from supervillain.generator.no_intersection import SurfaceWormGas
    assert 'windingInSampler' not in inspect.signature(SurfaceWormGas.__init__).parameters
