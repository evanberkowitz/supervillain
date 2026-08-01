import numpy as np
import pytest
import h5py as h5
from supervillain.generator.no_intersection.surface_worm.weights import (
    SectorWeights, PairUmbrella, pair_separation_squared)

def test_fugacity_table_is_linear():
    w = SectorWeights.fugacity(0.09, cap=16)
    lg = np.log(0.09)
    for D in range(17):
        assert np.isclose(w(D), D * lg)
    assert np.isclose(w.delta(3, 5), 2 * lg)

def test_hard_wall_is_minus_inf():
    w = SectorWeights.fugacity(0.09, cap=8)
    assert w(9) == -np.inf and w(8) > -np.inf

def test_logweight_anchored_at_zero():
    w = SectorWeights(np.array([3.0, 4.0, 6.0]), tailSlope=-2.0)
    assert w.logWeight[0] == 0.0 and np.isclose(w(2), 3.0)

def test_pair_umbrella_off_is_identity():
    u = PairUmbrella.off(4)
    assert u.logW(7) == 0.0 and u.logW(None) == 0.0
    assert u.delta(None, 5) == 0.0 and u.weight(9) == 1.0

def test_pair_umbrella_shape_check():
    with pytest.raises(ValueError):
        PairUmbrella(np.zeros(5), N=4)

def test_pair_separation_squared():
    assert pair_separation_squared(None, 4) is None
    assert pair_separation_squared({(0,0,0,0): 1, (1,0,0,0): -1}, 4) == 1
    assert pair_separation_squared({(0,0,0,0): 1, (3,0,0,0): -1}, 4) == 1   # minimal image
    assert pair_separation_squared({(0,0,0,0): 1, (1,0,0,0): 1}, 4) is None  # not +/-1

def test_readwriteable_roundtrip(tmp_path):
    w = SectorWeights.fugacity(0.09, cap=8)
    u = PairUmbrella(np.linspace(0, 2, 17), N=4)
    with h5.File(tmp_path / 'w.h5', 'w') as f:
        w.to_h5(f.create_group('w')); u.to_h5(f.create_group('u'))
    with h5.File(tmp_path / 'w.h5', 'r') as f:
        w2 = SectorWeights.from_h5(f['w']); u2 = PairUmbrella.from_h5(f['u'])
    assert np.allclose(w2.logWeight, w.logWeight) and w2.hardWall == w.hardWall
    assert np.allclose(u2.logWeight, u.logWeight) and u2.N == 4
