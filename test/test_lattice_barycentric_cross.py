import numpy as np
import pytest

import supervillain.lattice.barycentric.doubled as dbl
import supervillain.lattice.barycentric.compact as cpt


def _pair(D, N):
    return cpt.Lattice(D, N), dbl.Lattice(D, N)


def test_round_trip_is_the_identity():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        for p in range(D + 1):
            a = clat.random(p)
            back = cpt.Form.from_doubled(a.to_doubled(), clat)
            assert not np.allclose(np.asarray(a), 0), f'D={D} p={p}: vacuous'
            assert np.allclose(np.asarray(back), np.asarray(a)), (D, p)
            assert back.degree == p and back.dual is False


def test_to_doubled_lands_only_on_real_simplices():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, dlat = _pair(D, N)
        for p in range(D + 1):
            f = clat.random(p).to_doubled()
            assert (np.asarray(f)[dlat.support(p) == 0] == 0).all(), (D, p)
            assert (np.asarray(f) != 0).sum() == clat.count(p), (D, p)


def test_from_doubled_ignores_nothing():
    """Every real slot of the doubled form reaches a compact component."""
    for D, N in ((2, 3), (4, 3)):
        clat, dlat = _pair(D, N)
        for p in range(D + 1):
            f = dlat.random(p)
            a = cpt.Form.from_doubled(f, clat)
            assert np.isclose(np.asarray(a).sum(), np.asarray(f).sum()), (D, p)


def test_to_doubled_refuses_a_dual_form():
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError):
        clat.zeros(2, dual=True).to_doubled()
