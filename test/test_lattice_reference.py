import numpy as np
import pytest
from supervillain.lattice import Lattice
from supervillain.lattice.compact import d, delta
from supervillain.lattice import reference as ref


@pytest.mark.parametrize("D,N", [(D, N) for D in range(2, 6) for N in (3, 4, 5)])
def test_reference_d_delta_match_current(D, N):
    L = Lattice(D=D, N=N)
    for p in range(D):
        f = L.random(p)
        assert (np.asarray(ref.reference_d(f)) == np.asarray(d(f))).all()
    for p in range(1, D + 1):
        f = L.random(p)
        assert (np.asarray(ref.reference_delta(f)) == np.asarray(delta(f))).all()


@pytest.mark.parametrize("D,N", [(D, N) for D in range(2, 6) for N in (3, 4, 5)])
def test_reference_face_coface_match_current(D, N):
    L = Lattice(D=D, N=N)
    for p in range(1, D + 1):
        f = L.random(p)
        assert (np.asarray(ref.reference_face_sum(f)) == np.asarray(f.face_sum())).all()
    for p in range(D):
        f = L.random(p)
        assert (np.asarray(ref.reference_coface_sum(f)) == np.asarray(f.coface_sum())).all()
