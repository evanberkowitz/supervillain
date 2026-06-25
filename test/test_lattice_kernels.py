import numpy as np
import pytest
from math import comb
from supervillain.lattice import Lattice, d, delta
from supervillain.lattice import reference as ref
from supervillain.lattice import _kernels   # RED until Task 3 Step 3 creates it

_DN = [(D, N) for D in range(2, 6) for N in (3, 4, 5)]


def test_kernels_module_exposes_operator_tuples():
    # A genuine RED: these names don't exist until _kernels.py is written.
    for name in ("D_KERNELS", "DELTA_KERNELS", "FACE_KERNELS", "COFACE_KERNELS"):
        assert len(getattr(_kernels, name)) == 2  # (serial, parallel)


@pytest.mark.parametrize("D,N", _DN)
@pytest.mark.parametrize("dtype", [float, int])
def test_kernels_match_reference(D, N, dtype):
    L = Lattice(D=D, N=N)
    for p in range(D):
        f = L.form(p, dtype=dtype); f[...] = (np.random.default_rng(p).integers(-3, 4, f.shape)
                                              if dtype is int else np.random.default_rng(p).standard_normal(f.shape))
        assert (np.asarray(d(f)) == np.asarray(ref.reference_d(f))).all()
        assert np.asarray(d(f)).dtype == np.dtype(dtype)
        assert (np.asarray(f.coface_sum()) == np.asarray(ref.reference_coface_sum(f))).all()
        assert np.asarray(f.coface_sum()).dtype == np.dtype(dtype)
    for p in range(1, D + 1):
        f = L.form(p, dtype=dtype); f[...] = (np.random.default_rng(p).integers(-3, 4, f.shape)
                                              if dtype is int else np.random.default_rng(p).standard_normal(f.shape))
        assert (np.asarray(delta(f)) == np.asarray(ref.reference_delta(f))).all()
        assert np.asarray(delta(f)).dtype == np.dtype(dtype)
        assert (np.asarray(f.face_sum()) == np.asarray(ref.reference_face_sum(f))).all()
        assert np.asarray(f.face_sum()).dtype == np.dtype(dtype)


def test_scalar_zero_ends():
    L = Lattice(D=3, N=4)
    assert d(L.random(3)) == 0
    assert delta(L.random(0)) == 0
    assert L.random(0).face_sum() == 0
    assert L.random(3).coface_sum() == 0
