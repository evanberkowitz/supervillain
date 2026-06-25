import numpy as np
import pytest
from supervillain.lattice import Lattice
from supervillain.lattice import reference as ref


def _apply_table_numpy(f, table, out_degree, combine, forward):
    # Re-derive an operator from its table with plain numpy, to prove the
    # table encodes the same incidence the reference computes.
    lat = f.lattice
    result = lat.zeros(out_degree, dtype=f.dtype)
    for out_idx, in_idx, axis, sign in table:
        spatial = f[in_idx]
        shifted = np.roll(spatial, -1 if forward else +1, axis=axis)
        if combine == "diff":
            result[out_idx] += sign * ((shifted - spatial) if forward else -(spatial - shifted))
        else:
            result[out_idx] += spatial + shifted
    return result


@pytest.mark.parametrize("D,N", [(D, N) for D in range(2, 6) for N in (3, 4, 5)])
def test_tables_reproduce_reference(D, N):
    L = Lattice(D=D, N=N)
    for p in range(D):
        f = L.random(p)
        got = _apply_table_numpy(f, L.operator_table("d", p), p + 1, "diff", True)
        assert (np.asarray(got) == np.asarray(ref.reference_d(f))).all()
        got = _apply_table_numpy(f, L.operator_table("coface_sum", p), p + 1, "sum", True)
        assert (np.asarray(got) == np.asarray(ref.reference_coface_sum(f))).all()
    for p in range(1, D + 1):
        f = L.random(p)
        got = _apply_table_numpy(f, L.operator_table("delta", p), p - 1, "diff", False)
        assert (np.asarray(got) == np.asarray(ref.reference_delta(f))).all()
        got = _apply_table_numpy(f, L.operator_table("face_sum", p), p - 1, "sum", False)
        assert (np.asarray(got) == np.asarray(ref.reference_face_sum(f))).all()
