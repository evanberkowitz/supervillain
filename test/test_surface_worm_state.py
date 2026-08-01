import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice, d
from supervillain.generator.no_intersection.surface_worm.state import FState

def _S(N=4, kappa=0.2):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)

def test_cold_state_is_legal_vacuum():
    st = FState(_S())
    assert st.D == 0 and st.Q == 0 and st.legal_vacuum
    assert not st.periods.any() and not st.winding.any()

def test_from_configuration_roundtrip_and_direct_J():
    S = _S()
    # deterministic constraint-safe configuration: n has only a spatial component
    # with no x0 dependence, so every dn component carrying a 0-index vanishes and
    # q = dn ^ dn = 0 identically -- yet F and J are generically nonzero
    n = np.zeros((4,)+(4,)*4, dtype=np.int64)
    n[1][:, :, 0, :] = 1
    n[2][:, 1, :, :] = 1
    cfg = S.configurations(1); cfg[0] = {'n': n, 'phi': np.zeros((4,)*4)}
    st = FState.from_configuration(S, cfg[0])
    assert st.legal_vacuum
    e = supervillain.Ensemble(S).from_configurations(cfg)
    J_lib = np.asarray(e.IntersectionWinding).astype(np.int64)[0]
    assert np.array_equal(st.intersection_winding(), J_lib)

def test_class_one_sheet_is_not_legal():
    S = _S()
    F = np.zeros((6,)+(4,)*4, dtype=np.int64); F[0, 0, 0, :, :] = 1
    st = FState(S, F)
    assert st.D == 0 and st.Q == 0
    assert tuple(st.periods) == (16, 0, 0, 0, 0, 0)
    assert not st.legal_vacuum

def test_check_passes_on_fresh_state():
    S = _S()
    F = np.zeros((6,)+(4,)*4, dtype=np.int64); F[3, 1, 2, 0, 3] = 2
    st = FState(S, F)
    assert st.check()

def test_intersection_winding_rejects_class_one_sheet():
    # D=0, Q=0 but periods != 0 -- closed with a nonzero H^2 class, not a legal
    # vacuum (no n with dn=F exists). The numeric spread/distance alarms alone
    # demonstrably miss this: the FFT Green's function silently drops F's nonzero
    # mean, and the resulting slice sums for this particular F still land inside
    # spread_tol of each other and of an integer (they round to [0,0,0,0]), so an
    # explicit legal_vacuum guard is required as the first line of defense.
    S = _S()
    F = np.zeros((6,)+(4,)*4, dtype=np.int64); F[0, 0, 0, :, :] = 1
    st = FState(S, F)
    assert not st.legal_vacuum
    with pytest.raises(ValueError):
        st.intersection_winding()

def test_intersection_winding_rejects_open_F():
    # A single toggled plaquette: D != 0 (dF is generically nonzero), so the same
    # legal_vacuum guard fires -- this is the more common off-shell case a worm
    # move passes through mid-proposal.
    S = _S()
    F = np.zeros((6,)+(4,)*4, dtype=np.int64); F[0, 0, 0, 0, 0] = 1
    st = FState(S, F)
    assert st.D != 0
    assert not st.legal_vacuum
    with pytest.raises(ValueError):
        st.intersection_winding()
