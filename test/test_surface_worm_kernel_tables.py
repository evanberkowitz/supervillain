import numpy as np
import supervillain
from supervillain.lattice import Lattice, d, wedge
from supervillain.generator.no_intersection.surface_worm import kernel

def test_scalar_green_inverts_laplacian():
    g0, se = kernel.scalar_green(4)
    src = np.zeros((4,)*4); src[0,0,0,0] = 1.0
    lap = sum(np.roll(g0, -1, m) + np.roll(g0, 1, m) - 2*g0 for m in range(4))
    assert np.allclose(-lap, src - 1/4**4, atol=1e-12)   # zero mode removed
    assert np.isclose(se, g0.reshape(-1)[0])

def test_stencils_match_library_d_and_wedge():
    N = 4
    rng = np.random.default_rng(0)
    L = Lattice(4, N)
    F = rng.integers(-2, 3, (6,)+(N,)*4).astype(np.int64)
    f = L.form(2); np.asarray(f)[...] = F
    dF, q = np.asarray(d(f)).astype(np.int64), np.asarray(wedge(f, f)).astype(np.int64)
    dsten, wsten = kernel.stencils(N)
    # toggling one plaquette by s changes dF on exactly dsten[c]'s 4 cells by s*sign
    c, x, s = 2, (1, 2, 3, 0), 1
    F2 = F.copy(); F2[(c,)+x] += s
    f2 = L.form(2); np.asarray(f2)[...] = F2
    dF2 = np.asarray(d(f2)).astype(np.int64)
    diff = dF2 - dF
    assert int((diff != 0).sum()) <= 4
    for (cc, off, sign) in dsten[c]:
        cell = (cc,) + tuple((x[i]+off[i]) % N for i in range(4))
        assert diff[cell] == s * sign
        diff[cell] = 0
    assert not diff.any()

def test_cob_tables_are_exact():
    N = 4
    S = supervillain.action.NoIntersections(Lattice(4, N), kappa=0.2)
    cob_pc, cob_off, cob_sign, Kcob = kernel.build_cob(S)
    L = S.Lattice
    for mu in range(4):
        F = np.zeros((6,)+(N,)*4, dtype=np.int64)
        for j in range(6):
            F[(int(cob_pc[mu,j]),) + tuple(int(v) % N for v in cob_off[mu,j])] += int(cob_sign[mu,j])
        f = L.form(2); np.asarray(f)[...] = F
        assert not np.asarray(d(f)).any()          # da is exact => closed
        assert F.sum() == 0                        # per-component totals cancel overall
