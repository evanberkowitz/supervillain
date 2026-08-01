import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice, d, wedge
from supervillain.generator.no_intersection.surface_worm.gas import SurfaceWormGas
from supervillain.generator.no_intersection.surface_worm.state import FState

# Vacuum-healthy defect prices, NOT the test_surface_worm_reference.py defaults
# (0.2, 0.3): at kappa=0.2 that pair sits in the "spanning network" branch
# (weights.py's own bimodality) where the class coordinate F.periods -- the
# off-shell H^2 direction legal_vacuum also gates on -- is UNCONFINED (pot()
# drops the zero Fourier mode, so C(F) never sees it, and the winding tilt
# prices a different linear functional of F, state.winding, not periods).
# Once the chain drifts there, the exactness-gated joint vacuum (D=Q=0 AND
# periods=0) starves: probed directly, the SAME chain that sits at
# periods=[0,0,-1,0,0,1] after 200k moves from cold still hasn't returned
# after 127M further moves (maxWaitTicks's warning is exactly this runtime
# symptom). Production regimes avoid this with a TUNED sector-weight table
# that keeps the closed shell populated (see weights.py); these tests use a
# strongly-suppressive bare fugacity instead, which keeps the chain hugging
# the closed shell so periods stays pinned at 0. Probed at (0.05, 0.1):
# P(legal_vacuum) = 0.545 over strided compiled-sweep samples, and a cold
# chain is already legal after its first 20k-move burn-in. kappa is NEVER
# changed here -- the oracle test's coset weight a = 2*pi**2*0.2/4**4 is
# keyed to kappa=0.2 specifically.
def _gas(kappa=0.2, **kw):
    S = supervillain.action.NoIntersections(Lattice(4, 4), kappa=kappa)
    kw.setdefault('openSurfaceFugacity', 0.05)
    kw.setdefault('intersectionFugacity', 0.1)
    kw.setdefault('ticksPerStep', 200)
    kw.setdefault('stride', 100)
    kw.setdefault('measure', False)
    return S, SurfaceWormGas(S, seed=9, **kw)

def test_emit_record_is_physical():
    S, g = _gas()
    st = FState(S)
    g.sweep(st, 50_000);
    while not st.legal_vacuum:
        g.sweep(st, 2_000)
    rec = g.emit(st, np.random.default_rng(1))
    n = np.asarray(rec['n']).astype(np.int64)
    nf = S.Lattice.form(1); np.asarray(nf)[...] = n
    assert np.array_equal(np.asarray(d(nf)).astype(np.int64), st.F)
    assert not np.asarray(wedge(d(nf), d(nf))).any()
    assert 'logWeight_SurfaceWormGas' not in rec        # the column is GONE

def test_emit_refuses_illegal_state():
    S, g = _gas()
    F = np.zeros((6,)+(4,)*4, dtype=np.int64); F[0, 0, 0, :, :] = 1
    with pytest.raises(ValueError):
        g.emit(FState(S, F), np.random.default_rng(0))

def test_winding_coset_resample_marginal():
    # the emitted total winding M sits in the coset M0 + c N^3 with the exact
    # Gaussian coset weights; chi^2-lite acceptance over 400 draws
    S, g = _gas(kappa=0.2)
    st = FState(S)                       # cold: F = 0, M0 = 0, quantum = 64
    rng = np.random.default_rng(4)
    Ms = []
    for _ in range(400):
        rec = g.emit(st, rng)
        Ms.append(np.asarray(rec['n']).astype(np.int64).reshape(4, -1).sum(axis=1))
    Ms = np.array(Ms)
    assert (Ms % 64 == 0).all()
    a = 2 * np.pi**2 * 0.2 / 4**4
    cs = np.arange(-4, 5)
    p = np.exp(-a * (cs * 64.0) ** 2); p /= p.sum()
    counts = np.array([(Ms[:, 0] // 64 == c).sum() for c in cs])
    Np = 400 * p
    assert (np.abs(counts - Np) <= 6 * np.sqrt(Np * (1 - p)) + 2).all()

def test_generator_protocol_with_ensemble_generate():
    S, g = _gas()
    g.equilibrate(20_000)
    e = supervillain.Ensemble(S).generate(4, g, start='cold')
    J = np.asarray(e.IntersectionWinding)
    assert J.shape == (4, 4)

def test_tilt_vs_reweight_equivalence():
    # THE test-only seam: untilted chain + hand reweight by Z_wind == tilted chain
    S, g_t = _gas(kappa=0.2)
    S2, g_u = _gas(kappa=0.2)
    g_u._windingCoefficient = 0.0        # private seam; do not add a public option
    a_phys = 2 * np.pi**2 * 0.2 / 4**4
    def logZ(M):
        # PHYSICAL winding partition function -- never the (possibly zeroed)
        # chain coefficient: the reweight corrects the untilted chain to physics.
        total = 0.0
        for mu in range(4):
            cs = np.arange(int(round(-M[mu]/64)) - 4, int(round(-M[mu]/64)) + 5)
            lw = -a_phys * (M[mu] + cs * 64.0) ** 2
            total += lw.max() + np.log(np.exp(lw - lw.max()).sum())
        return total
    def wrapping2(g, seed, rows=60):
        rng = np.random.default_rng(seed)
        st = FState(g.S); g.sweep(st, 20_000)
        vals, logws = [], []
        for _ in range(rows):
            g.sweep(st, 3_000)
            while not st.legal_vacuum:
                g.sweep(st, 500)
            rec = g.emit(st, rng)
            n = np.asarray(rec['n']).astype(np.int64)
            vals.append(float((n.reshape(4, -1).sum(axis=1) ** 2).sum()))
            logws.append(logZ(st.winding))
        v, lw = np.array(vals), np.array(logws)
        return v, lw
    vt, _ = wrapping2(g_t, 1)
    vu, lwu = wrapping2(g_u, 2)
    w = np.exp(lwu - lwu.max())
    tilted = vt.mean()
    reweighted = (w * vu).sum() / w.sum()
    scale = max(vt.std() / len(vt) ** 0.5, 1e-9)
    assert abs(tilted - reweighted) < 8 * scale      # loose: short chains
