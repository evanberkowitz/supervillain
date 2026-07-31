# Surface Worm Gas Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the exactness-gated `SurfaceWormGas` instrument from the notebook toolchain into `supervillain/generator/no_intersection/surface_worm/`, refactored to library idioms (spec: `docs/superpowers/specs/2026-07-31-surface-worm-migration-design.md`).

**Architecture:** Seven focused modules under a new subpackage; the gas and its Generator adapter merge into one `SurfaceWormGas(ReadWriteable, Generator)` per the DefectGas precedent; state lives in an `FState` class instead of a dict; the winding tilt is always on (no option, no weight column); the exactness (periods) gate is baked into `legal_vacuum`, `reconstruct`, and the accumulator.

**Tech Stack:** numpy, numba (`@njit(cache=True)`), the library's `Lattice`/`d`/`delta`/`wedge`, `ReadWriteable` h5 persistence, pytest.

## Global Constraints

- Branch: `feature/surface-worm` in `/Users/evanberkowitz/physics/supervillain/library`.
- **Port source of truth** (audited, gates-passing): `/Users/evanberkowitz/physics/supervillain/no-intersections/swg-audit-2026-07-31/` — referred to below as `AUDIT/`. Tuner sources live in `/Users/evanberkowitz/physics/supervillain/no-intersections/j-vacuum-2026-07-31/` — referred to as `JVAC/`. Never modify either directory.
- Test runner (always this invocation, from the library root):
  `NUMBA_CACHE_DIR=/tmp/numba_sessions/swg-migrate .venv/bin/python -m pytest test/<file> -v`
- Renames applied throughout (spec "Naming"): `eta_dF`/`etaDF` → `openSurfaceFugacity`; `eta_q`/`etaQ` → `intersectionFugacity`. No other public-name churn.
- **No `windingInSampler` anywhere.** The tilt coefficient is `self._windingCoefficient = 2 * np.pi**2 * kappa / V`, set unconditionally in the constructor; tests may overwrite the attribute (the test-only seam). `emit` returns a record only — no log-weight.
- All new files carry sphinx-style docstrings (the AUDIT sources are already close; keep their physics notes, drop notebook-finding cross-references or rewrite them as self-contained statements).
- Every commit message ends with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- p_cob/pCob spelling: the library subpackage uses `pCob` everywhere (constructor and method arguments).

---

### Task 1: Subpackage skeleton + `weights.py`

**Files:**
- Create: `supervillain/generator/no_intersection/surface_worm/__init__.py`
- Create: `supervillain/generator/no_intersection/surface_worm/weights.py`
- Test: `test/test_surface_worm_weights.py`

**Interfaces:**
- Consumes: `supervillain.h5.ReadWriteable` (existing).
- Produces: `SectorWeights(logWeight, tailSlope, hardWall=True)` with `.fugacity(openSurfaceFugacity, cap=64)` classmethod, `.__call__(D)`, `.delta(old, new)`, `.arrays()`, `.flatness(histogram)`, `.interpolated(visited)`, `.cap`, `.hardWall`; `PairUmbrella(logWeight, N)` with `.off(N)`, `.logW(r2)`, `.delta(oldR2, newR2)`, `.weight(r2)`, `.normalized(occupancy)`, `.shifted(delta)`, `.rebalanced(sectorTicks, totalTicks, targetOdds)`; `pair_separation_squared(chargeSites, N)`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_surface_worm_weights.py
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
```

- [ ] **Step 2: Run to verify failure** — expected: `ModuleNotFoundError: ...surface_worm`.
- [ ] **Step 3: Implement.** Create the subpackage `__init__.py` containing only a one-line placeholder docstring (the narrative arrives in Task 11). Create `weights.py`: copy `AUDIT/sector_weights.py` lines 41–191 (`SectorWeights` + module docstring, keeping the physics notes) and `AUDIT/pair_umbrella.py` lines 31–153 (`PairUmbrella`, `pair_separation_squared`) into the one module. Transformations: rename `fugacity(cls, etaDF, cap=64)` → `fugacity(cls, openSurfaceFugacity, cap=64)` (update its body's `lg = np.log(openSurfaceFugacity)` and docstring); rewrite docstring references to notebook findings as self-contained sentences.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `surface-worm: weights (SectorWeights, PairUmbrella)`.

---

### Task 2: `staircase.py`

**Files:**
- Create: `supervillain/generator/no_intersection/surface_worm/staircase.py`
- Test: `test/test_surface_worm_staircase.py`

**Interfaces:**
- Produces: `d0(lam)`, `d1(a)`, `primitive_1form(m)`, `primitive_2form(b)` (integer arrays; `primitive_2form` takes a `(C(D,2),)+(N,)*D` integer 2-form and returns a `(D,)+(N,)*D` integer 1-form). `primitive_2form` remains **permissive** on non-exact input by design (spec Behavior 1) — its docstring must say so and point to `reconstruct` as the gate.

- [ ] **Step 1: Write the failing test** — port `AUDIT/staircase.py` `_gate2` (lines 156–163) and `_gate_generic` (lines 166–182) as two pytest functions `test_primitive2_2d` / `test_primitive_general_D` (assert `d(a) == b` exactly on random exact forms, D = 2, 3, 4, N = 3, 4, 5); add:

```python
def test_primitive2_raises_on_flux():
    import numpy as np, pytest
    from supervillain.generator.no_intersection.surface_worm.staircase import primitive2
    with pytest.raises(ValueError):
        primitive2(np.ones((4, 4), dtype=np.int64))

def test_primitive_2form_is_permissive_and_linear():
    # deliberately NO raise on closed-non-exact input: the sampler uses it as a linear map
    import numpy as np
    from supervillain.generator.no_intersection.surface_worm.staircase import primitive_2form
    F = np.zeros((6, 4, 4, 4, 4), dtype=np.int64); F[0, 0, 0, :, :] = 1
    a = primitive_2form(F)          # returns without raising
    b = primitive_2form(2 * F)
    assert np.array_equal(b, 2 * a)  # linearity
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — copy `AUDIT/staircase.py` lines 1–151 verbatim (docstring through `primitive_2form`), dropping the `__main__` gate block; extend the module docstring with the permissive-by-design note above.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `surface-worm: staircase integer primitives`.

---

### Task 3: `kernel.py` part 1 — tables, stencils, Green machinery

**Files:**
- Create: `supervillain/generator/no_intersection/surface_worm/kernel.py`
- Test: `test/test_surface_worm_kernel_tables.py`

**Interfaces:**
- Produces (all consumed by Tasks 4/6/7): `scalar_green(N) -> (g0, self_energy)`; `pot(Fc, N)`; `stencils(N) -> (dsten, wsten)` (the d-stencil `dsten[c] = [(cc, off, sign) x4]` and wedge stencil, cached per N); `build_arrays(N)` (the flat numba tables — was `_build_arrays`); `build_cob(S) -> (cob_pc, cob_off, cob_sign, Kcob)`; `group_hrel(N)` (was `gas._group_hrel`); `green_add` and `nb_seed` (njit helpers, was `_green_add`/`_nb_seed`).

- [ ] **Step 1: Write the failing test**

```python
# test/test_surface_worm_kernel_tables.py
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
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — assemble `kernel.py` from: `AUDIT/worm.py` lines 35–51 (`scalar_green`, `pot`); `AUDIT/stencils.py` (the `_build`/`get` machinery — rename `get` → `stencils`); `AUDIT/worm_numba.py` lines 26–72 (`_build_arrays` → `build_arrays`), 74–78 (`_nb_seed` → `nb_seed`), 98–109 (`_green_add` → `green_add`), 357–382 (`_build_cob` → `build_cob`); `AUDIT/gas.py` lines 838–848 (`_group_hrel` → `group_hrel`). Do NOT port the old-worm classes (`NumbaBoundaryWorm`, `_worm_batch`, `NumbaLocalCoboundary`, …).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `surface-worm: kernel tables (stencils, Green, coboundary, numba helpers)`.

---

### Task 4: `state.py` — FState

**Files:**
- Create: `supervillain/generator/no_intersection/surface_worm/state.py`
- Test: `test/test_surface_worm_state.py`

**Interfaces:**
- Consumes: `kernel.pot`, `kernel.scalar_green`; `staircase.primitive_2form`; library `Lattice, d, wedge, delta`.
- Produces: `FState(S, F=None)` (None → cold zeros; else integer `(6,)+(N,)*4`), `FState.from_configuration(S, configuration)` (F = d(n)); attributes `F, dF, q, G, counts, winding, periods, absoluteCharge, squaredCharge, chargeSites`; properties `D`, `Q` (read `counts`), `legal_vacuum` (`D==0 and Q==0 and not periods.any()`); methods `refresh_charge()` (O(V) rebuild after compiled batches), `recomputed()` (dict of from-scratch dF/q/D/Q/winding/periods/C), `check(atol=1e-9)` (compares incremental vs recomputed, raises `AssertionError` on mismatch, returns True), `intersection_winding(spread_tol=1e-8)` (direct-from-F J).

- [ ] **Step 1: Write the failing test**

```python
# test/test_surface_worm_state.py
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
```

(If the random-`n` seed in `test_from_configuration_roundtrip_and_direct_J` trips the
constraint, replace it with the deterministic spatial construction: `n[1] = z(x2)` any
integer profile constant in `x0, x1, x3` has `q ≡ 0`; use
`n = np.zeros(...); n[1][:, :, 0, :] = 1` which is exact-constraint-safe and carries
nonzero F.)

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.** Port the derived-state construction from `AUDIT/gas.py` `_init` (lines 286–314: dF, q, G via `pot`, D, Q, charge bookkeeping, `winding` via `primitive_2form` sum, `periods` component totals) into the constructor; `refresh_charge` from `AUDIT/gas.py` `_refresh_charge_state` (lines 775–790). `recomputed()`/`check()` follow `_log_extended_weight`'s global recomputes (lines 333–348) plus periods/winding. `intersection_winding()` implements the validated construction from `/Users/evanberkowitz/physics/supervillain/no-intersections/swg-audit-2026-07-31/probe_direct_J.py` (green_inverse → `delta` → `wedge(a, F)` → slice sums; slice-spread and integer-distance alarms raise `ValueError` beyond `spread_tol`), reusing the already-maintained `G` (which IS `Δ⁻¹F` per component) instead of a fresh FFT. `from_configuration` builds `F = d(configuration['n'])`.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `surface-worm: FState (incremental extended-ensemble state, legal_vacuum, direct J)`.

---

### Task 5: `reconstruct.py`

**Files:**
- Create: `supervillain/generator/no_intersection/surface_worm/reconstruct.py`
- Test: `test/test_surface_worm_reconstruct.py`

**Interfaces:**
- Consumes: `staircase.primitive_2form`; library `d`.
- Produces: `reconstruct_n(S, F, M)` (raises `ValueError` on non-exact F: `dF != 0` or any nonzero per-plane period map), `draw_phi(S, n, rng)`, `fft_symbols(N)` (inlined from the notebook's `surgery.py`).

- [ ] **Step 1: Write the failing test** — port `AUDIT/reconstruct.py` gates as pytest: `test_raises_on_class` (the class-1 sheet), `test_raises_on_open` (single plaquette), `test_roundtrip` (build 10 valid F = d(n) samples from the deterministic spatial construction plus random exact shifts, assert `d(reconstruct_n(S,F,M)) == F`, winding `Σn == M`, `q == 0`), `test_winding_quantum_mismatch_raises` (M off by a non-multiple of N³ → ValueError), `test_phi_conditional_action` (with `draw_phi`, the action density `0.5κΣ(dφ−2πn)²/V` over 50 draws has the equipartition-scale mean — port the check from `AUDIT/reconstruct.py` `gate_observables` in miniature, tolerance 5σ).
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — copy `AUDIT/reconstruct.py` lines 27–95 (module docstring, `reconstruct_n` INCLUDING the exactness raise added by the audit fix, `draw_phi`, `reconstruct`, `build_ensemble` — drop the last two: the gas emits directly and nothing else consumes them, YAGNI) plus `fft_symbols` from `AUDIT/surgery.py` lines 43–50 inlined at module top. Drop the gate functions (they became the tests).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `surface-worm: reconstruction (n-ification gated on exactness, exact phi draw)`.

---

### Task 6: `gas.py` — constructor + python-reference moves + detailed-balance gate

**Files:**
- Create: `supervillain/generator/no_intersection/surface_worm/gas.py`
- Test: `test/test_surface_worm_reference.py`

**Interfaces:**
- Consumes: `FState`, `SectorWeights`, `PairUmbrella`, `pair_separation_squared`, `kernel.*`, `staircase.primitive_2form`.
- Produces: `SurfaceWormGas(ReadWriteable, Generator)` with constructor signature
  `(S, openSurfaceFugacity=None, sectorWeights=None, intersectionFugacity=0.3, sectorWeightCap=64, targetFraction=0.0, pairUmbrella=None, ticksPerStep=1000, stride=200, pCob=0.5, maxWaitTicks=200000, hardWaitFactor=10, measure=True, seed=None, rng=None, absoluteChargeCap=64, squaredChargeCap=64, chargeBinWidth=1)`;
  this task delivers: the guardrails (exactly one of `openSurfaceFugacity`/`sectorWeights`; `targetFraction ∈ [0,1)`), `self._windingCoefficient` set unconditionally, the `windingSensitivity` precompute, `sweep_reference(state, nmoves, pCob=None)` (python moves), `_plaquette_log_acceptance` and `_coboundary_log_weights` split out for gates, `_log_extended_weight(F)` global recompute, `setSectorWeights(weights)`. (The numba `sweep`, `emit`, and the Generator protocol arrive in Tasks 7–8; until then the class simply doesn't define them.)

- [ ] **Step 1: Write the failing test**

```python
# test/test_surface_worm_reference.py
import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.gas import SurfaceWormGas
from supervillain.generator.no_intersection.surface_worm.state import FState

def _gas(N=4, kappa=0.2, **kw):
    S = supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)
    kw.setdefault('openSurfaceFugacity', 0.2)
    kw.setdefault('intersectionFugacity', 0.3)
    return S, SurfaceWormGas(S, seed=5, **kw)

def test_constructor_guardrails():
    S = supervillain.action.NoIntersections(Lattice(4, 4), kappa=0.2)
    with pytest.raises(ValueError):
        SurfaceWormGas(S)                                        # neither price
    with pytest.raises(ValueError):
        SurfaceWormGas(S, openSurfaceFugacity=0.1,
                       sectorWeights=object())                   # both prices
    with pytest.raises(ValueError):
        SurfaceWormGas(S, openSurfaceFugacity=0.1, targetFraction=1.0)

def test_tilt_is_always_on():
    S, g = _gas()
    assert g._windingCoefficient == pytest.approx(2 * np.pi**2 * 0.2 / 4**4)

def test_plaquette_acceptance_matches_global_recompute():
    S, g = _gas()
    st = FState(S)
    rng = np.random.default_rng(3)
    twopi2k = 2 * np.pi**2 * g.kappa
    for _ in range(60):
        c = int(rng.integers(6)); x = tuple(int(v) for v in rng.integers(4, size=4))
        s = 1 if rng.random() < 0.5 else -1
        before = g._log_extended_weight(st.F)
        lnA, dD, dQ, cube_new, q_new, idx = g._plaquette_log_acceptance(st, c, x, s)
        # apply unconditionally to walk into varied states
        g._apply_plaquette(st, c, x, s, dD, dQ, cube_new, q_new, idx)
        after = g._log_extended_weight(st.F)
        # the acceptance exponent must equal the true Delta log pi_ext up to the
        # Hastings proposal correction, which the global recompute does not carry;
        # with targetFraction=0 that correction is identically zero.
        assert lnA == pytest.approx(after - before, abs=1e-8)

def test_invariants_after_reference_sweep():
    S, g = _gas()
    st = FState(S)
    g.sweep_reference(st, 3000)
    assert st.check()
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.** Port from `AUDIT/gas.py`: constructor logic lines 112–277 with the transformations — renames per Global Constraints; delete the `windingInSampler` parameter and branch (set `self.samplerWindingCoefficient` → `self._windingCoefficient` unconditionally to the physical value); dict-state code becomes `FState` attribute access (`cfg['D']` → `state.D` via `state.counts`, `cfg['chargeSites']` → `state.chargeSites`, `cfg['winding']` → `state.winding`, `cfg['periods']` → `state.periods`); accept the new cadence parameters and store them (used in Task 8); store `self.measure = bool(measure)` and `self.accumulator = None` — the accumulator is constructed in Task 9; this task's tests never measure. Port `_log_proposal_density`, `_propose_plaquette`, `_plaquette_log_acceptance`, `_charge_sites_after`, `_plaquette_move`, `_apply_plaquette`, `_coboundary_log_weights`, `_coboundary_heatbath`, `sweep_reference` (was `step_reference`), `_log_extended_weight`, `_log_winding_weight`, `winding_of`, `setSectorWeights` from their AUDIT/gas.py bodies (lines 299–618, 745–767), with the same dict→FState mechanical change; `_plaquette_log_acceptance(self, state, c, x, s)` drops the redundant F/dF/q/G/N/twopi2k arguments (read them off `state`/`self`).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `surface-worm: SurfaceWormGas reference moves with detailed-balance gate`.

---

### Task 7: `kernel.py` part 2 — the batch kernel + `sweep`

**Files:**
- Modify: `supervillain/generator/no_intersection/surface_worm/kernel.py` (append)
- Modify: `supervillain/generator/no_intersection/surface_worm/gas.py` (add `_build_nb`, `sweep`, `sweep_measured` minus ticking)
- Test: `test/test_surface_worm_sweep.py`

**Interfaces:**
- Produces: `kernel.gas_batch(...)` — the `@njit` batch (was `_nb_gas_batch`), same argument list as `AUDIT/gas.py` lines 995–1004 **including the `periods` array**, with helper njits `log_winding_1d`, `log_proposal_density_nb`, `log_sector_weight`, `sep2_flat`, `pair_log_umbrella`, `cob_log_umbrella`; `SurfaceWormGas.sweep(state, nmoves, pCob=None)` (numba path, defaults `pCob` to the constructor value).

- [ ] **Step 1: Write the failing test**

```python
# test/test_surface_worm_sweep.py
import numpy as np
import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.gas import SurfaceWormGas
from supervillain.generator.no_intersection.surface_worm.state import FState

def test_sweep_invariants_through_charged_excursions():
    S = supervillain.action.NoIntersections(Lattice(4, 4), kappa=0.03)
    g = SurfaceWormGas(S, openSurfaceFugacity=0.09, intersectionFugacity=0.1,
                       targetFraction=0.8, seed=12, measure=False)
    st = FState(S)
    g.sweep(st, 200_000)
    st.refresh_charge()
    assert st.check()          # periods, winding, D, Q, dF, q, G all match recompute

def test_sweep_and_reference_agree_statistically():
    # same target, two kernels: mean D and mean Q agree loosely over short runs
    S = supervillain.action.NoIntersections(Lattice(4, 4), kappa=0.2)
    def meanD(sweeper, seed, moves=40_000, samples=40):
        g = SurfaceWormGas(S, openSurfaceFugacity=0.2, intersectionFugacity=0.3,
                           seed=seed, measure=False)
        st = FState(S); out = []
        for _ in range(samples):
            getattr(g, sweeper)(st, moves // samples)
            out.append(st.D)
        return np.mean(out), np.std(out) / len(out) ** 0.5
    m1, e1 = meanD('sweep', 21)
    m2, e2 = meanD('sweep_reference', 22)
    assert abs(m1 - m2) < 5 * np.hypot(e1, e2) + 0.5
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — move `AUDIT/gas.py` lines 851–1369+ (the njit helpers and `_nb_gas_batch`, WITH the periods updates at both commit points — plaquette commit `periods[c] += s`, coboundary commit `periods[pc] += D * cob_sign[mu, j]`) into `kernel.py` under the public names above, unchanged logic. In `gas.py`, port `_build_nb` (AUDIT lines 672–718) and `step` → `sweep` (lines 719–744): the winding/periods arrays now live in `state.winding`/`state.periods` (`np.ascontiguousarray` round-trip exactly as the audit fix does).
- [ ] **Step 4: Run to verify pass** (the statistical test takes ~1 min; that is acceptable).
- [ ] **Step 5: Commit** — `surface-worm: numba batch kernel and sweep`.

---

### Task 8: `gas.py` — emit + the Generator protocol

**Files:**
- Modify: `supervillain/generator/no_intersection/surface_worm/gas.py`
- Test: `test/test_surface_worm_emit.py`

**Interfaces:**
- Produces: `emit(state, rng=None) -> dict` (record with `'n'`, `'phi'`, plus harvest keys when measuring; **no logWeight**; raises `ValueError` if `not state.legal_vacuum`); `warm_start(configuration)` (chain state ← `FState.from_configuration`); Generator protocol `step(configuration) -> dict` (advance `ticksPerStep × stride` moves — via `sweep` when `measure=False`, `sweep_measured` once Task 9 lands ticking — then advance in 10-tick chunks until `legal_vacuum`, warning at `maxWaitTicks`, `RuntimeError` at `hardWaitFactor ×` that, then emit); `inline_observables(steps)`; `equilibrate(moves)`; `report()`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_surface_worm_emit.py
import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice, d, wedge
from supervillain.generator.no_intersection.surface_worm.gas import SurfaceWormGas
from supervillain.generator.no_intersection.surface_worm.state import FState

def _gas(kappa=0.2, **kw):
    S = supervillain.action.NoIntersections(Lattice(4, 4), kappa=kappa)
    kw.setdefault('openSurfaceFugacity', 0.2)
    kw.setdefault('intersectionFugacity', 0.3)
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
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.** `emit` ports `AUDIT/gas.py` lines 624–678 with: the legal-vacuum raise replacing the periods check; **the winding resample always uses the physical coefficient** `2π²κ/V` regardless of `self._windingCoefficient` (the seam detunes the *chain*, never the emitted conditional); no logZ return — return the record only; harvest attached when `measure=True`. `step`/`inline_observables`/`report`/`equilibrate` port `AUDIT/swg_generator.py` lines 59–137 onto the class (chain state in `self._state`, lazily `FState(S)`; `warm_start(configuration)` sets it from a configuration; `inline_observables` returns `{}` when `measure=False` and defers harvest-key sizing to Task 9).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `surface-worm: emit and Generator protocol; winding tilt always on`.

---

### Task 9: `accumulator.py`

**Files:**
- Create: `supervillain/generator/no_intersection/surface_worm/accumulator.py`
- Modify: `supervillain/generator/no_intersection/surface_worm/gas.py` (construct accumulator when `measure=True`; `sweep_measured(state, nticks, stride=None, pCob=None)`; `step` uses it; `inline_observables` sizes from a fresh template harvest)
- Test: `test/test_surface_worm_accumulator.py`

**Interfaces:**
- Produces: `CorrelatorAccumulator(N, intersectionFugacity, absoluteChargeCap=64, squaredChargeCap=64, chargeBinWidth=1, sectorCap=64)` with `.tick(state)`, `.harvest()`, `.reset()`, `.pairUmbrella` (assigned by the gas to the SAME object the acceptance uses), and the audit's `NontrivialClassTicks` exclusion.

- [ ] **Step 1: Write the failing test**

```python
# test/test_surface_worm_accumulator.py
import numpy as np
import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.accumulator import CorrelatorAccumulator
from supervillain.generator.no_intersection.surface_worm.state import FState

def _S(kappa=0.2):
    return supervillain.action.NoIntersections(Lattice(4, 4), kappa=kappa)

def test_vacuum_and_class_classification():
    acc = CorrelatorAccumulator(4, 0.3)
    st = FState(_S())                      # legal vacuum
    acc.tick(st)
    F = np.zeros((6,)+(4,)*4, dtype=np.int64); F[0, 0, 0, :, :] = 1
    acc.tick(FState(_S(), F))              # closed, q=0, class 1: EXCLUDED
    h = acc.harvest()
    assert h['VacuumTicks'] == 1 and h['ClosedTicks'] == 1
    assert h['NontrivialClassTicks'] == 1

def test_pair_bin_divides_umbrella_out():
    from supervillain.generator.no_intersection.surface_worm.weights import PairUmbrella
    acc = CorrelatorAccumulator(4, 0.3)
    u = PairUmbrella(np.log(np.full(17, 2.0)), 4)   # w2 = 2 everywhere
    acc.pairUmbrella = u
    st = FState(_S())
    # manufacture a +/-1 pair state at separation 1 by hand
    st.q[0, 0, 0, 0] = 1; st.q[1, 0, 0, 0] = -1
    st.counts[1] = 2
    st.chargeSites = {(0, 0, 0, 0): 1, (1, 0, 0, 0): -1}
    st.absoluteCharge = 2; st.squaredCharge = 2
    acc.tick(st)
    h = acc.harvest()
    dx = tuple(np.argwhere(h['Theta_Theta'] != 0)[0])
    V = 4 ** 4
    assert np.isclose(h['Theta_Theta'][dx], (1 / 2.0) / (V * 0.3 ** 2))

def test_harvest_resets():
    acc = CorrelatorAccumulator(4, 0.3)
    acc.tick(FState(_S()))
    acc.harvest()
    assert acc.harvest()['VacuumTicks'] == 0
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — port `AUDIT/correlator.py` (the whole `CorrelatorAccumulator`, ~200 lines, including `NontrivialClassTicks` init/tick/harvest) with: `etaQ` → `intersectionFugacity` throughout; `tick(cfg)` → `tick(state)` reading `state.Q`, `state.D`, `state.periods`, `state.chargeSites`, `state.absoluteCharge`, `state.squaredCharge`. In `gas.py`: construct it in `__init__` when `measure=True` and share `self.pairUmbrella` with it; `sweep_measured` ports `AUDIT/gas.py` lines 786–828 (`sweep` per stride + `state.refresh_charge()` + `acc.tick(state)`); `step` switches to `sweep_measured` when measuring; `inline_observables` ports `AUDIT/swg_generator.py` lines 117–137 minus the logWeight entry.
- [ ] **Step 4: Run to verify pass**, and re-run Task 8's tests (the generator path changed).
- [ ] **Step 5: Commit** — `surface-worm: correlator accumulator with exactness-gated dwells`.

---

### Task 10: `tuners.py`

**Files:**
- Create: `supervillain/generator/no_intersection/surface_worm/tuners.py`
- Test: `test/test_surface_worm_tuners.py`

**Interfaces:**
- Produces: `SectorWeightTuner(S, intersectionFugacity, cap, openSurfaceFugacity=0.09, iterations=20, ticks=2000, stride=200, damping=0.7, targetFlatness=1.5, minimumReachable=3, pCob=0.6, targetFraction=0.8, seed=None)` with `.tune() -> SectorWeights` and `.history` (list of per-iteration dicts: `flatness`, `visited`, `histogram`); `PairUmbrellaTuner(S, sectorWeights, intersectionFugacity, targetFraction, pCob, iterations=20, ticks=6000, stride=100, equilibrate=2_000_000, targetOdds=0.5, seed=None)` with `.tune() -> PairUmbrella` and `.history`. Both drive a `SurfaceWormGas(measure=False)` internally via `sweep`, reusing one gas across iterations through `setSectorWeights` (the multicanonical warm start).

- [ ] **Step 1: Write the failing test**

```python
# test/test_surface_worm_tuners.py
import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.tuners import (
    SectorWeightTuner, PairUmbrellaTuner)
from supervillain.generator.no_intersection.surface_worm.weights import (
    SectorWeights, PairUmbrella)

def _S():
    return supervillain.action.NoIntersections(Lattice(4, 4), kappa=0.2)

def test_sector_weight_tuner_smoke():
    t = SectorWeightTuner(_S(), intersectionFugacity=0.3, cap=8,
                          iterations=3, ticks=300, stride=50, seed=1)
    w = t.tune()
    assert isinstance(w, SectorWeights) and w.cap == 8
    assert len(t.history) == 3 and 'flatness' in t.history[0]

def test_pair_umbrella_tuner_smoke():
    w = SectorWeights.fugacity(0.2, cap=8)
    t = PairUmbrellaTuner(_S(), sectorWeights=w, intersectionFugacity=0.3,
                          targetFraction=0.5, pCob=0.5,
                          iterations=2, ticks=300, stride=50, equilibrate=5_000, seed=2)
    u = t.tune()
    assert isinstance(u, PairUmbrella) and u.N == 4
    assert len(t.history) == 2
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.** `SectorWeightTuner` wraps `JVAC/tune_sector_weights.py` `visit_histogram` (lines 48–64) and `tune` (lines 66–211) as methods — the free-function bodies move essentially verbatim; renames per Global Constraints; the CLI block is not ported. `PairUmbrellaTuner` wraps `JVAC/tune_pair_umbrella.py` `achievable_shells` (57–82), `measure` (84–94), `update` (96–143), `OffsetBisector` (145–197, kept as a private module class), `coverage` (199–205); its docstring carries the known limitation verbatim in spirit: *the bisection tuner oscillates at N=4 between high-occupancy and dead-sector tables (observed 2026-07-31); inspect `.history` and prefer the best iteration rather than the last.* Both tuners construct one internal `SurfaceWormGas(..., measure=False)` and iterate `sweep` + table updates via `setSectorWeights` / rebuilding `PairUmbrella` (the umbrella table is passed to the gas constructor per iteration — port the pattern from the JVAC script faithfully).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `surface-worm: sector-weight and pair-umbrella tuners`.

---

### Task 11: Exports, narrative docs, changes.rst, full suite

**Files:**
- Modify: `supervillain/generator/no_intersection/surface_worm/__init__.py`
- Modify: `supervillain/generator/no_intersection/__init__.py`
- Modify: `changes.rst`
- Test: `test/test_surface_worm_api.py`

**Interfaces:**
- Produces: `from supervillain.generator.no_intersection import SurfaceWormGas, SectorWeightTuner, PairUmbrellaTuner` (the public API); subpackage exports additionally `SectorWeights, PairUmbrella, FState, CorrelatorAccumulator` (power users, unadvertised).

- [ ] **Step 1: Write the failing test**

```python
# test/test_surface_worm_api.py
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
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.** Subpackage `__init__.py`: the narrative docstring (extended ensemble `pi_ext(F) ∝ exp[−2π²κC(F)] · w(D) · intersectionFugacity^Q · Z_wind(F) · w₂`, both moves, the legal vacuum D=0 ∧ Q=0 ∧ periods=0 and WHY closed ≠ exact on T⁴, emission and the always-on tilt, what the accumulator measures) modeled on `no_intersection/__init__.py`'s style, plus the imports. Parent `__init__.py`: add `from .surface_worm import SurfaceWormGas, SectorWeightTuner, PairUmbrellaTuner` and one sentence in its module docstring situating the SWG next to the DefectGas. `changes.rst`: one entry under the current release describing the new subpackage, the exactness gate, and the always-on tilt.
- [ ] **Step 4: Run the FULL new suite** — `NUMBA_CACHE_DIR=/tmp/numba_sessions/swg-migrate .venv/bin/python -m pytest test/test_surface_worm_*.py -v` — all pass; also run `test/test_defect_gas_weights.py` to confirm no collateral damage.
- [ ] **Step 5: Commit** — `surface-worm: public API, narrative docs, changes entry`.

---

### Task 12: Notebook-side statistical revalidation

**Files (in `/Users/evanberkowitz/physics/supervillain/no-intersections`):**
- Create: `swg-library-revalidation-2026-07-31/NOTES.md` (template: `.claude/docs/notes-template.md`; date = actual start date if later)
- Create: `swg-library-revalidation-2026-07-31/revalidate.py`

**Interfaces:**
- Consumes: the library `SurfaceWormGas` (Task 11 API) and the frozen `AUDIT/` toolchain, both importable from the notebook venv.

- [ ] **Step 1: Create the experiment directory and NOTES.md before the script** (lab rule), recording provenance: library commit under test, no-intersections commit, the comparison design.
- [ ] **Step 2: Write `revalidate.py`.** Two comparisons, each library-vs-AUDIT-toolchain with per-arm seeds and blocked errors:
  (a) **N=4, κ=0.03, production-tuned tables** — load `AUDIT/design_N4_k0.03.npz` + `AUDIT/umbrella_N4_k0.03.npz`, build both samplers with the same tables (`SectorWeights(design['weights'].logWeight...)` on the library side — the npz stores the toolchain objects; construct library `SectorWeights`/`PairUmbrella` from the same arrays), warm-start both from `j-vacuum-2026-07-31/ensemble_N4_k0.03.h5` row 5000 (the audited dense-branch protocol), run 100 emitted rows each: compare `⟨J²⟩`, J transition fraction, `ActionDensity`, `WrappingSquared` — agreement within 3σ of the blocked errors.
  (b) **N=4, κ=0.2 untuned** (`openSurfaceFugacity=0.2, intersectionFugacity=0.3`): 200 rows each from cold, same observables — the historical Hammer-agreement grounds.
  The toolchain arm computes raw (tilted) observables exactly as `AUDIT/probe_h2_class.py` does; the library arm reads them off `Ensemble.generate`'s output.
- [ ] **Step 3: Run**, record numbers + verdicts in NOTES.md findings as they land.
- [ ] **Step 4: If any comparison fails 3σ**, STOP — do not tune tolerances; investigate with the systematic-debugging skill (the library refactor has a bug until proven otherwise).
- [ ] **Step 5: Commit (no-intersections repo)** — `swg-library-revalidation: library SurfaceWormGas vs frozen audit toolchain`, and commit any library fix separately on `feature/surface-worm`.
