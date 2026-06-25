# Numba Form Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pure-numpy `d`, `δ`, `coface_sum`, and `face_sum` operators with bit-exact numba kernels, keeping the current numpy code as an importable reference oracle.

**Architecture:** A new `reference.py` holds verbatim numpy copies (the oracle). A new `_kernels.py` holds `@njit` table-driven shift-and-accumulate kernels (serial + parallel twins, combine-mode and direction baked as closure constants). `Lattice` gains cached incidence tables. `compact.py`'s `d`/`delta`/`Form.face_sum`/`Form.coface_sum` become thin wrappers that handle scalar-`0` ends, contiguity, dtype, table lookup, and serial/parallel dispatch, then re-wrap as a `Form`.

**Tech Stack:** Python 3.14, numpy ≥ 2.4.6, numba ≥ 0.65.1, pytest, `uv`.

## Global Constraints

- Always run Python/pytest via `uv run` (never `python3` or `.venv/bin/...`).
- Kernels must be **bit-exact** vs the numpy reference (tests compare float forms with `==`): use `@njit(cache=True, fastmath=True)`; `fastmath` is verified bit-exact for these ±1-multiply operators.
- Operators must **preserve input dtype** (integer forms stay integer): allocate output as `f.dtype`.
- Operators must return the scalar Python `0` at the degenerate ends (`d` of a D-form, `δ`/`face_sum` of a 0-form, `coface_sum` of a D-form), exactly as today.
- Public API is unchanged: `d(f)` / `delta(f)` are free functions in `supervillain.lattice`; `face_sum` / `coface_sum` are `Form` methods.
- `_kernels.py` must NOT import `Form` (avoid a circular import): it operates on raw 2-D arrays; all `Form` handling lives in `compact.py`.
- `PARALLEL_SITE_THRESHOLD = 30_000` (sites = Nᴰ); below it use serial, at/above use parallel.

---

### Task 1: Reference module (the oracle)

**Files:**
- Create: `supervillain/lattice/reference.py`
- Test: `test/test_lattice_reference.py`

**Interfaces:**
- Produces: `reference_d(f)`, `reference_delta(f)`, `reference_face_sum(f)`, `reference_coface_sum(f)` — free functions taking a `Form`, returning a `Form` of the appropriate degree or the scalar `0`. Identical logic to today's `compact.py`.

- [ ] **Step 1: Write the failing test**

`test/test_lattice_reference.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_lattice_reference.py -q`
Expected: FAIL — `ModuleNotFoundError: supervillain.lattice.reference`.

- [ ] **Step 3: Write `reference.py`**

Copy the four current implementations verbatim. `reference_d` and `reference_delta` are the current free functions `d`/`delta` from `compact.py` (lines 949–999 and 1006–1061) with the names changed. `reference_face_sum` and `reference_coface_sum` are the current `Form.face_sum` (806–844) and `Form.coface_sum` (846–884) method bodies with `self` → `f`.

```python
#!/usr/bin/env python
# Reference (pure-numpy) implementations of the lattice shift-and-accumulate
# operators.  These are the correctness oracle for the numba kernels in
# _kernels.py; keep them simple and obviously-correct, never "optimize" them.

import numpy as np


def reference_d(f):
    lat = f.lattice
    p = f.degree
    if p == lat.D:
        return 0
    result = lat.zeros(p + 1, dtype=f.dtype)
    for out_comp in lat.components[p + 1]:
        out_idx = lat.comp_index[p + 1][out_comp]
        for j, k_j in enumerate(out_comp):
            in_comp = tuple(k for k in out_comp if k != k_j)
            in_idx = lat.comp_index[p][in_comp]
            sign = (-1) ** j
            spatial = f[in_idx]
            fwd_diff = np.roll(spatial, -1, axis=k_j) - spatial
            result[out_idx] += sign * fwd_diff
    return result


def reference_delta(f):
    lat = f.lattice
    p = f.degree
    if p == 0:
        return 0
    result = lat.zeros(p - 1, dtype=f.dtype)
    all_dirs = set(range(lat.D))
    for out_comp in lat.components[p - 1]:
        out_idx = lat.comp_index[p - 1][out_comp]
        M_set = set(out_comp)
        for e in sorted(all_dirs - M_set):
            j = sum(1 for m in out_comp if m < e)
            sign = (-1) ** j
            in_comp = tuple(sorted(M_set | {e}))
            in_idx = lat.comp_index[p][in_comp]
            spatial = f[in_idx]
            bwd_diff = spatial - np.roll(spatial, +1, axis=e)
            result[out_idx] -= sign * bwd_diff
    return result


def reference_face_sum(f):
    lat = f.lattice
    p = f.degree
    if p == 0:
        return 0
    result = lat.zeros(p - 1, dtype=f.dtype)
    all_dirs = set(range(lat.D))
    for M_comp in lat.components[p - 1]:
        out_idx = lat.comp_index[p - 1][M_comp]
        M_set = set(M_comp)
        for e in sorted(all_dirs - M_set):
            in_comp = tuple(sorted(M_set | {e}))
            in_idx = lat.comp_index[p][in_comp]
            spatial = f[in_idx]
            result[out_idx] += spatial
            result[out_idx] += np.roll(spatial, +1, axis=e)
    return result


def reference_coface_sum(f):
    lat = f.lattice
    p = f.degree
    if p == lat.D:
        return 0
    result = lat.zeros(p + 1, dtype=f.dtype)
    for O_comp in lat.components[p + 1]:
        out_idx = lat.comp_index[p + 1][O_comp]
        for j, k_j in enumerate(O_comp):
            in_comp = tuple(k for k in O_comp if k != k_j)
            in_idx = lat.comp_index[p][in_comp]
            spatial = f[in_idx]
            result[out_idx] += spatial
            result[out_idx] += np.roll(spatial, -1, axis=k_j)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_lattice_reference.py -q`
Expected: PASS (all parametrizations).

- [ ] **Step 5: Commit**

```bash
git add supervillain/lattice/reference.py test/test_lattice_reference.py
git commit -m "Add pure-numpy reference implementations of d, δ, face_sum, coface_sum"
```

---

### Task 2: Incidence tables on `Lattice`

**Files:**
- Modify: `supervillain/lattice/compact.py` (add to `class Lattice`, after the `cached_property` block near line 150)
- Test: `test/test_lattice_tables.py`

**Interfaces:**
- Consumes: `Lattice.components`, `Lattice.comp_index` (existing).
- Produces: `Lattice.operator_table(op, degree) -> np.ndarray` of shape `(rows, 4)` `int64`, columns `(out_idx, in_idx, axis, sign)`. Valid `op`: `'d'`, `'delta'`, `'face_sum'`, `'coface_sum'`. Valid `degree`: `0..D-1` for `d`/`coface_sum`, `1..D` for `delta`/`face_sum`.

- [ ] **Step 1: Write the failing test**

`test/test_lattice_tables.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_lattice_tables.py -q`
Expected: FAIL — `AttributeError: 'Lattice' object has no attribute 'operator_table'`.

- [ ] **Step 3: Add the tables to `Lattice`**

In `supervillain/lattice/compact.py`, inside `class Lattice`, add after the existing `cells_of_codegree` cached property (around line 141):
```python
    @cached_property
    def _operator_tables(self):
        """Incidence tables (out_idx, in_idx, axis, sign) for the shift-and-accumulate
        operators, built once per lattice and shared across every call."""
        D = self.D
        tables = {}
        # d and coface_sum map p -> p+1 (output component drops one direction k_j).
        for p in range(D):
            d_rows, co_rows = [], []
            for out_comp in self.components[p + 1]:
                out_idx = self.comp_index[p + 1][out_comp]
                for j, k_j in enumerate(out_comp):
                    in_idx = self.comp_index[p][tuple(k for k in out_comp if k != k_j)]
                    d_rows.append((out_idx, in_idx, k_j, (-1) ** j))
                    co_rows.append((out_idx, in_idx, k_j, 1))
            tables[("d", p)] = np.array(d_rows, dtype=np.int64).reshape(-1, 4)
            tables[("coface_sum", p)] = np.array(co_rows, dtype=np.int64).reshape(-1, 4)
        # delta and face_sum map p -> p-1 (output component gains one direction e).
        all_dirs = set(range(D))
        for p in range(1, D + 1):
            de_rows, fa_rows = [], []
            for out_comp in self.components[p - 1]:
                out_idx = self.comp_index[p - 1][out_comp]
                M_set = set(out_comp)
                for e in sorted(all_dirs - M_set):
                    in_idx = self.comp_index[p][tuple(sorted(M_set | {e}))]
                    j = sum(1 for m in out_comp if m < e)
                    de_rows.append((out_idx, in_idx, e, (-1) ** j))
                    fa_rows.append((out_idx, in_idx, e, 1))
            tables[("delta", p)] = np.array(de_rows, dtype=np.int64).reshape(-1, 4)
            tables[("face_sum", p)] = np.array(fa_rows, dtype=np.int64).reshape(-1, 4)
        return tables

    def operator_table(self, op, degree):
        """Return the (rows, 4) int64 incidence table for ``op`` at input ``degree``."""
        return self._operator_tables[(op, degree)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_lattice_tables.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add supervillain/lattice/compact.py test/test_lattice_tables.py
git commit -m "Add cached incidence tables for the shift-and-accumulate operators"
```

---

### Task 3: Serial numba kernels + wire up the operators

**Files:**
- Create: `supervillain/lattice/_kernels.py`
- Modify: `supervillain/lattice/compact.py` (`d`, `delta`, `Form.face_sum`, `Form.coface_sum`)
- Test: `test/test_lattice_kernels.py`

**Interfaces:**
- Consumes: `Lattice.operator_table` (Task 2).
- Produces in `_kernels.py`:
  - `PARALLEL_SITE_THRESHOLD = 30_000`
  - `D_KERNELS`, `DELTA_KERNELS`, `COFACE_KERNELS`, `FACE_KERNELS` — each a `(serial, parallel)` tuple of compiled kernels with signature `kernel(F2d, res2d, table, N, D)` where `F2d`/`res2d` are `(C, Nᴰ)` arrays.
  - `select(kernels, sites)` → returns the serial or parallel member.
- Produces in `compact.py`: `d`, `delta`, `Form.face_sum`, `Form.coface_sum` rewritten as kernel wrappers, preserving signatures, scalar-`0` ends, and dtype.

- [ ] **Step 1: Write the failing oracle test**

`test/test_lattice_kernels.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_lattice_kernels.py -q`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'supervillain.lattice._kernels'` (the top-level import). This is the clean RED.

- [ ] **Step 3: Write `_kernels.py`**

```python
#!/usr/bin/env python
# Numba kernels for the lattice shift-and-accumulate operators.  Operates on
# raw (C, N**D) arrays only — NO Form import (avoids a circular import with
# compact.py).  Combine mode and neighbor direction are baked in as closure
# constants by _make_kernel so the contiguous inner loop is branch-free.

import numba
from numba import prange

PARALLEL_SITE_THRESHOLD = 30_000


def _make_kernel(combine, forward, parallel):
    is_diff = (combine == "diff")
    fwd = forward

    @numba.njit(cache=True, fastmath=True, parallel=parallel)
    def kernel(F, res, table, N, D):
        for t in range(table.shape[0]):
            oi = table[t, 0]; ii = table[t, 1]; e = table[t, 2]; sign = table[t, 3]
            A = N ** e
            B = N ** (D - 1 - e)
            for a in (prange(A) if parallel else range(A)):
                base = a * N * B
                for k in range(N):
                    kn = (k + 1 if k < N - 1 else 0) if fwd else (k - 1 if k > 0 else N - 1)
                    s0 = base + k * B
                    sn = base + kn * B
                    if is_diff and fwd:                      # d
                        for b in range(B):
                            res[oi, s0 + b] += sign * (F[ii, sn + b] - F[ii, s0 + b])
                    elif is_diff:                            # delta
                        for b in range(B):
                            res[oi, s0 + b] -= sign * (F[ii, s0 + b] - F[ii, sn + b])
                    else:                                    # face_sum / coface_sum
                        for b in range(B):
                            res[oi, s0 + b] += F[ii, s0 + b] + F[ii, sn + b]
        return res

    return kernel


D_KERNELS      = (_make_kernel("diff", True,  False), _make_kernel("diff", True,  True))
DELTA_KERNELS  = (_make_kernel("diff", False, False), _make_kernel("diff", False, True))
COFACE_KERNELS = (_make_kernel("sum",  True,  False), _make_kernel("sum",  True,  True))
FACE_KERNELS   = (_make_kernel("sum",  False, False), _make_kernel("sum",  False, True))


def select(kernels, sites):
    """Pick the parallel kernel for large lattices, serial otherwise."""
    return kernels[1] if sites >= PARALLEL_SITE_THRESHOLD else kernels[0]
```

- [ ] **Step 4: Rewrite the wrappers in `compact.py`**

At the top of `compact.py`, add the import (near the existing imports, ~line 32):
```python
from supervillain.lattice import _kernels
```

Replace the body of `d` (lines 949–999) with:
```python
def d(f):
    # (keep the existing docstring verbatim)
    lat = f.lattice
    p = f.degree
    if p == lat.D:
        return 0
    return _apply_operator(_kernels.D_KERNELS, "d", f, p + 1)
```

Replace the body of `delta` (lines 1006–1061) with:
```python
def delta(f):
    # (keep the existing docstring verbatim)
    lat = f.lattice
    p = f.degree
    if p == 0:
        return 0
    return _apply_operator(_kernels.DELTA_KERNELS, "delta", f, p - 1)
```

Replace the body of `Form.face_sum` (lines 806–844) with:
```python
    def face_sum(self):
        # (keep the existing docstring verbatim)
        if self.degree == 0:
            return 0
        return _apply_operator(_kernels.FACE_KERNELS, "face_sum", self, self.degree - 1)
```

Replace the body of `Form.coface_sum` (lines 846–884) with:
```python
    def coface_sum(self):
        # (keep the existing docstring verbatim)
        if self.degree == self.lattice.D:
            return 0
        return _apply_operator(_kernels.COFACE_KERNELS, "coface_sum", self, self.degree + 1)
```

Add this shared helper as a module-level function in `compact.py`, just above `def d(f):` (~line 948):
```python
def _apply_operator(kernels, op, f, out_degree):
    """Dispatch a shift-and-accumulate operator through its numba kernel.

    Handles contiguity, dtype preservation, and serial/parallel selection,
    then re-wraps the result as a Form of degree ``out_degree``.
    """
    lat = f.lattice
    N, D, S = lat.N, lat.D, lat.sites
    table = lat.operator_table(op, f.degree)
    src = np.ascontiguousarray(np.asarray(f)).reshape(f.shape[0], S)
    out = np.zeros((comb(D, out_degree), S), dtype=f.dtype)
    _kernels.select(kernels, S)(src, out, table, N, D)
    return Form(out.reshape((comb(D, out_degree),) + (N,) * D), degree=out_degree, lattice=lat)
```
(`comb` and `np` and `Form` are already imported/defined in `compact.py`.)

- [ ] **Step 5: Run the oracle test and the full lattice suite**

Run:
```bash
uv run pytest test/test_lattice_kernels.py test/test_lattice.py test/test_field_dtypes.py -q
```
Expected: PASS (kernels equal reference bit-for-bit; dtypes preserved; scalar-0 ends intact).

- [ ] **Step 6: Run the in-file self-tests and the broader suite**

Run:
```bash
uv run python supervillain/lattice/compact.py --D 4 --N 6
uv run pytest test -q
```
Expected: the `compact.py` self-tests print all ✅ and "All tests passed."; the full pytest suite is green.

- [ ] **Step 7: Commit**

```bash
git add supervillain/lattice/_kernels.py supervillain/lattice/compact.py test/test_lattice_kernels.py
git commit -m "Route d, δ, face_sum, coface_sum through bit-exact numba kernels (serial)"
```

- [ ] **Step 8 (tuning, oracle-gated): try faster inner-loop formulations**

The inner loop indexes a 2-D `(C, S)` array (`res[oi, s0 + b]`). Try, one at a time, re-measuring with `benchmark/form_kernels.py` (Task 5) and re-running `test/test_lattice_kernels.py` after each:
  - pass per-component 1-D views (`res[oi]`, `F[ii]`) into the kernel instead of 2-D indexing;
  - reshape the active axis to a literal `(A, N, B)` view per row.
Keep a change only if it is faster **and** the oracle test still passes. If none helps, leave the committed version. Commit any kept improvement:
```bash
git commit -am "Tune numba kernel inner loop (oracle still bit-exact)"
```

---

### Task 4: Parallel dispatch verification

**Files:**
- Test: `test/test_lattice_kernels.py` (extend)

**Interfaces:**
- Consumes: `_kernels.select`, `_kernels.PARALLEL_SITE_THRESHOLD`, the parallel kernels (all from Task 3).

- [ ] **Step 1: Write the failing test**

Append to `test/test_lattice_kernels.py`:
```python
from supervillain.lattice import _kernels


def test_select_threshold():
    s = _kernels.DELTA_KERNELS
    assert _kernels.select(s, _kernels.PARALLEL_SITE_THRESHOLD - 1) is s[0]
    assert _kernels.select(s, _kernels.PARALLEL_SITE_THRESHOLD) is s[1]


@pytest.mark.parametrize("op,kernels,degree", [
    ("d", "D_KERNELS", 1),
    ("delta", "DELTA_KERNELS", 2),
    ("face_sum", "FACE_KERNELS", 2),
    ("coface_sum", "COFACE_KERNELS", 1),
])
def test_parallel_kernel_matches_serial(op, kernels, degree):
    # Force the parallel kernel on a small lattice and demand bit-exactness
    # against the serial reference path.
    import numpy as np
    from math import comb
    L = Lattice(D=3, N=5)
    f = L.random(degree)
    table = L.operator_table(op, degree)
    S = L.sites
    out_degree = degree + 1 if op in ("d", "coface_sum") else degree - 1
    src = np.ascontiguousarray(np.asarray(f)).reshape(f.shape[0], S)
    serial = np.zeros((comb(3, out_degree), S)); parallel = np.zeros((comb(3, out_degree), S))
    ks = getattr(_kernels, kernels)
    ks[0](src, serial, table, L.N, L.D)
    ks[1](src, parallel, table, L.N, L.D)
    assert (serial == parallel).all()
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `uv run pytest test/test_lattice_kernels.py -q -k "threshold or parallel"`
Expected: PASS already (Task 3 built the parallel kernels). If a parallel kernel mis-compiles or races, this is where it surfaces RED — fix the parallel member of the offending `*_KERNELS` tuple (ensure the `prange` axis writes disjoint output cells) until bit-exact.

- [ ] **Step 3: Commit**

```bash
git add test/test_lattice_kernels.py
git commit -m "Verify parallel kernels are bit-exact and threshold selection is correct"
```

---

### Task 5: Benchmark script

**Files:**
- Create: `benchmark/form_kernels.py`
- Create: `benchmark/__init__.py` (empty, so it is importable for the smoke test)
- Test: `test/test_benchmark_smoke.py`

**Interfaces:**
- Consumes: `Lattice`, `d`, `delta`, `Form.face_sum`, `Form.coface_sum`, `reference.*`.
- Produces: `benchmark.form_kernels.sweep(Ds, Ns) -> list[dict]` returning per-op timing rows, and a `__main__` that prints a table.

- [ ] **Step 1: Write the failing smoke test**

`test/test_benchmark_smoke.py`:
```python
def test_sweep_runs_small():
    from benchmark.form_kernels import sweep
    rows = sweep(Ds=(2, 3), Ns=(3,))
    assert rows, "sweep produced no rows"
    for r in rows:
        assert {"D", "N", "op", "ref_us", "kernel_us", "speedup"} <= set(r)
        assert r["kernel_us"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_benchmark_smoke.py -q`
Expected: FAIL — `ModuleNotFoundError: benchmark.form_kernels`.

- [ ] **Step 3: Write the benchmark**

`benchmark/__init__.py`: empty file.

`benchmark/form_kernels.py`:
```python
#!/usr/bin/env python
# Performance benchmark for the numba Form kernels vs the numpy reference.
# Not a pytest: a dev tool to keep the speedup claims and parallel crossover
# honest.  Run: uv run python benchmark/form_kernels.py
import timeit
import numpy as np
from supervillain.lattice import Lattice, d, delta
from supervillain.lattice import reference as ref

# (production op, reference op, input degree picker) for each operator.
_OPS = {
    "d":          (lambda f: d(f),            ref.reference_d,          lambda D: max(0, D - 2)),
    "delta":      (lambda f: delta(f),        ref.reference_delta,      lambda D: 2),
    "face_sum":   (lambda f: f.face_sum(),    ref.reference_face_sum,   lambda D: 2),
    "coface_sum": (lambda f: f.coface_sum(),  ref.reference_coface_sum, lambda D: max(0, D - 2)),
}


def _time(fn, n):
    fn()  # warm up / trigger numba compilation
    return timeit.timeit(fn, number=n) / n * 1e6  # microseconds


def sweep(Ds=(4,), Ns=(7, 9, 11, 13)):
    rows = []
    for D in Ds:
        for N in Ns:
            L = Lattice(D=D, N=N)
            n = max(50, int(2e6 / N ** D))
            for op, (prod, refop, pick) in _OPS.items():
                p = pick(D)
                f = L.random(p)
                k_us = _time(lambda: prod(f), n)
                r_us = _time(lambda: refop(f), max(20, n // 4))
                rows.append({"D": D, "N": N, "op": op, "sites": N ** D,
                             "ref_us": r_us, "kernel_us": k_us, "speedup": r_us / k_us})
    return rows


if __name__ == "__main__":
    print(f"{'D':>2} {'N':>3} {'sites':>8} {'op':>11} {'ref_us':>9} {'kernel_us':>10} {'speedup':>8}")
    for r in sweep():
        print(f"{r['D']:>2} {r['N']:>3} {r['sites']:>8} {r['op']:>11} "
              f"{r['ref_us']:>9.1f} {r['kernel_us']:>10.1f} {r['speedup']:>7.1f}x")
```

- [ ] **Step 4: Run the smoke test and the benchmark**

Run:
```bash
uv run pytest test/test_benchmark_smoke.py -q
uv run python benchmark/form_kernels.py
```
Expected: smoke test PASS; benchmark prints a table with speedups (≈3–5× at small N, dropping toward ~1× at large N as documented).

- [ ] **Step 5: Commit**

```bash
git add benchmark/__init__.py benchmark/form_kernels.py test/test_benchmark_smoke.py
git commit -m "Add benchmark/form_kernels.py comparing numba kernels to the reference"
```

---

## Final verification

- [ ] Run the entire suite: `uv run pytest test -q` — all green.
- [ ] Run the self-tests at a non-trivial size: `uv run python supervillain/lattice/compact.py --D 4 --N 6` — all ✅.
- [ ] Re-profile the original workload to confirm the projected win:
  `uv run python -m cProfile -o /tmp/prof_after.out example/action-comparison.py --D 4 --N 9 --configurations 1000 --figure /tmp/after.pdf`, then compare `delta`/`coface_sum` cumulative time against `numba-profiling-report.md`.
