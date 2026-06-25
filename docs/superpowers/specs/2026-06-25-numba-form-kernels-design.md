# Numba kernels for `d`, `δ`, `coface_sum`, `face_sum`

**Date:** 2026-06-25
**Branch:** `numba-improvements`
**Scope:** Replace the pure-numpy implementations of the four lattice
shift-and-accumulate operators with numba kernels, keeping the current numpy
code as an importable reference and an exhaustive equality oracle.

Out of scope (natural follow-ons, not this spec): `star`, `wedge`, `laplacian`,
and the algorithmic sparse-δ in the Worldline generators.

---

## 1. Motivation

Profiling `example/action-comparison.py --D 4 --N 9 --configurations 1000`
(see `numba-profiling-report.md`) showed that >50% of runtime is `Form`
ndarray-subclass dispatch (`__array_ufunc__`, 23.1 M calls) plus `np.roll`
(7.4 M calls), and that essentially all of it originates inside four operators:

| Operator | cumulative | calls | direction |
|---|---:|---:|---|
| `delta` (δ) | 182.6 s | 388 k | p → p−1, backward diff, signed |
| `coface_sum` | 69.3 s | 192 k | p → p+1, forward, unsigned sum |
| `d` | 10.9 s | 72 k | p → p+1, forward diff, signed |
| `face_sum` | ~1.5 s | 128 k | p → p−1, backward, unsigned sum |

`np.roll` and `__array_ufunc__` are not costs beside these operators — they are
the cost *of* them, so porting the operators to numba kernels on raw `ndarray`
deletes those buckets. Measured `delta` speedups (D4 N9 two-form, bit-exact):
3–5× at the overhead-dominated small-N regime.

### Performance expectations (calibrated, honest)

The win is size-dependent because the numpy reference's `np.roll` is vectorized,
cache-friendly C that amortizes the Form overhead at large N:

| Regime (D=4) | kernel vs numpy reference |
|---|---|
| N ≤ ~11 (e.g. the N=9 profile) | 3–5× serial |
| N ≈ 13–17 | ties to ~1.3× |
| N ≳ 21 (a stated production target) | serial ~ties; ~1.3–1.6× with parallel |

So at the anticipated D=4, N≫13 target the d/δ/face/coface kernels alone buy a
modest multiplier; the larger large-N levers are the follow-on numba work and
the sparse-δ algorithm. These kernels are the necessary foundation that unblocks
both, and they remove the small-N penalty entirely (thermalization runs, smaller
D, tests).

---

## 2. Architecture

Three pieces, each with one clear purpose.

### 2.1 `supervillain/lattice/reference.py` (new) — the oracle
Verbatim copies of the *current* numpy implementations, as free functions:
`reference_d`, `reference_delta`, `reference_face_sum`, `reference_coface_sum`.
Logic unchanged from today's `compact.py`. First-class and importable. These
define correctness; the kernels must match them bit-for-bit.

### 2.2 `supervillain/lattice/_kernels.py` (new) — the numba kernels
`@njit(cache=True, fastmath=True)` kernels operating on a **flattened** form
array of shape `(C(D,p), S)` with `S = Nᴰ`. A small factory bakes two
compile-time constants — combine mode (signed difference vs unsigned sum) and
neighbor direction (forward `x+ê` vs backward `x−ê`) — into specialized kernels,
so nothing branches in the inner loop (branches block vectorization, measured).
That yields the four operator specializations, each with a serial and a
`parallel=True` twin (identical bodies):

| Operator | combine | direction |
|---|---|---|
| `d` | signed difference | forward `x+ê` |
| `δ` | signed difference | backward `x−ê` |
| `coface_sum` | unsigned sum | forward `x+ê` |
| `face_sum` | unsigned sum | backward `x−ê` |

Each consumes an incidence **table** of rows `(out_idx, in_idx, axis, sign)`
(`sign` is `+1` for the unsigned-sum operators). Loop nest is
`(row, A, N, B)` with `A = Nᵃˣⁱˢ`, `B = Nᴰ⁻¹⁻ᵃˣⁱˢ`, so the active axis sits in
the middle of a `(A, N, B)` view: the wrap touches only that axis and the inner
`B` loop is contiguous (no gather). `fastmath=True` is verified bit-exact here
(the only multiply is by `±1`, exact; no cross-term reassociation).

### 2.3 `compact.py` wrappers — unchanged public API
`d(f)` / `delta(f)` (free functions) and `Form.face_sum` / `Form.coface_sum`
(methods) keep their signatures. Each wrapper:
1. handles the scalar-`0` ends (`d` of a D-form, `δ`/`face_sum` of a 0-form,
   `coface_sum` of a D-form) exactly as today;
2. takes `np.ascontiguousarray(np.asarray(f))`, reshapes to `(C, S)`;
3. looks up the cached incidence table for this degree;
4. selects serial vs parallel kernel by `lattice.sites` (see §4);
5. calls the kernel into a preallocated output of `f.dtype`;
6. reshapes back and re-wraps as a `Form` of the correct degree once.

### 2.4 `Lattice` incidence tables — cached
Add `cached_property`-style per-degree tables built once from
`components`/`comp_index`: for each operator a dict `degree → np.ndarray` of
`(out_idx, in_idx, axis, sign)` rows (`int64`). Tables are tiny and shared
across all calls and all configurations.

---

## 3. Correctness invariants (must hold for every existing test)

1. **Bit-exact vs reference**, hence vs the interlaced implementation:
   `test_compact_d_matches_interlaced` / `test_compact_delta_matches_interlaced`
   compare *float* forms with `==`. Kernels must equal `reference_*` with `==`.
2. **dtype preservation:** integer forms stay integer (output allocated as
   `f.dtype`; numba specializes int64/float64). Covers
   `test_d_and_delta_preserve_input_dtype`, `test_form_operators_preserve_dtype`.
3. **Scalar-`0` ends** returned as today (Python `0`).
4. d²=0, δ²=0, adjointness ⟨da,b⟩=⟨a,δb⟩, nontriviality, and the Hammer
   end-to-end dtype tests all stay green.

---

## 4. Serial vs parallel dispatch

- Both kernels exist in serial and `parallel=True` forms with identical bodies.
- The wrapper picks parallel when `lattice.sites >= PARALLEL_SITE_THRESHOLD`,
  else serial. Measured crossover ≈ **30 000 sites ≈ Nᴰ**, roughly constant in D
  (D=4 → N≈13, D=5 → N≈8). Default constant: `30_000`, module-level and tunable.
- The exact high-N parallel kernel structure needs benchmark-driven tuning (the
  naïve per-row `prange` is uneven because `A = Nᵃˣⁱˢ` varies per row). The
  parallel axis must be chosen so each kernel call opens few parallel regions
  over enough work, while preserving the no-write-race property (a thread writes
  disjoint output cells). Correctness is gated by the oracle test regardless of
  the structure chosen.

---

## 5. Testing

- **New `test/test_lattice_kernels.py`** — the oracle: for every
  `D ∈ 2..6`, `N ∈ {3,4,5}` (plus a couple of larger N to exercise the parallel
  path), every degree, and `dtype ∈ {int, float}`, assert
  `kernel(f) == reference_*(f)` with `==`, for all four operators, in both
  serial and parallel dispatch.
- **Entire existing suite stays green:** `test_lattice.py`,
  `test_field_dtypes.py`, the generator/observable tests that exercise these
  operators, and the `compact.py` `__main__` self-tests.
- **Benchmark script** (not a pytest): a small `example`/dev script reporting
  per-operator µs and speedup vs reference across a size sweep, so the
  performance claims and the parallel crossover stay measurable and honest.

TDD order: write the oracle test first (it fails because kernels don't exist),
then implement kernels until oracle + full suite pass, then tune for speed
without breaking the oracle.

---

## 6. Risks / open questions

- **Parallel tuning at large N** is the least-pinned-down part; expectation is
  ~1.3–1.6× there, not the small-N 3–5×. Treated as measure-and-tune, gated by
  the oracle.
- **`fastmath` bit-exactness** verified for these specific ops (±1 multiplies);
  if any future change introduces real reassociation it must be re-verified or
  `fastmath` dropped for that kernel.
- **Contiguity:** sliced/strided Forms are forced contiguous in the wrapper;
  this copy is cheap relative to the kernel and keeps the kernel simple.
- **One-time compile cost** (~1–2 s per kernel signature) is amortized by
  `cache=True`.
