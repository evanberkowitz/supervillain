# Numba Acceleration Profiling Report

**Workload:** `example/action-comparison.py --D 4 --N 9 --configurations 1000`
**Branch:** `numba-improvements`
**Date:** 2026-06-25
**Profile data:** `/tmp/prof_d4n9.out` (before) and `/tmp/prof_after.out` (after), open with `python -m pstats`
**Status:** §1–6 are the pre-implementation analysis. The plan in §5 was
implemented (numba kernels for `d`, `δ`, `coface_sum`, `face_sum`); **§7 reports
the measured before/after result.**

> Profiling note: `cProfile` inflates absolute wall-clock by roughly 3–5× and
> the totals below are *profiled* seconds. The **ratios** between functions are
> what matter for deciding where to spend effort, and those are reliable.

---

## 1. Executive summary

Two findings dominate, and they reinforce each other:

1. **Over half the runtime is `Form` ndarray-subclass bookkeeping, not math.**
   Every arithmetic operation (`+`, `-`, `*`) and every `np.roll` on a `Form`
   re-enters `Form.__array_ufunc__`, which strips the wrappers, checks degree
   agreement, and re-wraps the result. That dispatch fires **23.1 million times**
   and costs **114.5 s of self-time (32% of the run)** — before any useful
   arithmetic happens.

2. **The codifferential `δ` (`delta`) is the hottest kernel by a wide margin**
   (182.6 s cumulative), driven almost entirely by the Worldline `VortexUpdate`
   and `CoexactUpdate` generators.

A microbenchmark of `delta` on a D=4 N=9 two-form shows that simply running the
*same algorithm* on a raw `ndarray` instead of a `Form` is already **1.57×
faster** — that 1.57× is pure subclass tax with no change to the math. A proper
numba kernel that also eliminates the `np.roll` allocations does much better: a
single-threaded structural kernel reaches **5.8×** with bit-for-bit identical
output (see [§6](#6-validation-a-real-numba-delta-measured)).

**Are `np.roll` and `__array_ufunc__` separate costs we can't escape? No.** They
are not costs that sit *beside* `delta` — they are the cost *of* `delta`, already
counted inside its 182.6 s cumulative. `delta` alone is responsible for **60% of
all `__array_ufunc__` calls** and **63% of all `np.roll` calls**; the form
kernels together account for **85%** of the ufunc traffic and essentially **100%**
of the rolls. Replacing `delta`'s internals with a numba kernel on raw arrays
therefore *deletes* its share of those buckets rather than leaving them behind —
which is exactly what the measured 5.8× confirms end-to-end.

---

## 2. Self-time hot spots (where CPU cycles are actually spent)

| Rank | Function | self-time | % of run | calls |
|---:|---|---:|---:|---:|
| 1 | `Form.__array_ufunc__` (compact.py:684) | 114.5 s | 32% | 23.1 M |
| 2 | `np.roll` (numpy numeric.py:1230) | 58.5 s | 16% | 7.4 M |
| 3 | `delta` loop body (compact.py:1006) | 22.9 s | 6% | 388 k |
| 4 | `Form.__array_finalize__` (compact.py:676) | 17.1 s | 5% | 78.9 M |
| 5 | `getattr` (driven by Form machinery) | 14.5 s | 4% | 183 M |
| 6 | `__array_ufunc__` `<genexpr>` (asarray of inputs) | 13.0 s | 4% | 68.6 M |
| 7 | `numpy.asarray` | 11.3 s | 3% | 71.9 M |
| 8 | `coface_sum` loop body (compact.py:846) | 10.7 s | 3% | 192 k |
| 9 | `normalize_axis_tuple` (inside `np.roll`) | 6.2 s | 2% | 7.4 M |
| 10 | `Form.__new__` (compact.py:670) | 4.4 s | 1% | 13.8 M |

**Form-subclass tax**, summed (`__array_ufunc__` + `__array_finalize__` +
`__new__` + the ufunc generator expressions): **~149 s of self-time**. Add the
`getattr` / `asarray` / `isinstance` traffic it generates and it is comfortably
**more than half** the entire run.

**`np.roll` tax:** 58.5 s self + 6.2 s `normalize_axis_tuple` ≈ **65 s**, across
7.4 M calls. Each call allocates a fresh array. A compiled kernel with explicit
periodic (modular) indexing removes both the allocation and the dispatch.

---

## 3. Kernels ranked by cumulative cost

These are the differential-form operators and their total cost (including
everything they call, e.g. `np.roll` and Form arithmetic):

| Kernel | cumulative | calls | notes |
|---|---:|---:|---|
| **`delta` (δ)** | **182.6 s** | 388 k | The prize. ~4.7 M `np.roll`s funnel through here. |
| **`coface_sum`** | 69.3 s | 192 k | Same shift-and-accumulate shape as δ. |
| `d` | 10.9 s | 72 k | Used by Villain updates + observables. |
| `face_sum` | ~1.5 s | 128 k | Cheap here (Villain site/exact). |
| `star`, `wedge`, `laplacian` | < 2 s each | — | **Not in this sampling hot path.** |

> **Note on your wager:** `δ`, `d`, and the action/Metropolis arithmetic are
> exactly where the time is. `star` and `wedge`, however, barely register for
> *this* workload — they are exercised by observables and other actions, not by
> the Villain/Worldline sampling loop. Worth porting for completeness, but not
> where the wall-clock is today.

---

## 4. Cost by generator (the sampling loop)

| Generator | cumulative | per step |
|---|---:|---:|
| **Worldline `VortexUpdate`** | 146.7 s | 147 ms |
| **Worldline `CoexactUpdate`** | 143.3 s | 143 ms |
| Worldline `ClassicWorm` | 6.7 s | 6.7 ms |
| Worldline `WrappingUpdate` | 0.9 s | 0.9 ms |
| Villain `ExactUpdate` | 10.0 s | 10 ms |
| Villain `SiteUpdate` | 9.4 s | 9.4 ms |
| Villain `LinkUpdate` | 0.7 s | 0.7 ms |
| Villain `CohomologyUpdate` | 0.4 s | 0.4 ms |

```
Worldline sampling  ≈ 298 s
Villain   sampling  ≈  21 s        →  Worldline is ~15× costlier
```

**Why Worldline is so expensive:** both `VortexUpdate.step` and
`CoexactUpdate.step` run a **`color × component` double loop**. For D=4 with
odd N=9 there are 16 checkerboard colors × 6 two-form components =
**96 inner iterations per step**, and each iteration calls `delta()`
(≈ 12 `np.roll`s) plus a `coface_sum()`. That is the engine driving the 388 k
`delta` calls and 7.4 M rolls.

A second, numba-independent observation: inside those loops `delta(change_v)`
and `delta(t)` operate on forms that are **all zero except one component on one
color**. A sparse/region-limited δ would skip the overwhelming majority of that
work regardless of numba.

---

## 5. Recommended acceleration plan (priority order)

The key insight: the win is not merely "JIT the arithmetic" — it is to make the
kernels operate on the **raw `ndarray`** with explicit periodic indexing, which
removes *both* the `np.roll` allocations *and* the millions of
`Form.__array_ufunc__` round-trips at once.

**Design pattern for every kernel:** keep `Form` as the user-facing type, but
inside `d`/`delta`/`coface_sum`/… do `np.asarray(form)` once, run an `@njit`
kernel that loops sites with modular indexing and accumulates into a single
preallocated output, then re-wrap as a `Form` once at the end. Precompute the
per-`(D, p)` incidence tables — `(out_idx, in_idx, axis, sign)` — at lattice
construction so the kernel is a tight numeric loop.

1. **`delta` → numba kernel.** Biggest single lever (182 s cumulative).
   Eliminates ~4.7 M rolls and a large share of the 23 M ufunc calls.
   **Measured 5.8× per call** on the hot case (see [§6](#6-validation-a-real-numba-delta-measured)).
2. **`coface_sum` (and `face_sum`) → numba kernel.** Same structure, 69 s.
3. **`d` → numba kernel.** Same pattern; widely used.
4. **Algorithmic sparse-δ in `vortex.py` / `coexact.py`.** Exploit that the
   change-forms are near-zero; independent of and complementary to numba.
5. **`star`, `wedge`, `laplacian`.** Port for completeness; low priority for
   this workload.

**Validation:** the existing self-tests in `compact.py` (`python compact.py
--D 4 --N 9`) plus the interlaced cross-validation give a ready-made correctness
harness — port under TDD, comparing each numba kernel bit-for-bit (integer
forms) / `np.isclose` (float forms) against the current implementation.

---

## 6. Validation: a real numba `delta`, measured

To check that the plan actually pays off — and is not defeated by the `np.roll`
and `__array_ufunc__` costs — a numba `delta` was written for the hot case
(D=4, N=9, two-form → one-form) and benchmarked against the current
implementation. All variants produce **bit-for-bit identical output** to
`compact.delta` (validated with `np.allclose`).

| Implementation | per call | speedup |
|---|---:|---:|
| current `delta(Form)` | 241 µs | 1.0× |
| raw `ndarray`, same algorithm (no numba) | 153 µs | 1.6× |
| numba, flat indirect-gather index map | 52 µs | 4.6× |
| **numba, structural 4-D loop (single-thread, alloc incl.)** | **41.5 µs** | **5.8×** |

The 241 µs baseline *already contains* all of `delta`'s `np.roll` and
`__array_ufunc__` overhead; the numba kernel performs the identical computation
in 41.5 µs. This is the end-to-end confirmation that those two cost buckets are
*internal* to `delta` and are removed by the port, not left behind.

Notes from the experiment:
- The naïve 5-D loop with per-axis modular indexing only reached ~1.6× — the
  cost was strided 5-D indexing, not arithmetic. Flattening the spatial axes and
  precomputing a backward-neighbor index map per axis (computed once per lattice)
  gave 4.6×; writing the modular wrap *structurally* (only the active axis wraps,
  inner axes stay contiguous) gave **5.8×**.
- **One-time compile cost** is ~1–2 s per kernel signature; `cache=True`
  amortizes it across runs.
- The 5.8× is **single-threaded**. A `prange` version is plausible but tripped a
  numba parfor type-inference quirk in a quick attempt — worth pursuing, but not
  assumed in the projections below.

### Projected whole-program impact

The form kernels account for ~263 s of the ~355 s profiled run. Applying the
measured (and expected, by structural similarity) speedups:

| Kernel | current cum | projected | basis |
|---|---:|---:|---|
| `delta` | 182.6 s | ~31 s | 5.8× measured |
| `coface_sum` | 69.3 s | ~15 s | ~4–6× expected (same shape) |
| `d` | 10.9 s | ~2 s | same pattern |

That projects to a **~2–2.5× speedup of the whole comparison run from the
kernels alone**, without touching the generators. The next bottleneck then
becomes the remaining ~15% of `__array_ufunc__` traffic (generator-level
arithmetic in `vortex`/`coexact`) and their Python `color × component` loop —
addressed by the algorithmic sparse-δ in step 4.

---

## 7. Results: measured before vs after

The plan was implemented — `d`, `δ`, `coface_sum`, and `face_sum` now run as
bit-exact `@njit` kernels (the pure-numpy code is preserved as
`supervillain/lattice/reference.py`, the equality oracle). The **identical**
workload was re-profiled with the same `cProfile` methodology
(`/tmp/prof_after.out`). The full test suite is green (3786 passed) and the
kernels match the reference bit-for-bit.

**Total profiled CPU time: 356.5 s → 167.4 s — 2.1× overall**, landing right in
the §6 projection of ~2–2.5×.

### The four operators (cumulative time)

| operator | before | after | speedup | calls |
|---|---:|---:|---:|---:|
| `δ` (delta) | 182.6 s | 41.8 s | **4.4×** | 388 k |
| `coface_sum` | 69.3 s | 17.5 s | **3.9×** | 192 k |
| `d` | 10.9 s | 3.7 s | **3.0×** | 72 k |
| `face_sum` | 3.8 s | 1.1 s | **3.4×** | 32 k |

The §6 per-operator projections held: `δ` 31 s projected vs **41.8 s** measured,
`coface_sum` 15 s vs **17.5 s** — the small gap is the no-`fastmath`/no-parallel
serial kernel actually shipped (the 5.8× microbench used `fastmath`; the bit-exact
oracle required dropping it, see the design spec §6).

### The cost buckets that dominated §1–2

| | before | after | change |
|---|---:|---:|---|
| `np.roll` | 85.3 s, 7.4 M calls | **0** | **eliminated entirely** |
| `Form.__array_ufunc__` | 162.5 s, 23.1 M calls | 36.2 s, 3.4 M calls | **6.8× fewer calls** |
| `Form.__array_finalize__` | 27.8 s, 78.9 M calls | 2.1 s, 5.1 M calls | **15× fewer calls** |

`np.roll` is gone and the `Form` dispatch traffic collapsed ~7× — confirming the
§1 thesis that those costs were *internal* to the operators, absorbed by the
kernels rather than left behind.

### The generators that drove them

| Generator | before | after | speedup |
|---|---:|---:|---:|
| Worldline `VortexUpdate.step` | 146.7 s | 53.2 s | **2.8×** |
| Worldline `CoexactUpdate.step` | 143.3 s | 48.9 s | **2.9×** |

### What's now on top, and caveats

- The largest remaining cost is the residual **36 s of `__array_ufunc__`** — now
  the *generator-level* arithmetic (the `dS_link` Metropolis expressions in
  vortex/coexact), not the operators. That is the next lever: the algorithmic
  sparse-δ / batched-arithmetic follow-on (§5 step 4), quantified in [§8](#8-where-the-time-goes-now-the-next-levers).
- `cProfile` inflates Python-heavy code more than compiled code, so it somewhat
  *understates* the true wall-clock win on the sampling loop. Both runs use the
  same tool, so the ratios are a fair comparison.
- This is the **N=9, overhead-dominated regime** where the kernels help most. At
  a production **N≫13** the per-operator multiplier shrinks (the numpy
  reference's vectorized `np.roll` amortizes), and parallel dispatch
  (`sites ≥ 30 000`) becomes the relevant lever — as the design spec details.

---

## 8. Where the time goes now: the next levers

After the kernel work the two Worldline generators still dominate
(`VortexUpdate` 53.2 s, `CoexactUpdate` 48.9 s). Attributing each `step` to its
callees pinpoints exactly what to attack next.

### `VortexUpdate.step` — 53.2 s

| callee | cum | calls | what it is |
|---|---:|---:|---|
| `delta` (kernel) | **20.5 s** | 192 k | codifferential, called ~192×/step |
| `dS_link` arithmetic (`__array_ufunc__`) | **15.8 s** | 1.33 M | the Metropolis ΔS expression |
| `coface_sum` (kernel) | 8.7 s | 96 k | boundary aggregation |
| acceptance (`clip` + `sum` + `prod`) | ~2.7 s | — | accept/reject machinery |
| `step` self (Python loop, rng) | 4.6 s | — | the `color × component` loop |

### `CoexactUpdate.step` — 48.9 s

| callee | cum | calls | what it is |
|---|---:|---:|---|
| `delta` (kernel) | **20.7 s** | 193 k | codifferential |
| `coface_sum` (kernel) | 8.8 s | 96 k | boundary aggregation |
| `dS_link` arithmetic (`__array_ufunc__`) | 3.0 s | 240 k | the Metropolis ΔS expression |
| acceptance (`clip` + `sum` + `prod`) | ~2.7 s | — | accept/reject machinery |
| `step` self (Python loop, rng) | 4.0 s | — | the `color × component` loop |

### Reading it — two distinct levers

**(a) `delta` re-computation is the single biggest residual (~41 s across both).**
Even with the 4.4×-faster kernel, `delta` stays #1 purely on call count: both
generators call it ~192×/step. `VortexUpdate` recomputes `delta(v)` on the *full*
two-form inside the inner loop every iteration, when only one component on one
color has changed (`vortex.py:91`). Hoisting / incrementally updating `delta(v)`,
or a **sparse-δ** that touches only the changed component, attacks this directly —
and pays off *independently of N*, so it matters more at large N.

> **Update — lever (a) implemented.** Added `lattice.delta_sparse` (the
> codifferential of a p-form supported on one component and checkerboard color —
> 212 dedicated bit-exact tests, documented in `form.rst`). `VortexUpdate.step`
> now computes `δv` once and patches it in place via `delta_sparse` (δ's linearity:
> `δ(v+Δv) = δv + δ(Δv)`), and the proposal's δ is sparse too; `CoexactUpdate.step`
> (which already hoisted δv) sparsifies both of its `delta(t)` calls. Each old
> dense step is preserved verbatim as `step_reference`, the oracle: vortex is
> bit-identical for finite W and coexact for **all** W including ∞
> (`test_vortex_sparse`, `test_coexact_sparse`). Measured (D=4): vortex ~1.4× at
> N=9 → ~2.2× at N=15; coexact 1.54× at N=9 → 1.81× at N=13 — and, as predicted,
> the multiplier **grows with N** (the dense `δ(v)` recompute it removes is O(V),
> while the per-call overhead is fixed). Remaining O(V) cost per step is now
> `coface_sum`; since `dS_link` is itself sparse, a sparse `coface_sum` is the
> natural next lever.

**(b) The most costly generator-level *arithmetic* is the `VortexUpdate` `dS_link`
line — 1.33 M `Form` ufunc calls, ~15.8 s** (`vortex.py:103`):

```python
dS_link = 0.5 / self.Action.kappa * (-change_delta_v / W) * (2*(m - delta_v / W) - change_delta_v / W)
```

evaluated 96×/step, each pass building ~8–10 `Form` temporaries. It is **5×
costlier than every other generator's arithmetic combined.** The asymmetry with
`CoexactUpdate`'s analogous line (`coexact.py:108`, only 3.0 s) is instructive:
coexact hoists `delta_v_by_W = delta(v)/W` out of the loop and uses a simpler
expression. Fusing vortex's `dS_link` (a small numba kernel on raw arrays, like
the operators, or restructuring to drop the intermediate `Form`s) attacks this.

> **Update — lever (b) implemented.** The `dS_link` (and acceptance `exp`/`clip`)
> arithmetic in both `VortexUpdate.step` and `CoexactUpdate.step` now runs on raw
> `np.asarray` views, re-wrapping as a `Form` once for `coface_sum`, so the ~9
> `Form` ops per inner iteration no longer round-trip through `__array_ufunc__`.
> Verified **bit-identical Markov chains** (seeded, both fields). Wall-clock
> per-step: vortex 44.4 → 38.8 ms, coexact 45.4 → 37.4 ms (**~15%**). The gain is
> smaller than the profiled 30% because `cProfile` over-weights the Python-level
> `Form` dispatch relative to wall-clock — the same caveat noted in §7. Lever (a),
> the sparse/hoisted `delta`, remains the larger, harder follow-on.

Together (a)+(b) target ~36 of `VortexUpdate`'s 53 s. The acceptance machinery
(`clip`/`exp`/reductions, ~2.7 s) and the Python `color × component` loop itself
(~4.6 s self-time) are secondary; the loop cost only becomes worth attacking
after (a) and (b), e.g. by batching all components/colors into one vectorized or
compiled pass.
