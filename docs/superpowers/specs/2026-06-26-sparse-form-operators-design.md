# Sparse form operators: `d`, `face_sum`, `coface_sum` (+ the existing `delta`)

**Date:** 2026-06-26
**Branch:** `numba-improvements`
**Status:** design — extends the shipped `delta_sparse` (lever a) to the operators
the Villain updates and the Worldline acceptance need.

## 1. Motivation

`delta_sparse` already lets the Worldline `VortexUpdate`/`CoexactUpdate` maintain
`δv` and form proposals without a full `δ`. Two things remain:

1. An upcoming **large-D=4 Villain** project will sparsify `SiteUpdate` and
   `ExactUpdate`, which are the Villain analogues of vortex/coexact. They use
   **`d`** (φ/z → links) and **`face_sum`** (links → sites), not `δ`/`coface_sum`.
2. The Worldline acceptance still computes a **full `coface_sum`** of the sparse
   `dS_link` every inner iteration — now the dominant O(V) cost per step (report
   §8). A sparse `coface_sum` removes it.

So we need sparse `d`, `face_sum`, and `coface_sum`. All the math below is
**validated bit-exact against the dense operators** (D=2–5, every degree,
component, color).

## 2. There are two sparsity flavors

The generators use sparsity in two distinct ways; conflating them is the trap.

### Flavor A — input-sparse *spread* (`*_sparse`)
The input is a form supported on **one component + one checkerboard color**
(a proposal, or an accepted field change); the operator spreads it to a few
output cells. This is exactly what `delta_sparse` does. Used for:
- the proposal's image (`d(change_φ)`, `δ(change_v)`, …), and
- the in-place field patch (`dφ += d(Δφ)`, `n += d(Δz)`, `δv += δ(Δv)`, `m += δ(Δt)`).

**Rule** (for a value `a` at site `x` of input component `c`, per operator-table
row `(out, in=c, axis e, sign)`):

| operator | degree | output cells | written values |
|---|---|---|---|
| `delta_sparse` (shipped) | p→p−1 | `x`, `x+ê_e` | `−sign·a`, `+sign·a` |
| **`d_sparse`** | p→p+1 | `x`, `x−ê_e` | `−sign·a`, `+sign·a` |
| `face_sum` (flavor A, if needed) | p→p−1 | `x`, `x+ê_e` | `+a`, `+a` |
| `coface_sum` (flavor A, if needed) | p→p+1 | `x`, `x−ê_e` | `+a`, `+a` |

Pattern: **p→p−1 spreads forward (`x+ê`), p→p+1 spreads backward (`x−ê`)**;
signed ops (`d`,`δ`) write `∓sign·a`, unsigned (`face`,`coface`) write `+a`,`+a`.
Tables come from `Lattice.operator_table(op, degree)` filtered to `in==c`.

### Flavor B — output-sparse *gather* (`*_at`)
The acceptance only needs the reduction at **one output component + color**
(`dS[comp][color]`), from a `dS_link` that is itself sparse. Gather the few
input links into those output cells:

| reduction | degree | `dS[O,x] = Σ over table rows out==O of …` | neighbor |
|---|---|---|---|
| **`coface_sum_at`** | p→p+1 | `f[in,x] + f[in,x+ê_e]` | forward |
| **`face_sum_at`** | p→p−1 | `f[in,x] + f[in,x−ê_e]` | backward |

**Critical:** accumulate the two terms as **two separate `+=`** in table-row
order (`R += f[in,x]; R += f[in,neighbor]`), matching the dense kernel — a
combined `f[x]+f[neighbor]` diverges at machine epsilon (float non-associativity,
same lesson as the kernel sum-ops). Verified: sequential adds are bit-exact;
combined are not.

## 3. Where each is used

| Generator | proposal/patch (Flavor A) | acceptance reduction (Flavor B) | bit-exact? |
|---|---|---|---|
| Worldline `VortexUpdate` | `delta_sparse` (shipped) | **`coface_sum_at`** | finite W |
| Worldline `CoexactUpdate` | `delta_sparse` (shipped) | **`coface_sum_at`** | all W |
| Villain `SiteUpdate` | **`d_sparse`** (`dφ` patch) | **`face_sum_at`** | bit-exact (see below) |
| Villain `ExactUpdate` | **`d_sparse`** (`n` patch) | **`face_sum_at`** | bit-exact (integer `n`) |

Both Villain updates are bit-exact under sparsification, with **no float-drift
caveat** (unlike the W=∞ vortex case):
- `SiteUpdate` *already* maintains `dφ = d(φ)` incrementally
  (`dphi = dphi + d(change_phi)`), so replacing those `d(...)` calls with
  `d_sparse` reproduces the identical float accumulation — bit-exact, no resync.
- `ExactUpdate` holds `dφ` fixed and patches the **integer** `n` by `d(change_z)`;
  `d_sparse` of integer input is exact.

## 4. API

Add to `compact.py` next to `delta_sparse`, exported from `supervillain.lattice`:

```python
def d_sparse(lattice, degree, component, color, values, out=None): ...   # Flavor A, p->p+1
def coface_sum_at(f, component, color): ...                              # Flavor B, returns 1-D array at (component,color)
def face_sum_at(f, component, color): ...                                # Flavor B
```

`d_sparse` mirrors `delta_sparse`'s signature exactly (accumulates into `out`).
The `*_at` reductions take a Form `f` and a target `(component, color)` and return
the gathered values as a 1-D array aligned with `color`. (A flavor-A
`face_sum_sparse`/`coface_sum_sparse` is easy to add for symmetry but is **not**
needed by any current generator — YAGNI unless a caller appears.)

## 5. The fully-local inner body (the real payoff)

Sparsifying the operators is necessary but, alone, leaves the `dS_link`
*arithmetic* full (it multiplies the sparse change-image by the **full**
background field `m − δv/W` or `dφ − 2πn`). The complete win localizes that too:
the change-image is nonzero only on a known link set `L`, and `coface_sum_at`/
`face_sum_at` read `dS_link` only on `L` — so compute `dS_link` only on `L`,
gathering the background field at those links. Then the whole inner body is
O(changed sites), not O(V). This is the deepest phase and the one that makes the
generator-level speedup approach the per-call `n_colors×`.

## 6. Phasing

1. **Primitives + tests.** `d_sparse`, `coface_sum_at`, `face_sum_at` in
   `compact.py`; a `test/test_sparse_operators.py` modeled on
   `test/test_sparse_delta.py` — bit-exact vs dense across D=2–5, every degree,
   component, color, dtype; plus the `*_at` sequential-add bit-exactness. Document
   in `form.rst` beside `delta_sparse`.
2. **Worldline `coface_sum_at`.** Replace the full `dS_link.coface_sum()` +
   `dS[comp][color]` in vortex/coexact with `coface_sum_at(dS_link, comp_idx, color)`.
   Keep `step_reference`; the existing `test_vortex_sparse`/`test_coexact_sparse`
   bit-exact comparisons already gate it.
3. **Villain `SiteUpdate`/`ExactUpdate`.** Sparsify with `d_sparse` (proposal +
   field patch) and `face_sum_at` (acceptance), preserving each old dense `step`
   as `step_reference`, with a `test_site_sparse`/`test_exact_sparse` bit-exact
   comparison (mirroring the worldline pattern). Both are bit-exact (table in §3):
   `SiteUpdate` already accumulates `dφ` incrementally, `ExactUpdate` patches
   integer `n` — no float-drift caveat.
4. **Deep localization (optional).** Compute `dS_link` only on the changed link
   set (§5) so the inner body is O(changed sites). Biggest large-N win; do only
   if profiling after phases 2–3 still shows `dS_link`/background arithmetic on top.

## 7. Testing & correctness invariants

- Every sparse primitive is **bit-exact vs its dense operator** (`==`), the same
  oracle discipline as `delta_sparse` (212 tests) and the kernels.
- Every sparsified generator keeps a verbatim `step_reference` and a seeded
  bit-exact `step == step_reference` chain test (finite W where the maintained
  field is float; all W where it is integer).
- The existing constraint tests (`δm=0`, `dn≡0`) and the full suite stay green.
- Sum-type reductions/operators accumulate **sequentially** (two `+=`) to stay
  bit-exact — enforced by the oracle tests.
