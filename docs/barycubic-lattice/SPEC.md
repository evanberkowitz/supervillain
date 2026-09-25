# Barycubic lattice: implementation spec

Scope:

- Fully general. Every legal degree `p = 0..D` gets `d`, the diagonal Hodge star, `δ`, and the
  Laplacian. Steenrod products `∪_i` are included for every legal `i`. General dimension `D`,
  mirroring `supervillain.lattice.Lattice` / `Form` in `supervillain/lattice/compact.py`.
- No dual or worldline formulation on this lattice. Nothing here builds dual-complex operators.
- An independent implementation may exist, and the two should be comparable exactly (section 8).

---

## 1. The complex

- `K` is the cubical cellulation of the torus `(R/NZ)^D` by unit hypercubes, with `D ≥ 1` and
  `N ≥ 2`. Both parities of `N` are allowed. The parameter name is `N`, matching `Lattice(D, N)`.
- `S = sd K` is its barycentric subdivision. It has one fine vertex `b(F)` per cell `F` of `K`,
  at the barycenter of `F`, and one fine p-simplex per flag `F_0 ⊊ F_1 ⊊ … ⊊ F_p`.
- The **rank** of `b(F)` is `dim F`. Every fine simplex has vertices of distinct ranks, so the
  rank order is a total order on each simplex, and it is inherited by faces. This local ordering
  is what the Alexander–Whitney and Steenrod products need.
- For `N ≥ 2`, `S` must be a genuine simplicial complex after periodic identification: distinct
  simplices have distinct vertex sets. This is a tested invariant (section 7), with `N = 2` as
  the edge case.

## 2. Labels and storage

### 2.1 Fine vertices as the doubled lattice
Use **doubled coordinates** `ξ ∈ (Z/2N)^D`, where `ξ = 2x` for a point `x ∈ (½Z)^D`. Every point
`ξ` is the barycenter of exactly one cell:

- the cell's **type** is the direction set `T(ξ) = {k : ξ_k odd}`;
- its rank is `|T(ξ)|`;
- its **anchor site** is `floor(ξ/2) ∈ (Z/N)^D`.

Fine vertices are exactly the doubled lattice, with `2^D` per original site.

### 2.2 Fine simplices
A fine p-simplex is a rank-ordered vertex list `ξ^0, …, ξ^p` with strictly nested types
`S_0 ⊊ S_1 ⊊ … ⊊ S_p ⊆ {0..D-1}`. It is determined by three pieces of data.

- **Top vertex:** `ξ^p = 2x + 1_{S_p}`, where `x` is the anchor site of the simplex.
- **Signs:** one sign `σ_k ∈ {−1, +1}` for each `k ∈ S_p \ S_0`. It records which side of the
  top cell the lower faces sit on.
- **Lower vertices:** `ξ^a_k = ξ^p_k + σ_k` for `k ∈ S_p \ S_a`, and `ξ^a_k = ξ^p_k` otherwise.
  All coordinates are taken mod `2N`.

The **local label** of the simplex is `(S_0, …, S_p; σ)`. Local labels are ordered
lexicographically by the tuple `(mask(S_0), …, mask(S_p), bits(σ))`, where:

- `mask(S) = Σ_{k∈S} 2^k`;
- `bits(σ)` lists `0` for `σ_k = −1` and `1` for `σ_k = +1`, over `k ∈ S_p \ S_0` in increasing
  `k`.

**Common lift (mandatory).** Stored coordinates are reduced mod `2N`. Every metric or
displacement computation must use the covering-space lift built from the label *before*
reduction: `ξ^p = 2x + 1_{S_p}` with `x` the anchor in `[0, N)^D`, and `ξ^a = ξ^p + Σ σ_k e_k`
as unreduced integers. This covers determinants, orientation, centroids, volumes, and winding
displacements. Differences of reduced representatives are wrong across a seam. Example:
`D = 1, N = 2`, top `ξ = 3`, `σ = +1`. The lower vertex is stored as `0`, so the raw difference
is `3`, but the true displacement is `−1`. Using raw differences also makes the winding cochain
exact, with period 0. Minimum-image differences agree with the lift, because true
differences are at most one doubled unit and `2N ≥ 4`.

The number of labels per site is `f_p(D) = Σ over chains S_0⊊…⊊S_p of 2^{|S_p \ S_0|}`.
Equivalently, by inclusion–exclusion, `f_p(D) = Σ_{j=0}^{p} (−1)^{p−j} C(p,j) (2j+2)^D`.

| D=4 | p=0 | 1 | 2 | 3 | 4 | Euler characteristic |
|---|---|---|---|---|---|---|
| f_p | 16 | 240 | 800 | 960 | 384 | 0 |

Also `f_1(D) = 4^D − 2^D` and `f_D(D) = 2^D · D!`.

### 2.3 Forms
A barycubic p-form is an ndarray subclass `BarycubicForm` with shape `(f_p, N, …, N)`: `D`
spatial axes, index `[label, x]`. Its semantics mirror `Form`:

- attributes `degree` and `lattice`;
- elementwise arithmetic between same-degree forms returns a form; mixed degrees return a plain
  ndarray;
- reductions return plain numpy values;
- a separate batch/h5 tag, so hypercubic and barycubic forms can never be confused on read.

**Serialization work in scope.** A new `__batch_tag__` alone does nothing: the only tag
`supervillain/batch.py::resolve_batch_cls` knows is `"Form"`, and upstream has the same
limitation. The implementation must do two things.

1. Register `BarycubicForm` (and `BarycubicDualForm`, if kept) in `resolve_batch_cls`.
2. Add a standalone-form HDF5 strategy that stores `degree`, `dual`, the lattice (`D`, `N`), and
   the dtype. The existing NumPy strategy reads ndarray subclasses back as plain arrays, and
   loses all of these.

**Dtypes.**
- Integer forms stay integer under `d`, `∪_i`, `coarsen`, and `transform`.
- The weighted operators (`star`, `inverse_star`, `δ`, the Laplacian, `inner`) promote to float
  or complex. For example, at `D = 1` the vertex weight is `M_0 = 1/2`, so `star` of an integer
  basis form is `1/2`.
- Complex forms are supported, and `inner` is the Hermitian extension (section 5.2).

### 2.4 Faces and cofaces
Dropping a vertex from a simplex anchored at `x` gives a face anchored at `x + o` with
`o ∈ {0,1}^D`. Only dropping the top vertex can move the anchor: it moves by `+1` in each
direction `k ∈ S_p \ S_{p−1}` with `σ_k = +1`. Cofaces therefore have offsets in `{0,−1}^D`.

## 3. Orientation

- **Reference orientation.** Stored values are for the rank-ordered vertex list:
  `ω[σ] = ω(ξ^0 … ξ^p)`.
- **Spacetime orientation of top simplices.** For a D-simplex with rank steps
  `k_a = S_{a+1} \ S_a`, the lifted edge vectors (section 2.2) are
  `ξ^{a+1} − ξ^a = −σ_{k_a} e_{k_a}`. Hence
  `ε_σ = sgn(k_0, …, k_{D−1}) · Π_k (−σ_k)`. This is exposed as the integer top-form
  `lattice.orientation`.
- **Integration and the fundamental cycle.** `∫ ω = Σ_σ ε_σ ω[σ]` for D-forms. The fundamental
  cycle is `[M] = Σ ε_σ σ`, and it must satisfy `∂[M] = 0`, which is tested.

## 4. Coboundary

`(dω)(v_0 … v_{p+1}) = Σ_j (−1)^j ω(v_0 … v̂_j … v_{p+1})`, with vertices in rank order. For
`p = D` it returns the scalar `0`, as `compact.d` does.

## 5. Metric operators and products

### 5.1 Hodge weights (the diagonal star)
For a fine p-simplex `σ`, the weight is `M_p(σ) = Vol_{D−p}(⋆σ) / Vol_p(σ)`.

- **Dual cell.** `⋆σ` is the barycentric dual cell of `σ` inside `S`. It is the union, over all
  maximal chains of fine simplices `σ = τ_0 ⊊ τ_1 ⊊ … ⊊ τ_{D−p}`, of the geometric simplices
  `[c(τ_0) … c(τ_{D−p})]`, where `c` is the centroid.
- **Volumes.** Metric volumes are Euclidean, `Vol_k = sqrt(det Gram)/k!`, and `Vol_0 = 1`. The
  dual cell's volume is the sum of the volumes of its pieces, because pieces need not be coplanar.
- **Boundary degrees.** `M_D(σ) = 1/Vol_D(σ) = 2^D D!`, and `M_0(v)` is the D-volume of the dual
  cell of `v`.
- **Symmetry.** `M_p` depends only on the **rank set** `R(σ) = {|S_0|, …, |S_p|}`, because the
  hyperoctahedral group times translations acts transitively on flags of each type, including
  types that skip ranks. There are `2^{D+1} − 1` rank-set entries. These are not all distinct
  numbers: `M(R) = M(D − R)` forces equalities, and D=4 has 18 distinct values among its 31
  entries. The implementation computes one entry per rank set, from exact rational coordinates,
  in float64.
- **Vertex weights in closed form** (from the review): `M_0(r) = 1 / ((D+1) C(D,r))`. This makes
  the vertex sum rule exact, since `Σ_r C(D,r) M_0(r) = 1` per site.

### 5.2 Inner product, δ, Laplacian
- `inner(a, b) = Σ_σ M_p(σ) conj(a[σ]) b[σ]`. On real forms this is `Σ M a b`; on complex
  forms it is Hermitian and positive definite.
- `δ = M_{p−1}^{-1} dᵀ M_p`. This is the adjoint of `d` in `inner`, matching the convention of
  `compact.delta`, where `⟨df, g⟩ = ⟨f, δg⟩`. For `p = 0` it returns the scalar `0`.
- `laplacian = dδ + δd`, the same definition as `test/test_lattice.py::_laplacian_via_d_delta`.

### 5.3 Star
`star(f)` returns `M_p · f`, typed as a `BarycubicDualForm`:

- it records `degree = D − p` and a `dual = True` flag;
- it supports arithmetic, `inverse_star`, and pairing with primal forms;
- it has no `d`, because there is no dual formulation.

This exists so the kinetic term can be written as `inner(ξ, ξ) = Σ ξ · star(ξ)`, as in the note.
Open question Q1: drop `star` entirely and keep only `inner`.

### 5.4 Cup product (i = 0)
Alexander–Whitney in rank order:
`(a ∪ b)(v_0 … v_{p+q}) = a(v_0 … v_p) · b(v_p … v_{p+q})`.
If `p + q > D`, it raises `ValueError`, as `compact.cup` does.

### 5.5 Steenrod products ∪_i (frozen sign rule)
Let `a` have degree `p`, `b` degree `q`, `0 ≤ i ≤ min(p, q)`, and `n = p + q − i ≤ D`. On a
rank-ordered simplex `(v_0 … v_n)`:

```
(a ∪_i b)(v_0…v_n) = Σ_{J = (j_0 < … < j_i) ⊆ {0..n}} ε(J) · a(v_A) · b(v_B)
  A = [0, j_0] ∪ [j_1, j_2] ∪ [j_3, j_4] ∪ …      (even-numbered intervals of the cut)
  B = [j_0, j_1] ∪ [j_2, j_3] ∪ …                 (odd-numbered intervals), last interval ends at n
  terms with |A| ≠ p+1 or |B| ≠ q+1 are omitted
  ε(J) = (−1)^{ C(p,2) + C(n,2) + Σ_{u ∈ U_B} ind_U(u) }
  U = {0..n} \ J,  ind_U(u) = #{w ∈ U : w < u},  U_B = {u ∈ U : u − ind_U(u) odd}
```

`U_B` is the set of uncut vertices that appear only in `b`'s face. The cut points `J` appear in
both faces.

**Provenance.** The rule was derived and checked in this session, not quoted from the literature.

1. **Uniqueness certificate.** `docs/barycubic-lattice/solve_cup_i_signs.py 10 5 6` works on the
   standard 9-simplex. Every interval-cut term gets an unknown coefficient, `∪_0` is fixed to
   Alexander–Whitney, and the Steenrod identity is solved level by level for `i = 1..5`, using
   6 random integer cochain pairs per `(p, q)`.
   - All 912 coefficients with output degree `n ≤ 8` lie outside the null space of the
     constraints. Each equals ±1, and each agrees with the closed form above.
   - Fewer random samples leave spurious null directions. With 3 samples, only 84 of the 210
     coefficients at `i = 4` came out determined.
2. **Exact check.** `docs/barycubic-lattice/verify_cup_i_signs.py 11` checks, in exact integer
   arithmetic:
   - the identity for every `(p, q, i)` with `n ≤ 9` (220 cases, `i ≤ 9`);
   - three negative controls, each of which must break the identity;
   - the order-reversal law below;
   - the pointwise top product.
3. **Independent check.** An independent re-implementation of the rule expanded every
   bilinear coefficient symbolically: 220 identity cases pass, and 161 even-`i` reversal cases
   pass.

**Scope of "forced".** The signs are forced as a *universal* family, i.e. on a simplex of
dimension at least `n + 2`. They are **not** all forced inside a D-dimensional complex. There,
the only legal product with `i = D` is `D`-form `∪_D` `D`-form. Its identity has degree `D + 1`,
where every term vanishes, so flipping its sign breaks no legal identity. The review computed
exactly one free coefficient at each `D = 2..6`, and mod-2 and reversal tests miss it too.

The pointwise test `a ∪_p b = a·b` for p-forms pins that coefficient. It is the analogue of the
existing `test_top_cup_is_pointwise_product`, and it is **mandatory for every p, including
p = D**. The implementation's sign table must also be compared entry by entry with the closed
form (section 7, item 10).

Given the universal family, any correct implementation with interval-cut support agrees term
by term. That makes it a good interop check against an independent implementation.

**Steenrod identity** (the fork's convention, `test/test_cup_products.py::_steenrod_residual`):
```
d(a ∪_i b) = da ∪_i b + (−1)^p a ∪_i db + (−1)^{p+q−i} a ∪_{i−1} b + (−1)^{pq+p+q} b ∪_{i−1} a
```
Here `∪_{−1} ≡ 0`, and any term whose degree exceeds `D` is `0`.

**Order reversal.** For even `i`: `rev(a ∪_i b) = (−1)^{pq + i + C(i+1,2)} rev(b) ∪_i rev(a)`,
where `rev` is pullback along vertex-order reversal. For odd `i` it holds only on the diagonal
`p = q = i`, where `a ∪_i b = a·b`. The checker confirms `(1,1,1)`, `(3,3,3)`, and `(5,5,5)`,
and 32 other odd-`i` cases fail. This is relevant because the half-diagonal translation reverses rank order
(section 6.2).

## 6. Maps between lattices, and symmetries

### 6.1 `coarsen`: fine to hypercubic
`coarsen(ω)` returns `Sdᵀ ω`, a `compact.Form` on `Lattice(D, N)`:
`(Sdᵀ ω)(F) = Σ_{σ ⊆ F, dim σ = dim F = p} o(σ, F) ω[σ]`, where `o(σ, F) = ±1` compares `σ`'s
rank orientation with `F`'s orientation in increasing-direction order.

- It is a cochain map: `coarsen(d_S ω) = d_K coarsen(ω)`, tested exactly.
- It uses only `compact.d` and `Lattice`, both present upstream. It does **not** use the fork-only
  hypercubic `cup`.
- A coarse-to-fine refinement map is out of scope, since no chain map `S → K` is needed yet.

### 6.2 Symmetries
`transform(form, perm, flips, shift)` applies the hyperoctahedral group and integer translations
by acting on `ξ`.

- These preserve rank, so the pullback is a pure relabelling with no sign.
- `d` and `∪_i` commute exactly, weights are invariant, and `ε` picks up `det g`.
- **Half-diagonal translation.** `half_shift(form)` implements `τ: ξ ↦ ξ + (1,…,1)`, which maps
  rank `r` to `D − r` and reverses rank order. On a p-simplex the pullback carries the sign
  `(−1)^{C(p+1,2)}` of the order reversal.
  - `d` commutes with `τ`.
  - Weights satisfy `M(R) = M(D − R)`.
  - `∪_i` satisfies the reversal law of section 5.5 for even `i` only.

### 6.3 Coloring
Rank is a proper vertex coloring of the fine 1-skeleton with `D + 1` colors, for **every** `N`.
Every link joins different ranks, and a top simplex is a `(D+1)`-clique, so this is optimal.

**Intentional API difference from `Lattice`.** Rank depends on the label axis, and every anchor
site carries vertices of every rank. So the existing spatial-only selector,
`form[(slice(None), *color)]`, cannot express this coloring. `lattice.checkerboarding` instead
returns, per color, an index tuple over **all** `D + 1` axes `(labels, x_0, …, x_{D−1})`, used
as `form[color]`. Tests cover:
- the colors together cover every vertex;
- the colors are disjoint;
- no link joins two vertices of the same color, at even and odd `N`.

Link colorings for local updates of `n` depend on the action and are deferred (Q3).

## 7. Tests (written before the implementation)

Parameters, unless a test says otherwise:

- `D ∈ {1, 2, 3, 4}` and `N ∈ {2, 3, 4}`;
- `D = 4` only at `N ∈ {2, 3}` where the test is expensive;
- one `D = 5, N = 2` smoke test for counts and `d∘d`.

Integer random cochains are used wherever an identity is exact.

**Oracles independent of the implementation**

1. **Counts.** `f_p` per site from the formula in section 2.2, the Euler characteristic 0,
   `f_D = 2^D D!` per hypercube, exactly one vertex of each rank per top simplex, and fine-link
   lengths `sqrt(s − r)/2`.
2. **Simplicial-complex invariant.** Vertex sets are pairwise distinct at `N = 2` and `N = 3`.
3. **Topology by Fourier blocks.** By translation invariance, `d` block-diagonalizes by momentum
   `k` into matrices `d(k)` of size `f_{p+1} × f_p`, with entries `Σ_rows sign · e^{ik·offset}`.
   Adjoints and Laplacian blocks use the conjugate transpose `d(k)^†`, not the plain transpose.
   - At `k = 0`, the cohomology of the unit-cell complex has Betti numbers `C(D, p)`.
   - At every `k ≠ 0` on the `N`-grid, it is acyclic.
   - The weighted Laplacian block has null-space dimension `C(D, p)` at `k = 0` and none
     otherwise.

   This makes `D = 4` cheap, since the matrices are at most 960 × 800.
4. **Reference build.** A slow, independent construction enumerates the cubical cells `(x, S)`
   and their flags explicitly, not through the doubled-coordinate shortcut, and builds sparse
   incidence matrices. Every fast operator must equal the reference exactly: `d` and cup/∪_i on
   integers, and weights to 1e-13 relative. This mirrors `test/test_lattice_reference.py`.
   The reference makes the *geometry* independent. It does not make the *cup signs*
   independent, because both builds use the same sign rule. The sign oracle is the closed form
   checked by the two `docs/barycubic-lattice/` scripts, plus the pointwise test (item 10).
5. **Independently derived reference values.** Match independently derived values for the ten
   `D = 4` link weights, the simplex symmetries, and plaquette factorization. That means the
   cup-square `q = f∪f` on a top simplex equals `f[v0v1v2] · f[v2v3v4]`.
6. **Weight sum rules.**
   - `Σ_v M_0(v) = N^D` (vertex dual cells tile the torus).
   - `Σ_σ 1/M_D(σ) = N^D`.
   - The affine-field normalization at `D = 4` is `(√5 + 2√6 + 3 + 2√2)/10 ≈ 1.296`, reproduced
     independently from the weights derived in section 5.1.
   - Weights are positive, and `M(R) = M(D − R)`.

**Algebraic identities** (the ones carried over from `test_lattice.py` / `test_cup_products.py`)

7. `d∘d = 0`; `d` is nonzero below top degree; `d` is linear.
8. `δ∘δ = 0`; `inner(da, b) = inner(a, δb)`; the Laplacian is self-adjoint and positive
   semidefinite; the Laplacian equals `dδ + δd`.
9. Cup: bilinear; translation covariant; `(a ∪ f)(σ) = a(σ) f(v_D)` and
   `(f ∪ a)(σ) = f(v_0) a(σ)` for a top form `a` and a 0-form `f` (the analogue of
   `test_top_cup_is_pointwise_product` for `i = 0`); out-of-range raises; Leibniz rule exact;
   strict associativity exact. The existing `test_lattice.py` tests already check the latter
   two for the hypercubic `wedge`, so these are ports.
10. Steenrod products:
    - the identity for all legal `(p, q, i)`, over the integers and mod 2;
    - Steenrod squares of mod-2 cocycles are cocycles;
    - **pointwise `a ∪_p b = a·b` for every `p` including `p = D`**, which is the only guard on the
      sign of `∪_D` (section 5.5);
    - the cached sign table equals the closed form of section 5.5 entry by entry;
    - reversal: exact for every even `i`, true on the odd diagonal `p = q = i`, and at least one
      off-diagonal odd-`i` failure is asserted.
11. **Cohomology ring.** Take the winding 1-cochains `w_μ(e) = Δξ_μ(e)/2`, using the lifted
    fine-link displacement (section 2.2). Each is closed, and its period around the μ cycle is
    `N`. Then `∫ w_{π(0)} ∪ … ∪ w_{π(D−1)} = sgn(π) N^D` exactly. Also add a regression test for
    the seam: at `D = 1, N = 2` the lifted displacement across the seam is `−1`, and the period
    is `2`.
12. **Symmetries.** `transform` commutes with `d` and `∪_i` and preserves weights. `half_shift`
    commutes with `d`, maps `M(R)` to `M(D − R)`, and obeys the even-`i` reversal law. The
    fundamental cycle satisfies `∂[M] = 0`.
13. **Bridge.** `coarsen ∘ d_S = d_K ∘ coarsen`, and `∫_M coarsen(ω) = ∫_M ω` for top forms.
14. **I/O.** HDF5 round trips, both standalone and inside a `Batch`:
    - the lattice (`D`, `N`);
    - forms of every degree, with integer, float, and complex dtypes;
    - class, degree, `dual` flag, and lattice all preserved;
    - mismatched type metadata rejected on read.
15. **Dual forms** (if Q1 keeps `star`): `inverse_star(star(a)) = a`; dual shape, type, and
    degree `D − p`; unsupported dual operations raise.

## 8. Interop format (comparison with an independent implementation)

`lattice.export_complex(p)` returns, for every fine p-simplex:

- its canonical global label, a tuple of rank-ordered vertices in doubled coordinates in
  `[0, 2N)^D`;
- `d` as a sparse matrix in canonical-label order;
- `M_p` per simplex;
- `ε` for top simplices.

A small comparison script diffs two exports. It also checks `∪_i` on shared random integer
cochains, keyed by canonical label. Internal layouts can then differ freely.

## 9. Performance design

- **Incidence tables.** `operator_table(op, degree)` for `op ∈ {d, dT}` returns int64 rows
  `(out_label, in_label, offset[D], sign)`, generalizing the hypercubic `(out, in, axis, sign)`.
  They are cached per `(D, op, degree)` and are independent of `N`. Face offsets lie in
  `{0,1}^D` and coface offsets in `{0,−1}^D`, so each table has at most `2^D` distinct offsets.
- **Incidence application.** For each distinct offset, roll the input once and apply a small
  sparse `f_out × f_in` matrix over labels. Memory is O(form size).
- **Weighted operators** are compositions, never one fused stencil. `δ = M^{-1} ∘ dT ∘ M` is a
  diagonal scaling, one `dT`, and another diagonal scaling. The Laplacian is `dδ + δd` from those
  pieces. Its effective stencil spans offsets in `{−1,0,1}^D`: at `D = 1` there are already 3
  offsets against `2^D = 2`. So the `2^D` bound does not apply to it.
- **Cup products** are bilinear, so they are a separate contraction. The table has rows
  `(out_label, a_label, a_offset[D], b_label, b_offset[D], sign)` and is cached per
  `(D, p, q, i)`, keyed on both input degrees. Application gathers `a` and `b` at their
  offsets, multiplies, and accumulates into `out_label`. There is no "one sparse matrix on one
  rolled input" form.
- **No Python loops over sites** anywhere; this is a hard requirement. A global CSR export
  exists for tests only.
- **Size to plan for.** 240 fine links per site against 4 hypercubic ones, a factor of 60. At
  `D = 4, N = 8` that is about 0.98 M links and 3.3 M triangles.
- **Measurement, not a gate.** Benchmark `d`, `δ`, `∪` (1∪1), and `∪_1` (2∪_1 2) at `D = 4`,
  `N ∈ {4, 8}`. Report time per fine output cell next to hypercubic `compact.d` per cell. Numba
  kernels are added only if measurement shows a need.
- **Deferred.** Local-update analogues of `d_sparse`, `delta_sparse`, and `coface_sum_at`. The
  tables above already contain the coface lists they need.

## 10. Module layout and upstream constraints

- `supervillain/lattice/barycubic.py`: `BarycubicLattice(D, N)`, `BarycubicForm`,
  `BarycubicDualForm`, and module functions `d`, `delta`, `laplacian`, `star`, `inverse_star`,
  `inner`, `cup(a, b, i=0)`, `coarsen`, `transform`, `half_shift`, `integrate`.
- Re-export from `supervillain/lattice/__init__.py` under distinct names (`barycubic`), with no
  changes to `compact.py`.
- Serialization (section 2.3): a one-line tag registration in `supervillain/batch.py`, plus a
  standalone-form strategy under `supervillain/h5/strategy/`. These are the only edits to shared
  upstream files.
- Tests: `test/test_barycubic_complex.py` (1–6), `test_barycubic_operators.py` (7–8, 13–15),
  `test_barycubic_cup.py` (9–12), and `test_barycubic_reference.py` (4).
- Branch from a freshly fetched `main`. The module imports nothing outside the package's
  existing dependencies (numpy, scipy, numba, h5py; Python ≥ 3.14).

## 11. Design decisions

- **Q1 → keep `star`.** `BarycubicDualForm` is implemented as in section 5.3, with tests
  (section 7, item 15).
- **Q2 → dispatch by form type.** The package-level `supervillain.lattice.d`, `delta`,
  `laplacian`, `star`, `wedge`, and `cup` look at the argument's type. Barycubic forms go to
  `barycubic.py`, hypercubic forms to `compact.py`.
  - The per-module functions stay importable directly.
  - Each dispatcher carries the hypercubic docstring, plus a note, so the existing docs are
    unchanged.
  - Upstream has no hypercubic `cup`, so on a hypercubic `Form`, `cup(a, b, 0)` is `wedge` and
    `i > 0` raises `NotImplementedError`. When the branch is merged into the fork, that branch
    of the dispatcher points at the fork's `compact.cup`.
  - The dispatch layer lives in `supervillain/lattice/__init__.py`.
- **Q3 → defer** local-update machinery until there is an action.
  - **Link colouring, deferred with Q3.** Take single-link
    `n` updates at fixed φ under q = f∪f in D = 4. Each top simplex's charge reads exactly one
    link from each of the six classes (0,1), (0,2), (1,2), (2,3), (2,4), (3,4). So colouring
    links by class is a proper colouring, and it is optimal, because every read set is a
    6-clique. It works for every N.
    - The four unconstrained classes (0,3), (0,4), (1,3), (1,4) conflict with nothing at fixed φ.
    - A greedy colouring of the explicit conflict graph also found exactly 6 colours, at N = 4
      and N = 5. The maximum conflict degree is 84.
    - A future `link_coloring` can be read off `rank_sets[1]`.

## 12. Verification history

Two rounds of independent adversarial review were performed against this design and its
implementation, in addition to the derivation checks described in section 5.5.

The first round reviewed the initial design and confirmed the counts and the `N = 2`
simplicial property; the orientation, fundamental-cycle, coarsen-chain-map, and
cohomology-ring sign; the closed-form cup signs (by an independent symbolic check); all Hodge
sum rules and rank-set invariance; the Fourier topology statement; and the availability of
`compact.Lattice`, `Form`, and `d` in the base package. It also raised nine findings, all
accepted and addressed in this revision:

- The sign of `∪_D` is invisible to every legal Steenrod identity in dimension `D` (one free
  coefficient at each `D = 2..6`), and the pointwise `a ∪_p b = ab` test had been dropped.
  Fixed: "forced" is scoped to the universal family (5.5); the pointwise test is mandatory for
  all `p ≤ D`; the sign table is compared with the closed form; a uniqueness certificate script
  was added.
- A new batch tag alone is not enough for HDF5; `resolve_batch_cls` only recognizes `"Form"`.
  Fixed: registration and a standalone serialization strategy are in scope (2.3, 10), with
  round-trip tests (7.14).
- Rank coloring cannot use the spatial-only checkerboard index format. Fixed: full-axis
  selectors, stated as an intentional API difference, with tests (6.3).
- Differences of mod-`2N` representatives are wrong across seams. Fixed: mandatory common lift
  (2.2), plus a seam regression test (7.11).
- Odd-`i` reversal also holds on the whole diagonal `p = q = i`. Fixed: text corrected; the
  checker asserts both the successes and the failures.
- The dtype and complex contracts were incomplete. Fixed: `star` promotes; `inner` is Hermitian;
  Fourier tests use `d(k)^†`.
- The application recipe conflated incidence operators, the Laplacian, and cups, and the cup
  cache key lacked both degrees. Fixed: separate contracts in section 9; the cup key is
  `(D, p, q, i)`.
- A remark claiming the cubical cup fails Leibniz or associativity was wrong (it satisfies
  both). Fixed: the remark was removed (7.9).
- `2^{D+1} − 1` counts rank-set entries, not distinct values. Fixed: reworded (5.1).

The second round of review, after the first implementation and test pass, used mutation
testing and independent kernel re-derivation, and found:

- `wedge`/`cup` on mixed hypercubic and barycubic forms gave a silent wrong answer: fixed, with
  a regression test.
- HDF5 writes of form views could not be read back: fixed, with a regression test.
- Shared per-`D` tables could be mutated by users: fixed, with a regression test.
- Unsigned and bool dtypes were accepted where they should not be: fixed, with a regression
  test.
- Degree and shape checks were missing in places: fixed, with regression tests.
- Forms lost their metadata when pickled: fixed, with a regression test.
- Integer overflow inside `laplacian` could make `⟨f, Δf⟩` negative (this predated the
  performance work below): fixed, with a regression test.
- No test covered complex values in the cup kernel (a mutation dropping imaginary parts
  survived): fixed, with a regression test.
- The export comparer ignored duplicate CSR entries: fixed.
- `transform` and the lattice constructor silently truncated non-integer arguments: fixed, with
  a regression test.
- Three documentation claims were wrong: the complex kinetic pairing (needs a conjugate), an
  allocation claim, and "one shared vertex" for `∪_i`: all corrected.

The same pass moved operator application into numba kernels (2–5x faster, with peak memory
about the size of the output), verified against the previous numpy path over 3264 comparisons
across all dtypes, `D = 1..4`, `N = 2, 3`, and several memory layouts, and against the weights
in exact rational arithmetic through `D = 5`. It also added `export_cup`, the export writer and
comparer scripts, and a benchmark.

**Known and deferred.**
- The package's numba kernel caching (`_kernels.py`, `cache=True`) could collide on disk across
  several closures, producing intermittent `NRT_adapt_ndarray_to_python` errors in hypercubic
  code. This predated the barycubic module. It is fixed separately by giving each cached variant
  its own qualname, with a two-process regression test (`test/test_lattice_kernel_cache.py`).
- `transform` rebuilds its site-index arrays on each call, about 0.5 of 3.2 ms at `D = 4, N = 8`,
  because caching them per translation would grow with `N^D`.

## 13. Documentation standard

The rest of the package is carefully documented, largely by hand, and maintainability is the
point. The barycubic code must read the same way.

- **Docstrings.** Every public class, method, and function gets a numpy-style docstring:
  Parameters, Returns, Raises, and the defining formula in `.. math::`. Match
  `supervillain/lattice/compact.py`. Explain *what* is computed and *why* the convention is what
  it is. No references to "SPEC §" in user-facing docstrings; those are for developers, in
  comments.
- **Comments.** Explain the reason for every non-obvious step, such as the common lift, sign
  bookkeeping, or table layouts. Do not restate the code.
- **Sphinx page.** Add `supervillain/barycubic.rst` to the `index.rst` toctree after
  `supervillain/form`. It is a pedagogical walk-through:
  - why the subdivision (the no-intersection constraint);
  - the construction, with a two-dimensional figure via `.. plot::` from a new
    `example/plot/barycubic-2d.py`;
  - ranks, doubled coordinates, and labels;
  - orientation;
  - `d`;
  - weights, including a table of the D=4 link weights;
  - `δ`, the Laplacian, and `star`;
  - the cup product and the Steenrod products, with the sign rule and one small worked example;
  - symmetries, `coarsen`, and dispatch;
  - interop export.

  Autodoc entries are woven into the narrative, following `supervillain/form.rst`, and tested
  properties are cited with `:source:` links to the tests.
- **Build.** `sphinx-build` must add no new warnings.

## 14. Clarifications from the build

These open points were resolved during implementation and are now part of the contract.

- **`operator_table('dT', degree)`.** `degree` is the *input* degree. So `'dT'` at `p+1` maps
  `(p+1)`-cochains to `p`-cochains, with offsets in `{0,−1}^D`.
- **`transform`.** `flips` are ±1 integers, and `perm[i]` names the source coordinate of output
  coordinate `i`. So `det g = sgn(perm) · Π flips`.
- **Dual forms.**
  - `BarycubicDualForm.dual` is `True`, and `degree` is the dual degree `D − p`.
  - Pairing means `(a * star(b)).sum() == inner(a, b)` for real forms.
  - Mixing primal and dual forms in arithmetic gives a plain ndarray.
  - Operations that make no sense on a dual form (`d`, `delta`, `laplacian`, `star`, `inner`,
    `cup`, `integrate`, `coarsen`, `transform`, `half_shift`) raise `TypeError`.
- **Pickle size.** `Batch` pickles its item arguments, including the lattice, into a single HDF5
  attribute, and HDF5 caps attributes at 64 KB. `BarycubicLattice.__reduce__` therefore
  reconstructs from `(D, N)` alone and carries no cached tables. A D = 4 lattice with its tables
  cached would otherwise exceed the limit.
- **Winding test.** `w = Δξ/2` takes values ±1/2. The exact integer statement tested is
  `∫ u_{π(0)} ∪ … ∪ u_{π(D−1)} = sgn(π) (2N)^D` for `u = 2w`.
- **Operator tables are per `D`.** They are built once per `D` and shared by every lattice of
  that `D`. At D = 4 the cost is 12 ms for labels, 52 ms for the `d` tables, 18 ms for the
  weights, and 313 ms for all 35 cup tables.
