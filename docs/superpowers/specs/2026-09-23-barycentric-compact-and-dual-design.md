# The compact barycentric lattice, and its dual — design

## Problem

`supervillain/lattice/barycentric.py` implements the barycentric subdivision
$S = \mathrm{sd}\,K$ in the obvious way: a degree-$p$ cochain is an array of
shape $(T_p,) + (2N,)^D$, a type axis over a doubled site array, with a slot
realized only where every direction the type releases is even at the anchor.
That is sparse by construction, and increasingly so with degree — occupancy
runs $100\%, 19\%, 11\%, 8\%, 6\%$ for $p = 0 \ldots 4$ in $D = 4$.

| all five degrees, float64 | $N=4$ | $N=6$ | $N=8$ | $N=12$ |
| --- | --- | --- | --- | --- |
| doubled (today) | 53.0 MiB | 268.5 MiB | 848.5 MiB | 4295.5 MiB |
| compact (this spec) | 4.7 MiB | 23.7 MiB | 75.0 MiB | 379.7 MiB |

The doubled layout is fine to $N \approx 6$ and impossible by $N = 12$. This
spec adds a space-efficient implementation beside it, in the relationship
`compact.py` already has with `interlaced.py`, and — because it turns out to
cost almost nothing once the layout exists — a Hodge star, which the doubled
module does not have.

## The compact layout

A *component* is a realized (anchor parity pattern, type) pair: a parity
pattern $S \subseteq \{0,\ldots,D-1\}$ together with a type whose released
directions all avoid $S$. Anchors of a given parity form an $N^D$ sublattice,
so each component occurs exactly once per original site, and

$$\#\{\text{components of degree } p\} = \#\{p\text{-simplices per original site}\} = S_p .$$

Verified: $S_p = 16, 240, 800, 960, 384$ for $p = 0 \ldots 4$ at $D = 4$,
matching the simplex counts exactly. So a degree-$p$ cochain is **one dense
array** of shape $(S_p,) + (N,)^D$ at 100% occupancy — no ragged blocks, no
dict of arrays, no sparse structure. This is the same shape
`compact.Form` already uses, $\bigl(\binom{D}{p}, N, \ldots, N\bigr)$, with a
richer component index.

Components are ordered lexicographically by (parity bitmask, type index), so
the order is reproducible and greppable.

## Operators are local tables

Each operator becomes a table: for every out-component, a list of
(in-component, site shift, sign). This is the form
`compact.operator_table(op, degree)` already takes, and it is what a later
numba pass would consume.

The shifts are **local**. The coboundary re-anchors by a $\pm1$ step in doubled
coordinates, which could in principle cross an original site; measured, every
doubled shift the coboundary uses has components in $\{-1,0,+1\}$, so the
original-site displacement lies in $\{-1,0\}$ per direction. Every operator is
a one-site stencil.

## `Form` carries `.dual`

Dual blocks are in bijection with primal simplices: the dual of a $p$-simplex
is a $(D-p)$-cell, one per simplex. A dual $(D-p)$-cochain therefore holds one
number per $p$-simplex and lives in **exactly the primal $p$ layout** —
identical shape, identical component index, identical bytes. What differs is
only how the numbers are read.

That is precisely why this needs guarding. A primal 2-form and a dual 2-form on
a $D=4$ lattice have the *same array shape*, so numpy will silently add them
unless we stop it.

`Form` therefore carries, alongside `.lattice`:

- `.degree` — the dimension of the cell the number lives on;
- `.dual` — which complex it lives on.

The component axis is always the **primal** index: length $S_{\text{degree}}$
for a primal form, $S_{D-\text{degree}}$ for a dual one. Keeping `.degree` the
cell dimension (rather than the index degree) makes the operators compose:
$\star$ takes primal $p$ to dual $D-p$, and $\star\star$ returns to primal $p$.

## Legality

The guard is in two places.

**Arithmetic.** `__array_ufunc__` currently requires all `Form` inputs to agree
on `degree`; it will require agreement on `(degree, dual)`. Mixed pairs degrade
to a plain `ndarray` exactly as mixed degrees already do, so an illegal
combination silently loses its `Form`-ness rather than pretending to be one.

**Operators.** Each checks `.dual` and either dispatches or refuses:

| operator | primal input | dual input |
| --- | --- | --- |
| `d` | primal $p \to$ primal $p+1$ | dual $k \to$ dual $k+1$, via the transpose tables |
| `delta` | primal $p \to$ primal $p-1$ | dual $k \to$ dual $k-1$ |
| `cup` / `wedge` | primal $\times$ primal $\to$ primal | **refused** |
| `star` | primal $p \to$ dual $D-p$ | dual $k \to$ primal $D-k$ |
| `push`, `pull`, `translate` | either | either |
| `reflect`, `permute` | either | either |
| `tau`, `translate_diagonal` | either | either |

`cup` is refused on dual input rather than dispatched: Alexander–Whitney is a
simplicial construction, and the dual block complex is not simplicial. There is
no dual cup to dispatch *to*, so a silent answer would be a wrong one.

`d` and `delta` do dispatch, because the coboundary genuinely exists on both
complexes and the dual one is the transpose of the primal — machinery `delta`
already builds. Every refusal names both the operator and the offending flag.

## The star

Combinatorially, $\star$ is diagonal in the component index:

$$(\star a)\bigl(D(\sigma)\bigr) = \pm\, a(\sigma)$$

the sign being the orientation convention relating $\sigma$ to $D(\sigma)$. It
costs a table of signs and no storage at all, which is why it ships with the
layout rather than later.

The **metric** star, carrying $|D(\sigma)|/|\sigma|$, is deferred. The cheap
diagonal DEC star is exact only when primal and dual cells meet orthogonally,
which needs a circumcentric dual on a well-centred mesh; measured, every fine
simplex here is an orthoscheme and the circumcentre of every top simplex has
minimum barycentric coordinate exactly $0$ — it lies *on* the boundary, in
$D = 2, 3, 4$. So the circumcentric dual is degenerate and the diagonal ratio is
not the metric star. A correct metric star is a Whitney/Galerkin mass matrix,
dense rather than a relabeling, and belongs in its own spec if a model needs it.

## Cross-validation is the deliverable

`Form` gains `to_doubled()` and `from_doubled()`, mirroring
`compact.Form.to_interlaced()` / `from_interlaced()`. The tests that matter
assert **compact $\equiv$ doubled** for every operator at $D = 2, 3, 4$ — the
analogue of `test_compact_wedge_matches_interlaced`. Without those, the compact
layout is an unverified rewrite; with them, the doubled module stops being a
nice artifact and becomes a guarantee.

## Layout on disk

```
supervillain/lattice/barycentric/
    __init__.py     # hoists the production names
    compact.py      # this spec
    doubled.py      # today's module, renamed
```

Imported as `supervillain.lattice.barycentric as bary`, giving `bary.Lattice`,
`bary.d`, `bary.delta`, `bary.cup`, `bary.star`. The names are not hoisted into
`supervillain.lattice`, where they would collide with the hypercubic ones.

Every lattice kind stays under the lattice layer; the top level is organized by
layer (`action`, `observable`, `generator`, …) and a lattice kind does not
belong there.

## API parity, and one deliberate divergence

The compact module presents the same names as the doubled one: `Lattice(D, N)`
with `.D`, `.N`, `count(p)`, `measure(p)`, `planes(p)`, `pvector(p)`,
`link_class`, `orientation_sign`, `zeros`, `random`, `random_integer`.

`support(p)` is **not** carried over. In the doubled layout it masks the slots
that are not simplices; in the compact layout every slot is one, which is the
entire point. Code that multiplies by `support` is doing something the compact
layout makes unnecessary, and should fail loudly rather than silently get a mask
of ones.

## Testing

1. **Counts** — $S_p$ matches the simplices per original site at every degree,
   $D = 2, 3, 4$.
2. **Round trip** — `from_doubled(to_doubled(a)) == a`, and `to_doubled` lands
   only on slots the doubled `support` admits.
3. **Cross-validation** — for `d`, `delta`, `cup`, `push`/`pull`, `reflect`,
   `permute`, `tau`, `translate_diagonal`: compact and doubled agree after
   `to_doubled`, at $D = 2, 3, 4$. This is the spec's reason to exist.
4. **Intrinsic identities** re-checked in the compact layout, so it stands on
   its own if the doubled module is ever dropped: $d^2 = 0$, adjointness, cup
   associativity and Leibniz, the point-group order, $\tau$'s class map.
5. **Star** — $\star$ toggles `.dual`, $\star\star$ returns to the primal with
   the expected sign, and the pairing $\langle a, \star b \rangle$ agrees with
   the plain sum over components.
6. **Legality** — primal + dual arithmetic degrades to `ndarray`; `cup` on a
   dual form raises; `d` on a dual form dispatches to the dual coboundary and
   agrees with the transpose of the primal one.
7. **$\star d \star = \pm\,\delta$.** Composing gives
   $p \to D-p \to D-p+1 \to p-1$, so the composite *lowers* the degree and
   should reproduce the codifferential we already build as the transpose. This
   ties the new star to existing machinery and is the sharpest check that the
   $\sigma \leftrightarrow D(\sigma)$ orientation convention is self-consistent.

   .. note: on the hypercubic lattice this identity holds only up to a
   translation (see the interlaced documentation).  With a combinatorial star
   that is the identity on the component index, it may well hold exactly here
   --- to be measured, not assumed, and whichever way it comes out is worth
   recording.
7. **Memory** — a degree-$p$ compact cochain is $S_p N^D$ entries, asserted
   rather than assumed.

## Open, to resolve during implementation

- **$\tau$'s sign on a dual form.** On a primal $p$-cochain $\tau$ carries
  $(-1)^{p(p+1)/2}$, from reversing the simplex's vertex order. A dual form is
  indexed by primal simplices, so the sign should follow the *index* degree
  rather than the cell dimension — but this needs deriving and testing, not
  assuming, and the cross-validation against the doubled module will settle it.
- **The orientation convention relating $\sigma$ to $D(\sigma)$**, which fixes
  the sign in $\star$ and hence the sign in $\star\star$. Fix it by requiring
  $\star\star = \pm 1$ with a stated sign and testing it.

## Deferred

- The metric star (Whitney/Galerkin mass matrix) and dual block volumes.
- numba kernels over the operator tables.
- Any action, weight, or Monte Carlo built on this lattice.
