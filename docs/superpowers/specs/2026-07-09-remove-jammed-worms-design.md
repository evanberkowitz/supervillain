# Retiring the jammed no-intersection worms

**Date:** 2026-07-09
**Status:** design, pending review

## Goal

The `no_intersection` generator package carries four worms.  Only two earn their keep as
documented objects of study, and none is fit to be the production sampler.  Retire the
two that are pure dead weight, keep the two that bracket the story, and replace prose
assertions about *why* the worms failed with measurements.

Concretely:

- **Delete** `AdaptiveIntersectionWorm` and `TwoLinkAdaptiveWorm` (the whole
  `adaptive_worm.py`).
- **Keep** `IntersectionWorm` (the naive worm) and `FreeTargetWorm` (the culmination of
  the adaptive idea).  Both remain opt-in; neither is in the `Hammer`.
- **Replace** the worm in `Hammer` with the `DefectGas`.
- **Document** the retired worms with measured evidence that they jammed, and correct
  the doc's explanation of *why*, which measurement shows is wrong.

## Measured evidence

All numbers: $D = 4$, $N = 6$, thermalized from cold with the current `Hammer`.
Clean sets sampled over (head, direction, sign) for the direction-drawing worms and over
head for the free worm.  The adaptive and free-target rows are reproduced by
`example/no-intersection/worm_jam.py` (new); the two-link rows were measured against the
live class before deletion and are historical (see §5).

### The clean sets

| $\kappa$ | worm | median $\lvert C\rvert$ | mean | **frac $\lvert C\rvert = 0$** | max |
|---|---|---|---|---|---|
| 0.1 | adaptive | **0** | 0.24 | **0.814** | 5 |
| 0.1 | two-link | **0** | 8.51 | **0.572** | 225 |
| 0.1 | free-target | 1208 | 1560 | 0.008 | 5618 |
| 0.5 | adaptive | 2 | 2.90 | **0.497** | 6 |
| 0.5 | two-link | 12 | 14.36 | 0.004 | 279 |
| 0.5 | free-target | 192 | 486 | 0.000 | 15409 |

The adaptive and two-link worms jam: on the warm, dense $\kappa = 0.1$ sheet the median
hypercube admits **no clean mover** in a drawn direction.  The adaptive worm is dead in
81% of (head, direction, sign) triples; the two-link worm in 57%.

### The free-target worm does *not* jam

On the same backgrounds it finds a median of **1208** clean movers.  Its clean set is not
the problem.

Composition of the free-target clean union at $\kappa = 0.1$ (40 heads):

| set | count | share |
|---|---|---|
| full union $C$ | 704,713 | — |
| idles | 635,880 | **90.2%** |
| movers | 68,833 | 9.8% |

Movers by link count: 2-link 68,675; 1-link 120; 3-link 26; 4-link 12.
Movers by target distance ($L_1$): 1 → 34,168; 2 → 34,454; $\geq 3$ → 211.

So the free-target worm's movers are overwhelmingly **local two-link transports** — the
very shapes `TwoLinkAdaptiveWorm` was built to find, and starves for.  The difference is
family breadth: free-target admits $\lvert c\rvert \leq 2$ over a proximity ball with
*either* slot touching the head, and allows $L_1 = 2$ and diagonal targets.

### The free-target worm fails on transport, not starvation

Worm lengths, 160 worms per cell.  `Worm_Length = 1` is a **zero-length worm**: the worm
opened, its opening proposal was rejected or was an accepted idle, head never left tail,
nothing was transported.

| $\kappa$ | worm | median len | mean | **frac(len = 1)** | max |
|---|---|---|---|---|---|
| 0.1 | `IntersectionWorm` | 25 | 221.1 | 0.019 | 23738 |
| 0.1 | `FreeTargetWorm` | **1** | **1.00** | **1.0000** | **1** |
| 0.5 | `IntersectionWorm` | 20 | 29.4 | 0.025 | 210 |
| 0.5 | `FreeTargetWorm` | **1** | **1.00** | **1.0000** | **1** |

`FreeTargetWorm` transported the defect **zero times in 160 worms, at both $\kappa$**.

`_run_free_worm` auto-closes the instant the head returns to the tail, and the worm opens
with head $=$ tail.  So an accepted *idle* at the opening step closes the worm at length 1,
and transport requires drawing a mover **and** accepting it.  Probing the opening step
directly (25 heads, $\Delta S$ from `_delta_S`; the Hastings factor
$\lvert C\rvert/\lvert C'\rvert \approx 1$ is neglected):

| $\kappa$ | $P(\text{draw mover})$ | $\mathbb{E}[\min(1,e^{-\Delta S}) \mid \text{mover}]$ | median $\Delta S$ | frac $\Delta S \leq 0$ | $P(\text{transport at open})$ |
|---|---|---|---|---|---|
| 0.1 | 0.0656 | 0.0627 | 6.01 | 0.023 | $4.1 \times 10^{-3}$ |
| 0.5 | 0.0007 | $\sim 0$ (max $5\times10^{-4}$) | 21.06 | 0.000 | $2.3 \times 10^{-10}$ |

Both mechanisms contribute, and **acceptance dominates**:

- At $\kappa = 0.1$ the two are comparable, and their product $\approx 1/243$ makes zero
  transports in 160 worms unsurprising.
- At $\kappa = 0.5$ acceptance annihilates the worm: median $\Delta S = 21.06$, and not one
  of 4756 movers had $\Delta S \leq 0$.  The union is 99.93% idles.

Consequently `idle_probability=0.0` (movers-only draw) would rescue transport at
$\kappa = 0.1$ — roughly 1 opening in 16 — but not at $\kappa = 0.5$, where it buys
$\sim 3 \times 10^{-7}$.  Idle swamping is real but **secondary**.  The killer is that
demanding exact repair at every move forces expensive multi-link shapes (movers are 99.8%
two-link), whose cost scales with $\kappa$.

This is exactly what the `DefectGas` escapes: it pays $\zeta^{\Delta D}$ on a *cheap
single-link* move rather than $e^{-21}$ on a *mandatory two-link* exact repair.

### The inclusion lemma

Every adaptive or two-link mover at head $h$ changes $q$ at $h$ and at $h + \hat d$, so its
charge-reach support contains the head; `FreeTargetWorm._build_family` blocks (a)–(c)
admit every such shape (block (c) explicitly folds in $\pm$ the parent library's
multi-link templates at every support-anchored placement).  Hence

$$C_\text{adaptive}(\hat d) \subseteq C_\text{two-link}(\hat d) \subseteq C_\text{free}(\text{movers to } h + \hat d).$$

Verified pointwise at $\kappa = 0.1$ over 960 (head, direction, sign) samples: the
inclusion held in every sample.  This is recorded as documentation of why the retired
worms could not have done better than the surviving one; it is not enforced by a test,
since after deletion there is no longer a superset relation to protect.

## Corrected narrative

The current `no_intersection.rst` says:

> enriching the candidate family (wider coefficients, farther-flung pairs, millions of
> shapes) does not help, because the jam is structural

**This is false.**  Enriching the family is exactly what unjams the clean set: median 0 →
1208 on identical backgrounds.  The starvation comes from **drawing a direction first**,
not from the constraint's structure.

The corrected hierarchy — each fix exposing the next failure:

1. **`IntersectionWorm`** — one fixed stencil, chosen before looking at $F$.  Rejected on
   cold backgrounds; constraint-violating on warm ones.
2. **`AdaptiveIntersectionWorm`** — look at $F$ before proposing.  Still starves: it
   commits to a *direction*, then asks which of a fixed library advances the head that
   way.  81% of the time nothing does.
3. **`TwoLinkAdaptiveWorm`** — enrich the two-link sector by live enumeration.  Helps by
   ~35× in the mean, but the median is still zero: the direction is still drawn first.
4. **`FreeTargetWorm`** — drop the direction; let $\Delta q$ land where it wants.  The
   starvation **vanishes**.  And the worm still fails, now on *acceptance*: its movers are
   99.8% two-link exact repairs costing a median $\Delta S$ of 6 at $\kappa = 0.1$ and 21
   at $\kappa = 0.5$.  It transported zero times in 320 worms.

This is what makes the `DefectGas` *necessary* rather than merely convenient.  A worm's
clean set can be unjammed — `FreeTargetWorm` proves it.  What cannot be fixed, while
insisting the constraint be exactly repaired at every move, is that exact repair *forces
expensive shapes*: the cheapest legal mover is a two-link move whose cost grows with
$\kappa$, and the excursion is all-or-nothing.  The gas stops insisting, and buys a cheap
single-link move by paying $\zeta^{\Delta D}$ instead.

## Design

### 1. Code surgery

Delete `supervillain/generator/no_intersection/adaptive_worm.py`
(`AdaptiveIntersectionWorm`, `TwoLinkAdaptiveWorm`, `_ravel`).

`FreeTargetWorm` re-parents onto `IntersectionWorm`, absorbing exactly what it used from
the deleted middle class and nothing more:

| absorbed | why needed |
|---|---|
| `_ravel` | flat index of a hypercube site |
| `_shape_self_charge` | a shape's background-independent $d\Delta n \wedge d\Delta n$ |
| `_scaled_self_charge` | scales the unit cross-charge by $c_1 c_2$ |
| `_reach_touching_slots` | family blocks (a) and (b) |
| `self._scratch` lattice | `_shape_self_charge` derives on it |

It does **not** use `_ortho`, `_accept`, `_mover_shapes`, or any `clean_set_*` /
`clean_idle_*` method; those die with the file.

`two_link_kernel.clean_mask` becomes orphaned (only `TwoLinkAdaptiveWorm` called it) —
delete it and fix the `flatten_family` docstring that references it.  `dq_stencil_arrays`,
`flatten_family`, and `classify_mask` all survive (`defect_gas_kernel` and
`FreeTargetWorm` need them).

### 2. `Hammer`

`Hammer(S, zeta=None)`; `None` calls `DefectGas.tune(S)`.  The worm slot becomes
`DefectGas(S, zeta)`.  The remaining generators are unchanged.

Note: `Hammer(S)` construction now runs Monte Carlo (≈60-sweep probes down a 6-rung
ladder).  Test call sites pass an explicit small `zeta` to stay cheap; only production
`Hammer(S)` tunes.

### 3. Observables

`Hammer` no longer emits `Intersection_Intersection`, because only a worm emits it.  The
gas emits `Theta_Theta` + `Vacuum_Ticks`, from which the correlator is a ratio.

- **`Intersection_Intersection` is promoted from `Observable` to `DerivedQuantity`**, with
  a `NoIntersections` implementation

  ```python
  @staticmethod
  def NoIntersections(S, Theta_Theta, Vacuum_Ticks):
      # Theta_Theta / Vacuum_Ticks, with the identically-unit origin bin restored.
  ```

  mirroring the existing `IntersectionSusceptibility(S, Theta_Theta, Vacuum_Ticks)`.
  It is now the absolutely-normalized $\Theta_{\Delta x}$.  The gas's `Theta_Theta` origin
  bin is *empty by construction* (a coincident pair is the vacuum), so the implementation
  must write $\Theta_0 = 1$ into `S.Lattice.origin` rather than divide it out.

- The worms' inline histogram becomes a new inline-only `Observable`,
  **`IntersectionTwoPoint`**.  This follows the library's existing raw-ingredient /
  physical-correlator convention: `ActionTwoPoint` (`Observable`) → `Action_Action`
  (`DerivedQuantity`), and `TopologicalTwoPoint` → `Topological_Topological`.  (Note
  `Spin_Spin` and `Vortex_Vortex` are *themselves* the worm-emitted inline observables,
  so they are not the precedent to copy here.)  `IntersectionWorm.inline_observables` and
  `FreeTargetWorm._run_free_worm` emit under the new name.

- `Intersection_Intersection_Normalized` keeps its `default(S, Intersection_Intersection)`.
  For a gas ensemble it now returns $\Theta$ unchanged (the origin is already 1).  Worm
  users normalize `IntersectionTwoPoint` by its origin bin themselves; the worm histogram
  is a raw ingredient, exactly as `ActionTwoPoint` is.

**Migration cost:** existing h5 ensembles store an inline `Intersection_Intersection`
field.  After the promotion, that name resolves to a `DerivedQuantity` on `Bootstrap` and
has no `Observable` on `Ensemble`, so the stored field is orphaned.  The
`NoIntersections` production was split into its own repository at 3655d3f, so the blast
radius should be small, but this must be checked against any archived ensembles before
merge.

### 4. Documentation — `supervillain/no_intersection.rst`

The escalation narrative stays **inline**: it is the motivation for the `DefectGas`, and
splitting it out would leave the gas looking arbitrary.

- `.. autoclass::` for `AdaptiveIntersectionWorm` and `TwoLinkAdaptiveWorm` → prose, in a
  `.. note::` marking them retired, citing the git SHA where the code lives.
- `FreeTargetWorm` gets a live `.. autoclass::`, presented as the culmination *and* as a
  failure — the experiment that separates starvation from transport.
- New subsection carrying the evidence tables and the inclusion lemma, citing
  `:source:example/no-intersection/worm_jam.py`.
- The false "enriching the family does not help / the jam is structural" claim is
  rewritten per **Corrected narrative** above.
- `.. autoclass::` for the promoted `Intersection_Intersection` moves to the
  `DerivedQuantity` list.

### 5. Evidence script — `example/no-intersection/worm_jam.py`

Reproduces the adaptive and free-target rows from **live code only**:

- $C_\text{adaptive}(\hat d)$ reconstructed exactly from `IntersectionWorm._library[dd]`,
  `_change_from_shape`, and `_local_dq` — all `IntersectionWorm` methods that survive.
  (`AdaptiveIntersectionWorm.clean_set_local` was nothing more than this loop.)
- $C_\text{free}$ measured directly, with the idle/mover split and the mover composition.
- Worm-length distributions for `IntersectionWorm` and `FreeTargetWorm`.

The **two-link row is historical**: it was measured against the live
`TwoLinkAdaptiveWorm` before deletion (numbers above) and is recorded in the docs with the
git SHA.  The script does not attempt to reconstruct it.  The inclusion lemma is stated in
prose as a claim, not enforced by a test — there is no longer a superset to protect.

### 6. Tests

- Delete `test_adaptive_worm.py`, `test_two_link_worm.py`.
- `test_free_target_worm.py`: **delete** `test_merged_set_contains_two_link_worm_moves`.
  Its whole purpose was to guard free-target's family against `TwoLinkAdaptiveWorm`'s;
  with that class gone there is no superset relation left to protect.  Update the
  inline-histogram name throughout the rest of the file.
- `test_no_intersection_generators.py`:
  - `test_hammer_includes_constraint_preserving_villain_updates` asserts
    `'IntersectionWorm' in str(H)` and passes **today only because
    `"AdaptiveIntersectionWorm"` contains that substring**.  It becomes `'DefectGas'`.
  - `test_intersection_intersection_normalized_is_one_at_origin` rewrites against the
    gas's `Theta_Theta` / `Vacuum_Ticks`.
  - `test_intersection_intersection_is_inline_only` inverts: the observable is no longer
    inline-only, it is a `DerivedQuantity` with a `NoIntersections` implementation.
  - Hammer call sites pass an explicit `zeta`.
- `test_worm_local.py`: inline-histogram rename only.

### 7. Files touched

| file | change |
|---|---|
| `generator/no_intersection/adaptive_worm.py` | **deleted** |
| `generator/no_intersection/free_target_worm.py` | re-parent onto `IntersectionWorm`; absorb 4 helpers + `_scratch`; rename inline histogram |
| `generator/no_intersection/worm.py` | rename inline histogram |
| `generator/no_intersection/two_link_kernel.py` | delete `clean_mask`; fix `flatten_family` docstring |
| `generator/no_intersection/__init__.py` | drop two exports; rewrite `Hammer` (signature, body, docstring) |
| `observable/intersection.py` | promote `Intersection_Intersection` to DQ; add inline `IntersectionTwoPoint` |
| `observable/__init__.py` | update the `.intersection` import line |
| `supervillain/no_intersection.rst` | retired-worm prose, evidence tables, corrected narrative, autoclass moves |
| `example/no-intersection/worm_jam.py` | **new** — the evidence script |
| `example/no-intersection/observables.py` | update correlator names |
| `test/test_adaptive_worm.py`, `test/test_two_link_worm.py` | **deleted** |
| `test/test_free_target_worm.py`, `test/test_no_intersection_generators.py`, `test/test_worm_local.py` | per §6 |

## Open questions

1. **h5 back-compat** for archived ensembles carrying inline `Intersection_Intersection`
   (see §3 migration cost).  Must be checked against any archived ensembles before merge.

2. **`zeta` default in tests.**  §2 has test call sites pass an explicit small `zeta` so
   `Hammer(S)` construction does not run `DefectGas.tune`'s Monte-Carlo probes.  A value
   must be chosen that emits vacuum ticks promptly at the test volumes ($N = 3$–$6$) —
   `tune`'s ladder starts at 0.1 — without hanging.  Pick during implementation and assert
   the emitted configurations satisfy the constraint.
