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

**$\kappa$ is a temperature knob: small $\kappa$ is the warm, dense, jammed phase.**  The
jammed phase lies *below* $\kappa \approx 0.02$; $\kappa = 0.1$ and $0.5$ are outside it.

frac $\lvert C\rvert = 0$, the fraction of proposals with no legal move:

| $\kappa$ | adaptive | two-link | free-target | free-target median $\lvert C\rvert$ |
|---|---|---|---|---|
| 0.01 | **0.977** | **0.948** | **0.488** | **1** |
| 0.03 | 0.947 | 0.848 | 0.038 | 450 |
| 0.05 | 0.869 | 0.739 | 0.000 | 594 |
| 0.1 | 0.814 | 0.572 | 0.008 | 1208 |
| 0.5 | 0.497 | 0.004 | 0.000 | 192 |

Two facts, and they pull in different directions:

- **The directional worms are dead everywhere.**  The adaptive worm never gets below a
  50% dead fraction, even at $\kappa = 0.5$.  In the jammed phase it is dead 98% of the
  time, with a maximum clean set of **one** move.
- **Free-target starves only in the jammed phase.**  Below $\kappa \approx 0.02$ half its
  hypercubes admit no mover at all and the median is a single move.  Above it, enrichment
  rescues the clean set completely (median 450–1208).

So enriching the candidate family *does* help — it takes the dead fraction from 97.7% to
48.8% at $\kappa = 0.01$ — but in the jammed phase it does **not rescue** the worm.  Both
halves of that sentence matter.

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

### Outside the jammed phase, free-target fails on acceptance instead

> **`Worm_Length` is not a transport counter.**  `IntersectionWorm` uses the
> Prokof'ev--Svistunov open/close branch and tallies head--tail dwell on *every* iteration
> including rejected stay-puts (`worm.py:730`: *"We tally on EVERY step, including these
> stay-puts"*).  Its length can be large while the head never leaves the tail.  Only
> `FreeTargetWorm` auto-closes when head returns to tail, so for **it alone**
> `Worm_Length = 1` $\iff$ zero transport.  The two lengths are not comparable; transport
> must be read off the off-origin weight of the inline displacement histogram, which is
> the correlator itself.

`FreeTargetWorm` worm lengths (auto-close, so `len = 1` means the defect never moved):

| $\kappa$ | worms | median len | mean | **frac(len = 1)** | max |
|---|---|---|---|---|---|
| 0.01 | 60 | 1 | 1.00 | **1.0000** | 1 |
| 0.03 | 60 | 1 | 1.15 | 0.983 | 10 |
| 0.05 | 60 | 1 | 1.17 | 0.950 | 7 |
| 0.1 | 160 | 1 | 1.00 | **1.0000** | 1 |
| 0.5 | 160 | 1 | 1.00 | **1.0000** | 1 |

It essentially never transports, at any $\kappa$ — but for *different reasons* at the two
ends: starvation in the jammed phase (there is nothing to draw), acceptance outside it
(there is plenty to draw and none of it is affordable).

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

### The naive worm in the jammed phase (provisional)

The `IntersectionWorm` also jams at small $\kappa$ — established by the author from prior
experience, **not yet by a measurement in this document**.  It is taken as given for the
design; the docs must not present it as measured until it is.

The measurement that would settle it is running: the off-origin weight of the inline
displacement histogram (transport) at $\kappa = 0.01, 0.03, 0.1$, together with the worm's
own `tallies` — `unclean/drawn` is the constraint-violating-stencil rate the docs attribute
to warm backgrounds, `accepted/clean` the costly-rejection rate they attribute to cold
ones.  **Falsifier:** if off-origin weight at $\kappa = 0.01$ is comparable to that at
$\kappa = 0.1$, the naive worm does *not* jam in the jammed phase, and the hierarchy in
**Corrected narrative** step 1 needs rewriting.

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

**Half right, and the half that is right is the half that matters.**  In the jammed phase
($\kappa \lesssim 0.02$) the jam *is* structural: free-target's vastly enriched family
still leaves 48.8% of hypercubes with no legal mover, and a median clean set of one.  But
"does not help" overstates it — enrichment halves the dead fraction (97.7% → 48.8%), and
*outside* the jammed phase it eliminates starvation outright (median 1208 at
$\kappa = 0.1$, where the adaptive worm is dead 81% of the time).

The sharper statement, which the measurements support:

> Enriching the family buys real ground, and buys it where the worm was already losing.
> It does not buy enough.  In the jammed phase no local exact-repair family suffices, and
> outside the jammed phase the moves that exist are too expensive to accept.

The corrected hierarchy — each fix exposing the next failure:

1. **`IntersectionWorm`** — one fixed stencil, chosen before looking at $F$.  Rejected on
   cold backgrounds; constraint-violating on warm ones.
2. **`AdaptiveIntersectionWorm`** — look at $F$ before proposing.  Still starves: it
   commits to a *direction*, then asks which of a fixed library advances the head that
   way.  Dead in 81% of proposals at $\kappa = 0.1$, and 98% in the jammed phase.
3. **`TwoLinkAdaptiveWorm`** — enrich the two-link sector by live enumeration.  Real
   improvement (dead fraction 0.814 → 0.572 at $\kappa = 0.1$), still a zero median.  The
   direction is still drawn first.
4. **`FreeTargetWorm`** — drop the direction; let $\Delta q$ land where it wants.  Outside
   the jammed phase the starvation **vanishes** (median 1208 movers) and the worm fails on
   *acceptance* instead: its movers are 99.8% two-link exact repairs costing a median
   $\Delta S$ of 6 at $\kappa = 0.1$ and 21 at $\kappa = 0.5$.  *Inside* the jammed phase
   the starvation returns anyway (48.8% dead, median clean set 1).  It transports
   essentially never, at any $\kappa$.

This is what makes the `DefectGas` *necessary* rather than merely convenient.  The worm is
squeezed from both sides, and no amount of family engineering escapes the squeeze:

- **In the jammed phase**, a background with multi-unit flux everywhere admits no local
  exact-repair shape.  Enrichment halves the dead fraction and no more.
- **Outside it**, exact repair *forces expensive shapes* — the cheapest legal mover is a
  two-link move whose $\Delta S$ grows with $\kappa$ — and the excursion is all-or-nothing,
  so one unlucky draw ends it.

Both failures come from the same insistence: that the constraint be exactly repaired at
every single move.  The gas stops insisting.  It buys a *cheap single-link* move and pays
$\zeta^{\Delta D}$ for the mess, instead of demanding a mandatory two-link repair and
paying $e^{-21}$ for it.

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
ladder).  Only production `Hammer(S)` tunes; **test call sites pass `zeta = 0.025`**.  If a
test condenses the gas (the chain stops revisiting the vacuum sector, and `step` raises),
lower it for that test rather than globally.

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

**No backward compatibility.**  Existing h5 ensembles storing an inline
`Intersection_Intersection` are not supported after the promotion, by decision.  No read
shim, no deprecation alias.

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
- **Transport**, measured as the off-origin weight of the inline displacement histogram —
  *not* `Worm_Length`, which counts dwell including rejected stay-puts and is therefore
  not comparable between the auto-closing and open/close worms.
- `FreeTargetWorm` worm lengths (where auto-close does make `len = 1` mean zero transport)
  and the opening-step $P(\text{mover}) \times \mathbb{E}[\text{accept}]$ decomposition.

It must sweep $\kappa$ across the jammed boundary ($\kappa \approx 0.02$); a script that
only samples $\kappa \geq 0.1$ measures outside the phase of interest and will draw the
wrong conclusion, as this design's first draft did.

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

None blocking.  Resolved: no h5 backward compatibility (§3); tests use `zeta = 0.025` (§2);
`IntersectionTwoPoint` names the worms' inline histogram (§3); the `IntersectionWorm`'s
low-$\kappa$ jam is provisional pending the running transport measurement, with its
falsifier stated above.
