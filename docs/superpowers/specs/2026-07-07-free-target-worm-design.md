# FreeTargetWorm — design

## Problem

`AdaptiveIntersectionWorm` / `TwoLinkAdaptiveWorm` draw a menu slot **first** (one of 8
signed orthogonal directions, or idle) and then keep only the templates whose $\Delta q$
is the dipole to that pre-committed target. A template whose $\Delta q$ is a perfectly
clean dipole to some *other* site — the direction not drawn, a diagonal, distance 2 — is
discarded. At small $\kappa$ (vortex proliferation) this jams: flux is everywhere, the
clean set for the drawn direction is usually empty, and the head freezes. The jam is
**constraint-dominated** (empty clean sets), not action-dominated ($\Delta S$ rejection is
the large-$\kappa$ problem), so enlarging what counts as a proposable move is the right
attack. Physics goal: resolve whether `Intersection_Intersection` develops long-range
order at small $\kappa$, as the mixed anomaly between the Villain $U(1)$ and the shift
symmetry of the $dn \wedge dn$ Lagrange multiplier suggests it must when `Spin_Spin`
disorders.

## Core idea — let the charge transport go where it wants

Drop the direction pre-commitment. One enumeration per iteration:

1. Generate every candidate shape anchored at the head (raw family, background-independent).
2. Compute each candidate's $\Delta q$ on the current $F = dn$ and **classify**:
   - $\Delta q \equiv 0$ → **idle**;
   - $\Delta q = \{h{:}\,{-1},\ y{:}\,{+1}\}$ for any $y$ → **mover to $y$** (the head
     walks to wherever the template transports the charge — orthogonal, diagonal,
     distance-2, …);
   - anything else → discard.
3. Flat draw over the deduped clean union $C$; tentatively apply; enumerate $C'$ at the
   new head on the arrived background; accept with
   $\min\!\big(1,\ (\left|C\right|/\left|C'\right|)\,e^{-\Delta S}\big)$.

This is ordinary state-dependent MH, same status as the existing adaptive worms — and
with **less** bookkeeping: one slot, so the fact-(2) cross-slot-uniqueness argument
becomes trivial ($\Delta n$ uniquely determines both $s'$ and the new head, via $\Delta q$).

## Correctness — the closure property

The load-bearing requirement, stated so the involution test can assert it:

> **Closure.** If the raw enumerator at head $h$ generates $\Delta n$ and
> $\Delta q(\Delta n \text{ on } F) = \{h{:}\,{-1},\ y{:}\,{+1}\}$, then the raw
> enumerator at head $y$ generates $-\Delta n$.

Given closure, safety composes from known facts: the reversal identity
$\Delta q_{F+d\Delta n}[-\Delta n] = -\Delta q_F[\Delta n]$ guarantees $-\Delta n$ is
*clean* on the arrived background (cleanliness is physics, free); closure guarantees it
is *generated* at $y$ (enumerability is a property of our family rule, **not** free); so
it survives the $\Delta q$ filter and lands in $C'$ — giving both the correct
$1/\left|C'\right|$ reverse-proposal probability and the $\left|C'\right| \geq 1$
no-division guarantee. Without closure the chain can propose $s \to s'$ but never
$s' \to s$: detailed balance breaks while every constraint-validity test stays green.

**Construction that gives closure by fiat:** anchor each shape to every cell in its
**charge-reach support** — $G(h) = \{\Delta n : h \in \mathrm{support}(\Delta n)\}$,
where the support is the background-independent union of the links' single-link reaches
(`_link_reach`) **plus the two-link self-charge (cross-term) support** $c_1 c_2 M_{12}$,
which can be nonzero where neither link's linear reach is. Then $\Delta q \neq 0$ at $y$
forces $y \in \mathrm{support}(\Delta n) = \mathrm{support}(-\Delta n)$, so
$-\Delta n \in G(y)$ — no box-geometry argument to get wrong. Idles need the same closure
at *fixed* head (negation-closed family there); the sign-symmetric `COEFF_BOX` already
provides it.

The involution test stays regardless: enumerators are code, and a later "optimization" to
the dedup or the support computation is exactly the edit that would break closure
silently.

## Closure of the worm — auto-close, no pivot bookkeeping

Adopt auto-close (the original worm-paper convention):

- The worm ends the **instant** the head returns to the tail. No close-vs-move coin flip.
- The emitted histogram is **pre-populated with 1 at the origin** — the honest tally of
  the iteration the worm spends at the pivot when it opens, not injected data.
- A worm whose *opening* proposal rejects (or whose opening clean set is empty) is a
  legitimate **zero-length worm**: emit the unchanged config with origin count 1. Keep it
  (it is pivot dwell time), do not retry.

Why exact (the derivation to be reproduced in the code's comment block with fact-(1)–(3)
rigor): the chain lives on $(n, \mathrm{head}, \mathrm{tail})$ with **plain weight
$w(n)$ for every state** — no pivot excess weight, because no proposal probability is
diverted to a close branch. Closing + reopening at a uniformly-random tail is a
tail-relabel on the pivot class $\{(n, x, x) : x\}$; $w(n)$ does not care where a
coincident head/tail sits, so the relabel is a free symmetry move. Quotienting by it, the
$1/V$ open factor cancels the pivot class's $V$-fold multiplicity and the balance
condition collapses to the ordinary uncorrected MH condition
$w(n)\,\frac{1}{\left|C\right|}A = w(n')\,\frac{1}{\left|C'\right|}A'$. The whole
fact-(3) menu/(menu+1) tally-ordering machinery disappears.

Estimator consequences: the origin bin is $\geq 1$ per worm by construction (retiring the
per-worm-origin-can-be-0 wart noted in `test_runs_in_ensemble_and_normalizes`);
`Worm_Length` stays "sum of displacement tallies". $G(0)$ is a UV contact term whose only
job is normalization (as in `Spin_Spin_Normalized`); the physics question — long-range
plateau vs decay of $G(r)/G(0)$ — lives at large $\left|r\right|$ and is
normalization-independent.

## Decisions

1. **New opt-in sibling class `FreeTargetWorm`** alongside `TwoLinkAdaptiveWorm`
   ($D = 4$, orthogonal-lattice two-link machinery reused, **not** in the default
   `Hammer`). House pattern: readable global-recompute `step_reference` oracle + compiled
   `step`, validated bit-for-bit on a shared seed.
2. **Raw family** = the same ingredients as `TwoLinkAdaptiveWorm` — single links and
   `COEFF_BOX` link pairs — anchored by charge-reach support (closure by construction).
   Strict superset of today's per-direction families, so mobility can only go up.
3. **Flat draw with class weights carried in the derivation.** The class of a move
   (idle vs mover) is a deterministic function of $(F, \Delta n)$, and the reverse has
   the *same* class (reversal identity), so per-class draw weights give the Hastings
   factor $W/W'$ with $W = w_{\mathrm{idle}}\left|I\right| + w_{\mathrm{move}}\left|M\right|$
   ($I$ = clean idles, $M$ = clean movers, $C = M \cup I$) — counts only, no extra
   $\Delta S$ work. **Ship with all weights $= 1$** (pure flat
   draw, $W/W' = \left|C\right|/\left|C'\right|$); the knob exists in the
   derivation so turning it later is a one-liner plus a detailed-balance rerun. Idle
   swamping of head transport on hot backgrounds is the risk the knob hedges; measure
   before turning.
4. **No backtracking suppression.** Alet–Sørensen-style directed-worm machinery is
   explicitly out of scope (empirically distrusted: a $D = 2$ Villain/Worldline duality
   cross-check failed with it). The bounce move sits in $C'$ with its honest weight;
   worm idling is physics, embraced. Weighted-proposal MH ($w(\Delta n) \propto$ e.g.
   $e^{-\Delta S/2}$, Hastings factor $\frac{w(-\Delta n)/W'}{w(\Delta n)/W}$) is exact
   and history-free but attacks the large-$\kappa$ $\Delta S$ problem, not this one —
   shelved, separate effort if ever.
5. **Kernel returns a classification, not a boolean.** `two_link_kernel.clean_mask`
   currently scores a family against one fixed target; the new kernel returns, per shape,
   (invalid | idle | mover→$y$). Each link pair is scored **once** instead of appearing
   re-anchored in up to 8 direction families.

## Architecture

- `FreeTargetWorm(TwoLinkAdaptiveWorm)` (or a shared-base refactor if inheritance gets
  awkward — implementation's call, oracle/fast/bit-for-bit pattern is the invariant).
- `classified_set_reference(n_arr, q0, head) → (movers=[(change, y), …], idles=[change, …])`
  — global `charge` recompute per candidate; the readable oracle.
- `classified_set_local(F, head)` — local stencils on maintained $F$; compiled twin.
- `step_reference` / `step` via a `_run_worm_local`-style shared walk: open at a uniform
  tail, pre-populate origin, loop {enumerate, flat-draw, tentative apply, enumerate
  reverse at the realized new head, accept with $W/W'\,e^{-\Delta S}$, tally
  displacement}, return the instant head == tail.
- Reuse: `_link_reach`, `_shape_self_charge`/`COEFF_BOX` family builders,
  `_delta_S`, `local_charge` stencils, `charge`, inline observables
  (`Intersection_Intersection`, `Worm_Length`).

## Testing

- **Involution / closure**: for every kept mover over *all realized targets* $y$ (not
  just orthogonal), the negated change appears in the enumeration at $y$; same for idles
  at fixed head.
- **Elementary detailed balance** per move (prototype style, $\sim 10^{-12}$), movers and
  idles, on real Hammer-generated backgrounds.
- **No-regression superset**: at every (head, background) sampled, the merged clean set
  contains the union over all 8 directions of `TwoLinkAdaptiveWorm`'s clean sets and its
  idle set.
- **Bit-for-bit** `step` vs `step_reference` on a shared seed.
- **Constraint preservation**: every emitted config exactly valid ($q \equiv 0$); origin
  bin $\geq 1$ per worm.
- **Physics cross-checks**: `Intersection_Intersection` and standard observables agree
  with `TwoLinkAdaptiveWorm` at moderate $\kappa$ (same physics, different sampler);
  small-$\kappa$ validation against the soft-constraint conditioning reference — where
  this worm has to earn its keep.

## Deferred / open

- Class-weight tuning ($w_{\mathrm{idle}} \neq w_{\mathrm{move}}$) — knob exists, measure
  idle fraction before turning.
- Larger reach box (more distant targets) — pure superset, bolt on only if the merged
  worm still jams.
- Weighted-proposal MH for the large-$\kappa$ $\Delta S$ regime — separate effort.
- Accept/reject-reason counters (empty set vs Hastings rejection) as a cheap sanity
  readout during implementation — diagnostic, not gating.
- Absolute normalization of the correlator (inherited open item from the adaptive-worm
  spec; unchanged by this design).
