# Pair-separation umbrella: w2(r) in the D=2 and D=4 sectors

**Date:** 2026-07-13
**Status:** approved design, pre-implementation
**Branch:** feature/r-umbrella (follows the weighted DefectGas,
2026-07-13-weighted-defect-gas-design.md, merged same day)

## Problem

The sector table w(D) flattens *sector* occupancies, but within the single-pair
sector visits to separation r arrive proportional to Theta(r) itself: tails in
decaying phases are exponentially starved, and even at the chi peak the far bins
carry the worst variance.  The four-defect sector inherits the same disease
twice over: entries arrive as (spread r1, newborn-compact r2) -- creation is
always local -- and without a weight gradient the second pair never spreads, so
the both-pairs-spread corner that dominates the ordered-phase Binder signal
(U -> 1/2 needs the disconnected-product regime) stays empty.

Measured class shares in the D=4 dwell (stored campaign ensembles, window
couplings): {+1,+1,-1,-1} >= 99.6%; all doubled-charge classes < 0.4% combined.

## The weight

One geometric factor multiplies the enlarged-ensemble weight,

    W(state) = 1                                   D = 0, D >= 6, any doubled class
             = w2(|r|)                             D = 2, single +-1 pair
             = w2(|x1-y1|) w2(|x2-y2|)
             + w2(|x1-y2|) w2(|x2-y1|)             D = 4, class {+1,+1,-1,-1}

with |.| the min-image separation (translation- and hypercubic-invariant) and
w2 a learned table over the distinct min-image r^2 shells.  The D=4 form is the
Wick-inspired symmetrization (Evan, 2026-07-13): manifestly bosonic (invariant
under head and tail relabelings), smooth (no matching discontinuities), and
cheap (two products).  It is NOT complete -- for Gaussian theta the four-point
is a six-pair product including same-charge suppression factors that no sum of
(+-) pair products reproduces -- but completeness is not required: any
deterministic W is exactness-preserving, incompleteness costs only flatness,
and the mismatch regions (all-close, like-charge encounters) are not starved.
If the coarse (|r1|, |r2|) occupancy histogram shows residual starvation, a
multiplicative correction can be learned by the same recursion later.

The empty-table sentinel convention extends: no w2 table means W = 1
identically and the accept expression is untouched (geometric and plain-w
paths stay bit-for-bit).

**Normalization:** after learning, rescale w2 so the total pair-sector weight
is unchanged (sum over shells of multiplicity x Theta-hat x w2 fixed).  This
decouples the knobs: w(D) owns sector traffic, w2 owns the within-sector
profile.  (The D=4 weight shifts quadratically under this rescaling; the
sector-total change is absorbed in the next w(D) recursion pass.)

## Accept test and kernel

The Metropolis factor becomes

    u < exp(-dS) * (w[k']/w[k]) * (W(state')/W(state))

`tick_batch` (and `_tick`, in lockstep for the bit-for-bit test) must therefore
evaluate the CANDIDATE state's geometry before accepting: the Delta-q stencil
already enumerates the touched cells, so the proposed defect multiset is known
pre-accept; when either endpoint state is a weighted case (D in {2,4}, all
unit charges) compute W from the tracked positions, else W = 1 shortcuts.
Positions ride in the existing nzc/q bookkeeping; min-image r^2 -> shell index
via a precomputed lookup (size N^2 + 1).  This positional plumbing is shared
infrastructure: the defect-adjacent-proposal work (separate, later spec)
reuses it.

## Tallies and estimators

* `Theta_Theta` bins are pure in r, so integer tallies survive: divide binwise
  by w2 at emission (alongside the existing 1/(V w1)).
* `H_four` is NOT pure in W (W varies within the class), so the kernel
  accumulates 1/W per tick -- H_four becomes float64 -- and
  `FourDefectDistribution` keeps its emission scaling and the Binder formula
  is untouched.
* `Vacuum_Ticks`, `SectorTicks`, `RoundTrips` unchanged (W = 1 at D = 0).
* w2-independence generalizes the w-independence self-test: two materially
  different w2 tables must agree on Theta AND on the Binder.

## Tuning

`DefectGasWeightTuner` grows a second stage: with w(D) frozen, learn w2 by the
same damped, clipped histogram recursion, driven by the r-resolved D=2 dwell
(`Theta_Theta` raw bins, already accumulated) over min-image shells; converge
on r-histogram flatness + stationarity + unchanged round-trip health.  Warm
start w2 = 1/Theta-hat from the unumbrella'd stage (or a stored campaign
correlator).  Optionally alternate stages once.

**Light-by-policy** (Evan, 2026-07-13): after the w(D) recursion freezes,
divide w[k] by lighten^k (default lighten ~ 1.5) so production sits safely on
the vacuum branch of the sector-coexistence point rather than AT it -- the
umbrella now carries the tail statistics that coexistence traffic used to.
This replaces reliance on condensation retries and protects any future
tempering ladder from fracture.

## Validation

1. Empty-w2 sentinel: geometric and plain-w paths bit-for-bit unchanged
   (existing tests pass untouched).
2. Kernel vs reference bit-for-bit with a nontrivial w2, including D=4 Wick
   weights and float H_four.
3. Staged switch: w2 active in D=2 with W = 1 forced in D=4 must reproduce a
   D=2-only umbrella run exactly; then enable the product.
4. w2-independence (statistical): Theta and Binder agree across two different
   frozen w2 at a healthy coupling.
5. Tuner second stage converges on a small lattice; frozen w2 flattens the
   production r-histogram within a factor ~3.

## Non-goals

Defect-adjacent proposals (separate spec; reuses this spec's positional
plumbing).  Kappa parallel tempering (branch feature/kappa-tempering; design
constraints recorded in the no-intersections workspace notes).  Doubled-class
weights (< 0.4% of dwell).  R-dependent (inter-pair) weights: entropy-flat
where it matters.  Learned residual corrections to the Wick form (future, by
recursion, if the coarse histogram demands).
