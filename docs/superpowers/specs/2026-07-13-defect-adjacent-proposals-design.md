# Defect-adjacent proposal targeting with the full Metropolis--Hastings ratio

**Date:** 2026-07-13
**Status:** approved design, pre-implementation
**Branch:** feature/defect-adjacent (follows the pair-separation umbrella,
2026-07-13-pair-separation-umbrella-design.md, merged same day)

## Problem

The DefectGas proposes uniformly over the 4V links, so once a pair separates
the fraction of proposals that touch either defect is ~48/4V: the pair's
r-space motion is diffusive with a per-tick step probability that vanishes
with volume.  The umbrella evidence isolated this as the binding constraint:
at N=6 kappa=0.02 a 1/Theta-hat umbrella flattens the shell dwell (the
stationary distribution is right) but the chain completes 0 vacuum round
trips and production condenses, because the flat pair must DIFFUSE back to
adjacency to annihilate.  At N=12 the same dilution is the plain-scan
throughput ceiling (~1.5 configs/s at kappa=0.05).

## The proposal distribution

In sector k = D/2, with probability gamma_k draw the legacy uniform proposal
(link uniform over 4V, +-1 uniform); otherwise draw a defect cell c with
probability |q_c|/D (prefix walk over the tracked nonzero-cell list), one of
c's 32 edges uniformly, and +-1 uniform.  Charge weighting costs nothing
extra: the density's normalizer is D, already the tracked sector variable,
and doubled charges get attention proportional to the charge they must shed
(Evan, 2026-07-13).  In vacuum (D = 0) the adjacent branch is empty and the
distribution degenerates to pure uniform -- a one-line guard.

The proposal density at link l is

    p(l | n) = gamma_k / (4V)  +  (1 - gamma_k) * s(l, q) / (32 D),

    s(l, q) = sum over l's 8 containing hypercubes of |q_c|,

with the exact normalization identity  sum_l s(l, q) = 32 D  (each unit of
charge contributes its cell's 32 edges; in 4D each link lies in exactly 8
hypercubes and each hypercube has 32 edges: 4V * 8 = V * 32).

**gamma is a (K+1)-vector indexed by sector**, constant-filled when given as
a scalar (default 0.5).  The general signature is baked in from day one so a
tuned or D-dependent gamma is later a parameter choice, not a kernel change;
only the constant table is exercised initially, plus one exactness test at a
deliberately nonuniform vector to certify the density bookkeeping.

## Accept test

    u < exp(-dS) * (w[k']/w[k]) * (W'/W) * p(l | n') / p(l | n)

with the reverse density evaluated at the POST-move charges, sector D', and
gamma_{k'} -- all known pre-accept from the Delta-q stencil the umbrella
work already enumerates.  The +-1 factor is symmetric and cancels.

**What targeting buys -- stated honestly.**  For a reversible
creation/annihilation pair the Hastings factor exactly cancels the proposal
concentration: net creation and annihilation FLUXES are invariant, by
construction.  The win is in-sector TRANSPORT, where both endpoint states
carry the adjacency term and the ratio is s' D / (s D') = O(1): the ~V-fold
proposal concentration survives in full, and the pair's r-space diffusion
constant per tick rises by ~(1 - gamma) 4V / (32 <s>).  That is precisely
the kinetics gap the umbrella needs closed, and the reason the two features
are partners rather than alternatives.

## Kernel and RNG

gamma = 1 takes the CURRENT code path with the current pre-drawn batch
layout: bit-for-bit legacy preservation, the same discipline as the empty-w2
sentinel.  gamma < 1 pre-draws a wider per-tick batch (component u, cell u,
edge u, sign, accept u) consumed deterministically; `_tick` /
`step_reference` mirror the kernel exactly so the twin tests stay
bit-for-bit at every gamma.

## API

* `DefectGas(..., gamma=None)`: `None` means legacy (identically the
  gamma = 1 path).  A scalar broadcasts to the (K+1)-vector; a vector is
  validated (length K+1, entries in (0, 1]).  Requires a capped gas
  (`weights` or `D_max`) since the vector length is K+1.  Rides through
  ReadWriteable/h5 alongside `weights` and `w2`.
* `DefectGasWeightTuner(..., gamma=0.5)`: the SAME gamma is used in every
  probe and in the production gas `generator()` builds, so `mixing_sweeps`,
  the measured dwell, and `emit_every` stay self-consistent.  The umbrella
  second stage is unchanged (it inherits the probe gas).

## Validation

1. gamma = None/1: kernel and reference bit-for-bit with the current
   sampler; the existing test suite passes untouched.
2. Kernel vs reference bit-for-bit twins at gamma = 0.5 and at a nonuniform
   gamma-vector, on a table that visits the quartic sector.
3. Density unit test: sum_l p(l | n) = 1 exactly on constructed states
   (vacuum, one pair, doubled charge, quartic class), via the
   normalization identity.
4. gamma-independence (statistical): Theta AND the Binder cumulant agree
   between gamma = 1 and gamma = 0.5 runs at a healthy coupling.
5. Evidence gate A -- umbrella revival: N=6 kappa=0.02 with the 1/Theta-hat
   warm-started umbrella and targeting on: vacuum round trips revive,
   production survives, far Theta bins populate (all three failed at every
   umbrella cap without targeting).
6. Evidence gate B -- throughput: N=12 kappa=0.05 matched wall clock vs the
   campaign point: configs/hour and far-bin reach (max pair r^2, error at
   fixed dx).

## Evidence (2026-07-14, post-implementation)

All runs N=6, D_max=8, gamma=0.5, against the corresponding
weighted-campaign-2026-07-13 points (uniform proposals, same seeds/grids).

* **Gate A (umbrella revival), kinetics confirmed / co-tuning still open.**
  At kappa=0.02 with the cap-30 1/Theta-hat umbrella -- the configuration
  whose shell dwell flattened but whose round trips were exactly 0 without
  targeting -- the gamma-targeted tuner's probes complete 90 round trips at
  the default budget and 259 at 3x the budget: the return-to-adjacency
  kinetics the umbrella lacked are demonstrably supplied, scaling with
  horizon.  Production nevertheless still condenses, and the diagnosis
  moved: (i) a cap-30 table cannot satisfy the flat-dwell convergence
  criterion at a 4-decade coupling (2.5 decades of decay remain by
  construction, so the tuner correctly refuses to converge and the
  health-tier freeze keeps the warm start); (ii) the full-cap (1e5) table
  makes the stage-1 w recursion itself non-stationary (probes visit 3
  shells); (iii) fundamentally, stage 1 tunes w WITHOUT the umbrella that
  stage 2 then imposes, so the frozen pair is mutually inconsistent at
  strong lift.  The umbrella in production awaits alternating (w, w2)
  co-tuning -- this spec's "optionally alternate stages once" was an
  underestimate.  Recorded as follow-on work; the narrative docs' "off or
  gently capped" production guidance stands.

* **Gate B (transport and throughput): decisively passed, umbrella not
  required.**  The gamma=0.5 campaign rerun (weighted-campaign-2026-07-14)
  over the identical grids/seeds: at N=6 kappa=0.005 (the LEAST favorable
  coupling for targeting), 2.6x the round trips in half the ticks (~5x
  per-tick transport) and a ~7x smaller chi error at matched
  configurations, with chi agreeing (1.0104(1) vs 1.0102(7) -- the
  gamma-independence exactness test in production).  All three N=6
  condensation holes of the 07-13 campaign (kappa = 0.01, 0.04, 0.05)
  completed on the first pass, and the far Theta bins that motivated
  everything are now MEASURED from plain (w, gamma) runs -- no umbrella --
  e.g. kappa=0.03: Theta(3,0,0,0) = 1.02(7)e-4 and Theta(antipode) =
  2.96(57)e-5 with the transport ceiling at the lattice maximum r^2 = 36;
  kappa=0.06 (the chi peak): Theta(antipode) = 8.2(9)e-4 at matched
  configurations (a 9-sigma far-bin measurement).  N=8/N=12 tier timings
  land with the overnight campaign and extend this table.

## Non-goals

A gamma-tuning stage (bolt on later if fixed 0.5 underperforms; the vector
plumbing already admits it).  Nontrivial D-dependent gamma values (the
plumbing is in; choosing values is future work).  Targeting for other
generators.  Any change to estimators, weights, or the umbrella -- the
Hastings factor is the entire footprint of this feature in the accept test.
