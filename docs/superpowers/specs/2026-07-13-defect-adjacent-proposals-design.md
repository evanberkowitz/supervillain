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

## Non-goals

A gamma-tuning stage (bolt on later if fixed 0.5 underperforms; the vector
plumbing already admits it).  Nontrivial D-dependent gamma values (the
plumbing is in; choosing values is future work).  Targeting for other
generators.  Any change to estimators, weights, or the umbrella -- the
Hastings factor is the entire footprint of this feature in the accept test.
