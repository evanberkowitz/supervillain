# Weighted DefectGas: multicanonical sector table and histogram tuner

**Date:** 2026-07-13
**Status:** approved design, pre-implementation

## Problem

`DefectGas` prices the enlarged ensemble with a single geometric knob,
$\zeta^{D}$.  The restricted partition functions grow like
$Z_k/Z_0 \approx \lambda^k/k!$ in the pair count $k = D/2$ (dilute pairs, one
factor $\lambda = V\chi_\theta$ per pair), so the sector-occupancy ratio
$Z_{k+1}/Z_k = \lambda/(k+1)$ **decreases** with $k$: the entropy is
log-concave, and a geometric weight can balance exactly one adjacent pair of
sectors.  Consequences, in decreasing order of pain:

1. **Quadratic mistuning cost.**  Tail bins of `Theta_Theta` fill at a rate
   $\propto \zeta^2$ (excursion launches) while each pair tick is weighted
   $1/\zeta^2$; the compute needed for fixed tail error scales as
   $(\zeta^*/\zeta)^2$.  The ladder tuner's 15% dwell target plus its coarse
   rungs (adjacent $\zeta$ ratios 2–2.5, hence Poisson-mean jumps of 4–6×)
   routinely parks $\zeta$ an order of magnitude under the edge — a ~100×
   compute penalty paid silently in empty far bins.
2. **Multi-pair sectors are exponentially priced.**  Mean $m$ pairs in flight
   costs vacuum dwell $e^{-m}$ under any geometric weight.  The multi-pair
   sectors are the partner-swap/relay channel (a long-lived excursion meets
   $O(k)$ foreign partners per lifetime, volume-independently, by 4D
   transience) and the corridors through jammed backgrounds.
3. **Fragile tuning.**  Ladder probes are vacuum-tick-budgeted: an unhealthy
   rung dies by `RuntimeError` instead of producing a measurement, so the
   tuner must search blind around a cliff.  Worse, short vacuum-anchored
   probes are fooled by the **metastable vacuum**: observed 2026-07-13 at
   $N=6$, $\kappa=0.05$, `tune()` accepted $\zeta=0.02$ (its 120-step probe
   dwelled happily) and production then condensed irrecoverably after ~1360
   steps; `tune_edge()` blessed $\zeta=0.08$ with a measured dwell of ~1.0
   because its probe steps are only ~5 vacuum ticks each — 200 of them never
   carried the chain away from its vacuum-rich start.  Nucleation out of the
   metastable vacuum is slow precisely where the entropy pressure is highest,
   so probe-length honesty, not just target choice, is a failure axis.

**Evidence (probe study, 2026-07-13, `weighted_gas_probes.py` in the
no-intersections workspace).**  A fugacity grid at $N=6$, $\kappa=0.05$,
`D_max=8`, 800 configurations per rung with a 2000-sweep horizon: every
surviving rung ($\zeta \le 0.01$) tallied **exactly zero** in every bin with
$r^2 \gtrsim 5$ over $\sim 4\times10^6$ ticks each (98–100% of steps empty
beyond the half-lattice; the correlator dies at $r^2 \approx 4$–$5$), while
$\zeta \ge 0.02$ condensed out of the metastable vacuum after 1–2×10³
sweeps.  **The geometric weight has no working window at this volume and
coupling**: everything that survives never reaches the tail, everything that
would reach the tail condenses.  The tuner picks were also run-to-run
unstable (`tune()`: 0.02 then 0.01; `tune_edge()`: 0.08 then 0.03 across two
attempts differing only in consumed rng draws).  An `emit_every` scan at
$\zeta=0.01$ (×1/×4/×16 at matched total ticks) moved the empty-far-bin step
fraction (98.75% → 92%) but not the information content (far bins zero
throughout), confirming that `emit_every` repackages statistics and cannot
create them.

The fix is standard multicanonical machinery specialized to a tiny sector
space: replace $\zeta^{D}$ by a learned table $w[k]$, $k = 0 \ldots K$, chosen
so the sector-tick histogram is flat.  Flat occupancy pays $1/(K{+}1)$ vacuum
dwell (linear, not exponential), cannot condense by construction, and the
estimator is $w$-independent — tuning moves only variance, never the answer.

## Design

Three pieces: the generator generalization, the kernel plumbing, and a new
tuner.  `DefectGasFugacityTuner` stays as-is (the cheap option for the
geometric path).

### 1. `DefectGas` accepts a sector table

```python
DefectGas(S, fugacity, D_max=None, ...)           # unchanged, geometric
DefectGas(S, weights=w, ...)                      # new: w[k] for k = 0..K
```

- `fugacity` and `weights` are mutually exclusive; exactly one is required.
- `weights` is a positive 1D array, stored normalized to `w[0] == 1`.  Its
  length pins the cap: `D_max = 2*(len(w)-1)`.  Passing both `weights` and a
  disagreeing `D_max` is a `ValueError`; `D_max=None` (uncapped) is only
  meaningful on the geometric path, which keeps it.
- Sector parity is safe: $|q| \equiv q \pmod 2$ per cell and $\sum q = 0$
  identically, so $D$ and every $\Delta D$ are even and $k = D/2$ always
  indexes the table.
- The Metropolis factor becomes `w[k']/w[k]` on the table path and stays the
  literal `fugacity**ΔD` on the geometric path — **the geometric
  floating-point stream is preserved bit-for-bit**, so existing seeds, the
  kernel-vs-reference test, and archived chains reproduce exactly.  (The
  kernel takes both a scalar and a table; an empty table selects the scalar
  path.)
- Emission scalings unify through `w1 = fugacity**2` or `w[1]` and
  `w4 = fugacity**4` or `w[2]`: `Theta_Theta = H_pair/(V*w1)`,
  `FourDefectDistribution = H_four/w4`.  Vacuum ticks stay unscaled
  (`w[0] == 1` is the absolute anchor).
- `ReadWriteable`: the table rides in `__dict__` like every other attribute;
  `fugacity` is absent on the table path (and vice versa).  `__str__` reports
  whichever the instance carries.
- The `step()` horizon `RuntimeError` message gains the table case ("retune
  the weights" alongside "lower fugacity").

### 2. Kernel and instrumentation

`tick_batch` (and `_tick`/`step_reference`, kept in lockstep for the
bit-for-bit test) changes:

- **New arguments:** `w` (float64 array; empty selects the geometric scalar
  path) and `t_sector` (int64 array, length `D_max//2 + 1`), incremented at
  the current sector every tick.
- **Round trips:** a counter incremented at each vacuum tick that follows a
  visit to the top sector since the previous vacuum tick (rides in the
  existing `tstate`-style bookkeeping).

Two new per-step inline quantities, emitted whenever the chain is capped
(either path): **`SectorTicks`** (shape `(D_max//2 + 1,)`) and
**`RoundTrips`** (scalar).  Like `Ticks`, these are generator bookkeeping,
not physics: no `Observable` class, no docs autoclass entry.  `SectorTicks`
is the tuner's input and the production health check (a frozen table whose
histogram drifts from flat is the early warning); `RoundTrips` is the
mixing diagnostic that flatness alone cannot certify.

### 3. New `DefectGasWeightTuner` (alongside the generator)

```python
t = DefectGasWeightTuner(S, companions=None, D_max=16, rng=None)
w, emit_every = t.tune(start='cold', ...)
chain         = t.generator(start='cold', ...)    # production-ready chain
```

Same tuner-is-not-a-generator contract, `companions` default
`(SiteUpdate(S),)`, and `generator()` shape as `DefectGasFugacityTuner`.
Default `D_max=16` ($K=8$): deep enough for several pairs in flight (the
swap/relay regime saturates linearly in $k$), shallow enough that vacuum
keeps $\gtrsim 10\%$ of a flat histogram.

**Probes are sweep-budgeted, not vacuum-budgeted.**  A private
`DefectGas._probe_sweeps(configuration, sweeps)` advances the enlarged chain
a fixed number of sweeps — never waiting for vacuum, never emitting — and
returns the accumulated `SectorTicks` plus the raw (possibly invalid) state
for the next block; the tuner interleaves companions on a fixed sweep
cadence between blocks (φ changes trigger the existing state rebuild).  A
chain stuck at high $D$ therefore produces a *measurement* (a lopsided
histogram) instead of a `RuntimeError`: the tuner's condensation failure
mode does not exist.

**The recursion.**  Starting from the Poisson-envelope warm start
$w[k] = k!\,u^k$ with $u = 1/V$ by default (any start converges; a good one
saves iterations):

1. Probe `probe_sweeps` (default ~2000) sweeps; collect `t[k]`.
2. For **visited** sectors, $w[k] \mathrel{*}= (\bar t / t[k])^{\alpha}$ with
   $\bar t$ the mean visited count; renormalize `w[0] = 1`.  Damping
   $\alpha = 1$ on the first pass, $0.5$ after (short-probe noise must not
   drive oscillation).  Unvisited sectors are left alone — they become
   reachable as the sectors below them flatten; extrapolating boosts into
   unmeasured territory is how multicanonical recursions blow up.
3. Converged when every sector was visited, $\max_k t[k] / \min_k t[k] \le 3$,
   the probe completed ≥ 5 round trips, **and** the histogram is stationary
   (second-half sector counts within a factor ~2 of the first half) — the
   metastable-vacuum trap in the Problem section is detected by drift and
   round-trip starvation, never by flatness alone.  Iteration cap (default 12):
   on expiry, `logger.warning` and return the current table — production's
   `SectorTicks` histogram is the check that catches a bad freeze.
4. **Freeze.**  The returned table is fixed for production (detailed balance
   requires it); tuning tallies are discarded.
5. `emit_every` is matched from the final probe's measured vacuum fraction
   exactly as `tune_edge` does: one production step ≈ `step_sweeps`
   (default 25) sweeps.

### Explicit non-goals (recorded so they aren't relitigated)

- **Checkerboard/broadcast updates** and the delayed-acceptance block scheme
  that would reconcile them with a global weight.  The kernel is sequential;
  $q = dn \wedge dn$ is quadratic in $n$, so parallel flips need
  stencil-disjoint colorings — a separate project if profiling ever demands
  it.
- **Within-sector umbrella** $w_2(r)$ over the pair separation (translation
  invariant, but a kernel-invasive change).  Only worth revisiting if flat
  sectors still leave the near-$\kappa_c$ tails starved.
- **Site-dependent weights**: never — breaks translation invariance.
- $D_{\max} \sim V$: finite defect density abolishes vacuum returns and with
  them the absolute normalization.  The cap is a rail, not a physics knob.

## Validation

1. **Geometric equivalence (unit):** `DefectGas(S, fugacity=z, D_max=2K)`
   and the same call spelled `weights=[z**(2k)]` produce identical accept
   decisions on a shared proposal stream (table path vs scalar path differ
   only in fp association; assert agreement of the decisions, not the
   intermediate floats).
2. **Kernel vs reference (bit-for-bit):** extend
   `test_defect_gas_kernel.py` with a deliberately non-geometric table;
   `step` and `step_reference` must agree tick-for-tick, including
   `SectorTicks` and `RoundTrips`.
3. **$w$-independence (statistical):** two frozen, materially different
   tables on a small lattice give compatible `Theta_Theta` and Binder — the
   direct generalization of the existing $\zeta$-independence self-test.
4. **Tuner convergence (small, slow-marked):** on a small lattice the
   recursion reaches its convergence criteria within the iteration cap and
   the frozen table's production histogram stays flat within a factor ~3.

Consumer-side driver changes (flags to select the weight tuner, npz fields
for the table) belong to the project repositories, not this library change.
