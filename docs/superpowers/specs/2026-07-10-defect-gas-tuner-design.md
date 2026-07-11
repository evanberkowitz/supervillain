# DefectGas Tuner extraction and Generator-route driver migration

**Date:** 2026-07-10
**Status:** approved design, pre-implementation

## Problem

`DefectGas` is three things wearing one class:

1. a **generator** (`step`/`step_reference` + inline observables) — the only role every
   other generator in the library has;
2. a **standalone chain-driver with its own measurement stack** (`run`, `blocks`,
   `close_block`, `correlator`, `binder`, the run-path half of `report`) — a parallel
   path to `Ensemble`/`Bootstrap` that no other generator carries;
3. a **tuner** (`tune`, `tune_edge` classmethods) that drives probe chains through (2),
   with a `SiteUpdate` hardwired into `_phi_sweep`.

The companion drivers in the separate `no-intersections` repository straddle (2) and
(3): `land_in_vacuum`, hand-rolled block-jackknife loops, and retry ladders are scar
tissue around the straddle.  The sharp edge is idiom-breaking: you cannot treat
`DefectGas` like any other generator without tripping over its second API surface.

## Design

Three crisply separated roles:

| Object | Role | Plane |
|---|---|---|
| `DefectGas` | plain generator: `step` + inline observables | data |
| `DefectGasFugacityTuner` (new) | runs probe experiments to pick ζ, assembles the production chain | control |
| `Hammer` | thin function: roster sugar (explicit ζ) or one-line delegation to the tuner | sugar |

A **Generator** steps configurations; a **Tuner** runs experiments to decide *which*
generator to build.  Generators ride into ensembles and h5 (`ReadWriteable`), so probe
machinery, ladders, and tuning rngs must not live inside them (cf. the generator-
stripping workarounds around issue #65 in `campaign.py`).

### 1. `DefectGas` becomes a plain generator

**Deleted:** `run`, `correlator`, `binder`, `blocks`, `_new_block`, `close_block`,
`tune`, `tune_edge`, `_phi_sweep`, the `update_phi` flag and `st.site_update` (the φ
machinery existed only for the standalone path — `step()` freezes φ by design), and
`report`'s run-path sector-dwell line.  The `SiteUpdate` import goes with them.

**Kept:** constructor signature (`S`, `fugacity`, `D_max`, `emit_every`,
`max_step_sweeps`, `rng`), `step`, `step_reference`, `inline_observables`, all chain
internals (`_init_state`, `_draw_batch`, `_tick`, `_kernel_ticks`, `_step_body`),
`D_trace`, `accepted`/`proposed`, and a trimmed `report()` (acceptance +
pairs-in-flight).

**Added:** one inline quantity, **`Ticks`** — the number of proposals the step
consumed, emitted from `_step_body` (so both `step` and `step_reference` produce it).
Per-step vacuum dwell is then `Vacuum_Ticks / Ticks`, measurable from ordinary emitted
steps.  This is the missing piece that lets tuning live outside the class; it is also a
useful production diagnostic (a live condensation early-warning).  It is
generator-specific bookkeeping, not physics: the ensemble carries it like the other
inline batches, but it gets **no `Observable` class** in
`supervillain/observable/intersection.py` and no docs autoclass entry; the tuner reads
it directly from its probe ensembles.

The `step()` horizon-exhaustion `RuntimeError` message re-points from `DefectGas.tune`
to the `DefectGasFugacityTuner`.  The class docstring drops the standalone-path
advertisement.

**Compatibility note:** `DefectGas` is `ReadWriteable`; the attribute removals change
its h5 footprint.  Freshly written ensembles are unaffected (generators are stripped
before writing in the campaign; the library writes whatever `__dict__` holds).  Confirm
during implementation that no test round-trips a saved pre-change `DefectGas`.

### 2. New `DefectGasFugacityTuner` class (in `defect_gas.py`, alongside the generator)

```python
t = DefectGasFugacityTuner(S, companions=None, D_max=8, rng=None)
# companions: ordered iterable of generators interleaved with the probe gas.
#             Default (SiteUpdate(S),) — matching today's tune() behavior.

fugacity             = t.tune(start='cold', ladder=..., target=0.15, ...)
fugacity, emit_every = t.tune_edge(start='cold', ladder=..., min_vacuum_ticks=500,
                                   ..., floor=None)
chain                = t.generator(start='cold', edge=False)
```

**Probe mechanics.**  Each ladder rung builds a throwaway probe chain —
`Sequentially((*companions, DefectGas(S, ζ, ...)))` — and drives it through the
ordinary `Ensemble.generate` route from `start` (any value `Ensemble.generate`
accepts), reading dwell as `Σ Vacuum_Ticks / Σ Ticks` from the probe ensemble's inline
data and treating the step `RuntimeError` (condensation) as rung rejection.  The
mechanics need only *roughly* match the current tuning mechanics — per-rung
equilibration before believing dwell, and the current default constants (ladders,
`target=0.15`, `min_vacuum_ticks=500`, `step_sweeps=25`) carry over; details (probe
`emit_every`, per-step sweep caps, exact probe lengths) are the implementation's to
pick, guided by the current classmethods.

Policy semantics are preserved: `tune` descends its ladder and keeps the first rung
whose dwell exceeds `target` (falling back to the smallest rung if none does, as
today); `tune_edge` ascends and keeps the largest measurable,
stationary rung, returning ζ with a matched `emit_every` (sized so one production step
costs about `step_sweeps` sweeps at the measured dwell).

**`generator()`** is the normal way to consume a tune: it probes from `start`, then
returns the production-ready `Sequentially((*companions, DefectGas(S, ζ, ...)))` with
ζ and `emit_every` **matched by construction** — the matched pair is only meaningful
together, and hand-threading it through a fresh `DefectGas` and `Sequentially` is the
boilerplate (and footgun) this method removes.  `edge=True` selects the `tune_edge`
policy.  The returned chain carries lightweight metadata (`fugacity`, `emit_every` as
plain floats/ints) for introspection — never a reference to the tuner itself.

Because every `step()` emission is a vacuum configuration, the Generator route cannot
hand the tuner an invalid start: `land_in_vacuum` and `defect_count` in the drivers
become deletable, not merely movable.

### 3. `Hammer` stays a thin function

`Hammer` is a library-wide idiom (`villain.Hammer`, `worldline.Hammer`) and its
dominant usage is the explicit-fugacity roster sugar (`Hammer(S, fugacity=0.025)` in
tests and examples), which involves no tuning at all.  It survives as:

```python
def Hammer(S, fugacity=None):
    if fugacity is None:
        return DefectGasFugacityTuner(S, companions=<roster minus gas>).generator()
    return Sequentially((*roster, DefectGas(S, fugacity)))           # roster sugar
```

Careful production code (the drivers) rightly bypasses it: thermalize → `DefectGasFugacityTuner(...)
.generator(start=hot)` → `Ensemble.generate`.  Docstring updated (the `fugacity=None`
branch now names the tuner).  `DefectGasFugacityTuner` is exported from
`supervillain.generator.no_intersection`.

### 4. Observable and docs

- `Ticks` is registered in `inline_observables` with `Batch(steps, shape=(),
  dtype=float)` but gets **no `Observable` class** and no docs autoclass entry (it is
  generator bookkeeping, not part of the physics package).
- `supervillain/no_intersection.rst`: autoclass entry for `DefectGasFugacityTuner`;
  prose references to `DefectGas.tune`/`tune_edge` and the standalone path re-pointed.

### 5. Kernel test rework (`test/test_defect_gas_kernel.py`)

The three `run()`-based twin tests (`test_run_matches_reference`, `_quartic_sector`,
`_uncapped`) re-express as `step()` vs `step_reference()` twins from identical seeds
and starts, comparing the emitted `n`, all inline observables (including the new
`Ticks`), `proposed`/`accepted`, and `D_trace` — preserving the quartic-sector and
uncapped coverage.  Any existing step-based or interleaving tests in the file are
retained.  This is the comparison the file was really making; `run()` was only its
vehicle.

### 6. Companion driver migrations (`no-intersections` repository)

All three drivers move to the single library idiom: `Ensemble.generate` →
autocorrelation cut → decorrelate → `Bootstrap` → observable classes
(`Intersection_Intersection`, `IntersectionSusceptibility`, `ThetaBinderCumulant`,
`Spin_Spin_Normalized`).

- **`defect_gas.py`** (ζ-scan CLI): production via `Ensemble.generate` with the
  `Sequentially` chain; Θ and the Binder from `Bootstrap` + observables, replacing the
  hand-rolled block-jackknife `correlator()`/`binder()`; `--tune` → the tuner.  The
  two-ζ exactness self-test survives (the mean is ζ-independent).  npz fields adapt:
  correlator/Binder values now bootstrap means/errors; block-tally fields (`H_four`,
  `H_Z`) become sums of the inline observables.
- **`campaign.py`**: thermalization from cold via `Ensemble.generate` with a
  conservative-ζ chain instead of `run(tally=False)`; `land_in_vacuum` and
  `defect_count` deleted; ζ from `DefectGasFugacityTuner(S, companions=<production
  companions>).tune(start=hot)`.  The mid-run condensation retry ladder **stays** (that is physics
  policy, not scar tissue); the retry candidates assemble chains with explicit
  fugacities, which needs no tuner.
- **`censoring_probe.py`**: same substitutions; the tuned-vs-edge protocol becomes two
  tuner policy calls; the recorded-condensation → fresh-chain-one-rung-lower
  semantics survive via a re-tune with a truncated ladder.

The two repositories commit separately: library changes in `library/`, driver
migrations in `no-intersections/`.

### 7. Verification

Targeted only (the NoIntersections suite is slow; no bare full-suite runs):

- the reworked `test_defect_gas_kernel.py`;
- a small new `DefectGasFugacityTuner` test at the kernel test's small volume (tune from cold, assert
  ζ in the ladder, assert `generator()` returns a chain whose `DefectGas` carries the
  matched pair; ensemble-generate a few configurations from it);
- the `Hammer(S, fugacity=...)` tests in `test_no_intersection_generators.py` that
  don't exercise tuning continue to pass by construction (roster unchanged);
- short smoke runs of each migrated driver at N=4 with tiny budgets.

## Non-goals

- No change to the enlarged-ensemble algorithm, the compiled kernel, proposal
  distribution, or any estimator's definition.
- The block-jackknife error path is **removed, not relocated**: `Bootstrap` is the
  library's error machinery and the drivers adopt it.
- No adaptive-during-warmup ζ (rejected: puts control flow inside the generator).
- `Hammer` does not become a class (rejected: the explicit-fugacity sugar and
  cross-action idiom are its real job).

## Baseline

The library working tree carries in-progress docstring revisions to
`defect_gas.py`, `no_intersection.rst`, and `observable/intersection.py`; this work
builds on top of them.
