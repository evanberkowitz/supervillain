# Surface worm gas migration — design

2026-07-31. Migrate the exactness-gated `SurfaceWormGas` — the F-space
extended-ensemble sampler developed and validated in the `no-intersections`
notebook (nperp-surgery-2026-07-24 → umbrella-2026-07-30 →
j-vacuum-2026-07-31, audited in swg-audit-2026-07-31) — into the library on
`feature/surface-worm`. Strategy: **refactor while migrating** — design the
ideal library form; revalidation is statistical, not bit-for-bit.

## What the sampler is

A grand-canonical extended-ensemble sampler of the No-Intersections model's
F-space with both physical constraints relaxed and priced:

    pi_ext(F) ∝ exp[−2π²κ C(F)] · w(D(F)) · intersectionFugacity^Q(F)
               · Z_wind(F) · w₂(pair separation)

Moves: uniform/defect-targeted plaquette toggles (Metropolis, manifest
detailed balance) interleaved with an exact coboundary heatbath. Physical
configurations are emitted from the **legal vacuum** — D = 0, Q = 0, and all
six H² periods zero (closed AND exact; the exactness condition is the
swg-audit-2026-07-31 fix, without which a worm that recloses around a
non-contractible 2-cycle emits a "vacuum" no (n, φ) represents) — by
reconstructing n from F (integer staircase primitive + winding-coset
resample) and drawing φ from its exact Gaussian conditional.

## Scope

**In:** the full instrument — gas + numba kernel, FState, staircase,
reconstruction, `SectorWeights`, `PairUmbrella`, `CorrelatorAccumulator`,
and both tuners (`SectorWeightTuner`, `PairUmbrellaTuner`).

**Out (stays notebook-side):** production/campaign drivers, tuning campaign
scripts, all h5 data, the frozen per-experiment toolchain copies (historical
record), and the transport tuner another session is testing — it migrates
separately when it has evidence, as a third class in `tuners.py`.

## Layout

`supervillain/generator/no_intersection/surface_worm/` — one module per
concern:

| module | contents |
| --- | --- |
| `state.py` | `FState`: F plus derived `dF, q, G, D, Q, winding, periods` and charge bookkeeping, maintained incrementally; `legal_vacuum` property (D=0 ∧ Q=0 ∧ periods=0); `recompute()` for gates; `intersection_winding()` — direct-from-F J via δG∧F periods (audit finding 8), with slice-spread and integrality alarms, so J-mobility is measurable on the chain without emitting. Constructible from an F array or from an (n, φ) configuration via F = dn (warm starts from stored rows). |
| `weights.py` | `SectorWeights` (w(D) table; `fugacity()` constructor for the linear special case, cap + hard wall) and `PairUmbrella` (w₂ over pair separation; `off()` identity constructor). Both `ReadWriteable` so tuned tables persist inside h5 ensembles — the side-car npz pattern retires. |
| `staircase.py` | Integer primitives of exact forms (verbatim math). Candidate to graduate to `supervillain.lattice` later; not in this migration. |
| `reconstruct.py` | n-ification: **raises on any non-exact F** (dF ≠ 0 or nonzero H² periods); winding-coset resample of the harmonic sector from its exact conditional; exact Gaussian φ draw. |
| `kernel.py` | The numba batch kernel and its table builders (`worm_numba.py` + `stencils.py` merged), including incremental periods maintenance at both commit points. |
| `gas.py` | `SurfaceWormGas(ReadWriteable, Generator)`. **Gas and generator-adapter merge** (the DefectGas precedent): `step(configuration)` advances the internal chain (ticksPerStep × stride measured moves), waits for the legal vacuum (slow-emit warning, hard-wait failure), emits an (n, φ) record carrying the accumulator's harvest; `inline_observables` sizes storage from a fresh harvest. The adapter's cadence parameters (`ticksPerStep`, `stride`, `pCob`, `maxWaitTicks`, `hardWaitFactor`) move onto the gas constructor; the notebook's separate `SurfaceWormGasGenerator` adapter disappears. `measure=False` skips the accumulator (cheap mobility/tuning chains) and its inline columns. |
| `accumulator.py` | `CorrelatorAccumulator`: strided dwell-ratio Θ measurement; excludes closed-but-non-exact ticks from the closed-shell dwells and reports them as `NontrivialClassTicks`; divides the umbrella weight out per pair bin using the SAME `PairUmbrella` object the acceptance multiplies by. |
| `tuners.py` | `SectorWeightTuner` (w(D) flattening) and `PairUmbrellaTuner` (w₂ bisection; its N=4 oscillation — audit-era finding — documented as a known limitation in the docstring). Extensible for the transport tuner. |

**Public API** (re-exported from `no_intersection/__init__.py`, mirroring
`DefectGas, DefectGasFugacityTuner, DefectGasWeightTuner`):
`SurfaceWormGas`, `SectorWeightTuner`, `PairUmbrellaTuner`. Everything else
is importable from the subpackage but not advertised; promotion later is a
one-line change.

## Naming

Fugacities get descriptive names — variable names convey meaning, they are
not mathematical symbols:

- `eta_dF` → **`openSurfaceFugacity`** (price per open cell, dF ≠ 0)
- `eta_q` → **`intersectionFugacity`** (price per intersecting hypercube, q ≠ 0)

Constructor guardrail: exactly one of `openSurfaceFugacity` and
`sectorWeights` (a tuned table already carries the open-surface price).

## Behavioral decisions (all settled by swg-audit-2026-07-31)

1. **Exactness everywhere.** `FState.legal_vacuum` requires periods = 0;
   `reconstruct` raises on non-exact F; the accumulator excludes non-exact
   closed ticks. `staircase.primitive_2form` itself stays permissive — the
   sampler legitimately uses it as a linear map on open F for the
   winding-sensitivity table; the gate lives at the n-ification boundary.
2. **The winding tilt is always on; the `windingInSampler` option and the
   `logWeight_SurfaceWormGas` column are removed** (supersedes the deferred
   j-vacuum finding-4 fix by making the whole bug class unrepresentable).
   The winding factor Z_wind(F) — the difference between the gas's native
   F-marginal and the physical one — is folded into the chain
   unconditionally: emitted rows are physical raw, there is no importance
   weight, and nothing can be double-counted. The tilt costs O(1) per move
   (winding-sensitivity table) and the reweighted alternative is strictly
   worse (measured ESS 0.43). The untilted path survives ONLY as a
   test-only seam (the tilt coefficient set to zero through a private
   hook — the identical code route), so the tilt-vs-reweight equivalence
   check remains a permanent library test without a public knob.
3. Constructor guardrails carried over: the fugacity/table exclusivity,
   `targetFraction ∈ [0, 1)` with the ergodicity rationale, the
   accumulator sharing the umbrella object.

## Validation

**Library tests** (`test/test_surface_worm*.py`, fast):

- class-1 stress: a manufactured closed-non-exact "vacuum" refused at every
  layer (not `legal_vacuum`, emit raises, reconstruct raises, accumulator
  books `NontrivialClassTicks` and keeps closed-shell dwells clean), while
  a legal vacuum passes;
- incremental-vs-recompute invariants (periods, winding, D, Q, G) on both
  the python-reference and numba paths, through charged excursions;
- staircase gates (d(primitive) == F on exact forms, D = 2, 3, 4);
- reconstruct gates: roundtrip (d(n) = F, winding kept, q = 0), gauge
  independence, observable agreement vs a standard-stack ensemble;
- winding-coset emit oracle: the resampled harmonic sector's marginal
  matches the exact conditional (the independent-oracle construction from
  the external validation);
- tilt-vs-reweight equivalence: an untilted chain (test-only seam, tilt
  coefficient zeroed) reweighted by Z_wind agrees with the tilted chain on
  physical observables — the permanent form of the TILT-XOR-WEIGHT check;
- detailed balance: the move's acceptance exponent vs a globally recomputed
  Δlog pi_ext;
- direct-from-F J (FState.intersection_winding) == library
  `IntersectionWinding` of the reconstruction;
- tuner smoke tests.

**Statistical revalidation** (notebook-side, new dated experiment dir,
because refactoring forfeits bit-equivalence): library gas vs the frozen
swg-audit toolchain at N = 4, κ = 0.03 with the same tuned tables —
⟨J²⟩, transition rate, ActionDensity, WrappingSquared, Θ moments within
errors — plus the κ = 0.2 N = 4 Hammer-agreement point. Recorded in that
experiment's NOTES per lab rules.

## Documentation

Sphinx docstrings throughout; a narrative subpackage docstring on
`surface_worm/__init__.py` in the style of `no_intersection/__init__.py`
(the extended ensemble, what is priced, what emission means, the exactness
story, the two placements of Z_wind); a `changes.rst` entry.

## Error handling

- Non-exact F at n-ification: `ValueError` naming the class (never silent).
- Emit reached with the gate bypassed: `ValueError` (defense in depth).
- Pinned chain (legal vacuum unreachable): slow-emit warning at
  `maxWaitTicks`, `RuntimeError` at `hardWaitFactor ×` that — the
  generator's existing contract, preserved so `Ensemble.generate` callers
  can salvage completed rows.
- Coboundary window fails tolerance within cap: raise (existing behavior).
