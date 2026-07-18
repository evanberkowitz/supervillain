# StreamingBootstrap — design (2026-07-17)

## Motivation

`Bootstrap.__getattr__` computes an observable over the whole ensemble
(`getattr(self.Ensemble, name)`), then `_resample` forms `obs[self.indices]`,
which materializes a **(configs × draws × observable-shape)** tensor. For a
correlator on a large lattice this explodes: N = 16, `Spin_Spin`, 920
decorrelated configs, 100 draws, V = 65536 → 920·100·65536·8 ≈ **48 GB**, an
OOM. This blocks the θ-sector susceptibilities (`SpinSusceptibility`,
`IntersectionSusceptibility`, `DoubleIntersectionSusceptibility`) at N = 16,
even though the *specific heat* (a scalar) is fine.

The fix is to compute the bootstrap **without ever holding the whole ensemble
or the (configs × draws × shape) tensor** — stream the configs from disk in
blocks, and accumulate each bootstrap draw incrementally. The result is written
to h5 in the layout `Bootstrap.to_h5` produces, so it round-trips through the
existing `Bootstrap.from_h5` / `estimate()` machinery.

## The resample, rewritten for streaming

`Bootstrap._resample` computes, for draw `d`,

    result[d, …] = mean_c( w[idx[c,d]] · obs[idx[c,d], …] ) / mean_c( w[idx[c,d]] )

where `idx` is the `(configs × draws)` index matrix. Grouping the sum by unique
config `i` with a **count matrix** `n[i,d] = #{c : idx[c,d] == i}`:

    numerator[d, …]   = (1/configs) Σ_i n[i,d] · w[i] · obs[i, …]
    denominator[d]    = (1/configs) Σ_i n[i,d] · w[i]
    result[d, …]      = numerator[d, …] / denominator[d]

`n` is computed once from `idx` (`configs × draws` integers, ~1 MB). The
numerator is then accumulated block by block: for a block of configs
`[i0:i1]`, add `einsum('bd,b...->d...', n[i0:i1]·w[i0:i1], obs_block)`. Peak
memory is one block's observable + the `(draws × shape)` accumulator (~50 MB
for the N = 16 correlator) + `n` — never the 48 GB tensor. The result equals
`Bootstrap._resample`'s to floating-point (summation order differs, so not
bit-identical, but statistically identical and equal to ~1e-12 on a scalar).

## Why streaming primary observables suffices (and derived quantities come free)

Observables register as **descriptors on the `Ensemble` class**
(`Observable.__init_subclass__` → `setattr(Ensemble, name, cls())`), and
`Observable.__get__` computes a non-inline observable **per configuration**
(a comprehension over configs, looking up the measure function's arguments —
`phi`, `n`, or other observables — as ensemble attributes). Two consequences
that make streaming correct:

- A block's `from_configurations` sub-Ensemble is a real `Ensemble` instance,
  so it carries every observable descriptor; computing `sub.Spin_Spin` needs
  only the `Action` (carried by the sub-Ensemble) and the argument fields
  (`phi`/`n`, sliced into the block). **Primary observables not stored on disk
  compute fine from a block.**
- Because the computation is per-configuration, computing an observable
  block-by-block and concatenating equals computing it on the full ensemble.
  This is the correctness guarantee for streaming, and it holds for every
  primary observable and for chains (`IntersectionWindingSquared` →
  `IntersectionWinding` → `IntersectionCurrent` → `n`). Inline observables
  (`Theta_Theta`) come straight from the sliced fields.

**Derived quantities** (`SpinSusceptibility`, `IntersectionSusceptibility`,
`IntersectionBinderCumulant`, …) register as descriptors on **`Bootstrap`**
(`DerivedQuantity.__init_subclass__` → `setattr(Bootstrap, name, cls())`) and
`__get__` computes them **per draw** by looking up dependencies *on the
bootstrap* — e.g. `getattr(bootstrap, 'Spin_Spin')`, which triggers the
resample of the primary. So `StreamingBootstrap` (subclassing `Bootstrap`)
inherits every DQ descriptor unchanged: a DQ's `getattr(self, primary)` returns
the *streamed* `(draws × shape)` resample, and the DQ derives per draw on top.
**`StreamingBootstrap` therefore only needs to stream primary observables; the
derived layer runs for free.** Accessing/`estimate`-ing either a primary name
or a derived-quantity name works — requesting `SpinSusceptibility` streams
`Spin_Spin` underneath and persists the derived `(draws,)` result (see the
write-through interface below).

## Two units

The work splits along a clean seam: **reading a serialized Ensemble in blocks**
(fiddly, I/O-bound) versus **the resample math and h5 output** (purely
numerical). Two independently-testable classes.

### `EnsembleStreamer` (ReadWriteable)

Encapsulates memory-bounded iteration of an h5-serialized Ensemble.

- **`EnsembleStreamer(source_group, block=64)`** — `source_group` is an h5py
  group holding an `Ensemble.to_h5`. Reads only cheap metadata eagerly: the
  configuration **count** (a dataset shape), the per-config `weight` (scalars),
  and `Action`. Does **not** load the configuration fields.
- **`__len__`** → number of configs.
- **`.weight`, `.Action`** → the cheap metadata.
- **`.blocks()`** → generator yielding `(start, sub_ensemble)` where
  `sub_ensemble` is an in-memory `Ensemble` of ≤ `block` configs, built by
  slicing each configuration field dataset `[start:start+block]` into a
  `Configurations({field: slice})` and wrapping with
  `Ensemble(Action).from_configurations(...)`. Consumers compute whatever
  observable they need on each `sub_ensemble` and discard it.
- **`to_h5` / `from_h5`** (custom, since `source_group` is a live h5 handle,
  not data): `to_h5` writes an h5 **link** to `source_group` (`SoftLink` if
  same file, else `ExternalLink(source_filename, source_path)`) plus `block`
  as an attr; `from_h5` resolves the link back to a group handle and
  reconstructs the streamer. Being `ReadWriteable` is what lets
  `StreamingBootstrap`'s inherited `from_h5` reconstruct its `streamer` field
  **automatically**.

Reusable beyond bootstrap: any memory-bounded pass over a big ensemble
(re-measuring an observable at N = 16 without OOM) is the same primitive.

*Test:* for every field and a scalar + a correlator observable, the
concatenation of `blocks()` reproduces the full-load Ensemble's values exactly;
`to_h5`/`from_h5` round-trips (link resolves to the same source).

### `StreamingBootstrap(Bootstrap)`

Subclasses `Bootstrap` to inherit `estimate`, `plot_band`, `plot_correlator`,
and every `DerivedQuantity` descriptor unchanged. **Write-through design**: a
target h5 group is bound at construction, and observable access *is*
persistence — no separate `to_h5(list)` step.

- **`StreamingBootstrap(streamer, target_group, rng=None)`** — `streamer` is an
  `EnsembleStreamer` (which owns `block`); `target_group` is where results
  land. Reads `len`, `weight`, `Action` from the streamer; builds `indices`
  `(configs × draws)` (via `rng` or the global RNG) and the count matrix `n`.
  Writes the metadata — the `streamer` (serialized as its source link),
  `draws`, `indices`, `Action` — into `target_group` at construction (via the
  same `Data.write` field convention `ReadWriteable.to_h5` uses), so the target
  is a valid, `from_h5`-readable Bootstrap layout from the first observable on.
  `target_group` itself is a live handle, not a serialized field.
- **`_resample_streaming(name)`** — the streaming accumulation above, iterating
  `streamer.blocks()` and computing `getattr(sub, name)` per block. The
  memory-safe core, invoked by `__getattr__` for primaries.
- **`__getattribute__(name)`** — **the single disk-cache + write-through
  gate**, guarded to fire only on names in the Observable/DerivedQuantity
  registries (everything else — `estimate`, `target_group`, `streamer`,
  methods — falls straight through to normal lookup, no recursion):
  - if `target_group[name]` exists → load and return the dataset (**resumable**
    — re-running skips work already on disk);
  - else `super().__getattribute__(name)` computes it — a **primary** falls to
    `__getattr__` → `_resample_streaming`; a **derived quantity** runs its
    descriptor, which pulls the (streamed, hence written) primaries it depends
    on — then write the draws-first result into `target_group` and return it.

  So `sb.SpinSusceptibility` and `sb.Spin_Spin` behave identically: check disk,
  build if absent, persist, return. Computing a DQ therefore also persists its
  underlying primary (the expensive `(draws×V)` array), so any later DQ built
  on it loads from disk. `estimate(name)` is then just
  `getattr(self, name)` → `(mean, std)` — persistence already handled by the
  gate.
- **`__getattr__(name)`** — the streaming resample of a **primary** (called by
  `super().__getattribute__` when the name isn't a class attribute); no
  persistence logic here (that lives in `__getattribute__`).
- **No `to_h5(list)`** — the `target_group` *is* the incrementally-written
  Bootstrap layout; nothing to flush beyond the construction-time metadata.
- **`from_h5(group)`** — the **inherited** `ReadWriteable.from_h5` does the
  work: it reconstructs each field, including `streamer` (via
  `EnsembleStreamer.from_h5`, which resolves the source link) and the cached
  observable datasets. `StreamingBootstrap` adds only a one-line touch to set
  `target_group = group` on the result so continued access keeps writing
  through. Cached observables load from their datasets; an un-streamed one
  streams through the reconstructed streamer and persists. A **broken link**
  (source moved) surfaces when the streamer is first used — cached observables
  still work; an un-streamed request raises a clear "source unavailable" error.

**Compatibility note.** A plain `Bootstrap.from_h5(group)` also reads the
output: `estimate()` on a *streamed* observable works (the dataset is cached in
`__dict__`, so `__getattr__` never fires); an *un-streamed* observable would
eagerly follow the link, load the ensemble, and OOM on the in-memory
`_resample` — i.e. exactly the behaviour `StreamingBootstrap` exists to
replace. Streamed observables are portable to plain `Bootstrap`; the streaming
benefit requires `StreamingBootstrap.from_h5`.

## Data flow

    source h5 (Ensemble)  ──EnsembleStreamer.blocks()──▶  small Ensembles
            │                                                    │
            │                                        getattr(sub, observable)
            ▼                                                    ▼
     weight, Action, len ──▶ indices, count  ──accumulate──▶ (draws×shape) result
                                                                 │
                                          write dataset (write-through) ▼
                                     target h5 (Bootstrap layout, incremental)

## Error handling

- Accessing an observable that fails on a block (a name the Ensemble can't
  compute) propagates the underlying error at the first block, not silently —
  no partial dataset is written for that observable (write the target dataset
  only after the full streamed result is in hand).
- Mismatched shapes across blocks (should never happen) raise loudly.
- Re-access of an on-disk observable loads the dataset; it is not recomputed
  (resumability).
- `from_h5` with a broken source link (source file moved): `EnsembleStreamer`
  reconstructs but its source is unresolvable; cached observables still
  `estimate()` from their datasets, and only a request that must call
  `streamer.blocks()` (an un-streamed observable) raises a clear "source
  ensemble unavailable" error.

## Testing

1. **Equivalence (the load-bearing test).** On a small stored ensemble, a
   `StreamingBootstrap` and a `Bootstrap` built with the *same* `indices`
   agree in `estimate()` to ~1e-12 for a scalar (`ActionDensity`), a correlator
   (`Spin_Spin`), and a derived quantity (`SpinSusceptibility`).
2. **Streamer fidelity.** `EnsembleStreamer.blocks()` concatenated equals the
   full-load Ensemble, field by field and for one computed observable.
3. **Write-through round-trip.** After accessing/`estimate`-ing a few
   observables, `StreamingBootstrap.from_h5(target_group)` reconstructs a
   streamer from the `Ensemble` link; cached observables `estimate()` match the
   pre-write values, and an *un-streamed* observable requested after `from_h5`
   streams through the reconstructed streamer and matches a direct `Bootstrap`.
4. **Resumability.** Re-accessing an on-disk observable returns the stored
   dataset without recomputing (assert via a call counter or timing).
5. **Portability.** `Bootstrap.from_h5` of the target estimates the streamed
   observables correctly.
6. **Memory.** Peak RSS over a streamed correlator scales with `block`, not
   `configs` (a coarse assertion / manual check on the N = 16 case).

## Out of scope (YAGNI)

- No re-implementation of derived quantities; `StreamingBootstrap` computes the
  same primary observables `Bootstrap` does, then the existing
  `DerivedQuantity` machinery consumes them unchanged.
- No parallelism across blocks (streaming is I/O-bound; add later if needed).
- No change to `Bootstrap` itself.
