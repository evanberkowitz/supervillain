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

### `EnsembleStreamer`

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

Reusable beyond bootstrap: any memory-bounded pass over a big ensemble
(re-measuring an observable at N = 16 without OOM) is the same primitive.

*Test:* for every field and a scalar + a correlator observable, the
concatenation of `blocks()` reproduces the full-load Ensemble's values exactly.

### `StreamingBootstrap(Bootstrap)`

Subclasses `Bootstrap` to inherit `estimate`, `plot_band`, `plot_correlator`,
and every `DerivedQuantity` descriptor unchanged. **Write-through design**: a
target h5 group is bound at construction, and observable access *is*
persistence — no separate `to_h5(list)` step.

- **`StreamingBootstrap(streamer, target_group, rng=None)`** — `streamer` is an
  `EnsembleStreamer` (which owns `block`); `target_group` is where results
  land. Reads `len`, `weight`, `Action` from the streamer; builds `indices`
  `(configs × draws)` (via `rng` or the global RNG) and the count matrix `n`.
  Writes the metadata — `draws`, `indices`, `Action`, and the **`Ensemble`
  link** → the streamer's source group (`SoftLink` if same file, else
  `ExternalLink(source_filename, source_path)`) — into `target_group` at
  construction, so the target is a valid Bootstrap layout from the first
  observable on. (Also stores `block` as an attr so `from_h5` can rebuild the
  streamer faithfully.)
- **`_resample_streaming(name)`** — the streaming accumulation above, iterating
  `streamer.blocks()` and computing `getattr(sub, name)` per block. The
  memory-safe core.
- **`__getattr__(name)`** (primaries) — **disk-cache + write-through**: if
  `target_group[name]` already exists, load and return it (this is what makes a
  run **resumable/idempotent** — re-running skips completed observables); else
  `_resample_streaming(name)`, write the draws-first dataset into
  `target_group`, cache in `__dict__`, return.
- **`estimate(name)`** — computes `getattr(self, name)` (streaming primaries as
  needed; a `DerivedQuantity` derives per-draw on top of the streamed
  primaries), **persists `name`'s resampled array to `target_group` if not
  already there**, and returns `(mean, std)`. This is where *derived
  quantities* land on disk: DQ descriptors bypass `__getattr__`, so their
  scalar result is persisted here rather than by the primary hook. (A bare
  `sb.SomeDQ` attribute access persists its underlying primary but not the DQ
  scalar; go through `estimate` — the normal path — to persist the DQ. No
  `__getattribute__` magic unless we later decide bare-DQ persistence matters.)
- **No `to_h5(list)`** — the `target_group` *is* the incrementally-written
  Bootstrap layout; there is nothing to flush beyond the metadata already
  written at construction.
- **`from_h5(group)`** (classmethod) — reconstructs a `StreamingBootstrap` in
  the same state as a fresh one: reads the cached observable datasets +
  `indices`/`draws`/`Action`/`block` into `__dict__`, and **resolves the
  `Ensemble` link to an `EnsembleStreamer`** wrapping the linked source group
  (symmetric with construction). Cached observables `estimate()` from their
  datasets (streamer untouched); an un-streamed observable streams through the
  reconstructed streamer. A **broken link** (source file moved) → `streamer =
  None`: cached observables still work, and requesting an un-streamed one
  raises a clear "source ensemble unavailable" error.

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
- `from_h5` with a broken `Ensemble` link (source file moved) sets
  `streamer = None`: cached observables still `estimate()`, and requesting an
  un-streamed one raises a clear "source ensemble unavailable" error.

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
