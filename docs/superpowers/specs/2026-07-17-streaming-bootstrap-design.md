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

Subclasses `Bootstrap` to inherit `estimate`, `plot_band`, `plot_correlator`
unchanged (they route through `getattr(self, obs)`), overriding init, the
resample, `__getattr__`, `to_h5`, `from_h5`.

- **`StreamingBootstrap(source_group, draws=100, block=64, rng=None)`** — holds
  an `EnsembleStreamer(source_group, block)`; reads `len`, `weight`, `Action`
  from it; builds `indices` `(configs × draws)` exactly as `Bootstrap` (via
  `rng` or the global RNG) and the derived count matrix `n`.
- **`_resample_streaming(name)`** — the streaming accumulation above, iterating
  `streamer.blocks()` and computing `getattr(sub, name)` per block.
- **`__getattr__(name)`** — like `Bootstrap`'s but calls
  `_resample_streaming`; caches into `__dict__`. Interactive use and
  *un-streamed* observables both work (via streaming, no OOM).
- **`to_h5(output_group, observables=(...))`** — streams each requested
  observable (union with any already cached on the object) and writes its
  draws-first dataset into `output_group` in the same layout `Bootstrap.to_h5`
  produces, plus `draws`, `indices`, `Action`. In place of the Ensemble it
  writes an **h5 link** named `Ensemble` → the source group: `SoftLink(path)`
  if `output_group` is in the source group's file, else
  `ExternalLink(source_filename, source_path)`.
- **`from_h5(group)`** (classmethod) — reads the cached observable datasets +
  `indices`/`draws`/`Action` into `__dict__`, and resolves the `Ensemble` link
  to a **group handle held lazily** (kept for streaming, not deserialized into
  an in-memory Ensemble). `estimate()` on a streamed observable reads its
  cached dataset (ensemble untouched); an un-streamed observable streams from
  the linked group on demand.

**Compatibility note.** A plain `Bootstrap.from_h5(group)` also reads the
output: `estimate()` on a *streamed* observable works (the dataset is cached in
`__dict__`, so `__getattr__` never fires); an *un-streamed* observable would
eagerly follow the link, load the ensemble, and OOM on the in-memory
`_resample` — i.e. exactly the behaviour `StreamingBootstrap` exists to
replace. This is acceptable: streamed observables are portable to plain
`Bootstrap`; the streaming benefit requires `StreamingBootstrap.from_h5`.

## Data flow

    source h5 (Ensemble)  ──EnsembleStreamer.blocks()──▶  small Ensembles
            │                                                    │
            │                                        getattr(sub, observable)
            ▼                                                    ▼
     weight, Action, len ──▶ indices, count  ──accumulate──▶ (draws×shape) result
                                                                 │
                                              write dataset + link ▼
                                                          output h5 (Bootstrap layout)

## Error handling

- `to_h5` with an observable that fails on a block (e.g. a name the Ensemble
  can't compute) propagates the underlying error at the first block, not
  silently — no partial dataset is written for that observable.
- Mismatched shapes across blocks (should never happen) raise loudly.
- `from_h5` where the `Ensemble` link is broken (source file moved) succeeds
  for cached observables and raises only when an un-streamed observable is
  requested (the link is followed lazily).

## Testing

1. **Equivalence (the load-bearing test).** On a small stored ensemble, a
   `StreamingBootstrap` and a `Bootstrap` built with the *same* `indices`
   agree in `estimate()` to ~1e-12 for a scalar (`ActionDensity`) and a
   correlator (`Spin_Spin`).
2. **Streamer fidelity.** `EnsembleStreamer.blocks()` concatenated equals the
   full-load Ensemble, field by field and for one computed observable.
3. **Round-trip.** `to_h5` → `StreamingBootstrap.from_h5` → `estimate()`
   matches the pre-write values; the `Ensemble` link resolves; an un-streamed
   observable computed after `from_h5` matches a direct `Bootstrap`.
4. **Portability.** `Bootstrap.from_h5` of the streamed output estimates the
   streamed observables correctly.
5. **Memory.** Peak RSS over a streamed correlator scales with `block`, not
   `configs` (a coarse assertion / manual check on the N = 16 case).

## Out of scope (YAGNI)

- No re-implementation of derived quantities; `StreamingBootstrap` computes the
  same primary observables `Bootstrap` does, then the existing
  `DerivedQuantity` machinery consumes them unchanged.
- No parallelism across blocks (streaming is I/O-bound; add later if needed).
- No change to `Bootstrap` itself.
