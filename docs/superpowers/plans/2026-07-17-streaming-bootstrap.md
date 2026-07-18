# StreamingBootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute bootstrap estimates of observables on a large on-disk `Ensemble` without ever holding the whole ensemble or the `(configs × draws × shape)` resample tensor in memory, fixing the N=16 correlator OOM.

**Architecture:** Two classes in `supervillain/analysis/bootstrap.py`. `EnsembleStreamer(ReadWriteable)` iterates an h5-serialized `Ensemble` in blocks (memory-bounded), reading cheap metadata eagerly and slicing configuration fields lazily; it serializes as an h5 link to its source group. `StreamingBootstrap(Bootstrap)` binds a target h5 group at construction, builds the bootstrap index/count matrices, and computes each primary observable by streaming-accumulation over the streamer's blocks; a registry-guarded `__getattribute__` gate makes observable/derived-quantity access check disk → compute-if-absent → persist → return (write-through, resumable). Derived quantities are inherited unchanged.

**Tech Stack:** Python, numpy, h5py, the supervillain `ReadWriteable`/`Data`/`Batch`/`Observable`/`DerivedQuantity` machinery.

## Global Constraints

- New code lives in `supervillain/analysis/bootstrap.py`; **no change to `Bootstrap` itself** (subclass only).
- Tests live in `test/` (pytest `testpaths = ["test"]`, `pythonpath = ["."]`). Run with `uv run --project . pytest`.
- The streaming resample must equal `Bootstrap._resample` to ~1e-12 given the *same* `indices` (summation order differs, so not bit-identical).
- Peak memory scales with `block`, never `configs`.
- Observable datasets are written into the target with `Data.write` (so `ReadWriteable.from_h5` round-trips them); metadata fields (`draws`, `indices`, `Action`, `streamer`) likewise.
- `EnsembleStreamer` serializes as a link only (`h5py.SoftLink` same file else `h5py.ExternalLink`) + a `block` attr; a broken link must not crash `from_h5`, only surface when `blocks()` is first used.

---

### Task 1: `EnsembleStreamer` — block iteration of a serialized Ensemble

**Files:**
- Modify: `supervillain/analysis/bootstrap.py` (add class + a module helper)
- Test: `test/test_streaming_bootstrap.py`

**Interfaces:**
- Consumes: `supervillain.h5.ReadWriteable`, `supervillain.h5.Data`, `supervillain.configurations.Configurations`, `supervillain.ensemble.Ensemble`, `supervillain.batch.Batch`/`resolve_batch_cls`, `supervillain.h5.extendable`.
- Produces:
  - `EnsembleStreamer(source_group, block=64)` with `__len__`, `.weight`, `.Action`, `.block`, `.blocks()` → generator of `(start:int, sub_ensemble:Ensemble)`, and custom `to_h5(group, _top=True)` / `from_h5(group, strict=True, _top=True)`.
  - module helper `_read_batch_block(field_group, start, stop) -> Batch`.

- [ ] **Step 1: Write the failing test** (fidelity: concatenated blocks equal the full-load ensemble, field-by-field and for a computed observable)

Add to `test/test_streaming_bootstrap.py`:

```python
import numpy as np
import h5py
import pytest

import supervillain
from supervillain.action import NoIntersections
from supervillain.lattice import Lattice
import supervillain.generator.no_intersection as gen
import supervillain.generator.villain as villain
from supervillain.analysis import Bootstrap
from supervillain.analysis.bootstrap import EnsembleStreamer, StreamingBootstrap


def _small_ensemble(tmp_path, N=4, kappa=0.1, configs=40, seed=17):
    """Generate a small NoIntersections DefectGas ensemble and store it to h5.
    Returns the path; the ensemble group is at '/ensemble'."""
    L = Lattice(4, N)
    S = NoIntersections(L, kappa=kappa)
    rng = np.random.default_rng(seed)
    companions = (villain.SiteUpdate(S), villain.ExactUpdate(S),
                  villain.CohomologyUpdate(S), gen.WrappingLoopUpdate(S))
    for g in companions:
        g.rng = rng
    tuner = gen.DefectGasFugacityTuner(S, companions=companions, max_defects=8, rng=rng)
    zeta = tuner.tune(start='cold')
    from supervillain.generator.combining import Sequentially
    chain = Sequentially(*companions, gen.DefectGas(S, fugacity=zeta, max_defects=8, rng=rng))
    e = supervillain.Ensemble(S).generate(configs, chain, start='cold')
    path = tmp_path / 'ens.h5'
    with h5py.File(path, 'w') as f:
        g = e.__dict__.pop('generator', None)
        e.__dict__.pop('start', None)
        e.to_h5(f.create_group('ensemble'))
    return path


def test_streamer_fidelity(tmp_path):
    path = _small_ensemble(tmp_path)
    with h5py.File(path, 'r') as f:
        full = supervillain.Ensemble.from_h5(f['ensemble'])
        full.generator = None
        streamer = EnsembleStreamer(f['ensemble'], block=7)

        assert len(streamer) == len(full)

        # Field concatenation, block by block.
        phi_blocks = []
        theta_blocks = []
        for start, sub in streamer.blocks():
            phi_blocks.append(np.asarray(sub.phi))
            theta_blocks.append(np.asarray(sub.Theta_Theta))
        phi = np.concatenate(phi_blocks, axis=0)
        theta = np.concatenate(theta_blocks, axis=0)
        assert np.array_equal(phi, np.asarray(full.phi))
        assert np.array_equal(theta, np.asarray(full.Theta_Theta))

        # A computed (non-inline) observable concatenated equals the full computation.
        iw_blocks = [np.asarray(sub.IntersectionWinding) for _, sub in streamer.blocks()]
        iw = np.concatenate(iw_blocks, axis=0)
        assert np.allclose(iw, np.asarray(full.IntersectionWinding))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project . pytest test/test_streaming_bootstrap.py::test_streamer_fidelity -x -q`
Expected: FAIL with `ImportError: cannot import name 'EnsembleStreamer'`.

- [ ] **Step 3: Write minimal implementation**

At the top of `supervillain/analysis/bootstrap.py` add imports and, after the `Bootstrap` class, the helper + `EnsembleStreamer`:

```python
import os
import pickle

import h5py

from supervillain.h5 import ReadWriteable, Data
import supervillain.h5.extendable as extendable
from supervillain.batch import Batch, resolve_batch_cls
from supervillain.configurations import Configurations
import supervillain.ensemble


def _read_batch_block(field_group, start, stop):
    r'''Reconstruct a :class:`~supervillain.batch.Batch` from configs
    ``[start:stop]`` of a stored field group, mirroring the ``batch`` storage
    strategy but slicing the ``data`` dataset instead of reading it whole.'''
    if 'H5Batch_item_kwargs' not in field_group.attrs:
        raise ValueError(
            f'{field_group.name} is not a stored Batch; EnsembleStreamer only '
            'streams Batch-valued configuration fields.')
    tag = field_group.attrs.get('H5Batch_cls', '')
    if isinstance(tag, bytes):
        tag = tag.decode()
    cls = resolve_batch_cls(tag) if tag else None
    data = extendable.array(field_group['data'][start:stop])
    item_kwargs = pickle.loads(field_group.attrs['H5Batch_item_kwargs'].tobytes())
    return Batch(data, cls=cls, dtype=data.dtype, **item_kwargs)


class EnsembleStreamer(ReadWriteable):
    r'''Memory-bounded block iteration of an h5-serialized :class:`~.Ensemble`.

    Reads only cheap metadata (config count, per-config weight, Action) eagerly;
    configuration fields are sliced lazily in :meth:`blocks`.  Serializes as an
    h5 link to its source group so a :class:`StreamingBootstrap` reconstructs it
    automatically.
    '''

    def __init__(self, source_group, block=64):
        self._source = source_group
        self.block = block
        self.Action = Data.read(source_group['Action'])
        self.weight = Data.read(source_group['weight'])
        self._length = len(np.asarray(self.weight))

    def __len__(self):
        return self._length

    def blocks(self):
        r'''Yield ``(start, sub_ensemble)`` covering the configs contiguously in
        blocks of at most ``self.block``.  Each ``sub_ensemble`` is a fresh
        in-memory :class:`~.Ensemble` carrying only the sliced configuration
        fields, so any observable recomputes from those fields.'''
        if self._source is None:
            raise RuntimeError(
                'EnsembleStreamer source ensemble unavailable (broken h5 link); '
                'only cached observables can be estimated.')
        fields = self._source['configuration/fields']
        for start in range(0, self._length, self.block):
            stop = min(start + self.block, self._length)
            block_fields = {name: _read_batch_block(fields[name], start, stop)
                            for name in fields}
            cfgs = Configurations(block_fields)
            sub = supervillain.ensemble.Ensemble(self.Action).from_configurations(cfgs)
            yield start, sub

    def to_h5(self, group, _top=True):
        group.attrs['block'] = self.block
        source = self._source
        source_path = source.name
        source_file = source.file.filename
        try:
            same = os.path.samefile(source_file, group.file.filename)
        except OSError:
            same = (source_file == group.file.filename)
        if same:
            group['source'] = h5py.SoftLink(source_path)
        else:
            group['source'] = h5py.ExternalLink(source_file, source_path)

    @classmethod
    def from_h5(cls, group, strict=True, _top=True):
        o = cls.__new__(cls)
        o.block = int(group.attrs['block'])
        try:
            source = group['source']
            o._source = source
            o.Action = Data.read(source['Action'])
            o.weight = Data.read(source['weight'])
            o._length = len(np.asarray(o.weight))
        except (KeyError, OSError):
            o._source = None
            o.Action = None
            o.weight = None
            o._length = None
        return o
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project . pytest test/test_streaming_bootstrap.py::test_streamer_fidelity -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evanberkowitz/physics/supervillain/library
git add supervillain/analysis/bootstrap.py test/test_streaming_bootstrap.py
git commit -m "feat(analysis): EnsembleStreamer for memory-bounded block iteration"
```

---

### Task 2: `StreamingBootstrap` — streaming resample + write-through gate

**Files:**
- Modify: `supervillain/analysis/bootstrap.py` (add `StreamingBootstrap` after `EnsembleStreamer`)
- Test: `test/test_streaming_bootstrap.py`

**Interfaces:**
- Consumes: `EnsembleStreamer`, `Bootstrap`, `supervillain.observables`, `supervillain.derivedQuantities`, `Batch`.
- Produces: `StreamingBootstrap(streamer, target_group, draws=100, rng=None)` with `_resample_streaming(name)`, `__getattribute__`, `__getattr__`, an `Ensemble` property aliasing the streamer, and inherited `estimate`/`from_h5`.

- [ ] **Step 1: Write the failing equivalence test** (the load-bearing test: same `indices` → agree to ~1e-12 for a scalar, a correlator, and a derived quantity)

Append to `test/test_streaming_bootstrap.py`:

```python
def test_streaming_equivalence(tmp_path):
    path = _small_ensemble(tmp_path)
    with h5py.File(path, 'r+') as f:
        full = supervillain.Ensemble.from_h5(f['ensemble'])
        full.generator = None

        # A reference Bootstrap on the in-memory ensemble.
        ref = Bootstrap(full, draws=50)

        # A StreamingBootstrap forced to use the SAME indices.
        streamer = EnsembleStreamer(f['ensemble'], block=9)
        target = f.create_group('boot')
        sb = StreamingBootstrap(streamer, target, draws=50)
        sb.indices = ref.indices  # force identical resampling
        # rebuild the count matrix from the forced indices
        sb._rebuild_counts()

        for name in ('ActionDensity', 'Theta_Theta', 'IntersectionSusceptibility'):
            m_ref, e_ref = ref.estimate(name)
            m_sb, e_sb = sb.estimate(name)
            assert np.allclose(np.asarray(m_ref), np.asarray(m_sb), atol=1e-10, rtol=1e-8), name
            assert np.allclose(np.asarray(e_ref), np.asarray(e_sb), atol=1e-10, rtol=1e-8), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project . pytest test/test_streaming_bootstrap.py::test_streaming_equivalence -x -q`
Expected: FAIL with `ImportError` / `AttributeError` (`StreamingBootstrap` undefined).

- [ ] **Step 3: Write minimal implementation**

Add `StreamingBootstrap` after `EnsembleStreamer` in `supervillain/analysis/bootstrap.py`:

```python
class StreamingBootstrap(Bootstrap):
    r'''A :class:`Bootstrap` that resamples an on-disk ensemble in blocks and
    writes each observable through to a target h5 group, never holding the whole
    ensemble or the ``(configs × draws × shape)`` tensor in memory.

    Parameters
    ----------
    streamer: EnsembleStreamer
        Source of blocks; owns ``block`` and the per-config ``weight``/``Action``.
    target_group: h5py.Group
        Where metadata and streamed observables are written (Bootstrap layout).
    draws: int
        Number of bootstrap resamplings.
    rng: numpy.random.Generator, optional
        Draws the resampling indices; defaults to the global RNG.
    '''

    # names that must never be routed through the disk-cache gate
    _PASSTHROUGH = frozenset({
        'target_group', 'streamer', 'draws', 'indices', 'Action', '_n',
        '_resample_streaming', '_rebuild_counts', 'Ensemble', 'estimate',
    })

    def __init__(self, streamer, target_group, draws=100, rng=None):
        self.streamer = streamer
        self.target_group = target_group
        self.draws = draws
        self.Action = streamer.Action
        cfgs = len(streamer)
        draw = (rng.integers if rng is not None else np.random.randint)
        if rng is not None:
            self.indices = rng.integers(0, cfgs, (cfgs, draws))
        else:
            self.indices = np.random.randint(0, cfgs, (cfgs, draws))
        self._rebuild_counts()
        # Construction-time metadata so the target is a valid Bootstrap layout.
        Data.write(target_group, 'draws', self.draws)
        Data.write(target_group, 'indices', self.indices)
        Data.write(target_group, 'Action', self.Action)
        Data.write(target_group, 'streamer', self.streamer)

    @property
    def Ensemble(self):
        # DerivedQuantity.__get__ and plot_* reach the action through .Ensemble.
        return self.streamer

    def _rebuild_counts(self):
        cfgs, draws = self.indices.shape
        n = np.zeros((cfgs, draws), dtype=np.int64)
        for d in range(draws):
            n[:, d] = np.bincount(self.indices[:, d], minlength=cfgs)
        self._n = n

    def _resample_streaming(self, name):
        weight = np.asarray(self.streamer.weight)
        n = self._n
        draws = self.draws
        numerator = None
        denominator = np.zeros(draws)
        for start, sub in self.streamer.blocks():
            obs = Batch.as_array(getattr(sub, name))
            b = obs.shape[0]
            wn = n[start:start + b] * weight[start:start + b, None]  # (b, draws)
            contrib = np.einsum('bd,b...->d...', wn, obs)            # (draws, ...)
            numerator = contrib if numerator is None else numerator + contrib
            denominator += wn.sum(axis=0)
        shape = (draws,) + (1,) * (numerator.ndim - 1)
        return numerator / denominator.reshape(shape)

    def __getattr__(self, name):
        # Reached (via the tp_getattro hook) when `name` is not a class/instance
        # attribute — i.e. a primary observable to stream.
        return self._resample_streaming(name)

    def __getattribute__(self, name):
        if name.startswith('__') or name in StreamingBootstrap._PASSTHROUGH:
            return super().__getattribute__(name)
        gated = (name in supervillain.observables) or (name in supervillain.derivedQuantities)
        if not gated:
            return super().__getattribute__(name)
        target = super().__getattribute__('target_group')
        if target is not None and name in target:
            value = Data.read(target[name])
            self.__dict__[name] = value
            return value
        try:
            value = super().__getattribute__(name)      # derived-quantity descriptor
        except AttributeError:
            value = super().__getattribute__('_resample_streaming')(name)  # primary
        if target is not None and name not in target:
            Data.write(target, name, np.asarray(value))
        return value

    @classmethod
    def from_h5(cls, group, strict=True, _top=True):
        o = super().from_h5(group, strict=strict, _top=_top)
        o.target_group = group
        return o
```

Add `import supervillain` at the top of the module (needed for `supervillain.observables` / `supervillain.derivedQuantities` in the gate). Remove the dead `draw = ...` line if you prefer; the `rng`/global branch below it is what runs.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project . pytest test/test_streaming_bootstrap.py::test_streaming_equivalence -x -q`
Expected: PASS (scalar, correlator, and derived-quantity estimates agree to ~1e-10).

- [ ] **Step 5: Commit**

```bash
cd /Users/evanberkowitz/physics/supervillain/library
git add supervillain/analysis/bootstrap.py test/test_streaming_bootstrap.py
git commit -m "feat(analysis): StreamingBootstrap streaming resample + write-through gate"
```

---

### Task 3: Persistence — write-through round-trip, resumability, portability

**Files:**
- Test: `test/test_streaming_bootstrap.py`
- Modify: `supervillain/analysis/bootstrap.py` only if a test exposes a defect.

**Interfaces:**
- Consumes: everything from Tasks 1–2. No new public surface.

- [ ] **Step 1: Write the failing tests** (round-trip via `StreamingBootstrap.from_h5`; an un-streamed observable after reload; resumability without recompute; portability to plain `Bootstrap`)

Append to `test/test_streaming_bootstrap.py`:

```python
def test_write_through_roundtrip_and_portability(tmp_path):
    path = _small_ensemble(tmp_path)
    with h5py.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], block=9)
        target = f.create_group('boot')
        sb = StreamingBootstrap(streamer, target, draws=40)
        # Stream two observables (one primary, one derived).
        m_theta, e_theta = sb.estimate('Theta_Theta')
        m_chi, e_chi = sb.estimate('IntersectionSusceptibility')
        assert 'Theta_Theta' in target
        assert 'IntersectionSusceptibility' in target

    # Reload via StreamingBootstrap.from_h5: cached values match; streamer rebuilt.
    with h5py.File(path, 'r+') as f:
        sb2 = StreamingBootstrap.from_h5(f['boot'])
        assert isinstance(sb2.streamer, EnsembleStreamer)
        m2, e2 = sb2.estimate('Theta_Theta')
        assert np.allclose(np.asarray(m2), np.asarray(m_theta))
        # An un-streamed observable now streams through the reconstructed streamer.
        m3, e3 = sb2.estimate('ActionDensity')
        assert np.isfinite(np.asarray(m3)).all()
        assert 'ActionDensity' in f['boot']

    # Portability: a plain Bootstrap.from_h5 estimates the streamed observable.
    with h5py.File(path, 'r') as f:
        plain = Bootstrap.from_h5(f['boot'])
        mp, ep = plain.estimate('Theta_Theta')
        assert np.allclose(np.asarray(mp), np.asarray(m_theta))


def test_resumability_no_recompute(tmp_path):
    path = _small_ensemble(tmp_path)
    with h5py.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], block=9)
        target = f.create_group('boot')
        sb = StreamingBootstrap(streamer, target, draws=30)
        sb.estimate('Theta_Theta')

        # Break the streamer so any recompute would raise; a cached read must not.
        sb.streamer._source = None
        m, e = sb.estimate('Theta_Theta')   # served from disk
        assert np.isfinite(np.asarray(m)).all()
        with pytest.raises(RuntimeError):
            sb.estimate('ActionDensity')     # not cached -> must stream -> raises
```

- [ ] **Step 2: Run tests to verify they fail (or pass) — diagnose any failure**

Run: `uv run --project . pytest test/test_streaming_bootstrap.py -x -q`
Expected: the two new tests initially fail only if a defect exists; fix the class until all pass. Likely-correct code passes directly.

- [ ] **Step 3: Fix any defect surfaced**

If `Data.read(target[name])` returns an `extendable.array` that trips a later `np.asarray` write, or if the `Ensemble` property collides with a stored field, adjust minimally. No speculative changes.

- [ ] **Step 4: Run the whole test module**

Run: `uv run --project . pytest test/test_streaming_bootstrap.py -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evanberkowitz/physics/supervillain/library
git add supervillain/analysis/bootstrap.py test/test_streaming_bootstrap.py
git commit -m "test(analysis): StreamingBootstrap round-trip, resumability, portability"
```

---

### Task 4: Export + regression guard

**Files:**
- Modify: `supervillain/analysis/__init__.py` (export the new names)
- Test: `test/test_streaming_bootstrap.py`

**Interfaces:**
- Produces: `supervillain.analysis.StreamingBootstrap`, `supervillain.analysis.EnsembleStreamer` importable from the package root path used elsewhere.

- [ ] **Step 1: Write the failing import test**

Append:

```python
def test_public_exports():
    from supervillain.analysis import StreamingBootstrap as SB, EnsembleStreamer as ES
    assert SB is StreamingBootstrap
    assert ES is EnsembleStreamer
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --project . pytest test/test_streaming_bootstrap.py::test_public_exports -x -q`
Expected: FAIL (`ImportError` if not exported).

- [ ] **Step 3: Add the exports**

In `supervillain/analysis/__init__.py`, alongside the existing `Bootstrap` export, add:

```python
from supervillain.analysis.bootstrap import Bootstrap, StreamingBootstrap, EnsembleStreamer
```

(Match the existing import style in that file; only add the two new names.)

- [ ] **Step 4: Run to verify it passes and run the full suite**

Run: `uv run --project . pytest test/test_streaming_bootstrap.py -q && uv run --project . pytest -q`
Expected: new tests PASS; the pre-existing suite is unaffected.

- [ ] **Step 5: Commit**

```bash
cd /Users/evanberkowitz/physics/supervillain/library
git add supervillain/analysis/__init__.py test/test_streaming_bootstrap.py
git commit -m "feat(analysis): export StreamingBootstrap and EnsembleStreamer"
```

---

## Downstream (notebook repo, separate from the library change)

After the library suite is green, in `no-intersections/`:

1. **Journal** the design + implementation in `journal/2026-07-17.md` and update `topic/` + indexes.
2. **Bootstrap the θ observables from an N=16 weight-table ensemble** using `StreamingBootstrap` (the original `tbltest` raw file was pruned, so regenerate one weight-table raw ensemble at κ=0.05 via `transition_dg.py --weight-table`, then stream `IntersectionSusceptibility`, `DoubleIntersectionSusceptibility`, `IntersectionWindingSquared`, `IntersectionBinderCumulant` from its raw h5).
3. **Campaign** just below the transition (κ ≈ 0.044–0.050 and lower) with the tuned weight table, streaming the θ observables per κ.

## Self-Review

- **Spec coverage:** streaming resample (Task 2), streamer block iteration + link serialization (Task 1), write-through `__getattribute__` gate (Task 2), inherited `from_h5` reconstructing the streamer (Task 2), equivalence/fidelity/round-trip/resumability/portability tests (Tasks 1–3), no `Bootstrap` change (subclass only). Memory test is covered qualitatively by the block-sized accumulator design and demonstrated downstream on the real N=16 case rather than as a brittle RSS unit assertion.
- **Type consistency:** `EnsembleStreamer(source_group, block)`, `.blocks()` → `(start, Ensemble)`, `StreamingBootstrap(streamer, target_group, draws, rng)`, `_resample_streaming`/`_rebuild_counts`/`_n`, `Ensemble` property → streamer; all referenced consistently across tasks.
- **Placeholder scan:** none.
