#!/usr/bin/env python

import os
import pickle

import numpy as np
import h5py

import supervillain
from supervillain.batch import Batch, resolve_batch_cls
from supervillain.h5 import ReadWriteable, Data
import supervillain.h5.extendable as extendable
from supervillain.configurations import Configurations
import supervillain.ensemble
from supervillain.performance import Timer

import logging
logger = logging.getLogger(__name__)

class Bootstrap(ReadWriteable):
    r'''
    The bootstrap is a resampling technique for estimating uncertainties.

    For samples with weights :math:`w` the expectation value of an observable is

    .. math::
        \left\langle O \right\rangle = \frac{\left\langle O w \right\rangle}{\left\langle w \right\rangle}

    and an accurate bootstrap estimate of the left-hand side requires tracking the correlations between the numerator and denominator.
    Moreover, quoting correlated uncertainties requires resampling different observables in the same way.

    Parameters
    ----------
        ensemble:   Ensemble
            The ensemble to resample.
        draws:      int
            The number of times to resample.

    Any :ref:`primary observables` that :class:`~.Ensemble` supports can be called from the :class:`~.Bootstrap`.
    :class:`~.Bootstrap` uses :code:`getattr` trickery under the hood to intercept calls and perform the weighted average transparently.

    Each observable returns an array of the same dimension as the ensemble's observable.  However, rather than configurations first, :code:`draws` are first.

    Each draw is a weighted average over the resampled weight, as shown above, and is therefore an estimator for the expectation value.
    These are guaranteed (by the `central limit theorem`_) to be normally distributed as long as you have not sinned.
    To get an uncertainty estimate one need only take the :code:`mean()` for a central value and :code:`std()` for the uncertainty on the mean.

    .. _central limit theorem: https://en.wikipedia.org/wiki/Central_limit_theorem
    '''

    def __init__(self, ensemble, draws=100):
        self.Ensemble = ensemble
        r'''The ensemble from which to resample.'''
        self.Action = ensemble.Action
        r'''The action underlying the ensemble.'''
        self.draws = draws
        r'''The number of resamplings.'''
        cfgs = len(ensemble)
        self.indices = np.random.randint(0, cfgs, (cfgs, draws))
        r'''The random draws themselves; configurations × draws.'''
        
    def __len__(self):
        return self.draws

    def _resample(self, obs):
        # Each observable should be multiplied by its respective weight.
        # Each draw should be divided by its average weight.
        obs = Batch.as_array(obs)
        w = Batch.as_array(self.Ensemble.weight)[self.indices]

        # This index ordering is needed to broadcast the weights division correctly.
        # See https://github.com/evanberkowitz/two-dimensional-gasses/issues/55
        # We return the bootstrap axis to the front to provide an analogous interface for Bootstrap and Ensemble quantities.
        return np.einsum('...d->d...', np.einsum('cd,cd...->c...d', w, obs[self.indices]).mean(axis=0) / w.mean(axis=0))
    
    def __getattr__(self, name):
        
        with Timer(logger.info, f'Bootstrapping {name}', per=len(self)):

            try:
                forward = getattr(self.Ensemble, name)
            except Exception as e:
                raise AttributeError(f"... and so 'Bootstrap' object has no attribute '{name}'") from e

            self.__dict__[name] = self._resample(forward)
            return self.__dict__[name]

    def plot_band(self, axis, observable, color=None):
        r'''
        Plots the single-number-valued observable as a horizontal band.

        Parameters
        ----------
        axis: matplotlib.pyplot.axis
            The axis on which to plot.
        observable: string
            Name of the observable or derived quantity.
        color: matplotlib color
            See the `matplotlib color API <https://matplotlib.org/stable/api/colors_api.html#module-matplotlib.colors>`_\. Defaults to the previously-used color.

        '''
        data = getattr(self, observable)
        mean = data.mean(axis=0)
        err  = data.std (axis=0)

        if mean.shape != ():
            raise ValueError(f'{observable} has shape {mean.shape}')

        if color is None:
            color = axis.get_lines()[-1].get_color()
        axis.axhspan(mean-err, mean+err, color=color, alpha=0.5, linestyle='none')

    def plot_correlator(self, axis, correlator, offset=0., symmetrize=True, multiplier=1., linestyle='none', marker='o', markerfacecolor='none', **kwargs):
        r'''
        Plots the space-dependent correlator against $\Delta x$ on the axis.
        Plotting options and kwargs are forwarded.

        Parameters
        ----------
        axis: matplotlib.pyplot.axis
            The axis on which to plot.
        correlator: string
            Name of the observable or derived quantity.
        offset: float
            Horizontal displacement, good for visually separating two correlators.
        symmetrize: bool
            If True (default), project onto the totally-symmetric irrep via :py:meth:`~.Lattice.symmetrize`.
        multiplier: float
            Rescales the observable by an overall constant.
        '''

        L = self.Ensemble.Action.Lattice
        Δx = L.linearize(L.R_squared)**0.5
        C = getattr(self, correlator).real
        if symmetrize:
            C = L.symmetrize(C)

        axis.errorbar(
                Δx+offset,
                multiplier * L.linearize(C.mean(axis=0)),
                multiplier * L.linearize(C.std(axis=0)),
                linestyle=linestyle,
                marker=marker,
                markerfacecolor=markerfacecolor,
                **kwargs
                )
        axis.set_xlabel('∆x')

    def estimate(self, observable):
        r'''
        Parameters
        ----------
        observable: string
            Name of the observable or derived quantity

        Returns
        -------
        tuple:
            A tuple with the central value and uncertainty estimate for the observable.  Need not be scalars, if the observable has other indices, the pieces of the tuples have those indices.
        '''
        o = getattr(self, observable)
        return (np.mean(o, axis=0), np.std(o, axis=0))


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


def _stream_weight(source_group):
    r'''Derive the per-config importance weight from an on-disk ensemble exactly
    as :attr:`Ensemble.weight` does in memory: sum the cheap ``logWeight_*``
    scalar columns and exponentiate with a SINGLE global ``max`` subtraction.

    The global max is a correctness requirement, not just overflow safety: the
    streaming resample forms ⟨Ow⟩/⟨w⟩ across blocks, and the ``exp`` offset
    cancels in that ratio only if it is one constant shared by every config --- a
    per-block max would silently bias the estimate.  Reads only the scalar
    ``data`` datasets of the log columns (never the heavy fields), so the whole
    weight vector is materialized eagerly before any block streams.  Falls back
    to a legacy stored ``weight`` dataset, then to unit weights.'''
    fields = source_group['configuration/fields']
    cols = sorted(k for k in fields.keys() if k.startswith('logWeight_'))
    if cols:
        lw = sum(np.asarray(fields[k]['data'][:]) for k in cols)
        return np.exp(lw - lw.max())
    if 'weight' in source_group:
        return np.asarray(Batch.as_array(Data.read(source_group['weight'])))
    name = next(iter(fields.keys()))
    return np.ones(len(fields[name]['data']))


class EnsembleStreamer(ReadWriteable):
    r'''Memory-bounded block iteration of an h5-serialized :class:`~.Ensemble`.

    Reads only cheap metadata (config count, per-config weight, Action) eagerly;
    configuration fields are sliced lazily in :meth:`blocks`.  Serializes as an
    h5 link to its source group so a :class:`StreamingBootstrap` reconstructs it
    automatically through the inherited :meth:`~.ReadWriteable.from_h5`.

    Parameters
    ----------
    source_group: h5py.Group
        A group holding an :meth:`~.Ensemble.to_h5` dump.
    block: int
        The maximum number of configurations to hold in memory at once.
    '''

    def __init__(self, source_group, block=64):
        self._source = source_group
        self.block = block
        self.Action = Data.read(source_group['Action'])
        self.weight = _stream_weight(source_group)
        self._length = len(np.asarray(self.weight))

    def __len__(self):
        return self._length

    def blocks(self):
        r'''Yield ``(start, sub_ensemble)`` covering the configurations
        contiguously in blocks of at most ``self.block``.  Each ``sub_ensemble``
        is a fresh in-memory :class:`~.Ensemble` carrying only the sliced
        configuration fields, so any observable recomputes from those fields.'''
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
            o.weight = _stream_weight(source)
            o._length = len(np.asarray(o.weight))
        except (KeyError, OSError):
            o._source = None
            o.Action = None
            o.weight = None
            o._length = None
        return o


class StreamingBootstrap(Bootstrap):
    r'''A :class:`Bootstrap` that resamples an on-disk ensemble in blocks and
    writes each observable through to a target h5 group, never holding the whole
    ensemble or the ``(configs × draws × shape)`` resample tensor in memory.

    A single registry-guarded :meth:`__getattribute__` gate makes every
    :ref:`primary observable <primary observables>` and
    :class:`~.DerivedQuantity` access *check the target group on disk, compute by
    streaming if absent, persist, and return* -- so access is persistence
    (write-through) and re-running skips work already on disk (resumable).

    Parameters
    ----------
    streamer: EnsembleStreamer
        Source of blocks; owns ``block`` and the per-config ``weight``/``Action``.
    target_group: h5py.Group
        Where metadata and streamed observables are written (Bootstrap layout).
    draws: int
        The number of bootstrap resamplings.
    rng: numpy.random.Generator, optional
        Draws the resampling indices; defaults to the global RNG.
    '''

    # Names that must never be routed through the disk-cache gate (they are the
    # object's own machinery, not observables/derived quantities).
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
        if rng is not None:
            self.indices = rng.integers(0, cfgs, (cfgs, draws))
        else:
            self.indices = np.random.randint(0, cfgs, (cfgs, draws))
        self._rebuild_counts()
        # Construction-time metadata so the target is a valid, from_h5-readable
        # Bootstrap layout from the first observable on.
        Data.write(target_group, 'draws', self.draws)
        Data.write(target_group, 'indices', self.indices)
        Data.write(target_group, 'Action', self.Action)
        Data.write(target_group, 'streamer', self.streamer)

    @property
    def Ensemble(self):
        # DerivedQuantity.__get__ and the plot_* helpers reach the action through
        # .Ensemble.Action; the streamer carries Action, weight, and __len__.
        return self.streamer

    def _rebuild_counts(self):
        r'''Rebuild the count matrix ``n[i, d] = #{c : indices[c, d] == i}`` from
        ``self.indices``.  Call after overriding ``indices`` directly.'''
        cfgs, draws = self.indices.shape
        n = np.zeros((cfgs, draws), dtype=np.int64)
        for d in range(draws):
            n[:, d] = np.bincount(self.indices[:, d], minlength=cfgs)
        self._n = n

    def _resample_streaming(self, name):
        r'''The memory-safe streaming resample of a primary observable ``name``.

        Accumulates ``numerator[d] = sum_i n[i,d] w[i] obs[i]`` and
        ``denominator[d] = sum_i n[i,d] w[i]`` block by block; the result
        ``numerator / denominator`` equals :meth:`Bootstrap._resample` (given the
        same ``indices``) to floating point.'''
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
        # Reached (via the tp_getattro hook) when `name` is neither a class nor an
        # instance attribute.  Stream it only if it is a genuine primary
        # observable; otherwise raise so a missing internal surfaces as an honest
        # AttributeError instead of an accidental (recursive) resample.
        if name.startswith('_') or name not in supervillain.observables:
            raise AttributeError(name)
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
        # The inherited ReadWriteable.from_h5 reconstructs every field, including
        # the streamer (via EnsembleStreamer.from_h5, which resolves the link) and
        # the cached observable datasets; we only bind the live target handle.
        o = super().from_h5(group, strict=strict, _top=_top)
        o.target_group = group
        o._rebuild_counts()   # the count matrix is derived from indices, not stored
        return o
