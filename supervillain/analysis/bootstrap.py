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
    r'''Derive the per-configuration importance weight of an on-disk ensemble,
    without reading a single configuration field.  Returns one weight per
    configuration.

    Three sources are consulted, in order.  A generator that reweights emits its
    contribution as an inline logWeight_<name> scalar column, one per contributing
    generator; logs sum so that weights multiply, and the namespacing keeps two
    generators composed with Sequentially from colliding.  Failing that, an
    explicitly stored weight is read.  Failing that, every configuration weighs
    the same --- which is every ensemble, until a reweighting generator lands.

    The single global max subtracted from the summed logs is a correctness
    requirement, not merely overflow safety.  _resample_streaming accumulates
    <Ow> and <w> separately, block by block, and the exp(-max) offset cancels
    between them only if it is one constant shared by every configuration; a
    per-block max would silently bias the estimate.  That is why the whole weight
    vector is built eagerly, here, before any block streams -- and why only the
    cheap scalar columns are touched, never the heavy fields.'''
    fields = source_group['configuration/fields']
    logWeights = sorted(k for k in fields.keys() if k.startswith('logWeight_'))
    if logWeights:
        logWeight = np.sum([np.asarray(fields[k]['data'][:]) for k in logWeights], axis=0)
        return np.exp(logWeight - logWeight.max())
    if 'weight' in source_group:
        return np.asarray(Batch.as_array(Data.read(source_group['weight'])))
    name = next(iter(fields.keys()))
    return np.ones(len(fields[name]['data']))


class EnsembleStreamer(ReadWriteable):
    r'''
    Hands a stored :class:`~.Ensemble` out a few configurations at a time, so that
    an ensemble too large to read can still be analyzed.

    Making one is cheap.  It reads only the :attr:`Action`, the number of
    configurations, and the :attr:`weight` of each, and leaves the configurations
    themselves on disk until :meth:`blocks` asks for them.  Each block comes back
    as an ordinary in-memory :class:`~.Ensemble`, so you can measure any
    :ref:`primary observable <primary observables>` on it in the usual way.

    .. note::
       A streamer is a view of its source, not a copy of it.  Writing one stores a
       link to the ensemble rather than the configurations, so a
       :class:`StreamingBootstrap` finds its way back to the ensemble on its own
       and you never have to say where it went.

    .. warning::
       Moving or deleting that ensemble breaks the link.  A streamer that cannot
       find its source can no longer hand out configurations, though a
       :class:`StreamingBootstrap` can still serve whatever it has already saved.

    Parameters
    ----------
    source_group: h5py.Group
        A group holding an :meth:`~.Ensemble.to_h5` dump.
    block: int
        The maximum number of configurations to hold in memory at once.
    '''

    def __init__(self, source_group, block=64):
        if block < 1:
            # range(0, length, block) is empty for a non-positive stride, so
            # blocks() would quietly yield nothing and a resample would come back
            # with no configurations in it rather than an error.
            raise ValueError(f'block should be at least 1 configuration, not {block}.')
        self._source = source_group
        self.block = block
        r'''The maximum number of configurations held in memory at once.'''
        self.Action = Data.read(source_group['Action'])
        r'''The action underlying the ensemble.'''
        self.weight = _stream_weight(source_group)
        r'''The importance weight of each configuration.'''
        self._length = len(np.asarray(self.weight))

    def __len__(self):
        return self._length

    def blocks(self):
        r'''
        Go through the ensemble in order, at most :attr:`block` configurations at
        a time.  Each piece is an ordinary :class:`~.Ensemble` of its own, and
        only one of them is in memory at once.

        Yields
        ------
        tuple:
            ``(start, sub_ensemble)``, where ``start`` counts the configurations
            that came before this piece.

        Raises
        ------
        RuntimeError
            If the ensemble this streamer was made from can no longer be found.
        '''
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
        r'''
        Write the streamer as :attr:`block` and a link to its source ensemble ---
        an :class:`h5py.SoftLink` if the source lives in the same file as
        ``group``, an :class:`h5py.ExternalLink` if it does not.  No
        configuration is copied.
        '''
        group.attrs['block'] = self.block
        source = self._source
        if source is None:
            raise RuntimeError(
                'EnsembleStreamer source ensemble unavailable (broken h5 link); '
                'there is no source to link to.')
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
        r'''
        Follow the stored link back to the source ensemble and re-read its cheap
        metadata.  If the link cannot be resolved the streamer still reads back,
        but with :attr:`Action` and :attr:`weight` set to ``None`` and
        :meth:`blocks` unavailable.
        '''
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
    r'''
    A :class:`Bootstrap` that resamples an ensemble as an :class:`EnsembleStreamer`
    hands it out, a few configurations at a time, and saves each result into a
    target h5 group.

    Every :ref:`primary observable <primary observables>` and
    :class:`~.DerivedQuantity` a :class:`Bootstrap` offers is available here and
    means the same thing.  What differs is where the answer comes from: the
    ensemble on disk rather than the ensemble in memory.  Resampled the same way
    the two differ only by floating-point roundoff --- they are the same sum added
    up in a different order --- so this is a way of affording an estimate, not of
    approximating one.

    .. note::
       Asking for a quantity is what saves it.  An analysis interrupted halfway
       through therefore resumes rather than restarts, and a quantity already
       computed costs nothing to ask for again.

    .. warning::
       A resampling is drawn over a fixed number of configurations, so a bootstrap
       does not survive its ensemble being extended.  If you
       :meth:`~.Ensemble.continue_from` an ensemble you have already bootstrapped,
       bootstrap the longer one into a new group; reading the old one back against
       the grown ensemble raises rather than report estimates of its beginning as
       though they described the whole.

    .. note::
       The target group is an ordinary :class:`Bootstrap`, so
       :meth:`~.ReadWriteable.from_h5` reads your results anywhere, with or
       without the ensemble they came from.  Reading it back as a
       :class:`StreamingBootstrap` instead recovers the link to the ensemble too,
       so you can go on to ask for more.

    .. seealso::
       :source:`example/streaming-bootstrap.py`, which bootstraps one ensemble
       both ways and tabulates the agreement.

    Parameters
    ----------
    streamer: EnsembleStreamer
        Hands out the ensemble, and carries its ``Action`` and ``weight``.
    target_group: h5py.Group
        Where the results are saved.
    draws: int
        The number of bootstrap resamplings.
    indices: numpy.ndarray, optional
        Resampling indices, configurations × draws, to use instead of fresh random
        ones; they set ``draws``.  Pass another :class:`Bootstrap`\'s
        :attr:`~.Bootstrap.indices` to resample the two identically, as when
        comparing a streamed estimate against an in-memory one.  Overrides ``rng``.
    rng: numpy.random.Generator, optional
        Draws the resampling indices; defaults to the global RNG.
    '''

    # Names that must never be routed through the disk-cache gate (they are the
    # object's own machinery, not observables/derived quantities).
    _PASSTHROUGH = frozenset({
        'target_group', 'streamer', 'draws', 'indices', 'Action', '_n',
        '_resample_streaming', '_rebuild_counts', 'Ensemble', 'estimate',
    })

    def __init__(self, streamer, target_group, draws=100, indices=None, rng=None):
        if 'indices' in target_group:
            raise ValueError(
                f'{target_group.name} already holds a StreamingBootstrap.  Read it '
                'back with StreamingBootstrap.from_h5 to carry on with the '
                'resampling it already used; constructing a new one here would draw '
                'new indices, which would not describe the results already stored.')
        self.streamer = streamer
        r'''The :class:`EnsembleStreamer` from which to resample.'''
        self.target_group = target_group
        r'''The h5 group into which streamed quantities are written.'''
        self.Action = streamer.Action
        r'''The action underlying the ensemble.'''
        cfgs = len(streamer)
        if indices is not None:
            indices = np.asarray(indices)
            if indices.ndim != 2 or indices.shape[0] != cfgs:
                raise ValueError(
                    f'indices should be ({cfgs}, draws) --- configurations × draws '
                    f'--- but are {indices.shape}.')
            self.indices = indices
            draws = indices.shape[1]   # the given indices, not the default, set the draws.
        elif rng is not None:
            self.indices = rng.integers(0, cfgs, (cfgs, draws))
        else:
            self.indices = np.random.randint(0, cfgs, (cfgs, draws))
        self.draws = draws
        r'''The number of resamplings.'''
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
        :attr:`indices`.  The counts are derived, not stored, so :meth:`from_h5`
        rebuilds them; call it yourself only if you assign ``indices`` directly
        rather than passing them to the constructor.'''
        cfgs, draws = self.indices.shape
        n = np.zeros((cfgs, draws), dtype=np.int64)
        for d in range(draws):
            n[:, d] = np.bincount(self.indices[:, d], minlength=cfgs)
        self._n = n

    def _resample_streaming(self, name):
        r'''The memory-safe streaming resample of a primary observable ``name``.

        Bootstrap._resample builds obs[indices], a (configurations, draws, ...)
        tensor, and averages it.  That tensor is what makes a correlator on a big
        lattice unaffordable, and it is avoidable: the resample never needed the
        draws themselves, only how many times each configuration was drawn.  With

            n[i,d] = #{c : indices[c,d] == i}

        the d-th resampled expectation value of O with weight w is

            <O>_d = sum_i n[i,d] w[i] O[i] / sum_i n[i,d] w[i]

        which is the same sum with the configuration index summed rather than
        stored.  Numerator and denominator each accumulate a block at a time, so
        nothing larger than one block plus the (draws, ...) answer is ever in
        memory, however long the chain.  The result equals Bootstrap._resample on
        the same indices to floating point --- reassociating a sum is all that
        separates them, which is what test_streaming_equivalence checks.'''
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
        if numerator is None:
            raise RuntimeError(
                f'{name} could not be resampled: the streamer yielded no '
                'configurations at all.')
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
        # Already in hand.  This gate intercepts *every* access to a gated name,
        # so without an in-memory check a correlator would be re-read from disk
        # each time it is touched, making an innocent-looking loop expensive.
        cached = super().__getattribute__('__dict__')
        if name in cached:
            return cached[name]
        target = super().__getattribute__('target_group')
        if target is not None and name in target:
            value = Data.read(target[name])
            cached[name] = value
            return value
        if target is not None and target.file.mode == 'r':
            raise RuntimeError(
                f'{name} has not been streamed yet, and {target.file.filename} is '
                'open read-only, so it could not be saved.  Reopen the file with '
                "mode 'r+' to stream new quantities; those already stored can be "
                'read from a read-only file as they are.')
        try:
            value = super().__getattribute__(name)      # derived-quantity descriptor
        except AttributeError:
            value = super().__getattribute__('_resample_streaming')(name)  # primary
        if target is not None and name not in target:
            Data.write(target, name, np.asarray(value))
        cached[name] = value
        return value

    @classmethod
    def from_h5(cls, group, strict=True, _top=True):
        r'''
        Read back a streaming bootstrap, including everything it has already
        streamed and the link to the ensemble it streamed from, so that further
        quantities pick up where the last pass left off.
        '''
        # The inherited ReadWriteable.from_h5 reconstructs every field, including
        # the streamer (via EnsembleStreamer.from_h5, which resolves the link) and
        # the cached observable datasets; we only bind the live target handle.
        o = super().from_h5(group, strict=strict, _top=_top)
        o.target_group = group
        o._rebuild_counts()   # the count matrix is derived from indices, not stored

        # The stored indices resample a fixed number of configurations.  If the
        # ensemble has grown since -- continue_from and extend_h5 make that easy,
        # and routine -- then every stored result describes a prefix of the
        # ensemble now on disk.  Serving those as though they described the whole
        # thing is the one way this class can be quietly wrong, so it does not.
        configurations = o.indices.shape[0]
        if o.streamer._source is not None and len(o.streamer) != configurations:
            raise ValueError(
                f'{group.name} resamples {configurations} configurations but its '
                f'ensemble now has {len(o.streamer)}.  Every result stored here '
                f'describes only the first {configurations}; bootstrap the extended '
                'ensemble into a new group rather than adding to this one.')
        return o
