#!/usr/bin/env python

r'''
Analysis of an :class:`~.Ensemble` too large to read.

An :class:`EnsembleStreamer` hands a stored ensemble out a few configurations at
a time; :class:`StreamingBlocking` averages consecutive samples of one as the
measurements go past; and :class:`StreamingBootstrap` resamples either.  Between
them they do what :meth:`~.Ensemble.cut`, :meth:`~.Ensemble.every`,
:class:`~.Blocking`, and :class:`~.Bootstrap` do, without ever holding the whole
ensemble.

What they hand each other is a SampleSource; the contract is written out above
that class.
'''

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
from supervillain.analysis.bootstrap import Bootstrap
from supervillain.analysis.autocorrelation import sample_autocorrelation_time

import logging
logger = logging.getLogger(__name__)


def _read_batch_chunk(field_group, start, stop, step=1):
    r'''Reconstruct a :class:`~supervillain.batch.Batch` from configurations
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
    data = extendable.array(field_group['data'][start:stop:step])
    item_kwargs = pickle.loads(field_group.attrs['H5Batch_item_kwargs'].tobytes())
    return Batch(data, cls=cls, dtype=data.dtype, **item_kwargs)


def _stream_weight(source_group):
    r'''Derive the per-configuration importance weight of an on-disk ensemble,
    without reading a single configuration field.  Returns one weight per
    configuration.

    A generator that reweights emits its contribution as an inline
    logWeight_<name> scalar column, one per contributing generator; logs sum so
    that weights multiply, and the namespacing keeps two generators composed with
    Sequentially from colliding.  With no such column every configuration weighs
    the same, exactly as Ensemble.weight decides it.

    An ensemble stored before the weight was derived carries a `weight` dataset
    alongside the configurations.  It is not consulted: Ensemble.weight does not
    consult it either, and honouring it here would make the streamed and in-memory
    paths disagree on the same file.  Nothing is lost --- every weight the library
    ever stored was one.

    The single global max subtracted from the summed logs is a correctness
    requirement, not merely overflow safety.  _resample_streaming accumulates
    <Ow> and <w> separately, a chunk at a time, and the exp(-max) offset cancels
    between them only if it is one constant shared by every configuration; a
    per-chunk max would silently bias the estimate.  That is why the whole weight
    vector is built eagerly, here, before any chunk streams -- and why only the
    cheap scalar columns are touched, never the heavy fields.'''
    fields = source_group['configuration/fields']
    logWeights = sorted(k for k in fields.keys() if k.startswith('logWeight_'))
    if logWeights:
        logWeight = np.sum([np.asarray(fields[k]['data'][:]) for k in logWeights], axis=0)
        return np.exp(logWeight - logWeight.max())
    name = next(iter(fields.keys()))
    return np.ones(len(fields[name]['data']))


def _stream_index(source_group):
    r'''The Markov-chain index of each configuration: one integer apiece, so it
    is read whole rather than streamed.  Falls back to counting if the stored
    ensemble predates the index.'''
    if 'index' in source_group:
        return np.asarray(Batch.as_array(Data.read(source_group['index'])))
    fields = source_group['configuration/fields']
    name = next(iter(fields.keys()))
    return np.arange(len(fields[name]['data']))


def _stream_index_stride(source_group):
    r'''The distance in the Markov chain between one stored configuration and the
    next.'''
    if 'index_stride' in source_group:
        return int(np.asarray(Data.read(source_group['index_stride'])))
    return 1


# The contract, for anyone writing a new source rather than using one.  A subclass
# supplies
#
#     Action        the action underlying the samples
#     weight, index one of each, per sample
#     __len__       how many samples there are
#     values(name)  the measurement of an observable, a chunk of samples at a time
#     available     whether the ensemble underneath can still be reached
#
# and inherits measured, timeseries, and autocorrelation_time.
#
# Configurations are only how a streamer produces its values; a blocking produces
# them by averaging.  That is why values(), rather than any access to
# configurations, is what StreamingBootstrap consumes: it resamples blocks and
# configurations without knowing which it has.
#
# __len__ must report what the source offers NOW, not what it offered when it was
# written.  Ensembles grow -- continue_from and extend_h5 make that easy, and a
# production campaign does it constantly -- and a resampling drawn over the
# shorter chain describes only its beginning.  A length read back from disk cannot
# notice that, so a source which stores a length derived from another must
# re-derive it on the way back in and refuse if the two disagree, as
# StreamingBlocking.from_h5 does.  Getting this wrong does not crash; it serves a
# stale analysis without complaint, which is how it went unnoticed the first time.
class SampleSource(ReadWriteable):
    r'''
    Samples a :class:`StreamingBootstrap` can resample without their all being in
    memory: an :class:`EnsembleStreamer`, whose samples are configurations, or a
    :class:`StreamingBlocking`, whose samples are blocks of them.

    Either offers how many samples there are, the weight and index of each, and
    the measurement of an observable a chunk of samples at a time --- so the
    bootstrap resamples either without knowing which it has.
    '''

    @property
    def measured(self):
        r'''Nothing is measured ahead of time; a source measures on demand.'''
        return set()

    @property
    def available(self):
        r'''Whether the ensemble underneath can still be reached.'''
        raise NotImplementedError

    def values(self, name):
        r'''
        Measure observable ``name``, a chunk of samples at a time.

        Yields
        ------
        tuple:
            ``(start, values)``, where ``values`` holds the measurement on
            samples ``[start:start+len(values)]``.
        '''
        raise NotImplementedError

    def timeseries(self, name):
        r'''
        The measurement of observable ``name`` on every sample, as one array.

        .. warning::
           The configurations are still streamed and thrown away a chunk at a
           time; what accumulates here is the *measurement*.  So this costs one
           chunk of configurations plus the whole measurement --- one number per
           sample for a scalar, which is why :meth:`autocorrelation_time` can
           afford it, but a number per site per sample for a correlator, which is
           comparable to the ensemble and hands back the memory that streaming
           just saved.
        '''
        # Filled in place rather than concatenated from a list of chunks, which
        # would hold every chunk and the joined result at the same time --- twice
        # the peak, for the one case where the peak is worth caring about.
        chunks = self.values(name)
        try:
            start, first = next(chunks)
        except StopIteration:
            raise RuntimeError(
                f'{name} has no samples to measure; this source offers none.') from None

        measurement = np.empty((len(self),) + first.shape[1:], dtype=first.dtype)
        measurement[start:start + len(first)] = first
        for start, values in chunks:
            measurement[start:start + len(values)] = values
        return measurement

    def autocorrelation_time(self, observables=None, every=False):
        r'''
        As :meth:`.Ensemble.autocorrelation_time`, but measuring as it streams:
        the autocorrelation time of *these* samples, which for a
        :class:`StreamingBlocking` means of the blocks rather than of the
        configurations underneath.  That is the number blocking is meant to bring
        down, and how you tell whether the width was wide enough.

        Only observables that opt in via :meth:`.Observable.autocorrelation` are
        considered, and those are scalars, so this costs one number per sample per
        observable however large the lattice.

        .. note::
           An :class:`~.Ensemble` considers the observables it has already
           measured; a source has measured nothing, so it considers them all.
        '''
        return sample_autocorrelation_time(self, observables=observables, every=every)


class EnsembleStreamer(SampleSource):
    r'''
    Hands a stored :class:`~.Ensemble` out a few configurations at a time, so that
    an ensemble too large to read can still be analyzed.

    Making one is cheap.  It reads only the :attr:`Action` and, one number per
    configuration, the :attr:`weight` and :attr:`index`; the configurations
    themselves stay on disk until :meth:`chunks` asks for them.  Each chunk comes back
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
    chunk: int
        The maximum number of configurations to hold in memory at once.
    start: int
        How many leading configurations to skip.  Prefer :meth:`cut`, which says
        what you mean and composes.
    stride: int
        Present one configuration in every ``stride``.  Prefer :meth:`every`.
    '''

    def __init__(self, source_group, chunk=64, start=0, stride=1):
        if chunk < 1:
            # range(0, length, chunk) is empty for a non-positive stride, so
            # chunks() would quietly yield nothing and a resample would come back
            # with no configurations in it rather than an error.
            raise ValueError(f'chunk should be at least 1 configuration, not {chunk}.')
        if stride < 1:
            raise ValueError(f'stride should be at least 1, not {stride}.')
        if start < 0:
            raise ValueError(f'start should not be negative, but is {start}.')

        self._source = source_group
        self.chunk = chunk
        r'''The maximum number of configurations held in memory at once.'''
        self.start = start
        r'''How many leading configurations of the stored ensemble are skipped.'''
        self.stride = stride
        r'''One configuration in every ``stride`` is presented.'''
        self.Action = Data.read(source_group['Action'])
        r'''The action underlying the ensemble.'''

        self.weight = _stream_weight(source_group)[start::stride]
        r'''The importance weight of each configuration.'''
        self.index = _stream_index(source_group)[start::stride]
        r'''The Markov-chain index of each configuration.'''
        self.index_stride = _stream_index_stride(source_group) * stride
        r'''The distance in the Markov chain between one configuration and the next.'''

        self._length = len(np.asarray(self.weight))

    def __len__(self):
        return self._length

    @property
    def available(self):
        r'''Whether the ensemble this streamer presents can still be reached.'''
        return self._source is not None

    def _require_source(self):
        if self._source is None:
            raise RuntimeError(
                'EnsembleStreamer source ensemble unavailable (broken h5 link); '
                'there is no ensemble to present.')

    def cut(self, start):
        r'''
        Drop the first ``start`` configurations, as :meth:`.Ensemble.cut` does.
        Nothing is read or copied; the result is another streamer presenting fewer
        configurations of the same stored ensemble.

        Returns
        -------
        EnsembleStreamer
        '''
        self._require_source()
        return EnsembleStreamer(self._source, chunk=self.chunk,
                                start=self.start + start * self.stride,
                                stride=self.stride)

    def every(self, stride):
        r'''
        Keep one configuration in every ``stride``, as :meth:`.Ensemble.every`
        does.  Nothing is read or copied.

        .. seealso::
           :class:`StreamingBlocking`, which averages consecutive configurations
           rather than discarding them --- usually the better trade, and much the
           better one when a rare configuration carries a lot of weight.

        Returns
        -------
        EnsembleStreamer
        '''
        self._require_source()
        return EnsembleStreamer(self._source, chunk=self.chunk,
                                start=self.start, stride=self.stride * stride)

    def values(self, name):
        r'''
        Measure observable ``name``, a chunk of configurations at a time.

        This is what a :class:`StreamingBootstrap` resamples.  A
        :class:`StreamingBlocking` provides the same thing, but its samples are
        blocks rather than configurations --- which is the whole reason this is
        the contract rather than :meth:`chunks`.

        Yields
        ------
        tuple:
            ``(start, values)``, where ``values`` holds the measurement on
            samples ``[start:start+len(values)]``.
        '''
        for start, sub in self.chunks():
            yield start, Batch.as_array(getattr(sub, name))

    def chunks(self):
        r'''
        Go through the ensemble in order, at most :attr:`chunk` configurations at
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
        self._require_source()
        fields = self._source['configuration/fields']
        for start in range(0, self._length, self.chunk):
            stop = min(start + self.chunk, self._length)
            # Sample i is configuration self.start + i*self.stride of the file.
            first = self.start + start * self.stride
            last  = self.start + (stop - 1) * self.stride + 1
            chunk_fields = {name: _read_batch_chunk(fields[name], first, last, self.stride)
                            for name in fields}
            cfgs = Configurations(chunk_fields)
            sub = supervillain.ensemble.Ensemble(self.Action).from_configurations(cfgs)
            yield start, sub

    def to_h5(self, group, _top=True):
        r'''
        Write the streamer as its :attr:`chunk`, :attr:`start` and :attr:`stride`,
        and an ``ensemble`` link to the ensemble it presents --- an
        :class:`h5py.SoftLink` if that lives in the same file as ``group``, an
        :class:`h5py.ExternalLink` if it does not.  No configuration is copied.
        '''
        group.attrs['chunk'] = self.chunk
        group.attrs['start'] = self.start
        group.attrs['stride'] = self.stride
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
            group['ensemble'] = h5py.SoftLink(source_path)
        else:
            group['ensemble'] = h5py.ExternalLink(source_file, source_path)

    @classmethod
    def from_h5(cls, group, strict=True, _top=True):
        r'''
        Follow the stored link back to the source ensemble and re-read its cheap
        metadata.  If the link cannot be resolved the streamer still reads back,
        but with :attr:`Action` and :attr:`weight` set to ``None`` and
        :meth:`chunks` unavailable.
        '''
        o = cls.__new__(cls)
        o.chunk = int(group.attrs['chunk'])
        o.start = int(group.attrs['start'])
        o.stride = int(group.attrs['stride'])
        try:
            source = group['ensemble']
            o._source = source
            o.Action = Data.read(source['Action'])
            o.weight = _stream_weight(source)[o.start::o.stride]
            o.index = _stream_index(source)[o.start::o.stride]
            o.index_stride = _stream_index_stride(source) * o.stride
            o._length = len(np.asarray(o.weight))
        except (KeyError, OSError):
            o._source = None
            o.Action = None
            o.weight = None
            o.index = None
            o.index_stride = None
            o._length = None
        return o


class StreamingBlocking(SampleSource):
    r'''
    Averages consecutive samples of an :class:`EnsembleStreamer` together, without
    ever holding the unaveraged measurements.

    :class:`~.Blocking` does the same thing to an :class:`~.Ensemble` it already
    has in memory, and gives identical numbers.  When the ensemble is too large
    for that, this does it as the
    measurements stream past, so neither the ensemble nor its unblocked
    timeseries is ever assembled.  What comes out is a source of samples ---
    blocks --- that a :class:`StreamingBootstrap` resamples exactly as it would
    resample configurations.

    Blocking is the right way to handle autocorrelation when rare configurations
    matter.  :meth:`~.EnsembleStreamer.every` throws configurations away, and near
    a phase transition the one it throws away may be the one that carried the
    signal; blocking averages that configuration in instead.

    .. note::
       A block is not a configuration, so this deliberately offers no field access
       and no :meth:`~.EnsembleStreamer.chunks`.  The average of some
       configurations need not even *be* one: the :class:`~.Villain` $n$ and the
       :class:`~.Worldline` $m$ and $v$ are integers, and the mean of some
       integers is not an integer, so the average does not live in the field space
       at all.  Where the average does have the right type it is still the wrong
       thing to measure on --- observables have to be measured on configurations
       and *then* averaged, since measuring an average is not averaging a
       measurement for anything nonlinear.

    .. note::
       A block cannot be handed out before all of its samples have been measured,
       so this holds up to ``width`` measurements at once --- or the source's
       chunk, whichever is larger.  With a width of a few autocorrelation times
       that is nothing; if you block very wide, size the two together.

    .. seealso::
       :class:`~.Blocking`, for an ensemble that fits in memory.

    Parameters
    ----------
    source: EnsembleStreamer
        The samples to average together.  Cut and decimate it first, if you mean
        to; blocking is the last step.
    width: int or 'auto'
        How many samples go into each block; if ``'auto'``, the source's
        :meth:`~.EnsembleStreamer.autocorrelation_time`.
    '''

    def __init__(self, source, width='auto'):
        self.source = source
        r'''The samples being averaged together.'''
        self.width = source.autocorrelation_time() if width == 'auto' else width
        r'''The number of samples in each block.'''
        if self.width < 1:
            raise ValueError(f'width should be at least 1 sample, not {self.width}.')

        samples = len(source)
        self.drop = samples % self.width
        r'''How many leading samples are dropped so that the blocking comes out evenly.'''
        self.blocks = (samples - self.drop) // self.width
        r'''How many blocks there are.'''
        if self.blocks < 1:
            raise ValueError(
                f'{samples} samples do not fill even one block of {self.width}.')

        self.Action = source.Action
        r'''The action underlying the ensemble.'''
        self.weight = np.asarray(source.weight)[self.drop:].reshape(-1, self.width).mean(axis=1)
        r'''The average importance weight of each block.'''
        self.index = np.asarray(source.index)[self.drop:].reshape(-1, self.width).mean(axis=1)
        r'''The average Markov-chain index of each block.'''
        self.index_stride = source.index_stride * self.width
        r'''The distance in the Markov chain between one block and the next.'''

    def __len__(self):
        return self.blocks

    @property
    def available(self):
        r'''Whether the ensemble underneath can still be reached.'''
        return self.source.available

    def values(self, name):
        r'''
        Measure observable ``name`` and average it into blocks as it streams.

        Each block is $\left\langle wO\right\rangle_b / \left\langle w\right\rangle_b$,
        which pairs with the block's own average weight in :attr:`weight` so that
        an average over blocks telescopes back to the average over configurations.
        With unit weights that is the plain block mean, and a streamed blocking and
        an in-memory one give identical numbers.

        Yields
        ------
        tuple:
            ``(start, values)``, where ``values`` holds the blocked measurement
            on blocks ``[start:start+len(values)]``.
        '''
        # Divide each block by its own average weight, so that
        #     sum_b <w>_b (<wO>_b / <w>_b) / sum_b <w>_b  =  <wO> / <w>
        # and a Bootstrap of these blocks estimates what a Bootstrap of the
        # configurations would.  Blocking._block does the same, and says why at
        # more length; the two must agree, which test_blocking_matches_in_memory
        # checks.
        weight = np.asarray(self.source.weight)

        held = None       # samples read but not yet part of a whole block
        dropped = 0       # of the leading self.drop
        emitted = 0       # blocks handed out so far

        for start, values in self.source.values(name):
            values = np.asarray(values)
            w = weight[start:start + len(values)]
            values = values * w.reshape((-1,) + (1,) * (values.ndim - 1))

            if dropped < self.drop:
                take = min(self.drop - dropped, len(values))
                dropped += take
                values = values[take:]
                if len(values) == 0:
                    continue

            held = values if held is None else np.concatenate([held, values], axis=0)

            whole = len(held) // self.width
            if whole == 0:
                continue
            full, held = held[:whole * self.width], held[whole * self.width:]
            blocked = full.reshape(whole, self.width, *full.shape[1:]).mean(axis=1)
            block_weight = self.weight[emitted:emitted + whole]
            yield emitted, blocked / block_weight.reshape(
                    (-1,) + (1,) * (blocked.ndim - 1))
            emitted += whole

    @classmethod
    def from_h5(cls, group, strict=True, _top=True):
        r'''
        Read a blocking back, and check that the ensemble underneath is still the
        one it was blocked from.
        '''
        o = super().from_h5(group, strict=strict, _top=_top)

        # How many blocks there are, and how many leading samples get dropped to
        # make them come out evenly, both depend on how many samples there are.
        # Those were computed once and stored; if the ensemble has grown since,
        # they describe a shorter chain than the one now on disk, and every value
        # blocked through them would silently be a value of that shorter chain.
        if o.available:
            samples = len(o.source)
            drop = samples % o.width
            blocks = (samples - drop) // o.width
            if (drop, blocks) != (o.drop, o.blocks):
                raise ValueError(
                    f'{group.name} blocks {o.blocks} × {o.width} samples (dropping '
                    f'{o.drop}), but its source now offers {samples} samples, which '
                    f'block into {blocks} (dropping {drop}).  Block the longer '
                    'ensemble afresh rather than reusing this.')
        return o


class StreamingBootstrap(Bootstrap):
    r'''
    A :class:`Bootstrap` that resamples a :class:`SampleSource` as it hands its
    samples out, a chunk at a time, and saves each result into a target h5 group.

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
    source: SampleSource
        Hands out the samples, and carries their ``Action`` and ``weight``.  An
        :class:`EnsembleStreamer` to resample configurations, a
        :class:`StreamingBlocking` to resample blocks of them.
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

    # Names of the object's own machinery, checked before the registries so that
    # an observable sharing a name with one of them could never send the gate off
    # to resample the machinery itself.  No such collision exists today --- so this
    # check changes no outcome, and test_machinery_names_are_shielded_from_the_gate
    # is what will notice if one ever appears.
    _PASSTHROUGH = frozenset({
        'target_group', 'source', 'draws', 'indices', 'Action', '_n',
        '_resample_streaming', '_rebuild_counts', 'Ensemble', 'estimate',
    })

    def __init__(self, source, target_group, draws=100, indices=None, rng=None):
        if 'indices' in target_group:
            raise ValueError(
                f'{target_group.name} already holds a StreamingBootstrap.  Read it '
                'back with StreamingBootstrap.from_h5 to carry on with the '
                'resampling it already used; constructing a new one here would draw '
                'new indices, which would not describe the results already stored.')
        self.source = source
        r'''The :class:`SampleSource` from which to resample.'''
        self.target_group = target_group
        r'''The h5 group into which streamed quantities are written.'''
        self.Action = source.Action
        r'''The action underlying the ensemble.'''
        cfgs = len(source)
        if cfgs < 1:
            raise ValueError(
                'there is nothing to resample; the source offers no samples at all.')
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
        Data.write(target_group, 'source', self.source)

    @property
    def Ensemble(self):
        # DerivedQuantity.__get__ and the plot_* helpers reach the action through
        # .Ensemble.Action; a SampleSource carries Action, weight, and __len__.
        return self.source

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
        stored.  Numerator and denominator each accumulate a chunk at a time, so
        nothing larger than one chunk plus the (draws, ...) answer is ever in
        memory, however long the chain.  The result equals Bootstrap._resample on
        the same indices to floating point --- reassociating a sum is all that
        separates them, which is what test_streaming_equivalence checks.'''
        weight = np.asarray(self.source.weight)
        n = self._n
        draws = self.draws
        numerator = None
        denominator = np.zeros(draws)
        for start, obs in self.source.values(name):
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
        # Reached only when __getattribute__ has raised AttributeError, which for
        # a gated name it never does --- so nothing that could be streamed arrives
        # here and there is nothing to do but raise.
        #
        # It has to be said explicitly, though, because Bootstrap.__getattr__
        # would otherwise look the name up on .Ensemble --- which is the streamer
        # --- and resample whatever it found: asking a StreamingBootstrap for
        # .chunk would fetch the streamer's chunk size and try to bootstrap the
        # integer 64.
        raise AttributeError(name)

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
        # the source (via EnsembleStreamer.from_h5, which resolves the link) and
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
        if o.source.available and len(o.source) != configurations:
            raise ValueError(
                f'{group.name} resamples {configurations} configurations but its '
                f'ensemble now has {len(o.source)}.  Every result stored here '
                f'describes only the first {configurations}; bootstrap the extended '
                'ensemble into a new group rather than adding to this one.')
        return o
