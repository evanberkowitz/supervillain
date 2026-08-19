#!/usr/bin/env python
r"""Tests for the streaming bootstrap: EnsembleStreamer (memory-bounded chunked
iteration of a serialized Ensemble) and StreamingBootstrap (chunk-accumulated
resample with a write-through disk cache).

Two tests are load-bearing.  `test_streaming_equivalence` is the correctness
claim: given the SAME resampling indices, the streaming estimate must equal a
plain Bootstrap's to floating point.
`test_refuses_to_serve_a_bootstrap_its_ensemble_has_outgrown` is the safety
claim: a resampling is drawn over a fixed number of configurations, and once
continue_from and extend_h5 have grown the ensemble past that, every stored
result describes a prefix of what is on disk --- the one way this class could be
quietly and plausibly wrong.

The rest guard properties that would otherwise fail silently: that chunks
reassemble the ensemble exactly, that the h5 link survives a round trip within a
file and across two, that cached quantities are served without touching the
source, that a target which cannot be written says so before streaming rather
than after, and that the importance weight is normalized by ONE global maximum.
"""

import numpy as np
import h5py as h5
import pytest

import supervillain
import supervillain.h5
from supervillain.batch import Batch
from supervillain.analysis import Bootstrap, Blocking
from supervillain.analysis.streaming import (
        EnsembleStreamer, StreamingBlocking, StreamingBootstrap, _stream_weight)
import generate

# Small and cheap: these tests check bookkeeping, not physics, so the ensemble
# only has to be a genuine one --- it does not have to be a good one.
N = 4
KAPPA = 0.1
CONFIGURATIONS = 40

# A scalar, a correlator, and a derived quantity built from primaries; between
# them they exercise every shape the streaming accumulator has to handle.
QUANTITIES = ('ActionDensity', 'Spin_Spin', 'InternalEnergyDensityVariance')


def villain_h5(tmp_path, configurations=CONFIGURATIONS, file='streaming.h5'):
    r'''Generate a 2D Villain ensemble and store it at ``/ensemble``.  Returns
    the path to the file.'''
    e = generate.villain(configurations, N, KAPPA)

    path = tmp_path / file
    with h5.File(path, 'w') as f:
        e.to_h5(f.create_group('ensemble'))

    return path


def test_streamer_fidelity(tmp_path):
    r'''The chunks a streamer yields must reassemble into the whole ensemble ---
    both the stored configuration fields and an observable computed from them.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r') as f:
        full = supervillain.Ensemble.from_h5(f['ensemble'])
        streamer = EnsembleStreamer(f['ensemble'], chunk=7)

        assert len(streamer) == len(full)

        # A chunk boundary that divides the ensemble evenly would not notice an
        # off-by-one in the final chunk; 7 does not divide 40.
        assert len(full) % streamer.chunk != 0

        # Stored fields, a chunk at a time.
        for field in ('phi', 'n'):
            chunks = [np.asarray(getattr(sub, field)) for _, sub in streamer.chunks()]
            assert np.array_equal(np.concatenate(chunks, axis=0),
                                  np.asarray(getattr(full, field)))

        # Starts must tile the ensemble contiguously from zero.
        starts = [start for start, _ in streamer.chunks()]
        assert starts == list(range(0, len(full), streamer.chunk))

        # An observable computed (not stored inline) from the sliced fields.
        chunks = [np.asarray(sub.Spin_Spin) for _, sub in streamer.chunks()]
        assert np.allclose(np.concatenate(chunks, axis=0), np.asarray(full.Spin_Spin))


def test_streaming_equivalence(tmp_path):
    r'''On the same resampling indices the streamed estimate must equal the
    in-memory one to floating point.  This is the whole point of the class.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        full = supervillain.Ensemble.from_h5(f['ensemble'])
        reference = Bootstrap(full, draws=50)

        streamer = EnsembleStreamer(f['ensemble'], chunk=9)
        streaming = StreamingBootstrap(
                streamer, f.create_group('bootstrap'), indices=reference.indices)

        assert streaming.draws == reference.draws

        for quantity in QUANTITIES:
            mean, error = (np.asarray(_) for _ in reference.estimate(quantity))
            streamed_mean, streamed_error = (
                    np.asarray(_) for _ in streaming.estimate(quantity))

            assert np.allclose(mean, streamed_mean, atol=1e-10, rtol=1e-8), quantity
            assert np.allclose(error, streamed_error, atol=1e-10, rtol=1e-8), quantity


def test_indices_set_the_draws(tmp_path):
    r'''Handing over indices should set the number of draws, and indices of the
    wrong shape should be refused rather than silently mis-resampled.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)
        configurations = len(streamer)

        indices = np.random.randint(0, configurations, (configurations, 13))
        streaming = StreamingBootstrap(
                streamer, f.create_group('bootstrap'), draws=100, indices=indices)
        assert streaming.draws == 13
        assert np.asarray(streaming.ActionDensity).shape == (13,)

        with pytest.raises(ValueError):
            StreamingBootstrap(streamer, f.create_group('malformed'),
                               indices=indices[:-1])


def test_write_through_roundtrip_and_portability(tmp_path):
    r'''Accessing a quantity is what stores it; the stored group is a plain
    Bootstrap layout that reads back with or without the source ensemble.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)
        streaming = StreamingBootstrap(streamer, f.create_group('bootstrap'), draws=40)

        estimates = {q: streaming.estimate(q) for q in QUANTITIES}
        for quantity in QUANTITIES:
            assert quantity in f['bootstrap']

    # Reload the streaming bootstrap: the cached values come back and the
    # streamer is rebuilt from the stored link.
    with h5.File(path, 'r+') as f:
        streaming = StreamingBootstrap.from_h5(f['bootstrap'])
        assert isinstance(streaming.streamer, EnsembleStreamer)

        for quantity in QUANTITIES:
            mean, _ = streaming.estimate(quantity)
            assert np.allclose(np.asarray(mean), np.asarray(estimates[quantity][0]))

        # A derived quantity persists the primaries it is built from, so they
        # need not be re-streamed to recompute or to reuse it.
        assert 'InternalEnergyDensity' in f['bootstrap']
        assert 'InternalEnergyDensitySquared' in f['bootstrap']

        # A quantity that was not streamed before now streams through the
        # reconstructed streamer, and is itself persisted.
        assert 'Vortex_Vortex' not in f['bootstrap']
        mean, _ = streaming.estimate('Vortex_Vortex')
        assert np.isfinite(np.asarray(mean)).all()
        assert 'Vortex_Vortex' in f['bootstrap']

    # A plain Bootstrap reads the streamed results, so they are usable on a
    # machine that never sees the ensemble.
    with h5.File(path, 'r') as f:
        plain = Bootstrap.from_h5(f['bootstrap'])
        for quantity in QUANTITIES:
            mean, error = plain.estimate(quantity)
            assert np.allclose(np.asarray(mean), np.asarray(estimates[quantity][0]))
            assert np.allclose(np.asarray(error), np.asarray(estimates[quantity][1]))


def test_resumability_no_recompute(tmp_path):
    r'''A cached quantity must be served from disk without touching the source;
    an uncached one must honestly fail when the source is gone.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)
        streaming = StreamingBootstrap(streamer, f.create_group('bootstrap'), draws=30)
        mean, error = streaming.estimate('ActionDensity')

        # A streamed quantity is held in memory, so touching it again neither
        # recomputes it nor goes back to disk for it.
        assert streaming.ActionDensity is streaming.ActionDensity

        # Break the streamer, so that any recomputation would raise.
        streaming.streamer._source = None

        cached_mean, cached_error = streaming.estimate('ActionDensity')
        assert np.allclose(np.asarray(cached_mean), np.asarray(mean))
        assert np.allclose(np.asarray(cached_error), np.asarray(error))

        with pytest.raises(RuntimeError):
            streaming.estimate('Spin_Spin')

        # Nor can a streamer with no source be re-serialized: there is nothing
        # left to link to, and a link to nothing would fail silently later.
        with pytest.raises(RuntimeError):
            streaming.streamer.to_h5(f.create_group('relink'))


def test_extended_ensemble_streams(tmp_path):
    r'''An ensemble grown on disk by continue_from + extend_h5 streams as one
    ensemble: the streamer must see every configuration, including those added
    after it was first written.'''
    path = tmp_path / 'extended.h5'
    continuations = 2

    e = generate.villain(CONFIGURATIONS, N, KAPPA)
    with h5.File(path, 'w') as f:
        e.to_h5(f.create_group('ensemble'))

    for _ in range(continuations):
        with h5.File(path, 'r+') as f:
            supervillain.Ensemble.continue_from(
                    f['ensemble'], CONFIGURATIONS).extend_h5(f['ensemble'])

    with h5.File(path, 'r+') as f:
        full = supervillain.Ensemble.from_h5(f['ensemble'])
        assert len(full) == (1 + continuations) * CONFIGURATIONS

        streamer = EnsembleStreamer(f['ensemble'], chunk=17)
        assert len(streamer) == len(full)

        chunks = [np.asarray(sub.phi) for _, sub in streamer.chunks()]
        assert np.array_equal(np.concatenate(chunks, axis=0), np.asarray(full.phi))

        reference = Bootstrap(full, draws=30)
        streaming = StreamingBootstrap(
                streamer, f.create_group('bootstrap'), indices=reference.indices)

        mean, error = (np.asarray(_) for _ in reference.estimate('ActionDensity'))
        streamed_mean, streamed_error = (
                np.asarray(_) for _ in streaming.estimate('ActionDensity'))
        assert np.allclose(mean, streamed_mean, atol=1e-10, rtol=1e-8)
        assert np.allclose(error, streamed_error, atol=1e-10, rtol=1e-8)


def _weighted_group(f, logWeights):
    r'''Fabricate the part of a stored ensemble that _stream_weight reads: the
    scalar ``data`` datasets of inline logWeight_ columns.'''
    fields = f.create_group('ensemble/configuration/fields')
    for name, column in logWeights.items():
        fields.create_dataset(f'logWeight_{name}/data', data=np.asarray(column))
    return f['ensemble']


def test_stream_weight_logs_sum_with_one_global_max():
    r'''Several generators each contribute a logWeight_ column; the logs must sum
    (so the weights multiply) and be exponentiated after subtracting a SINGLE
    global maximum.

    The global maximum is a correctness requirement, not overflow safety: the
    streaming resample accumulates <Ow> and <w> separately across chunks, and the
    exp offset cancels between them only if every configuration shares it.  A
    per-chunk --- or per-column --- maximum would silently bias the estimate.
    '''
    a = np.array([0.0, 1.0, -2.0, 5.0, 3.0, -1.0, 0.5, 2.0])
    b = np.array([1.0, -3.0, 4.0, 0.0, 2.0, 1.5, -0.5, 1.0])

    with h5.File('weights.h5', 'w', driver='core', backing_store=False) as f:
        weight = _stream_weight(_weighted_group(f, {'a': a, 'b': b}))

    logWeight = a + b
    assert np.allclose(weight, np.exp(logWeight - logWeight.max()))

    # The maximum is global: exactly one configuration sits at weight 1, and no
    # configuration exceeds it.
    assert weight.max() == pytest.approx(1.)
    assert (weight <= 1.).all()

    # Only the ratio is physical, so the estimator must not care about an overall
    # rescaling of the weights --- which is what the offset amounts to.
    observable = np.array([3., 1., 4., 1., 5., 9., 2., 6.])
    unnormalized = np.exp(logWeight)
    assert ((weight * observable).sum() / weight.sum()
            == pytest.approx((unnormalized * observable).sum() / unnormalized.sum()))


def test_stream_weight_prefers_logs_then_stored_then_unit(tmp_path):
    r'''The three sources of the weight, in order.'''
    logWeights = np.array([0.0, 1.0, -2.0, 5.0])

    # Inline logWeight_ columns win over an explicitly stored weight.
    with h5.File('precedence.h5', 'w', driver='core', backing_store=False) as f:
        group = _weighted_group(f, {'a': logWeights})
        supervillain.h5.Data.write(group, 'weight', Batch(np.full(4, 7.)))
        assert np.allclose(_stream_weight(group),
                           np.exp(logWeights - logWeights.max()))

    # With no logWeight_ columns, a stored weight is read.
    with h5.File('stored.h5', 'w', driver='core', backing_store=False) as f:
        group = _weighted_group(f, {})
        f.create_dataset('ensemble/configuration/fields/phi/data', data=np.zeros((4, 3, 3)))
        supervillain.h5.Data.write(group, 'weight', Batch(np.arange(4.)))
        assert np.allclose(_stream_weight(group), np.arange(4.))

    # With neither, every configuration weighs the same.
    with h5.File('unit.h5', 'w', driver='core', backing_store=False) as f:
        group = _weighted_group(f, {})
        f.create_dataset('ensemble/configuration/fields/phi/data', data=np.zeros((4, 3, 3)))
        assert np.allclose(_stream_weight(group), np.ones(4))

    # And a real, unweighted ensemble comes out at unit weight.
    path = villain_h5(tmp_path)
    with h5.File(path, 'r') as f:
        assert np.allclose(EnsembleStreamer(f['ensemble']).weight, 1.)


def test_public_exports():
    from supervillain.analysis import (
            StreamingBootstrap as exported_bootstrap,
            EnsembleStreamer as exported_streamer)

    assert exported_bootstrap is StreamingBootstrap
    assert exported_streamer is EnsembleStreamer


def test_chunk_must_tile_the_ensemble(tmp_path):
    r'''A non-positive chunk makes ``range(0, length, chunk)`` empty, so chunks()
    would quietly hand back no configurations at all rather than complain.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r') as f:
        for chunk in (0, -1, -9):
            with pytest.raises(ValueError):
                EnsembleStreamer(f['ensemble'], chunk=chunk)


def test_refuses_to_reuse_a_populated_target(tmp_path):
    r'''Constructing a second bootstrap over a group that already holds one would
    draw fresh indices, which do not describe the results already stored there.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)
        target = f.create_group('bootstrap')
        StreamingBootstrap(streamer, target, draws=20).estimate('ActionDensity')

        with pytest.raises(ValueError):
            StreamingBootstrap(streamer, target, draws=20)


def test_refuses_to_serve_a_bootstrap_its_ensemble_has_outgrown(tmp_path):
    r'''The load-bearing safety test.  An ensemble grown by continue_from and
    extend_h5 no longer matches indices drawn over the shorter chain, so every
    result already stored describes a prefix of what is now on disk.  Serving
    those as if they described the whole ensemble is the one way this class can be
    quietly and plausibly wrong.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)
        streaming = StreamingBootstrap(streamer, f.create_group('bootstrap'), draws=20)
        stale, _ = streaming.estimate('ActionDensity')

    with h5.File(path, 'r+') as f:
        supervillain.Ensemble.continue_from(
                f['ensemble'], CONFIGURATIONS).extend_h5(f['ensemble'])

    with h5.File(path, 'r+') as f:
        # The stale estimate is a perfectly good bootstrap of the first half, and
        # differs from the truth by far more than roundoff --- which is exactly why
        # it must not come back unannounced.
        whole = supervillain.Ensemble.from_h5(f['ensemble'])
        assert len(whole) == 2 * CONFIGURATIONS
        truth, error = Bootstrap(whole, draws=20).estimate('ActionDensity')
        assert abs(float(stale) - float(truth)) > 1e-6

        with pytest.raises(ValueError):
            StreamingBootstrap.from_h5(f['bootstrap'])


def test_read_only_target_serves_cached_and_refuses_fresh(tmp_path):
    r'''Results already stored can be read from a read-only file; one that would
    have to be saved must say so, and say so before streaming for it.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)
        streaming = StreamingBootstrap(streamer, f.create_group('bootstrap'), draws=20)
        mean, _ = streaming.estimate('ActionDensity')

    with h5.File(path, 'r') as f:
        streaming = StreamingBootstrap.from_h5(f['bootstrap'])
        cached, _ = streaming.estimate('ActionDensity')
        assert np.allclose(np.asarray(cached), np.asarray(mean))

        with pytest.raises(RuntimeError):
            streaming.estimate('WindingSquared')


def test_streams_across_files(tmp_path):
    r'''A streamer whose source is in another file serializes as an external link,
    so results can be written somewhere other than alongside the ensemble --- and
    still find their way back to it.'''
    source = villain_h5(tmp_path, file='source.h5')
    target = tmp_path / 'target.h5'

    with h5.File(source, 'r') as s, h5.File(target, 'w') as t:
        streamer = EnsembleStreamer(s['ensemble'], chunk=9)
        streaming = StreamingBootstrap(streamer, t.create_group('bootstrap'), draws=20)
        mean, _ = streaming.estimate('ActionDensity')

    # Nothing of the ensemble was copied; only a link to it.
    with h5.File(target, 'r') as t:
        link = t['bootstrap/streamer'].get('source', getlink=True)
        assert isinstance(link, h5.ExternalLink)
        assert 'configuration' not in t['bootstrap/streamer']

    # Reopened on its own, the target follows that link back to the ensemble and
    # streams a quantity it did not have before.
    with h5.File(target, 'r+') as t:
        streaming = StreamingBootstrap.from_h5(t['bootstrap'])
        cached, _ = streaming.estimate('ActionDensity')
        assert np.allclose(np.asarray(cached), np.asarray(mean))

        fresh, _ = streaming.estimate('WindingSquared')
        assert np.isfinite(np.asarray(fresh)).all()


def test_only_batch_valued_fields_stream(tmp_path):
    r'''EnsembleStreamer slices stored Batches.  A configuration field stored some
    other way cannot be sliced, and must say so rather than be skipped.'''
    from supervillain.analysis.streaming import _read_batch_chunk

    path = villain_h5(tmp_path)
    with h5.File(path, 'r+') as f:
        impostor = f['ensemble/configuration/fields'].create_group('NotABatch')
        impostor.create_dataset('data', data=np.zeros(CONFIGURATIONS))

        with pytest.raises(ValueError):
            _read_batch_chunk(impostor, 0, 4)


# The analysis pipeline: cut, every, autocorrelation_time, and blocking.  Each of
# these already works on an in-memory Ensemble, so every test here is a parity
# test --- the streamed answer must be the in-memory answer.

VIEWS = (
        ('raw',              lambda e: e,                      lambda s: s),
        ('cut',              lambda e: e.cut(30),              lambda s: s.cut(30)),
        ('every',            lambda e: e.every(3),             lambda s: s.every(3)),
        ('cut.every',        lambda e: e.cut(30).every(3),     lambda s: s.cut(30).every(3)),
        ('every.cut',        lambda e: e.every(3).cut(7),      lambda s: s.every(3).cut(7)),
        )


@pytest.mark.parametrize('label,of_ensemble,of_streamer', VIEWS, ids=[v[0] for v in VIEWS])
def test_cut_and_every_match_the_ensemble(tmp_path, label, of_ensemble, of_streamer):
    r'''cut and every are index arithmetic, so a streamer can do them without
    reading anything --- but it has to land on exactly the configurations the
    Ensemble would, in either order of composition.'''
    path = villain_h5(tmp_path, configurations=120)

    with h5.File(path, 'r') as f:
        memory = of_ensemble(supervillain.Ensemble.from_h5(f['ensemble']))
        streamed = of_streamer(EnsembleStreamer(f['ensemble'], chunk=17))

        assert len(streamed) == len(memory)
        assert streamed.index_stride == memory.index_stride
        assert np.array_equal(np.asarray(streamed.index), np.asarray(memory.index))
        assert np.allclose(np.asarray(streamed.weight), np.asarray(memory.weight))

        # A scalar and a correlator, to be sure the strided read is right for
        # both the shape that fits in memory and the shape that does not.
        for quantity in ('ActionDensity', 'Spin_Spin'):
            assert np.allclose(streamed.timeseries(quantity),
                               np.asarray(getattr(memory, quantity))), quantity


def test_autocorrelation_time_matches(tmp_path):
    r'''The streamer measures the scalars as it goes; the answer must be the one
    the ensemble in memory gives.'''
    path = villain_h5(tmp_path, configurations=120)

    with h5.File(path, 'r') as f:
        memory = supervillain.Ensemble.from_h5(f['ensemble'])
        streamed = EnsembleStreamer(f['ensemble'], chunk=17)

        assert streamed.autocorrelation_time() == memory.autocorrelation_time()

        per_observable = streamed.autocorrelation_time(every=True)
        assert per_observable == memory.autocorrelation_time(every=True)
        # Only scalars opt in, which is what keeps this affordable.
        assert per_observable
        assert all(np.asarray(getattr(memory, o)).ndim == 1 for o in per_observable)


@pytest.mark.parametrize('width', (2, 5, 7, 16))
def test_blocking_matches_in_memory(tmp_path, width):
    r'''StreamingBlocking must reproduce Blocking exactly --- the same number of
    blocks, the same configurations dropped from the front to make them come out
    evenly, and the same averaged values.  Width 7 does not divide 120, so the
    drop is exercised.'''
    path = villain_h5(tmp_path, configurations=120)

    with h5.File(path, 'r') as f:
        memory = Blocking(supervillain.Ensemble.from_h5(f['ensemble']), width=width)
        streamed = StreamingBlocking(EnsembleStreamer(f['ensemble'], chunk=17), width=width)

        assert len(streamed) == len(memory)
        assert streamed.drop == memory.drop
        assert streamed.index_stride == memory.index_stride
        assert np.allclose(streamed.weight, np.asarray(memory.weight))
        assert np.allclose(streamed.index, np.asarray(memory.index))

        for quantity in ('ActionDensity', 'Spin_Spin'):
            assert np.allclose(streamed.timeseries(quantity),
                               np.asarray(getattr(memory, quantity))), quantity


def test_blocked_pipeline_equivalence(tmp_path):
    r'''The load-bearing test for the pipeline, and the counterpart of
    test_streaming_equivalence.  Thermalize, decorrelate, block, bootstrap --- in
    memory and on disk, on the same resampling --- and the two must agree to
    floating point.'''
    path = villain_h5(tmp_path, configurations=120)

    with h5.File(path, 'r+') as f:
        memory = Blocking(
                supervillain.Ensemble.from_h5(f['ensemble']).cut(20).every(2), width=5)
        streamed = StreamingBlocking(
                EnsembleStreamer(f['ensemble'], chunk=17).cut(20).every(2), width=5)

        reference = Bootstrap(memory, draws=40)
        streaming = StreamingBootstrap(
                streamed, f.create_group('bootstrap'), indices=reference.indices)

        for quantity in QUANTITIES:
            mean, error = (np.asarray(_) for _ in reference.estimate(quantity))
            blocked_mean, blocked_error = (
                    np.asarray(_) for _ in streaming.estimate(quantity))

            assert np.allclose(mean, blocked_mean, atol=1e-10, rtol=1e-8), quantity
            assert np.allclose(error, blocked_error, atol=1e-10, rtol=1e-8), quantity


def test_a_block_is_not_a_configuration(tmp_path):
    r'''A blocking offers no configurations at all --- rather than offer the
    unblocked ones, whose count would not even match its own length.

    There is no blocked configuration to offer instead.  The Villain n and the
    Worldline m and v are integers, so their average is not even of the right
    type, let alone a configuration of the model; and where the type does survive,
    measuring an average is still not averaging a measurement.
    '''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r') as f:
        streamed = StreamingBlocking(EnsembleStreamer(f['ensemble'], chunk=9), width=4)

        assert not hasattr(streamed, 'chunks')
        assert not hasattr(streamed, 'phi')
        assert not hasattr(streamed, 'configuration')

        # It is a source of samples, though, and its samples are blocks.
        assert len(streamed) == CONFIGURATIONS // 4
        assert streamed.timeseries('ActionDensity').shape == (CONFIGURATIONS // 4,)


def test_blocking_width_auto(tmp_path):
    r'''width='auto' asks the source for its autocorrelation time, which a
    streamer can now answer.'''
    path = villain_h5(tmp_path, configurations=120)

    with h5.File(path, 'r') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=17)
        assert StreamingBlocking(streamer).width == streamer.autocorrelation_time()


def test_views_and_blocking_reject_nonsense(tmp_path):
    path = villain_h5(tmp_path)

    with h5.File(path, 'r') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)

        for stride in (0, -1):
            with pytest.raises(ValueError):
                streamer.every(stride)
        with pytest.raises(ValueError):
            EnsembleStreamer(f['ensemble'], start=-1)
        with pytest.raises(ValueError):
            StreamingBlocking(streamer, width=0)
        # Too few samples to fill even one block.
        with pytest.raises(ValueError):
            StreamingBlocking(streamer, width=len(streamer) + 1)


def test_a_view_survives_a_round_trip(tmp_path):
    r'''A cut, decimated, blocked source has to come back off disk as itself; a
    StreamingBootstrap that forgot its view would resample different samples.'''
    path = villain_h5(tmp_path, configurations=120)

    with h5.File(path, 'r+') as f:
        source = StreamingBlocking(
                EnsembleStreamer(f['ensemble'], chunk=17).cut(20).every(2), width=5)
        streaming = StreamingBootstrap(source, f.create_group('bootstrap'), draws=20)
        mean, _ = streaming.estimate('ActionDensity')
        samples = len(source)

    with h5.File(path, 'r+') as f:
        reloaded = StreamingBootstrap.from_h5(f['bootstrap'])
        assert isinstance(reloaded.streamer, StreamingBlocking)
        assert reloaded.streamer.width == 5
        assert reloaded.streamer.source.start == 20
        assert reloaded.streamer.source.stride == 2
        assert len(reloaded.streamer) == samples

        # And it can still stream something new through that same view.
        fresh, _ = reloaded.estimate('WindingSquared')
        assert np.isfinite(np.asarray(fresh)).all()


def test_blocking_divides_by_the_block_weight(tmp_path):
    r'''Each block must be <wO>_b / <w>_b, not <wO>_b.

    Paired with the block's own average weight, dividing is what makes an average
    over blocks telescope back to the average over configurations:

        sum_b <w>_b (<wO>_b / <w>_b) / sum_b <w>_b  =  <wO> / <w>

    Leaving it undivided lets a Bootstrap apply the weight a second time.  The
    two agree exactly at unit weight --- which is every ensemble on main, and why
    test_blocking_matches_in_memory cannot see the difference --- so this drives
    genuinely unequal weights through to tell them apart.
    '''
    path = villain_h5(tmp_path, configurations=48)
    width = 4

    with h5.File(path, 'r') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=7)

        rng = np.random.default_rng(11)
        weight = rng.uniform(0.1, 3.0, size=len(streamer))
        streamer.weight = weight

        blocking = StreamingBlocking(streamer, width=width)
        assert blocking.drop == 0

        observable = np.asarray(
                supervillain.Ensemble.from_h5(f['ensemble']).ActionDensity)

        block_weight = weight.reshape(-1, width).mean(axis=1)
        undivided = (observable * weight).reshape(-1, width).mean(axis=1)

        assert np.allclose(blocking.weight, block_weight)
        assert np.allclose(blocking.timeseries('ActionDensity'), undivided / block_weight)
        # The undivided form is what we must NOT be producing.
        assert not np.allclose(blocking.timeseries('ActionDensity'), undivided)

        # And the point of dividing: blocked and unblocked estimate the same thing.
        assert (blocking.weight * blocking.timeseries('ActionDensity')).sum() / blocking.weight.sum() \
                == pytest.approx((weight * observable).sum() / weight.sum())
