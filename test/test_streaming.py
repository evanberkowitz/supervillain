#!/usr/bin/env python
r"""Tests for supervillain.analysis.streaming: EnsembleStreamer (memory-bounded chunked
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
        assert isinstance(streaming.source, EnsembleStreamer)

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
        streaming.source._source = None

        cached_mean, cached_error = streaming.estimate('ActionDensity')
        assert np.allclose(np.asarray(cached_mean), np.asarray(mean))
        assert np.allclose(np.asarray(cached_error), np.asarray(error))

        with pytest.raises(RuntimeError):
            streaming.estimate('Spin_Spin')

        # Nor can a streamer with no source be re-serialized: there is nothing
        # left to link to, and a link to nothing would fail silently later.
        with pytest.raises(RuntimeError):
            streaming.source.to_h5(f.create_group('relink'))


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
        # /bootstrap/source is the sample source; its own 'ensemble' is the link.
        link = t['bootstrap/source'].get('ensemble', getlink=True)
        assert isinstance(link, h5.ExternalLink)
        assert 'configuration' not in t['bootstrap/source']

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

    width = 4

    with h5.File(path, 'r') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)
        streamed = StreamingBlocking(streamer, width=width)

        assert not hasattr(streamed, 'chunks')
        assert not hasattr(streamed, 'phi')
        assert not hasattr(streamed, 'configuration')

        # The premise: n really is integer-valued, so its average really is not a
        # field of the model.  If that ever stopped being true this test would be
        # arguing for a restriction that no longer had a reason.
        for _, sub in streamer.chunks():
            n = np.asarray(Batch.as_array(sub.n))
            assert np.issubdtype(n.dtype, np.integer)
            assert not np.issubdtype(n.mean(axis=0).dtype, np.integer)
            break

        # It is a source of samples, though, and its samples are blocks.
        assert len(streamed) == CONFIGURATIONS // width
        assert streamed.timeseries('ActionDensity').shape == (CONFIGURATIONS // width,)


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
        assert isinstance(reloaded.source, StreamingBlocking)
        assert reloaded.source.width == 5
        assert reloaded.source.source.start == 20
        assert reloaded.source.source.stride == 2
        assert len(reloaded.source) == samples

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


def test_a_chunk_is_a_bound_not_a_suggestion(tmp_path):
    r'''Nothing a streamer hands out may exceed its chunk --- that number is the
    whole promise, and an off-by-one in the strided read would break it silently
    on the final piece.'''
    path = villain_h5(tmp_path, configurations=120)

    with h5.File(path, 'r') as f:
        for chunk in (1, 7, 17, 119, 120, 121):
            for streamer in (EnsembleStreamer(f['ensemble'], chunk=chunk),
                             EnsembleStreamer(f['ensemble'], chunk=chunk).every(3),
                             EnsembleStreamer(f['ensemble'], chunk=chunk).cut(11)):
                sizes = [len(values) for _, values in streamer.values('ActionDensity')]
                assert sizes, chunk
                assert max(sizes) <= chunk, (chunk, sizes)
                assert sum(sizes) == len(streamer), (chunk, sizes)


@pytest.mark.parametrize('chunk,width', ((7, 30), (3, 30), (2, 40), (64, 30)))
def test_blocking_drop_spanning_several_chunks(tmp_path, chunk, width):
    r'''The samples dropped to make the blocking come out evenly can outnumber a
    whole chunk, so dropping has to survive being spread over several of them ---
    a case the widths that divide the ensemble never reach.'''
    path = villain_h5(tmp_path, configurations=100)

    with h5.File(path, 'r') as f:
        memory = Blocking(supervillain.Ensemble.from_h5(f['ensemble']), width=width)
        streamed = StreamingBlocking(EnsembleStreamer(f['ensemble'], chunk=chunk), width=width)

        assert memory.drop > chunk or chunk >= width   # the case is actually exercised
        assert streamed.drop == memory.drop
        assert len(streamed) == len(memory)
        for quantity in ('ActionDensity', 'Spin_Spin'):
            assert np.allclose(streamed.timeseries(quantity),
                               np.asarray(getattr(memory, quantity))), quantity


def test_refuses_a_blocking_its_ensemble_has_outgrown(tmp_path):
    r'''The counterpart of test_refuses_to_serve_a_bootstrap_its_ensemble_has_outgrown,
    for the blocked path --- which the bootstrap's own check cannot see.

    A blocking stores how many blocks it has and how many samples it dropped to
    make them come out evenly.  Both were computed from a length, and its length
    is what a StreamingBootstrap compares its indices against; so if the ensemble
    grows, the number of blocks does not, the bootstrap's check passes, and every
    stored result silently describes the shorter chain.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        blocking = StreamingBlocking(EnsembleStreamer(f['ensemble'], chunk=9), width=4)
        streaming = StreamingBootstrap(blocking, f.create_group('bootstrap'), draws=20)
        stale, _ = streaming.estimate('ActionDensity')
        blocks = len(blocking)

    with h5.File(path, 'r+') as f:
        supervillain.Ensemble.continue_from(
                f['ensemble'], 2 * CONFIGURATIONS).extend_h5(f['ensemble'])

    with h5.File(path, 'r+') as f:
        # The blocking would otherwise look unchanged: same stored block count,
        # so the bootstrap's own guard sees nothing wrong.
        whole = supervillain.Ensemble.from_h5(f['ensemble'])
        truth, _ = Bootstrap(Blocking(whole, width=4), draws=20).estimate('ActionDensity')
        assert abs(float(stale) - float(truth)) > 1e-6
        assert len(whole) // 4 != blocks

        with pytest.raises(ValueError):
            StreamingBootstrap.from_h5(f['bootstrap'])


def test_views_reject_a_lost_ensemble(tmp_path):
    r'''cut and every build a new streamer over the same source, so with no source
    there is nothing to build one from.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)
        streamer._source = None

        with pytest.raises(RuntimeError):
            streamer.cut(1)
        with pytest.raises(RuntimeError):
            streamer.every(2)


def test_nothing_to_resample(tmp_path):
    r'''Cutting an ensemble away entirely leaves a streamer with no samples;
    bootstrapping it should say so, not build an empty resampling that fails
    later and elsewhere.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        empty = EnsembleStreamer(f['ensemble'], chunk=9).cut(CONFIGURATIONS)
        assert len(empty) == 0

        with pytest.raises(ValueError):
            StreamingBootstrap(empty, f.create_group('bootstrap'), draws=5)


def test_a_blocking_notices_growth_smaller_than_one_block(tmp_path):
    r'''Growing an ensemble by less than a block width leaves the number of
    blocks alone and moves only the drop, so a check on the block count alone
    would wave it through.  Since samples = blocks*width + drop with drop < width,
    comparing both is comparing the sample count, and nothing slips past.'''
    path = villain_h5(tmp_path)
    width = 8

    with h5.File(path, 'r+') as f:
        blocking = StreamingBlocking(EnsembleStreamer(f['ensemble'], chunk=9), width=width)
        StreamingBootstrap(blocking, f.create_group('bootstrap'), draws=20).estimate('ActionDensity')
        blocks, drop = blocking.blocks, blocking.drop

    with h5.File(path, 'r+') as f:
        supervillain.Ensemble.continue_from(f['ensemble'], 2).extend_h5(f['ensemble'])

    with h5.File(path, 'r+') as f:
        grown = StreamingBlocking(EnsembleStreamer(f['ensemble'], chunk=9), width=width)
        assert grown.blocks == blocks      # the count alone notices nothing ...
        assert grown.drop != drop          # ... and only the drop gives it away

        with pytest.raises(ValueError):
            StreamingBootstrap.from_h5(f['bootstrap'])


def test_growth_a_strided_view_cannot_see_is_allowed(tmp_path):
    r'''The guard must not cry wolf.  extend_h5 only appends, so a strided view of
    a longer ensemble is a prefix-extension of the same view of the shorter one:
    if its length has not changed it is looking at the identical configurations,
    and a resampling drawn over them is still exactly valid.'''
    path = villain_h5(tmp_path)

    def view(f):
        return EnsembleStreamer(f['ensemble'], chunk=9).cut(1).every(2)

    with h5.File(path, 'r+') as f:
        before = np.asarray(view(f).timeseries('ActionDensity'))
        streaming = StreamingBootstrap(view(f), f.create_group('bootstrap'), draws=20)
        mean, _ = streaming.estimate('ActionDensity')

    # One more configuration, which a stride of two starting at one steps over.
    with h5.File(path, 'r+') as f:
        supervillain.Ensemble.continue_from(f['ensemble'], 1).extend_h5(f['ensemble'])

    with h5.File(path, 'r+') as f:
        assert np.array_equal(np.asarray(view(f).timeseries('ActionDensity')), before)

        reloaded = StreamingBootstrap.from_h5(f['bootstrap'])   # must not raise
        assert np.allclose(np.asarray(reloaded.estimate('ActionDensity')[0]),
                           np.asarray(mean))


def test_a_second_handle_reads_what_the_first_streamed(tmp_path):
    r'''Two bootstraps over one stored analysis: what either streams, the other
    reads off disk rather than recomputing.

    Reading a bootstrap back loads what it already has into memory, so the
    resume tests above are answered out of memory and never touch the stored copy.
    This is the path that genuinely reads it: a handle opened *before* a quantity
    existed on disk has no memory of it, and must find it there.
    '''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)
        StreamingBootstrap(streamer, f.create_group('bootstrap'),
                           draws=20).estimate('ActionDensity')

    with h5.File(path, 'r+') as f:
        first = StreamingBootstrap.from_h5(f['bootstrap'])
        second = StreamingBootstrap.from_h5(f['bootstrap'])
        assert 'WindingSquared' not in second.__dict__

        streamed, _ = first.estimate('WindingSquared')
        assert 'WindingSquared' in f['bootstrap']
        assert 'WindingSquared' not in second.__dict__

        # Break the second one's link, so it cannot possibly recompute.
        second.source._source = None
        from_disk, _ = second.estimate('WindingSquared')
        assert np.allclose(np.asarray(from_disk), np.asarray(streamed))


def test_inherited_bootstrap_methods_survive_the_gate(tmp_path):
    r'''__getattribute__ intercepts every attribute access, so the methods
    StreamingBootstrap inherits have to make it through unmolested --- they are
    not observables, and must not be routed through the disk cache.'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        whole = supervillain.Ensemble.from_h5(f['ensemble'])
        reference = Bootstrap(whole, draws=20)
        streaming = StreamingBootstrap(
                EnsembleStreamer(f['ensemble'], chunk=9),
                f.create_group('bootstrap'), indices=reference.indices)

        assert len(streaming) == reference.draws
        # plot_correlator reaches the lattice through .Ensemble.Action; a streamer
        # reads its own Action off the disk, so this is an equal lattice rather
        # than the very same object.
        assert (repr(streaming.Ensemble.Action.Lattice)
                == repr(whole.Action.Lattice))

        figure, axes = plt.subplots(1, 2)
        streaming.plot_band(axes[0], 'ActionDensity', color='C0')
        streaming.plot_correlator(axes[1], 'Spin_Spin')
        plt.close(figure)

        with pytest.raises(ValueError):
            # plot_band refuses a non-scalar, exactly as it does for a Bootstrap.
            figure, axes = plt.subplots(1, 2)
            streaming.plot_band(axes[0], 'Spin_Spin', color='C0')
            plt.close(figure)


def test_a_missing_attribute_is_an_honest_attribute_error(tmp_path):
    r'''A mistyped observable must raise, not recurse into a resample of nothing
    or quietly return some other attribute.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        streaming = StreamingBootstrap(EnsembleStreamer(f['ensemble'], chunk=9),
                                       f.create_group('bootstrap'), draws=20)

        for name in ('NoSuchObservable', 'ActionDensityy', '_not_an_internal'):
            with pytest.raises(AttributeError):
                getattr(streaming, name)


def test_rng_makes_the_resampling_reproducible(tmp_path):
    r'''The same generator seed must draw the same indices, and so give the same
    estimate; a different one must not.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=9)

        def bootstrap(group, seed):
            return StreamingBootstrap(streamer, f.create_group(group), draws=20,
                                      rng=np.random.default_rng(seed))

        one, again, other = bootstrap('one', 17), bootstrap('again', 17), bootstrap('other', 18)

        assert np.array_equal(one.indices, again.indices)
        assert not np.array_equal(one.indices, other.indices)
        assert np.allclose(np.asarray(one.estimate('ActionDensity')[0]),
                           np.asarray(again.estimate('ActionDensity')[0]))


def test_a_dangling_link_degrades_rather_than_explodes(tmp_path):
    r'''A streamer stores a link, not a copy, so the ensemble can genuinely go
    away.  What was already computed must still read back --- that is the whole
    reason results are written through --- and anything else must say plainly that
    the ensemble is gone.

    The tests above simulate this by clearing the source by hand; this one lets
    the link really dangle.
    '''
    source = villain_h5(tmp_path, file='source.h5')
    target = tmp_path / 'target.h5'

    with h5.File(source, 'r') as s, h5.File(target, 'w') as t:
        streaming = StreamingBootstrap(EnsembleStreamer(s['ensemble'], chunk=9),
                                       t.create_group('bootstrap'), draws=20)
        mean, error = streaming.estimate('ActionDensity')

    source.unlink()          # the ensemble is gone; only the link to it remains

    with h5.File(target, 'r+') as t:
        streaming = StreamingBootstrap.from_h5(t['bootstrap'])
        assert not streaming.source.available
        assert streaming.source.Action is None

        # Already computed: still perfectly readable.
        recovered, recovered_error = streaming.estimate('ActionDensity')
        assert np.allclose(np.asarray(recovered), np.asarray(mean))
        assert np.allclose(np.asarray(recovered_error), np.asarray(error))

        # Anything else needs the ensemble, and says so.
        with pytest.raises(RuntimeError):
            streaming.estimate('WindingSquared')

    # A plain Bootstrap never wanted the ensemble in the first place.
    with h5.File(target, 'r') as t:
        assert np.allclose(np.asarray(Bootstrap.from_h5(t['bootstrap']).estimate('ActionDensity')[0]),
                           np.asarray(mean))


def test_streams_an_assembled_ensemble(tmp_path):
    r'''An Ensemble put together with from_configurations, rather than generated,
    carries no weight, index, or index_stride --- only an Action and the
    configurations themselves.  A streamer has to supply the obvious defaults for
    what is missing rather than fail to read it.'''
    generated = generate.villain(CONFIGURATIONS, N, KAPPA)
    assembled = supervillain.Ensemble(generated.Action).from_configurations(
            generated.configuration)
    assert set(assembled.__dict__) == {'Action', 'configuration'}

    path = tmp_path / 'assembled.h5'
    with h5.File(path, 'w') as f:
        assembled.to_h5(f.create_group('ensemble'))

    with h5.File(path, 'r') as f:
        for stored in ('weight', 'index', 'index_stride'):
            assert stored not in f['ensemble']

        streamer = EnsembleStreamer(f['ensemble'], chunk=7)
        assert len(streamer) == CONFIGURATIONS
        assert np.allclose(streamer.weight, 1.)
        assert np.array_equal(streamer.index, np.arange(CONFIGURATIONS))
        assert streamer.index_stride == 1

        # And it measures the same as the ensemble it was assembled from.
        assert np.allclose(streamer.timeseries('ActionDensity'),
                           np.asarray(generated.ActionDensity))


def test_a_blocking_measures_its_own_autocorrelation(tmp_path):
    r'''A blocking reports the autocorrelation time of its blocks, not of the
    configurations underneath --- which is how you tell whether the width was
    wide enough.

    This checks the plumbing, not the physics: that the answer is the one you get
    by handing the blocked timeseries to autocorrelation_time yourself.  What the
    number *is* depends on the chain and is no business of a test.
    '''
    path = villain_h5(tmp_path, configurations=120)

    with h5.File(path, 'r') as f:
        blocking = StreamingBlocking(EnsembleStreamer(f['ensemble'], chunk=17), width=4)

        per_observable = blocking.autocorrelation_time(every=True)
        assert per_observable

        for name, tau in per_observable.items():
            assert tau == supervillain.analysis.autocorrelation_time(
                    blocking.timeseries(name)), name
            assert 1 <= tau <= len(blocking)

        assert blocking.autocorrelation_time() == max(per_observable.values())


def test_machinery_names_are_shielded_from_the_gate(tmp_path):
    r'''__getattribute__ routes any name the observable or derived-quantity
    registries claim through the disk cache.  If a registered name ever collided
    with something the bootstrap uses for its own purposes, the gate would try to
    resample the machinery; _PASSTHROUGH is what prevents that, and it is a hand
    written list.  Nothing collides today, so the list is doing no work --- this
    is what will fail if that changes.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        streaming = StreamingBootstrap(EnsembleStreamer(f['ensemble'], chunk=9),
                                       f.create_group('bootstrap'), draws=20)

        # The machinery proper: what __init__ set, and what the class defines
        # itself.  Not dir(), which includes every DerivedQuantity descriptor ---
        # those are registered names and being gated is exactly their purpose.
        own = set(streaming.__dict__) | set(vars(StreamingBootstrap))
        registered = set(supervillain.observables) | set(supervillain.derivedQuantities)

        assert own & registered <= StreamingBootstrap._PASSTHROUGH


def test_autocorrelation_time_never_materializes_a_correlator(tmp_path):
    r'''timeseries() accumulates the measurement --- not the configurations, which
    keep streaming past regardless, but the measurement.  For a scalar that is one
    number per sample and costs nothing; for a correlator it is a number per site
    per sample, comparable to the ensemble, and would hand back the memory that
    streaming just saved.

    autocorrelation_time is the one place the library calls it, so it must only
    ever ask for scalars.  It does, because Observable.autocorrelation is false for
    everything else --- and it filters on that even for observables named
    explicitly, which is the case worth pinning.
    '''
    path = villain_h5(tmp_path, configurations=120)

    with h5.File(path, 'r') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=17)
        whole = supervillain.Ensemble.from_h5(f['ensemble'])

        asked = []
        materialize = streamer.timeseries
        streamer.timeseries = lambda name: (asked.append(name), materialize(name))[1]

        streamer.autocorrelation_time()
        assert asked
        for name in asked:
            assert np.asarray(getattr(whole, name)).ndim == 1, name

        # Naming a correlator outright does not get it materialized either.
        asked.clear()
        streamer.autocorrelation_time(observables=['Spin_Spin', 'ActionDensity'])
        assert asked == ['ActionDensity']

        # And the configurations really are streamed, whatever is measured.
        alive = []
        chunks = streamer.chunks
        def watched():
            for start, sub in chunks():
                alive.append(len(sub))
                yield start, sub
        streamer.chunks = watched
        streamer.timeseries('Spin_Spin')
        assert max(alive) <= streamer.chunk
