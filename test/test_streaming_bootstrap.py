#!/usr/bin/env python
r"""Tests for the streaming bootstrap: EnsembleStreamer (memory-bounded block
iteration of a serialized Ensemble) and StreamingBootstrap (block-accumulated
resample with a write-through disk-cache gate).

The load-bearing test is `test_streaming_equivalence`: given the SAME resampling
indices, the streaming estimate must equal a plain Bootstrap's to floating point.
Everything else here guards a property that would otherwise fail silently ---
that blocks reassemble the ensemble exactly, that the h5 link survives a
round trip, that cached quantities are served without touching the source, and
that the importance weight is normalized by ONE global maximum.
"""

import numpy as np
import h5py as h5
import pytest

import supervillain
import supervillain.h5
from supervillain.batch import Batch
from supervillain.analysis import Bootstrap
from supervillain.analysis.bootstrap import (
        EnsembleStreamer, StreamingBootstrap, _stream_weight)
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
    r'''The blocks a streamer yields must reassemble into the whole ensemble ---
    both the stored configuration fields and an observable computed from them.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r') as f:
        full = supervillain.Ensemble.from_h5(f['ensemble'])
        streamer = EnsembleStreamer(f['ensemble'], block=7)

        assert len(streamer) == len(full)

        # A block boundary that divides the ensemble evenly would not notice an
        # off-by-one in the final block; 7 does not divide 40.
        assert len(full) % streamer.block != 0

        # Stored fields, block by block.
        for field in ('phi', 'n'):
            blocks = [np.asarray(getattr(sub, field)) for _, sub in streamer.blocks()]
            assert np.array_equal(np.concatenate(blocks, axis=0),
                                  np.asarray(getattr(full, field)))

        # Starts must tile the ensemble contiguously from zero.
        starts = [start for start, _ in streamer.blocks()]
        assert starts == list(range(0, len(full), streamer.block))

        # An observable computed (not stored inline) from the sliced fields.
        blocks = [np.asarray(sub.Spin_Spin) for _, sub in streamer.blocks()]
        assert np.allclose(np.concatenate(blocks, axis=0), np.asarray(full.Spin_Spin))


def test_streaming_equivalence(tmp_path):
    r'''On the same resampling indices the streamed estimate must equal the
    in-memory one to floating point.  This is the whole point of the class.'''
    path = villain_h5(tmp_path)

    with h5.File(path, 'r+') as f:
        full = supervillain.Ensemble.from_h5(f['ensemble'])
        reference = Bootstrap(full, draws=50)

        streamer = EnsembleStreamer(f['ensemble'], block=9)
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
        streamer = EnsembleStreamer(f['ensemble'], block=9)
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
        streamer = EnsembleStreamer(f['ensemble'], block=9)
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
        streamer = EnsembleStreamer(f['ensemble'], block=9)
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

        streamer = EnsembleStreamer(f['ensemble'], block=17)
        assert len(streamer) == len(full)

        blocks = [np.asarray(sub.phi) for _, sub in streamer.blocks()]
        assert np.array_equal(np.concatenate(blocks, axis=0), np.asarray(full.phi))

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
    streaming resample accumulates <Ow> and <w> separately across blocks, and the
    exp offset cancels between them only if every configuration shares it.  A
    per-block --- or per-column --- maximum would silently bias the estimate.
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
