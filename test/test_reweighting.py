#!/usr/bin/env python
r"""Tests for importance reweighting.

A reweighting generator emits its per-configuration log-weight as an inline
`logWeight_<name>` column.  Ensemble.weight is derived from those on access ---
logs sum, so weights multiply --- and the analysis stack then corrects every
estimate with no further intervention.

The load-bearing test is `test_reweighted_bootstraps_agree`: a weighted ensemble
must give the same answer read whole as it does streamed, since the two derive
the weight independently and would otherwise disagree in silence.
"""

import numpy as np
import h5py as h5
import pytest

import supervillain
import supervillain.h5
from supervillain.batch import Batch
from supervillain.analysis import Bootstrap, Blocking, autocorrelation_time
from supervillain.analysis import EnsembleStreamer, StreamingBlocking, StreamingBootstrap
import generate

N = 4
KAPPA = 0.2
CONFIGURATIONS = 60


def weighted(configurations=CONFIGURATIONS, seed=7, spreads=(1.5, 0.7)):
    r'''An ensemble carrying two generators' worth of log-weight, as
    Sequentially would leave it.  Returns the ensemble and the weight it implies.'''
    e = generate.villain(configurations, N, KAPPA)
    rng = np.random.default_rng(seed)

    logs = []
    names = ('alpha', 'beta', 'gamma', 'delta')[:len(spreads)]
    assert len(names) == len(spreads), 'name a column for every spread'
    for name, spread in zip(names, spreads):
        log = rng.normal(0., spread, configurations)
        e.configuration.fields[f'logWeight_{name}'] = Batch(log)
        logs.append(log)

    if not logs:
        # No reweighting generator ran, which is the ordinary case.
        return e, np.ones(configurations)

    total = sum(logs)
    return e, np.exp(total - total.max())


def test_logs_sum_so_weights_multiply():
    r'''Each generator emits its own contribution under its own name; the total
    weight is their product, which is the exponential of the summed logs.'''
    e, expected = weighted()

    assert np.allclose(np.asarray(Batch.as_array(e.weight)), expected)
    assert expected.max() == pytest.approx(1.)
    # Genuinely non-trivial, or the test proves nothing.
    assert expected.min() < 1e-2


def test_no_columns_means_unit_weight():
    r'''An ordinary generator emits nothing, and an unweighted ensemble is the
    ordinary sample mean --- which is what makes reweighting free to everyone who
    does not use it.'''
    e = generate.villain(CONFIGURATIONS, N, KAPPA)

    assert not [k for k in e.configuration.fields if k.startswith('logWeight_')]
    assert np.allclose(np.asarray(Batch.as_array(e.weight)), 1.)


def test_weight_is_derived_not_stored(tmp_path):
    r'''The logs are the canonical stored quantity, and the weight is taken from
    whatever configurations are in hand.  So cut and every need not rewrite it,
    and cannot disagree with it.'''
    e, _ = weighted()

    for view, expected in ((e.cut(11), np.asarray(Batch.as_array(e.weight))[11:]),
                           (e.every(3), np.asarray(Batch.as_array(e.weight))[::3])):
        got = np.asarray(Batch.as_array(view.weight))
        # Equal up to the global max, which is retaken over what remains; only
        # ratios are physical, so compare after normalizing both the same way.
        assert np.allclose(got / got.max(), expected / expected.max())

    path = tmp_path / 'weighted.h5'
    with h5.File(path, 'w') as f:
        e.to_h5(f.create_group('ensemble'))
    with h5.File(path, 'r') as f:
        assert 'weight' not in f['ensemble']
        assert [k for k in f['ensemble/configuration/fields'] if k.startswith('logWeight_')]


@pytest.mark.parametrize('spreads', ((), (1.5,), (1.5, 0.7), (1.5, 0.7, 0.4)),
                         ids=('none', 'one', 'two', 'three'))
def test_reweighted_bootstraps_agree(tmp_path, spreads):
    r'''The load-bearing test.  Ensemble.weight and EnsembleStreamer's weight are
    derived independently, from the same columns, and a bootstrap of one file must
    not depend on which of them read it.

    Parametrized over how many generators contributed, because the two derivations
    are separate implementations of the same sum: with no column at all they must
    both fall back to unit weight, with one they must not mangle a single-element
    sum, and with several they must agree on the total and on the one global
    maximum taken from it.
    '''
    e, _ = weighted(spreads=spreads)

    path = tmp_path / 'weighted.h5'
    with h5.File(path, 'w') as f:
        e.to_h5(f.create_group('ensemble'))

    with h5.File(path, 'r+') as f:
        whole = supervillain.Ensemble.from_h5(f['ensemble'])
        streamer = EnsembleStreamer(f['ensemble'], chunk=13)

        assert np.allclose(np.asarray(Batch.as_array(whole.weight)),
                           np.asarray(streamer.weight))

        reference = Bootstrap(whole, draws=40)
        streaming = StreamingBootstrap(
                streamer, f.create_group('bootstrap'), indices=reference.indices)

        for quantity in ('ActionDensity', 'WindingSquared', 'Spin_Spin',
                         'SpinSusceptibility'):
            mean, error = (np.asarray(_) for _ in reference.estimate(quantity))
            streamed_mean, streamed_error = (
                    np.asarray(_) for _ in streaming.estimate(quantity))
            assert np.allclose(mean, streamed_mean, atol=1e-10, rtol=1e-8), quantity
            assert np.allclose(error, streamed_error, atol=1e-10, rtol=1e-8), quantity


def test_the_weight_actually_changes_the_answer(tmp_path):
    r'''Reweighting that made no difference would pass every test above.  The same
    configurations, weighted and not, must give different estimates --- and the
    weighted one must be the ratio estimator <Ow>/<w>.'''
    e, weight = weighted()

    unweighted = supervillain.Ensemble(e.Action).from_configurations(
            supervillain.configurations.Configurations(
                {k: v for k, v in e.configuration.fields.items()
                 if not k.startswith('logWeight_')}))

    observable = np.asarray(Batch.as_array(e.ActionDensity))
    assert np.allclose(np.asarray(Batch.as_array(unweighted.weight)), 1.)

    # Resample each configuration exactly once: then the bootstrap is not a
    # sample of the estimator, it *is* the estimator, and the comparison is
    # algebra rather than statistics.
    identity = np.arange(len(e)).reshape(-1, 1)
    with_weight = Bootstrap(e, draws=1);          with_weight.indices = identity
    without     = Bootstrap(unweighted, draws=1); without.indices = identity

    assert float(np.asarray(with_weight.ActionDensity)[0]) == pytest.approx(
            (weight * observable).sum() / weight.sum())
    assert float(np.asarray(without.ActionDensity)[0]) == pytest.approx(
            observable.mean())

    # And those are not the same number, so the weights are doing something.
    assert not np.isclose(float(np.asarray(with_weight.ActionDensity)[0]),
                          float(np.asarray(without.ActionDensity)[0]))


def test_blocking_of_a_weighted_ensemble_telescopes(tmp_path):
    r'''Blocking a weighted ensemble and bootstrapping the blocks must estimate
    what bootstrapping the configurations does: each block carries <wO>_b/<w>_b
    against a block weight <w>_b, so the weighted average over blocks collapses
    back to <wO>/<w>.'''
    e, weight = weighted(configurations=64)
    observable = np.asarray(Batch.as_array(e.ActionDensity))
    truth = (weight * observable).sum() / weight.sum()

    blocking = Blocking(e, width=4)
    blocked = np.asarray(Batch.as_array(blocking.ActionDensity))
    block_weight = np.asarray(Batch.as_array(blocking.weight))

    assert (block_weight * blocked).sum() / block_weight.sum() == pytest.approx(truth)


def test_streamed_blocking_of_a_weighted_ensemble_matches(tmp_path):
    r'''And the streamed blocking of the same weighted ensemble must match the
    in-memory one --- which at unit weight is trivial, and here is not.'''
    e, _ = weighted(configurations=64)

    path = tmp_path / 'weighted.h5'
    with h5.File(path, 'w') as f:
        e.to_h5(f.create_group('ensemble'))

    with h5.File(path, 'r') as f:
        memory = Blocking(supervillain.Ensemble.from_h5(f['ensemble']), width=4)
        streamed = StreamingBlocking(
                EnsembleStreamer(f['ensemble'], chunk=13), width=4)

        assert np.allclose(streamed.weight, np.asarray(Batch.as_array(memory.weight)))
        for quantity in ('ActionDensity', 'Spin_Spin'):
            assert np.allclose(streamed.timeseries(quantity),
                               np.asarray(Batch.as_array(getattr(memory, quantity)))), quantity


def test_autocorrelation_uses_the_influence_function():
    r'''On a reweighted ensemble the estimator is the ratio <Ow>/<w>, and the tau
    that inflates its variance is that of the influence function
    f = w(O - Obar)/<w>, not of O.  At unit weight the two coincide, which is what
    keeps this free for everyone else.'''
    e, weight = weighted()
    observable = np.asarray(Batch.as_array(e.ActionDensity))

    unit = np.ones(len(observable))
    assert (autocorrelation_time(observable, weight=unit)
            == autocorrelation_time(observable))

    # With real weights it is the influence function that is correlated.
    mean = (weight * observable).sum() / weight.sum()
    influence = weight * (observable - mean) / weight.mean()
    assert (autocorrelation_time(observable, weight=weight)
            == autocorrelation_time(influence))


def test_a_second_generator_contributes_rather_than_replaces():
    r'''Two reweighting generators each emit under their own namespaced key, and
    the total weight is the product --- the sum of the logs.  If the second column
    replaced the first rather than accumulating, or if one silently won, every
    other test here would still pass, since they all carry both columns from the
    start.'''
    e = generate.villain(CONFIGURATIONS, N, KAPPA)
    rng = np.random.default_rng(3)
    alpha = rng.normal(0., 1.1, CONFIGURATIONS)
    beta = rng.normal(0., 0.9, CONFIGURATIONS)

    def weight_now():
        return np.asarray(Batch.as_array(e.weight))

    def normalized(w):
        return w / w.max()

    e.configuration.fields['logWeight_alpha'] = Batch(alpha)
    one = weight_now()
    assert np.allclose(one, normalized(np.exp(alpha)))

    e.configuration.fields['logWeight_beta'] = Batch(beta)
    both = weight_now()

    # The product of the two, not either alone.
    assert np.allclose(both, normalized(np.exp(alpha) * np.exp(beta)))
    assert not np.allclose(both, one)
    assert not np.allclose(both, normalized(np.exp(beta)))

    # And the estimate moves, so the second contribution is doing something.
    observable = np.asarray(Batch.as_array(e.ActionDensity))
    assert not np.isclose((one * observable).sum() / one.sum(),
                          (both * observable).sum() / both.sum())


def test_every_path_gives_the_same_weighted_estimate(tmp_path):
    r'''Four ways to the same number, on one weighted ensemble: read whole or
    streamed, blocked or not.  Each applies the weight in a different place --- the
    plain bootstrap once, the blocked ones inside the block and again over blocks
    --- so agreement is what says none of them applies it twice, or drops it.

    Resampling each sample exactly once makes this the estimator itself rather
    than a draw from it, so the four are compared as algebra.
    '''
    width = 4
    e, weight = weighted(configurations=64)
    observable = np.asarray(Batch.as_array(e.ActionDensity))
    truth = (weight * observable).sum() / weight.sum()

    def once(n):
        return np.arange(n).reshape(-1, 1)

    def estimate(bootstrap):
        return float(np.asarray(bootstrap.ActionDensity)[0])

    path = tmp_path / 'weighted.h5'
    with h5.File(path, 'w') as f:
        e.to_h5(f.create_group('ensemble'))

    plain = Bootstrap(e, draws=1); plain.indices = once(len(e))

    blocking = Blocking(e, width=width)
    assert blocking.drop == 0
    blocked = Bootstrap(blocking, draws=1); blocked.indices = once(len(blocking))

    with h5.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], chunk=13)
        streamed = StreamingBootstrap(streamer, f.create_group('streamed'),
                                      indices=once(len(streamer)))

        streamed_blocking = StreamingBlocking(streamer, width=width)
        streamed_blocked = StreamingBootstrap(
                streamed_blocking, f.create_group('streamed_blocked'),
                indices=once(len(streamed_blocking)))

        for label, bootstrap in (('in memory', plain),
                                 ('in memory, blocked', blocked),
                                 ('streamed', streamed),
                                 ('streamed, blocked', streamed_blocked)):
            assert estimate(bootstrap) == pytest.approx(truth), label

    # The weights are not decoration: unweighted would give something else.
    assert not np.isclose(truth, observable.mean())
