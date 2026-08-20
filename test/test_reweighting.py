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


def strongly_autocorrelated_and_weighted(seed=0, samples=8000, rho=0.98):
    r'''A chain built so that the observable and its influence function have
    conspicuously different autocorrelation.

    The observable is an AR(1) walk with a long memory, and the weight
    $w = e^{-3O}$ is strongly anticorrelated with it, so multiplying by $w$
    largely undoes the drift.  The point is separation: correlating $O$ gives a
    $\tau$ several times larger than correlating $w(O-\bar O)$, so a test can
    tell which one was correlated.  Ordinary ensembles do not separate them --- a
    coarse integer $\tau$ comes out the same either way, and a test built on one
    cannot see the difference.
    '''
    rng = np.random.default_rng(seed)
    O = np.empty(samples)
    O[0] = rng.normal()
    for t in range(1, samples):
        O[t] = rho * O[t-1] + np.sqrt(1 - rho**2) * rng.normal()

    return O, np.exp(-3 * O)


def test_autocorrelation_uses_the_influence_function():
    r'''On a reweighted ensemble the estimator is the ratio
    $\langle Ow\rangle/\langle w\rangle$, and the $\tau$ that inflates its
    variance is that of the influence function
    $f = w(O - \bar O)/\langle w\rangle$, not of $O$.

    Two claims, and the construction is chosen so that both can fail.  The weight
    must be used at all --- correlating $O$ instead gives a $\tau$ five times
    larger here, which no rounding hides.  And it must be used in that
    combination: taking the unweighted mean, squaring the weight, or dividing by
    it rather than multiplying each give a different answer on this chain, where
    on an ordinary one they do not.
    '''
    O, w = strongly_autocorrelated_and_weighted()

    # At unit weight the influence function is O - Obar, so nothing changes.
    assert (autocorrelation_time(O, weight=np.ones(len(O)))
            == autocorrelation_time(O))

    weighted = autocorrelation_time(O, weight=w)

    # It is the influence function that is correlated ...
    mean = (w * O).sum() / w.sum()
    influence = w * (O - mean) / w.mean()
    assert weighted == autocorrelation_time(influence)

    # ... and not the observable, which on this chain is far more correlated.
    # Stated as a band rather than a number, so it holds whatever the seed.
    assert weighted < 15
    assert autocorrelation_time(O) > 25



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


@pytest.mark.parametrize('label,of_ensemble,of_streamer', (
        ('cut',       lambda e: e.cut(20),          lambda s: s.cut(20)),
        ('every',     lambda e: e.every(3),         lambda s: s.every(3)),
        ('cut.every', lambda e: e.cut(20).every(3), lambda s: s.cut(20).every(3)),
        ), ids=('cut', 'every', 'cut.every'))
def test_a_view_weighs_its_own_configurations(tmp_path, label, of_ensemble, of_streamer):
    r'''Ensemble.weight takes its maximum over whatever configurations are in
    hand, so a cut or decimated view renormalizes.  A streamer of the same view
    must do the same.

    Only ratios of weights are physical, so an estimate cannot tell the difference
    --- which is exactly why this needs asserting separately, and why the streaming
    suite's version of it could not: with unit weights every normalization looks
    alike, and the assertion passed without ever being able to fail.
    '''
    e, _ = weighted(configurations=64)

    path = tmp_path / 'weighted.h5'
    with h5.File(path, 'w') as f:
        e.to_h5(f.create_group('ensemble'))

    with h5.File(path, 'r+') as f:
        memory = of_ensemble(supervillain.Ensemble.from_h5(f['ensemble']))
        streamed = of_streamer(EnsembleStreamer(f['ensemble'], chunk=13))

        m = np.asarray(Batch.as_array(memory.weight))
        s = np.asarray(streamed.weight)

        assert m.max() == pytest.approx(1.), 'the ensemble renormalizes over the view'
        assert s.max() == pytest.approx(1.), 'and so must the streamer'
        assert np.allclose(m, s), label

        # And the weights are not all alike, or none of the above means anything.
        assert m.min() < 0.5

        # The estimate agrees too, as it would have either way.
        once = np.arange(len(memory)).reshape(-1, 1)
        reference = Bootstrap(memory, draws=1); reference.indices = once
        streaming = StreamingBootstrap(streamed, f.create_group('b'), indices=once)
        assert float(np.asarray(streaming.ActionDensity)[0]) == pytest.approx(
                float(np.asarray(reference.ActionDensity)[0]))


# The library has no reweighting generator, so the tests above inject the columns
# directly.  That exercises the analysis but not generation --- and a generator is
# where a weight comes from.  Here is the smallest honest one: a single real
# variable, sampled from the wrong distribution on purpose, emitting the log of
# the weight that corrects for it.  Everything about it is checkable by hand.

class Gaussian(supervillain.h5.ReadWriteable):
    r'''One real variable $x$ with action $S = x^2$, so the distribution wanted is
    $e^{-x^2}$: a Gaussian of variance $1/2$, under which $\langle x^2\rangle = 1/2$.'''

    # There is no lattice here and nothing to wind around, but the library asks
    # every action whether it is winding-constrained --- Constrained.autocorrelation
    # reads Action.W to decide whether an observable belongs in the autocorrelation
    # time.  W = 1 is the unconstrained answer, and the true one here.
    W = 1

    def configurations(self, steps):
        return supervillain.configurations.Configurations({
            'x': Batch(steps, shape=(), dtype=float),
        })

    def __str__(self):
        return 'Gaussian'


class XSquared(supervillain.observable.Scalar, supervillain.observable.Observable):
    r'''$x^2$, whose expectation value under $e^{-x^2}$ is $1/2$ and under the
    $e^{-x^2/2}$ actually sampled is $1$.

    A real observable rather than a number the generator emits, so that it is
    registered --- which is what lets a :class:`~.StreamingBootstrap` resample it,
    since its write-through cache is keyed on the registries.'''

    @staticmethod
    def Gaussian(S, x):
        return x**2


class SampleWide(supervillain.generator.Generator):
    r'''Samples $x$ from $e^{-x^2/2}$ --- a Gaussian of variance 1, wider than the
    one wanted --- and emits the log of the correcting weight

    .. math::
        w = \frac{e^{-x^2}}{e^{-x^2/2}} = e^{-x^2/2},

    so that :math:`\log w = -x^2/2`.  Draws are independent, so this is exact
    sampling of the wrong distribution; only the weight makes it the right one.
    '''

    def __init__(self, action, seed=0):
        self.Action = action
        self.rng = np.random.default_rng(seed)

    def inline_observables(self, steps):
        # Only the weight; XSquared is measured from x, not emitted.
        return {'logWeight_gauss': Batch(steps, shape=(), dtype=float)}

    def step(self, configuration):
        x = self.rng.normal(0., 1.)
        return {'x': x, 'logWeight_gauss': -x**2 / 2}

    def report(self):
        return 'SampleWide: x ~ exp(-x^2/2), reweighted to exp(-x^2)'


def test_reweighting_recovers_a_known_distribution():
    r'''Reweighting on a problem with an answer.

    :class:`SampleWide` samples $x$ from $e^{-x^2/2}$ when $e^{-x^2}$ is wanted,
    and emits the correcting log-weight as any reweighting generator would.  Then

        $\langle x^2 \rangle = 1$    under the distribution actually sampled, and
        $\langle x^2 \rangle = 1/2$  under the one wanted,

    and the whole point is that the second is what comes out.  Every other test
    here checks that the machinery agrees with itself; this one checks that it is
    right, against an answer that does not come from the code.
    '''
    samples = 20000
    action = Gaussian()
    e = supervillain.Ensemble(action).generate(samples, SampleWide(action), start='cold')

    assert 'logWeight_gauss' in e.configuration.fields

    # The weight really is the ratio of the two distributions, up to the
    # normalization that cancels from the estimator.
    x = np.asarray(Batch.as_array(e.x))
    w = np.asarray(Batch.as_array(e.weight))
    assert np.allclose(w, np.exp(-x**2 / 2) / np.exp(-x**2 / 2).max())

    indices = np.random.default_rng(1).integers(0, samples, (samples, 100))
    weighted_bootstrap = Bootstrap(e, draws=100); weighted_bootstrap.indices = indices

    mean, error = (float(_) for _ in weighted_bootstrap.estimate('XSquared'))
    assert abs(mean - 0.5) < 5 * error, f'{mean} +/- {error} is not 1/2'

    # Drop the weight and it is the distribution that was sampled, not the one
    # wanted --- a factor of two away, so this could not pass by accident.
    unweighted = supervillain.Ensemble(action).from_configurations(
            supervillain.configurations.Configurations(
                {'x': e.configuration.fields['x']}))
    plain = Bootstrap(unweighted, draws=100); plain.indices = indices

    unweighted_mean, unweighted_error = (float(_) for _ in plain.estimate('XSquared'))
    assert abs(unweighted_mean - 1.0) < 5 * unweighted_error
    assert abs(mean - unweighted_mean) > 20 * error


def test_a_reweighting_generator_survives_continue_from_and_extend(tmp_path):
    r'''The production shape, on the toy: generate, store, continue from disk,
    extend, and again.

    continue_from reuses the stored generator, so the continuation emits the same
    logWeight_ column and extend_h5 has a column to extend.  The weight of the
    grown ensemble must be the weight of the whole chain --- and because the toy
    has an answer, the grown ensemble must still give it.
    '''
    steps = 4000
    action = Gaussian()
    e = supervillain.Ensemble(action).generate(steps, SampleWide(action), start='cold')

    path = tmp_path / 'grown.h5'
    with h5.File(path, 'w') as f:
        e.to_h5(f.create_group('ensemble'))

    continuations = 2
    for _ in range(continuations):
        with h5.File(path, 'r+') as f:
            supervillain.Ensemble.continue_from(
                    f['ensemble'], steps).extend_h5(f['ensemble'])

    with h5.File(path, 'r+') as f:
        grown = supervillain.Ensemble.from_h5(f['ensemble'])
        assert len(grown) == (1 + continuations) * steps

        # Every column grew with the configurations; nothing is short.
        fields = f['ensemble/configuration/fields']
        assert set(fields[k]['data'].shape[0] for k in fields) == {len(grown)}

        # The weight is the whole chain's, derived from the whole column.
        logs = np.asarray(Batch.as_array(grown.configuration.fields['logWeight_gauss']))
        assert np.allclose(np.asarray(Batch.as_array(grown.weight)),
                           np.exp(logs - logs.max()))

        # And the grown ensemble still knows the answer.
        indices = np.random.default_rng(2).integers(0, len(grown), (len(grown), 100))
        bootstrap = Bootstrap(grown, draws=100); bootstrap.indices = indices
        mean, error = (float(_) for _ in bootstrap.estimate('XSquared'))
        assert abs(mean - 0.5) < 5 * error, f'{mean} +/- {error} is not 1/2'

        # And because XSquared is a registered observable rather than a number the
        # generator emitted, the streamed path can be asked for it too --- so the
        # analytic answer anchors that path directly, not only by its agreement
        # with the in-memory one.
        streamer = EnsembleStreamer(f['ensemble'], chunk=333)
        streaming = StreamingBootstrap(
                streamer, f.create_group('streamed'), indices=indices)
        streamed_mean, streamed_error = (
                float(_) for _ in streaming.estimate('XSquared'))
        assert abs(streamed_mean - 0.5) < 5 * streamed_error
        assert streamed_mean == pytest.approx(mean)

        blocked = StreamingBlocking(streamer, width=4)
        once = np.arange(len(blocked)).reshape(-1, 1)
        blocked_bootstrap = StreamingBootstrap(
                blocked, f.create_group('blocked'), indices=once)
        assert float(np.asarray(blocked_bootstrap.XSquared)[0]) == pytest.approx(0.5, abs=5*error)

        assert np.allclose(np.asarray(Batch.as_array(grown.weight)),
                           np.asarray(streamer.weight))
        assert np.allclose(
                np.asarray(Batch.as_array(grown.cut(500).every(3).weight)),
                np.asarray(streamer.cut(500).every(3).weight))


def test_a_toy_ensemble_answers_the_whole_stack():
    r'''The library asks an action questions beyond generating configurations ---
    Constrained.autocorrelation reads Action.W to decide whether an observable
    belongs in the autocorrelation time --- and asks them of every registered
    observable when nothing has been measured yet.

    Villain and Worldline both carry W, so a toy is the first action that can fail
    to answer.  It should not be the first thing a new action discovers.
    '''
    action = Gaussian()
    e = supervillain.Ensemble(action).generate(500, SampleWide(action), start='cold')

    assert not e.measured                  # nothing measured, so everything is asked
    assert e.autocorrelation_time() >= 1
    assert 'XSquared' in e.measure()
    assert len(Blocking(e, width=5)) == 100
    assert len(e.cut(10).every(3)) == len(np.arange(500)[10::3])


def test_every_weight_zero_is_refused(tmp_path):
    r'''A reweighting with no overlap at all --- every log-weight minus infinity ---
    has no weighted expectation value: <Ow>/<w> is 0/0, and the global maximum
    subtraction is minus infinity less minus infinity.

    numpy answers that with nan and a warning, which is the worst outcome: the
    estimates come back, and they are all nan.  Both derivations refuse instead.
    One weight of zero among many is fine, and stays fine.
    '''
    def ensemble(logs):
        return supervillain.Ensemble(Gaussian()).from_configurations(
                supervillain.configurations.Configurations({
                    'x': Batch(np.zeros(len(logs))),
                    'logWeight_a': Batch(np.asarray(logs, dtype=float)),
                }))

    # One configuration carrying no weight is ordinary.
    fine = np.asarray(Batch.as_array(ensemble([0., -np.inf, 1., 2.]).weight))
    assert np.isfinite(fine).all()
    assert fine.min() == 0.
    assert fine.max() == pytest.approx(1.)

    # All of them is not.
    with pytest.raises(ValueError):
        ensemble([-np.inf] * 4).weight

    # And the streamed derivation refuses it too, rather than differ.
    path = tmp_path / 'zero.h5'
    with h5.File(path, 'w') as f:
        supervillain.Ensemble(Gaussian()).from_configurations(
                supervillain.configurations.Configurations({
                    'x': Batch(np.zeros(4)),
                    'logWeight_a': Batch(np.full(4, -np.inf)),
                })).to_h5(f.create_group('ensemble'))

    with h5.File(path, 'r') as f:
        with pytest.raises(ValueError):
            EnsembleStreamer(f['ensemble'], chunk=2)
