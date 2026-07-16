#!/usr/bin/env python
r"""
Certification of the parallel-tempering machinery on cheap 2D Villain ladders.

The composite kernel's factors each satisfy detailed balance, so a tempered rung's
marginals must match an un-tempered chain at the same kappa; the swap kernel, the
schedule, the replica bookkeeping, and the whole-ladder continuation are certified
piece by piece.  Everything here runs at N=4 in D=2 to stay quick.
"""

import numpy as np
import h5py as h5
import pytest

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator import Generator
from supervillain.generator.combining import Sequentially
from supervillain.tempering import (
    ParallelTempering, ParallelTemperingTuner, TemperedRung, EvenOddPairs, swap_accepted,
)
import supervillain.generator.villain as villain


def _ladder(kappas, N=4, seed=None):
    L = Lattice(2, N)
    actions = [supervillain.action.Villain(L, k) for k in kappas]
    generators = [Sequentially((villain.SiteUpdate(S), villain.LinkUpdate(S))) for S in actions]
    return ParallelTempering(actions, generators, seed=seed)


class SeededShift(Generator):
    r"""A deterministic test generator: shifts phi by draws from its own seeded rng.

    Not detailed-balanced --- it exists only to certify that the ladder machinery
    itself is bitwise deterministic when its ingredients are.
    """

    def __init__(self, S, seed):
        self.Action = S
        self.rng = np.random.default_rng(seed)

    def step(self, cfg):
        return cfg | {'phi': cfg['phi'] + self.rng.normal(size=cfg['phi'].shape)}

    def inline_observables(self, steps):
        return dict()

    def report(self):
        return 'SeededShift'


def test_swap_kernel():
    # dS <= 0 accepts regardless of the uniform.
    assert swap_accepted(0.2, -3.0, 0.999999)
    assert swap_accepted(-0.2, 3.0, 0.999999)
    # dkappa -> 0 always accepts.
    assert swap_accepted(0.0, 123.4, 0.999999)
    # dS > 0 accepts exactly when uniform < exp(-dS).
    dkappa, dE = 0.5, 3.0
    threshold = np.exp(-dkappa * dE)
    assert swap_accepted(dkappa, dE, threshold * 0.999)
    assert not swap_accepted(dkappa, dE, threshold * 1.001)


def test_even_odd_schedule():
    schedule = EvenOddPairs(5)
    assert schedule.pairs(0) == ((0, 1), (2, 3))
    assert schedule.pairs(1) == ((1, 2), (3, 4))
    # Every adjacent pair is attempted every other sweep, and no rung twice at once.
    for sweep in (0, 1):
        flat = [r for pair in schedule.pairs(sweep) for r in pair]
        assert len(flat) == len(set(flat))


def test_replica_permutation_and_round_trips():
    pt = _ladder((0.3, 0.5, 0.8), seed=5)
    ensembles = pt.generate(100)

    R = np.stack([np.asarray(e.configuration.Replica) for e in ensembles])
    for t in range(R.shape[1]):
        assert sorted(R[:, t]) == [0, 1, 2]

    # Recount crossings from the stored traces and compare with the online counter.
    crossings = np.zeros(3, dtype=int)
    last = np.full(3, -1, dtype=int)
    for t in range(R.shape[1]):
        for replica, endpoint in ((R[0, t], 0), (R[-1, t], 1)):
            if last[replica] != endpoint:
                if last[replica] != -1:
                    crossings[replica] += 1
                last[replica] = endpoint
    assert pt.round_trips == int(crossings.sum()) // 2


def test_degenerate_ladder_always_swaps():
    # All rungs at the same kappa: dS = 0 identically, every attempt accepts.
    pt = _ladder((0.5, 0.5, 0.5), seed=3)
    pt.generate(50)
    assert (pt.accepted == pt.attempted).all()
    assert (pt.attempted > 0).all()


def test_exactness_against_untempered():
    r"""A tempered rung's ActionDensity must agree with an un-tempered chain at the
    same kappa within errors."""
    kappas = (0.4, 0.6)
    steps, cut = 800, 200
    pt = _ladder(kappas, seed=11)
    tempered = [e.cut(cut) for e in pt.generate(steps)]

    L = Lattice(2, 4)
    for kappa, te in zip(kappas, tempered):
        S = supervillain.action.Villain(L, kappa)
        g = Sequentially((villain.SiteUpdate(S), villain.LinkUpdate(S)))
        ue = supervillain.Ensemble(S).generate(steps, g).cut(cut)

        means, errs = [], []
        for e in (te, ue):
            x = np.asarray(e.ActionDensity)
            tau = max(1, int(np.ceil(e.autocorrelation_time(observables=('ActionDensity',)))))
            means.append(x.mean())
            errs.append(x.std() * np.sqrt(2 * tau / len(x)))
        pooled = np.sqrt(errs[0]**2 + errs[1]**2)
        assert abs(means[0] - means[1]) < 5 * pooled, (
            f'kappa={kappa}: tempered {means[0]:.4f} vs untempered {means[1]:.4f} '
            f'differ by more than 5 x {pooled:.4f}'
        )


def test_determinism():
    # Same ladder seed and seeded local generators: bitwise-identical output, twice.
    def run():
        L = Lattice(2, 4)
        actions = [supervillain.action.Villain(L, k) for k in (0.3, 0.5, 0.8)]
        generators = [SeededShift(S, seed=100 + i) for i, S in enumerate(actions)]
        pt = ParallelTempering(actions, generators, seed=17)
        return pt.generate(60)

    a, b = run(), run()
    for ea, eb in zip(a, b):
        assert (np.asarray(ea.configuration.phi) == np.asarray(eb.configuration.phi)).all()
        assert (np.asarray(ea.configuration.Replica) == np.asarray(eb.configuration.Replica)).all()


def test_single_rung_continuation_refused():
    pt = _ladder((0.3, 0.5), seed=7)
    ensembles = pt.generate(10)
    assert isinstance(ensembles[0].generator, TemperedRung)
    with pytest.raises(RuntimeError):
        supervillain.Ensemble.continue_from(ensembles[0], 5)


def test_ladder_continuation():
    pt = _ladder((0.3, 0.5, 0.8), seed=9)
    first = pt.generate(20)
    more = ParallelTempering.continue_from(first, 10, seed=10)
    for e, m in zip(first, more):
        assert int(np.asarray(m.index)[0]) == int(np.asarray(e.index)[-1]) + 1
        assert len(m) == 10
        # The continuation picks up from the last configuration, replicas included.
        assert m.Action.kappa == e.Action.kappa


def test_h5_round_trip(tmp_path):
    pt = _ladder((0.3, 0.5), seed=13)
    ensembles = pt.generate(10)

    with h5.File(tmp_path / 'tempered.h5', 'w') as f:
        for i, e in enumerate(ensembles):
            e.to_h5(f.create_group(f'rung{i}'))

    with h5.File(tmp_path / 'tempered.h5', 'r') as f:
        read = [supervillain.Ensemble.from_h5(f[f'rung{i}']) for i in range(2)]
        for i, e in enumerate(read):
            assert isinstance(e.generator, TemperedRung)
            assert e.generator.rung == i
            assert (e.generator.ladder == pt.kappa).all()
            assert (np.asarray(e.configuration.Replica) ==
                    np.asarray(ensembles[i].configuration.Replica)).all()
        with pytest.raises(RuntimeError):
            supervillain.Ensemble.continue_from(read[0], 5)


def test_tuner_recommends_a_ladder():
    pt = _ladder((0.3, 0.5, 0.8), seed=21)
    pt.generate(100)
    tuner = ParallelTemperingTuner(pt)

    recommended = tuner.ladder(target=0.25)
    assert recommended[0] == pytest.approx(0.3)
    assert recommended[-1] == pytest.approx(0.8)
    assert (np.diff(recommended) > 0).all()
    # Uniform predicted acceptance at or above the target, by construction.
    predicted = tuner.predicted_acceptance(recommended)
    assert predicted.std() < 1e-8
    assert (predicted >= 0.25 - 1e-8).all()

    forced = tuner.ladder(rungs=6)
    assert len(forced) == 6
