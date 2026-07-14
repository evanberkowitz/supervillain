#!/usr/bin/env python
r"""
The DefectGasFugacityTuner probes candidate fugacities through the ordinary
Ensemble/step() route and assembles a production-ready chain with the fugacity and
emit_every matched by construction.  Small probes at N=4 keep these tests quick.
"""

import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.combining import Sequentially
from supervillain.generator.no_intersection import DefectGas, DefectGasFugacityTuner


def _action(N=4, kappa=0.05):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def test_tune_returns_a_rung():
    S = _action()
    t = DefectGasFugacityTuner(S, rng=np.random.default_rng(7))
    ladder = (0.05, 0.02)
    fugacity = t.tune(ladder=ladder, steps=40)
    assert fugacity in ladder


def test_generator_matched_chain():
    S = _action()
    t = DefectGasFugacityTuner(S, rng=np.random.default_rng(11))
    chain = t.generator(ladder=(0.05, 0.02), steps=40)
    assert isinstance(chain, Sequentially)
    gas = chain.generators[-1]
    assert isinstance(gas, DefectGas)
    assert chain.fugacity == gas.fugacity
    assert chain.emit_every == gas.emit_every
    # The chain is production-ready: it generates, every emission carries the
    # inline quantities, and the dwell bound VacuumTicks <= Ticks holds.
    e = supervillain.Ensemble(S).generate(3, chain)
    vac, ticks = np.asarray(e.VacuumTicks), np.asarray(e.Ticks)
    assert np.all(vac > 0) and np.all(ticks >= vac)


def test_tune_edge_matched_pair():
    S = _action()
    t = DefectGasFugacityTuner(S, rng=np.random.default_rng(13))
    ladder = (0.002, 0.01)
    fugacity, emit_every = t.tune_edge(ladder=ladder, steps=40,
                                       min_vacuum_ticks=100, max_probe_sweeps=800)
    assert fugacity in ladder
    assert emit_every >= 1


def test_hammer_explicit_fugacity_is_cheap_sugar():
    # The explicit-fugacity branch must not tune: construction is instant and the
    # gas is last so emitted configurations are its vacuum ticks.
    import supervillain.generator.no_intersection as gen
    S = _action()
    H = gen.Hammer(S, fugacity=0.025)
    assert isinstance(H.generators[-1], DefectGas)
    assert H.generators[-1].fugacity == 0.025


def test_tune_selection_logic_canned():
    # Pin the ladder-walk logic itself with canned probes: dwell = vac/ticks per rung,
    # no MCMC.  tune() must keep the FIRST rung whose second-half dwell beats target.
    S = _action()
    t = DefectGasFugacityTuner(S, rng=np.random.default_rng(1))
    dwells = {0.1: 0.01, 0.05: 0.05, 0.02: 0.2, 0.01: 0.5}

    def canned(fugacity, start, steps, emit_every, max_step_sweeps):
        vac = np.full(steps, 100.0)
        return vac, vac / dwells[fugacity]

    t._probe = canned
    assert t.tune(ladder=(0.1, 0.05, 0.02, 0.01), steps=40, target=0.15) == 0.02


def test_tune_edge_selection_logic_canned():
    # tune_edge must keep the LARGEST measurable stationary rung and stop past the edge.
    S = _action()
    t = DefectGasFugacityTuner(S, rng=np.random.default_rng(2))

    def canned(fugacity, start, steps, emit_every, max_step_sweeps):
        if fugacity > 0.05:
            return None                      # condensed: past the edge
        vac = np.full(steps, float(emit_every))
        return vac, vac / 0.3                # healthy, stationary dwell
    t._probe = canned
    fugacity, emit_every = t.tune_edge(ladder=(0.002, 0.01, 0.05, 0.1), steps=40,
                                       min_vacuum_ticks=100, max_probe_sweeps=800)
    assert fugacity == 0.05
    assert emit_every >= 1


def test_tune_edge_empty_ladder_raises_runtime_error():
    S = _action()
    t = DefectGasFugacityTuner(S, rng=np.random.default_rng(3))
    try:
        t.tune_edge(ladder=())
    except RuntimeError:
        pass
    else:
        assert False, 'tune_edge(ladder=()) must raise RuntimeError'


def test_tune_edge_non_divisible_quota():
    # Ceil-division: a (min_vacuum_ticks, steps) pair that does NOT divide evenly must
    # still be able to meet its own quota (regression: floor-division made 550/200
    # deterministically fail every rung).
    S = _action()
    t = DefectGasFugacityTuner(S, rng=np.random.default_rng(4))

    def canned(fugacity, start, steps, emit_every, max_step_sweeps):
        vac = np.full(steps, float(emit_every))
        return vac, vac / 0.3
    t._probe = canned
    fugacity, emit_every = t.tune_edge(ladder=(0.01,), steps=200,
                                       min_vacuum_ticks=550, max_probe_sweeps=4000)
    assert fugacity == 0.01
