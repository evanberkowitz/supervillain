#!/usr/bin/env python
r"""
The DefectGasWeightTuner learns the per-sector table by damped histogram recursion
over sweep-budgeted probes: overweighted sectors are cut toward the mean, unvisited
sectors are left alone, and convergence demands flatness AND round trips AND
stationarity.  Small probes at N=4 keep these tests quick.
"""

import numpy as np
import pytest

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.combining import Sequentially
from supervillain.generator.no_intersection import DefectGas, DefectGasWeightTuner


def _action(N=4, kappa=0.05):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def test_recursion_update_and_convergence_canned():
    S = _action()
    V = S.Lattice.N ** 4
    t = DefectGasWeightTuner(S, D_max=4, rng=np.random.default_rng(5))
    calls = []
    canned = [
        (np.array([900, 90, 0]), 3),      # lopsided, k=2 unvisited, too few trips
        (np.array([400, 300, 200]), 8),   # flat within 3x, enough trips -> converge
    ]

    def probe(w, cfg, probe_sweeps, companion_every):
        calls.append(w.copy())
        t_, trips = canned[min(len(calls) - 1, 1)]
        return t_, t_ // 2, t_ - t_ // 2, trips, cfg

    t._probe = probe
    w, emit_every = t.tune(probe_sweeps=100, lighten=1.0)

    assert len(calls) == 2                       # converged on the second probe
    warm, updated = calls
    u = 1.0 / V
    assert np.allclose(warm, [1.0, u, 2 * u**2])  # k! u^k warm start, w[0] = 1
    # alpha = 1 on the first pass: w[k] *= (tbar/t[k]) on visited sectors, then
    # renormalize w[0] = 1.  tbar = mean(900, 90) = 495; k = 2 is untouched.
    tbar = 495.0
    expect = warm.copy()
    expect[0] *= tbar / 900
    expect[1] *= tbar / 90
    expect /= expect[0]
    assert np.allclose(updated, expect)
    assert np.allclose(w, updated)               # the frozen table is the probed one
    # emit_every from the FINAL probe's vacuum dwell: 400/900 of 4V, x25 sweeps.
    assert emit_every == max(1, round((400 / 900) * 4 * V * 25))
    # The measured mixing timescale: the converging probe ran 200 sweeps (doubled
    # once after the first probe's 3 < 5 round trips) and completed 8 trips.
    assert t.mixing_sweeps == 200 / 8


def test_no_convergence_warns_and_returns_best_probed(caplog):
    S = _action()
    t = DefectGasWeightTuner(S, D_max=4, rng=np.random.default_rng(6))
    calls = []

    def probe(w, cfg, probe_sweeps, companion_every):
        calls.append(w.copy())
        t_ = np.array([1000, 1, 0])              # never flattens, never converges
        return t_, t_ // 2, t_ - t_ // 2, 0, cfg

    t._probe = probe
    w, emit_every = t.tune(probe_sweeps=100, max_iterations=3, lighten=1.0)
    assert len(w) == 3 and w[0] == 1.0 and emit_every >= 1
    # The frozen table must be one that was actually PROBED -- freezing the
    # post-update table hands production an unmeasured chain.  All probes score
    # equally here, so the first (the warm start) is kept.
    assert np.allclose(w, calls[0])
    assert any('no convergence' in r.message for r in caplog.records)


def test_update_clipped_and_probes_lengthen():
    # A sector visited by a handful of ticks must not receive an enormous noisy
    # boost (clip at max_update per iteration), and zero-round-trip probes signal
    # that the sectors mix slower than the probe: lengthen before re-measuring.
    S = _action()
    t = DefectGasWeightTuner(S, D_max=4, rng=np.random.default_rng(8))
    seen = []

    def probe(w, cfg, probe_sweeps, companion_every):
        seen.append((w.copy(), probe_sweeps))
        t_ = np.array([10_000_000, 100, 0])      # naive boost for k=1 would be ~5e4
        return t_, t_ // 2, t_ - t_ // 2, 0, cfg

    t._probe = probe
    t.tune(probe_sweeps=100, max_iterations=3)
    (w0, s0), (w1, s1), (w2, s2) = seen
    ratio = w1 / w0                              # per-sector update, post-renormalization
    assert np.max(ratio) / np.min(ratio) <= 100 + 1e-9   # each raw factor in [1/10, 10]
    assert (s0, s1, s2) == (100, 200, 400)


def test_tune_real_tiny():
    S = _action()
    t = DefectGasWeightTuner(S, D_max=8, rng=np.random.default_rng(7))
    w, emit_every = t.tune(probe_sweeps=400, max_iterations=8)
    assert w.shape == (5,) and w[0] == 1.0 and np.all(w > 0)
    assert emit_every >= 1


def test_generator_produces():
    S = _action()
    t = DefectGasWeightTuner(S, D_max=8, rng=np.random.default_rng(11))
    chain = t.generator(probe_sweeps=400, max_iterations=8)
    assert isinstance(chain, Sequentially)
    gas = chain.generators[-1]
    assert isinstance(gas, DefectGas)
    assert np.array_equal(chain.sectorWeights, gas.sectorWeights)
    assert chain.emit_every == gas.emit_every
    e = supervillain.Ensemble(S).generate(3, chain)
    vac, ticks = np.asarray(e.Vacuum_Ticks), np.asarray(e.Ticks)
    assert np.all(vac > 0) and np.all(ticks >= vac)
    assert np.asarray(e.SectorTicks).sum() == ticks.sum()


def test_w_independence():
    # The estimator is w-independent: tables differing 2x in the PAIR sector (the
    # exactness-critical rung -- Theta_Theta divides its tallies by w[1]) must agree
    # on the near correlator.  kappa = 0.2 and light deep sectors keep the vacuum
    # stable over the whole run (at kappa = 0.05 every table eventually nucleates
    # out of the metastable vacuum -- the physics this machinery exists for).
    # Ratio-of-sums with blocked jackknife errors; statistical but seeded, 5 sigma.
    import supervillain.generator.villain as villain

    S = _action(kappa=0.2)
    tables = ([1.0, 0.09, 8e-4, 8e-6, 8e-8],
              [1.0, 0.045, 8e-4, 8e-6, 8e-8])
    results = []
    for seed, w in enumerate(tables):
        gas = DefectGas(S, sectorWeights=w, emit_every=200,
                        rng=np.random.default_rng(100 + seed))
        chain = Sequentially((villain.SiteUpdate(S), gas))
        e = supervillain.Ensemble(S).generate(400, chain)
        T = np.asarray(e.Theta_Theta).real[:, 1, 0, 0, 0]
        V = np.asarray(e.Vacuum_Ticks).astype(float)
        B = 20
        n = len(T) // B
        Tb = T[:B * n].reshape(B, n).sum(axis=1)
        Vb = V[:B * n].reshape(B, n).sum(axis=1)
        jk = np.array([(Tb.sum() - Tb[b]) / (Vb.sum() - Vb[b]) for b in range(B)])
        results.append((Tb.sum() / Vb.sum(), np.sqrt((B - 1) * jk.var())))
    (m1, e1), (m2, e2) = results
    assert m1 > 0 and m2 > 0                          # actual signal, not 0 == 0
    assert abs(m1 - m2) < 5 * np.hypot(e1, e2)


def test_lighten_policy_canned():
    # After freezing the probed table, light-by-policy divides w[k] by
    # lighten^k so production sits below sector coexistence.
    S = _action()
    t = DefectGasWeightTuner(S, D_max=4, rng=np.random.default_rng(5))

    def probe(w, cfg, probe_sweeps, companion_every):
        t_ = np.array([400, 300, 200])
        return t_, t_ // 2, t_ - t_ // 2, 8, cfg

    t._probe = probe
    w_ref, _ = t.tune(probe_sweeps=100, lighten=1.0)
    t2 = DefectGasWeightTuner(S, D_max=4, rng=np.random.default_rng(5))
    t2._probe = probe
    w_light, _ = t2.tune(probe_sweeps=100)              # default lighten=1.5
    assert np.allclose(w_light, w_ref / 1.5 ** np.arange(3))


def test_tune_umbrella_canned():
    import supervillain.generator.no_intersection.defect_gas as dg

    S = _action()
    _, values = dg.pair_shells(4)
    t = DefectGasWeightTuner(S, D_max=4, rng=np.random.default_rng(6))
    calls = []
    # Shell dwell histograms (per-bin): first lopsided, then flat -> converge.
    canned = [np.geomspace(1000.0, 1.0, len(values)),
              np.full(len(values), 50.0)]

    def uprobe(w, w2, cfg, probe_sweeps, companion_every):
        calls.append(w2.copy())
        h = canned[min(len(calls) - 1, 1)]
        return h, h / 2, h - h / 2, 8, cfg

    t._probe_umbrella = uprobe
    w2 = t.tune_umbrella(np.array([1.0, 0.04, 2.4e-3]), probe_sweeps=100)
    assert len(calls) == 2
    # First update: visited shells scaled toward the mean, clipped at
    # max_update, then dwell-renormalized; the frozen table is the probed one.
    assert np.allclose(w2, calls[1])
    assert w2.shape == values.shape and np.all(w2 > 0)


def test_tune_umbrella_scoring_health_tier_canned():
    # Round-trip health must be a boolean TIER, not a magnitude: a probe with
    # MORE round trips but worse coverage/flatness must not beat a probe that
    # clears min_round_trips and is flatter/fully-visited.  None of the three
    # canned probes converge (each fails a different convergence condition),
    # so tune_umbrella freezes the best-SCORED probed table on cap expiry.
    import supervillain.generator.no_intersection.defect_gas as dg

    S = _action()
    _, values = dg.pair_shells(4)
    n = len(values)
    t = DefectGasWeightTuner(S, D_max=4, rng=np.random.default_rng(6))
    calls = []
    # probe 0: huge round trips (100), but one shell is unvisited and the
    #          visited shells are wildly lopsided -- under the OLD scoring
    #          (trips, visited, -flatness) the raw trip count alone would
    #          make this win outright.
    h0 = np.geomspace(1000.0, 1.0, n)
    h0[n // 2] = 0.0                              # leave one shell unvisited
    # probe 1: healthy trips (5, >= min_round_trips), every shell visited,
    #          flattest of the three (within 2x) -- fails convergence only on
    #          stationarity (the two halves are deliberately uneven).  Under
    #          the NEW health-tier scoring this is the winner.
    h1 = np.linspace(20.0, 40.0, n)
    h1_1, h1_2 = 0.1 * h1, 0.9 * h1               # uneven halves: non-stationary
    # probe 2: healthy trips, every shell visited, stationary halves, but
    #          flatness (20x) exceeds the convergence threshold (flat=3.0).
    h2 = np.geomspace(100.0, 5.0, n)
    h2_1, h2_2 = h2 / 2, h2 / 2

    canned = [
        (h0, h0 / 2, h0 / 2, 100),
        (h1, h1_1, h1_2, 5),
        (h2, h2_1, h2_2, 5),
    ]

    def uprobe(w, w2, cfg, probe_sweeps, companion_every):
        calls.append(w2.copy())
        h, ha, hb, trips = canned[min(len(calls) - 1, 2)]
        return h, ha, hb, trips, cfg

    t._probe_umbrella = uprobe
    w2 = t.tune_umbrella(np.array([1.0, 0.04, 2.4e-3]), probe_sweeps=100,
                         max_iterations=3, min_round_trips=5)
    assert len(calls) == 3      # none converged: all three iterations ran

    # Honestly recompute the expected winner from the implemented scoring
    # (int(trips >= min_round_trips), shells visited, -flatness), and confirm
    # the frozen table is the w2 that probe actually saw (best is scored on
    # the w2 IN FLIGHT when it was probed, not a post-update table).
    best = None
    for idx, (h, ha, hb, trips) in enumerate(canned):
        visited = h > 0
        flatness = (h[visited].max() / h[visited].min()) if visited.any() else np.inf
        score = (int(trips >= 5), int(visited.sum()), -flatness)
        if best is None or score > best[0]:
            best = (score, idx)
    _, winner = best
    assert winner == 1                                # the healthy-flattest probe, not the max-trips one (0)
    assert np.allclose(w2, calls[winner])
    assert w2.shape == (n,) and np.all(w2 > 0)


def test_tune_umbrella_warm_start_canned():
    import supervillain.generator.no_intersection.defect_gas as dg

    S = _action()
    _, values = dg.pair_shells(4)
    n = len(values)
    t = DefectGasWeightTuner(S, D_max=4, rng=np.random.default_rng(6))
    calls = []

    def uprobe(w, w2, cfg, probe_sweeps, companion_every):
        calls.append(w2.copy())
        h = np.full(n, 50.0)                      # flat and healthy: converge immediately
        return h, h / 2, h / 2, 8, cfg

    t._probe_umbrella = uprobe
    w2_0 = np.geomspace(5.0, 0.1, n)
    t.tune_umbrella(np.array([1.0, 0.04, 2.4e-3]), probe_sweeps=100, warmStart=w2_0)
    assert len(calls) == 1                        # converged on the very first probe
    assert np.array_equal(calls[0], w2_0)          # iteration 0 probed exactly w2_0

    with pytest.raises(ValueError):
        t.tune_umbrella(np.array([1.0, 0.04, 2.4e-3]), probe_sweeps=100,
                        warmStart=np.ones(n - 1))               # wrong shape
    with pytest.raises(ValueError):
        bad = np.ones(n)
        bad[0] = -1.0
        t.tune_umbrella(np.array([1.0, 0.04, 2.4e-3]), probe_sweeps=100,
                        warmStart=bad)                           # non-positive


def test_tune_umbrella_real_tiny(caplog):
    S = _action()
    t = DefectGasWeightTuner(S, D_max=8, rng=np.random.default_rng(7))
    w, _ = t.tune(probe_sweeps=400, max_iterations=8, lighten=1.5)
    w2 = t.tune_umbrella(w, probe_sweeps=2000, max_iterations=12, max_probe_growth=8)
    assert np.all(w2 > 0)
    # Validation bullet 5: convergence must be genuinely achievable here, not
    # just a freeze-on-cap-expiry that happens to leave every entry positive.
    assert not any('tune_umbrella' in r.message for r in caplog.records)


def test_generator_with_umbrella():
    S = _action()
    t = DefectGasWeightTuner(S, D_max=8, rng=np.random.default_rng(11))
    chain = t.generator(probe_sweeps=400, max_iterations=6, umbrella=True)
    gas = chain.generators[-1]
    assert gas.pairSeparationUmbrella.size > 0
    assert np.array_equal(chain.pairSeparationUmbrella, gas.pairSeparationUmbrella)
    e = supervillain.Ensemble(S).generate(3, chain)
    assert np.all(np.asarray(e.Vacuum_Ticks) > 0)
