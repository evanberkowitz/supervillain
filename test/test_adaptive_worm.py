#!/usr/bin/env python

import numpy as np
import supervillain
import supervillain.generator.no_intersection as gen
from supervillain.lattice import Lattice, Form, d
from supervillain.generator.no_intersection.charge import charge


def _valid_configs(kappa=0.3, N=6, steps=140, burn=100, stride=8):
    L = Lattice(4, N)
    S = supervillain.action.NoIntersections(L, kappa=kappa)
    H = gen.Hammer(S)
    e = supervillain.Ensemble(S).generate(steps, H, start='cold')
    return S, [e.configuration[i] for i in range(burn, steps, stride)]


def test_clean_set_members_are_exact_dipoles():
    S, configs = _valid_configs()
    w = gen.AdaptiveIntersectionWorm(S)
    L = S.Lattice
    N = L.N
    rng = np.random.default_rng(0)
    checked = 0
    for cfg in configs:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        q0 = charge(cfg['n'])
        for _ in range(20):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            dd = w._ortho[int(rng.integers(0, len(w._ortho)))]
            sign = +1 if rng.integers(0, 2) == 0 else -1
            target = tuple((head[k] + sign * dd[k]) % N for k in range(4))
            want = {head: -1, target: 1}
            for change, tgt in w.clean_set_reference(n_arr, q0, head, dd, sign):
                trial = n_arr.copy()
                for link, c in change.items():
                    trial[link] += c
                dq = charge(Form(trial, degree=1, lattice=L)) - q0
                defects = {tuple(int(x) for x in z[1:]): int(dq[tuple(z)])
                           for z in np.argwhere(dq != 0)}
                assert defects == want and tgt == target
                checked += 1
    assert checked > 0  # the ensemble really exercised some clean moves


def test_head_move_detailed_balance_and_involution():
    S, configs = _valid_configs()
    w = gen.AdaptiveIntersectionWorm(S)
    L = S.Lattice
    N = L.N
    rng = np.random.default_rng(1)
    two_M = 2 * len(w._ortho)
    tested = 0
    for cfg in configs:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        dphi = np.asarray(d(cfg['phi']))
        q0 = charge(cfg['n'])
        for _ in range(20):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            dd = w._ortho[int(rng.integers(0, len(w._ortho)))]
            sign = +1 if rng.integers(0, 2) == 0 else -1
            C = w.clean_set_reference(n_arr, q0, head, dd, sign)
            if not C:
                continue
            change, target = C[int(rng.integers(0, len(C)))]
            trial = n_arr.copy()
            for lnk, c in change.items():
                trial[lnk] += c
            q1 = charge(Form(trial, degree=1, lattice=L))
            Cp = w.clean_set_reference(trial, q1, target, dd, -sign)
            # involution: m^{-1} (negated change) is in C'
            inv = frozenset((lnk, -c) for lnk, c in change.items() if c != 0)
            assert any(frozenset((l, c) for l, c in ch.items() if c != 0) == inv
                       for ch, _ in Cp)
            # elementary detailed balance
            dS = w._delta_S(dphi, n_arr, change)
            q_fwd = (1.0 / two_M) / len(C)
            q_rev = (1.0 / two_M) / len(Cp)
            A_fwd = min(1.0, (len(C) / len(Cp)) * np.exp(-dS))
            A_rev = min(1.0, (len(Cp) / len(C)) * np.exp(+dS))
            assert abs(q_fwd * A_fwd - np.exp(-dS) * q_rev * A_rev) < 1e-12
            tested += 1
    assert tested > 0


def test_step_reference_emits_valid_config():
    S, configs = _valid_configs()
    w = gen.AdaptiveIntersectionWorm(S)
    w.rng = np.random.default_rng(7)
    cfg = configs[0]
    for _ in range(10):
        cfg = w.step_reference(cfg)
        q = charge(cfg['n'])
        assert np.abs(q).max() == 0                      # constraint preserved
        assert cfg['Intersection_Intersection'].shape == S.Lattice.dims
        assert np.isfinite(cfg['Worm_Length'])


def test_idle_moves_are_charge_neutral_and_present_cold():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.0)
    w = gen.AdaptiveIntersectionWorm(S)
    cold = S.configurations(1)[0]
    n_arr = np.asarray(cold['n']).astype(np.int64)
    q0 = charge(cold['n'])
    head = (2, 2, 2, 2)
    idles = w.clean_idle_reference(n_arr, q0, head)
    assert len(idles) > 0                                  # cold sheet: idles available
    for change in idles:
        trial = n_arr.copy()
        for lnk, c in change.items():
            trial[lnk] += c
        dq = charge(Form(trial, degree=1, lattice=L)) - q0
        assert np.abs(dq).max() == 0                        # Δq ≡ 0
        inv = {lnk: -c for lnk, c in change.items()}
        assert any(ch == inv for ch in idles)              # inverse also enumerated


def test_idle_detailed_balance():
    S, configs = _valid_configs()
    w = gen.AdaptiveIntersectionWorm(S)
    L = S.Lattice
    rng = np.random.default_rng(3)
    tested = 0
    for cfg in configs:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        dphi = np.asarray(d(cfg['phi']))
        q0 = charge(cfg['n'])
        for _ in range(15):
            head = tuple(int(x) for x in rng.integers(0, L.N, size=4))
            I = w.clean_idle_reference(n_arr, q0, head)
            if not I:
                continue
            change = I[int(rng.integers(0, len(I)))]
            trial = n_arr.copy()
            for lnk, c in change.items():
                trial[lnk] += c
            q1 = charge(Form(trial, degree=1, lattice=L))
            Ip = w.clean_idle_reference(trial, q1, head)
            dS = w._delta_S(dphi, n_arr, change)
            A_fwd = min(1.0, (len(I) / len(Ip)) * np.exp(-dS))
            A_rev = min(1.0, (len(Ip) / len(I)) * np.exp(+dS))
            assert abs((A_fwd / len(I)) - np.exp(-dS) * (A_rev / len(Ip))) < 1e-12
            tested += 1
    assert tested > 0


def test_step_matches_reference_bit_for_bit():
    S, configs = _valid_configs()
    cfg = configs[0]
    a = gen.AdaptiveIntersectionWorm(S)
    b = gen.AdaptiveIntersectionWorm(S)
    for _ in range(8):
        a.rng = np.random.default_rng(2024)
        b.rng = np.random.default_rng(2024)
        ra = a.step(cfg)
        rb = b.step_reference(cfg)
        assert np.array_equal(np.asarray(ra['n']), np.asarray(rb['n']))
        assert np.array_equal(ra['Intersection_Intersection'],
                              rb['Intersection_Intersection'])
        assert ra['Worm_Length'] == rb['Worm_Length']
        cfg = rb


def test_runs_in_ensemble_and_normalizes():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.3)
    w = gen.AdaptiveIntersectionWorm(S)
    e = supervillain.Ensemble(S).generate(60, w, start='cold')
    q2 = np.asarray(e.TopologicalChargeDensitySquared)
    assert np.abs(q2).max() == 0                           # every emitted config valid
    theta = np.asarray(e.Intersection_Intersection)         # (steps,) + dims
    # The normalizer is the ENSEMBLE-averaged origin bin (a single immediately-closed
    # worm contributes an all-zero histogram, so per-worm origins can be 0).
    mean = theta.mean(axis=0)
    assert mean[L.origin] > 0                               # <Theta_0> populated
    normalized = mean / mean[L.origin]                      # the DerivedQuantity's formula
    assert normalized[L.origin] == 1
    assert 'adaptive worms' in w.report()
