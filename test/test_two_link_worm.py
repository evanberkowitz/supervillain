#!/usr/bin/env python

import time

import numpy as np
import supervillain
import supervillain.generator.no_intersection as gen
from supervillain.lattice import Lattice, Form, d
from supervillain.generator.no_intersection.charge import charge


def _action(kappa=0.3, N=5):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def _valid_flux_config(S, seed=23, steps=6):
    # A valid (q ≡ 0) configuration carrying flux, built from constraint-preserving moves.
    g = gen.PlanarFluxUpdate(S)
    g.rng = np.random.default_rng(seed)
    cfg = S.configurations(1)[0]
    for _ in range(steps):
        cfg = g.step(cfg)
    assert S.valid(cfg)
    return cfg


def test_two_link_mover_family_shapes_well_formed():
    S = _action()
    w = gen.TwoLinkAdaptiveWorm(S)
    assert set(w._two_movers) == set(w._ortho)
    for dd, shapes in w._two_movers.items():
        assert len(shapes) > 0
        for shape in shapes:
            assert len(shape) == 2                                   # two links
            links = {(mu,) + tuple(rs) for mu, rs, c in shape}
            assert len(links) == 2                                   # distinct links
            for mu, rs, c in shape:
                assert c in gen.TwoLinkAdaptiveWorm.COEFF_BOX
            assert shape in w._self_charge                           # self-charge registered


def test_two_link_self_charge_matches_global_recompute():
    # The registered self-charge (dΔn∧dΔn, background-independent) must reproduce a global
    # charge recompute of the placed shape at a random anchor.  Derivation is on a lattice
    # of the SAME extent, so small-N periodic-image cross terms are exact.
    N = 5
    S = _action(N=N)
    L = S.Lattice
    w = gen.TwoLinkAdaptiveWorm(S)
    rng = np.random.default_rng(11)
    checked = 0
    for dd, shapes in w._two_movers.items():
        for shape in shapes[:: max(1, len(shapes) // 25)]:           # sample to stay fast
            anchor = tuple(int(x) for x in rng.integers(0, N, size=4))
            dn = L.zeros(1, dtype=int)   # already a Form
            for mu, rs, c in shape:
                dn[(mu,) + tuple((anchor[k] + rs[k]) % N for k in range(4))] += c
            q = np.asarray(charge(dn))
            got = {tuple(int(x) for x in h[1:]): int(q[tuple(h)]) for h in np.argwhere(q != 0)}
            expect = {}
            for off, v in w._self_charge[shape]:
                cell = tuple((anchor[k] + off[k]) % N for k in range(4))
                expect[cell] = expect.get(cell, 0) + v
            assert got == {c: v for c, v in expect.items() if v}
            checked += 1
    assert checked > 0


def test_two_link_clean_set_local_matches_reference():
    # The local-stencil clean set (fast) must equal the global-recompute clean set
    # (reference), member-for-member and in the same order, on real flux backgrounds.
    S = _action()
    L = S.Lattice
    N = L.N
    w = gen.TwoLinkAdaptiveWorm(S)
    cfg = _valid_flux_config(S)
    n_arr = np.asarray(cfg['n']).astype(np.int64)
    q0 = charge(cfg['n'])
    F = np.asarray(d(cfg['n'])).astype(np.int64)
    rng = np.random.default_rng(4)
    for _ in range(40):
        head = tuple(int(x) for x in rng.integers(0, N, size=4))
        dd = w._ortho[int(rng.integers(0, len(w._ortho)))]
        sign = +1 if rng.integers(0, 2) == 0 else -1
        loc = w.clean_set_local(F, head, dd, sign)
        ref = w.clean_set_reference(n_arr, q0, head, dd, sign)
        assert [ch for ch, _ in loc] == [ch for ch, _ in ref]
        assert [t for _, t in loc] == [t for _, t in ref]


def test_two_link_mover_fires_on_flux_background():
    # The enrichment is pointless unless some clean move in a clean set is genuinely a
    # two-link shape.  Over many head/direction draws on a flux background, at least one.
    S = _action()
    L = S.Lattice
    N = L.N
    w = gen.TwoLinkAdaptiveWorm(S)
    cfg = _valid_flux_config(S)
    F = np.asarray(d(cfg['n'])).astype(np.int64)
    two_link_seen = 0
    rng = np.random.default_rng(5)
    for _ in range(200):
        head = tuple(int(x) for x in rng.integers(0, N, size=4))
        dd = w._ortho[int(rng.integers(0, len(w._ortho)))]
        sign = +1 if rng.integers(0, 2) == 0 else -1
        for change, _t in w.clean_set_local(F, head, dd, sign):
            if len([c for c in change.values() if c != 0]) == 2:
                two_link_seen += 1
    assert two_link_seen > 0


def test_two_link_idle_family_well_formed():
    S = _action()
    w = gen.TwoLinkAdaptiveWorm(S)
    assert len(w._two_idles) > 0
    for shape in w._two_idles:
        assert len(shape) == 2
        assert len({(mu,) + tuple(rs) for mu, rs, c in shape}) == 2   # distinct links
        for mu, rs, c in shape:
            assert c in gen.TwoLinkAdaptiveWorm.COEFF_BOX
        assert shape in w._self_charge


def test_two_link_idle_clean_set_local_matches_reference():
    S = _action()
    L = S.Lattice
    N = L.N
    w = gen.TwoLinkAdaptiveWorm(S)
    cfg = _valid_flux_config(S)
    n_arr = np.asarray(cfg['n']).astype(np.int64)
    q0 = charge(cfg['n'])
    F = np.asarray(d(cfg['n'])).astype(np.int64)
    rng = np.random.default_rng(6)
    for _ in range(30):
        head = tuple(int(x) for x in rng.integers(0, N, size=4))
        assert w.clean_idle_local(F, head) == w.clean_idle_reference(n_arr, q0, head)


def test_two_link_idles_are_charge_neutral_and_inverse_present():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.0)
    w = gen.TwoLinkAdaptiveWorm(S)
    cold = S.configurations(1)[0]
    n_arr = np.asarray(cold['n']).astype(np.int64)
    q0 = charge(cold['n'])
    head = (2, 2, 2, 2)
    idles = w.clean_idle_reference(n_arr, q0, head)
    two_link = [ch for ch in idles if len([c for c in ch.values() if c != 0]) == 2]
    assert len(two_link) > 0                                         # 2-link idles present
    for change in idles:
        trial = n_arr.copy()
        for lnk, c in change.items():
            trial[lnk] += c
        dq = charge(Form(trial, degree=1, lattice=L)) - q0
        assert np.abs(dq).max() == 0                                 # Δq ≡ 0
        inv = {lnk: -c for lnk, c in change.items()}
        assert any(ch == inv for ch in idles)                        # inverse enumerated


def test_construction_is_fast():
    # The M-pattern build (one charge() per link-pair, scaled by c1*c2) must be far
    # cheaper than a charge() per (pair, c1, c2).  A generous ceiling that still fails
    # loudly if the per-coefficient recompute ever returns.
    S = _action(N=5)
    t0 = time.perf_counter()
    supervillain.generator.no_intersection.TwoLinkAdaptiveWorm(S)
    assert time.perf_counter() - t0 < 8.0


def test_self_charge_is_c1c2_times_unit_cross_term():
    # Self-charge scales exactly linearly in c1*c2: the (c1,c2) shape's registered
    # self-charge equals c1*c2 times the unit (1,1) shape's, value-for-value.
    S = _action(N=5)
    w = supervillain.generator.no_intersection.TwoLinkAdaptiveWorm(S)
    checked = 0
    for dd, shapes in w._two_movers.items():
        for shape in shapes[:: max(1, len(shapes) // 40)]:
            (mu1, r1, c1), (mu2, r2, c2) = shape
            unit = ((mu1, r1, 1), (mu2, r2, 1))
            m = dict(w._self_charge[unit]) if unit in w._self_charge else \
                dict(w._shape_self_charge(unit))
            got = dict(w._self_charge[shape])
            assert got == {off: c1 * c2 * v for off, v in m.items() if c1 * c2 * v}
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# Exactness gate.  The detailed-balance loops use the FAST local clean sets
# (clean_set_local / clean_idle_local) rather than the O(family) global oracle:
# clean_set_local == clean_set_reference is independently established by
# test_two_link_clean_set_local_matches_reference (and the idle twin).  What
# actually gates correctness here is (i) a per-background global-oracle
# cross-check run on a clean set that CONTAINS a two-link move -- so the two-link
# enumeration itself is validated against a full charge() recompute on every
# background -- and (ii) the involution check, run EXPLICITLY on a two-link
# member (not only on a random draw, which the base library dominates).  The
# numeric balance equation is, as in the base worm's tests, an algebraic identity
# for any positive |C|, |C'|; it catches formula-transcription bugs (a flipped
# ratio or dS sign), while the oracle cross-check and involution catch
# enumeration-content bugs.
# ---------------------------------------------------------------------------


def _flux_configs(S, seeds=(23, 41, 59)):
    return [_valid_flux_config(S, seed=s, steps=6) for s in seeds]


def _two_link(change):
    return len([c for c in change.values() if c != 0]) == 2


def test_two_link_head_move_detailed_balance():
    # Elementary detailed balance + involution for the ENRICHED head-move clean set.
    # Every background is oracle-cross-checked on a clean set containing a two-link move,
    # and every such move is run through the involution + balance explicitly.
    S = _action(N=5)
    L = S.Lattice
    N = L.N
    w = gen.TwoLinkAdaptiveWorm(S)
    two_M = 2 * len(w._ortho)
    rng = np.random.default_rng(1)
    tested = 0
    two_link_tested = 0
    oracle_on_two_link = 0
    maxerr = 0.0
    for cfg in _flux_configs(S):
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        dphi = np.asarray(d(cfg['phi']))
        q0 = charge(cfg['n'])
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        cfg_crosschecked = False
        for _ in range(8):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            dd = w._ortho[int(rng.integers(0, len(w._ortho)))]
            sign = +1 if rng.integers(0, 2) == 0 else -1
            C = w.clean_set_local(F, head, dd, sign)
            if not C:
                continue
            twolinks = [(ch, t) for ch, t in C if _two_link(ch)]
            # Per-background global-oracle cross-check, on a clean set that contains a
            # two-link move so the recompute validates the two-link enumeration itself.
            if not cfg_crosschecked and twolinks:
                assert C == w.clean_set_reference(n_arr, q0, head, dd, sign)
                oracle_on_two_link += 1
                cfg_crosschecked = True
            # Test the random draw AND, when present, a two-link member explicitly.
            moves = [C[int(rng.integers(0, len(C)))]]
            if twolinks:
                moves.append(twolinks[0])
            for change, target in moves:
                trial = n_arr.copy()
                for lnk, c in change.items():
                    trial[lnk] += c
                Fp = np.asarray(d(Form(trial, degree=1, lattice=L))).astype(np.int64)
                Cp = w.clean_set_local(Fp, target, dd, -sign)
                inv = frozenset((lnk, -c) for lnk, c in change.items() if c != 0)
                assert any(frozenset((l, c) for l, c in ch.items() if c != 0) == inv
                           for ch, _ in Cp)            # involution: m^{-1} in C'
                dS = w._delta_S(dphi, n_arr, change)
                q_fwd = (1.0 / two_M) / len(C)
                q_rev = (1.0 / two_M) / len(Cp)
                A_fwd = min(1.0, (len(C) / len(Cp)) * np.exp(-dS))
                A_rev = min(1.0, (len(Cp) / len(C)) * np.exp(+dS))
                maxerr = max(maxerr, abs(q_fwd * A_fwd - np.exp(-dS) * q_rev * A_rev))
                tested += 1
                two_link_tested += _two_link(change)
    assert tested > 0
    assert two_link_tested > 0, 'no two-link move run through involution + balance'
    assert oracle_on_two_link == len(_flux_configs(S)), \
        'a background lacked a global-oracle cross-check on a two-link-bearing clean set'
    assert maxerr < 1e-12


def test_two_link_idle_detailed_balance():
    # Elementary detailed balance + inverse-presence for the ENRICHED idle set.  Every
    # background is oracle-cross-checked on an idle set containing a two-link idle, and
    # every such idle is run through the involution + balance explicitly.
    S = _action(N=5)
    L = S.Lattice
    N = L.N
    w = gen.TwoLinkAdaptiveWorm(S)
    rng = np.random.default_rng(3)
    tested = 0
    two_link_tested = 0
    oracle_on_two_link = 0
    maxerr = 0.0
    for cfg in _flux_configs(S):
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        dphi = np.asarray(d(cfg['phi']))
        q0 = charge(cfg['n'])
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        cfg_crosschecked = False
        for _ in range(8):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            I = w.clean_idle_local(F, head)
            if not I:
                continue
            twolinks = [ch for ch in I if _two_link(ch)]
            if not cfg_crosschecked and twolinks:
                assert I == w.clean_idle_reference(n_arr, q0, head)
                oracle_on_two_link += 1
                cfg_crosschecked = True
            moves = [I[int(rng.integers(0, len(I)))]]
            if twolinks:
                moves.append(twolinks[0])
            for change in moves:
                trial = n_arr.copy()
                for lnk, c in change.items():
                    trial[lnk] += c
                Fp = np.asarray(d(Form(trial, degree=1, lattice=L))).astype(np.int64)
                Ip = w.clean_idle_local(Fp, head)
                inv = {lnk: -c for lnk, c in change.items()}
                assert any(ch == inv for ch in Ip)      # reverse idle present
                dS = w._delta_S(dphi, n_arr, change)
                A_fwd = min(1.0, (len(I) / len(Ip)) * np.exp(-dS))
                A_rev = min(1.0, (len(Ip) / len(I)) * np.exp(+dS))
                maxerr = max(maxerr, abs((A_fwd / len(I)) - np.exp(-dS) * (A_rev / len(Ip))))
                tested += 1
                two_link_tested += _two_link(change)
    assert tested > 0
    assert two_link_tested > 0, 'no two-link idle run through involution + balance'
    assert oracle_on_two_link == len(_flux_configs(S)), \
        'a background lacked a global-oracle cross-check on a two-link-bearing idle set'
    assert maxerr < 1e-12


def test_two_link_step_matches_reference_bit_for_bit():
    # The compiled step (numba clean_mask + incrementally maintained F) must reproduce the
    # pure-Python local step_reference EXACTLY on a shared seed.  Small N and few worms:
    # step_reference enumerates the ~15k-shape family in interpreted Python per proposal.
    S = _action(N=4)
    cfg = _valid_flux_config(S, seed=23, steps=5)
    a = gen.TwoLinkAdaptiveWorm(S)
    b = gen.TwoLinkAdaptiveWorm(S)
    for _ in range(2):
        a.rng = np.random.default_rng(2024)
        b.rng = np.random.default_rng(2024)
        ra = a.step(cfg)
        rb = b.step_reference(cfg)
        assert np.array_equal(np.asarray(ra['n']), np.asarray(rb['n']))
        assert np.array_equal(ra['Intersection_Intersection'],
                              rb['Intersection_Intersection'])
        assert ra['Worm_Length'] == rb['Worm_Length']
        cfg = rb


def test_two_link_runs_in_ensemble_and_stays_valid():
    S = _action(N=4)
    w = gen.TwoLinkAdaptiveWorm(S)
    e = supervillain.Ensemble(S).generate(6, w, start='cold')
    q2 = np.asarray(e.TopologicalChargeDensitySquared)
    assert np.abs(q2).max() == 0                          # every emitted config valid
    assert np.asarray(e.Intersection_Intersection).shape == (len(e),) + S.Lattice.dims


# ---------------------------------------------------------------------------
# Compiled clean-set kernel: bit-for-bit against the pure-Python enumeration.
# ---------------------------------------------------------------------------


def test_numba_clean_set_matches_python():
    # The njit clean_set_local must equal the pure-Python _clean_set_local_py
    # member-and-order (change dicts and targets) on real flux backgrounds.
    S = _action(N=5)
    w = gen.TwoLinkAdaptiveWorm(S)
    N = S.Lattice.N
    rng = np.random.default_rng(7)
    for cfg in _flux_configs(S):
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        for _ in range(20):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            dd = w._ortho[int(rng.integers(0, len(w._ortho)))]
            sign = +1 if rng.integers(0, 2) == 0 else -1
            assert w.clean_set_local(F, head, dd, sign) == \
                w._clean_set_local_py(F, head, dd, sign)


def test_numba_clean_idle_matches_python():
    # The njit clean_idle_local must equal the pure-Python _clean_idle_local_py.
    S = _action(N=5)
    w = gen.TwoLinkAdaptiveWorm(S)
    N = S.Lattice.N
    rng = np.random.default_rng(8)
    for cfg in _flux_configs(S):
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        for _ in range(20):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            assert w.clean_idle_local(F, head) == w._clean_idle_local_py(F, head)
