#!/usr/bin/env python

import numpy as np
import pytest
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
