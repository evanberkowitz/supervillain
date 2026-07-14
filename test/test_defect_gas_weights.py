#!/usr/bin/env python
r"""
DefectGas with a per-sector weight table w[k] (k = D/2) generalizing the geometric
zeta^D pricing.  The table path must validate its inputs, reproduce the fugacity
path when handed a geometric table, and match the pure-python reference chain
bit-for-bit; the geometric path itself is untouched.
"""

import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection import DefectGas


def _action(N=4, kappa=0.05):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def _cold(S):
    L = S.Lattice
    return (np.zeros((1,) + tuple(L.dims)),
            np.zeros((4,) + tuple(L.dims), dtype=np.int64))


def _twins(S, seed=17, **kwargs):
    return (DefectGas(S, rng=np.random.default_rng(seed), **kwargs),
            DefectGas(S, rng=np.random.default_rng(seed), **kwargs))


def test_exactly_one_pricing():
    S = _action()
    for kwargs in ({}, {'fugacity': 0.1, 'weights': (1.0, 0.01)}):
        try:
            DefectGas(S, **kwargs)
        except ValueError:
            pass
        else:
            assert False, f'DefectGas(S, **{kwargs}) must raise ValueError'


def test_weights_normalized_and_pin_D_max():
    S = _action()
    g = DefectGas(S, weights=(2.0, 1.0, 0.5))
    assert g.fugacity is None
    assert g.D_max == 4
    assert np.array_equal(g.w, [1.0, 0.5, 0.25])
    assert g._w1 == 0.5 and g._w4 == 0.25


def test_weights_short_table_has_no_four_sector():
    S = _action()
    g = DefectGas(S, weights=(1.0, 0.25))
    assert g.D_max == 2 and g._w4 == 1.0


def test_weights_D_max_mismatch():
    S = _action()
    try:
        DefectGas(S, weights=(1.0, 0.1, 0.01), D_max=8)
    except ValueError:
        pass
    else:
        assert False, 'disagreeing D_max must raise ValueError'


def test_weights_must_be_positive():
    S = _action()
    for w in ((1.0, 0.0, 0.1), (1.0, -0.1, 0.1), (1.0,)):
        try:
            DefectGas(S, weights=w)
        except ValueError:
            pass
        else:
            assert False, f'weights={w} must raise ValueError'


def test_geometric_path_unchanged():
    S = _action()
    g = DefectGas(S, fugacity=0.1)
    assert g.fugacity == 0.1 and g.w.size == 0 and g.D_max is None
    assert g._w1 == 0.1**2 and g._w4 == 0.1**4


def test_geometric_table_equivalence():
    # A geometric table must reproduce the fugacity path decision-for-decision:
    # same accepted moves, same fields, same tallies, on a shared proposal stream.
    S = _action()
    z, K = 0.1, 4
    geo = DefectGas(S, fugacity=z, D_max=2 * K, emit_every=300,
                    rng=np.random.default_rng(31))
    tab = DefectGas(S, weights=[z**(2 * k) for k in range(K + 1)], emit_every=300,
                    rng=np.random.default_rng(31))
    phi, n = _cold(S)
    a = {'phi': phi, 'n': n}
    b = {'phi': phi, 'n': n}
    for _ in range(3):
        a = geo.step(a)
        b = tab.step(b)
        assert np.array_equal(np.asarray(a['n']), np.asarray(b['n']))
        assert a['Vacuum_Ticks'] == b['Vacuum_Ticks']
        assert a['Ticks'] == b['Ticks']
        assert np.allclose(a['Theta_Theta'], b['Theta_Theta'])
    assert geo.proposed == tab.proposed
    assert geo.accepted == tab.accepted


def test_step_matches_reference_nongeometric_table():
    # The compiled kernel and the pure-python reference must agree bit-for-bit on a
    # deliberately NON-geometric table (not expressible as any zeta^D).
    S = _action()
    # A perturbed-geometric table (rung ratios .04/.06/.021/.08): non-geometric but
    # light enough that the N=4 kappa=0.05 chain still comes home; this seed visits
    # the quartic sector.  (Fatter tables condense here -- the cliff is real.)
    w = (1.0, 0.04, 2.4e-3, 5e-5, 4e-6)
    fast, slow = _twins(S, seed=5, weights=w, emit_every=100)
    phi, n = _cold(S)
    a = {'phi': phi, 'n': n}
    b = {'phi': phi, 'n': n}
    four = 0.
    for _ in range(3):
        a = fast.step(a)
        b = slow.step_reference(b)
        assert np.array_equal(np.asarray(a['n']), np.asarray(b['n']))
        assert np.array_equal(a['Theta_Theta'], b['Theta_Theta'])
        assert np.array_equal(a['FourDefectDistribution'], b['FourDefectDistribution'])
        assert a['Vacuum_Ticks'] == b['Vacuum_Ticks']
        assert a['Ticks'] == b['Ticks']
        four += a['FourDefectDistribution'].sum()
    assert fast.proposed == slow.proposed
    assert fast.accepted == slow.accepted
    assert fast.D_trace == slow.D_trace
    assert four > 0                     # the fat table populates the quartic sector


def test_sector_ticks_and_round_trips():
    # Every tick lands in exactly one sector; vacuum is sector 0; a round trip is a
    # vacuum return after touching the top sector.  Both paths agree exactly.
    S = _action()
    w = (1.0, 0.04, 2.4e-3)                   # D_max = 4: a reachable top sector
    fast, slow = _twins(S, seed=5, weights=w, emit_every=300)
    phi, n = _cold(S)
    a = {'phi': phi, 'n': n}
    b = {'phi': phi, 'n': n}
    trips = 0
    for _ in range(3):
        a = fast.step(a)
        b = slow.step_reference(b)
        assert np.array_equal(a['SectorTicks'], b['SectorTicks'])
        assert a['RoundTrips'] == b['RoundTrips']
        assert a['SectorTicks'].sum() == a['Ticks']
        assert a['SectorTicks'][0] == a['Vacuum_Ticks']
        trips += a['RoundTrips']
    assert trips > 0                          # the chain shuttles to the top and back


def test_uncapped_emits_no_sector_ticks():
    S = _action()
    g = DefectGas(S, fugacity=0.1, emit_every=100, rng=np.random.default_rng(43))
    assert 'SectorTicks' not in g.inline_observables(2)
    assert 'RoundTrips' not in g.inline_observables(2)
    phi, n = _cold(S)
    out = g.step({'phi': phi, 'n': n})
    assert 'SectorTicks' not in out and 'RoundTrips' not in out


def test_probe_sweeps_budget_and_continuity():
    # A sweep-budgeted probe never waits for vacuum; the tick budget is exact from a
    # fresh chain, and two 2-sweep blocks equal one 4-sweep block.
    S = _action()
    w = (1.0, 0.04, 2.4e-3, 5e-5, 4e-6)
    one = DefectGas(S, weights=w, rng=np.random.default_rng(51))
    two = DefectGas(S, weights=w, rng=np.random.default_rng(51))
    phi, n = _cold(S)
    cfg1, t1, r1, _ = one._probe_sweeps({'phi': phi, 'n': n}, 4)
    cfg2, t2a, r2a, _ = two._probe_sweeps({'phi': phi, 'n': n}, 2)
    cfg2, t2b, r2b, _ = two._probe_sweeps(cfg2, 2)
    assert np.array_equal(np.asarray(cfg1['n']), np.asarray(cfg2['n']))
    assert np.array_equal(t1, t2a + t2b)
    assert r1 == r2a + r2b
    assert t1.sum() == 4 * 4 * S.Lattice.N**4       # ticks == sweeps * links, exactly
    assert one.proposed == two.proposed
    assert one.accepted == two.accepted


def test_probe_sweeps_requires_cap():
    S = _action()
    g = DefectGas(S, fugacity=0.1)                  # D_max None
    phi, n = _cold(S)
    try:
        g._probe_sweeps({'phi': phi, 'n': n}, 1)
    except ValueError:
        pass
    else:
        assert False, 'uncapped _probe_sweeps must raise ValueError'


def test_capped_geometric_emits_sector_ticks():
    S = _action()
    g = DefectGas(S, fugacity=0.1, D_max=8, emit_every=100,
                  rng=np.random.default_rng(47))
    obs = g.inline_observables(2)
    assert 'SectorTicks' in obs and 'RoundTrips' in obs
    phi, n = _cold(S)
    out = g.step({'phi': phi, 'n': n})
    assert out['SectorTicks'].shape == (5,)
    assert out['SectorTicks'].sum() == out['Ticks']
