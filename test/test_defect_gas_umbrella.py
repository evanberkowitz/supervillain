#!/usr/bin/env python
r"""
The pair-separation umbrella w2(r): a learned shell table multiplying the
enlarged-ensemble weight in the D=2 (single unit pair) and D=4 (four-unit,
Wick-sum) sectors.  An empty table is the off switch, preserving all prior
paths bit-for-bit.
"""

import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection import DefectGas
from supervillain.generator.no_intersection.defect_gas import pair_shells


def _action(N=4, kappa=0.05):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def _cold(S):
    L = S.Lattice
    return (np.zeros((1,) + tuple(L.dims)),
            np.zeros((4,) + tuple(L.dims), dtype=np.int64))


def _twins(S, seed=17, **kwargs):
    return (DefectGas(S, rng=np.random.default_rng(seed), **kwargs),
            DefectGas(S, rng=np.random.default_rng(seed), **kwargs))


def test_pair_shells():
    N = 4
    lookup, values = pair_shells(N)
    # Realized min-image r^2 on N=4: components in {0, 1, 4}; r^2 = 0 excluded.
    d = np.minimum(np.arange(N), N - np.arange(N))**2
    rsq = (d[:, None, None, None] + d[None, :, None, None]
           + d[None, None, :, None] + d[None, None, None, :]).ravel()
    expected = np.unique(rsq[rsq > 0])
    assert np.array_equal(values, expected)
    assert lookup.shape == (N**2 + 1,)
    for r2 in range(N**2 + 1):
        if r2 in set(expected.tolist()):
            assert values[lookup[r2]] == r2
        else:
            assert lookup[r2] == -1


def test_w2_constructor():
    S = _action()
    _, values = pair_shells(4)
    w2 = np.linspace(1.0, 2.0, len(values))
    g = DefectGas(S, weights=(1.0, 0.04, 2.4e-3, 5e-5, 4e-6), w2=w2)
    assert np.array_equal(g.w2, w2)
    # The per-displacement field carries w2 by shell and 1.0 at the origin,
    # and is hypercubically symmetric by construction.
    L = Lattice(4, 4)
    f = g._w2_field
    assert f[0, 0, 0, 0] == 1.0
    assert f[1, 0, 0, 0] == f[0, 0, 0, 1] == f[3, 0, 0, 0]
    assert np.allclose(np.asarray(L.symmetrize(f)), f, rtol=0, atol=1e-14)


def test_w2_validation():
    S = _action()
    _, values = pair_shells(4)
    for bad in (np.ones(len(values) - 1),          # wrong length
                -np.ones(len(values)),             # not positive
                np.zeros(len(values))):            # not positive
        try:
            DefectGas(S, fugacity=0.1, w2=bad)
        except ValueError:
            pass
        else:
            assert False, 'bad w2 must raise ValueError'


def test_no_w2_is_sentinel():
    S = _action()
    g = DefectGas(S, fugacity=0.1)
    assert g.w2.size == 0 and g._w2_field is None


def test_ones_table_matches_no_table():
    # w2 identically 1.0 must reproduce the no-umbrella chain decision-for-
    # decision (the accept factor is exactly 1 in every sector; emission
    # divides by 1): the D=4 branch's averaged Wick sum makes an all-ones w2
    # neutral there too, so the full table is certified here.
    S = _action()
    _, values = pair_shells(4)
    w = (1.0, 0.04, 2.4e-3, 5e-5, 4e-6)
    off = DefectGas(S, weights=w, emit_every=100, rng=np.random.default_rng(5))
    on = DefectGas(S, weights=w, w2=np.ones(len(values)), emit_every=100,
                   rng=np.random.default_rng(5))
    phi, n = _cold(S)
    a = {'phi': phi, 'n': n}
    b = {'phi': phi, 'n': n}
    for _ in range(3):
        a = off.step(a)
        b = on.step(b)
        assert np.array_equal(np.asarray(a['n']), np.asarray(b['n']))
        assert np.array_equal(a['Theta_Theta'], b['Theta_Theta'])
        assert a['Vacuum_Ticks'] == b['Vacuum_Ticks']
    assert off.proposed == on.proposed and off.accepted == on.accepted


def test_step_matches_reference_umbrella_D2():
    # Nontrivial w2 with D_max = 2: only the single-pair sector exists, so this
    # certifies the D=2 weight factors on both paths before the Wick sum enters.
    S = _action()
    _, values = pair_shells(4)
    w2 = np.geomspace(1.0, 30.0, len(values))     # strong outward push
    fast, slow = _twins(S, seed=5, weights=(1.0, 0.04), w2=w2, emit_every=100)
    phi, n = _cold(S)
    a = {'phi': phi, 'n': n}
    b = {'phi': phi, 'n': n}
    for _ in range(3):
        a = fast.step(a)
        b = slow.step_reference(b)
        assert np.array_equal(np.asarray(a['n']), np.asarray(b['n']))
        assert np.array_equal(a['Theta_Theta'], b['Theta_Theta'])
        assert a['Vacuum_Ticks'] == b['Vacuum_Ticks']
        assert a['Ticks'] == b['Ticks']
    assert fast.proposed == slow.proposed
    assert fast.accepted == slow.accepted
    assert fast.D_trace == slow.D_trace


def test_step_matches_reference_umbrella_D4():
    # Full table, D_max = 8: the Wick-sum weight and the inverse-weighted
    # float H_four, bit-for-bit across paths, with the quartic sector visited.
    # The outward geomspace(1, 10) push makes the fully-quartic class ({+1,+1,
    # -1,-1}) a genuine trap under the averaged-Wick dynamics -- a step can
    # take millions of ticks to next hit vacuum -- so max_step_sweeps needs
    # real headroom over the DefectGas default (verified empirically: a step
    # here can take ~4e6 ticks against the default 500*n_links = 512000 cap).
    S = _action()
    _, values = pair_shells(4)
    w2 = np.geomspace(1.0, 10.0, len(values))
    fast, slow = _twins(S, seed=5, weights=(1.0, 0.04, 2.4e-3, 5e-5, 4e-6),
                        w2=w2, emit_every=100, max_step_sweeps=20000)
    phi, n = _cold(S)
    a = {'phi': phi, 'n': n}
    b = {'phi': phi, 'n': n}
    four = 0.
    for _ in range(3):
        a = fast.step(a)
        b = slow.step_reference(b)
        assert np.array_equal(np.asarray(a['n']), np.asarray(b['n']))
        assert np.array_equal(a['Theta_Theta'], b['Theta_Theta'])
        assert np.array_equal(a['FourDefectDistribution'],
                              b['FourDefectDistribution'])
        four += a['FourDefectDistribution'].sum()
    assert fast.proposed == slow.proposed
    assert fast.accepted == slow.accepted
    assert four > 0


def test_w2_independence():
    # The estimator is w2-independent: materially different shell tables must
    # agree on the correlator AND the Binder.  kappa = 0.2 (stable vacuum),
    # ratio-of-sums with blocked jackknife, 5 sigma.
    import supervillain.generator.villain as villain
    from supervillain.generator.combining import Sequentially

    S = _action(kappa=0.2)
    _, values = pair_shells(4)
    w = (1.0, 0.09, 8e-4, 8e-6, 8e-8)
    tables = (np.ones(len(values)),
              np.geomspace(1.0, 20.0, len(values)))
    results = []
    for seed, w2 in enumerate(tables):
        gas = DefectGas(S, weights=w, w2=w2, emit_every=200,
                        rng=np.random.default_rng(300 + seed))
        chain = Sequentially((villain.SiteUpdate(S), gas))
        e = supervillain.Ensemble(S).generate(400, chain)
        T = np.asarray(e.Theta_Theta).real[:, 1, 0, 0, 0]
        V = np.asarray(e.Vacuum_Ticks).astype(float)
        B = 20
        n = len(T) // B
        Tb = T[:B * n].reshape(B, n).sum(axis=1)
        Vb = V[:B * n].reshape(B, n).sum(axis=1)
        jk = np.array([(Tb.sum() - Tb[b]) / (Vb.sum() - Vb[b]) for b in range(B)])
        from supervillain.analysis import Bootstrap
        auto = e.autocorrelation_time(observables=('ActionDensity',))
        b = Bootstrap(e.cut(10 * auto).every(max(1, auto)))
        U = np.asarray(b.ThetaBinderCumulant).real
        results.append((Tb.sum() / Vb.sum(), np.sqrt((B - 1) * jk.var()),
                        float(U.mean()), float(U.std())))
    (m1, e1, U1, dU1), (m2, e2, U2, dU2) = results
    assert m1 > 0 and m2 > 0
    assert abs(m1 - m2) < 5 * np.hypot(e1, e2)
    assert abs(U1 - U2) < 5 * max(1e-6, np.hypot(dU1, dU2))
