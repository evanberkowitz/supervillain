#!/usr/bin/env python

import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice, d
from supervillain.generator.no_intersection.charge import charge


def _action(kappa=0.3, N=5):
    L = Lattice(4, N)
    return supervillain.action.NoIntersections(L, kappa=kappa)


def _cold(S):
    return S.configurations(1)[0]


def _worm(S, seed=None):
    w = supervillain.generator.no_intersection.IntersectionWorm(S)
    if seed is not None:
        w.rng = np.random.default_rng(seed)
    return w


FAMILIES = ('ortho3', 'ortho2', 'elbow2', 'same4', '1link')


def test_family_classification_covers_library():
    # Every shape in every bucket gets exactly one of the five family names, and the
    # name matches the bucket geometry (taxicab length, sign pattern) and shape size.
    S = _action()
    worm = _worm(S)
    for dd in worm._directions:
        fams = worm._family[dd]
        assert len(fams) == len(worm._library[dd])
        taxicab = sum(abs(x) for x in dd)
        for fam, shape in zip(fams, worm._library[dd]):
            assert fam in FAMILIES
            if len(shape) == 1:
                assert fam == '1link'
            elif taxicab == 1:
                assert fam == ('ortho2' if len(shape) == 2 else 'ortho3')
            elif sum(dd) == 0:
                assert fam == 'elbow2'
            else:
                assert fam == 'same4'


def test_tallies_are_consistent_after_steps():
    S = _action()
    worm = _worm(S, seed=17)
    cfg = _cold(S)
    for _ in range(10):
        cfg = worm.step(cfg)
    total_drawn = 0
    for fam, t in worm.tallies.items():
        assert t['drawn'] == t['unclean'] + t['clean'] + t['idle']
        assert t['accepted'] <= t['clean']
        assert t['accepted_idle'] <= t['idle']
        total_drawn += t['drawn']
    assert total_drawn > 0
    assert 'drawn' in worm.report()


def test_one_link_self_charge_is_empty():
    # A single link's self-wedge d(delta)∧d(delta) vanishes identically, so its
    # self-charge pattern must be empty.
    S = _action()
    worm = _worm(S)
    for dd in worm._directions:
        for shape in worm._library[dd]:
            if len(shape) == 1:
                assert worm._self_charge[shape] == ()


def test_self_charge_matches_global_recompute_at_random_anchor():
    # The pattern is derived at one anchor; verify it translates: placing the template
    # at a random anchor and recomputing charge globally must reproduce the stored
    # pattern shifted to that anchor.  Derivation is on a lattice of the SAME extent,
    # so small-N wrap-around cross terms are captured exactly.
    N = 5
    S = _action(N=N)
    L = S.Lattice
    worm = _worm(S)
    rng = np.random.default_rng(11)
    for dd in worm._directions:
        for shape in worm._library[dd]:
            anchor = tuple(int(x) for x in rng.integers(0, N, size=4))
            dn = L.zeros(1, dtype=int)
            for mu, rs, c in shape:
                site = tuple((anchor[k] + rs[k]) % N for k in range(4))
                dn[(mu,) + site] += c
            q = np.asarray(charge(dn))
            got = {tuple(int(x) for x in h[1:]): int(q[tuple(h)])
                   for h in np.argwhere(q != 0)}
            expect = {}
            for off, v in worm._self_charge[shape]:
                cell = tuple((anchor[k] + off[k]) % N for k in range(4))
                expect[cell] = expect.get(cell, 0) + v
            assert got == {cell: v for cell, v in expect.items() if v}


def _flux_background(S, seed=23, steps=6):
    # A valid configuration with a nonzero sheet, built from constraint-preserving
    # updates so S.valid holds by construction.
    gen = supervillain.generator.no_intersection.PlanarFluxUpdate(S)
    gen.rng = np.random.default_rng(seed)
    cfg = _cold(S)
    for _ in range(steps):
        cfg = gen.step(cfg)
    assert S.valid(cfg)
    return cfg


@pytest.mark.parametrize('N', (4, 5))
@pytest.mark.parametrize('seed', (3, 5))
def test_local_dq_matches_global_recompute(N, seed):
    # The identity Δq = F∧dΔn + dΔn∧F + dΔn∧dΔn holds for ARBITRARY integer n --
    # valid or not, mid-worm or not -- so a random background is the strongest test.
    S = _action(N=N)
    L = S.Lattice
    worm = _worm(S)
    rng = np.random.default_rng(seed)
    n = L.zeros(1, dtype=int)
    n += rng.integers(-1, 2, size=n.shape)
    q0 = charge(n)
    F = np.asarray(d(n)).astype(np.int64)
    for _ in range(50):
        head = tuple(int(x) for x in rng.integers(0, N, size=4))
        dd = worm._directions[rng.integers(0, len(worm._directions))]
        sign = 1 if rng.integers(0, 2) == 0 else -1
        shapes = worm._library[dd]
        shape = shapes[rng.integers(0, len(shapes))]
        change = worm._change_from_shape(head, dd, sign, shape)
        anchor = tuple((head[k] + dd[k]) % N for k in range(4)) if sign > 0 else head
        local = worm._local_dq(F, change, anchor, shape)
        trial = n.copy()
        for link, c in change.items():
            trial[link] += c
        dq = charge(trial) - q0
        glob = {tuple(int(x) for x in h[1:]): int(dq[tuple(h)])
                for h in np.argwhere(dq != 0)}
        assert local == glob


@pytest.mark.parametrize('seed', (1, 2, 3))
def test_step_matches_reference_bit_for_bit(seed):
    # Same seed, same start => the accelerated step and the global-recompute oracle
    # must draw identical RNG streams and emit identical configurations and inline
    # observables.  Any classification disagreement would desynchronize the streams
    # and fail loudly here.
    S = _action(kappa=0.3, N=5)
    cfg = _flux_background(S)
    fast = _worm(S, seed=seed)
    ref = _worm(S, seed=seed)
    out_f = fast.step(cfg)
    out_r = ref.step_reference(cfg)
    assert np.array_equal(np.asarray(out_f['n']), np.asarray(out_r['n']))
    assert np.array_equal(out_f['Intersection_Intersection'],
                          out_r['Intersection_Intersection'])
    assert out_f['Worm_Length'] == out_r['Worm_Length']


def test_accelerated_step_preserves_validity_and_closes():
    S = _action()
    worm = _worm(S, seed=9)
    cfg = _cold(S)
    for _ in range(10):
        cfg = worm.step(cfg)
        assert S.valid(cfg)


def test_class_weights_validation():
    S = _action()
    W = supervillain.generator.no_intersection.IntersectionWorm
    with pytest.raises(ValueError):
        W(S, class_weights={'bogus': 1.0})
    with pytest.raises(ValueError):
        W(S, class_weights={'ortho2': -1.0})
    # Same-sign diagonal buckets hold only same4 and 1link shapes; zeroing both
    # leaves those buckets with nothing to draw.
    with pytest.raises(ValueError):
        W(S, class_weights={'same4': 0.0, '1link': 0.0})


def test_weighted_draws_respect_zero_weight():
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(
        S, class_weights={'same4': 0.0})
    worm.rng = np.random.default_rng(5)
    cfg = _cold(S)
    for _ in range(10):
        cfg = worm.step(cfg)
        assert S.valid(cfg)
    assert worm.tallies['same4']['drawn'] == 0
    assert worm.tallies['ortho2']['drawn'] > 0


@pytest.mark.parametrize('seed', (4, 8))
def test_weighted_step_matches_reference_bit_for_bit(seed):
    # The weighted draw goes through the shared _draw_shape, so fast and reference
    # consume identical RNG streams at ANY weights.
    S = _action(N=5)
    cfg = _flux_background(S)
    weights = {'ortho2': 3.0, 'ortho3': 1.0, 'elbow2': 1.0, 'same4': 0.5, '1link': 0.25}
    fast = supervillain.generator.no_intersection.IntersectionWorm(S, class_weights=weights)
    ref = supervillain.generator.no_intersection.IntersectionWorm(S, class_weights=weights)
    fast.rng = np.random.default_rng(seed)
    ref.rng = np.random.default_rng(seed)
    out_f = fast.step(cfg)
    out_r = ref.step_reference(cfg)
    assert np.array_equal(np.asarray(out_f['n']), np.asarray(out_r['n']))
