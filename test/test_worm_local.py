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
