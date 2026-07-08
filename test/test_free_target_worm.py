#!/usr/bin/env python

import numpy as np
import supervillain
import supervillain.generator.no_intersection as gen
from supervillain.lattice import Lattice, Form, d
from supervillain.generator.no_intersection.charge import charge


def _worm(N=6, kappa=0.3):
    L = Lattice(4, N)
    S = supervillain.action.NoIntersections(L, kappa=kappa)
    return gen.FreeTargetWorm(S)


def _key(shape):
    return frozenset(shape)


def _valid_configs(kappa=0.3, N=6, steps=140, burn=100, stride=8):
    L = Lattice(4, N)
    S = supervillain.action.NoIntersections(L, kappa=kappa)
    H = gen.Hammer(S)
    e = supervillain.Ensemble(S).generate(steps, H, start='cold')
    return S, [e.configuration[i] for i in range(burn, steps, stride)]


def _apply(n_arr, change):
    trial = n_arr.copy()
    for link, c in change.items():
        trial[link] += c
    return trial


def _defects(L, trial, q0):
    dq = charge(Form(trial, degree=1, lattice=L)) - q0
    return {tuple(int(x) for x in z[1:]): int(dq[tuple(z)])
            for z in np.argwhere(dq != 0)}


def test_family_is_deduped_and_negation_closed():
    w = _worm()
    keys = [_key(s) for s in w._family]
    assert len(keys) == len(set(keys))                     # no duplicate placements
    keyset = set(keys)
    for shape in w._family:
        neg = _key(tuple((mu, r, -c) for mu, r, c in shape))
        assert neg in keyset                               # negation-closed
        assert shape in w._self_charge                     # self-charge registered


def test_family_touches_the_head():
    # Support-anchoring: every family member's charge-reach support contains the
    # head (the origin, in relative coordinates) through at least one link.
    w = _worm()
    origin = (0, 0, 0, 0)
    for shape in w._family:
        assert any(origin in w._slot_support(mu, r) for mu, r, _c in shape)


def test_family_contains_singles_pairs_and_library():
    w = _worm()
    sizes = {len(shape) for shape in w._family}
    assert 1 in sizes and 2 in sizes                       # singles and pairs
    assert max(sizes) >= 3                                 # library 3-/4-link templates


def test_classified_set_reference_classifies_exactly():
    # The family is ~2e5 shapes and the reference does a global charge() recompute per
    # shape, so full-family reference enumerations are hand-run only.  Here the oracle's
    # semantic contract is checked on random SUBSAMPLES of the family (the ``shapes``
    # parameter): everything it returns is exactly an idle or a head-rooted dipole,
    # deduped.  (Full-set claims are covered by the local/kernel equalities of Tasks
    # 5 and 7 plus the pointwise subsample equality of Task 5.)
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S)
    L, N = S.Lattice, S.Lattice.N
    rng = np.random.default_rng(0)
    movers = idles = 0
    for cfg in configs:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        q0 = charge(cfg['n'])
        for _ in range(2):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            sub = [w._family[i] for i in
                   rng.choice(len(w._family), size=1500, replace=False)]
            C = w.classified_set_reference(n_arr, q0, head, shapes=sub)
            keys = [frozenset((l, c) for l, c in ch.items() if c != 0) for ch, _ in C]
            assert len(keys) == len(set(keys))              # deduped
            for change, target in C:
                dq_map = _defects(L, _apply(n_arr, change), q0)
                if target == head:
                    assert dq_map == {}                     # idle: Δq ≡ 0
                    idles += 1
                else:
                    assert dq_map == {head: -1, target: 1}  # exact dipole, any y
                    movers += 1
    assert movers + idles > 0


def test_merged_set_contains_two_link_worm_moves():
    # Full-family reference enumeration is infeasible (~2e5 global recomputes), so the
    # superset guarantee is checked per TwoLink move: its Delta n must be PRESENT in the
    # placed family, and the reference classifier restricted to exactly that shape (the
    # ``shapes`` parameter) must classify it as the same mover/idle.
    S, configs = _valid_configs()
    free = gen.FreeTargetWorm(S)
    two = gen.TwoLinkAdaptiveWorm(S)
    N = S.Lattice.N
    rng = np.random.default_rng(2)
    for cfg in configs[:2]:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        q0 = charge(cfg['n'])
        for _ in range(2):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            placed = {frozenset((l, c) for l, c in ch.items() if c != 0): shape
                      for ch, shape in free._placed(head)}

            def present_as(ch, want_target):
                k = frozenset((l, c) for l, c in ch.items() if c != 0)
                assert k in placed                          # Δn in the merged family
                got = free.classified_set_reference(n_arr, q0, head,
                                                    shapes=[placed[k]])
                assert len(got) == 1
                gch, gt = got[0]
                assert frozenset((l, c) for l, c in gch.items() if c != 0) == k
                assert gt == want_target

            for dd in two._ortho:
                for sign in (+1, -1):
                    target = tuple((head[k] + sign * dd[k]) % N for k in range(4))
                    for ch, _tgt in two.clean_set_reference(n_arr, q0, head, dd, sign):
                        present_as(ch, target)              # mover present, same target
            for ch in two.clean_idle_reference(n_arr, q0, head):
                present_as(ch, head)                        # idle present
