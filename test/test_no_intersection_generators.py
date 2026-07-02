#!/usr/bin/env python

import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice, d, wedge


def _action(kappa=0.3, N=5):
    L = Lattice(4, N)
    return supervillain.action.NoIntersections(L, kappa=kappa)


def _cold(S):
    return S.configurations(1)[0]


def test_charge_matches_topological_charge():
    from supervillain.generator.no_intersection.charge import charge
    from supervillain.observable.topological import _topological_charge
    L = Lattice(4, 5)
    n = L.zeros(1, dtype=int)
    n[0, 0, 0, 0, 0] = 1
    n[1, 1, 0, 0, 0] = 1
    assert np.array_equal(charge(n), np.asarray(_topological_charge(L, n)))


def test_intersection_worm_requires_no_intersections_action():
    L = Lattice(4, 5)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.IntersectionWorm(V)


def test_intersection_worm_requires_D4():
    # NoIntersections cannot even be built in D != 4, so a Villain stand-in in D=2
    # exercises the worm's own dimensional guard.
    L = Lattice(2, 4)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.IntersectionWorm(V)


def test_intersection_worm_preserves_validity_and_closes():
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    out = worm.step(_cold(S))
    assert S.valid(out)
    assert np.asarray(out['Intersection_Intersection']).shape == S.Lattice.dims
    assert np.isscalar(out['Worm_Length']) or np.asarray(out['Worm_Length']).shape == ()


def test_intersection_worm_inline_observable_keys():
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    obs = worm.inline_observables(3)
    assert set(obs) == {'Intersection_Intersection', 'Worm_Length'}


def test_constrained_link_update_requires_no_intersections_action():
    L = Lattice(4, 5)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.ConstrainedLinkUpdate(V)


def test_constrained_link_update_preserves_validity():
    S = _action()
    gen = supervillain.generator.no_intersection.ConstrainedLinkUpdate(S)
    cfg = _cold(S)
    out = gen.step(cfg)
    assert S.valid(out)
    assert out['n'].shape == cfg['n'].shape


def test_wrapping_loop_update_requires_no_intersections_action():
    L = Lattice(4, 5)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.WrappingLoopUpdate(V)


def test_wrapping_loop_update_preserves_validity():
    S = _action()
    gen = supervillain.generator.no_intersection.WrappingLoopUpdate(S)
    cfg = _cold(S)
    for _ in range(5):
        cfg = gen.step(cfg)
        assert S.valid(cfg)


def test_hammer_includes_constraint_preserving_villain_updates():
    # The Hammer reuses the Villain ExactUpdate and CohomologyUpdate, which change n
    # by a closed form and so leave dn (hence q = dn∧dn) untouched.
    S = _action()
    H = str(supervillain.generator.no_intersection.Hammer(S))
    for name in ('SiteUpdate', 'ExactUpdate', 'CohomologyUpdate',
                 'ConstrainedLinkUpdate', 'WrappingLoopUpdate', 'IntersectionWorm'):
        assert name in H


def test_hammer_steps_stay_valid():
    S = _action()
    H = supervillain.generator.no_intersection.Hammer(S)
    cfg = _cold(S)
    for _ in range(5):
        cfg = H.step(cfg)
        assert S.valid(cfg)


def test_ensemble_generate_stays_valid():
    S = _action()
    H = supervillain.generator.no_intersection.Hammer(S)
    e = supervillain.Ensemble(S).generate(10, H, start='cold')
    for c in e.configuration:
        assert S.valid(c)


def test_intersection_intersection_is_inline_only():
    # The correlator has no closed-form estimator, so it carries no measurement
    # method for any action; it is only ever available when the IntersectionWorm
    # produces it inline.  That is what effectively scopes it (and the normalized
    # derived quantity built on it) to the NoIntersections model.
    obs = supervillain.observable.Intersection_Intersection
    assert not hasattr(obs, 'default')
    assert not hasattr(obs, 'Villain')
    assert not hasattr(obs, 'Worldline')
    assert not hasattr(obs, 'NoIntersections')


def test_intersection_intersection_normalized_is_one_at_origin():
    L = Lattice(4, 3)
    S = supervillain.action.NoIntersections(L, kappa=0.3)
    H = supervillain.generator.no_intersection.Hammer(S)
    e = supervillain.Ensemble(S).generate(40, H, start='cold')

    # The worm fills the inline Intersection_Intersection histogram.
    assert np.asarray(e.Intersection_Intersection).shape == (len(e),) + L.dims

    b = supervillain.analysis.Bootstrap(e, 25)
    norm = np.asarray(b.Intersection_Intersection_Normalized)
    # Normalized to 1 at the origin on every bootstrap sample.
    assert np.allclose(norm[(slice(None),) + L.origin], 1)


def test_intersection_worm_library_covers_all_directions():
    # The library holds clean shapes for every one of the 8 unit dipole separations
    # with equal bucket sizes by symmetry.  The orbit expansion of the 93 seed
    # classes in moves.py must reproduce exactly the 828 moves per direction found
    # by the exhaustive enumeration in example/no-intersection-move-search.py.
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    buckets = {sep: len(shapes) for sep, shapes in worm._library.items()}
    units = {tuple(int(k == mu) * s for k in range(4)) for mu in range(4) for s in (+1, -1)}
    assert set(buckets) == units
    assert set(buckets.values()) == {828}


def test_intersection_worm_local_dq_matches_global():
    # The worm's O(1) linearized Δq must agree exactly with a global recomputation
    # of q = dn ∧ dn on arbitrary (even invalid, multiply-occupied) backgrounds.
    from supervillain.generator.no_intersection.charge import charge

    S = _action()
    L = S.Lattice
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    rng = np.random.default_rng(11)

    for trial in range(20):
        n = L.zeros(1, dtype=int)
        n[...] = rng.integers(0, 2, size=n.shape) * rng.integers(-2, 3, size=n.shape)
        F = np.asarray(d(n)).astype(int)

        head = tuple(int(x) for x in rng.integers(0, L.N, size=4))
        mu = int(rng.integers(0, 4))
        sign = int(rng.choice([1, -1]))
        step = tuple(sign if k == mu else 0 for k in range(4))
        target = tuple((head[k] + step[k]) % L.N for k in range(4))
        direct = worm._library[step]
        negated = worm._library[tuple(-x for x in step)]
        i = int(rng.integers(0, len(direct) + len(negated)))
        if i < len(direct):
            change = worm._place(direct[i], target, +1)
        else:
            change = worm._place(negated[i - len(direct)], head, -1)

        trial_n = n.copy()
        for link, c in change.items():
            trial_n[link] += c
        dq = charge(trial_n) - charge(n)
        nz = np.argwhere(dq != 0)
        expected = {tuple(int(x) for x in h[1:]): int(dq[tuple(h)]) for h in nz}

        assert worm._dq(F, change) == expected


def test_intersection_worm_moves_invert_exactly():
    # Every forward library move anchored at the target is exactly undone by the
    # negated placement anchored at the head --- the pairing detailed balance rests on.
    S = _action()
    L = S.Lattice
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    rng = np.random.default_rng(13)

    for trial in range(20):
        head = tuple(int(x) for x in rng.integers(0, L.N, size=4))
        mu = int(rng.integers(0, 4))
        step = tuple(int(k == mu) for k in range(4))
        target = tuple((head[k] + step[k]) % L.N for k in range(4))
        shapes = worm._library[step]
        shape = shapes[int(rng.integers(0, len(shapes)))]

        forward = worm._place(shape, target, +1)
        backward = worm._place(shape, target, -1)
        net = dict(forward)
        for link, c in backward.items():
            net[link] = net.get(link, 0) + c
        assert all(v == 0 for v in net.values())


def test_constrained_link_update_local_check_matches_global():
    # Along a trajectory of accepted single-link changes on a valid configuration,
    # the local constraint check (Δq empty) must agree with the global one
    # (q of the trial configuration vanishes everywhere), and the incrementally
    # maintained F must track d(n).
    from supervillain.generator.no_intersection.charge import charge, dF_entries, local_dq

    S = _action(N=4)
    L = S.Lattice
    rng = np.random.default_rng(17)

    n = L.zeros(1, dtype=int)
    F = np.asarray(d(n)).astype(int)

    checked = accepted = 0
    for trial in range(300):
        link = (int(rng.integers(0, 4)),) + tuple(int(x) for x in rng.integers(0, L.N, size=4))
        c = int(rng.choice([1, -1]))
        change = {link: c}

        local_ok = not local_dq(L, F, change, pairs=None)
        trial_n = n.copy()
        trial_n[link] += c
        global_ok = bool(np.all(charge(trial_n) == 0))
        assert local_ok == global_ok
        checked += 1

        if local_ok:
            n = trial_n
            for (idx, site), v in dF_entries(L, change).items():
                F[(idx,) + site] += v
            accepted += 1

    assert np.array_equal(F, np.asarray(d(n)))
    assert accepted > 0 and accepted < checked  # both branches exercised
