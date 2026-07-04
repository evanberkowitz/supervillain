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


def test_intersection_worm_library_has_orthogonal_and_diagonal_moves():
    # The move library carries orthogonal steps to the 4 face neighbours (ê_μ) and
    # diagonal steps to all 12 in-plane diagonals (ê_μ ± ê_ν): 6 opposite-sign
    # (ê_μ - ê_ν) and 6 same-sign (ê_μ + ê_ν).
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    orthogonal = [d for d in worm._directions if sum(abs(x) for x in d) == 1]
    diagonal = [d for d in worm._directions if sum(abs(x) for x in d) == 2]
    opposite = [d for d in diagonal if sum(d) == 0]   # ê_μ - ê_ν
    same = [d for d in diagonal if sum(d) == 2]        # ê_μ + ê_ν
    assert len(orthogonal) == 4
    assert len(diagonal) == 12
    assert len(opposite) == 6
    assert len(same) == 6
    # Every stored direction is canonical (first nonzero component is +1).
    for d in worm._directions:
        assert next(x for x in d if x != 0) > 0
    # Orthogonal buckets carry the 3-link shape, the leaner 2-link shape, and the
    # background-activated 1-link shapes.
    for d in orthogonal:
        lengths = {len(shape) for shape in worm._library[d]}
        assert lengths == {1, 2, 3}
    # The opposite-sign diagonal is the minimal 2-link elbow; the same-sign diagonal,
    # unreachable by two links, is a 4-link shape; both also carry 1-link shapes.
    for d in opposite:
        assert {len(shape) for shape in worm._library[d]} == {1, 2}
    for d in same:
        assert {len(shape) for shape in worm._library[d]} == {1, 4}
    # Every 1-link shape carries coefficient ±1 only: Δq = c·L_ℓ(F) is exactly linear
    # in c (the self-wedge vanishes), so a unit head dipole demands c | 1 --- larger
    # magnitudes are provably useless for a unit-charge worm (see _build_library).
    for d in worm._directions:
        for shape in worm._library[d]:
            if len(shape) == 1:
                assert shape[0][2] in (+1, -1)


def test_intersection_worm_one_link_moves_fire_on_flux_and_never_on_vacuum():
    # A 1-link shape has zero self-charge (dΔn∧dΔn ≡ 0), so on the vacuum its Δq
    # vanishes identically and it can never move the head; on a flux background its
    # linear response Δq = c·L_ℓ(F) can be exactly the unit head dipole.  Pin one
    # concrete instance on the valid 3×3 patch background (F only in (0,ν) planes, so
    # q ≡ 0), and check the cleanliness-reversibility that plain Metropolis relies on:
    # the same shape, negated, undoes the move from the arrived configuration.
    from supervillain.generator.no_intersection.charge import charge
    L = Lattice(4, 6)
    N = L.N
    S = supervillain.action.NoIntersections(L, kappa=0.3)
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)

    head, sep, sign, shape = (1, 0, 0, 0), (0, 0, 0, 1), +1, ((1, (0, 0, 0, 0), +1),)
    assert shape in worm._library[sep]

    def defects(n, q0, change):
        trial = n.copy()
        for link, c in change.items():
            trial[link] += c
        dq = charge(trial) - q0
        return {tuple(int(x) for x in v[1:]): int(dq[tuple(v)])
                for v in np.argwhere(dq != 0)}, trial

    # Vacuum: the shape's Δq is identically zero --- a stay-put, never a head move.
    vacuum = L.zeros(1, dtype=int)
    change = worm._change_from_shape(head, sep, sign, shape)
    assert defects(vacuum, charge(vacuum), change)[0] == {}

    # Flux background: the same shape cleanly transports the head by +ê_3.
    n = L.zeros(1, dtype=int)
    for i, j in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)):
        n[(0, 1, i, j, 1)] += 1
    q0 = charge(n)
    assert not np.asarray(q0).any()
    target = tuple((head[k] + sign * sep[k]) % N for k in range(4))
    moved, trial = defects(n, q0, change)
    assert moved == {target: 1, head: -1}

    # Reversibility: from the arrived configuration, the backward draw (same bucket,
    # opposite sign) proposes exactly the negated links and restores q exactly.
    back = worm._change_from_shape(target, sep, -sign, shape)
    assert back == {link: -c for link, c in change.items()}
    undone, _ = defects(trial, charge(trial), back)
    assert undone == {head: 1, target: -1}


def test_intersection_worm_idle_one_link_classification():
    # On a locally flat background a 1-link shape has Δq ≡ 0, and _sheet_segment must
    # classify it as an IDLE move: return the change with target == head (sheet moves,
    # head does not).  Its reverse is the coefficient-negated shape from the same bucket
    # at the same sign, which must also be registered and anchor the same links.
    import types
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    n = S.Lattice.zeros(1, dtype=int)
    from supervillain.generator.no_intersection.charge import charge
    q0 = charge(n)
    head, sep = (0, 0, 0, 0), (0, 0, 0, 1)

    # 1-link shapes are appended after the seed orbits, so the last shape is 1-link.
    shape = worm._library[sep][-1]
    assert len(shape) == 1
    worm.rng = types.SimpleNamespace(integers=lambda lo, hi: hi - 1)
    change, target = worm._sheet_segment(n, q0, head, sep, +1)
    assert target == head
    assert change == worm._change_from_shape(head, sep, +1, shape)

    # The mirror slot: same bucket, same sign, coefficient-negated shape.
    (mu, rel, c) = shape[0]
    mirror = ((mu, rel, -c),)
    assert mirror in worm._library[sep]
    undo = worm._change_from_shape(head, sep, +1, mirror)
    assert undo == {link: -x for link, x in change.items()}


def test_intersection_worm_idle_moves_fire_and_stay_valid():
    # At κ = 0 every idle proposal is accepted, so idles must actually fire during worm
    # evolution from the cold start, and every emitted configuration must stay valid.
    S = _action(kappa=0.0)
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    idle = {'proposals': 0}
    orig = worm._sheet_segment

    def spy(n, q_now, head, hop, sign):
        change, target = orig(n, q_now, head, hop, sign)
        if change is not None and target == head:
            assert len(change) == 1      # idles are 1-link only
            idle['proposals'] += 1
        return change, target

    worm._sheet_segment = spy
    cfg = _cold(S)
    for _ in range(5):
        cfg = worm.step(cfg)
        assert S.valid(cfg)
    assert idle['proposals'] > 0


def test_intersection_worm_uses_diagonal_moves_and_stays_valid():
    # A diagonal 2-link step must be accepted at least once, and every emitted
    # configuration must still satisfy q = dn∧dn = 0.
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    used = {'diagonal': 0}
    orig = worm._sheet_segment

    def spy(n, q_now, head, hop, sign):
        change, target = orig(n, q_now, head, hop, sign)
        if change is not None and sum(abs(x) for x in hop) == 2:
            used['diagonal'] += 1
        return change, target

    worm._sheet_segment = spy
    cfg = _cold(S)
    for _ in range(40):
        cfg = worm.step(cfg)
        assert S.valid(cfg)
    assert used['diagonal'] > 0


def test_intersection_worm_uses_every_move_family_and_stays_valid():
    # All four shape families must fire a clean, accepted step at least once, and every
    # emitted configuration must satisfy q = dn∧dn = 0: the 2- and 3-link orthogonal
    # shapes (±ê_μ), the 2-link opposite-sign elbow (ê_μ - ê_ν), and the 4-link same-sign
    # diagonal (ê_μ + ê_ν).
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    used = {'ortho2': 0, 'ortho3': 0, 'opposite': 0, 'same': 0}
    orig = worm._sheet_segment

    def spy(n, q_now, head, hop, sign):
        change, target = orig(n, q_now, head, hop, sign)
        if change is not None:
            taxicab = sum(abs(x) for x in hop)
            if taxicab == 1:
                used['ortho2' if len(change) == 2 else 'ortho3'] += 1
            elif sum(hop) == 0:
                used['opposite'] += 1
            else:
                used['same'] += 1
        return change, target

    worm._sheet_segment = spy
    cfg = _cold(S)
    for _ in range(80):
        cfg = worm.step(cfg)
        assert S.valid(cfg)
    for family, count in used.items():
        assert count > 0, f'move family {family!r} never produced a clean accepted step'


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


def test_planar_flux_update_requires_no_intersections_action():
    L = Lattice(4, 5)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.PlanarFluxUpdate(V)


def test_planar_flux_update_preserves_validity():
    S = _action()
    gen = supervillain.generator.no_intersection.PlanarFluxUpdate(S)
    cfg = _cold(S)
    for _ in range(5):
        cfg = gen.step(cfg)
        assert S.valid(cfg)
        assert cfg['n'].shape == (S.Lattice.D,) + S.Lattice.dims


def test_scattershot_requires_no_intersections_action():
    L = Lattice(4, 5)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.ScattershotUpdate(V)


def test_scattershot_ratio_default_and_validation():
    S = _action(kappa=0.3)
    # The default ratio tracks the acceptance cliff between |Δn| = 1 and 2 …
    gen = supervillain.generator.no_intersection.ScattershotUpdate(S)
    assert np.isclose(gen.ratio, np.exp(-6 * np.pi**2 * 0.3))
    # … is capped at 1/2 for κ = 0 (normalizability) …
    cold = supervillain.generator.no_intersection.ScattershotUpdate(_action(kappa=0.0))
    assert cold.ratio == 0.5
    # … stays strictly positive even at huge κ (the tail is what makes the
    # irreducibility argument a theorem) …
    hot = supervillain.generator.no_intersection.ScattershotUpdate(_action(kappa=100.0))
    assert hot.ratio > 0
    # … and explicit ratios must sit in (0, 1).
    for bad in (0, 1, -0.1, 2):
        with pytest.raises(ValueError):
            supervillain.generator.no_intersection.ScattershotUpdate(S, ratio=bad)


def test_scattershot_preserves_validity():
    S = _action()
    gen = supervillain.generator.no_intersection.ScattershotUpdate(S)
    cfg = _cold(S)
    for _ in range(20):
        cfg = gen.step(cfg)
        assert S.valid(cfg)
        assert cfg['n'].shape == (S.Lattice.D,) + S.Lattice.dims


def test_scattershot_moves_at_kappa_zero():
    # At κ = 0 every clean proposal is accepted; near the vacuum single-link changes
    # are clean, so with λ = 2 expected touched links the configuration must change
    # within a modest number of steps.
    S = _action(kappa=0.0)
    gen = supervillain.generator.no_intersection.ScattershotUpdate(S)
    cfg = _cold(S)
    for _ in range(50):
        cfg = gen.step(cfg)
        assert S.valid(cfg)
    assert gen.accepted > 0
    assert gen.clean >= gen.accepted


def test_scattershot_proposals_are_joint():
    # The whole point: several links are changed in ONE accepted move.  At κ = 0 from
    # cold, scattered ±1 links are clean, so an accepted multi-link jump appears
    # quickly.  A sequential sweep can never do this --- it accepts links one at a time.
    S = _action(kappa=0.0)
    gen = supervillain.generator.no_intersection.ScattershotUpdate(S, links=3)
    cfg = _cold(S)
    jumped = 0
    for _ in range(100):
        before = np.asarray(cfg['n']).copy()
        cfg = gen.step(cfg)
        changed = int((np.asarray(cfg['n']) != before).sum())
        if changed >= 2:
            jumped += 1
        assert S.valid(cfg)
    assert jumped > 0, 'no accepted joint (multi-link) move in 100 steps at κ = 0'


def test_hammer_includes_constraint_preserving_villain_updates():
    # The Hammer reuses the Villain ExactUpdate and CohomologyUpdate, which change n
    # by a closed form and so leave dn (hence q = dn∧dn) untouched.
    S = _action()
    H = str(supervillain.generator.no_intersection.Hammer(S))
    for name in ('SiteUpdate', 'ExactUpdate', 'CohomologyUpdate', 'ConstrainedLinkUpdate',
                 'WrappingLoopUpdate', 'PlanarFluxUpdate', 'ScattershotUpdate',
                 'IntersectionWorm'):
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
