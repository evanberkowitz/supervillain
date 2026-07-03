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
    # Orthogonal buckets carry both the 3-link and the leaner 2-link shape.
    for d in orthogonal:
        lengths = {len(shape) for shape in worm._library[d]}
        assert lengths == {2, 3}
    # The opposite-sign diagonal is the minimal 2-link elbow; the same-sign diagonal,
    # unreachable by two links, is a 4-link shape.
    for d in opposite:
        assert all(len(shape) == 2 for shape in worm._library[d])
    for d in same:
        assert all(len(shape) == 4 for shape in worm._library[d])


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


def test_hammer_includes_constraint_preserving_villain_updates():
    # The Hammer reuses the Villain ExactUpdate and CohomologyUpdate, which change n
    # by a closed form and so leave dn (hence q = dn∧dn) untouched.
    S = _action()
    H = str(supervillain.generator.no_intersection.Hammer(S))
    for name in ('SiteUpdate', 'ExactUpdate', 'CohomologyUpdate', 'ConstrainedLinkUpdate',
                 'WrappingLoopUpdate', 'PlanarFluxUpdate', 'IntersectionWorm'):
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
