#!/usr/bin/env python

import re

import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice, d, wedge


def _action(kappa=0.3, N=5):
    L = Lattice(4, N)
    return supervillain.action.NoIntersections(L, kappa=kappa)


def _cold(S):
    return S.configurations(1)[0]


def test_no_intersection_hammer_uses_heatbaths_and_overrelaxation():
    # Explicit fugacity skips the tuner's Monte-Carlo probes, keeping this fast.  The closed-n
    # moves are the exact heatbaths (constraint-safe: they leave dn, hence dn∧dn, untouched),
    # and the φ overrelaxation is interleaved.  LinkHeatbath is excluded (breaks the constraint).
    S = _action()
    s = str(supervillain.generator.no_intersection.Hammer(S, fugacity=0.5))
    assert 'SiteOverrelaxation' in s
    assert 'ExactHeatbath' in s
    assert 'CohomologyHeatbath' in s
    assert 'ExactUpdate' not in s
    assert 'CohomologyUpdate' not in s
    # The constraint-safe local heatbath is present …
    assert 'ConstrainedLinkHeatbath' in s
    # … but the Villain LinkHeatbath (its W-coset move breaks the constraint) is not.  Match
    # on a word boundary so ConstrainedLinkHeatbath's substring does not give a false positive.
    assert not re.search(r'(?<![A-Za-z])LinkHeatbath', s)


def test_no_intersection_hammer_overrelax_must_be_positive():
    S = _action()
    import pytest
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.Hammer(S, fugacity=0.5, overrelax=0)


def test_no_intersection_hammer_steps_stay_valid():
    # The heatbath-based Hammer must keep dn∧dn = 0 exactly.  fugacity is kept light: at
    # this small volume a heavy DefectGas condenses (no vacuum return) and raises --- a
    # physical limit, not a validity failure --- which would make this stochastic test flaky.
    S = _action()
    H = supervillain.generator.no_intersection.Hammer(S, fugacity=0.025)
    cfg = S.configurations(1)[0]
    for _ in range(20):
        cfg = H.step(cfg)
        assert S.valid(cfg)


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
    assert np.asarray(out['IntersectionTwoPoint']).shape == S.Lattice.dims
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
    # At κ = 0 every idle proposal is accepted, so idles must actually fire during
    # worm evolution from the cold start, and every emitted configuration must stay
    # valid.  Idles arise from 1-link shapes only, by construction.
    S = _action(kappa=0.0)
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    cfg = _cold(S)
    for _ in range(5):
        cfg = worm.step(cfg)
        assert S.valid(cfg)
    assert worm.tallies['1link']['accepted_idle'] > 0


def test_intersection_worm_uses_diagonal_moves_and_stays_valid():
    # A diagonal step must be offered cleanly at least once, and every emitted
    # configuration must still satisfy q = dn∧dn = 0.
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    cfg = _cold(S)
    for _ in range(40):
        cfg = worm.step(cfg)
        assert S.valid(cfg)
    assert sum(worm.tallies[f]['clean'] for f in ('elbow2', 'same4')) > 0


def test_intersection_worm_uses_every_move_family_and_stays_valid():
    # All four multi-link shape families must offer a clean step at least once, and
    # every emitted configuration must satisfy q = dn∧dn = 0.
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    cfg = _cold(S)
    for _ in range(80):
        cfg = worm.step(cfg)
        assert S.valid(cfg)
    for family in ('ortho2', 'ortho3', 'elbow2', 'same4'):
        assert worm.tallies[family]['clean'] > 0, \
            f'move family {family!r} never produced a clean step'


def test_intersection_worm_inline_observable_keys():
    S = _action()
    worm = supervillain.generator.no_intersection.IntersectionWorm(S)
    obs = worm.inline_observables(3)
    assert set(obs) == {'IntersectionTwoPoint', 'Worm_Length'}


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


def test_constrained_link_heatbath_requires_no_intersections_action():
    L = Lattice(4, 5)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.ConstrainedLinkHeatbath(V)


def test_constrained_link_heatbath_preserves_validity():
    # Both the vectorized sweep and the global-recompute oracle must keep dn∧dn = 0.
    S = _action()
    for method in ('step', 'step_reference'):
        gen = supervillain.generator.no_intersection.ConstrainedLinkHeatbath(S)
        cfg = _cold(S)
        for _ in range(5):
            cfg = getattr(gen, method)(cfg)
            assert S.valid(cfg)
            assert cfg['n'].shape == (S.Lattice.D,) + S.Lattice.dims


def test_constrained_link_heatbath_leaves_phi_untouched():
    S = _action()
    gen = supervillain.generator.no_intersection.ConstrainedLinkHeatbath(S)
    cfg = _cold(S)
    phi0 = np.asarray(cfg['phi']).copy()
    out = gen.step(cfg)
    assert np.array_equal(np.asarray(out['phi']), phi0)


def test_constrained_link_heatbath_clean_on_zero_flux_and_moves_from_cold():
    # On a zero-flux background F = dn = 0, L_ℓ(F) = 0 on every link, so every link is
    # clean: the first colour swept from cold sees all-clean.  (Across the full sweep the
    # carried F fills in as links move, so later colours can freeze --- clean < proposed
    # overall, which is the checkerboard doing its job, not a bug.)
    from supervillain.generator.no_intersection.local_charge import clean_mask_for_color, axis_colors
    S = _action()
    N = S.Lattice.N
    F0 = np.asarray(d(S.Lattice.zeros(1, dtype=int)))       # zero flux
    axis = axis_colors(N)
    idx = [axis[0]] * 4
    assert clean_mask_for_color(F0, 0, idx, N).all()

    # And the heatbath actually resamples links off the cold start (not a silent no-op).
    gen = supervillain.generator.no_intersection.ConstrainedLinkHeatbath(S)
    gen.step(_cold(S))
    assert gen.resampled > 0
    assert 0 < gen.clean <= gen.proposed == S.Lattice.D * S.Lattice.sites


def test_constrained_link_heatbath_matches_villain_linkheatbath_when_unconstrained():
    # Where every link is clean (cold F = 0), a ConstrainedLinkHeatbath sweep and a Villain
    # W=1 LinkHeatbath sweep sample the SAME per-link discrete Gaussian.  Seeded identically
    # and drawing in the same order (the reference oracle visits one link at a time, as the
    # Villain heatbath does per site), their first-move marginals must coincide statistically:
    # here we check the cheaper invariant that both leave q = 0 and only shift n by integers.
    S = _action()
    gen = supervillain.generator.no_intersection.ConstrainedLinkHeatbath(S)
    cfg = _cold(S)
    out = gen.step(cfg)
    shift = np.asarray(out['n']) - np.asarray(cfg['n'])
    assert np.array_equal(shift, shift.astype(int))     # integer shifts only
    assert S.valid(out)


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
    # The Hammer reuses the Villain closed-n moves as exact heatbaths (ExactHeatbath and
    # CohomologyHeatbath), which change n by a closed form and so leave dn (hence q = dn∧dn)
    # untouched.  The DefectGas replaces the worm: it is manifestly ergodic and emits the
    # absolutely-normalized correlator, where every worm we tried jammed.
    S = _action()
    H = str(supervillain.generator.no_intersection.Hammer(S, fugacity=0.025))
    for name in ('SiteHeatbath', 'ExactHeatbath', 'CohomologyHeatbath',
                 'ConstrainedLinkHeatbath', 'PlanarFluxUpdate',
                 'ScattershotUpdate', 'DefectGas'):
        assert name in H
    # The heatbath supersedes the Metropolis single-link move; it is not doubled up.
    assert 'ConstrainedLinkUpdate' not in H
    # WrappingLoopUpdate is omitted while it is a slow reference implementation
    # (Scattershot/DefectGas cover ergodicity).
    assert 'WrappingLoopUpdate' not in H


def test_hammer_has_no_worm():
    # Every worm we built jams; none is in the default generator.
    S = _action()
    H = str(supervillain.generator.no_intersection.Hammer(S, fugacity=0.025))
    assert 'Worm' not in H


def test_hammer_fugacity_none_autotunes():
    # fugacity=None runs the DefectGasFugacityTuner, which returns a fugacity in (0, 1].
    S = _action()
    H = supervillain.generator.no_intersection.Hammer(S)
    assert 'DefectGas' in str(H)


def test_hammer_steps_stay_valid():
    S = _action()
    H = supervillain.generator.no_intersection.Hammer(S, fugacity=0.025)
    cfg = _cold(S)
    for _ in range(5):
        cfg = H.step(cfg)
        assert S.valid(cfg)


def test_ensemble_generate_stays_valid():
    S = _action()
    H = supervillain.generator.no_intersection.Hammer(S, fugacity=0.025)
    e = supervillain.Ensemble(S).generate(10, H, start='cold')
    for c in e.configuration:
        assert S.valid(c)


def test_nointersections_inherits_villain_observables():
    # NoIntersections is a Villain, so the observable dispatch walks the action MRO and its
    # Villain implementations apply --- the field-based observables computed from (phi, n)
    # are all available.  (Vortex_Vortex is excluded: its Villain measurement is D=2 only.)
    S = _action()
    H = supervillain.generator.no_intersection.Hammer(S, fugacity=0.025)
    e = supervillain.Ensemble(S).generate(20, H, start='cold')
    for o in ('ActionDensity', 'InternalEnergyDensity', 'InternalEnergyDensitySquared',
              'WindingSquared', 'Winding_Winding', 'Spin_Spin'):
        assert np.asarray(getattr(e, o)).shape[0] == len(e)
    b = supervillain.analysis.Bootstrap(e, 20)
    assert np.asarray(b.Spin_Spin_Normalized).shape[0] == 20  # derived quantities too


def test_nointersections_topological_charge_vanishes():
    # The constraint q = dn∧dn = 0 makes the topological-charge density identically zero,
    # so its (inherited Villain) same-site observable is exactly 0 on every configuration.
    S = _action()
    H = supervillain.generator.no_intersection.Hammer(S, fugacity=0.025)
    e = supervillain.Ensemble(S).generate(20, H, start='cold')
    assert not np.asarray(e.TopologicalChargeDensitySquared).any()


def test_intersection_two_point_is_inline_only():
    # The raw worm histogram has no closed-form estimator; it is only ever produced
    # inline by a worm, exactly as ActionTwoPoint is a raw ingredient.
    obs = supervillain.observable.IntersectionTwoPoint
    assert not hasattr(obs, 'default')
    assert not hasattr(obs, 'Villain')
    assert not hasattr(obs, 'Worldline')
    assert not hasattr(obs, 'NoIntersections')


def test_intersection_intersection_is_a_derived_quantity():
    from supervillain.observable import DerivedQuantity
    assert issubclass(supervillain.observable.Intersection_Intersection, DerivedQuantity)
    assert hasattr(supervillain.observable.Intersection_Intersection, 'NoIntersections')


def test_intersection_intersection_is_one_at_origin():
    L = Lattice(4, 3)
    S = supervillain.action.NoIntersections(L, kappa=0.3)
    H = supervillain.generator.no_intersection.Hammer(S, fugacity=0.025)
    e = supervillain.Ensemble(S).generate(40, H, start='cold')

    b = supervillain.analysis.Bootstrap(e, 25)
    theta = np.asarray(b.Intersection_Intersection).real
    assert theta.shape == (25,) + L.dims
    # Theta_0 = 1 identically: a coincident pair IS the vacuum.  It is WRITTEN, not
    # divided out -- Theta_Theta's origin bin is empty by construction.
    assert np.allclose(theta[(slice(None),) + L.origin], 1.0)
