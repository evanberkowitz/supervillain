r"""
The compact layout against the doubled one.

.. warning ::
    These tests check the **re-indexing**, not the operator mathematics.  Every
    compact table here is *derived* from the corresponding doubled table, so
    agreement confirms only that the routing between components is right; that
    $d^2 = 0$, that adjointness holds, that the Leibniz rule holds and the rest
    is established by the doubled module's own tests, in
    ``test/test_lattice_barycentric.py``.  A passing cross-check is not
    independent confirmation of any of it.
"""
from math import comb

import numpy as np
import pytest

import supervillain.lattice.barycentric.doubled as dbl
import supervillain.lattice.barycentric.compact as cpt


def _pair(D, N):
    return cpt.Lattice(D, N), dbl.Lattice(D, N)


def test_round_trip_is_the_identity():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        for p in range(D + 1):
            a = clat.random(p)
            back = cpt.Form.from_doubled(a.to_doubled(), clat)
            assert not np.allclose(np.asarray(a), 0), f'D={D} p={p}: vacuous'
            assert np.allclose(np.asarray(back), np.asarray(a)), (D, p)
            assert back.degree == p and back.dual is False


def test_to_doubled_lands_only_on_real_simplices():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, dlat = _pair(D, N)
        for p in range(D + 1):
            f = clat.random(p).to_doubled()
            assert (np.asarray(f)[dlat.support(p) == 0] == 0).all(), (D, p)
            assert (np.asarray(f) != 0).sum() == clat.count(p), (D, p)


def test_from_doubled_ignores_nothing():
    """Every real slot of the doubled form reaches a compact component."""
    for D, N in ((2, 3), (4, 3)):
        clat, dlat = _pair(D, N)
        for p in range(D + 1):
            f = dlat.random(p)
            a = cpt.Form.from_doubled(f, clat)
            assert np.isclose(np.asarray(a).sum(), np.asarray(f).sum()), (D, p)


def test_to_doubled_refuses_a_dual_form():
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError):
        clat.zeros(2, dual=True).to_doubled()


def _some_indices(count, k=3):
    """A handful of component indices spanning [0, count), without repeats."""
    if count <= k:
        return list(range(count))
    step = max(1, count // k)
    return sorted(set(range(0, count, step)))[:k]


def test_single_component_lands_at_its_predicted_slot():
    r"""
    Pin the ``(parity, type)`` $\to$ anchor correspondence directly, rather
    than restating ``to_doubled``'s own indexing: a single nonzero compact
    entry must land at doubled anchor $2x + \pi$ --- computed here from
    scratch, not by calling into the implementation --- and nowhere else.

    .. note ::
        ``test_round_trip_is_the_identity`` cannot catch a *consistently*
        wrong parity convention (e.g. every ``parity[mu]`` flipped to
        ``1 - parity[mu]`` in both directions): the round trip would still be
        the identity.  This test locates the single nonzero directly, so it
        would.
    """
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        comps = cpt._compact_structures(D).components
        for p in range(D + 1):
            count = clat._c.count[p]
            sites = [(0,) * D, tuple((mu + 1) % N for mu in range(D))]
            for i in _some_indices(count):
                parity, t = comps[p][i]
                for x in sites:
                    a = clat.zeros(p, dtype=int)
                    np.asarray(a)[(i,) + x] = 7
                    anchor = tuple((2 * x[mu] + parity[mu]) % (2 * N)
                                   for mu in range(D))
                    doubled_form = a.to_doubled()
                    arr = np.asarray(doubled_form)
                    assert np.issubdtype(arr.dtype, np.integer), (D, p, i, x)
                    assert arr[(t,) + anchor] == 7, (D, p, i, x)
                    assert (arr != 0).sum() == 1, (D, p, i, x)


def test_d_matches_doubled():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        for p in range(D):
            a = clat.random(p)
            lhs = cpt.d(a).to_doubled()
            rhs = dbl.d(a.to_doubled())
            assert not np.allclose(np.asarray(lhs), 0), f'D={D} p={p}: vacuous'
            assert np.allclose(np.asarray(lhs), np.asarray(rhs)), (D, p)


def test_d_raises_the_degree_and_keeps_it_primal():
    """Degree bookkeeping only -- not the ValueError branches; see below."""
    clat = cpt.Lattice(4, 3)
    out = cpt.d(clat.random(1))
    assert out.degree == 2 and out.dual is False


def test_d_refuses_a_dual_form():
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError):
        cpt.d(clat.zeros(1, dual=True))


def test_d_refuses_past_the_top_degree():
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError):
        cpt.d(clat.random(4))


def test_delta_matches_doubled():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        for p in range(1, D + 1):
            a = clat.random(p)
            lhs = cpt.delta(a).to_doubled()
            rhs = dbl.delta(a.to_doubled())
            assert not np.allclose(np.asarray(lhs), 0), f'D={D} p={p}: vacuous'
            assert np.allclose(np.asarray(lhs), np.asarray(rhs)), (D, p)


def test_cup_matches_doubled():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        for p in range(D + 1):
            for q in range(D + 1 - p):
                a, c = clat.random(p), clat.random(q)
                lhs = cpt.cup(a, c).to_doubled()
                rhs = dbl.cup(a.to_doubled(), c.to_doubled())
                assert not np.allclose(np.asarray(lhs), 0), f'D={D} {p}u{q}: vacuous'
                assert np.allclose(np.asarray(lhs), np.asarray(rhs)), (D, p, q)


def test_adjointness_holds_compactly():
    """Re-checked here so the compact module stands alone."""
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        for p in range(D):
            a, c = clat.random(p), clat.random(p + 1)
            lhs = float((np.asarray(cpt.d(a)) * np.asarray(c)).sum())
            rhs = float((np.asarray(a) * np.asarray(cpt.delta(c))).sum())
            assert not np.isclose(lhs, 0), f'D={D} p={p}: vacuous'
            assert np.isclose(lhs, rhs), (D, p)


def test_delta_refuses_a_dual_form():
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError):
        cpt.delta(clat.zeros(1, dual=True))


def test_delta_refuses_a_0_cochain():
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError):
        cpt.delta(clat.zeros(0))


def test_cup_refuses_a_dual_form():
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError):
        cpt.cup(clat.zeros(1, dual=True), clat.zeros(1))


def test_cup_refuses_a_dual_form_in_the_second_argument():
    """A guard that checked only the first argument would miss this."""
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError):
        cpt.cup(clat.zeros(1), clat.zeros(1, dual=True))


def test_cup_refuses_past_the_top_degree():
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError):
        cpt.cup(clat.random(3), clat.random(2))


def test_wedge_is_cup():
    assert cpt.wedge is cpt.cup


def test_site_offsets_are_two_point():
    """The stencil reaches x-1 and x per direction, never x+1.

    A component is realized only where every direction its type releases is
    even at the anchor, and a coboundary shift is nonzero only in directions
    the type releases -- so parity + shift never reaches 2, which is the only
    route to a +1 offset.
    """
    for D in (2, 3, 4):
        c = cpt._compact_structures(D)
        seen = set()
        for P in range(1, D + 1):
            for terms in c.coboundary[P]:
                for _, shift, _ in terms:
                    seen |= set(shift)
        assert seen == {-1, 0}, f'D={D}: {sorted(seen)}'


def test_translations_match_doubled():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        shift = tuple(range(1, D + 1))
        for p in range(D + 1):
            a = clat.random(p)
            for name in ('push', 'pull'):
                lhs = getattr(cpt, name)(a, shift).to_doubled()
                rhs = getattr(dbl, name)(a.to_doubled(), shift)
                assert not np.allclose(np.asarray(lhs), 0), f'{name} D={D} p={p}: vacuous'
                assert np.allclose(np.asarray(lhs), np.asarray(rhs)), (name, D, p)


def test_translate_is_push():
    """An alias, in both modules -- so ``push`` above covers it."""
    assert cpt.translate is cpt.push
    assert dbl.translate is dbl.push


def test_symmetries_match_doubled():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        perm  = tuple((i + 1) % D for i in range(D))
        flips = (0,)
        cases = (('reflect', flips), ('permute', perm), ('tau', 1), ('tau', 2))
        for p in range(D + 1):
            a = clat.random(p)
            for name, arg in cases:
                lhs = getattr(cpt, name)(a, arg).to_doubled()
                rhs = getattr(dbl, name)(a.to_doubled(), arg)
                assert not np.allclose(np.asarray(lhs), 0), f'{name} D={D} p={p}'
                assert np.allclose(np.asarray(lhs), np.asarray(rhs)), (name, arg, D, p)


def test_translate_diagonal_matches_doubled():
    for D, N in ((3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        signs = tuple(1 if i % 2 == 0 else -1 for i in range(D))
        a = clat.random(1)
        lhs = cpt.translate_diagonal(a, signs).to_doubled()
        assert not np.allclose(np.asarray(lhs), 0), f'D={D}: vacuous'
        assert np.allclose(np.asarray(lhs),
                           np.asarray(dbl.translate_diagonal(a.to_doubled(), signs))), D


def test_package_hoists_the_compact_names():
    import supervillain.lattice.barycentric as bary
    for name in ('Lattice', 'Form', 'd', 'delta', 'cup', 'wedge',
                 'push', 'pull', 'translate', 'reflect', 'permute',
                 'tau', 'translate_diagonal'):
        assert hasattr(bary, name), name
        assert getattr(bary, name) is getattr(cpt, name), name
    assert bary.doubled.Lattice is dbl.Lattice


def test_planes_match_doubled():
    '''The coordinate planes depend only on D and p, so both layouts agree.'''
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, dlat = _pair(D, N)
        for p in range(D + 1):
            assert clat.planes(p) == dlat.planes(p), (D, p)
            assert len(clat.planes(p)) == comb(D, p), (D, p)


def test_pvector_is_the_doubled_table_gathered_onto_components():
    '''Each component's p-vector is its type's, so compact is a gather.'''
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, dlat = _pair(D, N)
        for p in range(D + 1):
            cpv, dpv = clat.pvector(p), dlat.pvector(p)
            assert cpv.shape == (clat._c.count[p], comb(D, p)), (D, p)
            assert dpv.shape[1] == cpv.shape[1], (D, p)
            for i, (_, t) in enumerate(clat._c.components[p]):
                assert np.array_equal(cpv[i], dpv[t]), (D, p, i)
            assert np.abs(cpv).sum() != 0, f'D={D} p={p}: vacuous'


def test_pvector_at_the_top_degree_is_the_orientation_sign():
    '''orientation_sign is the p = D case, where there is one plane.'''
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        top = clat.pvector(D)
        assert top.shape[1] == 1, D
        assert np.array_equal(top[:, 0].reshape((-1,) + (1,) * D),
                              clat.orientation_sign), D
        assert set(np.unique(top)) == {-1, 1}, D


def test_coordinate_aligned_components_have_exactly_one_unit_component():
    '''A type releasing one direction per step spans a single plane.'''
    for D, N in ((2, 3), (3, 3), (4, 3)):
        clat, _ = _pair(D, N)
        seen = 0
        for p in range(1, D + 1):
            pv = clat.pvector(p)
            for i, (_, t) in enumerate(clat._c.components[p]):
                if all(len(step) == 1 for step in clat._s.step_types[p][t]):
                    assert np.count_nonzero(pv[i]) == 1, (D, p, i)
                    assert abs(pv[i]).max() == 1, (D, p, i)
                    seen += 1
        assert seen > 0, f'D={D}: no coordinate-aligned component examined'
