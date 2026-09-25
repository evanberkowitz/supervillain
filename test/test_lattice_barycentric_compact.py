import itertools
import numpy as np
import pytest

import supervillain.lattice.barycentric.doubled as dbl
import supervillain.lattice.barycentric.compact as cpt


def test_component_counts_match_simplices_per_site():
    """S_p is the number of p-simplices per original site: a component occurs
    exactly once per site, which is what makes the layout 100% dense."""
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = dbl.Lattice(D, N)
        c = cpt._compact_structures(D)
        for p in range(D + 1):
            assert c.count[p] == lat.count(p) // N**D, (D, p)


def test_four_dimensional_component_counts():
    c = cpt._compact_structures(4)
    assert [c.count[p] for p in range(5)] == [16, 240, 800, 960, 384]


def test_components_are_valid_and_lexicographic():
    """A type is realized only where the directions it releases are all even."""
    D = 4
    s = dbl._structures(D)
    c = cpt._compact_structures(D)
    for p in range(D + 1):
        for parity, t in c.components[p]:
            S = {mu for mu in range(D) if parity[mu]}
            assert not (s.touched[p][t] & S), (p, parity, t)
        keys = [(sum(b << mu for mu, b in enumerate(par)), t)
                for par, t in c.components[p]]
        assert keys == sorted(keys), f'p={p}: components not in canonical order'


def test_component_index_inverts_components():
    c = cpt._compact_structures(4)
    for p in range(5):
        for i, comp in enumerate(c.components[p]):
            assert c.component_index[p][comp] == i


def test_form_carries_degree_and_dual():
    lat = cpt.Lattice(4, 3)
    a = lat.zeros(2)
    assert isinstance(a, cpt.Form)
    assert a.degree == 2 and a.dual is False and a.lattice is lat
    assert a.shape == (cpt._compact_structures(4).count[2],) + (3, 3, 3, 3)


def test_dual_form_uses_the_complementary_component_axis():
    """A dual (D-p)-cochain is indexed by p-simplices, so it has the primal p
    component axis -- identical bytes, different reading."""
    lat = cpt.Lattice(4, 3)
    c = cpt._compact_structures(4)
    b_ = lat.zeros(2, dual=True)          # a dual 2-form is indexed by 2-simplices
    assert b_.dual is True and b_.degree == 2
    assert b_.shape == (c.count[4 - 2],) + (3, 3, 3, 3)
    d1 = lat.zeros(1, dual=True)          # a dual 1-form is indexed by 3-simplices
    assert d1.shape == (c.count[4 - 1],) + (3, 3, 3, 3)


def test_same_degree_and_duality_arithmetic_stays_a_form():
    lat = cpt.Lattice(3, 3)
    a, b_ = lat.random(2), lat.random(2)
    assert isinstance(a + b_, cpt.Form) and (a + b_).degree == 2
    assert (a + b_).dual is False


def test_primal_plus_dual_degrades_even_though_shapes_match():
    """The dangerous case: identical shapes, different spaces.  numpy would add
    them silently, so the guard must drop the Form-ness."""
    lat = cpt.Lattice(4, 3)
    primal = lat.random(2)
    dual   = cpt.Form(np.asarray(lat.random(2)), degree=2, lattice=lat, dual=True)
    assert primal.shape == dual.shape, 'the test is pointless unless shapes match'
    assert not isinstance(primal + dual, cpt.Form)


def test_mixed_degree_still_degrades():
    lat = cpt.Lattice(2, 3)
    one = cpt.Form(np.asarray(lat.random(1))[:1], degree=1, lattice=lat)
    two = cpt.Form(np.asarray(lat.random(0))[:1], degree=0, lattice=lat)
    assert not isinstance(one * two, cpt.Form)


def test_random_is_dense():
    """Unlike the doubled layout there is nothing to mask: every slot is real."""
    lat = cpt.Lattice(4, 3)
    for p in range(5):
        assert (np.asarray(lat.random(p)) != 0).all()


def test_no_support_method():
    """support() is deliberately absent; every slot is a simplex."""
    assert not hasattr(cpt.Lattice(4, 3), 'support')


def test_count_is_lattice_wide():
    """The precise trap the brief named: _c.count[p] is per site, count(p) is
    lattice-wide.  Cross-check against doubled's count(), computed by a
    completely different route (masking a (2N)^D array), rather than merely
    restating _c.count[p] * N**D."""
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = cpt.Lattice(D, N)
        dlat = dbl.Lattice(D, N)
        for p in range(D + 1):
            assert lat.count(p) == dlat.count(p), (D, N, p)


def test_measure_matches_doubled():
    D, N = 4, 3
    lat = cpt.Lattice(D, N)
    c = cpt._compact_structures(D)
    s = dbl._structures(D)
    for p in range(D + 1):
        m = lat.measure(p)
        assert m.shape == (c.count[p],) + (1,) * D
        assert (m != 0).any(), f'p={p}: measure is degenerately all zero'
        for i, (parity, t) in enumerate(c.components[p]):
            assert m.ravel()[i] == s.measure[p][t], (p, i)


def test_orientation_sign_matches_doubled():
    D, N = 4, 3
    lat = cpt.Lattice(D, N)
    c = cpt._compact_structures(D)
    s = dbl._structures(D)
    sign = lat.orientation_sign
    assert (sign != 0).any(), 'orientation_sign is degenerately all zero'
    for i, (parity, t) in enumerate(c.components[D]):
        assert sign.ravel()[i] == s.orientation_sign[t], i


def test_link_class_multiset_matches_doubled():
    """Compare histograms, not element by element, so the test does not
    merely restate the implementation."""
    D, N = 4, 3
    lat = cpt.Lattice(D, N)
    dlat = dbl.Lattice(D, N)
    nclass = len(lat.link_classes)
    compact_hist = np.bincount(np.repeat(lat.link_class, N**D), minlength=nclass)
    doubled_class = dlat.link_class
    doubled_hist = np.bincount(doubled_class[doubled_class >= 0].ravel(), minlength=nclass)
    assert compact_hist.sum() == lat.count(1) > 0
    assert (compact_hist == doubled_hist).all()


def test_random_integer_respects_its_bounds():
    D, N, p = 3, 3, 2
    lat = cpt.Lattice(D, N)
    c = cpt._compact_structures(D)
    a = lat.random_integer(p, low=10, high=20)
    assert a.shape == (c.count[p],) + (N,) * D
    raw = np.asarray(a)
    assert np.issubdtype(raw.dtype, np.integer)
    assert (raw >= 10).all() and (raw < 20).all()
    assert (raw != raw.flat[0]).any(), 'random_integer is degenerately constant'


def test_push_and_pull_carry_a_dual_form_through():
    """A whole-cell roll needs no component table, so dual forms are fine."""
    clat = cpt.Lattice(4, 3)
    a = clat.random(2, dual=True)
    out = cpt.pull(cpt.push(a, (1, 0, 2, 1)), (1, 0, 2, 1))
    assert out.dual is True and out.degree == 2
    assert not np.allclose(np.asarray(a), 0)
    assert np.allclose(np.asarray(out), np.asarray(a))


@pytest.mark.parametrize('call', [
    lambda lat: cpt.reflect(lat.zeros(1, dual=True), (0,)),
    lambda lat: cpt.permute(lat.zeros(1, dual=True), (1, 2, 3, 0)),
    lambda lat: cpt.tau(lat.zeros(1, dual=True)),
    lambda lat: cpt.tau(lat.zeros(1, dual=True), 2),
    lambda lat: cpt.translate_diagonal(lat.zeros(1, dual=True), (1, 1, 1, 1)),
])
def test_the_point_group_and_tau_refuse_a_dual_form(call):
    """Each entry point guards: the tables are indexed by primal components."""
    with pytest.raises(ValueError):
        call(cpt.Lattice(4, 3))


def test_tau_refuses_a_dual_form_at_even_power_too():
    """An even power short-circuits to push, which would otherwise sail past."""
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError):
        cpt.tau(clat.zeros(1, dual=True), 2)


@pytest.mark.parametrize('signs', [(1, 1, 1), (1, 1, 1, 0), (1, -1, 1, 2)])
def test_translate_diagonal_refuses_bad_signs(signs):
    """``match`` matters: a wrong-length ``signs`` would also trip ``np.roll``
    downstream, so without it the guard is not what the test proves."""
    clat = cpt.Lattice(4, 3)
    with pytest.raises(ValueError, match='signs must be'):
        cpt.translate_diagonal(clat.random(1), signs)


def test_point_group_tables_are_N_independent_and_cached():
    """The table depends on D and the group element only -- never on N."""
    a, b = cpt.Lattice(3, 3), cpt.Lattice(3, 5)
    assert a._c is b._c
    assert a._c.point_group(1, (1, 2, 0), ()) is b._c.point_group(1, (1, 2, 0), ())
    assert a._c.half_diagonal(1) is b._c.half_diagonal(1)


def test_point_group_permutes_the_components_bijectively():
    """Every component is the image of exactly one -- a symmetry loses nothing."""
    for D in (2, 3, 4):
        c = cpt._compact_structures(D)
        perm = tuple((i + 1) % D for i in range(D))
        for p in range(D + 1):
            for element in ((tuple(range(D)), (0,)), (perm, ()), (perm, (0, 1))):
                images = [j for j, _ in c.point_group(p, *element)]
                assert sorted(images) == list(range(c.count[p])), (D, p, element)
            images = [j for j, _ in c.half_diagonal(p)[0]]
            assert sorted(images) == list(range(c.count[p])), (D, p, 'tau')


def test_the_general_point_group_element_is_permute_after_reflect():
    r"""``reflect`` and ``permute`` use only the pure-flip and pure-permutation
    branches, so the combined one would otherwise go unexercised.  It is
    $\xi_{\text{perm}[\mu]} = \epsilon_\mu \xi_\mu$, which is the flip first."""
    for D in (2, 3, 4):
        lat = cpt.Lattice(D, 3)
        perm, flips = tuple((i + 1) % D for i in range(D)), (0,)
        for p in range(D + 1):
            a = lat.random(p)
            combined = np.asarray(cpt._point_group(a, perm, flips))
            assert not np.allclose(combined, 0), (D, p)
            assert np.allclose(combined,
                               np.asarray(cpt.permute(cpt.reflect(a, flips), perm))), (D, p)


def test_poincare_star_toggles_dual_and_reflects_the_degree():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = cpt.Lattice(D, N)
        for p in range(D + 1):
            a = lat.random(p)
            s = cpt.poincare_star(a)
            assert s.degree == D - p, (D, p)
            assert s.dual is True, (D, p)
            assert s.lattice is lat, (D, p)


def test_poincare_star_moves_no_data():
    '''It is a relabeling: the numbers are untouched, in order.'''
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = cpt.Lattice(D, N)
        for p in range(D + 1):
            a = lat.random(p)
            assert np.abs(np.asarray(a)).sum() != 0, f'D={D} p={p}: vacuous'
            assert np.array_equal(np.asarray(cpt.poincare_star(a)),
                                  np.asarray(a)), (D, p)


def test_poincare_star_is_involutive_with_no_sign():
    '''The resolved convention makes star-star exactly +1, not (-1)^p(D-p).'''
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = cpt.Lattice(D, N)
        for p in range(D + 1):
            a = lat.random(p)
            back = cpt.poincare_star(cpt.poincare_star(a))
            assert back.degree == p and back.dual is False, (D, p)
            assert np.abs(np.asarray(a)).sum() != 0, f'D={D} p={p}: vacuous'
            assert np.array_equal(np.asarray(back), np.asarray(a)), (D, p)


def test_poincare_star_does_not_alias_its_input():
    '''Mutating the result must not corrupt the source.'''
    lat = cpt.Lattice(3, 3)
    a = lat.random(1)
    before = np.array(np.asarray(a), copy=True)
    s = cpt.poincare_star(a)
    np.asarray(s)[...] += 1.0
    assert np.array_equal(np.asarray(a), before)


def test_poincare_star_round_trips_a_dual_form():
    lat = cpt.Lattice(3, 3)
    d1 = cpt.poincare_star(lat.random(1))
    back = cpt.poincare_star(d1)
    assert back.degree == 1 and back.dual is False


def test_d_on_a_dual_form_raises_the_degree_and_stays_dual():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = cpt.Lattice(D, N)
        for k in range(D):
            a = cpt.poincare_star(lat.random(D - k))
            assert a.degree == k and a.dual is True
            out = cpt.d(a)
            assert out.degree == k + 1 and out.dual is True, (D, k)
            assert np.asarray(out).shape[0] == lat._c.count[D - k - 1], (D, k)


def test_d_on_the_dual_is_the_transpose_of_the_primal_coboundary():
    '''<d(star a), star b> == <star a, delta(star b)> -- the defining pairing.'''
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = cpt.Lattice(D, N)
        for k in range(D):
            a = cpt.poincare_star(lat.random(D - k))
            b = cpt.poincare_star(lat.random(D - k - 1))
            lhs = float((np.asarray(cpt.d(a)) * np.asarray(b)).sum())
            rhs = float((np.asarray(a) * np.asarray(cpt.delta(b))).sum())
            assert not np.isclose(lhs, 0), f'D={D} k={k}: vacuous'
            assert np.isclose(lhs, rhs), (D, k)


def test_d_squared_is_zero_on_the_dual_complex():
    for D, N in ((3, 3), (4, 3)):
        lat = cpt.Lattice(D, N)
        for k in range(D - 1):
            a = cpt.poincare_star(lat.random(D - k))
            assert np.abs(np.asarray(a)).sum() != 0, f'D={D} k={k}: vacuous'
            assert np.allclose(np.asarray(cpt.d(cpt.d(a))), 0), (D, k)


def test_delta_on_a_dual_form_lowers_the_degree_and_stays_dual():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = cpt.Lattice(D, N)
        for k in range(1, D + 1):
            a = cpt.poincare_star(lat.random(D - k))
            out = cpt.delta(a)
            assert out.degree == k - 1 and out.dual is True, (D, k)


def test_d_still_refuses_past_the_top_degree_on_each_complex():
    lat = cpt.Lattice(3, 3)
    with pytest.raises(ValueError):
        cpt.d(lat.zeros(3))
    with pytest.raises(ValueError):
        cpt.d(cpt.poincare_star(lat.zeros(0)))      # dual degree 3, the top


def test_delta_still_refuses_a_zero_cochain_on_each_complex():
    lat = cpt.Lattice(3, 3)
    with pytest.raises(ValueError):
        cpt.delta(lat.zeros(0))
    with pytest.raises(ValueError):
        cpt.delta(cpt.poincare_star(lat.zeros(3)))  # dual degree 0
