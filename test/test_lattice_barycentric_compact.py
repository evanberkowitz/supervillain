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
