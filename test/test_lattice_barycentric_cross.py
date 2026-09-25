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
        for p, q in ((0, 1), (1, 1), (1, 2), (2, 1), (2, 2)):
            if p + q > D:
                continue
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
