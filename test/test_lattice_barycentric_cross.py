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


def test_d_raises_the_degree_and_stays_primal():
    clat = cpt.Lattice(4, 3)
    out = cpt.d(clat.random(1))
    assert out.degree == 2 and out.dual is False


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
