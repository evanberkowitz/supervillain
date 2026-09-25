#!/usr/bin/env python
r'''
The slow reference construction of the barycubic complex, and the fast implementation against it.

``test/barycubic_reference.py`` builds the barycentric subdivision of the cubical torus from
explicit flags of cubical cells (not from the doubled-coordinate/sign shortcut the implementation
uses, SPEC §2.2), and computes Hodge weights from barycentric dual cells by exact Gram
determinants (SPEC §5.1).  This file has two halves.

1. **Self-checks of the reference** (``test_reference_*``).  These need no implementation and pass
   today.  They pin the reference to closed-form facts: the inclusion–exclusion counts, ``d∘d = 0``,
   the torus Betti numbers, ``∂[M] = 0``, the weight sum rules, the ten D = 4 link weights and the
   affine-field normalization from the notes, and the coarsening chain map against the existing
   hypercubic ``compact.d``.
2. **Implementation vs reference** (SPEC §7 item 4, mirroring ``test/test_lattice_reference.py``).
   Labels are matched through ``lattice.canonical_labels(p)`` (rank-ordered vertex lists in doubled
   coordinates mod 2N, SPEC §8), and every fast operator must agree exactly: ``d`` and ``∪_i`` on
   integer cochains, ``ε`` exactly, and weights to 1e-13 relative.

The reference makes the *geometry* independent; it shares the closed-form cup-i sign rule with the
implementation, which has its own oracles (``test_barycubic_cup.py``).
'''

import itertools
import math

import numpy as np
import pytest

import barycubic_reference as ref
from barycubic_reference import DN_REF, lattice, reference, int_form
from supervillain.lattice import Lattice, Form
from supervillain.lattice import compact


# =============================================================================================
# Part 1. Self-checks of the reference construction (no implementation needed)
# =============================================================================================

D4_LINK_WEIGHTS = {
    # Barycentric-dual Hodge weights of the ten D = 4 link classes, computed independently by hand.
    (0, 1): math.sqrt(5) / 10, (0, 2): math.sqrt(6) / 60, (0, 3): 1 / 40, (0, 4): math.sqrt(2) / 40,
    (1, 2): math.sqrt(6) / 30, (1, 3): math.sqrt(2) / 60, (1, 4): 1 / 40,
    (2, 3): math.sqrt(6) / 30, (2, 4): math.sqrt(6) / 60, (3, 4): math.sqrt(5) / 10,
}
D4_LINKS_PER_SITE = {
    # Links of each rank class per hypercubic site, counted independently.
    (0, 1): 8, (0, 2): 24, (0, 3): 32, (0, 4): 16, (1, 2): 24,
    (1, 3): 48, (1, 4): 32, (2, 3): 24, (2, 4): 24, (3, 4): 8,
}
D4_AFFINE_NORMALIZATION = (math.sqrt(5) + 2 * math.sqrt(6) + 3 + 2 * math.sqrt(2)) / 10


@pytest.mark.parametrize("D", [1, 2, 3, 4, 5])
def test_reference_counts_match_inclusion_exclusion(D):
    r'''Flags counted explicitly equal $f_p = \sum_j (-1)^{p-j}\binom{p}{j}(2j+2)^D$; χ = 0; $f_1 = 4^D-2^D$; $f_D = 2^D D!$.'''
    # SPEC §2.2, §7 item 1.
    counts = [len(ref.local_simplices(D, p)) for p in range(D + 1)]
    assert counts == [ref.simplices_per_site(D, p) for p in range(D + 1)]
    assert sum((-1) ** p * c for p, c in enumerate(counts)) == 0
    assert counts[1] == 4 ** D - 2 ** D
    assert counts[D] == 2 ** D * math.factorial(D)
    if D == 4:
        assert counts == [16, 240, 800, 960, 384]


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_reference_simplices_are_rank_ordered_flags(D):
    r'''Every reference simplex has strictly increasing ranks; top simplices have exactly one vertex of each rank.'''
    # SPEC §1, §7 item 1.
    for p in range(D + 1):
        ranks = ref.rank_of(ref.local_simplices(D, p))
        assert (np.diff(ranks, axis=1) > 0).all()
    assert (ref.rank_of(ref.local_simplices(D, D)) == np.arange(D + 1)).all()


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_reference_link_lengths_and_classes(D):
    r'''A fine link from rank r to rank s has length $\sqrt{s-r}/2$; there are $\binom{D}{s}\binom{s}{r}2^{s-r}$ per site.'''
    # SPEC §7 items 1 and 5.
    links = ref.local_simplices(D, 1)
    ranks = ref.rank_of(links)
    length2 = (((links[:, 1] - links[:, 0]) / 2) ** 2).sum(axis=1)
    assert np.allclose(length2, (ranks[:, 1] - ranks[:, 0]) / 4, rtol=0, atol=1e-15)
    for r, s in itertools.combinations(range(D + 1), 2):
        count = int(((ranks[:, 0] == r) & (ranks[:, 1] == s)).sum())
        assert count == ref.links_per_site_of_class(D, r, s)
        if D == 4:
            assert count == D4_LINKS_PER_SITE[(r, s)]


@pytest.mark.parametrize("D,N", [(D, N) for D in (1, 2, 3, 4) for N in (2, 3)])
def test_reference_is_simplicial_complex(D, N):
    r'''After periodic identification, distinct fine simplices have distinct vertex sets, and no simplex repeats a vertex.'''
    # SPEC §1, §7 item 2.  N = 2 is the edge case.
    R = reference(D, N)
    for p in range(D + 1):
        simp = R.reduced[p]
        assert len(R.index[p]) == len(simp) == N ** D * ref.simplices_per_site(D, p)
        vertex_sets = {frozenset(map(tuple, s)) for s in simp}
        assert len(vertex_sets) == len(simp)
        assert all(len(s) == p + 1 for s in vertex_sets)


@pytest.mark.parametrize("D,N", DN_REF)
def test_reference_d_squared_zero_and_fundamental_cycle(D, N):
    r'''The explicit coboundary matrices satisfy $d_{p+1} d_p = 0$, and $\partial[M] = d_{D-1}^\top \varepsilon = 0$ with $\varepsilon = \pm1$.'''
    # SPEC §3, §4.
    R = reference(D, N)
    for p in range(D - 1):
        assert abs(R.d(p + 1) @ R.d(p)).sum() == 0
    eps = R.orientation
    assert set(np.unique(eps)) <= {-1, 1}
    assert not (R.d(D - 1).T @ eps).any()


@pytest.mark.parametrize("D,N", [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2)])
def test_reference_torus_betti_numbers(D, N):
    r'''The global reference complex has the rational cohomology of the D-torus, $b_p = \binom{D}{p}$.'''
    # SPEC §7 item 3 (global version, small complexes only).
    assert reference(D, N).betti() == [math.comb(D, p) for p in range(D + 1)]


@pytest.mark.parametrize("D,N", [(D, N) for D in (1, 2, 3) for N in (2, 3, 4)] + [(4, 2)])
def test_reference_fourier_topology(D, N):
    r'''Unit-cell Fourier blocks of the reference: $b_p(k=0) = \binom{D}{p}$, acyclic at $k \ne 0$; weighted Laplacian nullities agree.'''
    # SPEC §7 item 3.  D = 4 uses one momentum per hyperoctahedral orbit (see momentum_orbit_representatives).
    sizes = [ref.simplices_per_site(D, p) for p in range(D + 1)]
    tables = [ref.local_d_table(D, p) for p in range(D)]
    weights = [ref.local_hodge_weights(D, p) for p in range(D + 1)]
    ks = ref.momenta(D, N) if D < 4 else ref.momentum_orbit_representatives(D, N)
    for k in ks:
        blocks = [ref.fourier_d_block(tables[p], sizes[p + 1], sizes[p], k) for p in range(D)]
        expected = [math.comb(D, p) for p in range(D + 1)] if not np.any(k) else [0] * (D + 1)
        assert ref.fourier_betti(blocks, sizes) == expected, k
        assert ref.fourier_laplacian_nullities(blocks, weights) == expected, k


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_reference_weight_sum_rules(D):
    r'''
    Geometric weights: positive; $\sum_v M_0 = 1$ and $\sum_\sigma 1/M_D = 1$ per site; $M_D = 2^D D!$;
    $M_0(r) = 1/((D+1)\binom{D}{r})$; $M(R) = M(D-R)$; one entry per nonempty rank set.
    '''
    # SPEC §5.1, §7 item 6.
    by_rank = ref.hodge_weights_by_rank_set(D)
    assert len(by_rank) == 2 ** (D + 1) - 1
    assert all(w > 0 for w in by_rank.values())
    for R_, w in by_rank.items():
        mirror = tuple(sorted(D - r for r in R_))
        assert w == pytest.approx(by_rank[mirror], rel=1e-13)
    for r in range(D + 1):
        assert by_rank[(r,)] == pytest.approx(1 / ((D + 1) * math.comb(D, r)), rel=1e-13)
    assert by_rank[tuple(range(D + 1))] == pytest.approx(2 ** D * math.factorial(D), rel=1e-13)
    assert ref.local_hodge_weights(D, 0).sum() == pytest.approx(1, rel=1e-13)
    assert (1 / ref.local_hodge_weights(D, D)).sum() == pytest.approx(1, rel=1e-13)


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_reference_weights_depend_only_on_rank_set(D):
    r'''Computed simplex by simplex, every geometric weight equals the weight of its rank set (hyperoctahedral transitivity).'''
    # SPEC §5.1 "Symmetry".
    by_rank = ref.hodge_weights_by_rank_set(D)
    for p in range(D + 1):
        ranks = ref.rank_of(ref.local_simplices(D, p))
        expected = np.array([by_rank[tuple(r)] for r in ranks])
        assert np.allclose(ref.local_hodge_weights(D, p), expected, rtol=1e-13, atol=0)


def test_reference_d4_link_weights_and_normalization():
    r'''At D = 4 the geometric link weights are the ten closed forms of the notes; 31 entries take 18 values; $\mathcal N = (\sqrt5+2\sqrt6+3+2\sqrt2)/10$.'''
    # SPEC §5.1, §7 items 5 and 6.
    by_rank = ref.hodge_weights_by_rank_set(4)
    for rs, w in D4_LINK_WEIGHTS.items():
        assert by_rank[rs] == pytest.approx(w, rel=1e-13), rs
    assert len(np.unique(np.round(list(by_rank.values()), 12))) == 18
    assert ref.affine_normalization(4, by_rank) == pytest.approx(D4_AFFINE_NORMALIZATION, rel=1e-13)


def test_reference_seam_displacement():
    r'''At D = 1, N = 2 the link stored as (0, 3) has lifted displacement −1 (not the raw +3), and the winding period is 2N.'''
    # SPEC §2.2 "Common lift", §7 item 11.
    R = reference(1, 2)
    row = R.lookup(1, np.array([[[0], [3]]]))[0]
    assert R.reduced[1][row, 1, 0] - R.reduced[1][row, 0, 0] == 3
    assert R.lifted[1][row, 1, 0] - R.lifted[1][row, 0, 0] == -1
    winding = (R.lifted[1][:, 1, 0] - R.lifted[1][:, 0, 0])
    assert (R.orientation * winding).sum() == 2 * 2        # ∫ Δξ = 2N, i.e. ∫ w = N


def _reference_winding(R, mu):
    r'''The integer winding 1-cochain $u_\mu(e) = \Delta\xi_\mu(e)$ (twice the physical displacement), from lifted coordinates.'''
    return R.lifted[1][:, 1, mu] - R.lifted[1][:, 0, mu]


@pytest.mark.parametrize("D,N", DN_REF)
def test_reference_cohomology_ring(D, N):
    r'''On the reference, $\int u_{\pi(0)} \cup \cdots \cup u_{\pi(D-1)} = {\rm sgn}(\pi)(2N)^D$ for the integer windings $u = 2w$.'''
    # SPEC §7 item 11, with the reference's own ε, d and cup.
    R = reference(D, N)
    u = [_reference_winding(R, mu) for mu in range(D)]
    if D > 1:
        for mu in range(D):
            assert not (R.d(1) @ u[mu]).any()
    perms = list(itertools.permutations(range(D)))
    if D == 4:
        perms = perms[:3]
    for perm in perms:
        prod = u[perm[0]]
        for k, mu in enumerate(perm[1:], start=1):
            prod = R.cup(prod, k, u[mu], 1, 0)
        sign = round(np.linalg.det(np.eye(D)[list(perm)]))
        assert (R.orientation * prod).sum() == sign * (2 * N) ** D


@pytest.mark.parametrize("D,N", [(D, N) for D in (1, 2, 3) for N in (2, 3)] + [(4, 2)])
def test_reference_coarsen_is_cochain_map(D, N):
    r'''The reference $Sd^\top$ intertwines the fine $d$ with the existing hypercubic ``compact.d``, and preserves the top integral.'''
    # SPEC §6.1, §7 item 13.  This pins the reference's orientation conventions to compact.py.
    R = reference(D, N)
    L = Lattice(D, N)
    rng = np.random.default_rng([D, N, 13])
    for p in range(D):
        omega = rng.integers(-3, 4, size=R.count(p))
        fine = R.coarsen(R.d(p) @ omega, p + 1)
        coarse = compact.d(Form(R.coarsen(omega, p), degree=p, lattice=L))
        assert np.array_equal(fine, np.asarray(coarse))
    top = rng.integers(-3, 4, size=R.count(D))
    assert R.coarsen(top, D).sum() == (R.orientation * top).sum()


# =============================================================================================
# Part 2. The implementation against the reference (SPEC §7 item 4)
# =============================================================================================

def _perm(L, R, p):
    return ref.permutation_to_reference(np.asarray(L.canonical_labels(p)), R, p)


@pytest.mark.parametrize("D,N", DN_REF)
def test_canonical_labels_match_reference(D, N):
    r'''The implementation's canonical labels are exactly the reference's fine simplices, each once.'''
    # SPEC §8.
    L, R = lattice(D, N), reference(D, N)
    for p in range(D + 1):
        labels = np.asarray(L.canonical_labels(p))
        assert labels.shape == (N ** D * ref.simplices_per_site(D, p), p + 1, D)
        assert labels.min() >= 0 and labels.max() < 2 * N
        perm = _perm(L, R, p)
        assert len(np.unique(perm)) == len(perm) == R.count(p)


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_local_vertices_match_reference(D):
    r'''``local_vertices(p)`` lists exactly the reference's lifted simplices anchored at the origin.'''
    # SPEC §2.2 (common lift): entries in {0,1,2}, top vertex = 1_{S_p}.
    L = lattice(D, 2)
    for p in range(D + 1):
        mine = np.asarray(L.local_vertices(p))
        assert mine.shape == (ref.simplices_per_site(D, p), p + 1, D)
        assert sorted(ref._key(mine)) == sorted(ref._key(ref.local_simplices(D, p)))


@pytest.mark.parametrize("D,N", DN_REF)
def test_d_matches_reference(D, N):
    r'''The fast $d$ equals the reference incidence matrix exactly on integer cochains, at every degree.'''
    # SPEC §4, §7 item 4.
    from supervillain.lattice.barycubic import d
    L, R = lattice(D, N), reference(D, N)
    for p in range(D):
        f = int_form(L, p, 4)
        expected = R.d(p) @ ref.to_reference_order(f, _perm(L, R, p))
        assert np.array_equal(ref.to_reference_order(d(f), _perm(L, R, p + 1)), expected)


@pytest.mark.parametrize("D,N", DN_REF)
def test_export_complex_matches_reference(D, N):
    r'''``export_complex(p)``: labels are the canonical labels; its sparse $d$, weights and ε equal the reference's in that order.'''
    # SPEC §8, the interop format.
    L, R = lattice(D, N), reference(D, N)
    for p in range(D + 1):
        ex = L.export_complex(p)
        assert np.array_equal(np.asarray(ex['labels']), np.asarray(L.canonical_labels(p)))
        perm = _perm(L, R, p)
        if p < D:
            expected = R.d(p)[_perm(L, R, p + 1)][:, perm]
            assert abs(ex['d'] - expected).sum() == 0
        else:
            assert 'd' not in ex
            assert np.array_equal(np.asarray(ex['orientation']), R.orientation[perm])
        # Map each global simplex to its local label by subtracting 2·anchor (anchor = floor(top/2)).
        lifted = R.lifted[p][perm]
        anchors = lifted[:, -1] // 2
        rows = [ref.local_index(D, p)[k] for k in ref._key(lifted - 2 * anchors[:, None, :])]
        geometric = ref.local_hodge_weights(D, p)[rows]
        assert np.allclose(np.asarray(ex['weights']).reshape(-1), geometric, rtol=1e-13, atol=0)


@pytest.mark.parametrize("D,N", DN_REF)
def test_orientation_matches_reference(D, N):
    r'''The integer top form ``lattice.orientation`` equals the sign of the determinant of the lifted edge vectors.'''
    # SPEC §3.
    L, R = lattice(D, N), reference(D, N)
    eps = L.orientation
    assert np.issubdtype(np.asarray(eps).dtype, np.integer)
    assert eps.degree == D
    assert np.array_equal(ref.to_reference_order(eps, _perm(L, R, D)), R.orientation)


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_weights_match_geometric_reference(D):
    r'''Per-label weights equal the geometric barycentric-dual weights to 1e-13 relative, and ``hodge_weight_by_rank_set`` likewise.'''
    # SPEC §5.1, §7 item 4.
    L = lattice(D, 2)
    geometric = ref.hodge_weights_by_rank_set(D)
    table = L.hodge_weight_by_rank_set
    assert set(map(tuple, table.keys())) == set(geometric.keys())
    for R_, w in geometric.items():
        assert table[R_] == pytest.approx(w, rel=1e-13, abs=0), R_
    for p in range(D + 1):
        mine = np.asarray(L.local_vertices(p), dtype=np.int64)
        rows = [ref.local_index(D, p)[k] for k in ref._key(mine)]
        assert np.allclose(np.asarray(L.hodge_weights(p)), ref.local_hodge_weights(D, p)[rows], rtol=1e-13, atol=0)


def _ref_cup_cases():
    cases = []
    for D, N in DN_REF:
        degrees = ref.legal_cup_degrees(D)
        if D == 4:
            # Keep the slow Python reference lean at D = 4: every i at output degree ≤ 2, plus the
            # top-degree products that carry the ∪_D and plaquette (2 ∪ 2) signs.
            degrees = [c for c in degrees if c[0] + c[1] - c[2] <= 2] + [(2, 2, 0), (3, 3, 2), (4, 4, 4), (2, 3, 1), (3, 2, 1)]
        cases += [(D, N, p, q, i) for p, q, i in degrees]
    return cases


@pytest.mark.parametrize("D,N,p,q,i", _ref_cup_cases())
def test_cup_matches_reference(D, N, p, q, i):
    r'''The fast $a \cup_i b$ equals the reference (vertex-tuple face lookups, closed-form signs) exactly on integers.'''
    # SPEC §5.4-§5.5, §7 item 4.
    from supervillain.lattice.barycubic import cup
    L, R = lattice(D, N), reference(D, N)
    a, b = int_form(L, p, 1, q, i), int_form(L, q, 2, p, i)
    expected = R.cup(ref.to_reference_order(a, _perm(L, R, p)), p,
                     ref.to_reference_order(b, _perm(L, R, q)), q, i)
    got = cup(a, b, i)
    assert got.degree == p + q - i
    assert np.array_equal(ref.to_reference_order(got, _perm(L, R, p + q - i)), expected)


# The cup kernel reaches neighbouring sites through per-lattice tables of shifted site indices.
# At N = 2 a shift of +1 and −1 lands on the same site, so those tables are only really exercised
# at N ≥ 3.  DN_REF already covers D = 4, N = 3 (test_cup_matches_reference, including i ≥ 1);
# this adds D = 3, N = 4, where every axis has four distinct site indices.
CUP_D3_N4_CASES = [(1, 1, 0), (1, 2, 0), (2, 1, 1), (1, 1, 1), (2, 2, 1), (2, 2, 2), (3, 1, 1), (3, 2, 2), (3, 3, 3), (1, 2, 1)]


@pytest.mark.parametrize("p,q,i", CUP_D3_N4_CASES)
def test_cup_matches_reference_d3_n4(p, q, i):
    r'''At D = 3, N = 4 the fast $a \cup_i b$ equals the reference exactly on integers, for products with i = 0 to 3.'''
    # SPEC §5.5, §7 item 4; replaces the retired chunk-size test (the kernel no longer chunks).
    from supervillain.lattice.barycubic import cup
    D, N = 3, 4
    L, R = lattice(D, N), reference(D, N)
    a, b = int_form(L, p, 60, q, i), int_form(L, q, 61, p, i)
    expected = R.cup(ref.to_reference_order(a, _perm(L, R, p)), p,
                     ref.to_reference_order(b, _perm(L, R, q)), q, i)
    assert expected.any()
    assert np.array_equal(ref.to_reference_order(cup(a, b, i), _perm(L, R, p + q - i)), expected)


@pytest.mark.parametrize("D,N", DN_REF)
def test_coarsen_matches_reference(D, N):
    r'''``coarsen`` equals the reference $Sd^\top$ (orientations from lifted determinants) exactly, at every degree.'''
    # SPEC §6.1.
    from supervillain.lattice.barycubic import coarsen
    L, R = lattice(D, N), reference(D, N)
    for p in range(D + 1):
        omega = int_form(L, p, 6)
        got = coarsen(omega)
        assert isinstance(got, Form) and got.degree == p
        assert got.lattice.D == D and got.lattice.N == N
        assert np.array_equal(np.asarray(got), R.coarsen(ref.to_reference_order(omega, _perm(L, R, p)), p))


# =============================================================================================
# Interop, end to end: export_cup against the reference.
#
# ``export_cup(p, q, i, seed)`` publishes deterministic integer cochains a, b (built from the
# canonical labels) and the values of a ∪_i b, so that two independent implementations can be
# diffed by canonical label (SPEC §8).  Here the reference construction is the second one.
# Layout (from the export_cup docstring): 'labels' are canonical_labels(p+q−i); 'a', 'b' are int64
# in canonical-label order of degrees p and q; 'values' are a ∪_i b in the order of 'labels'.
# =============================================================================================

EXPORT_HASH_P = 2 ** 31 - 1
EXPORT_HASH_B = 1000003


def export_cochain_value(label, seed, tag):
    r'''
    The documented export hash, re-implemented from the ``export_cup`` docstring alone:
    $h \leftarrow {\rm seed} \bmod P$, $h \leftarrow (hB + {\rm tag}) \bmod P$, then
    $h \leftarrow (hB + c_m + 1) \bmod P$ over the label's coordinates vertex by vertex, then
    $h \leftarrow (h^2 + B) \bmod P$; the value is $(h \bmod 7) - 3$.  ``tag`` is 0 for a, 1 for b.
    '''
    P, B = EXPORT_HASH_P, EXPORT_HASH_B
    h = seed % P
    h = (h * B + tag) % P
    for c in np.asarray(label, dtype=np.int64).reshape(-1):
        h = (h * B + int(c) + 1) % P
    h = (h * h + B) % P
    return h % 7 - 3


def export_cochain(labels, seed, tag):
    r'''The export cochain on every canonical label in ``labels`` ``(M, p+1, D)``.'''
    return np.array([export_cochain_value(lab, seed, tag) for lab in np.asarray(labels)], dtype=np.int64)


EXPORT_CUP_CASES = [(3, N, p, q, i) for N in (2, 3) for p, q, i in [(1, 1, 0), (2, 1, 1), (2, 2, 1), (1, 2, 1), (3, 3, 3)]]


def _flat(x):
    return np.asarray(x).reshape(-1)


@pytest.mark.parametrize("D,N,p,q,i", EXPORT_CUP_CASES)
def test_export_cup_matches_reference(D, N, p, q, i):
    r'''The exported a and b equal the documented hash of the canonical labels, and the exported $a \cup_i b$ equals the reference cup on them, matched by canonical label.'''
    L, R = lattice(D, N), reference(D, N)
    n = p + q - i
    ex = L.export_cup(p, q, i, seed=0)
    assert {'labels', 'a', 'b', 'values'} <= set(ex)
    assert np.array_equal(np.asarray(ex['labels']).reshape(-1, n + 1, D), np.asarray(L.canonical_labels(n)))
    a, b = _flat(ex['a']), _flat(ex['b'])
    assert a.dtype == np.int64 and b.dtype == np.int64
    assert a.any() and b.any()
    # The factors are rebuilt independently from the documented hash of the canonical labels.
    assert np.array_equal(a, export_cochain(L.canonical_labels(p), 0, 0))
    assert np.array_equal(b, export_cochain(L.canonical_labels(q), 0, 1))
    expected = R.cup(ref.to_reference_order(a, _perm(L, R, p)), p, ref.to_reference_order(b, _perm(L, R, q)), q, i)
    assert np.array_equal(ref.to_reference_order(_flat(ex['values']), _perm(L, R, n)), expected)


@pytest.mark.parametrize("N", [2, 3])
def test_export_cup_is_deterministic_and_seeded(N):
    r'''``export_cup`` gives identical output for the same seed and different cochains (hence values) for different seeds.'''
    L = lattice(3, N)
    first, again, other = L.export_cup(2, 2, 1, seed=0), L.export_cup(2, 2, 1, seed=0), L.export_cup(2, 2, 1, seed=1)
    for key in ('labels', 'a', 'b', 'values'):
        assert np.array_equal(np.asarray(first[key]), np.asarray(again[key])), key
    assert not np.array_equal(_flat(first['a']), _flat(other['a']))
    assert not np.array_equal(_flat(first['b']), _flat(other['b']))
    assert not np.array_equal(_flat(first['values']), _flat(other['values']))
    assert np.array_equal(_flat(other['a']), export_cochain(L.canonical_labels(2), 1, 0))


# =============================================================================================
# Complex and mixed-dtype cups against the reference (regression tests from adversarial review).
#
# A mutation that dropped the imaginary part in the cup kernel survived every test above, which
# only used integer cochains.  All values below have integer real and imaginary parts, so every
# comparison is exact in any floating dtype.
# =============================================================================================

COMPLEX_CUP_CASES = [(1, 1, 0), (2, 1, 1), (1, 2, 1), (2, 2, 1), (2, 2, 2), (3, 2, 2), (3, 3, 3)]


def _valued_form(L, p, dtype, *seed):
    r'''A form of the given dtype with small integer entries (and integer imaginary parts if complex).'''
    rng = np.random.default_rng([L.D, L.N, p, 90, *seed])
    shape = (ref.simplices_per_site(L.D, p),) + (L.N,) * L.D
    data = rng.integers(-3, 4, size=shape).astype(np.float64)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        data = data + 1j * rng.integers(-3, 4, size=shape)
    from supervillain.lattice.barycubic import BarycubicForm
    return BarycubicForm(data.astype(dtype), degree=p, lattice=L)


def _reference_cup_of(L, R, a, b, i):
    p, q = a.degree, b.degree
    dtype = np.result_type(np.asarray(a).dtype, np.asarray(b).dtype)
    return R.cup(ref.to_reference_order(np.asarray(a).astype(dtype), _perm(L, R, p)), p,
                 ref.to_reference_order(np.asarray(b).astype(dtype), _perm(L, R, q)), q, i)


def _check_against_reference(L, R, a, b, i):
    from supervillain.lattice.barycubic import cup
    got = cup(a, b, i)
    expected = _reference_cup_of(L, R, a, b, i)
    assert np.iscomplexobj(got) == np.iscomplexobj(expected)
    assert np.array_equal(ref.to_reference_order(got, _perm(L, R, a.degree + b.degree - i)), expected)
    return expected


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("p,q,i", COMPLEX_CUP_CASES)
def test_complex_cup_matches_reference(p, q, i, dtype):
    r'''Complex $a \cup_i b$ equals the reference exactly (real and imaginary parts), for complex128 and complex64 at D = 3, N = 3.'''
    L, R = lattice(3, 3), reference(3, 3)
    a, b = _valued_form(L, p, dtype, 1, q, i), _valued_form(L, q, dtype, 2, p, i)
    expected = _check_against_reference(L, R, a, b, i)
    assert np.imag(expected).any()


@pytest.mark.parametrize("p,q,i", COMPLEX_CUP_CASES)
def test_cup_complex_bilinear(p, q, i):
    r'''$\cup_i$ is complex bilinear: $(\mathrm{i}a) \cup_i b = a \cup_i (\mathrm{i}b) = \mathrm{i}\,(a \cup_i b)$ exactly.'''
    from supervillain.lattice.barycubic import cup
    L = lattice(3, 3)
    a, b = int_form(L, p, 3, q, i), int_form(L, q, 4, p, i)
    ia = ref.wrap(1j * np.asarray(a), a)
    ib = ref.wrap(1j * np.asarray(b), b)
    base = 1j * np.asarray(cup(a, b, i))
    assert np.array_equal(np.asarray(cup(ia, b, i)), base)
    assert np.array_equal(np.asarray(cup(a, ib, i)), base)
    assert np.array_equal(np.asarray(cup(ia, ib, i)), 1j * base)


MIXED_DTYPES = [(np.int64, np.complex128), (np.complex128, np.int64), (np.float32, np.complex128),
                (np.complex64, np.int32), (np.float64, np.complex64), (np.int32, np.float32)]


@pytest.mark.parametrize("dtype_a,dtype_b", MIXED_DTYPES)
def test_mixed_dtype_cup_matches_reference(dtype_a, dtype_b):
    r'''For factors of different dtypes, $a \cup_i b$ equals the reference on the promoted values, for several (p, q, i).'''
    L, R = lattice(3, 3), reference(3, 3)
    for p, q, i in [(1, 1, 0), (2, 1, 1), (2, 2, 2)]:
        _check_against_reference(L, R, _valued_form(L, p, dtype_a, 5, q, i), _valued_form(L, q, dtype_b, 6, p, i), i)


def _noncontiguous(f, kind):
    r'''A full-shape form with the same values as ``f`` whose storage is Fortran-ordered, negatively strided, or step-sliced.'''
    from supervillain.lattice.barycubic import BarycubicForm
    x = np.asarray(f)
    if kind == 'fortran':
        data = np.asfortranarray(x)
    elif kind == 'negative':
        data = np.flip(np.flip(x).copy())                  # a view with negative strides on every axis
    else:
        big = np.zeros((2 * x.shape[0],) + x.shape[1:], dtype=x.dtype)
        big[::2] = x
        data = big[::2]                                    # stride two along the label axis
    assert data.shape == x.shape and not data.flags.c_contiguous
    return BarycubicForm(data, degree=f.degree, lattice=f.lattice)


@pytest.mark.parametrize("kind", ['fortran', 'negative', 'strided'])
@pytest.mark.parametrize("dtype", [np.int64, np.complex128])
def test_cup_noncontiguous_inputs(kind, dtype):
    r'''Non-contiguous full-shape inputs (Fortran order, negative strides, step slices) give the same $a \cup_i b$ as the reference.'''
    L, R = lattice(3, 3), reference(3, 3)
    for p, q, i in [(1, 1, 0), (2, 1, 1), (2, 2, 1)]:
        a, b = _valued_form(L, p, dtype, 7, q, i), _valued_form(L, q, dtype, 8, p, i)
        for x, y in [(_noncontiguous(a, kind), b), (a, _noncontiguous(b, kind)),
                     (_noncontiguous(a, kind), _noncontiguous(b, kind))]:
            _check_against_reference(L, R, x, y, i)


# =============================================================================================
# The export comparer (docs/barycubic-lattice/compare_exports.py), regression tests from adversarial review.
#
# Interface used: compare_exports.compare(A, B, rtol) -> list of report lines, a line not ending in
# ': OK' being a disagreement; write_export.export(D, N, seed) -> dict of arrays.  Coboundary
# entries are compared as mathematical matrix entries: duplicate CSR entries sum.
# =============================================================================================

import importlib.util
import pathlib

_DOCS = pathlib.Path(__file__).resolve().parent.parent / 'docs' / 'barycubic-lattice'


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f'_barycubic_{name}', _DOCS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _disagreements(A, B):
    compare_exports = _load_script('compare_exports')
    return [line for line in compare_exports.compare(A, B, 1e-12) if not line.endswith(': OK')]


def _export(D, N):
    return {k: np.array(v) for k, v in _load_script('write_export').export(D, N, 0).items()}


def _with_row0(arrays, entries):
    r'''A copy of the export whose complex/0/d row 0 is replaced by the (column, value) pairs ``entries``, duplicates allowed.'''
    out = {k: np.array(v) for k, v in arrays.items()}
    data, indices, indptr = (arrays[f'complex/0/d/{k}'] for k in ('data', 'indices', 'indptr'))
    start, stop = indptr[0], indptr[1]
    cols = np.array([c for c, _ in entries], dtype=indices.dtype)
    vals = np.array([v for _, v in entries], dtype=data.dtype)
    out['complex/0/d/data'] = np.concatenate([vals, data[stop:]])
    out['complex/0/d/indices'] = np.concatenate([cols, indices[stop:]])
    out['complex/0/d/indptr'] = indptr - (stop - start) + len(entries)
    out['complex/0/d/indptr'][0] = 0
    return out


def _row0(arrays):
    data, indices, indptr = (arrays[f'complex/0/d/{k}'] for k in ('data', 'indices', 'indptr'))
    return list(zip(indices[indptr[0]:indptr[1]].tolist(), data[indptr[0]:indptr[1]].tolist()))


def test_export_comparer_accepts_identical_and_reordered_exports():
    r'''Two exports that list the same complex in different simplex orders compare equal, quantity by quantity.'''
    import scipy.sparse as sp
    A = _export(1, 2)
    assert _disagreements(A, A) == []
    # Reorder the 1-simplices: labels, weights, orientation, and the rows of d_0.
    B = {k: np.array(v) for k, v in A.items()}
    n1 = len(A['complex/1/labels'])
    perm = np.roll(np.arange(n1), 1)                     # a cyclic shift: never the identity
    for key in ('complex/1/labels', 'complex/1/weights', 'complex/1/orientation'):
        B[key] = A[key][perm]
    d0 = sp.csr_matrix((A['complex/0/d/data'], A['complex/0/d/indices'], A['complex/0/d/indptr']),
                       shape=tuple(A['complex/0/d/shape']))[perm]
    B['complex/0/d/data'], B['complex/0/d/indices'], B['complex/0/d/indptr'] = d0.data, d0.indices, d0.indptr
    for key in [k for k in A if k.startswith('cup/') and k.endswith('/labels') and A[k].shape[1] == 2]:
        prefix = key[:-len('/labels')]
        B[key], B[f'{prefix}/values'] = A[key][perm], A[f'{prefix}/values'][perm]
    for key in [k for k in A if k.startswith('cup/1/') and k.endswith('/a')]:
        B[key] = A[key][perm]
    for key in [k for k in A if k.startswith('cup/') and k.split('/')[2] == '1' and k.endswith('/b')]:
        B[key] = A[key][perm]
    assert not np.array_equal(B['complex/1/labels'], A['complex/1/labels'])
    assert _disagreements(A, B) == []


def test_export_comparer_sums_duplicate_entries():
    r'''CSR data ``[v]`` against ``[v, v]`` with duplicate column indices (entries v against 2v) is flagged as a disagreement in ``d``.'''
    A = _export(1, 2)
    (c, v), *rest = _row0(A)
    B = _with_row0(A, [(c, v), (c, v)] + rest)
    bad = _disagreements(A, B)
    assert any('complex/0/d' in line for line in bad), bad


def test_export_comparer_sees_cancelled_duplicates():
    r'''An entry stored as ``[−v, v]`` (summing to 0) against a single ``v`` is flagged, although the last duplicate matches.'''
    A = _export(1, 2)
    (c, v), *rest = _row0(A)
    B = _with_row0(A, [(c, -v), (c, v)] + rest)
    bad = _disagreements(A, B)
    assert any('complex/0/d' in line for line in bad), bad
