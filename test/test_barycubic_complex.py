#!/usr/bin/env python
r'''
Combinatorics, topology, and metric of the barycubic complex (SPEC §7 items 1, 2, 3, 5, 6).

The barycubic lattice is the barycentric subdivision of the periodic hypercubic lattice.  Before
any operator is trusted, the complex itself has to be right:

* **Counts and labels** (item 1): the number of fine p-simplices per site is the inclusion–exclusion
  formula $f_p(D)$, the labels follow the doubled-coordinate/sign rule and ordering of SPEC §2.2,
  every top simplex has one vertex of each rank, and links have length $\sqrt{s-r}/2$.
* **Simplicial-complex invariant** (item 2): after periodic identification distinct simplices have
  distinct vertex sets, even at N = 2.  The Alexander–Whitney and Steenrod products need this.
* **Topology** (item 3): translation invariance block-diagonalizes $d$ by momentum; the unit-cell
  complex at $k=0$ has Betti numbers $\binom{D}{p}$ (the D-torus), every $k \ne 0$ block is acyclic,
  and the weighted Laplacian's null spaces agree (Hodge theorem).  Adjoints use $d(k)^\dagger$.
* **Scripts in the notes** (item 5): the D = 4 link classes and the plaquette factorization of the
  cup square $q = dn \cup dn$.
* **Hodge weights** (item 6): sum rules, the closed-form vertex weights, the half-translation mirror
  $M(R) = M(D-R)$, the ten D = 4 link weights, and the affine-field normalization.
* **Rank coloring** (SPEC §6.3): a proper D+1 coloring of the fine 1-skeleton, as full-axis selectors.

Weights are compared against closed forms here; the independent geometric computation lives in
``test_barycubic_reference.py``.
'''

import itertools
import math

import numpy as np
import pytest

import barycubic_reference as ref
from barycubic_reference import DN_ALL, lattice, int_form


D4_LINK_WEIGHTS = {
    (0, 1): math.sqrt(5) / 10, (0, 2): math.sqrt(6) / 60, (0, 3): 1 / 40, (0, 4): math.sqrt(2) / 40,
    (1, 2): math.sqrt(6) / 30, (1, 3): math.sqrt(2) / 60, (1, 4): 1 / 40,
    (2, 3): math.sqrt(6) / 30, (2, 4): math.sqrt(6) / 60, (3, 4): math.sqrt(5) / 10,
}
D4_AFFINE_NORMALIZATION = (math.sqrt(5) + 2 * math.sqrt(6) + 3 + 2 * math.sqrt(2)) / 10


def _mask(S):
    return sum(1 << k for k in S)


def _set(mask, D):
    return [k for k in range(D) if mask >> k & 1]


# ---------------------------------------------------------------------------------------------
# Item 1: counts and labels
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("D,N", [(D, N) for D in (1, 2, 3, 4) for N in (2, 3)] + [(5, 2)])
def test_counts(D, N):
    r'''$f_p$ per site follows inclusion–exclusion; χ = 0; $f_1 = 4^D - 2^D$; $f_D = 2^D D!$; totals are $f_p N^D$.'''
    # SPEC §2.2, §7 item 1.  (D, N) = (5, 2) is the D = 5 smoke case.
    L = lattice(D, N)
    f = [ref.simplices_per_site(D, p) for p in range(D + 1)]
    assert L.sites == N ** D
    assert tuple(L.dims) == (N,) * D
    assert L.dim == D
    for p in range(D + 1):
        assert L.per_site[p] == f[p]
        assert len(L.labels[p]) == f[p]
        assert L.cells_of_degree[p] == f[p] * N ** D
    assert sum((-1) ** p * fp for p, fp in enumerate(f)) == 0
    assert f[1] == 4 ** D - 2 ** D
    assert f[D] == 2 ** D * math.factorial(D)
    if D == 4:
        assert f == [16, 240, 800, 960, 384]


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_labels_are_ordered_and_indexed(D):
    r'''Local labels are strictly nested chains with one sign per direction of $S_p \setminus S_0$, sorted by $({\rm mask}(S_0),…,{\rm mask}(S_p),{\rm bits}(\sigma))$; ``label_index`` inverts ``labels``.'''
    # SPEC §2.2 (ordering of local labels).
    L = lattice(D, 2)
    for p in range(D + 1):
        labels = L.labels[p]
        keys = []
        for c, lab in enumerate(labels):
            masks, signs = tuple(lab.masks), tuple(lab.signs)
            assert len(masks) == p + 1
            assert all(a & b == a and a != b for a, b in zip(masks, masks[1:]))   # S_a ⊊ S_{a+1}
            assert len(signs) == len(_set(masks[-1] & ~masks[0], D))
            assert set(signs) <= {-1, 1}
            keys.append(masks + tuple(0 if s == -1 else 1 for s in signs))
            assert L.label_index[p][lab] == c
        assert keys == sorted(keys) and len(set(keys)) == len(keys)


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_local_vertices_follow_label_rule(D):
    r'''Lifted vertices: top $\xi^p = 1_{S_p}$ (anchor 0) and $\xi^a_k = \xi^p_k + \sigma_k$ for $k \in S_p \setminus S_a$.'''
    # SPEC §2.2 (fine simplices, common lift).
    L = lattice(D, 2)
    for p in range(D + 1):
        V = np.asarray(L.local_vertices(p))
        assert V.shape == (ref.simplices_per_site(D, p), p + 1, D)
        for c, lab in enumerate(L.labels[p]):
            masks = tuple(lab.masks)
            moving = _set(masks[-1] & ~masks[0], D)
            sigma = dict(zip(moving, lab.signs))
            top = np.zeros(D, dtype=int)
            top[_set(masks[-1], D)] = 1
            for a in range(p + 1):
                expected = top.copy()
                for k in _set(masks[-1] & ~masks[a], D):
                    expected[k] += sigma[k]
                assert np.array_equal(V[c, a], expected), (p, c, a)


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_rank_sets_and_one_vertex_per_rank(D):
    r'''``rank_sets[p]`` lists $|S_0| < … < |S_p|$ (the parities of the vertices); every top simplex has ranks exactly $0, 1, …, D$.'''
    # SPEC §1, §7 item 1.
    L = lattice(D, 2)
    for p in range(D + 1):
        ranks = np.asarray(L.rank_sets[p])
        assert ranks.shape == (ref.simplices_per_site(D, p), p + 1)
        assert np.array_equal(ranks, ref.rank_of(np.asarray(L.local_vertices(p))))
        assert (np.diff(ranks, axis=1) > 0).all()
        for c, lab in enumerate(L.labels[p]):
            assert list(ranks[c]) == [bin(m).count('1') for m in lab.masks]
    assert (np.asarray(L.rank_sets[D]) == np.arange(D + 1)).all()


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_link_lengths_and_classes(D):
    r'''A fine link of class (r, s) has lifted length $\sqrt{s-r}/2$, and there are $\binom{D}{s}\binom{s}{r}2^{s-r}$ per site.'''
    # SPEC §7 items 1 and 5 (D = 4: 8, 24, 32, 16, 24, 48, 32, 24, 24, 8).
    L = lattice(D, 2)
    V = np.asarray(L.local_vertices(1))
    ranks = np.asarray(L.rank_sets[1])
    length = np.sqrt((((V[:, 1] - V[:, 0]) / 2) ** 2).sum(axis=1))
    assert np.allclose(length, np.sqrt(ranks[:, 1] - ranks[:, 0]) / 2, rtol=0, atol=1e-15)
    for r, s in itertools.combinations(range(D + 1), 2):
        assert ((ranks[:, 0] == r) & (ranks[:, 1] == s)).sum() == ref.links_per_site_of_class(D, r, s)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_canonical_labels_layout(D, N):
    r'''Canonical label of entry ``[c, x]`` (C order) is ``(local_vertices(p)[c] + 2x) mod 2N``.'''
    # SPEC §8.
    L = lattice(D, N)
    for p in range(D + 1):
        f = ref.simplices_per_site(D, p)
        got = np.asarray(L.canonical_labels(p)).reshape((f,) + (N,) * D + (p + 1, D))
        x = np.stack(np.meshgrid(*[np.arange(N)] * D, indexing='ij'), axis=-1)       # (N,…,N, D)
        V = np.asarray(L.local_vertices(p))
        expected = (V.reshape((f,) + (1,) * D + (p + 1, D)) + 2 * x[None, ..., None, :]) % (2 * N)
        assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------------------------
# Item 2: simplicial-complex invariant
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("D,N", [(D, N) for D in (1, 2, 3, 4) for N in (2, 3)])
def test_simplicial_complex_invariant(D, N):
    r'''After periodic identification, every simplex has p+1 distinct vertices and distinct simplices have distinct vertex sets.'''
    # SPEC §1, §7 item 2.  N = 2 is the edge case.
    L = lattice(D, N)
    for p in range(D + 1):
        labels = np.asarray(L.canonical_labels(p))
        sets = {frozenset(map(tuple, s)) for s in labels}
        assert len(sets) == len(labels) == N ** D * ref.simplices_per_site(D, p)
        assert all(len(s) == p + 1 for s in sets)


# ---------------------------------------------------------------------------------------------
# Item 3: topology by Fourier blocks
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("D,N", DN_ALL)
def test_fourier_topology(D, N):
    r'''
    $d(k)$ built from ``operator_table('d', p)``: Betti numbers $\binom{D}{p}$ at $k = 0$, acyclic at every
    $k \ne 0$; the weighted Laplacian blocks (with $d(k)^\dagger$) have the same null-space dimensions.
    '''
    # SPEC §7 item 3.  D = 4 checks one momentum per hyperoctahedral orbit, which is exhaustive
    # because the complex is point-group symmetric (checked in test_barycubic_cup.py).
    L = lattice(D, N)
    sizes = [L.per_site[p] for p in range(D + 1)]
    tables = [np.asarray(L.operator_table('d', p)) for p in range(D)]
    weights = [np.asarray(L.hodge_weights(p), dtype=float) for p in range(D + 1)]
    for t in tables:
        assert t.shape[1] == D + 3
    ks = ref.momenta(D, N) if D < 4 else ref.momentum_orbit_representatives(D, N)
    for k in ks:
        blocks = [ref.fourier_d_block(tables[p], sizes[p + 1], sizes[p], k) for p in range(D)]
        for p in range(D - 1):
            assert np.allclose(blocks[p + 1] @ blocks[p], 0, atol=1e-12)
        expected = [math.comb(D, p) for p in range(D + 1)] if not np.any(k) else [0] * (D + 1)
        assert ref.fourier_betti(blocks, sizes) == expected, k
        assert ref.fourier_laplacian_nullities(blocks, weights) == expected, k


# ---------------------------------------------------------------------------------------------
# Item 6: Hodge weights
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_weight_table_structure(D):
    r'''One positive entry per nonempty rank set ($2^{D+1}-1$), mirror symmetric $M(R) = M(D-R)$, and ``hodge_weights`` reads it by rank set.'''
    # SPEC §5.1 "Symmetry", §6.2, §7 item 6.
    L = lattice(D, 2)
    table = {tuple(k): v for k, v in L.hodge_weight_by_rank_set.items()}
    expected_keys = {R for n in range(1, D + 2) for R in itertools.combinations(range(D + 1), n)}
    assert set(table) == expected_keys
    assert len(table) == 2 ** (D + 1) - 1
    assert all(w > 0 for w in table.values())
    for R, w in table.items():
        assert w == pytest.approx(table[tuple(sorted(D - r for r in R))], rel=1e-13)
    for p in range(D + 1):
        w = np.asarray(L.hodge_weights(p))
        assert w.shape == (ref.simplices_per_site(D, p),)
        ranks = np.asarray(L.rank_sets[p])
        assert np.allclose(w, [table[tuple(r)] for r in ranks], rtol=1e-15, atol=0)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_weight_sum_rules(D, N):
    r'''Vertex dual cells tile the torus, $\sum_v M_0(v) = N^D$; top simplices do too, $\sum_\sigma 1/M_D(\sigma) = N^D$.'''
    # SPEC §7 item 6.
    from supervillain.lattice.barycubic import inner
    L = lattice(D, N)
    one = L.zeros(0) + 1.0
    assert inner(one, one) == pytest.approx(N ** D, rel=1e-13)
    assert N ** D * (1 / np.asarray(L.hodge_weights(D))).sum() == pytest.approx(N ** D, rel=1e-13)


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_boundary_degree_weights(D):
    r'''$M_D = 2^D D!$ (inverse fine-simplex volume) and the closed form $M_0(r) = 1/((D+1)\binom{D}{r})$.'''
    # SPEC §5.1 "Boundary degrees" and "Vertex weights in closed form".
    L = lattice(D, 2)
    assert np.allclose(np.asarray(L.hodge_weights(D)), 2 ** D * math.factorial(D), rtol=1e-14, atol=0)
    table = {tuple(k): v for k, v in L.hodge_weight_by_rank_set.items()}
    for r in range(D + 1):
        assert table[(r,)] == pytest.approx(1 / ((D + 1) * math.comb(D, r)), rel=1e-13)


def test_d4_link_weights_and_affine_normalization():
    r'''D = 4: the ten link weights of the notes; 18 distinct values among 31 entries; $\mathcal N = \sum M_1 (\Delta x_0)^2 = (\sqrt5+2\sqrt6+3+2\sqrt2)/10$ per site.'''
    # SPEC §5.1, §7 items 5 and 6.
    L = lattice(4, 2)
    table = {tuple(k): v for k, v in L.hodge_weight_by_rank_set.items()}
    for rs, w in D4_LINK_WEIGHTS.items():
        assert table[rs] == pytest.approx(w, rel=1e-13), rs
    assert len(np.unique(np.round(list(table.values()), 12))) == 18
    # Kinetic energy per site of the affine field φ = x_0, with lifted link displacements.
    V = np.asarray(L.local_vertices(1))
    dx0 = (V[:, 1, 0] - V[:, 0, 0]) / 2
    assert (np.asarray(L.hodge_weights(1)) * dx0 ** 2).sum() == pytest.approx(D4_AFFINE_NORMALIZATION, rel=1e-13)


def test_dimension_one_vertex_weight_is_one_half():
    r'''At D = 1 every vertex dual cell has length 1/2, so ``star`` of an integer basis 0-form is 1/2 (and floating point).'''
    # SPEC §2.3 "Dtypes".
    from supervillain.lattice.barycubic import star
    L = lattice(1, 3)
    e = L.zeros(0, dtype=np.int64)
    e[0, 0] = 1
    s = star(e)
    assert np.issubdtype(np.asarray(s).dtype, np.floating)
    assert np.asarray(s)[0, 0] == 0.5


# ---------------------------------------------------------------------------------------------
# Item 5: the plaquette factorization of the cup square at D = 4
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 3])
def test_plaquette_factorization(N):
    r'''
    For an integer link field n, the cup square $q = dn \cup dn$ on a top simplex $(v_0 … v_4)$ is
    $(n_{12} - n_{02} + n_{01})(n_{34} - n_{24} + n_{23})$, and $\int q = 0$.
    '''
    # SPEC §7 item 5 (verify-plaquette-factorization.py), with explicit vertex lookups.
    from supervillain.lattice.barycubic import d, cup, integrate
    L = lattice(4, N)
    n = int_form(L, 1, 55)
    q = np.asarray(cup(d(n), d(n))).reshape(-1)
    links = {k: i for i, k in enumerate(ref._key(np.asarray(L.canonical_labels(1))))}
    tops = np.asarray(L.canonical_labels(4), dtype=np.int64)
    flat = np.asarray(n).reshape(-1)

    def link(a, b):
        return flat[[links[k] for k in ref._key(tops[:, [a, b]])]]

    expected = (link(1, 2) - link(0, 2) + link(0, 1)) * (link(3, 4) - link(2, 4) + link(2, 3))
    assert np.array_equal(q, expected)
    assert integrate(cup(d(n), d(n))) == 0


# ---------------------------------------------------------------------------------------------
# SPEC §6.3: rank coloring
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("D,N", DN_ALL)
def test_rank_checkerboarding(D, N):
    r'''The D+1 color selectors (over all axes of a 0-form) cover every vertex once, each is a single rank, and no link joins one color.'''
    # SPEC §6.3.  Both parities of N.
    L = lattice(D, N)
    colors = L.checkerboarding
    assert len(colors) == D + 1
    count = np.zeros((L.per_site[0],) + (N,) * D, dtype=int)
    color_of = np.full(count.shape, -1)
    for c, sel in enumerate(colors):
        assert len(sel) == D + 1
        count[sel] += 1
        color_of[sel] = c
    assert (count == 1).all()
    vertex_rank = np.broadcast_to(np.asarray(L.rank_sets[0])[:, 0].reshape((-1,) + (1,) * D), count.shape)
    for c in range(D + 1):
        assert len(np.unique(vertex_rank[color_of == c])) == 1
    vertices = {k: i for i, k in enumerate(ref._key(np.asarray(L.canonical_labels(0))))}
    links = np.asarray(L.canonical_labels(1), dtype=np.int64)
    ends = [np.array([vertices[k] for k in ref._key(links[:, [a]])]) for a in (0, 1)]
    flat = color_of.reshape(-1)
    assert (flat[ends[0]] != flat[ends[1]]).all()
