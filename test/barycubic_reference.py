#!/usr/bin/env python
r'''
Slow, independent reference construction of the barycubic complex, for tests only.

The barycubic complex is the barycentric subdivision of the cubical cellulation of the torus
$(\mathbb{R}/N\mathbb{Z})^D$ (docs/barycubic-lattice/SPEC.md §1).  The production code in
``supervillain/lattice/barycubic.py`` builds it from the doubled-coordinate/sign shortcut of
SPEC §2.2.  This module deliberately does *not*: it starts from the cubical cells themselves,

* a cell is a pair ``(x, S)`` of an anchor site ``x`` and a sorted tuple of spanned directions ``S``;
* its faces are enumerated explicitly: ``(x + Σ_{k∈S\T} s_k e_k, T)`` for every proper ``T ⊊ S``
  and every choice of sides ``s_k ∈ {0,1}``;
* a fine p-simplex is a flag ``F_0 ⊊ F_1 ⊊ … ⊊ F_p`` of cells, found by walking down the face
  relation, and its vertices are the barycenters of the flag's cells, in rank (= dimension) order.

Everything is done in the covering space, with unreduced integer anchors, and only reduced mod
``2N`` (doubled coordinates) when a torus label is needed.  Metric quantities (orientations,
Hodge weights, winding displacements) therefore use lifted coordinates by construction
(SPEC §2.2, "common lift").

The geometry is independent of the implementation.  The *cup-product signs* are not: both
builds use the closed-form Steenrod sign rule of SPEC §5.5, ported here from
``docs/barycubic-lattice/verify_cup_i_signs.py`` (:func:`cup_i_sign`, :func:`cup_i_terms`).  That
rule has its own exact oracles (the two ``docs/barycubic-lattice`` scripts and the pointwise
test ``a ∪_p b = ab``).

Hodge weights are computed from geometry exactly as SPEC §5.1 defines them: the barycentric dual
cell of a fine simplex σ is the union, over maximal chains σ = τ_0 ⊊ … ⊊ τ_{D−p} of fine simplices,
of the geometric simplices spanned by the centroids ``c(τ_0) … c(τ_{D−p})``.  Every Gram determinant
is computed in exact integer arithmetic (coordinates scaled by ``2·lcm(1..D+1)``), and equal
determinants are grouped before the square roots are summed, so the float64 result is accurate to
a few ulp.

Conventions shared with the implementation (and only these): doubled coordinates
``ξ = 2x ∈ (Z/2N)^D`` for fine vertices, rank-ordered vertex tuples as canonical global labels
(SPEC §8), and the coboundary ``(dω)(v_0…v_{p+1}) = Σ_j (−1)^j ω(v_0…v̂_j…v_{p+1})`` (SPEC §4).
'''

import itertools
import math
from collections import Counter
from functools import lru_cache

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------------------------
# Closed-form counts (SPEC §2.2)
# ---------------------------------------------------------------------------------------------

def simplices_per_site(D, p):
    r'''$f_p(D) = \sum_{j=0}^{p} (-1)^{p-j} \binom{p}{j} (2j+2)^D$, fine p-simplices per site.'''
    return sum((-1) ** (p - j) * math.comb(p, j) * (2 * j + 2) ** D for j in range(p + 1))


def links_per_site_of_class(D, r, s):
    r'''Fine links joining rank r to rank s per site: $\binom{D}{s}\binom{s}{r}2^{s-r}$.'''
    return math.comb(D, s) * math.comb(s, r) * 2 ** (s - r)


# ---------------------------------------------------------------------------------------------
# The cubical complex and its flags, in the covering space
# ---------------------------------------------------------------------------------------------

def cube_faces(cell):
    r'''
    All proper faces of the cubical cell ``(x, S)``: ``(x + Σ_{k∈S\T} s_k e_k, T)`` for every
    ``T ⊊ S`` and every choice of sides ``s ∈ {0,1}^{S\T}``.  Anchors are unreduced integers.
    '''
    x, S = cell
    for r in range(len(S)):
        for T in itertools.combinations(S, r):
            rest = [k for k in S if k not in T]
            for sides in itertools.product((0, 1), repeat=len(rest)):
                y = list(x)
                for k, s in zip(rest, sides):
                    y[k] += s
                yield (tuple(y), T)


def descending_chains(cell, length):
    r'''Every strictly descending chain ``cell = F_p ⊋ F_{p−1} ⊋ … ⊋ F_0`` with ``length = p``.'''
    if length == 0:
        yield (cell,)
        return
    for face in cube_faces(cell):
        for chain in descending_chains(face, length - 1):
            yield (cell,) + chain


def barycenter(cell):
    r'''Doubled coordinates of the barycenter of ``(x, S)``: ``2x + 1_S`` (unreduced).'''
    x, S = cell
    xi = [2 * c for c in x]
    for k in S:
        xi[k] += 1
    return tuple(xi)


@lru_cache(maxsize=None)
def local_simplices(D, p):
    r'''
    Lifted rank-ordered vertex lists of every fine p-simplex whose top cell is anchored at the
    origin, as an int array ``(f_p, p+1, D)`` with entries in ``{0, 1, 2}``, sorted
    lexicographically (the reference's own order, unrelated to the implementation's).
    '''
    origin = (0,) * D
    out = []
    for r in range(p, D + 1):
        for S in itertools.combinations(range(D), r):
            for chain in descending_chains((origin, S), p):
                out.append(tuple(barycenter(c) for c in reversed(chain)))
    out.sort()
    return np.array(out, dtype=np.int64).reshape(len(out), p + 1, D)


def _key(rows):
    r'''Hashable keys (bytes) for each row of an int64 array ``(M, ...)``.'''
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    return [r.tobytes() for r in rows]


@lru_cache(maxsize=None)
def local_index(D, p):
    r'''dict: lifted local vertex list (as bytes) → row of :func:`local_simplices`.'''
    return {k: i for i, k in enumerate(_key(local_simplices(D, p)))}


def rank_of(xi):
    r'''Rank of a fine vertex = number of odd doubled coordinates (works on arrays, last axis).'''
    return (np.asarray(xi) % 2).sum(axis=-1)


# ---------------------------------------------------------------------------------------------
# Unit-cell incidence tables (for Fourier blocks) built from the reference local simplices
# ---------------------------------------------------------------------------------------------

def face_anchor_and_local(face):
    r'''
    Given the lifted vertex list of a face, return ``(offset, local_vertices)``: the face's anchor
    is ``floor(top vertex / 2)`` (the top vertex is the barycenter ``2x + 1_S`` of its top cell).
    '''
    face = np.asarray(face)
    offset = face[-1] // 2
    return offset, face - 2 * offset


@lru_cache(maxsize=None)
def local_d_table(D, p):
    r'''
    Rows ``(out, in, offset[D], sign)`` of the coboundary from degree p to p+1 on the unit cell:
    ``(dω)(out at x) = Σ sign · ω(in at x + offset)``.
    '''
    rows = []
    index = local_index(D, p)
    for out, V in enumerate(local_simplices(D, p + 1)):
        for j in range(p + 2):
            off, loc = face_anchor_and_local(np.delete(V, j, axis=0))
            rows.append((out, index[_key(loc[None])[0]], *off, (-1) ** j))
    return np.array(rows, dtype=np.int64)


def fourier_d_block(table, f_out, f_in, k):
    r'''$d(k)_{\rm out,in} = \sum_{\rm rows} {\rm sign}\, e^{i k\cdot{\rm offset}}$ for a table ``(out, in, off[D], sign)``.'''
    D = table.shape[1] - 3
    M = np.zeros((f_out, f_in), dtype=complex)
    phase = np.exp(1j * (table[:, 2:2 + D] @ np.asarray(k, dtype=float)))
    np.add.at(M, (table[:, 0], table[:, 1]), table[:, -1] * phase)
    return M


def momenta(D, N):
    r'''All lattice momenta ``2π m / N``, ``m ∈ {0..N−1}^D``.'''
    return [2 * np.pi * np.array(m) / N for m in itertools.product(range(N), repeat=D)]


def fourier_betti(d_blocks, sizes, tol=1e-8):
    r'''
    Betti numbers of the block complex ``C^0 → C^1 → … → C^D`` with coboundary blocks
    ``d_blocks[p]`` (shape ``sizes[p+1] × sizes[p]``): ``b_p = f_p − rank d_p − rank d_{p−1}``.
    '''
    ranks = [np.linalg.matrix_rank(M, tol=tol) for M in d_blocks]
    D = len(sizes) - 1
    return [sizes[p] - (ranks[p] if p < D else 0) - (ranks[p - 1] if p > 0 else 0) for p in range(D + 1)]


def fourier_laplacian_nullities(d_blocks, weights, tol=1e-9):
    r'''
    Null-space dimension of each weighted Laplacian block
    $\Delta_p(k) = d_{p-1} M_{p-1}^{-1} d_{p-1}^\dagger M_p + M_p^{-1} d_p^\dagger M_{p+1} d_p$,
    computed from the Hermitian similarity transform $M_p^{1/2} \Delta_p M_p^{-1/2}$.
    '''
    D = len(weights) - 1
    out = []
    for p in range(D + 1):
        Mh = np.sqrt(weights[p])
        # M^{1/2} Δ M^{-1/2} = A^† A + B B^†, with A = M_{p+1}^{1/2} d_p M_p^{-1/2}, B = M_p^{1/2} d_{p-1} M_{p-1}^{-1/2}.
        H = np.zeros((len(weights[p]),) * 2, dtype=complex)
        if p < D:
            A = np.sqrt(weights[p + 1])[:, None] * d_blocks[p] / Mh[None, :]
            H += A.conj().T @ A
        if p > 0:
            B = Mh[:, None] * d_blocks[p - 1] / np.sqrt(weights[p - 1])[None, :]
            H += B @ B.conj().T
        ev = np.linalg.eigvalsh(H)
        out.append(int((ev < tol * max(1.0, ev.max())).sum()))
    return out


# ---------------------------------------------------------------------------------------------
# Hodge weights from geometry (SPEC §5.1)
# ---------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _cover_top_simplices(D):
    r'''
    Lifted vertex lists of every fine D-simplex in the D-cubes anchored at ``{−1, 0}^D`` (these
    are all the top simplices whose closure can contain a simplex anchored at the origin), and
    their integer vertex codes.
    '''
    local = local_simplices(D, D)
    shifts = np.array(list(itertools.product((-1, 0), repeat=D)), dtype=np.int64)
    tops = (local[None, :, :, :] + 2 * shifts[:, None, None, :]).reshape(-1, D + 1, D)
    return tops, _vertex_codes(tops)


def _vertex_codes(V):
    r'''Distinct integer codes for lifted doubled coordinates in ``[−2, 2]^D`` (last axis).'''
    V = np.asarray(V, dtype=np.int64)
    return ((V + 2) * 5 ** np.arange(V.shape[-1])).sum(axis=-1)


def _bareiss_det(G):
    r'''Exact determinants of a stack of positive-definite integer matrices (object dtype), ``(P, m, m)``.'''
    M = G.copy()
    m = M.shape[1]
    if m == 0:
        return np.ones(M.shape[0], dtype=object)
    prev = np.ones(M.shape[0], dtype=object)
    for k in range(m - 1):
        piv = M[:, k, k].copy()
        for i in range(k + 1, m):
            for j in range(k + 1, m):
                M[:, i, j] = (M[:, i, j] * piv - M[:, i, k] * M[:, k, j]) // prev
        prev = piv
    return M[:, m - 1, m - 1]


def _gram_dets(E):
    r'''Exact Gram determinants of a stack of integer edge-vector sets ``E`` with shape ``(P, m, D)``.'''
    E = E.astype(object)
    G = np.einsum('pad,pbd->pab', E, E) if E.shape[1] else np.zeros((E.shape[0], 0, 0), dtype=object)
    return _bareiss_det(G)


def simplex_volume(V):
    r'''Euclidean p-volume of a simplex with lifted doubled-coordinate vertices ``V`` ``(p+1, D)``: $\sqrt{\det G}/(p!\,2^p)$.'''
    V = np.asarray(V, dtype=np.int64)
    p = len(V) - 1
    if p == 0:
        return 1.0
    det = _gram_dets((V[1:] - V[0])[None])[0]
    return math.sqrt(det) / (math.factorial(p) * 2 ** p)


def dual_cell_volume(V, D):
    r'''
    $(D-p)$-volume of the barycentric dual cell of the fine simplex with lifted local vertices
    ``V`` (anchored at the origin), summed over the pieces $[c(\tau_0) \ldots c(\tau_{D-p})]$.
    '''
    V = np.asarray(V, dtype=np.int64)
    p = len(V) - 1
    m = D - p
    if m == 0:
        return 1.0
    tops, codes = _cover_top_simplices(D)
    # Which top simplices contain every vertex of σ?  (Vertices compared through integer codes.)
    in_sigma = np.isin(codes, _vertex_codes(V))
    contains = in_sigma.sum(axis=1) == p + 1
    star, in_sigma = tops[contains], in_sigma[contains]
    # For each containing top simplex, the D−p vertices not in σ, in every order, give the
    # maximal chains σ = τ_0 ⊊ τ_1 ⊊ … ⊊ τ_m ending at that top simplex.
    extra = star[~in_sigma].reshape(len(star), m, D)                          # (T, m, D), rank order kept
    lcm = math.lcm(*range(1, D + 2))                                           # scale = 2·lcm
    pieces = []
    base = V.sum(axis=0)
    for order in itertools.permutations(range(m)):
        running = np.broadcast_to(base, (len(extra), D)).copy()
        pts = [running * (lcm // (p + 1))]
        for step, idx in enumerate(order):
            running = running + extra[:, idx]
            pts.append(running * (lcm // (p + 2 + step)))
        pts = np.stack(pts, axis=1)                                            # (T, m+1, D), scaled by 2·lcm
        pieces.append(pts[:, 1:] - pts[:, :1])
    dets = _gram_dets(np.concatenate(pieces, axis=0))
    scale = float(2 * lcm) ** m * math.factorial(m)
    return sum(count * math.sqrt(det) for det, count in Counter(dets.tolist()).items()) / scale


def hodge_weight(V, D):
    r'''$M_p(\sigma) = {\rm Vol}_{D-p}(\star\sigma) / {\rm Vol}_p(\sigma)$ for lifted local vertices ``V``.'''
    return dual_cell_volume(V, D) / simplex_volume(V)


@lru_cache(maxsize=None)
def hodge_weights_by_rank_set(D):
    r'''One geometric weight per nonempty rank set ``R ⊆ {0..D}``, from a representative simplex.'''
    out = {}
    for p in range(D + 1):
        local = local_simplices(D, p)
        ranks = rank_of(local)
        for R in itertools.combinations(range(D + 1), p + 1):
            rep = local[(ranks == np.array(R)).all(axis=1)][0]
            out[R] = hodge_weight(rep, D)
    return out


@lru_cache(maxsize=None)
def local_hodge_weights(D, p):
    r'''Geometric weight of every local p-simplex, computed simplex by simplex (no symmetry assumed).'''
    return np.array([hodge_weight(V, D) for V in local_simplices(D, p)])


def affine_normalization(D, weight_of_rank_set):
    r'''
    Kinetic energy per site of the affine field φ = x_0 (unit gradient),
    $\mathcal N = \sum_{\rm links\ per\ site} M_1(e)\,(\Delta x_0(e))^2$, with lifted displacements.
    '''
    local = local_simplices(D, 1)
    ranks = rank_of(local)
    dx = (local[:, 1, 0] - local[:, 0, 0]) / 2
    return sum(weight_of_rank_set[tuple(r)] * x * x for r, x in zip(ranks, dx))


# ---------------------------------------------------------------------------------------------
# The closed-form Steenrod cup-i sign rule (SPEC §5.5), ported from verify_cup_i_signs.py
# ---------------------------------------------------------------------------------------------

def cup_i_sign(p, q, i, n, J):
    r'''
    Sign $\varepsilon(J) = (-1)^{\binom{p}{2} + \binom{n}{2} + \sum_{u\in U_B} {\rm ind}_U(u)}$ of the
    interval-cut term with cut points ``J = (j_0 < … < j_i)``; ``U`` are the uncut vertices and
    ``U_B`` those that appear only in b's face.
    '''
    U = [u for u in range(n + 1) if u not in J]
    e = math.comb(p, 2) + math.comb(n, 2)
    for ind, u in enumerate(U):
        if (u - ind) % 2 == 1:
            e += ind
    return -1 if e % 2 else 1


@lru_cache(maxsize=None)
def cup_i_terms(p, q, i):
    r'''
    Terms ``(sign, A, B)`` of $(a \cup_i b)(v_0 … v_n) = \sum \pm a(v_A)\, b(v_B)$, where the cut
    ``[0, j_0], [j_0, j_1], …, [j_i, n]`` sends even-numbered intervals to A, odd ones to B.
    '''
    n = p + q - i
    out = []
    for J in itertools.combinations(range(n + 1), i + 1):
        cuts = [0, *J, n]
        A, B = [], []
        for k in range(i + 2):
            (A if k % 2 == 0 else B).extend(range(cuts[k], cuts[k + 1] + 1))
        if len(A) == p + 1 and len(B) == q + 1:
            out.append((cup_i_sign(p, q, i, n, J), tuple(A), tuple(B)))
    return tuple(out)


def expected_cup_table(local_vertices, p, q, i):
    r'''
    The contraction table implied by the closed-form sign rule, derived only from the
    implementation's rank-ordered lifted vertex lists ``local_vertices(degree)`` (a callable).

    Returns
    -------
    dict
        ``(out_label, a_label, a_offset, b_label, b_offset) → sign``, offsets as tuples.
    '''
    n = p + q - i
    index_p = {k: c for c, k in enumerate(_key(local_vertices(p)))}
    index_q = {k: c for c, k in enumerate(_key(local_vertices(q)))}
    out = {}
    for o, V in enumerate(np.asarray(local_vertices(n), dtype=np.int64)):
        for sign, A, B in cup_i_terms(p, q, i):
            a_off, a_loc = face_anchor_and_local(V[list(A)])
            b_off, b_loc = face_anchor_and_local(V[list(B)])
            key = (o, index_p[_key(a_loc[None])[0]], tuple(int(t) for t in a_off),
                   index_q[_key(b_loc[None])[0]], tuple(int(t) for t in b_off))
            out[key] = out.get(key, 0) + sign
    return {k: v for k, v in out.items() if v != 0}


def expected_d_table(local_vertices, p):
    r'''
    The incidence table of ``d`` from degree p to p+1 implied by rank-ordered face deletion,
    derived only from ``local_vertices``: ``(out, in, offset) → sign``.
    '''
    index_p = {k: c for c, k in enumerate(_key(local_vertices(p)))}
    out = {}
    for o, V in enumerate(np.asarray(local_vertices(p + 1), dtype=np.int64)):
        for j in range(p + 2):
            off, loc = face_anchor_and_local(np.delete(V, j, axis=0))
            key = (o, index_p[_key(loc[None])[0]], tuple(int(t) for t in off))
            out[key] = out.get(key, 0) + (-1) ** j
    return {k: v for k, v in out.items() if v != 0}


# ---------------------------------------------------------------------------------------------
# The global torus complex
# ---------------------------------------------------------------------------------------------

class ReferenceComplex:
    r'''
    The barycubic complex on $(\mathbb{Z}/N)^D$, built from explicit flags.

    Attributes
    ----------
    lifted : dict p → int array ``(count_p, p+1, D)``
        Lifted rank-ordered vertices of every fine p-simplex; the top cell's anchor lies in
        ``[0, N)^D``.
    reduced : dict p → int array
        The same, reduced mod ``2N``: the canonical global labels (SPEC §8).
    index : dict p → dict
        Reduced vertex list (as bytes) → row.
    '''

    def __init__(self, D, N):
        self.D, self.N = D, N
        anchors = np.array(list(itertools.product(range(N), repeat=D)), dtype=np.int64).reshape(-1, D)
        self.lifted, self.reduced, self.index = {}, {}, {}
        for p in range(D + 1):
            local = local_simplices(D, p)
            lifted = (local[None, :, :, :] + 2 * anchors[:, None, None, :]).reshape(-1, p + 1, D)
            self.lifted[p] = lifted
            self.reduced[p] = lifted % (2 * N)
            keys = _key(self.reduced[p])
            self.index[p] = {k: i for i, k in enumerate(keys)}
        self._d = {}
        self._cup = {}

    def count(self, p):
        return len(self.lifted[p])

    def lookup(self, p, vertices):
        r'''Row indices of fine p-simplices given reduced (or unreduced) vertex lists ``(M, p+1, D)``.'''
        vertices = np.asarray(vertices, dtype=np.int64) % (2 * self.N)
        idx = self.index[p]
        return np.array([idx[k] for k in _key(vertices)], dtype=np.int64)

    def d(self, p):
        r'''Sparse coboundary matrix from degree p to p+1: ``(dω)(v_0…v_{p+1}) = Σ_j (−1)^j ω(…v̂_j…)``.'''
        if p not in self._d:
            rows, cols, vals = [], [], []
            simp = self.reduced[p + 1]
            for j in range(p + 2):
                faces = np.delete(simp, j, axis=1)
                rows.append(np.arange(len(simp)))
                cols.append(self.lookup(p, faces))
                vals.append(np.full(len(simp), (-1) ** j, dtype=np.int64))
            self._d[p] = sp.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.count(p + 1), self.count(p)), dtype=np.int64)
        return self._d[p]

    @property
    def orientation(self):
        r'''$\varepsilon_\sigma$ from the sign of the determinant of the lifted edge vectors.'''
        V = self.lifted[self.D]
        E = (V[:, 1:] - V[:, :-1]).astype(float)
        return np.rint(np.linalg.det(E)).astype(np.int64)

    def cup(self, a, p, b, q, i):
        r'''Reference $a \cup_i b$ on vectors in reference order, via vertex-tuple face lookups.'''
        n = p + q - i
        key = (p, q, i)
        if key not in self._cup:
            simp = self.reduced[n]
            self._cup[key] = [(s, self.lookup(p, simp[:, list(A)]), self.lookup(q, simp[:, list(B)]))
                              for s, A, B in cup_i_terms(p, q, i)]
        out = np.zeros(self.count(n), dtype=np.result_type(a, b))
        for s, ia, ib in self._cup[key]:
            out += s * a[ia] * b[ib]
        return out

    def coarsen(self, omega, p):
        r'''
        $(Sd^\top \omega)(F) = \sum_{\sigma \subseteq F,\ \dim\sigma = p} o(\sigma, F)\, \omega[\sigma]$ as a
        hypercubic-layout array ``(C(D,p), N, …, N)``; $o$ is the sign of the determinant of σ's
        lifted edge vectors restricted to F's directions in increasing order.
        '''
        D, N = self.D, self.N
        out = np.zeros((math.comb(D, p),) + (N,) * D, dtype=np.asarray(omega).dtype)
        local = local_simplices(D, p)
        ranks = rank_of(local)
        for c, S in enumerate(itertools.combinations(range(D), p)):
            top = np.zeros(D, dtype=np.int64)
            top[list(S)] = 1
            inside = (ranks == np.arange(p + 1)).all(axis=1) & (local[:, -1] == top).all(axis=1)
            for V in local[inside]:
                E = (V[1:] - V[:-1])[:, list(S)].astype(float)
                o = int(np.rint(np.linalg.det(E))) if p else 1
                for x in itertools.product(range(N), repeat=D):
                    row = self.lookup(p, (V + 2 * np.array(x))[None])[0]
                    out[(c,) + x] += o * omega[row]
        return out

    def betti(self):
        r'''Betti numbers over the rationals from dense ranks (small complexes only).'''
        ranks = [np.linalg.matrix_rank(self.d(p).toarray().astype(float)) for p in range(self.D)]
        return [self.count(p) - (ranks[p] if p < self.D else 0) - (ranks[p - 1] if p > 0 else 0)
                for p in range(self.D + 1)]


def permutation_to_reference(canonical_labels, ref, p):
    r'''
    ``perm`` with ``perm[i]`` = reference row of the implementation's flat entry ``i``, found by
    matching the implementation's canonical labels (rank-ordered reduced vertex lists).
    '''
    return ref.lookup(p, canonical_labels)


def to_reference_order(values, perm):
    r'''Reorder a flattened implementation form into reference order.'''
    flat = np.asarray(values).reshape(-1)
    out = np.empty_like(flat)
    out[perm] = flat
    return out


def momentum_orbit_representatives(D, N):
    r'''
    One momentum per orbit of the hyperoctahedral group acting on the ``N``-grid of momenta in
    ``D`` dimensions: ``2π/N · (m_0 ≥ m_1 ≥ … ≥ m_{D−1})`` with ``0 ≤ m ≤ N//2``.  Because the
    complex is invariant under permutations and reflections of the axes, the Fourier blocks at
    ``k`` and ``g·k`` are unitarily equivalent, so their Betti numbers and Laplacian nullities
    agree; checking one momentum per orbit is therefore exhaustive.
    '''
    reps = itertools.combinations_with_replacement(range(N // 2, -1, -1), D)
    return [2 * np.pi * np.array(m) / N for m in reps]


# ---------------------------------------------------------------------------------------------
# Shared test parameters and implementation-facing helpers.
#
# Nothing above this line touches supervillain.lattice.barycubic.  The helpers below only wrap
# arrays for the tests; imports are local so that this module (and the self-checks) load even
# while the implementation is a stub.
# ---------------------------------------------------------------------------------------------

#: SPEC §7: D ∈ {1,2,3,4}, N ∈ {2,3,4}; D = 4 only at N ∈ {2,3}.
DN_ALL = [(D, N) for D in (1, 2, 3) for N in (2, 3, 4)] + [(4, 2), (4, 3)]
#: Bilinear (cup) identities with many (p, q, i) cases.
#: (4, 3) was added after the implementation review: at N = 2 an offset of +1 and −1 name the
#: same site, so sign or offset errors in the tables can hide; N = 3 separates them.
DN_CUP = [(D, N) for D in (1, 2, 3) for N in (2, 3, 4)] + [(4, 2), (4, 3)]
#: Comparisons against the slow Python reference build (same reason for (4, 3)).
DN_REF = [(D, N) for D in (1, 2, 3) for N in (2, 3)] + [(4, 2), (4, 3)]


@lru_cache(maxsize=None)
def lattice(D, N):
    r'''A cached :class:`~supervillain.lattice.barycubic.BarycubicLattice`.'''
    from supervillain.lattice.barycubic import BarycubicLattice
    return BarycubicLattice(D, N)


@lru_cache(maxsize=None)
def reference(D, N):
    r'''A cached :class:`ReferenceComplex`.'''
    return ReferenceComplex(D, N)


def int_form(L, p, *seed, low=-3, high=4, dtype=np.int64):
    r'''A :class:`BarycubicForm` of degree p with random integers in ``[low, high)``, reproducibly seeded.'''
    from supervillain.lattice.barycubic import BarycubicForm
    rng = np.random.default_rng([L.D, L.N, p, *seed])
    shape = (simplices_per_site(L.D, p),) + (L.N,) * L.D
    return BarycubicForm(rng.integers(low, high, size=shape).astype(dtype), degree=p, lattice=L)


def float_form(L, p, *seed, dtype=float):
    r'''A :class:`BarycubicForm` of degree p with standard-normal entries (complex if ``dtype`` is complex).'''
    from supervillain.lattice.barycubic import BarycubicForm
    rng = np.random.default_rng([L.D, L.N, p, 7, *seed])
    shape = (simplices_per_site(L.D, p),) + (L.N,) * L.D
    data = rng.standard_normal(shape)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        data = data + 1j * rng.standard_normal(shape)
    return BarycubicForm(data.astype(dtype), degree=p, lattice=L)


def wrap(values, like):
    r'''Re-wrap a plain array as a :class:`BarycubicForm` with the degree and lattice of ``like``.'''
    from supervillain.lattice.barycubic import BarycubicForm
    return BarycubicForm(np.asarray(values), degree=like.degree, lattice=like.lattice)


def roll_sites(form, shift):
    r'''Translate a form by ``shift`` sites: ``out[c, x] = form[c, x − shift]``, rewrapped.'''
    D = len(shift)
    rolled = np.roll(np.asarray(form), tuple(shift), axis=tuple(range(1, D + 1)))
    return wrap(rolled, form)


def is_zero(x):
    r'''True if ``x`` is the scalar 0 returned at the ends of the complex, or an all-zero array.'''
    return not np.asarray(x).any()


def legal_cup_degrees(D):
    r'''All ``(p, q, i)`` with ``0 ≤ i ≤ min(p, q)`` and ``p + q − i ≤ D``.'''
    return [(p, q, i) for p in range(D + 1) for q in range(D + 1)
            for i in range(min(p, q) + 1) if p + q - i <= D]
