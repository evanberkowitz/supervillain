#!/usr/bin/env python
# coding: utf-8
#
# barycubic.py — cochains on the barycentric subdivision of the D-dimensional
# periodic hypercubic lattice ("barycubic lattice").
#
# DEVELOPER NOTES
# ---------------
# The design is fixed by docs/barycubic-lattice/SPEC.md; section numbers (§) in
# comments refer to that file.  Three ideas carry the whole module.
#
# 1. Doubled coordinates (§2.1).  Every fine vertex is the barycenter of one
#    hypercubic cell, and 2 * barycenter is an integer vector ξ whose odd
#    entries are exactly the directions the cell spans.  So fine vertices are
#    the points of the doubled lattice (Z/2N)^D, and the rank of a vertex is
#    the number of odd entries of ξ.
#
# 2. Translation invariance.  Every fine simplex is a translate, by an integer
#    site x, of one of finitely many simplices "anchored at the origin" (§2.2).
#    A p-cochain is therefore stored as an array of shape (f_p, N, ..., N),
#    [label, x], exactly as a hypercubic p-form is stored as
#    (C(D, p), N, ..., N), [component, x].  All combinatorics (labels, faces,
#    cup-product terms, weights, orientations) is done ONCE per dimension D on
#    the origin-anchored simplices and cached at module level; it does not
#    depend on N.  Applying an operator is then one pass of a compiled kernel
#    (_barycubic_kernels.py) over the table rows, reading neighbors through a
#    per-lattice table of periodic site shifts: no Python loop ever runs over
#    sites, and no shifted copy of the input is made (§9).  Every cached array
#    is read-only, and accessors hand out read-only views of it.
#
# 3. The common lift (§2.2).  Stored vertex coordinates are reduced mod 2N,
#    but all geometry (volumes, orientations, displacements) is computed from
#    the UNREDUCED vertices of the origin-anchored simplex, whose entries lie
#    in {0, 1, 2}.  Reduced representatives give wrong differences across the
#    periodic seam (e.g. D = 1, N = 2: the lower vertex of the link with top
#    ξ = 3 and σ = +1 is stored as 0, but the true displacement is -1).  The
#    local tables never see the seam; it enters only through the periodic
#    neighbor-site tables (BarycubicLattice._neighbors).

from collections import namedtuple
from functools import cached_property, lru_cache
from itertools import combinations, permutations, product
from math import comb, factorial

import numpy as np
import scipy.sparse

from supervillain.h5 import ReadWriteable
from supervillain.lattice import _barycubic_kernels as _kernels


LocalLabel = namedtuple('LocalLabel', ['masks', 'signs'])
LocalLabel.__doc__ = r'''
Label of a fine simplex relative to its anchor site.

A fine p-simplex is a flag of hypercubic cells $F_0 \subsetneq \cdots \subsetneq F_p$, one vertex
at the barycenter of each.  Relative to the anchor site of its top cell $F_p$ it is fixed by the
chain of cell types (the sets of directions the cells span) and by which side of the top cell
each lower cell sits on.

``masks`` is the strictly increasing chain of cell types $S_0 \subsetneq \cdots \subsetneq S_p$,
each written as a bitmask $\sum_{k \in S} 2^k$.  ``signs`` holds one $\sigma_k \in \{-1,+1\}$ per
direction $k \in S_p \setminus S_0$, in increasing $k$: in doubled coordinates the lower vertices
are displaced from the top vertex by $\sigma_k$ along every such direction they do not span.
'''

LocalLabel.masks.__doc__ = r'''Bitmasks of the cell types $S_0 \subsetneq \cdots \subsetneq S_p$.'''
LocalLabel.signs.__doc__ = r'''One $\sigma_k = \pm 1$ per direction $k \in S_p \setminus S_0$, in increasing $k$.'''


# ---------------------------------------------------------------------------------------------
# Closed forms that do not depend on the geometry
# ---------------------------------------------------------------------------------------------

def _perm_sign(seq):
    '''Sign of the permutation that sorts a sequence of distinct integers.'''
    inversions = sum(1 for a in range(len(seq)) for b in range(a + 1, len(seq)) if seq[a] > seq[b])
    return -1 if inversions % 2 else 1


def _popcount(mask):
    return bin(mask).count('1')


def _directions(mask, D):
    '''The directions k with bit k set, in increasing order.'''
    return tuple(k for k in range(D) if (mask >> k) & 1)


@lru_cache(maxsize=None)
def _cup_terms(p, q, i):
    r'''
    The interval-cut terms of $\cup_i$ on one rank-ordered $n$-simplex, $n = p + q - i$.

    Returns a tuple of ``(sign, A, B)``: the term contributes ``sign * a(v_A) * b(v_B)``, where
    ``A`` and ``B`` are the vertex positions of the two faces.  This is the closed-form rule
    checked by docs/barycubic-lattice/verify_cup_i_signs.py (§5.5), transcribed term by term:
    the cut points J = (j_0 < ... < j_i) split 0..n into i + 2 intervals that share their
    endpoints; even-numbered intervals form a's face and odd-numbered ones form b's face.
    '''
    n = p + q - i
    terms = []
    for J in combinations(range(n + 1), i + 1):
        cuts = (0, *J, n)
        A, B = [], []
        for k in range(i + 2):
            (A if k % 2 == 0 else B).extend(range(cuts[k], cuts[k + 1] + 1))
        if len(A) != p + 1 or len(B) != q + 1:
            continue
        # ε(J) = (-1)^{C(p,2) + C(n,2) + Σ_{u ∈ U_B} ind_U(u)}: U are the uncut positions, and
        # an uncut u belongs only to b's face exactly when an odd number of cuts lie below it,
        # i.e. when u - ind_U(u) is odd.
        exponent = comb(p, 2) + comb(n, 2)
        uncut = [u for u in range(n + 1) if u not in J]
        for ind, u in enumerate(uncut):
            if (u - ind) % 2 == 1:
                exponent += ind
        terms.append((-1 if exponent % 2 else 1, tuple(A), tuple(B)))
    return tuple(terms)


def _is_degree(p):
    '''An integer usable as a degree (bool is excluded: True is not a degree).'''
    return isinstance(p, (int, np.integer)) and not isinstance(p, (bool, np.bool_))


def _check_degree(D, p, what='degree'):
    r'''Raise ValueError unless p is an integer in 0..D; return it as a Python int.'''
    if not _is_degree(p) or not 0 <= p <= D:
        raise ValueError(f"{what} must be an integer in 0..{D}, got {p!r}")
    return int(p)


def _check_cup_degrees(D, p, q, i):
    if not isinstance(i, (int, np.integer)) or isinstance(i, bool):
        raise TypeError(f"cup_i needs an integer i, got {i!r}")
    _check_degree(D, p, 'the degree p of the first factor')
    _check_degree(D, q, 'the degree q of the second factor')
    if i < 0 or i > min(p, q):
        raise ValueError(f"cup_{i} of a {p}-form and a {q}-form needs 0 <= i <= min(p, q) = {min(p, q)}")
    if p + q - i > D:
        raise ValueError(f"cup_{i} of a {p}-form and a {q}-form has degree {p + q - i} > D = {D}")


# ---------------------------------------------------------------------------------------------
# Per-dimension local complex: everything that depends on D but not on N
# ---------------------------------------------------------------------------------------------

def _frozen(array):
    r'''
    A read-only array that owns its memory, for the caches.  Callers only ever receive views of
    it (see _view).  An array that is itself a view is copied first: numpy lets a view be made
    writeable again whenever the array owning the memory is writeable, so only freezing the
    owner makes the freeze stick.
    '''
    if array.base is not None:
        array = array.copy()
    array.setflags(write=False)
    return array


def _view(owner):
    r'''
    A read-only view of a cached, read-only array.  numpy refuses to make a view writeable when
    its base is not writeable, so nothing handed out can corrupt the shared per-D caches.
    '''
    view = owner.view()
    view.setflags(write=False)
    return view


def _orientation_sign(masks, signs):
    r'''
    The sign sgn(k_0, ..., k_{p-1}) · Π_k (-σ_k) of a rank-ordered simplex whose type grows by
    direction k_a between ranks a and a+1 (§3, §6.1).  Its lifted rank-order edges are
    ξ^{a+1} - ξ^a = -σ_{k_a} e_{k_a}, so this is the sign of their determinant in the basis of
    the directions it spans, in increasing order.  It is the spacetime orientation ε of a top
    simplex and the coarsening sign o(σ, F) of a simplex with ranks 0..p inside its top cell F;
    _self_check compares it with a determinant.
    '''
    steps = [(masks[a + 1] ^ masks[a]).bit_length() - 1 for a in range(len(masks) - 1)]
    return _perm_sign(steps) * (-1) ** sum(1 for s in signs if s > 0)


class _LocalComplex:
    r'''
    The origin-anchored fine simplices of the barycubic complex in dimension D, and every
    operator table built from them.  One instance per D, shared by all lattices (see
    :func:`_local_complex`); nothing here depends on N.  Every array stored here is read-only.
    '''

    def __init__(self, D):
        self.D = D
        self.labels = {}
        self.index = {}
        self.vertices = {}
        for p in range(D + 1):
            labels = sorted(self._enumerate(p),
                            key=lambda L: (L.masks, tuple((s + 1) // 2 for s in L.signs)))
            self.labels[p] = labels
            self.index[p] = {L: j for j, L in enumerate(labels)}
            V = np.array([self._lifted_vertices(L) for L in labels], dtype=np.int64)
            self.vertices[p] = _frozen(V.reshape(len(labels), p + 1, D))
        self._tables = {}
        self._plans = {}

    # --- labels ---------------------------------------------------------------------------

    def _chains(self, length):
        '''All strictly increasing chains of `length` subsets of {0..D-1}, as bitmask tuples.'''
        full = (1 << self.D) - 1

        def extend(chain):
            if len(chain) == length:
                yield tuple(chain)
                return
            rest = full & ~chain[-1]
            sub = rest
            while sub:   # every nonempty subset of the missing directions
                yield from extend(chain + [chain[-1] | sub])
                sub = (sub - 1) & rest

        for S0 in range(1 << self.D):
            yield from extend([S0])

    def _enumerate(self, p):
        for masks in self._chains(p + 1):
            free = len(_directions(masks[-1] & ~masks[0], self.D))
            for signs in product((-1, 1), repeat=free):
                yield LocalLabel(masks, signs)

    def _lifted_vertices(self, label):
        r'''
        Rank-ordered vertices of the simplex anchored at the origin, in UNREDUCED doubled
        coordinates.  The top vertex is 1_{S_p}; a lower vertex ξ^a is displaced from it by σ_k
        along every direction k ∈ S_p \ S_a.  Entries lie in {0, 1, 2}.
        '''
        D = self.D
        masks, signs = label
        sigma = dict(zip(_directions(masks[-1] & ~masks[0], D), signs))
        top = [(masks[-1] >> k) & 1 for k in range(D)]
        verts = []
        for S in masks:
            verts.append(tuple(top[k] + (sigma[k] if (k in sigma and not (S >> k) & 1) else 0)
                               for k in range(D)))
        return verts

    def identify(self, vertices):
        r'''
        Given the rank-ordered vertices of a fine simplex in lifted doubled coordinates (any
        integers, not reduced), return ``(label index, anchor)``.

        The anchor is floor(top / 2) (§2.1); the masks are the odd-coordinate sets; each sign is
        the displacement of the bottom vertex from the top one along a free direction.
        Python's ``>>`` and ``&`` floor correctly for negative coordinates.
        '''
        D = self.D
        top = vertices[-1]
        masks = tuple(sum(1 << k for k in range(D) if v[k] & 1) for v in vertices)
        free = _directions(masks[-1] & ~masks[0], D)
        signs = tuple(int(vertices[0][k] - top[k]) for k in free)
        anchor = tuple(int(t) >> 1 for t in top)
        return self.index[len(vertices) - 1][(masks, signs)], anchor

    @cached_property
    def rank_sets(self):
        return {p: _frozen(np.array([[_popcount(S) for S in L.masks] for L in self.labels[p]],
                                    dtype=np.int64).reshape(-1, p + 1))
                for p in range(self.D + 1)}

    # --- incidence --------------------------------------------------------------------------

    def incidence_table(self, op, degree):
        r'''
        Rows ``(out_label, in_label, offset[D], sign)`` of ``(op f)[out, x] = Σ sign f[in, x+offset]``.

        'd' (input degree p): the face of the output (p+1)-simplex obtained by deleting
        position j, with sign (-1)^j.  Only deleting the top vertex can move the anchor (§2.4),
        so offsets lie in {0,1}^D.  'dT' (input degree p+1) is the transpose of 'd' at degree p:
        swap the labels and negate the offset.  Rows are sorted by output label.
        '''
        key = (op, degree)
        if key in self._tables:
            return self._tables[key]
        D = self.D
        if op == 'd' and _is_degree(degree) and 0 <= degree < D:
            rows = []
            for out, V in enumerate(self.vertices[degree + 1]):
                V = [tuple(v) for v in V]
                for j in range(degree + 2):
                    lab, anchor = self.identify(V[:j] + V[j + 1:])
                    rows.append((out, lab, *anchor, -1 if j % 2 else 1))
            table = np.array(rows, dtype=np.int64).reshape(-1, D + 3)
        elif op == 'dT' and _is_degree(degree) and 0 < degree <= D:
            d = self.incidence_table('d', degree - 1)
            table = np.concatenate([d[:, [1]], d[:, [0]], -d[:, 2:2 + D], d[:, [2 + D]]], axis=1)
            table = table[np.lexsort(table.T[::-1])]
        else:
            raise ValueError(f"no operator table for op={op!r} at degree={degree!r} in D={D}")
        table = self._tables[key] = _frozen(table)
        return table

    def incidence_plan(self, op, degree):
        r'''
        The incidence table in the form the numba kernel reads: the label columns, the signs,
        the distinct offsets, and for each row the index of its offset among them.  The lattice
        turns the offsets into neighbor-site tables once per N (BarycubicLattice._neighbors).
        '''
        key = ('incidence', op, degree)
        if key not in self._plans:
            D = self.D
            table = self.incidence_table(op, degree)
            offsets, row = np.unique(table[:, 2:2 + D], axis=0, return_inverse=True)
            self._plans[key] = {
                'out': _frozen(np.ascontiguousarray(table[:, 0])),
                'in': _frozen(np.ascontiguousarray(table[:, 1])),
                'row': _frozen(row.reshape(-1).astype(np.int64)),
                'sign': _frozen(np.ascontiguousarray(table[:, -1])),
                'offsets': _frozen(offsets),
            }
        return self._plans[key]

    def weighted_incidence_coefficients(self, degree):
        r'''
        Per-row coefficients of δ = M_{p-1}^{-1} dᵀ M_p at input degree p, in the row order of
        incidence_plan('dT', p): sign · M_p(in) / M_{p-1}(out).  Folding both diagonal scalings
        into the rows makes δ a single pass with no weighted copy of the input.
        '''
        key = ('delta', degree)
        if key not in self._plans:
            plan = self.incidence_plan('dT', degree)
            W = self.weights
            coefficient = plan['sign'] * W[degree][plan['in']] / W[degree - 1][plan['out']]
            self._plans[key] = _frozen(coefficient)
        return self._plans[key]

    # --- cup products -----------------------------------------------------------------------

    def cup_table(self, p, q, i):
        r'''
        Rows ``(out_label, a_label, a_offset[D], b_label, b_offset[D], sign)`` of
        ``(a ∪_i b)[out, x] = Σ sign a[a_label, x + a_offset] b[b_label, x + b_offset]``,
        ordered by output label and then by the cut points of :func:`_cup_terms`.
        '''
        _check_cup_degrees(self.D, p, q, i)
        key = ('cup', p, q, i)
        if key in self._tables:
            return self._tables[key]
        D, n = self.D, p + q - i
        terms = _cup_terms(p, q, i)
        rows = []
        for out, V in enumerate(self.vertices[n]):
            V = [tuple(v) for v in V]
            for sign, A, B in terms:
                a_lab, a_anchor = self.identify([V[x] for x in A])
                b_lab, b_anchor = self.identify([V[x] for x in B])
                rows.append((out, a_lab, *a_anchor, b_lab, *b_anchor, sign))
        table = _frozen(np.array(rows, dtype=np.int64).reshape(-1, 2 * D + 4))
        self._tables[key] = table
        return table

    def cup_plan(self, p, q, i):
        r'''
        The cup table in the form the numba contraction kernel reads.  Both factors' offsets lie
        in {0,1}^D, so one list of distinct offsets serves both, and each row names the offset
        rows of its a-face and its b-face.
        '''
        key = ('cup', p, q, i)
        if key not in self._plans:
            D = self.D
            table = self.cup_table(p, q, i)
            both = np.concatenate([table[:, 2:2 + D], table[:, 3 + D:3 + 2 * D]], axis=0)
            offsets, row = np.unique(both, axis=0, return_inverse=True)
            row = row.reshape(-1).astype(np.int64)
            self._plans[key] = {
                'out': _frozen(np.ascontiguousarray(table[:, 0])),
                'a': _frozen(np.ascontiguousarray(table[:, 1])),
                'a_row': _frozen(row[:len(table)].copy()),
                'b': _frozen(np.ascontiguousarray(table[:, 2 + D])),
                'b_row': _frozen(row[len(table):].copy()),
                'sign': _frozen(np.ascontiguousarray(table[:, -1])),
                'offsets': _frozen(offsets),
            }
        return self._plans[key]

    # --- metric -------------------------------------------------------------------------------

    @cached_property
    def weights_by_rank_set(self):
        r'''
        M_p(R) = Vol_{D-p}(⋆σ) / Vol_p(σ) for one representative σ per rank set R (§5.1).

        The representative has S_a = {0, ..., R_a - 1} and all signs +1; the hyperoctahedral
        group times translations maps it to every simplex with the same rank set, so one
        computation per rank set suffices.  Coordinates are the real ones, ξ/2.
        '''
        D = self.D
        tops = self.vertices[D]
        weights = {}
        for p in range(D + 1):
            for R in combinations(range(D + 1), p + 1):
                masks = tuple((1 << r) - 1 for r in R)
                signs = (1,) * (R[-1] - R[0])
                V = self.vertices[p][self.index[p][(masks, signs)]]
                sigma_volume = _simplex_volume(V[1:] - V[0]) / 2 ** p
                weights[R] = _dual_cell_volume(V, R, tops, D) / sigma_volume
        return weights

    @cached_property
    def weights(self):
        '''dict p → float array (f_p,) of M_p per local label.'''
        by_R = self.weights_by_rank_set
        return {p: _frozen(np.array([by_R[tuple(R)] for R in self.rank_sets[p]], dtype=float))
                for p in range(self.D + 1)}

    @cached_property
    def orientation(self):
        r'''ε for each top label, from _orientation_sign; _self_check compares it with det(edges).'''
        return _frozen(np.array([_orientation_sign(*L) for L in self.labels[self.D]], dtype=np.int64))

    def coarsen_matrix(self, p):
        r'''
        Sparse (C(D,p) × f_p) matrix of o(σ, F) (§6.1).  A fine p-simplex lies inside the
        hypercubic p-cell F = (x, I) exactly when its ranks are 0, 1, ..., p and its top type
        is I (its top vertex must be the barycenter of F itself), and then it is anchored at x.
        o compares its rank-order orientation with F's increasing-direction orientation, which
        is _orientation_sign restricted to the directions of I.
        '''
        key = ('coarsen', p)
        if key in self._plans:
            return self._plans[key]
        D = self.D
        components = {c: j for j, c in enumerate(combinations(range(D), p))}
        rows, cols, vals = [], [], []
        for j, (masks, signs) in enumerate(self.labels[p]):
            if [_popcount(S) for S in masks] != list(range(p + 1)):
                continue
            rows.append(components[_directions(masks[-1], D)])
            cols.append(j)
            vals.append(_orientation_sign(masks, signs))
        matrix = scipy.sparse.csr_matrix((np.array(vals, dtype=np.int8), (rows, cols)),
                                         shape=(comb(D, p), len(self.labels[p])))
        self._plans[key] = matrix
        return matrix


def _simplex_volume(edges):
    r'''
    k-volume of the simplex spanned by the k edge vectors (rows), batched over leading axes:
    sqrt(det Gram) / k!, with the 0-simplex (a point) given volume 1.
    '''
    edges = np.asarray(edges, dtype=float)
    k = edges.shape[-2]
    if k == 0:
        return np.ones(edges.shape[:-2])
    gram = edges @ np.swapaxes(edges, -1, -2)
    return np.sqrt(np.abs(np.linalg.det(gram))) / factorial(k)


def _dual_cell_volume(V, R, tops, D):
    r'''
    Euclidean (D-p)-volume of the barycentric dual cell of the p-simplex with lifted doubled
    vertices V and rank set R.

    The dual cell is the union, over maximal chains σ = τ_0 ⊊ τ_1 ⊊ ... ⊊ τ_{D-p} of fine
    simplices, of the geometric simplices [c(τ_0) ... c(τ_{D-p})] on the centroids.  A maximal
    chain is a top simplex T ⊇ σ together with an order in which T's remaining D - p vertices
    are added.  The pieces are not coplanar in general, so the volume is the SUM of the piece
    volumes, not the volume of a hull.
    '''
    p = len(R) - 1
    if p == D:
        return 1.0
    # A top simplex anchored at y has vertices in 2y + {0, 1, 2}^D, so only anchors y with
    # 2y <= v <= 2y + 2 for every vertex v of σ can contain σ.
    lo = np.ceil((V.max(axis=0) - 2) / 2).astype(int)
    hi = np.floor(V.min(axis=0) / 2).astype(int)
    extra_positions = [r for r in range(D + 1) if r not in R]
    extras = []
    for y in product(*(range(l, h + 1) for l, h in zip(lo, hi))):
        T = tops + 2 * np.array(y)
        # Vertices of a top simplex sit at the positions of their ranks, so σ ⊆ T exactly when
        # T agrees with σ at the positions R.
        inside = (T[:, list(R), :] == V[None]).all(axis=(1, 2))
        extras.append(T[inside][:, extra_positions, :])
    extras = np.concatenate(extras, axis=0).astype(float) / 2        # real coordinates
    base = V.astype(float).sum(axis=0) / 2
    volume = 0.0
    for order in permutations(range(D - p)):
        partial = np.cumsum(extras[:, list(order), :], axis=1) + base  # vertex sums of τ_1..τ_{D-p}
        centroids = partial / np.arange(p + 2, D + 2)[None, :, None]
        edges = centroids - base / (p + 1)
        volume += _simplex_volume(edges).sum()
    return volume


@lru_cache(maxsize=None)
def _local_complex(D):
    '''The shared, lazily filled :class:`_LocalComplex` of dimension D.'''
    return _LocalComplex(D)


_HASH_MODULUS = 2 ** 31 - 1
_HASH_BASE = 1000003


def _canonical_hash_values(labels, seed, tag):
    r'''
    The reproducible cochain values of BarycubicLattice.export_cup: a polynomial hash of each
    canonical label's coordinates mod the prime 2^31 - 1, one squaring for mixing, then mod 7
    shifted to -3..3.  Kept in int64, where every intermediate is exact (< 2^62).
    '''
    P, B = _HASH_MODULUS, _HASH_BASE
    coordinates = np.asarray(labels, dtype=np.int64).reshape(len(labels), -1)
    h = np.full(len(labels), int(seed) % P, dtype=np.int64)
    h = (h * B + int(tag)) % P
    for m in range(coordinates.shape[1]):
        h = (h * B + coordinates[:, m] + 1) % P
    h = (h * h + B) % P
    return h % 7 - 3


@lru_cache(maxsize=256)
def _pullback_map(D, p, perm, flips, half):
    r'''
    Label map of the doubled-coordinate map g(ξ) = L ξ + t on origin-anchored p-simplices,
    where (L ξ)_k = flips_k ξ_{perm[k]} and t = (1, ..., 1) if ``half`` else 0.

    g sends the simplex with label j at site x to the simplex with lifted vertices
    L V_j + t + 2 L x.  Identifying L V_j + t (an origin-anchored lift, possibly with negative
    entries) gives a label image[j] and an anchor offset offset[j], so
    g σ(j, x) = σ(image[j], L x + offset[j]).  The half shift reverses rank order, so for it the
    reversed vertex list is identified.  Integer translations are left out of the key: they only
    add to the offsets, and transform applies them afterwards.  Returns read-only arrays.
    '''
    local = _local_complex(D)
    perm = list(perm)
    flip = np.asarray(flips, dtype=np.int64)
    t = np.ones(D, dtype=np.int64) if half else np.zeros(D, dtype=np.int64)
    images, offsets = [], []
    for V in local.vertices[p]:
        W = flip[None, :] * V[:, perm] + t[None, :]
        W = [tuple(int(w) for w in v) for v in (W[::-1] if half else W)]
        label, anchor = local.identify(W)
        images.append(label)
        offsets.append(anchor)
    return (_frozen(np.array(images, dtype=np.int64)),
            _frozen(np.array(offsets, dtype=np.int64).reshape(-1, D)))


def _self_check(D):
    r'''
    Developer self-test of the per-D tables against independent closed forms; raises
    AssertionError on failure.  Not part of the API: the test suite checks the same facts.
    '''
    local = _local_complex(D)
    for p in range(D + 1):
        expected = sum((-1) ** (p - j) * comb(p, j) * (2 * j + 2) ** D for j in range(p + 1))
        assert len(local.labels[p]) == expected, (D, p)
    W = local.weights_by_rank_set
    for r in range(D + 1):
        assert np.isclose(W[(r,)], 1 / ((D + 1) * comb(D, r)), rtol=1e-13, atol=0), (D, r)
    for R, w in W.items():
        assert np.isclose(w, W[tuple(sorted(D - r for r in R))], rtol=1e-12, atol=0), (D, R)
    assert np.isclose(W[tuple(range(D + 1))], 2 ** D * factorial(D), rtol=1e-13), D
    tops = local.vertices[D]
    edges = tops[:, 1:] - tops[:, :-1]
    assert (np.rint(np.linalg.det(edges.astype(float))).astype(int) == local.orientation).all()
    return True


# ---------------------------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------------------------

class BarycubicLattice(ReadWriteable):
    r'''
    The barycentric subdivision of the periodic hypercubic lattice $(\mathbb{Z}/N)^D$.

    Every cell $F$ of the hypercubic lattice (site, link, plaquette, ..., hypercube) contributes one
    fine vertex $b(F)$ at its barycenter, and every flag of cells
    $F_0 \subsetneq F_1 \subsetneq \cdots \subsetneq F_p$ contributes one fine p-simplex with
    vertices $b(F_0), \ldots, b(F_p)$.  The **rank** of $b(F)$ is $\dim F$.  The vertices of a fine
    simplex have distinct ranks, so ordering them by rank orders every simplex, compatibly with
    its faces; that local order is what the Alexander–Whitney and Steenrod products need, and it
    is why this lattice supports $\cup_i$ products that the hypercubic lattice does not.

    **Doubled coordinates.**  Twice a barycenter is an integer vector
    $\xi \in (\mathbb{Z}/2N)^D$.  Its odd entries are the directions the cell spans (the cell's
    *type* $T(\xi)$), its rank is the number of odd entries, and its *anchor site* is
    $\lfloor \xi/2 \rfloor$.  The fine vertices are exactly the doubled lattice, $2^D$ per site.

    **Labels.**  A fine p-simplex anchored at site $x$ has rank-ordered vertices
    $\xi^0, \ldots, \xi^p$ with strictly nested types $S_0 \subsetneq \cdots \subsetneq S_p$.  Its
    top vertex is $\xi^p = 2x + 1_{S_p}$, and each lower vertex is displaced from it by one sign
    $\sigma_k = \pm 1$ along every direction $k \in S_p \setminus S_a$ it does not span:

    .. math::

        \xi^a_k = \begin{cases} \xi^p_k + \sigma_k & k \in S_p \setminus S_a \\ \xi^p_k & \text{otherwise.} \end{cases}

    The :class:`LocalLabel` $(S_0, \ldots, S_p; \sigma)$ fixes the simplex up to translation.  The
    number of labels per site is

    .. math::

        f_p(D) = \sum_{S_0 \subsetneq \cdots \subsetneq S_p} 2^{|S_p \setminus S_0|}
               = \sum_{j=0}^{p} (-1)^{p-j} \binom{p}{j} (2j+2)^D,

    so that for example $D = 4$ has $(f_0, \ldots, f_4) = (16, 240, 800, 960, 384)$.  Labels are
    ordered lexicographically by $(\text{mask}(S_0), \ldots, \text{mask}(S_p), \text{bits}(\sigma))$
    with $\text{mask}(S) = \sum_{k \in S} 2^k$ and bits $0$ for $\sigma_k = -1$, $1$ for $+1$.

    A p-cochain is stored as a :class:`BarycubicForm` of shape $(f_p, N, \ldots, N)$, indexed
    ``[label, x]``, just as a hypercubic :class:`~supervillain.lattice.Form` is indexed
    ``[component, x]``.

    All label combinatorics, operator tables, weights, and orientations depend only on $D$; they
    are computed once per $D$ and shared by every lattice of that dimension.

    Parameters
    ----------
    D : int
        Spacetime dimension, $D \geq 1$.
    N : int
        Sites per direction of the underlying hypercubic lattice, $N \geq 2$.  Both parities are
        allowed; $N \geq 2$ guarantees that the subdivision is a genuine simplicial complex after
        the periodic identification.

    Raises
    ------
    TypeError
        If ``D`` or ``N`` is not an integer (a float is rejected even when integral).
    ValueError
        If $D < 1$ or $N < 2$.

    Attributes
    ----------
    D : int
        Spacetime dimension.
    N : int
        Sites per direction.
    labels : dict[int, list[LocalLabel]]
        ``labels[p]`` is the ordered list of the $f_p$ local labels of fine p-simplices; the
        position in this list is the index along axis 0 of a p-form.
    label_index : dict[int, dict[LocalLabel, int]]
        Reverse map of :attr:`labels`.  Plain ``(masks, signs)`` tuples work as keys too.
    per_site : dict[int, int]
        $f_p$, the number of fine p-simplices per anchor site.
    '''

    def __init__(self, D, N):
        # Check the types before converting: int() would silently truncate 4.5 to 4.
        for name, value in (('D', D), ('N', N)):
            if not _is_degree(value):
                raise TypeError(f"BarycubicLattice needs an integer {name}, got {value!r}")
        D, N = int(D), int(N)
        if D < 1:
            raise ValueError(f"BarycubicLattice needs D >= 1, got D={D}")
        if N < 2:
            raise ValueError(f"BarycubicLattice needs N >= 2, got N={N}")
        self.D = D
        self.N = N
        self._local = _local_complex(D)
        # Per-lattice shallow copies, so that editing them cannot corrupt the shared tables.
        self.labels = {p: list(self._local.labels[p]) for p in range(D + 1)}
        self.label_index = {p: dict(self._local.index[p]) for p in range(D + 1)}
        self.per_site = {p: len(self.labels[p]) for p in range(D + 1)}
        # Neighbor-site tables for the numba kernels, keyed by the offsets they encode.
        self._neighbor_cache = {}

    def __reduce__(self):
        # Pickle as (D, N) only.  Batch stores its lattice by pickling, and the cached tables
        # would otherwise be dragged along; they are rebuilt (or found in the cache) on load.
        return (type(self), (self.D, self.N))

    def __eq__(self, other):
        return isinstance(other, BarycubicLattice) and (self.D, self.N) == (other.D, other.N)

    def __hash__(self):
        return hash((BarycubicLattice, self.D, self.N))

    def __repr__(self):
        return f"BarycubicLattice(D={self.D}, N={self.N})"

    # --- sizes, mirroring Lattice ----------------------------------------------------------

    @cached_property
    def sites(self):
        r'''Number of anchor sites $N^D$.'''
        return self.N ** self.D

    @cached_property
    def dims(self):
        '''Spatial shape ``(N,) * D``, the trailing axes of every form.'''
        return (self.N,) * self.D

    @property
    def dim(self):
        '''Number of dimensions (alias for ``D``).'''
        return self.D

    @cached_property
    def cells_of_degree(self):
        r'''dict[int, int]: p → total number of fine p-simplices, $f_p N^D$.'''
        return {p: self.per_site[p] * self.sites for p in range(self.D + 1)}

    @cached_property
    def _hypercubic(self):
        # The underlying hypercubic lattice, target of coarsen.  Imported late because
        # supervillain.lattice imports this module.
        from supervillain.lattice.compact import Lattice
        return Lattice(self.D, self.N)

    def _neighbors(self, offsets):
        r'''
        Neighbor-site table for the kernels: row k holds, for every flattened site x, the
        flattened site x + offsets[k] (periodic).  Built once per lattice and set of offsets;
        the offsets come from the per-D plans, so there are few distinct sets.
        '''
        key = (offsets.shape, offsets.tobytes())
        table = self._neighbor_cache.get(key)
        if table is None:
            sites = np.indices(self.dims).reshape(self.D, -1)                     # (D, S)
            table = np.stack([np.ravel_multi_index(tuple((sites + o[:, None]) % self.N), self.dims)
                              for o in offsets]).astype(np.int64)
            table = self._neighbor_cache[key] = _frozen(table)
        return table

    # --- combinatorics -----------------------------------------------------------------------

    @property
    def rank_sets(self):
        r'''
        The ranks of the vertices of every local label.

        Returns
        -------
        dict[int, numpy.ndarray]
            ``rank_sets[p]`` is a read-only int array of shape ``(f_p, p+1)`` whose row ``j`` is
            $(\lvert S_0\rvert, \ldots, \lvert S_p\rvert)$ for label ``j``: strictly
            increasing, and the rank-order positions of the simplex's vertices.
        '''
        return {p: _view(self._local.rank_sets[p]) for p in range(self.D + 1)}

    def local_vertices(self, p):
        r'''
        Rank-ordered vertices of every p-label anchored at the origin, in lifted doubled
        coordinates.

        The coordinates are the *common lift*: the unreduced integers
        $\xi^p = 1_{S_p}$ and $\xi^a = \xi^p + \sum_{k \in S_p \setminus S_a} \sigma_k e_k$, with
        entries in $\{0, 1, 2\}$.  Differences of these give true displacements (divide by 2 for
        lattice units), which reduced coordinates do not across the periodic seam.  The
        simplex with the same label anchored at $x$ has vertices $2x + \xi^a$.

        Parameters
        ----------
        p : int
            Simplex dimension, $0 \leq p \leq D$.

        Returns
        -------
        numpy.ndarray
            Read-only int array of shape ``(f_p, p+1, D)``.

        Raises
        ------
        ValueError
            If ``p`` is not an integer in $0, \ldots, D$.
        '''
        return _view(self._local.vertices[_check_degree(self.D, p)])

    def canonical_labels(self, p):
        r'''
        Global canonical labels of all fine p-simplices, for comparison between implementations.

        A simplex's canonical label is its rank-ordered list of vertices in doubled coordinates
        reduced to $[0, 2N)$.  Because $N \geq 2$ makes the subdivision a simplicial complex,
        this list determines the simplex, independently of any internal layout.

        Parameters
        ----------
        p : int
            Simplex dimension, $0 \leq p \leq D$.

        Returns
        -------
        numpy.ndarray
            Int array of shape ``(f_p * N**D, p+1, D)`` in the C order of a flattened
            ``(f_p, N, ..., N)`` form, so row ``j * N**D + ravel(x)`` is label ``j`` at site ``x``.

        Raises
        ------
        ValueError
            If ``p`` is not an integer in $0, \ldots, D$.
        '''
        p = _check_degree(self.D, p)
        sites = np.indices(self.dims).reshape(self.D, -1).T                   # (S, D), C order
        V = self._local.vertices[p]
        return ((V[:, None, :, :] + 2 * sites[None, :, None, :]) % (2 * self.N)).reshape(
            -1, p + 1, self.D)

    # --- operator tables ----------------------------------------------------------------------

    def operator_table(self, op, degree):
        r'''
        Incidence table of the coboundary or its transpose, generalizing the hypercubic
        ``(out, in, axis, sign)`` tables.

        Each row ``(out_label, in_label, offset_0, ..., offset_{D-1}, sign)`` contributes

        .. math::

            (\mathrm{op}\, f)[\text{out}, x] \mathrel{+}= \text{sign} \cdot f[\text{in}, x + \text{offset}].

        For ``'d'`` the input simplex is a face of the output simplex; deleting the top vertex is
        the only way to move the anchor, so offsets lie in $\{0,1\}^D$.  ``'dT'`` is the
        transpose (the signed coface sum), with offsets in $\{0,-1\}^D$.  Rows are sorted by
        output label.  The tables depend only on $D$ and are shared; they are read-only.

        Parameters
        ----------
        op : str
            ``'d'`` or ``'dT'``.
        degree : int
            Input degree: $0 \leq p < D$ for ``'d'``, $0 < p \leq D$ for ``'dT'``.

        Returns
        -------
        numpy.ndarray
            Read-only int64 array of shape ``(rows, D + 3)``.

        Raises
        ------
        ValueError
            If ``op`` is unknown or ``degree`` is out of range for it.
        '''
        return _view(self._local.incidence_table(op, degree))

    def cup_table(self, p, q, i=0):
        r'''
        Contraction table of the Steenrod product $\cup_i$ of a p-form and a q-form.

        Each row ``(out_label, a_label, a_offset[D], b_label, b_offset[D], sign)`` contributes

        .. math::

            (a \cup_i b)[\text{out}, x] \mathrel{+}= \text{sign} \cdot a[\text{a\_label}, x + \text{a\_offset}]\; b[\text{b\_label}, x + \text{b\_offset}],

        one row per interval-cut term of :func:`cup` on each output label, with the sign of the
        closed-form rule documented there.  Offsets lie in $\{0,1\}^D$.  Rows are ordered by
        output label, then by the cut points $J$ in lexicographic order.

        Parameters
        ----------
        p, q : int
            Degrees of the two factors.
        i : int
            Which product, $0 \leq i \leq \min(p, q)$ with $p + q - i \leq D$.

        Returns
        -------
        numpy.ndarray
            Read-only int64 array of shape ``(rows, 2D + 4)``.

        Raises
        ------
        ValueError
            If $(p, q, i)$ is not a legal product in dimension $D$.
        '''
        return _view(self._local.cup_table(p, q, i))

    # --- metric ------------------------------------------------------------------------------

    @cached_property
    def hodge_weight_by_rank_set(self):
        r'''
        The diagonal Hodge-star weights, one per rank set.

        For a fine p-simplex $\sigma$ the weight is the ratio of the Euclidean volume of its
        barycentric dual cell to its own volume,

        .. math::

            M_p(\sigma) = \frac{\mathrm{Vol}_{D-p}(\star\sigma)}{\mathrm{Vol}_p(\sigma)},
            \qquad \mathrm{Vol}_k = \frac{\sqrt{\det \mathrm{Gram}}}{k!},\quad \mathrm{Vol}_0 = 1.

        The dual cell $\star\sigma$ is the union, over all maximal chains of fine simplices
        $\sigma = \tau_0 \subsetneq \tau_1 \subsetneq \cdots \subsetneq \tau_{D-p}$, of the
        simplices spanned by their centroids; its volume is the sum of those pieces, which are not
        coplanar in general.  The hyperoctahedral group and translations act transitively on
        simplices with a given rank set $R(\sigma) = \{\lvert S_0\rvert, \ldots, \lvert S_p\rvert\}$, so $M_p$ depends on
        $\sigma$ only through $R$.  The half-diagonal translation (:func:`half_shift`) is an
        isometry mapping $R$ to $D - R$, so $M(R) = M(D-R)$.

        The computed values reproduce several closed forms, which the tests check.  The vertex
        weights are $M_0(r) = 1/((D+1)\binom{D}{r})$, so the $\binom{D}{r}$ rank-$r$ vertices of
        one site have dual cells of total volume $1/(D+1)$, and all of them together tile its unit
        cube.  The top weight is $M_D = 1/\mathrm{Vol}_D = 2^D D!$, since $2^D D!$ fine simplices
        tile each unit hypercube.  In units of the hypercubic lattice
        spacing, the $D=4$ link weights by rank class $(r, s)$ are $\sqrt5/10$ for $(0,1)$ and
        $(3,4)$, $\sqrt6/30$ for $(1,2)$ and $(2,3)$, $\sqrt6/60$ for $(0,2)$ and $(2,4)$,
        $1/40$ for $(0,3)$ and $(1,4)$, $\sqrt2/40$ for $(0,4)$, and $\sqrt2/60$ for $(1,3)$.

        Returns
        -------
        dict[tuple[int, ...], float]
            Maps each nonempty strictly increasing rank tuple $R \subseteq \{0, \ldots, D\}$ to
            $M_{\lvert R\rvert-1}(R)$: $2^{D+1} - 1$ entries.
        '''
        return dict(self._local.weights_by_rank_set)

    def hodge_weights(self, p):
        r'''
        The Hodge-star weight $M_p$ of every local p-label.

        See :attr:`hodge_weight_by_rank_set` for the definition.

        Parameters
        ----------
        p : int
            Simplex dimension, $0 \leq p \leq D$.

        Returns
        -------
        numpy.ndarray
            Read-only float array of shape ``(f_p,)``; broadcast it against the spatial axes
            with ``weights[(slice(None),) + (None,) * D]``.

        Raises
        ------
        ValueError
            If ``p`` is not an integer in $0, \ldots, D$.
        '''
        return _view(self._local.weights[_check_degree(self.D, p)])

    @cached_property
    def _orientation_owner(self):
        eps = np.broadcast_to(self._local.orientation[(slice(None),) + (None,) * self.D],
                              (self.per_site[self.D],) + self.dims)
        return _frozen(BarycubicForm(eps.copy(), degree=self.D, lattice=self))

    @property
    def orientation(self):
        r'''
        The orientation of every top simplex relative to the standard orientation of $\mathbb{R}^D$.

        For a D-simplex whose type grows by direction $k_a$ between ranks $a$ and $a+1$, the
        rank-order edge vectors are $\xi^{a+1} - \xi^a = -\sigma_{k_a} e_{k_a}$ in the common lift,
        so the sign of their determinant is

        .. math::

            \varepsilon_\sigma = \mathrm{sgn}(k_0, \ldots, k_{D-1}) \prod_{k} (-\sigma_k).

        The fundamental cycle $[M] = \sum_\sigma \varepsilon_\sigma \sigma$ has no boundary, and
        :func:`integrate` pairs top forms with it.

        Returns
        -------
        BarycubicForm
            Read-only integer top form (degree $D$) with entries $\pm 1$.  Arithmetic on it
            returns ordinary writeable forms.
        '''
        return _view(self._orientation_owner)

    # --- coloring ----------------------------------------------------------------------------

    @cached_property
    def _checkerboarding_owner(self):
        rank = np.broadcast_to(self._local.rank_sets[0][:, 0][(slice(None),) + (None,) * self.D],
                               (self.per_site[0],) + self.dims)
        return tuple(tuple(_frozen(axis) for axis in np.where(rank == r))
                     for r in range(self.D + 1))

    @property
    def checkerboarding(self):
        r'''
        Partition the fine vertices into colors with no same-color fine link.

        The rank is a proper coloring of the fine 1-skeleton: every fine link joins vertices of
        different ranks.  With $D+1$ colors it is optimal, since every top simplex is a
        $(D+1)$-clique, and it works for every $N$, even or odd.

        .. note::
            This differs intentionally from :attr:`Lattice.checkerboarding
            <supervillain.lattice.Lattice.checkerboarding>`, whose colors select sites from the
            spatial axes only.  Here the rank depends on the label axis and every site carries
            vertices of every rank, so each color selects over all $D+1$ axes of a 0-form::

                for r, color in enumerate(L.checkerboarding):
                    form[color] = r

        Returns
        -------
        tuple
            $D+1$ index tuples, one per rank $r = 0, \ldots, D$; each is a tuple of $D+1$
            read-only integer arrays ``(labels, x_0, ..., x_{D-1})`` as returned by
            ``numpy.where``.
        '''
        return tuple(tuple(_view(axis) for axis in color) for color in self._checkerboarding_owner)

    # --- factories, mirroring Lattice --------------------------------------------------------

    def zeros(self, p, dtype=float):
        r'''
        Return a zero p-form.

        Parameters
        ----------
        p : int
            Form degree, $0 \leq p \leq D$.
        dtype : data-type, optional
            NumPy dtype of the underlying array (default ``float``).

        Returns
        -------
        BarycubicForm
            Zero p-form of shape ``(f_p, N, ..., N)``.

        Raises
        ------
        ValueError
            If ``p`` is not an integer in $0, \ldots, D$.
        '''
        p = _check_degree(self.D, p)
        return BarycubicForm(np.zeros((self.per_site[p],) + self.dims, dtype=dtype),
                             degree=p, lattice=self)

    def random(self, p):
        r'''
        Return a p-form with entries drawn uniformly from $[0, 1)$ with ``numpy.random.random``.

        Parameters
        ----------
        p : int
            Form degree, $0 \leq p \leq D$.

        Returns
        -------
        BarycubicForm
            Random p-form of shape ``(f_p, N, ..., N)``.

        Raises
        ------
        ValueError
            If ``p`` is not an integer in $0, \ldots, D$.
        '''
        p = _check_degree(self.D, p)
        return BarycubicForm(np.random.random((self.per_site[p],) + self.dims),
                             degree=p, lattice=self)

    def form(self, p, dtype=float):
        r'''
        Return a zero p-form.  Alias for :meth:`zeros`.

        Parameters
        ----------
        p : int
            Form degree, $0 \leq p \leq D$.
        dtype : data-type, optional
            NumPy dtype (default ``float``).

        Returns
        -------
        BarycubicForm
            Zero p-form of shape ``(f_p, N, ..., N)``.

        Raises
        ------
        ValueError
            If ``p`` is not an integer in $0, \ldots, D$.
        '''
        return self.zeros(p, dtype=dtype)

    # --- interop -----------------------------------------------------------------------------

    def export_complex(self, p):
        r'''
        Export the degree-p data of the complex in a layout-independent form.

        Two implementations of the barycubic lattice can be compared exactly by matching
        simplices through their canonical labels (rank-ordered vertex lists in reduced doubled
        coordinates), regardless of how each stores its cochains.

        Parameters
        ----------
        p : int
            Simplex dimension, $0 \leq p \leq D$.

        Returns
        -------
        dict
            ``'labels'``
                :meth:`canonical_labels` ``(p)``, shape ``(f_p N^D, p+1, D)``.
            ``'d'``
                The coboundary from degree p to p+1 as a ``scipy.sparse.csr_matrix`` of
                shape ``(f_{p+1} N^D, f_p N^D)`` with entries $\pm 1$, rows and columns in the
                canonical order of ``labels``.  Absent for $p = D$.
            ``'weights'``
                $M_p$ of every simplex, shape ``(f_p N^D,)``.
            ``'orientation'``
                $\varepsilon$ of every simplex, shape ``(f_D N^D,)``.  Only for $p = D$.

        Raises
        ------
        ValueError
            If ``p`` is not an integer in $0, \ldots, D$.
        '''
        p = _check_degree(self.D, p)
        D, S = self.D, self.sites
        out = {
            'labels': self.canonical_labels(p),
            'weights': np.repeat(self._local.weights[p], S),
        }
        if p < D:
            table = self.operator_table('d', p)
            sites = np.indices(self.dims).reshape(D, -1).T                      # (S, D)
            shifted = (sites[None, :, :] + table[:, None, 2:2 + D]) % self.N      # (rows, S, D)
            rows = table[:, [0]] * S + np.arange(S)[None, :]
            cols = table[:, [1]] * S + np.ravel_multi_index(
                tuple(np.moveaxis(shifted, -1, 0)), self.dims)
            vals = np.broadcast_to(table[:, [-1]], rows.shape)
            out['d'] = scipy.sparse.csr_matrix(
                (vals.ravel(), (rows.ravel(), cols.ravel())),
                shape=(self.cells_of_degree[p + 1], self.cells_of_degree[p]))
        if p == D:
            out['orientation'] = np.repeat(self._local.orientation, S)
        return out

    def export_cup(self, p, q, i=0, seed=0):
        r'''
        Export one Steenrod product on reproducible integer cochains, for comparison between
        implementations.

        The factors are built from canonical labels alone, so that any implementation can
        rebuild exactly the same cochains without sharing code.  For a simplex with canonical
        label $(\xi^0, \ldots, \xi^p)$ (see :meth:`canonical_labels`), list its $(p+1)D$
        coordinates vertex by vertex, $c_1, c_2, \ldots$, and compute in integer arithmetic
        with $P = 2^{31} - 1$ and $B = 1000003$:

        .. math::

            h \leftarrow \text{seed} \bmod P, \qquad
            h \leftarrow (h B + \text{tag}) \bmod P, \qquad
            h \leftarrow (h B + c_m + 1) \bmod P \quad \text{for } m = 1, 2, \ldots,

        then one final mixing step $h \leftarrow (h^2 + B) \bmod P$.  The cochain value is
        $(h \bmod 7) - 3 \in \{-3, \ldots, 3\}$, with $\text{tag} = 0$ for $a$ and
        $\text{tag} = 1$ for $b$.  Every intermediate stays below $2^{62}$, so signed 64-bit
        arithmetic is exact; $\bmod$ is the nonnegative remainder.

        The scripts ``docs/barycubic-lattice/write_export.py`` and
        ``docs/barycubic-lattice/compare_exports.py`` write this export (with
        :meth:`export_complex`) for every legal product and diff two such exports.

        Parameters
        ----------
        p, q : int
            Degrees of the two factors.
        i : int
            Which product, $0 \leq i \leq \min(p, q)$ with $p + q - i \leq D$.
        seed : int
            Selects the cochains.

        Returns
        -------
        dict
            ``'labels'``
                :meth:`canonical_labels` ``(p + q - i)``, the simplices of the product.
            ``'a'``, ``'b'``
                The two factors, int64 of shapes ``(f_p N^D,)`` and ``(f_q N^D,)``, in the order
                of :meth:`canonical_labels` ``(p)`` and ``(q)``.
            ``'values'``
                $a \cup_i b$ in the order of ``'labels'``, int64.

        Raises
        ------
        ValueError
            If $(p, q, i)$ is not a legal product in dimension $D$.
        '''
        _check_cup_degrees(self.D, p, q, i)
        a = _canonical_hash_values(self.canonical_labels(p), seed, 0)
        b = _canonical_hash_values(self.canonical_labels(q), seed, 1)
        a_form = BarycubicForm(a.reshape(self.per_site[p], *self.dims), degree=p, lattice=self)
        b_form = BarycubicForm(b.reshape(self.per_site[q], *self.dims), degree=q, lattice=self)
        return {
            'labels': self.canonical_labels(p + q - i),
            'a': a,
            'b': b,
            'values': np.asarray(cup(a_form, b_form, i)).reshape(-1),
        }

    # --- I/O ---------------------------------------------------------------------------------

    def to_h5(self, group, _top=True):
        '''Write ``D`` and ``N`` to HDF5; everything else is recomputed on load.'''
        from supervillain.h5 import Data
        Data.write(group, 'D', self.D)
        Data.write(group, 'N', self.N)

    @classmethod
    def from_h5(cls, group, strict=True, _top=True):
        '''Reconstruct from HDF5 by reading ``D`` and ``N`` and rebuilding the rest.'''
        from supervillain.h5 import Data
        return cls(D=Data.read(group['D'], strict), N=Data.read(group['N'], strict))


# ---------------------------------------------------------------------------------------------
# Forms
# ---------------------------------------------------------------------------------------------

class BarycubicForm(np.ndarray):
    r'''
    A p-cochain on a :class:`BarycubicLattice`.

    The value on the fine p-simplex with local label ``j`` anchored at site ``x`` is stored at
    ``form[j, x_0, ..., x_{D-1}]``, so the shape is $(f_p, N, \ldots, N)$.  The stored value is the
    cochain evaluated on the simplex with its vertices in **rank order**,
    $\omega[\sigma] = \omega(\xi^0 \ldots \xi^p)$; any other vertex order differs by the sign of
    the permutation.

    Semantics mirror :class:`~supervillain.lattice.Form`.  A form knows its ``degree`` and its
    ``lattice``.  Elementwise operations (``±``, ``*``, ``/``, unary ``−``, ``abs``, ``**``,
    ``np.sqrt``, ``np.isclose``, ``==``, and so on) return a form of the same kind when all form
    operands share both degree and kind (primal or :class:`dual <BarycubicDualForm>`).  Mixed
    degrees or kinds return a plain ndarray, because the degree of the result would be ambiguous.
    Reductions (``sum``, ``max``, ...) return plain numpy scalars or arrays.

    Parameters
    ----------
    input_array : array_like
        Data of shape ``(f_p, N, ..., N)``.
    degree : int
        The form degree $p$.
    lattice : BarycubicLattice
        The lattice the form lives on.
    dtype : data-type, optional
        Passed to ``numpy.asarray``.

    Raises
    ------
    ValueError
        If the shape of ``input_array`` is not ``spatial_shape(degree=degree, lattice=lattice)``.
    '''

    __batch_tag__ = 'BarycubicForm'

    dual = False
    '''Whether the form lives on dual cells; primal cochains have ``dual = False``, and :class:`BarycubicDualForm` sets it to ``True``.'''

    @classmethod
    def spatial_shape(cls, *, degree, lattice):
        r'''
        Parameters
        ----------
        degree : int
            Form degree $p$.
        lattice : BarycubicLattice
            The lattice.

        Returns
        -------
        tuple
            ``(f_p, N, ..., N)``, the shape of a p-form on ``lattice``.

        Raises
        ------
        ValueError
            If ``degree`` is not an integer in $0, \ldots, D$.
        '''
        degree = _check_degree(lattice.D, degree)
        return (lattice.per_site[degree],) + (lattice.N,) * lattice.D

    def __new__(cls, input_array, *, degree, lattice, dtype=None):
        expected = cls.spatial_shape(degree=degree, lattice=lattice)
        obj = np.asarray(input_array, dtype=dtype).view(cls)
        if obj.shape != expected:
            raise ValueError(f"a {cls.__name__} of degree {degree} on {lattice} has shape "
                             f"{expected}, got {obj.shape}")
        obj.degree = int(degree)
        obj.lattice = lattice
        return obj

    def _is_whole(self):
        '''Whether this array is a whole form, not a slice, reshape, or fancy-indexed piece.'''
        if self.lattice is None or self.degree is None:
            return False
        try:
            return self.shape == type(self).spatial_shape(degree=self.degree, lattice=self.lattice)
        except ValueError:
            return False

    def __reduce__(self):
        # ndarray's own pickling would drop degree and lattice.  A whole form is rebuilt through
        # its constructor (the lattice pickles as (D, N)); a slice or other piece of a form is
        # not a form, and pickles as the plain array it is.
        data = np.array(self, copy=True, subok=False)
        if not self._is_whole():
            return data.__reduce__()
        return (_rebuild_form, (data, self.degree, self.lattice, self.dual))

    def __reduce_ex__(self, protocol):
        # numpy implements __reduce_ex__ itself (for out-of-band buffers under protocol 5),
        # which would bypass __reduce__; route every protocol through __reduce__.
        return self.__reduce__()

    def __array_finalize__(self, obj):
        # obj is None for an explicit constructor; otherwise it is the array being viewed.
        if obj is None:
            return
        self.degree = getattr(obj, 'degree', None)
        self.lattice = getattr(obj, 'lattice', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        forms = [x for x in inputs if isinstance(x, BarycubicForm)]
        # A primal p-form and a dual form of numerical degree p are different objects (the dual
        # one lives on dual cells), so both the kind and the degree must agree to stay a form.
        kinds = {(x.dual, x.degree) for x in forms}

        raw = tuple(np.asarray(x) for x in inputs)
        out = kwargs.get('out')
        if out is not None:
            kwargs['out'] = tuple(np.asarray(o) for o in out)

        result = getattr(ufunc, method)(*raw, **kwargs)

        # Only a full-shape result is a form: slices and fancy-indexed pieces of a form are still
        # instances of the class (numpy views keep the subclass) but not cochains.
        if len(kinds) == 1 and isinstance(result, np.ndarray):
            first = forms[0]
            if first._is_whole() and result.shape == first.shape:
                return type(first)(result, degree=first.degree, lattice=first.lattice)
        return result

    def __repr__(self):
        return (f"{type(self).__name__}(degree={self.degree}, shape={self.shape}, "
                f"lattice={self.lattice})")


class BarycubicDualForm(BarycubicForm):
    r'''
    The image of :func:`star`: a $(D-p)$-cochain on the barycentric dual cells.

    The dual cell $\star\sigma$ of a fine p-simplex $\sigma$ is a $(D-p)$-cell, and dual cells
    are in one-to-one correspondence with primal simplices.  A dual form is therefore stored on
    the primal p-labels, with shape $(f_p, N, \ldots, N)$ — the same as a primal p-form — while
    its ``degree`` is the dual degree $D - p$.  ``primal_degree`` is $p$.

    It supports elementwise arithmetic with dual forms of the same degree, :func:`inverse_star`,
    and pairing with a primal p-form by elementwise product and sum,

    .. math::

        \sum_\sigma \overline{a_\sigma}\, (\star b)_{\star\sigma} = \langle a, b \rangle,

    i.e. ``(a.conj() * star(b)).sum() == inner(a, b)`` (for real forms the conjugate does
    nothing).  There is no coboundary or product on dual
    forms (no dual complex is built), and the operators :func:`d`, :func:`delta`,
    :func:`laplacian`, :func:`star`, :func:`inner`, :func:`cup`, :func:`integrate`,
    :func:`coarsen`, :func:`transform`, and :func:`half_shift` raise ``TypeError`` on them.

    Parameters
    ----------
    input_array : array_like
        Data of shape ``(f_{D-degree}, N, ..., N)``.
    degree : int
        The dual degree $D - p$.
    lattice : BarycubicLattice
        The lattice.
    dtype : data-type, optional
        Passed to ``numpy.asarray``.
    '''

    __batch_tag__ = 'BarycubicDualForm'

    dual = True
    '''Whether the form lives on dual cells; always ``True`` here.'''

    @classmethod
    def spatial_shape(cls, *, degree, lattice):
        r'''
        Parameters
        ----------
        degree : int
            Dual degree $D - p$.
        lattice : BarycubicLattice
            The lattice.

        Returns
        -------
        tuple
            ``(f_p, N, ..., N)`` with $p = D - $ ``degree``, the primal labels of the dual cells.

        Raises
        ------
        ValueError
            If ``degree`` is not an integer in $0, \ldots, D$.
        '''
        degree = _check_degree(lattice.D, degree)
        return (lattice.per_site[lattice.D - degree],) + (lattice.N,) * lattice.D

    def __new__(cls, input_array, *, degree, lattice, dtype=None):
        return super().__new__(cls, input_array, degree=degree, lattice=lattice, dtype=dtype)

    @property
    def primal_degree(self):
        r'''The degree $p = D - \texttt{degree}$ of the primal simplices labelling the dual cells.'''
        return self.lattice.D - self.degree


def _rebuild_form(data, degree, lattice, dual):
    '''Unpickle a form: see BarycubicForm.__reduce__.'''
    cls = BarycubicDualForm if dual else BarycubicForm
    return cls(data, degree=degree, lattice=lattice)


# ---------------------------------------------------------------------------------------------
# Operators.  The package-level names supervillain.lattice.d, delta, laplacian, star, wedge,
# cup dispatch here for barycubic forms.
# ---------------------------------------------------------------------------------------------

# The dtypes the operators accept.  Unsigned integers would wrap around under the signs of d
# and the cup products, and bool is not a ring the signs make sense in; the numba kernels are
# compiled for exactly these.
_SUPPORTED_DTYPES = frozenset(np.dtype(t) for t in (
    np.int8, np.int16, np.int32, np.int64,
    np.float32, np.float64, np.complex64, np.complex128))


def _check_form(f, operation):
    r'''
    Reject an unsupported dtype (TypeError) or anything that is not a whole form of its degree
    (ValueError): a slice, reshape, or fancy-indexed piece of a form is still an instance of
    the class, but it is not a cochain.
    '''
    if f.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{operation} does not accept {f.dtype} forms: cast to a signed integer "
                        f"or float type first, for example f.astype(np.int64) or f.astype(float)")
    if not f._is_whole():
        expected = None
        if f.lattice is not None and _is_degree(f.degree) and 0 <= f.degree <= f.lattice.D:
            expected = type(f).spatial_shape(degree=f.degree, lattice=f.lattice)
        raise ValueError(f"{operation} needs a whole {type(f).__name__} of degree {f.degree} with "
                         f"shape {expected}, got shape {f.shape}")
    return f


def _primal(f, operation):
    '''Reject anything that is not a whole, primal BarycubicForm of a supported dtype.'''
    if isinstance(f, BarycubicDualForm):
        raise TypeError(f"{operation} is not defined on a BarycubicDualForm; dual forms support "
                        f"only arithmetic, inverse_star, and pairing with primal forms")
    if not isinstance(f, BarycubicForm):
        raise TypeError(f"{operation} needs a BarycubicForm, got {type(f).__name__}")
    return _check_form(f, operation)


def _flat(f):
    '''The raw data of a form as a C-contiguous (labels, sites) array, for the kernels.'''
    return np.ascontiguousarray(np.asarray(f)).reshape(f.shape[0], -1)


def _same_lattice(a, b, operation):
    if (a.lattice.D, a.lattice.N) != (b.lattice.D, b.lattice.N):
        raise ValueError(f"{operation} needs forms on the same lattice, got {a.lattice} and {b.lattice}")


def _weights_broadcast(lattice, p):
    return lattice._local.weights[p][(slice(None),) + (None,) * lattice.D]


def _accumulate_incidence(out, src, lattice, op, degree, coefficient=None):
    r'''
    out += (op src) for the incidence operator 'd' or 'dT' at input degree ``degree``, where
    out and src are raw (labels, sites) arrays.  ``coefficient`` replaces the ±1 signs of the
    table row by row (δ folds its weights in that way); by default the signs are used, cast to
    the output dtype so integer forms stay exact integers.
    '''
    plan = lattice._local.incidence_plan(op, degree)
    if coefficient is None:
        coefficient = plan['sign'].astype(out.dtype)
    _kernels.incidence(out, src, plan['out'], plan['in'], plan['row'], coefficient,
                       lattice._neighbors(plan['offsets']))
    return out


def _apply_incidence(f, op, out_degree):
    '''(op f) as a raw array of the output shape, in the dtype of f.'''
    lat = f.lattice
    out = np.zeros((lat.per_site[out_degree], lat.sites), dtype=f.dtype)
    _accumulate_incidence(out, _flat(f), lat, op, f.degree)
    return out.reshape((lat.per_site[out_degree],) + lat.dims)


def _apply_delta(f):
    '''δ f as a raw (labels, sites) array: one weighted dᵀ pass, see weighted_incidence_coefficients.'''
    lat, p = f.lattice, f.degree
    out = np.zeros((lat.per_site[p - 1], lat.sites), dtype=np.result_type(f.dtype, np.float64))
    _accumulate_incidence(out, _flat(f), lat, 'dT', p,
                          lat._local.weighted_incidence_coefficients(p))
    return out


def d(f):
    r'''
    The coboundary (exterior derivative) of a p-form, a (p+1)-form.

    With the vertices of every simplex in rank order,

    .. math::

        (df)(v_0 \ldots v_{p+1}) = \sum_{j=0}^{p+1} (-1)^j\, f(v_0 \ldots \widehat{v_j} \ldots v_{p+1}).

    Deleting a vertex from a rank-ordered list leaves a rank-ordered list, so every face value is
    read directly from storage without a reordering sign.  $d \circ d = 0$ exactly, and $d$
    commutes with the lattice symmetries (:func:`transform`, :func:`half_shift`) and with
    :func:`coarsen`.  Being a signed sum, it keeps integer forms integer.

    Parameters
    ----------
    f : BarycubicForm
        A p-form.

    Returns
    -------
    BarycubicForm
        The (p+1)-form $df$, or the scalar ``0`` if $f$ is a D-form (as for the hypercubic
        :func:`~supervillain.lattice.d`).

    Raises
    ------
    TypeError
        If ``f`` is a :class:`BarycubicDualForm`, not a barycubic form, or of an
        unsigned-integer, bool, or other unsupported dtype (cast to a signed integer or float
        type first).
    ValueError
        If a form argument is not whole: a slice or reshape of a form does not have the shape
        ``(f_p, N, ..., N)`` of its degree.
    '''
    _primal(f, 'd')
    lat, p = f.lattice, f.degree
    if p == lat.D:
        return 0
    return BarycubicForm(_apply_incidence(f, 'd', p + 1), degree=p + 1, lattice=lat)


def delta(f):
    r'''
    The codifferential of a p-form, a (p-1)-form: the adjoint of :func:`d` in :func:`inner`.

    .. math::

        \delta = M_{p-1}^{-1}\, d^\top M_p,
        \qquad \langle d a, b \rangle = \langle a, \delta b \rangle,

    where $M_p$ is the diagonal matrix of :meth:`Hodge weights <BarycubicLattice.hodge_weights>`
    and $d^\top$ is the transpose of the coboundary (the signed sum over cofaces).  This is the same
    adjointness convention as the hypercubic :func:`~supervillain.lattice.delta`.  It is
    evaluated as a diagonal scaling, one $d^\top$, and another diagonal scaling, never as a fused
    stencil.  The weights are irrational, so the result is floating point (complex stays complex).

    Parameters
    ----------
    f : BarycubicForm
        A p-form.

    Returns
    -------
    BarycubicForm
        The (p-1)-form $\delta f$, or the scalar ``0`` if $f$ is a 0-form.

    Raises
    ------
    TypeError
        If ``f`` is a :class:`BarycubicDualForm`, not a barycubic form, or of an
        unsigned-integer, bool, or other unsupported dtype (cast to a signed integer or float
        type first).
    ValueError
        If a form argument is not whole: a slice or reshape of a form does not have the shape
        ``(f_p, N, ..., N)`` of its degree.
    '''
    _primal(f, 'delta')
    lat, p = f.lattice, f.degree
    if p == 0:
        return 0
    # Both diagonal scalings are folded into the per-row coefficients of one dᵀ pass.
    return BarycubicForm(_apply_delta(f).reshape((lat.per_site[p - 1],) + lat.dims),
                         degree=p - 1, lattice=lat)


def laplacian(f):
    r'''
    The Hodge–de Rham Laplacian of a p-form, a p-form.

    .. math::

        \Delta = d\delta + \delta d.

    Because :func:`delta` is the adjoint of :func:`d` in :func:`inner`, $\Delta$ is
    self-adjoint and positive semidefinite in that inner product,
    $\langle \Delta f, f \rangle = \langle df, df \rangle + \langle \delta f, \delta f \rangle$,
    and its kernel on the $N^D$ torus has dimension $\binom{D}{p}$ (the harmonic forms).
    Unlike the hypercubic Laplacian it is not diagonal over labels, since the weights differ
    between simplex classes.

    Parameters
    ----------
    f : BarycubicForm
        A p-form.

    Returns
    -------
    BarycubicForm
        The p-form $\Delta f$ (floating point).

    Raises
    ------
    TypeError
        If ``f`` is a :class:`BarycubicDualForm`, not a barycubic form, or of an
        unsigned-integer, bool, or other unsupported dtype (cast to a signed integer or float
        type first).
    ValueError
        If a form argument is not whole: a slice or reshape of a form does not have the shape
        ``(f_p, N, ..., N)`` of its degree.
    '''
    _primal(f, 'laplacian')
    lat, p = f.lattice, f.degree
    # Promote BEFORE any derivative: d of an integer form stays in its integer dtype, and a
    # narrow one (int8, int16, ...) would wrap around before the weights are applied.  With the
    # promotion, laplacian(f) equals laplacian(f.astype(float)) exactly for integer f.
    dtype = np.result_type(f.dtype, np.float64)
    if f.dtype != dtype:
        f = BarycubicForm(np.asarray(f).astype(dtype), degree=p, lattice=lat)
    # Both terms accumulate into one preallocated output: δ(df) as a weighted dᵀ pass over df,
    # then d(δf) added on top.  The only temporaries are df and δf themselves.
    out = np.zeros((lat.per_site[p], lat.sites), dtype=dtype)
    if p < lat.D:
        df = _apply_incidence(f, 'd', p + 1).reshape(lat.per_site[p + 1], lat.sites)
        _accumulate_incidence(out, df, lat, 'dT', p + 1,
                              lat._local.weighted_incidence_coefficients(p + 1))
        del df
    if p > 0:
        _accumulate_incidence(out, _apply_delta(f), lat, 'd', p - 1)
    return BarycubicForm(out.reshape((lat.per_site[p],) + lat.dims), degree=p, lattice=lat)


def star(f):
    r'''
    The diagonal Hodge star of a p-form, a $(D-p)$-form on the barycentric dual cells.

    .. math::

        (\star f)_{\star\sigma} = M_p(\sigma)\, f_\sigma,
        \qquad M_p(\sigma) = \frac{\mathrm{Vol}_{D-p}(\star\sigma)}{\mathrm{Vol}_p(\sigma)}.

    The weights are those of :attr:`BarycubicLattice.hodge_weight_by_rank_set`.  The star is
    what turns a primal cochain into a density that can be summed against another primal
    cochain: the kinetic term of a field $\xi$ is
    $\langle \xi, \xi \rangle = \sum_\sigma \overline{\xi_\sigma}\, (\star \xi)_{\star\sigma}$,
    i.e. ``(xi.conj() * star(xi)).sum()``.  No coboundary
    acts on the result, because no dual complex is built.

    Parameters
    ----------
    f : BarycubicForm
        A primal p-form.

    Returns
    -------
    BarycubicDualForm
        The dual form with ``degree`` $D - p$ and ``primal_degree`` $p$, floating point.

    Raises
    ------
    TypeError
        If ``f`` is already a :class:`BarycubicDualForm`, not a barycubic form, or of an
        unsigned-integer, bool, or other unsupported dtype (cast to a signed integer or float
        type first).
    ValueError
        If a form argument is not whole: a slice or reshape of a form does not have the shape
        ``(f_p, N, ..., N)`` of its degree.
    '''
    _primal(f, 'star')
    lat, p = f.lattice, f.degree
    return BarycubicDualForm(_weights_broadcast(lat, p) * np.asarray(f),
                             degree=lat.D - p, lattice=lat)


def inverse_star(g):
    r'''
    The inverse of :func:`star`: divide a dual form by the weights of its primal simplices.

    .. math::

        (\star^{-1} g)_\sigma = g_{\star\sigma} / M_p(\sigma).

    Parameters
    ----------
    g : BarycubicDualForm
        A dual form of ``degree`` $D - p$.

    Returns
    -------
    BarycubicForm
        The primal p-form with ``inverse_star(star(f)) == f`` up to rounding.

    Raises
    ------
    TypeError
        If ``g`` is not a :class:`BarycubicDualForm`, or is of an
        unsigned-integer, bool, or other unsupported dtype (cast to a signed integer or float
        type first).
    ValueError
        If a form argument is not whole: a slice or reshape of a form does not have the shape
        ``(f_p, N, ..., N)`` of its degree.
    '''
    if not isinstance(g, BarycubicDualForm):
        raise TypeError(f"inverse_star needs a BarycubicDualForm, got {type(g).__name__}")
    _check_form(g, 'inverse_star')
    lat, p = g.lattice, g.primal_degree
    return BarycubicForm(np.asarray(g) / _weights_broadcast(lat, p), degree=p, lattice=lat)


def inner(a, b):
    r'''
    The Hodge inner product of two p-forms.

    .. math::

        \langle a, b \rangle = \sum_\sigma M_p(\sigma)\, \overline{a_\sigma}\, b_\sigma
                             = \sum_\sigma \overline{a_\sigma}\, (\star b)_{\star\sigma}.

    On real forms it is the symmetric bilinear form $\sum M_p a b$; on complex forms it is
    Hermitian (antilinear in the first slot) and positive definite, since every weight is
    positive.  :func:`delta` is the adjoint of :func:`d` in this inner product.

    Parameters
    ----------
    a, b : BarycubicForm
        Primal forms of the same degree on the same lattice.

    Returns
    -------
    numpy scalar
        $\langle a, b \rangle$ (floating point, or complex if either form is complex).

    Raises
    ------
    TypeError
        If either argument is a :class:`BarycubicDualForm`, not a barycubic form, or of an
        unsigned-integer, bool, or other unsupported dtype (cast to a signed integer or float
        type first).
    ValueError
        If the degrees or lattices differ, or a form is not whole (see :func:`d`).
    '''
    _primal(a, 'inner')
    _primal(b, 'inner')
    _same_lattice(a, b, 'inner')
    if a.degree != b.degree:
        raise ValueError(f"inner needs forms of equal degree, got {a.degree} and {b.degree}")
    A = _flat(a)
    if np.iscomplexobj(A):
        A = np.conj(A)
    # One contraction over (label, site), with no full-size weighted temporary.  einsum casts
    # every operand to the common dtype (float64 or complex128, because of the weights) before
    # multiplying, so integer forms cannot overflow in a * b.
    dtype = np.result_type(A.dtype, b.dtype, np.float64)
    return np.einsum('l,lx,lx->', a.lattice._local.weights[a.degree], A, _flat(b), dtype=dtype)


def cup(a, b, i=0):
    r'''
    Steenrod's cup-i product $a \cup_i b$ of a p-form and a q-form, a $(p+q-i)$-form.

    Every fine simplex is ordered by rank, which is all the Alexander–Whitney construction and
    its Steenrod generalizations need.  For $i = 0$ this is the Alexander–Whitney cup product,

    .. math::

        (a \cup b)(v_0 \ldots v_{p+q}) = a(v_0 \ldots v_p)\, b(v_p \ldots v_{p+q}),

    which is associative and obeys the Leibniz rule
    $d(a \cup b) = da \cup b + (-1)^p a \cup db$.  For $i > 0$, on a rank-ordered simplex
    $(v_0 \ldots v_n)$ with $n = p + q - i$,

    .. math::

        (a \cup_i b)(v_0 \ldots v_n) = \sum_{J = (j_0 < \cdots < j_i) \subseteq \{0..n\}}
            \varepsilon(J)\, a(v_A)\, b(v_B),

    where the cut points $J$ split $0..n$ into $i+2$ intervals sharing their endpoints,
    $A = [0, j_0] \cup [j_1, j_2] \cup \cdots$ collects the even-numbered intervals and
    $B = [j_0, j_1] \cup [j_2, j_3] \cup \cdots$ the odd-numbered ones (the last ends at $n$);
    terms with $\lvert A\rvert \neq p+1$ or $\lvert B\rvert \neq q+1$ are omitted.  The sign is

    .. math::

        \varepsilon(J) = (-1)^{\binom{p}{2} + \binom{n}{2} + \sum_{u \in U_B} \mathrm{ind}_U(u)},

    with $U = \{0..n\} \setminus J$ the uncut positions, $\mathrm{ind}_U(u) = \#\{w \in U : w < u\}$,
    and $U_B = \{u \in U : u - \mathrm{ind}_U(u) \text{ odd}\}$ the uncut positions that appear
    only in $b$'s face.  With these signs the products obey the Steenrod coboundary formula

    .. math::

        d(a \cup_i b) = da \cup_i b + (-1)^p a \cup_i db
            + (-1)^{p+q-i}\, a \cup_{i-1} b + (-1)^{pq+p+q}\, b \cup_{i-1} a,

    with $\cup_{-1} = 0$ and terms above degree $D$ dropped, and the top product is pointwise,
    $a \cup_p b = a\, b$ for two p-forms.  The signs are the unique solution of the coboundary
    formula on a simplex of sufficiently high dimension, with $\cup_0$ fixed to
    Alexander–Whitney; the pointwise identity fixes the one sign ($\cup_D$ of two D-forms) that
    no identity inside a D-dimensional complex can see.

    The product is bilinear and translation covariant, and keeps integer forms integer.

    Parameters
    ----------
    a : BarycubicForm
        A p-form.
    b : BarycubicForm
        A q-form on the same lattice.
    i : int
        Which product; ``0`` (the default) is the ordinary cup product.

    Returns
    -------
    BarycubicForm
        The $(p+q-i)$-form $a \cup_i b$.

    Raises
    ------
    ValueError
        Unless $0 \leq i \leq \min(p, q)$ and $p + q - i \leq D$, if the lattices differ,
        or if a form is not whole (see :func:`d`).
    TypeError
        If either argument is a :class:`BarycubicDualForm`, not a barycubic form, or of an
        unsigned-integer, bool, or other unsupported dtype (cast to a signed integer or float
        type first).
    '''
    _primal(a, 'cup')
    _primal(b, 'cup')
    _same_lattice(a, b, 'cup')
    lat, p, q = a.lattice, a.degree, b.degree
    _check_cup_degrees(lat.D, p, q, i)
    n = p + q - i
    # The kernel multiplies in the output dtype, so both factors are brought to it first; this
    # copies at most the factor whose dtype differs.
    dtype = np.result_type(a.dtype, b.dtype)
    A, B = _flat(a).astype(dtype, copy=False), _flat(b).astype(dtype, copy=False)
    plan = lat._local.cup_plan(p, q, i)
    out = np.zeros((lat.per_site[n], lat.sites), dtype=dtype)
    _kernels.contraction(out, A, B, plan['out'], plan['a'], plan['a_row'], plan['b'],
                         plan['b_row'], plan['sign'].astype(dtype), lat._neighbors(plan['offsets']))
    return BarycubicForm(out.reshape((lat.per_site[n],) + lat.dims), degree=n, lattice=lat)


def integrate(f):
    r'''
    The integral of a top form over the torus.

    .. math::

        \int f = \sum_\sigma \varepsilon_\sigma\, f_\sigma,

    the pairing with the fundamental cycle $[M] = \sum_\sigma \varepsilon_\sigma \sigma$, where
    $\varepsilon$ is :attr:`BarycubicLattice.orientation`.  Because $\partial [M] = 0$,
    $\int df = 0$ for every $(D-1)$-form $f$, and $\int$ agrees with the hypercubic sum of
    :func:`coarsen` ``(f)``.

    Parameters
    ----------
    f : BarycubicForm
        A D-form.

    Returns
    -------
    numpy scalar
        The integral; integer for integer forms.

    Raises
    ------
    ValueError
        If ``f`` is not a top form.
        Also if ``f`` is not whole: a slice or reshape of a form does not have the shape
        ``(f_p, N, ..., N)`` of its degree.
    TypeError
        If ``f`` is a :class:`BarycubicDualForm`, not a barycubic form, or of an
        unsigned-integer, bool, or other unsupported dtype (cast to a signed integer or float
        type first).
    '''
    _primal(f, 'integrate')
    lat = f.lattice
    if f.degree != lat.D:
        raise ValueError(f"integrate needs a top form (degree {lat.D}), got degree {f.degree}")
    return np.einsum('l,lx->', lat._local.orientation, _flat(f))


def coarsen(f):
    r'''
    Sum a fine p-cochain over each hypercubic p-cell, giving a hypercubic p-form.

    This is the pullback along the subdivision chain map $Sd$, which sends a hypercubic p-cell to
    the signed sum of the fine p-simplices that tile it:

    .. math::

        (Sd^\top f)(F) = \sum_{\sigma \subseteq F,\ \dim\sigma = p} o(\sigma, F)\, f_\sigma.

    A fine p-simplex lies in the p-cell $F$ (anchored at $x$ with directions $I$) exactly when its
    ranks are $0, 1, \ldots, p$ and its top vertex is the barycenter of $F$; there are $2^p p!$ of
    them.  $o(\sigma, F) = \pm 1$ compares the rank-order orientation of $\sigma$ with the
    orientation of $F$ in increasing direction order, which is the convention of the hypercubic
    :class:`~supervillain.lattice.Form`.  With it, coarsening is a cochain map,
    ``coarsen(d(f)) == supervillain.lattice.d(coarsen(f))``, it keeps integer forms integer, and
    for top forms the hypercubic sum of ``coarsen(f)`` equals :func:`integrate` ``(f)``.

    Parameters
    ----------
    f : BarycubicForm
        A p-form.

    Returns
    -------
    ~supervillain.lattice.Form
        The hypercubic p-form on ``Lattice(D, N)``.

    Raises
    ------
    TypeError
        If ``f`` is a :class:`BarycubicDualForm`, not a barycubic form, or of an
        unsigned-integer, bool, or other unsupported dtype (cast to a signed integer or float
        type first).
    ValueError
        If a form argument is not whole: a slice or reshape of a form does not have the shape
        ``(f_p, N, ..., N)`` of its degree.
    '''
    from supervillain.lattice.compact import Form
    _primal(f, 'coarsen')
    lat, p = f.lattice, f.degree
    matrix = lat._local.coarsen_matrix(p)
    src = np.asarray(f).reshape(lat.per_site[p], lat.sites)
    out = (matrix @ src).reshape((matrix.shape[0],) + lat.dims)
    return Form(out, degree=p, lattice=lat._hypercubic)


def _pullback(f, perm, flips, half, shift):
    r'''
    Pullback of f along g(ξ) = L ξ + t + 2 shift, with (L ξ)_k = flips_k ξ_{perm[k]} and
    t = (1, ..., 1) for the half shift, else 0.

    The per-D label map (_pullback_map, cached) gives g σ(j, x) = σ(image[j], L x + offset[j]),
    so (g* f)[j, x] = f[image[j], L x + offset[j] + shift].  Only the spatial part depends on N
    and the translation: one flattened-site table per distinct total offset, then one gather.
    '''
    lat, p = f.lattice, f.degree
    D, N = lat.D, lat.N
    image, offset = _pullback_map(D, p, tuple(perm), tuple(flips), half)
    distinct, image_row = np.unique(offset + np.asarray(shift, dtype=np.int64), axis=0,
                                    return_inverse=True)
    # Flattened index of (L x + o) mod N, assembled axis by axis: each axis contributes
    # ((flips_k x_{perm[k]} + o_k) mod N) * stride_k, and an axis takes only a few values o_k.
    sites = np.indices(lat.dims).reshape(D, -1)
    stride = [N ** (D - 1 - k) for k in range(D)]
    column = {}
    table = np.zeros((len(distinct), lat.sites), dtype=np.int64)
    for row, o in enumerate(distinct):
        for k in range(D):
            key = (k, int(o[k]))
            if key not in column:
                column[key] = ((flips[k] * sites[perm[k]] + o[k]) % N) * stride[k]
            table[row] += column[key]
    out = np.empty((lat.per_site[p], lat.sites), dtype=f.dtype)
    _kernels.gather(out, _flat(f), image, image_row.reshape(-1).astype(np.int64), table)
    return out.reshape((lat.per_site[p],) + lat.dims)


def transform(f, perm, flips, shift):
    r'''
    Pull a form back along a symmetry of the lattice: a hyperoctahedral transformation followed
    by an integer translation.

    On doubled coordinates the symmetry is

    .. math::

        g: \xi \mapsto \mathrm{flips} \odot (\xi_{\mathrm{perm}[0]}, \ldots, \xi_{\mathrm{perm}[D-1]}) + 2\,\mathrm{shift},

    and the pullback is $(g^* f)(\sigma) = f(g\sigma)$.  $g$ maps cells to cells of the same
    dimension, so it preserves rank and hence the rank order of every simplex: the pullback is
    a pure relabelling with no sign.  It commutes with :func:`d` and every :func:`cup`
    product, preserves the Hodge weights, and multiplies :attr:`BarycubicLattice.orientation` by
    $\det g$.

    Parameters
    ----------
    f : BarycubicForm
        A p-form.
    perm : sequence of int
        A permutation of $0, \ldots, D-1$.
    flips : sequence of int
        $D$ entries $\pm 1$.
    shift : sequence of int
        $D$ integers, the translation in hypercubic lattice units.

    Returns
    -------
    BarycubicForm
        The p-form $g^* f$, with the dtype of ``f``.

    Raises
    ------
    ValueError
        If ``perm`` is not a permutation of $0, \ldots, D-1$, ``flips`` are not $D$ entries
        $\pm 1$, or ``shift`` does not have $D$ entries.
        Also if ``f`` is not whole: a slice or reshape of a form does not have the shape
        ``(f_p, N, ..., N)`` of its degree.
    TypeError
        If ``perm``, ``flips``, or ``shift`` contain anything but integers (floats are rejected
        even when integral, rather than truncated), or if ``f`` is a
        :class:`BarycubicDualForm`, not a barycubic form, or of an unsigned-integer, bool, or
        other unsupported dtype (cast to a signed integer or float type first).
    '''
    _primal(f, 'transform')
    D = f.lattice.D
    # Check the types before converting: int() would silently truncate 0.5 to 0.  Only Python
    # and numpy integers are accepted; a float is rejected even when it is integral.
    for name, values in (('perm', perm), ('flips', flips), ('shift', shift)):
        try:
            values = list(values)
        except TypeError:
            raise TypeError(f"{name} must be a sequence of integers, got {values!r}") from None
        if not all(_is_degree(v) for v in values):
            raise TypeError(f"{name} must contain only integers, got {values!r}")
    perm = [int(k) for k in perm]
    flips = [int(s) for s in flips]
    shift = [int(s) for s in shift]
    if sorted(perm) != list(range(D)):
        raise ValueError(f"perm must be a permutation of 0..{D - 1}, got {perm}")
    if len(flips) != D or any(s not in (-1, 1) for s in flips):
        raise ValueError(f"flips must be {D} entries of ±1, got {flips}")
    if len(shift) != D:
        raise ValueError(f"shift must have {D} entries, got {shift}")
    out = _pullback(f, perm, flips, half=False, shift=shift)
    return BarycubicForm(out, degree=f.degree, lattice=f.lattice)


def half_shift(f):
    r'''
    Pull a form back along the half-diagonal translation.

    .. math::

        \tau: \xi \mapsto \xi + (1, \ldots, 1)

    moves every barycenter by half a lattice spacing in every direction.  It maps cells to
    cells, sending a vertex of rank $r$ to one of rank $D - r$ (it exchanges sites with
    hypercubes, links with cubes, ...), so it is a symmetry of the fine complex that is invisible
    on the hypercubic lattice.  It reverses the rank order of every simplex, and a stored
    value refers to rank order, so the pullback carries the sign of the order reversal:

    .. math::

        (\tau^* f)(\sigma) = (-1)^{\binom{p+1}{2}}\, f[\tau\sigma].

    It commutes with :func:`d`, maps weights $M(R)$ to $M(D-R)$ (which are equal), and obeys the
    order-reversal law $\tau^*(a \cup_i b) = (-1)^{pq + i + \binom{i+1}{2}}\, \tau^* b \cup_i \tau^* a$
    for even $i$; for odd $i$ the law holds only when $p = q = i$.

    Parameters
    ----------
    f : BarycubicForm
        A p-form.

    Returns
    -------
    BarycubicForm
        The p-form $\tau^* f$, with the dtype of ``f``.

    Raises
    ------
    TypeError
        If ``f`` is a :class:`BarycubicDualForm`, not a barycubic form, or of an
        unsigned-integer, bool, or other unsupported dtype (cast to a signed integer or float
        type first).
    ValueError
        If a form argument is not whole: a slice or reshape of a form does not have the shape
        ``(f_p, N, ..., N)`` of its degree.
    '''
    _primal(f, 'half_shift')
    lat, p = f.lattice, f.degree
    out = _pullback(f, tuple(range(lat.D)), (1,) * lat.D, half=True, shift=(0,) * lat.D)
    if comb(p + 1, 2) % 2:
        np.negative(out, out=out)
    return BarycubicForm(out, degree=p, lattice=lat)


def wedge(a, b):
    r'''
    The cup product $a \cup_0 b$ (Alexander–Whitney), under the name the hypercubic lattice uses.

    See :func:`cup`.

    Parameters
    ----------
    a : BarycubicForm
        A p-form.
    b : BarycubicForm
        A q-form on the same lattice.

    Returns
    -------
    BarycubicForm
        The $(p+q)$-form $a \cup b$.

    Raises
    ------
    ValueError
        If $p + q > D$.
    '''
    return cup(a, b, 0)
