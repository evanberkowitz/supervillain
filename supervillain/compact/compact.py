#!/usr/bin/env python
# coding: utf-8
#
# compact.py — Differential forms on a D-dimensional hypercubic lattice,
# stored without wasted space.
#
# MOTIVATION
# ----------
# The interlaced representation (see interlaced.py) places p-form data at
# sites with exactly p odd coordinates in a (2N)^D array.  That is correct
# but sparse: only C(D,p)/2^D of array elements are used.  Here we store a
# p-form compactly as an array
# of shape
#
#     (C(D,p), N, N, ..., N)       [D spatial axes follow]
#
# The first axis indexes the C(D,p) "components", listed in lexicographic
# order by the sorted tuple of directions that are "form directions".
#
# Example — D=3:
#   0-form components:  ()            → shape (1, N, N, N)
#   1-form components:  (0,),(1,),(2,) → shape (3, N, N, N)
#   2-form components:  (0,1),(0,2),(1,2) → shape (3, N, N, N)
#   3-form components:  (0,1,2)       → shape (1, N, N, N)

from itertools import combinations
from math import comb
import numpy as np


# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------

class Lattice:
    """
    A D-dimensional hypercubic lattice with N sites per direction.

    The lattice knows how to enumerate the components of any p-form and
    can create zero or random p-forms on demand.
    """

    def __init__(self, D, N):
        self.D = D
        self.N = N

        # components[p] — ordered list of component tuples for p-forms.
        # Each tuple is a sorted sequence of p direction indices (0-based).
        # Length = C(D, p).  Order is lexicographic.
        self.components = {
            p: list(combinations(range(D), p))
            for p in range(D + 1)
        }

        # comp_index[p][dirs] — reverse map: tuple of dirs → integer index.
        # Use comp_index[p][(i, j)] to find where component (i,j) lives
        # along axis 0 of a p-form array.
        self.comp_index = {
            p: {comp: idx for idx, comp in enumerate(self.components[p])}
            for p in range(D + 1)
        }

    # ------------------------------------------------------------------
    # Factory methods

    def zeros(self, p, dtype=float):
        """Return a zero p-form: shape (C(D,p), N,...,N)."""
        shape = (comb(self.D, p),) + (self.N,) * self.D
        return Form(np.zeros(shape, dtype=dtype), degree=p, lattice=self)

    def random(self, p):
        """Return a p-form with random entries uniform in [0, 1)."""
        shape = (comb(self.D, p),) + (self.N,) * self.D
        return Form(np.random.random(shape), degree=p, lattice=self)

    def __repr__(self):
        return f"Lattice(D={self.D}, N={self.N})"


# ---------------------------------------------------------------------------
# Form  (numpy.ndarray subclass)
# ---------------------------------------------------------------------------

class Form(np.ndarray):
    """
    A differential p-form on a Lattice, stored compactly.

    Underlying array shape: (C(D,p), N, N, ..., N).
    - Axis 0 : component index (C(D,p) values, lex order by direction tuple).
    - Axes 1…D : physical lattice site n = (n_0, …, n_{D-1}).

    RELATIONSHIP TO THE INTERLACED LAYOUT
    In the interlaced (2N)^D array a p-form component with odd directions I
    lives at interlaced coordinates x where:
        x_k = 2 n_k      for k ∉ I  (even — a "site" direction)
        x_k = 2 n_k + 1  for k ∈ I  (odd  — a "link/plaquette/…" direction)
    In both cases  x_k // 2 = n_k, so the physical site is always the floor
    of the interlaced coordinate divided by 2.  This is why from_interlaced
    can use slice(0,None,2) and slice(1,None,2) interchangeably to extract
    the same physical-site index range for every component.

    ARITHMETIC
    Element-wise operations (±, *, /, unary −, abs, **, np.sqrt, np.isclose,
    ==, …) return a Form of the same degree when all Form operands share a
    degree.  Reductions (sum, max, …) return plain numpy scalars or arrays.
    Mixed-degree arithmetic is left as a plain ndarray because the degree of
    the result would be ambiguous.

    Numpy subclassing protocol:
    - __new__          : attach degree and lattice when the object is created.
    - __array_finalize__: propagate them when numpy makes an internal view.
    - __array_ufunc__  : intercept every ufunc call to return a typed Form
                         when the result is unambiguous.
    """

    def __new__(cls, input_array, degree, lattice):
        obj = np.asarray(input_array).view(cls)
        obj.degree  = degree
        obj.lattice = lattice
        return obj

    def __array_finalize__(self, obj):
        # obj is None when called from an explicit constructor,
        # otherwise it is the "parent" array being viewed.
        if obj is None:
            return
        self.degree  = getattr(obj, 'degree',  None)
        self.lattice = getattr(obj, 'lattice', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Collect every Form among the inputs and check degree agreement.
        form_inputs = [x for x in inputs if isinstance(x, Form)]
        degrees     = {f.degree for f in form_inputs}

        # Strip Form wrappers so the ufunc operates on raw data.
        raw = tuple(np.asarray(x) for x in inputs)
        out = kwargs.get('out')
        if out is not None:
            kwargs['out'] = tuple(np.asarray(o) for o in out)

        result = getattr(ufunc, method)(*raw, **kwargs)

        # Re-wrap as a Form when the result is unambiguous:
        #   • all Form inputs share the same degree (mixed-degree → plain)
        #   • same shape as the inputs (element-wise, not a reduction)
        # Boolean results (np.isclose, ==, …) are included: "is component k
        # at site n close to zero?" has the same index structure as the form
        # itself, so the degree is still meaningful.
        if (len(degrees) == 1
                and isinstance(result, np.ndarray)
                and result.shape == form_inputs[0].shape):
            return Form(result,
                        degree=form_inputs[0].degree,
                        lattice=form_inputs[0].lattice)

        return result

    # ------------------------------------------------------------------
    # Component access

    def component(self, *dirs):
        """
        View of a single component's spatial data, shape (N,...,N).

        Pass the direction indices as separate arguments or as a tuple:
            f.component(0, 2)   — the (0,2)-component of a 2-form
            f.component((0, 2)) — same

        The return value is a *view*, so writes back to the Form.
        """
        # Accept either f.component(0,2) or f.component((0,2))
        if len(dirs) == 1 and hasattr(dirs[0], '__iter__'):
            dirs = tuple(dirs[0])
        dirs = tuple(sorted(dirs))
        idx = self.lattice.comp_index[self.degree][dirs]
        return self[idx]   # shape (N,...,N), a view

    def to_interlaced(self):
        """
        Embed the compact form into a (2N)^D interlaced array (see
        interlaced.py).  Every site that is not a p-form site is zero.

        Each component indexed by direction tuple `comp` occupies the
        sub-array where direction k uses odd indices (start=1) if k is
        in `comp`, and even indices (start=0) otherwise.

        Returns a plain numpy.ndarray of shape (2N, 2N, ..., 2N).
        """
        lat = self.lattice
        D, N = lat.D, lat.N
        result = np.zeros((2 * N,) * D, dtype=self.dtype)

        for comp_tuple, comp_idx in lat.comp_index[self.degree].items():
            dirs = set(comp_tuple)
            slc = tuple(
                slice(1 if k in dirs else 0, None, 2)
                for k in range(D)
            )
            result[slc] = self[comp_idx]

        return result

    @classmethod
    def from_interlaced(cls, p, data, lattice=None):
        """
        Construct a compact Form from an interlaced (2N)^D array.

        p       — form degree
        data    — interlaced array of shape (2N, 2N, ..., 2N); only the
                  sites with exactly p odd coordinates are read.
        lattice — optional Lattice; inferred from data.shape if omitted.

        This is the left-inverse of to_interlaced():
            Form.from_interlaced(p, f.to_interlaced()) == f   for any p-Form f.

        For a valid interlaced p-form it is also the right-inverse:
            Form.from_interlaced(p, data).to_interlaced() == data
            (provided data is zero at all non-p-form sites).
        """
        D = data.ndim
        N = data.shape[0] // 2
        if lattice is None:
            lattice = Lattice(D, N)

        result = lattice.zeros(p, dtype=data.dtype)
        for comp_tuple, comp_idx in lattice.comp_index[p].items():
            dirs = set(comp_tuple)
            slc  = tuple(slice(1 if k in dirs else 0, None, 2) for k in range(D))
            result[comp_idx] = data[slc]

        return result

    def __repr__(self):
        return (
            f"Form(degree={self.degree}, "
            f"shape={self.shape}, "
            f"lattice={self.lattice})"
        )


# ---------------------------------------------------------------------------
# Exterior derivative  d : Ω^p → Ω^{p+1}
# ---------------------------------------------------------------------------

def d(f):
    """
    Exterior derivative of a p-form, returning a (p+1)-form.

    For each output component O = (o_0, …, o_p) the formula is

        d(f)[O, n]  =  Σ_{j=0}^{p}  (-1)^j  ·  Δ_{o_j}  f[O \\ {o_j}]

    where  Δ_k A  is the forward finite difference in direction k:

        Δ_k A[n]  =  A[n + ê_k] − A[n]
                   =  np.roll(A, -1, axis=k) − A

    The sign alternates ±1 over the directions of the target form, matching
    the interlaced convention in interlaced.py.  Periodic boundary conditions
    come for free from np.roll.
    """
    lat = f.lattice
    p   = f.degree
    if p == lat.D:
        return 0

    result = lat.zeros(p + 1)

    for out_comp in lat.components[p + 1]:          # each (p+1)-form component
        out_idx = lat.comp_index[p + 1][out_comp]

        for j, k_j in enumerate(out_comp):
            # Remove direction k_j from out_comp to get the source p-form component.
            in_comp = tuple(k for k in out_comp if k != k_j)
            in_idx  = lat.comp_index[p][in_comp]

            sign = (-1) ** j

            # Forward finite difference of f[in_idx] in direction k_j.
            # np.roll(A, -1, axis=k) shifts data so result[n] = A[n+1 mod N].
            spatial = f[in_idx]   # shape (N,...,N)
            fwd_diff = np.roll(spatial, -1, axis=k_j) - spatial

            result[out_idx] += sign * fwd_diff

    return result


# ---------------------------------------------------------------------------
# Codifferential  δ : Ω^p → Ω^{p-1}
# ---------------------------------------------------------------------------

def delta(f):
    """
    Codifferential (formal adjoint of d) of a p-form, returning a (p-1)-form.

    For each output component M = (m_0, …, m_{p-1}) and each direction
    e ∉ M, let j = #{m ∈ M : m < e} (the position where e would be
    inserted to keep M ∪ {e} sorted).  Then

        δ(F)[M, n]  =  Σ_{e ∉ M}  (-1)^j  ·  ∇*_e  F[M ∪ {e}]

    where  ∇*_e A  is the backward finite difference in direction e:

        ∇*_e A[n]  =  A[n] − A[n − ê_e]
                    =  A − np.roll(A, +1, axis=e)

    The sign (-1)^j equals (-1)^(e−i) where i is the position of e in the
    sorted complement of M in {0,…,D-1}, because e − i = #{m ∈ M : m < e}.
    This matches the interlaced convention in interlaced.py.
    """
    lat = f.lattice
    p   = f.degree
    if p == 0:
        return 0

    result = lat.zeros(p - 1)

    all_dirs = set(range(lat.D))

    for out_comp in lat.components[p - 1]:          # each (p-1)-form component
        out_idx = lat.comp_index[p - 1][out_comp]
        M_set   = set(out_comp)

        for e in sorted(all_dirs - M_set):          # directions not in M
            # Position where e is inserted into sorted(M ∪ {e}).
            j    = sum(1 for m in out_comp if m < e)
            sign = (-1) ** j

            in_comp = tuple(sorted(M_set | {e}))
            in_idx  = lat.comp_index[p][in_comp]

            # Backward finite difference of F[in_idx] in direction e.
            # np.roll(A, +1, axis=e) shifts so result[n] = A[n-1 mod N].
            spatial  = f[in_idx]
            bwd_diff = spatial - np.roll(spatial, +1, axis=e)

            result[out_idx] += sign * bwd_diff

    return result

δ = delta


# ---------------------------------------------------------------------------
# Hodge star  ★ : Ω^p → Ω^{D-p}
# ---------------------------------------------------------------------------

def _perm_sign(seq):
    """Sign of the permutation that sorts a sequence of distinct integers."""
    return (-1) ** sum(
        1 for i in range(len(seq)) for j in range(i + 1, len(seq))
        if seq[i] > seq[j]
    )


def star(f):
    """
    Hodge star of a p-form, returning a (D-p)-form.

    For each output component J (a sorted (D-p)-tuple of directions),
    let I = complement of J in {0,…,D-1}.  Then

        (★f)[J, n]  =  σ(I,J) · f[I, n − ê_I]

    where
        σ(I,J)  = sign of the permutation sorting the concatenation (I, J)
                = (-1)^(#{(i,j) ∈ I×J : i > j})
        ê_I     = Σ_{k ∈ I} ê_k   (sum of unit vectors for the I directions)

    and  f[I, n − ê_I]  is implemented as successive  np.roll(…, +1, axis=k)
    calls for k ∈ I.

    WHY THE SPATIAL SHIFT?
    In the discrete interlaced geometry, a p-form and its Hodge dual are
    "centred" at different lattice positions.  The shift aligns them so that
    the inner-product identity holds pointwise after summing:

        Σ_{n, I}  a_I[n] · b_I[n]  =  Σ_n  (a ∧ ★b)_{0,…,D-1}[n]

    For p=0 and p=D the shift is trivial (|I|=0 or the shifts cancel in the
    wedge), which is why a pure push without signs already works for those
    two degrees.
    """
    lat = f.lattice
    p   = f.degree
    D   = lat.D
    result = lat.zeros(D - p)

    for J_comp in lat.components[D - p]:
        J_set  = set(J_comp)
        I_comp = tuple(k for k in range(D) if k not in J_set)

        sign = _perm_sign(I_comp + J_comp)

        # Roll f[I_comp] by +1 in each direction k ∈ I_comp to
        # evaluate f_I at site n − ê_I.
        spatial = f[lat.comp_index[p][I_comp]]
        for k in I_comp:
            spatial = np.roll(spatial, +1, axis=k)

        result[lat.comp_index[D - p][J_comp]] = sign * spatial

    return result


# ---------------------------------------------------------------------------
# Wedge product  ∧ : Ω^n × Ω^m → Ω^{n+m}
# ---------------------------------------------------------------------------

def wedge(a, b):
    """
    Wedge product of an n-form a and an m-form b, returning an (n+m)-form.

    For each output component O = (o_0, …, o_{n+m-1}) we sum over all
    ways to split O into

        B_dirs  (n directions, the a-component index)
        A_dirs  (m directions, the b-component index)

    with the formula

        (a ∧ b)[O, n]  =  Σ  sign · a[B_dirs, n] · b[A_dirs, n + shift_B]

    where shift_B = Σ_{k ∈ B_dirs} ê_k, implemented as successive
    np.roll(…, -1, axis=k) calls on the b-component array.

    SIGN CONVENTION
    ---------------
    Let ASSIGNMENT be the tuple indexed by position in O where entry j
    is 1 if O[j] ∈ A_dirs (→ shifts a) and 0 if O[j] ∈ B_dirs.
    sign = (-1)^(# inversions in ASSIGNMENT).

    An inversion is a pair (i < j) with ASSIGNMENT[i]=1, ASSIGNMENT[j]=0,
    i.e. an A-direction appearing before a B-direction in O.  Equivalently:

        sign = (-1)^(Σ_{k ∈ B_dirs}  #{j ∈ A_dirs : j < k})

    This matches the interlaced wedge in interlaced.py exactly (verified by
    tracing the push/pull shifts for several (n,m) pairs).
    """
    lat = a.lattice
    n, m = a.degree, b.degree
    if n + m > lat.D:
        raise ValueError(
            f"Cannot wedge a {n}-form and a {m}-form in D={lat.D}: {n}+{m} > {lat.D}"
        )

    result = lat.zeros(n + m)

    for out_comp in lat.components[n + m]:
        out_idx = lat.comp_index[n + m][out_comp]

        # Enumerate all size-m subsets of out_comp → these become A_dirs.
        # The complementary size-n subset becomes B_dirs.
        for A_dirs in combinations(out_comp, m):
            B_dirs = tuple(k for k in out_comp if k not in A_dirs)
            # B_dirs is already sorted because out_comp is sorted and we
            # remove elements while preserving order.

            # Compute the sign: count A-before-B inversions.
            A_set = set(A_dirs)
            inversions = sum(
                1
                for k in B_dirs
                for j in A_dirs
                if j < k
            )
            sign = (-1) ** inversions

            # a is evaluated at site n using component B_dirs.
            a_idx = lat.comp_index[n][B_dirs]
            a_spatial = a[a_idx]   # shape (N,...,N)

            # b is evaluated at site n + shift_B, where shift_B moves
            # +1 in each direction k ∈ B_dirs.
            # Rolling b by -1 in axis k maps b[n] ← b[n+1_k], which is
            # equivalent to evaluating b at the site one step forward in k.
            b_idx = lat.comp_index[m][A_dirs]
            b_spatial = b[b_idx]   # shape (N,...,N)
            for k in B_dirs:
                b_spatial = np.roll(b_spatial, -1, axis=k)

            result[out_idx] += sign * a_spatial * b_spatial

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from itertools import product as iproduct

    parser = argparse.ArgumentParser(description="Test compact differential forms.")
    parser.add_argument('--D', type=int, default=4, help='Spacetime dimension')
    parser.add_argument('--N', type=int, default=6, help='Lattice size per direction')
    args = parser.parse_args()

    D = args.D
    N = args.N
    lat = Lattice(D, N)
    print(f"Testing on {lat}\n")

    # --- Sparsity report ---------------------------------------------------
    # How many numbers do we actually store per lattice site?
    print("STORAGE PER SITE")
    for p in range(D + 1):
        n_comps = comb(D, p)
        old_fraction = n_comps / 2**D
        print(f"  {p}-form: {n_comps} component(s) per site "
              f"(was {old_fraction:.4f} × 2^D = {n_comps} nonzero in old layout)")

    # --- Nilpotency of d ---------------------------------------------------
    print("\nNILPOTENCY OF d")
    for p in range(D + 1):
        a  = lat.random(p)
        da = d(a)
        if p == D:
            assert da == 0, f"d({p}-form) should be scalar 0"
            print(f'  ✅ d({p}-form) = 0')
        else:
            assert not np.isclose(da, 0).all(), f"d({p}-form) is unexpectedly zero"
            dda = d(da)
            # np.asarray handles both the scalar-0 case (p == D-1) and the
            # Form case (p < D-1) uniformly.
            assert np.isclose(np.asarray(dda), 0).all(), f"d² ≠ 0 on {p}-form"
            print(f'  ✅ d² = 0 on {p}-forms')

    # --- Nilpotency of δ ---------------------------------------------------
    print("\nNILPOTENCY OF δ")
    for p in range(D + 1):
        a  = lat.random(p)
        da = delta(a)
        if p == 0:
            assert da == 0, f"δ({p}-form) should be scalar 0"
            print(f'  ✅ δ(0-form) = 0')
        else:
            assert not np.isclose(da, 0).all(), f"δ({p}-form) is unexpectedly zero"
            dda = delta(da)
            assert np.isclose(np.asarray(dda), 0).all(), f"δ² ≠ 0 on {p}-form"
            print(f'  ✅ δ² = 0 on {p}-forms')

    # --- Adjointness  Σ (da)·b = -Σ a·(δb) --------------------------------
    # Inner product of two p-forms: sum over all components and all sites.
    # In compact layout every array element is a genuine DOF, so np.sum()
    # is the correct lattice L² inner product.
    print("\nADJOINTNESS  ⟨da, b⟩ = ⟨a, δb⟩")
    for p in range(D):
        a = lat.random(p)
        b = lat.random(p + 1)
        LHS = +( d(a) * b ).sum()
        RHS = -( a * delta(b) ).sum()
        assert np.isclose(LHS, RHS), f"Adjointness failed for p={p}: {LHS} vs {RHS}"
        print(f"  ✅ ⟨da, b⟩ = ⟨a, δb⟩  for a ∈ Ω^{p}, b ∈ Ω^{p+1}")

    # --- Wedge: bilinearity ------------------------------------------------
    print("\nBILINEARITY")
    for n, m in iproduct(range(D + 1), repeat=2):
        if n + m > D:
            continue
        a = lat.random(n);  b = lat.random(n);  c = lat.random(m);  e = lat.random(m)
        assert np.isclose(wedge(a + b, c), wedge(a, c) + wedge(b, c)).all()
        assert np.isclose(wedge(a, c + e), wedge(a, c) + wedge(a, e)).all()
        print(f"  ✅ ∧ bilinear for {n} ∧ {m}")

    # --- Wedge: Leibniz rule  d(a∧b) = da∧b + (-1)^n a∧db ----------------
    print("\nLEIBNIZ RULE")
    for n, m in iproduct(range(D + 1), repeat=2):
        if n + m + 1 > D:
            continue
        a = lat.random(n);  b = lat.random(m)
        LHS = d(wedge(a, b))
        RHS = wedge(d(a), b) + (-1)**n * wedge(a, d(b))
        assert not np.isclose(LHS, 0).all()
        assert np.isclose(LHS, RHS).all(), f"Leibniz failed for {n}∧{m}"
        print(f"  ✅ d(a∧b) = da∧b + (-1)^{n} a∧db  for a ∈ Ω^{n}, b ∈ Ω^{m}")

    # --- Wedge: associativity  (a∧b)∧c = a∧(b∧c) -------------------------
    print("\nASSOCIATIVITY")
    for n, m, p in iproduct(range(D), repeat=3):
        if n + m + p > D:
            continue
        a = lat.random(n);  b = lat.random(m);  c = lat.random(p)
        LHS = wedge(wedge(a, b), c)
        RHS = wedge(a, wedge(b, c))
        assert not np.isclose(LHS, 0).all()
        assert np.isclose(LHS, RHS).all(), f"Associativity failed for {n},{m},{p}"
        print(f"  ✅ (a∧b)∧c = a∧(b∧c)  for degrees {n}, {m}, {p}")

    # --- Hodge star: inner product identity --------------------------------
    # Σ_{n,I} a_I[n] b_I[n]  =  Σ_n (a ∧ ★b)_{0,…,D-1}[n]
    print("\nHODGE STAR  Σ a·b = Σ (a∧★b)")
    for p in range(D + 1):
        a = lat.random(p)
        b = lat.random(p)
        LHS = (a * b).sum()
        RHS = wedge(a, star(b)).sum()
        assert np.isclose(LHS, RHS), \
            f"Hodge identity failed for p={p}: {LHS} vs {RHS}"
        print(f"  ✅ Σ a·b = Σ (a∧★b)  for p={p}")

    # --- to_interlaced round-trip ------------------------------------------
    # Verify that embedding a compact form into the (2N)^D array places
    # nonzero values only at sites with exactly p odd coordinates.
    print("\nTO_INTERLACED")
    TXYZ = np.mgrid[(slice(0, 2*N),) * D]
    odds = np.mod(TXYZ, 2).sum(axis=0)   # number of odd coords per site
    for p in range(D + 1):
        a = lat.random(p)
        big = a.to_interlaced()           # shape (2N,...,2N)
        # Nonzero only where exactly p coordinates are odd.
        wrong_sites = big[odds != p]
        assert np.all(wrong_sites == 0), f"to_interlaced put values at non-{p}-form sites"
        # All p-form sites are filled (random data is generically nonzero).
        right_sites = big[odds == p]
        assert not np.all(right_sites == 0), f"to_interlaced left {p}-form sites empty"
        print(f"  ✅ {p}-form embeds correctly into (2N)^D array")

    # --- Round-trip tests for from_interlaced / to_interlaced --------------
    print("\nROUND-TRIP: compact → interlaced → compact")
    for p in range(D + 1):
        f = lat.random(p)
        g = Form.from_interlaced(p, f.to_interlaced())
        assert (f == g).all(), f"compact→interlaced→compact failed for p={p}"
        print(f"  ✅ from_interlaced(to_interlaced(f)) == f  for p={p}")

    print("\nROUND-TRIP: interlaced → compact → interlaced")
    import interlaced as il
    il_lat = il.Lattice(D, N)
    for p in range(D + 1):
        data = il_lat.random(p)
        assert (Form.from_interlaced(p, data).to_interlaced() == data).all(), \
            f"interlaced→compact→interlaced failed for p={p}"
        print(f"  ✅ to_interlaced(from_interlaced(data)) == data  for p={p}")

    # --- Cross-validation against interlaced.py ----------------------------
    print("\nCROSS-VALIDATION: compact d vs interlaced d")
    for p in range(D):
        a = lat.random(p)
        path_A = d(a).to_interlaced()
        path_B = il.d(a.to_interlaced())
        assert (path_A == path_B).all(), f"compact d ≠ interlaced d on {p}-forms"
        print(f"  ✅ compact d = interlaced d  on {p}-forms")

    print("\nCROSS-VALIDATION: compact δ vs interlaced δ")
    for p in range(1, D + 1):
        a = lat.random(p)
        path_A = delta(a).to_interlaced()
        path_B = il.delta(a.to_interlaced())
        assert (path_A == path_B).all(), f"compact δ ≠ interlaced δ on {p}-forms"
        print(f"  ✅ compact δ = interlaced δ  on {p}-forms")

    print("\nCROSS-VALIDATION: compact wedge vs interlaced wedge")
    for p, q in iproduct(range(D + 1), repeat=2):
        if p + q > D:
            continue
        a = lat.random(p)
        b = lat.random(q)
        path_A = wedge(a, b).to_interlaced()
        path_B = il.wedge(p, q, a.to_interlaced(), b.to_interlaced())
        # The two paths can take different floating-point arithmetic orderings,
        # so compare with np.isclose.  Other cross-validations should be bit-exact matches.
        assert np.isclose(path_A, path_B).all(), f"compact ∧ ≠ interlaced ∧ for {p}∧{q}"
        print(f"  ✅ compact wedge = interlaced wedge  for {p}∧{q}")

    print("\nCROSS-VALIDATION: compact ★ vs interlaced ★")
    for p in range(D + 1):
        a = lat.random(p)
        path_A = star(a).to_interlaced()
        path_B = il.star(p, a.to_interlaced())
        assert (path_A == path_B).all(), f"compact ★ ≠ interlaced ★ on {p}-forms"
        print(f"  ✅ compact ★ = interlaced ★  on {p}-forms")

    print("\nAll tests passed.")
