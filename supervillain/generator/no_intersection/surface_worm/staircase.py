#!/usr/bin/env python

r"""Integer primitive (staircase) for the surface worm's closure step.

At worm closure the swept increment ΔF is a closed integer 2-form with
zero periods; the worm returns to the n-representation by finding an
integer 1-form a with da = ΔF, then n += a and φ is redrawn exactly.
This module builds that primitive.

Status (2026-07-25):
  * 2D core VERIFIED (`primitive2`): the seam construction — integrate in
    x1 to satisfy all non-seam rows, fix the single seam row with a1
    integrated in x0 (closes periodically precisely because the one
    period, the total flux, vanishes). Gate: da == b exactly, integer, on
    random exact 2-forms, N = 3..8.
  * GENERAL D BUILT + VERIFIED (`primitive_2form`, with `primitive_1form`):
    cone/homotopy along axis 0 with a seam correction supplied by a
    lower-dimensional 1-form primitive, recursing on the x₀=0 slice.
    Integrality is exact (only cumulative sums of integers). Gate
    `_gate_generic`: da == b exactly, D = 2,3,4, N = 3,4,5. This is the
    map's `F → n` step (used by `reconstruct.py`); the mean-based
    suspension sketched earlier was abandoned — means break integrality.

The d convention (verified against the library, 2026-07-24):
  (da)_{μν}(x) = a_μ(x) + a_ν(x+μ̂) − a_μ(x+ν̂) − a_ν(x),
  plaquette components ordered (01,02,03,12,13,23) = comps 0..5.

Note on exactness: `primitive_2form` is permissive by design — it does not
raise on closed-non-exact 2-forms, accepting them as a linear map for use
in the sampler. The `reconstruct` function serves as the gate for exactness;
gates in tests use `d1(a) == b` to verify the outcome on exact forms.
"""

import numpy as np


def d2(a0, a1):
    r"""(da)_{01} for a 1-form (a0,a1) on a 2-torus, library convention."""
    return a0 + np.roll(a1, -1, 0) - np.roll(a0, -1, 1) - a1


def primitive2(b):
    r"""Integer (a0,a1) with da==b, for closed integer b (Σb==0) on N×N.

    VERIFIED. Construction: integrate b in x1 via a0 (satisfies rows
    x1=0..N−2 exactly); the seam row x1=N−1 then carries residual equal to
    the per-x0 column sums C(x0), fixed by a1 on that row integrated in x0
    — which closes periodically because Σ_x0 C = Σ b = 0.
    """
    N = b.shape[0]
    if b.sum() != 0:
        raise ValueError('primitive2 needs zero total flux (the one period)')
    a0 = np.zeros_like(b)
    a1 = np.zeros_like(b)
    for x1 in range(N - 1):
        a0[:, x1 + 1] = a0[:, x1] - b[:, x1]
    C = b.sum(axis=1)
    for x0 in range(N - 1):
        a1[x0 + 1, N - 1] = a1[x0, N - 1] + C[x0]
    return a0, a1


# ---------------------------------------------------------------- generic D
# Forms on T^D (size N per axis), integer-valued:
#   0-form: array (N,)*D               1-form: (D,)+(N,)*D
#   2-form: (C(D,2),)+(N,)*D, components ordered by combinations(range(D),2).
# The exterior derivative uses the verified library convention.

from itertools import combinations


def _D_from_2form(b):
    C = b.shape[0]
    D = int(round((1 + (1 + 8 * C) ** 0.5) / 2))
    assert D * (D - 1) // 2 == C, f'{C} is not C(D,2)'
    return D


def _pidx(D):
    return {p: i for i, p in enumerate(combinations(range(D), 2))}


def d0(lam):
    r"""0-form → 1-form: (dλ)_μ(x) = λ(x+μ̂) − λ(x)."""
    D = lam.ndim
    return np.stack([np.roll(lam, -1, axis=mu) - lam for mu in range(D)])


def d1(a):
    r"""1-form → 2-form (library convention), general D."""
    D = a.shape[0]
    out = []
    for (mu, nu) in combinations(range(D), 2):
        out.append(a[mu] + np.roll(a[nu], -1, axis=mu)
                   - np.roll(a[mu], -1, axis=nu) - a[nu])
    return np.stack(out)


def primitive_1form(m):
    r"""Integer 0-form λ with dλ == m, for an EXACT integer 1-form m on T^D.

    Axial line integral from the origin along axes 0,1,…,D−1 in turn (each
    leg an exclusive cumulative sum). For an exact m the line integral is
    path-independent, so dλ == m holds on every link including the seams —
    no period correction needed (gated).
    """
    D = m.shape[0]
    N = m.shape[1]
    lam = np.zeros((N,) * D, dtype=np.int64)
    for a in range(D):
        base = m[a]
        for b in range(D - 1, a, -1):        # fix axes > a to index 0
            base = np.take(base, 0, axis=b)
        exc = np.cumsum(base, axis=a) - base  # exclusive prefix sum along axis a
        add = exc
        for _ in range(a + 1, D):             # broadcast over axes > a
            add = add[..., None]
        lam = lam + np.broadcast_to(add, (N,) * D)
    return lam


def primitive_2form(b):
    r"""Integer 1-form a with da == b, for an EXACT integer 2-form b on T^D
    (db = 0, all C(D,2) periods 0). General D ≥ 2, integrality exact.

    Cone/homotopy along axis 0 (rest = {1,…,D−1}):
      1. a_j(x) = Σ_{p<x₀} b_{0j}(p, x_≥1)         (j ∈ rest) — sets the
         (0,j) plaquettes off the x₀ seam.
      2. seam fix: R_j = Σ_{x₀} b_{0j} is an exact 1-form on T^{D−1};
         c = −primitive_1form(R); put a_0 on the x₀ = N−1 slab = c, which
         injects −d_rest c = R and closes the seam for every (0,j) at once.
      3. residual b_{ij}(x₀=0) (i,j ∈ rest) is an exact 2-form on T^{D−1};
         recurse and lift the result x₀-independently into a_i.

    Note: This function is permissive by design and does not raise on
    closed-non-exact input; it accepts the 2-form as a linear map for use
    in the sampler. The reconstruct function is the gate for exactness.
    """
    D = _D_from_2form(b)
    N = b.shape[1]
    pid = _pidx(D)
    a = np.zeros((D,) + (N,) * D, dtype=np.int64)
    if D < 2:
        return a
    # step 1
    for j in range(1, D):
        b0j = b[pid[(0, j)]]
        a[j] += np.cumsum(b0j, axis=0) - b0j
    # step 2
    R = np.stack([b[pid[(0, j)]].sum(axis=0) for j in range(1, D)])  # (D-1,)+(N,)*(D-1)
    c = -primitive_1form(R)
    a[0][N - 1] = c
    # step 3
    if D - 1 >= 2:
        rest = list(range(1, D))
        b0 = np.stack([b[pid[(i, j)]][0] for (i, j) in combinations(rest, 2)])
        arest = primitive_2form(b0)                # 1-form on T^{D-1}
        for k, i in enumerate(rest):
            a[i] += arest[k][None]                 # lift x0-independently
    return a
