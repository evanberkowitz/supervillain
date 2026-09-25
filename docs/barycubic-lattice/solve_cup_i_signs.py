#!/usr/bin/env python3
"""Uniqueness certificate for the Steenrod cup-i signs frozen in SPEC.md (section 5.5).

On the standard simplex with vertices 0..NV-1, give every interval-cut term of every
(p, q, i) an unknown coefficient.  Fix cup_0 = Alexander-Whitney, then solve level by level
in i for the coefficients imposed by the Steenrod identity (convention of
test/test_cup_products.py::_steenrod_residual) on random integer cochains.

Scope of the claim, stated carefully (see SPEC.md 5.5):
  * Only coefficients of products with output degree n <= NV-2 enter an identity that fits in
    the simplex; the script reports which of them the identity DETERMINES (they lie outside
    the null space of the constraint matrix) and checks that each equals +-1 and agrees with
    the closed-form rule of verify_cup_i_signs.py.
  * Inside a D-dimensional complex the top product u_D (n = D) enters no identity at all, so
    its sign is fixed only by this universal (larger-simplex) family, or by the pointwise test
    a u_p b = ab.  That is why the pointwise test is mandatory in the package test suite.

Requires numpy.  Usage:  python3 solve_cup_i_signs.py [NV] [max_i] [trials]   (default NV=10, max_i=5, trials=6)
"""
import itertools
import random
import sys
from math import comb

import numpy as np

NV = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MAX_I = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
SIMP = {k: list(itertools.combinations(range(NV), k + 1)) for k in range(NV)}
IDX = {k: {s: j for j, s in enumerate(SIMP[k])} for k in SIMP}


def closed_form(p, q, i, n, J):
    U = [u for u in range(n + 1) if u not in J]
    e = comb(p, 2) + comb(n, 2) + sum(ind for ind, u in enumerate(U) if (u - ind) % 2 == 1)
    return -1 if e % 2 else 1


def labels(p, q, i):
    n = p + q - i
    out = []
    for J in itertools.combinations(range(n + 1), i + 1):
        cuts = [0, *J, n]
        A, B = [], []
        for k in range(i + 2):
            (A if k % 2 == 0 else B).extend(range(cuts[k], cuts[k + 1] + 1))
        if len(A) == p + 1 and len(B) == q + 1:
            out.append((J, tuple(A), tuple(B)))
    return out


def rand(p, rng):
    return np.array([rng.randint(-3, 3) for _ in SIMP[p]], float)


def cob(w, p):
    out = np.zeros(len(SIMP[p + 1]))
    for t, s in enumerate(SIMP[p + 1]):
        out[t] = sum((-1) ** j * w[IDX[p][s[:j] + s[j + 1:]]] for j in range(p + 2))
    return out


def terms(a, p, b, q, i):
    n = p + q - i
    L = labels(p, q, i)
    T = np.zeros((len(L), len(SIMP[n])))
    for t, s in enumerate(SIMP[n]):
        for l, (J, A, B) in enumerate(L):
            T[l, t] = a[IDX[p][tuple(s[x] for x in A)]] * b[IDX[q][tuple(s[x] for x in B)]]
    return L, T


signs = {}


def cup(a, p, b, q, i):
    L, T = terms(a, p, b, q, i)
    return np.array([signs[(p, q, i)][J] for J, _, _ in L]) @ T


def main():
    for p in range(NV):
        for q in range(NV - p):
            signs[(p, q, 0)] = {J: 1 for J, _, _ in labels(p, q, 0)}
    rng = random.Random(0)
    determined = agree = 0
    for i in range(1, MAX_I + 1):
        keys = [(p, q) for p in range(i, NV) for q in range(i, NV) if p + q - i <= NV - 2]
        var = {}
        for p, q in keys:
            for J, _, _ in labels(p, q, i):
                var[(p, q, J)] = len(var)
        rows, rhs = [], []
        for _ in range(TRIALS):
            for p, q in keys:
                n = p + q - i
                if n + 1 > NV - 2:
                    continue
                a, b = rand(p, rng), rand(q, rng)
                M = np.zeros((len(SIMP[n + 1]), len(var)))
                L, T = terms(a, p, b, q, i)
                for l, (J, _, _) in enumerate(L):
                    M[:, var[(p, q, J)]] += cob(T[l], n)
                for aa, pa, bb, qb, c in [(cob(a, p), p + 1, b, q, 1), (a, p, cob(b, q), q + 1, (-1) ** p)]:
                    L2, T2 = terms(aa, pa, bb, qb, i)
                    for l, (J, _, _) in enumerate(L2):
                        M[:, var[(pa, qb, J)]] -= c * T2[l]
                r = (-1) ** n * cup(a, p, b, q, i - 1) + (-1) ** (p * q + p + q) * cup(b, q, a, p, i - 1)
                rows.append(M)
                rhs.append(r)
        M, r = np.vstack(rows), np.concatenate(rhs)
        x, *_ = np.linalg.lstsq(M, r, rcond=None)
        assert np.abs(M @ x - r).max() < 1e-8, "identity system inconsistent"
        _, S, Vt = np.linalg.svd(M)
        null = Vt[(S > 1e-8 * S.max()).sum():]
        det = (np.abs(null).max(axis=0) < 1e-8) if len(null) else np.ones(len(var), bool)
        for (p, q, J), v in var.items():
            signs.setdefault((p, q, i), {})[J] = int(round(x[v]))
            if det[v]:
                assert abs(abs(x[v]) - 1) < 1e-8, f"non-unit determined coefficient {(p, q, i, J)}"
                determined += 1
                agree += int(round(x[v])) == closed_form(p, q, i, p + q - i, J)
        print(f"i={i}: {len(var)} coefficients, {int(det.sum())} determined by the identity")
    print(f"NV={NV}: {determined} determined coefficients, {agree} agree with the closed form")
    assert agree == determined


if __name__ == "__main__":
    main()
