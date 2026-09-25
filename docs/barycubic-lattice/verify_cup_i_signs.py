#!/usr/bin/env python3
"""Exact check of the Steenrod cup-i sign rule frozen in SPEC.md (section 5.5).

Standard library only.  Works on the standard simplex with vertices 0..NV-1 and random
integer cochains, using the coboundary (dw)(v0..vk) = sum_j (-1)^j w(v0..^vj..vk).

Checks, all exact integer arithmetic:
  1. cup_0 is Alexander-Whitney with sign +1.
  2. Steenrod identity, for every (p, q, i) whose coboundary still fits in the simplex:
       d(a u_i b) = da u_i b + (-1)^p a u_i db + (-1)^(p+q-i) a u_(i-1) b
                    + (-1)^(pq+p+q) b u_(i-1) a,          with u_(-1) = 0.
     This is the convention of test/test_cup_products.py::_steenrod_residual.
  3. Negative controls: flipping the sign of any single family of terms breaks check 2.
  4. Order reversal k -> NV-1-k:  rev(a u_i b) = (-1)^(pq + i + C(i+1,2)) rev(b) u_i rev(a)
     holds exactly for every even i and for the diagonal p = q = i, and fails for every
     other odd-i case tested.
  5. Pointwise top product: a u_p b = a*b for p-cochains a, b (the only product whose sign
     no Steenrod identity inside a p-dimensional complex can see).

Usage:  python3 verify_cup_i_signs.py [NV]      (default NV=10; NV=11 checks i up to 9)
"""
import itertools
import random
import sys
from math import comb

NV = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def sign(p, q, i, n, J, mutate=None):
    """Sign of the interval-cut term labelled by cut points J = (j_0 < ... < j_i)."""
    U = [u for u in range(n + 1) if u not in J]
    e = comb(p, 2) + comb(n, 2)
    for ind, u in enumerate(U):
        if (u - ind) % 2 == 1:          # odd number of cut points below u: u is only in b's face
            e += ind
    if mutate is not None and mutate(p, q, i, n, J):
        e += 1
    return -1 if e % 2 else 1


def labels(p, q, i, mutate=None, _cache={}):
    key = (p, q, i, mutate)
    if key not in _cache:
        n = p + q - i
        out = []
        for J in itertools.combinations(range(n + 1), i + 1):
            cuts = [0, *J, n]
            A, B = [], []
            for k in range(i + 2):
                (A if k % 2 == 0 else B).extend(range(cuts[k], cuts[k + 1] + 1))
            if len(A) == p + 1 and len(B) == q + 1:
                out.append((sign(p, q, i, n, J, mutate), tuple(A), tuple(B)))
        _cache[key] = out
    return _cache[key]


def rnd(p, rng):
    return {s: rng.randint(-3, 3) for s in itertools.combinations(range(NV), p + 1)}


def cob(w, p):
    return {s: sum((-1) ** j * w[s[:j] + s[j + 1:]] for j in range(p + 2))
            for s in itertools.combinations(range(NV), p + 2)}


def cup(a, p, b, q, i, mutate=None):
    n = p + q - i
    if i < 0 or i > min(p, q) or n > NV - 1:
        return None
    L = labels(p, q, i, mutate)
    return {s: sum(sg * a[tuple(s[x] for x in A)] * b[tuple(s[x] for x in B)] for sg, A, B in L)
            for s in itertools.combinations(range(NV), n + 1)}


def steenrod_holds(mutate=None, seed=7):
    rng = random.Random(seed)
    cases = 0
    for p in range(NV - 1):
        for q in range(NV - 1):
            for i in range(min(p, q) + 1):
                n = p + q - i
                if n + 1 > NV - 1:
                    continue
                a, b = rnd(p, rng), rnd(q, rng)
                lhs = cob(cup(a, p, b, q, i, mutate), n)
                terms = [(1, cup(cob(a, p), p + 1, b, q, i, mutate)),
                         ((-1) ** p, cup(a, p, cob(b, q), q + 1, i, mutate)),
                         ((-1) ** n, cup(a, p, b, q, i - 1, mutate)),
                         ((-1) ** (p * q + p + q), cup(b, q, a, p, i - 1, mutate))]
                for s, v in lhs.items():
                    if v != sum(c * t[s] for c, t in terms if t is not None):
                        return False, (p, q, i)
                cases += 1
    return True, cases


def rev(w, p):
    return {s: (-1) ** comb(p + 1, 2) * w[tuple(sorted(NV - 1 - x for x in s))] for s in w}


def main():
    assert all(sg == 1 for p in range(NV) for q in range(NV - p) for sg, _, _ in labels(p, q, 0)), \
        "cup_0 is not Alexander-Whitney"
    ok, info = steenrod_holds()
    assert ok, f"Steenrod identity fails at (p,q,i)={info}"
    print(f"NV={NV}: cup_0 = Alexander-Whitney; Steenrod identity exact in {info} (p,q,i) cases")

    controls = {
        "flip all i=1 terms": lambda p, q, i, n, J: i == 1,
        "flip i=2 terms with j_0=0": lambda p, q, i, n, J: i == 2 and J[0] == 0,
        "flip i=3 terms with j_0=1": lambda p, q, i, n, J: i == 3 and J[0] == 1,
    }
    for name, m in controls.items():
        ok, info = steenrod_holds(mutate=m)
        assert not ok, f"negative control '{name}' unexpectedly passed"
        print(f"negative control '{name}': identity fails at {info}, as it must")

    rng = random.Random(3)
    odd_pass, odd_fail = [], []
    for p in range(6):
        for q in range(6):
            for i in range(min(p, q) + 1):
                if p + q - i > NV - 1:
                    continue
                a, b = rnd(p, rng), rnd(q, rng)
                c = (-1) ** (p * q + i + comb(i + 1, 2))
                lhs = rev(cup(a, p, b, q, i), p + q - i)
                rhs = cup(rev(b, q), q, rev(a, p), p, i)
                holds = all(lhs[s] == c * rhs[s] for s in lhs)
                if i % 2 == 0:
                    assert holds, f"even-i reversal fails at {(p, q, i)}"
                else:
                    (odd_pass if holds else odd_fail).append((p, q, i))
    assert all(p == q == i for p, q, i in odd_pass), f"unexpected odd-i reversal successes {odd_pass}"
    assert odd_fail, "odd-i reversal failures expected"
    print(f"order reversal: exact for even i; odd i holds only on the diagonal {odd_pass}, "
          f"fails in {len(odd_fail)} other cases")

    for p in range(NV):
        a, b = rnd(p, rng), rnd(p, rng)
        prod = cup(a, p, b, p, p)
        assert all(prod[s] == a[s] * b[s] for s in prod), f"a u_p b != ab at p={p}"
    print("pointwise top product: a u_p b = ab for every p")


if __name__ == "__main__":
    main()
