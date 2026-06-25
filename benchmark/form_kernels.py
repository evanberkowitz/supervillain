#!/usr/bin/env python
# Performance benchmark for the numba Form kernels vs the numpy reference.
# Not a pytest: a dev tool to keep the speedup claims and parallel crossover
# honest.  Run: uv run python benchmark/form_kernels.py
import timeit
from supervillain.lattice import Lattice, d, delta
from supervillain.lattice import reference as ref

# (production op, reference op, input degree picker) for each operator.
_OPS = {
    "d":          (lambda f: d(f),            ref.reference_d,          lambda D: max(0, D - 2)),
    "delta":      (lambda f: delta(f),        ref.reference_delta,      lambda D: 2),
    "face_sum":   (lambda f: f.face_sum(),    ref.reference_face_sum,   lambda D: 2),
    "coface_sum": (lambda f: f.coface_sum(),  ref.reference_coface_sum, lambda D: max(0, D - 2)),
}


def _time(fn, n):
    fn()  # warm up / trigger numba compilation
    return timeit.timeit(fn, number=n) / n * 1e6  # microseconds


def sweep(Ds=(4,), Ns=(7, 9, 11, 13)):
    rows = []
    for D in Ds:
        for N in Ns:
            L = Lattice(D=D, N=N)
            n = max(50, int(2e6 / N ** D))
            for op, (prod, refop, pick) in _OPS.items():
                p = pick(D)
                f = L.random(p)
                k_us = _time(lambda: prod(f), n)
                r_us = _time(lambda: refop(f), max(20, n // 4))
                rows.append({"D": D, "N": N, "op": op, "sites": N ** D,
                             "ref_us": r_us, "kernel_us": k_us, "speedup": r_us / k_us})
    return rows


if __name__ == "__main__":
    print(f"{'D':>2} {'N':>3} {'sites':>8} {'op':>11} {'ref_us':>9} {'kernel_us':>10} {'speedup':>8}")
    for r in sweep():
        print(f"{r['D']:>2} {r['N']:>3} {r['sites']:>8} {r['op']:>11} "
              f"{r['ref_us']:>9.1f} {r['kernel_us']:>10.1f} {r['speedup']:>7.1f}x")
