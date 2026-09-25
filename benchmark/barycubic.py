#!/usr/bin/env python
# Performance benchmark for the barycubic lattice operators, next to the hypercubic d.
# Not a pytest: a dev tool to keep the cost claims in supervillain/barycubic.rst honest.
# Run: uv run python benchmark/barycubic.py
#
# Times are per call and per OUTPUT cell: fine simplices of the output degree for the barycubic
# operators, hypercubic cells of the output degree for compact.d.  The per-D tables and the
# numba kernels are warmed up before timing, so these are steady-state costs.
import timeit
import numpy as np
from supervillain.lattice import Lattice, compact
from supervillain.lattice import barycubic as bc

_PERM, _FLIPS, _SHIFT = (1, 0, 3, 2), (1, -1, 1, -1), (1, 2, 0, 3)

# (operator, input degrees, output degree, callable)
_OPS = {
    "d 1->2":        ((1,),   2, lambda f: bc.d(f[0])),
    "delta 2->1":    ((2,),   1, lambda f: bc.delta(f[0])),
    "laplacian 1":   ((1,),   1, lambda f: bc.laplacian(f[0])),
    "cup(1,1,0)":    ((1, 1), 2, lambda f: bc.cup(f[0], f[1], 0)),
    "cup(2,2,1)":    ((2, 2), 3, lambda f: bc.cup(f[0], f[1], 1)),
    "transform 2":   ((2,),   2, lambda f: bc.transform(f[0], _PERM, _FLIPS, _SHIFT)),
    "half_shift 2":  ((2,),   2, lambda f: bc.half_shift(f[0])),
}


def _time(fn, n):
    fn()  # warm up: build the per-D tables and compile the kernels
    return timeit.timeit(fn, number=n) / n * 1e6  # microseconds


def sweep(D=4, Ns=(4, 8)):
    rows = []
    for N in Ns:
        L = bc.BarycubicLattice(D, N)
        H = Lattice(D=D, N=N)
        n = max(5, int(2e6 / (L.cells_of_degree[2])))
        h = H.random(1)
        us = _time(lambda: compact.d(h), max(50, 20 * n))
        rows.append({"N": N, "op": "hypercubic d 1->2", "cells": H.cells_of_degree[2], "us": us})
        for op, (degrees, out, call) in _OPS.items():
            forms = [L.random(p) for p in degrees]
            us = _time(lambda: call(forms), n)
            rows.append({"N": N, "op": op, "cells": L.cells_of_degree[out], "us": us})
    return rows


if __name__ == "__main__":
    print(f"{'N':>3} {'op':>18} {'out cells':>10} {'ms':>9} {'ns/cell':>8}")
    for r in sweep():
        print(f"{r['N']:>3} {r['op']:>18} {r['cells']:>10} {r['us'] / 1e3:>9.2f} "
              f"{r['us'] * 1e3 / r['cells']:>8.1f}")
