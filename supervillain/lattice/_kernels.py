#!/usr/bin/env python
# Numba kernels for the lattice shift-and-accumulate operators.  Operates on
# raw (C, N**D) arrays only — NO Form import (avoids a circular import with
# compact.py).  Combine mode and neighbor direction are baked in as closure
# constants by _make_kernel so the contiguous inner loop is branch-free.

import numba
from numba import prange

PARALLEL_SITE_THRESHOLD = 30_000


def _make_kernel(combine, forward, parallel):
    is_diff = (combine == "diff")
    fwd = forward

    @numba.njit(cache=True, fastmath=True, parallel=parallel)
    def kernel(F, res, table, N, D):
        for t in range(table.shape[0]):
            oi = table[t, 0]; ii = table[t, 1]; e = table[t, 2]; sign = table[t, 3]
            A = N ** e
            B = N ** (D - 1 - e)
            for a in (prange(A) if parallel else range(A)):
                base = a * N * B
                for k in range(N):
                    kn = (k + 1 if k < N - 1 else 0) if fwd else (k - 1 if k > 0 else N - 1)
                    s0 = base + k * B
                    sn = base + kn * B
                    if is_diff and fwd:                      # d
                        for b in range(B):
                            res[oi, s0 + b] += sign * (F[ii, sn + b] - F[ii, s0 + b])
                    elif is_diff:                            # delta
                        for b in range(B):
                            res[oi, s0 + b] -= sign * (F[ii, s0 + b] - F[ii, sn + b])
                    else:                                    # face_sum / coface_sum
                        # TWO separate accumulations, matching the reference's
                        # `result += spatial; result += shifted` order — float
                        # addition is not associative, so the combined
                        # `F[s0] + F[sn]` would diverge from the reference at
                        # machine epsilon and break the bit-exact (`==`) oracle.
                        for b in range(B):
                            res[oi, s0 + b] += F[ii, s0 + b]
                            res[oi, s0 + b] += F[ii, sn + b]
        return res

    return kernel


D_KERNELS      = (_make_kernel("diff", True,  False), _make_kernel("diff", True,  True))
DELTA_KERNELS  = (_make_kernel("diff", False, False), _make_kernel("diff", False, True))
COFACE_KERNELS = (_make_kernel("sum",  True,  False), _make_kernel("sum",  True,  True))
FACE_KERNELS   = (_make_kernel("sum",  False, False), _make_kernel("sum",  False, True))


def select(kernels, sites):
    """Pick the parallel kernel for large lattices, serial otherwise."""
    return kernels[1] if sites >= PARALLEL_SITE_THRESHOLD else kernels[0]
