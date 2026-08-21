#!/usr/bin/env python
# Reference (pure-numpy) implementations of the lattice shift-and-accumulate
# operators.  These are the correctness oracle for the numba kernels in
# _kernels.py; keep them simple and obviously-correct, never "optimize" them.

import numpy as np


def reference_d(f):
    lat = f.lattice
    p = f.degree
    if p == lat.D:
        return 0
    result = lat.zeros(p + 1, dtype=f.dtype)
    for out_comp in lat.components[p + 1]:
        out_idx = lat.comp_index[p + 1][out_comp]
        for j, k_j in enumerate(out_comp):
            in_comp = tuple(k for k in out_comp if k != k_j)
            in_idx = lat.comp_index[p][in_comp]
            sign = (-1) ** j
            spatial = f[in_idx]
            fwd_diff = np.roll(spatial, -1, axis=k_j) - spatial
            result[out_idx] += sign * fwd_diff
    return result


def reference_delta(f):
    lat = f.lattice
    p = f.degree
    if p == 0:
        return 0
    result = lat.zeros(p - 1, dtype=f.dtype)
    all_dirs = set(range(lat.D))
    for out_comp in lat.components[p - 1]:
        out_idx = lat.comp_index[p - 1][out_comp]
        M_set = set(out_comp)
        for e in sorted(all_dirs - M_set):
            j = sum(1 for m in out_comp if m < e)
            sign = (-1) ** j
            in_comp = tuple(sorted(M_set | {e}))
            in_idx = lat.comp_index[p][in_comp]
            spatial = f[in_idx]
            bwd_diff = spatial - np.roll(spatial, +1, axis=e)
            result[out_idx] -= sign * bwd_diff
    return result


def reference_face_sum(f):
    lat = f.lattice
    p = f.degree
    if p == 0:
        return 0
    result = lat.zeros(p - 1, dtype=f.dtype)
    all_dirs = set(range(lat.D))
    for M_comp in lat.components[p - 1]:
        out_idx = lat.comp_index[p - 1][M_comp]
        M_set = set(M_comp)
        for e in sorted(all_dirs - M_set):
            in_comp = tuple(sorted(M_set | {e}))
            in_idx = lat.comp_index[p][in_comp]
            spatial = f[in_idx]
            result[out_idx] += spatial
            result[out_idx] += np.roll(spatial, +1, axis=e)
    return result


def reference_coface_sum(f):
    lat = f.lattice
    p = f.degree
    if p == lat.D:
        return 0
    result = lat.zeros(p + 1, dtype=f.dtype)
    for O_comp in lat.components[p + 1]:
        out_idx = lat.comp_index[p + 1][O_comp]
        for j, k_j in enumerate(O_comp):
            in_comp = tuple(k for k in O_comp if k != k_j)
            in_idx = lat.comp_index[p][in_comp]
            spatial = f[in_idx]
            result[out_idx] += spatial
            result[out_idx] += np.roll(spatial, -1, axis=k_j)
    return result
