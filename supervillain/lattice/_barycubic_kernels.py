#!/usr/bin/env python
# Numba kernels for the barycubic lattice operators (supervillain/lattice/barycubic.py).
#
# Every kernel works on raw C-contiguous arrays whose leading axis is the label and whose
# second axis is the flattened site, shape (labels, N**D).  A spatial offset never appears
# directly: the caller passes `neighbors`, one row per distinct anchor offset, with
# neighbors[k, x] the flattened site x + offset_k (periodic).  Each table row then names the
# neighbor row it reads through, so one kernel handles every offset, and no rolled copy of the
# input is ever materialized.  Outputs are accumulated in place (+=), so callers can sum
# several operators into one preallocated array.
#
# The kernels are serial on purpose.  Rows are sorted by output label, so a prange over output
# labels would be race-free, but Monte Carlo chains usually run one per core already; a
# parallel variant should be a separate function (numba's on-disk cache keys dispatchers by
# function, so two dispatchers of one function with different flags would collide).
#
# No imports from the lattice package, to avoid a circular import.

import numba


@numba.njit(cache=True)
def incidence(out, src, out_label, in_label, neighbor_row, coefficient, neighbors):
    r'''out[o, x] += c * src[i, neighbors[k, x]] for every row (o, i, k, c).'''
    S = out.shape[1]
    for r in range(out_label.shape[0]):
        o = out_label[r]
        i = in_label[r]
        c = coefficient[r]
        nb = neighbors[neighbor_row[r]]
        for x in range(S):
            out[o, x] += c * src[i, nb[x]]


@numba.njit(cache=True)
def contraction(out, a, b, out_label, a_label, a_row, b_label, b_row, sign, neighbors):
    r'''out[o, x] += s * a[i, neighbors[ka, x]] * b[j, neighbors[kb, x]] for every row.'''
    S = out.shape[1]
    for r in range(out_label.shape[0]):
        o = out_label[r]
        i = a_label[r]
        j = b_label[r]
        s = sign[r]
        na = neighbors[a_row[r]]
        nb = neighbors[b_row[r]]
        for x in range(S):
            out[o, x] += s * a[i, na[x]] * b[j, nb[x]]


@numba.njit(cache=True)
def gather(out, src, image, image_row, sites):
    r'''out[l, x] = src[image[l], sites[image_row[l], x]] for every output label l.'''
    S = out.shape[1]
    for l in range(out.shape[0]):
        i = image[l]
        st = sites[image_row[l]]
        for x in range(S):
            out[l, x] = src[i, st[x]]
