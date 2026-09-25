#!/usr/bin/env python3
"""Compare two exports of the barycubic complex, matching simplices by canonical label.

Each export is a .npz file in the layout written by write_export.py, or a directory whose .npz
files together hold that layout.  The two exports may come from different implementations and
list simplices in different orders: every row is matched through its canonical label, the
rank-ordered vertex list in doubled coordinates reduced to [0, 2N).  Only numpy is needed.

For every quantity it prints OK or the first mismatch:

  complex/{p}/labels        the two exports name the same set of simplices
  complex/{p}/d             the same coboundary entries, keyed by (row label, column label)
  complex/{p}/weights       the same Hodge weight per simplex, to --rtol
  complex/{p}/orientation   the same orientation per top simplex
  cup/{p}/{q}/{i}/a, b      the same reproducible factors per simplex
  cup/{p}/{q}/{i}/values    the same a cup_i b per simplex, exactly

Usage:  python compare_exports.py FIRST SECOND [--rtol 1e-12]
Exit status 0 when everything agrees, 1 otherwise.

The logic is importable: compare(A, B, rtol) takes two dicts of arrays in the layout above and
returns the report, agrees(report) says whether it is all OK, and compare_csr compares two
coboundary matrices given as (data, indices, indptr, shape).
"""
import argparse
import pathlib
import sys

import numpy as np


def load(path):
    path = pathlib.Path(path)
    files = sorted(path.glob('*.npz')) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f'no .npz files in {path}')
    arrays = {}
    for f in files:
        with np.load(f) as data:
            arrays.update({k: data[k] for k in data.files})
    return arrays


def keys(labels):
    '''One hashable key per simplex: the bytes of its canonical label as int64.'''
    labels = np.ascontiguousarray(np.asarray(labels, dtype=np.int64)).reshape(len(labels), -1)
    return [row.tobytes() for row in labels]


def show(key, width):
    '''Render a key back as a vertex list.'''
    coords = np.frombuffer(key, dtype=np.int64).reshape(-1, width)
    return '(' + ' '.join('[' + ','.join(str(c) for c in v) + ']' for v in coords) + ')'


class Matching:
    '''Row correspondence between two label lists, or the first label present on only one side.'''

    def __init__(self, first, second, D):
        self.D = D
        self.first, self.second = keys(first), keys(second)
        where = {k: j for j, k in enumerate(self.second)}
        self.problem = None
        if len(where) != len(self.second) or len(set(self.first)) != len(self.first):
            self.problem = 'duplicate canonical labels'
            return
        missing = [k for k in self.first if k not in where]
        extra = set(self.second) - set(self.first)
        if missing:
            self.problem = f'{show(missing[0], D)} only in the first export'
        elif extra:
            self.problem = f'{show(next(iter(extra)), D)} only in the second export'
        else:
            self.to_second = np.array([where[k] for k in self.first], dtype=np.int64)

    def label(self, row):
        return show(self.first[row], self.D)


def first_difference(x, y, rtol=0.0):
    bad = ~np.isclose(x, y, rtol=rtol, atol=0) if rtol else (x != y)
    hits = np.flatnonzero(bad)
    return int(hits[0]) if len(hits) else None


def compare_values(name, match, x, y, rtol=0.0):
    if match.problem:
        return f'{name}: cannot match rows ({match.problem})'
    y = np.asarray(y)[match.to_second]
    x = np.asarray(x)
    if x.shape != y.shape:
        return f'{name}: shapes {x.shape} and {y.shape}'
    r = first_difference(x, y, rtol)
    if r is None:
        return None
    return f'{name}: first mismatch at {match.label(r)}: {x[r]} vs {y[r]}'


def csr_entries(data, indices, indptr, shape, expected_shape):
    r"""
    The nonzero entries of a CSR matrix as sorted flat keys row * ncols + col and their values.

    The arrays are validated first (shape, monotone indptr, index bounds), because a malformed
    export must be reported, not silently read.  Duplicate coordinates, which CSR allows, are
    summed, and entries that are exactly zero after summing are dropped, so two matrices with
    the same meaning compare equal whatever their storage.  Raises ValueError when invalid.
    """
    data, indices, indptr = np.asarray(data), np.asarray(indices), np.asarray(indptr)
    shape = tuple(int(n) for n in np.asarray(shape).reshape(-1))
    if shape != tuple(expected_shape):
        raise ValueError(f'shape {shape}, but the labels give {tuple(expected_shape)}')
    nrows, ncols = shape
    if indptr.shape != (nrows + 1,) or indptr[0] != 0 or np.any(np.diff(indptr) < 0):
        raise ValueError('indptr is not a valid CSR row pointer for the shape')
    if indptr[-1] != len(indices) or len(indices) != len(data):
        raise ValueError(f'indptr ends at {indptr[-1]} but there are {len(indices)} indices '
                         f'and {len(data)} values')
    if len(indices) and (indices.min() < 0 or indices.max() >= ncols):
        raise ValueError('a column index is out of range')
    rows = np.repeat(np.arange(nrows, dtype=np.int64), np.diff(indptr))
    flat = rows * ncols + indices.astype(np.int64)
    keys, position = np.unique(flat, return_inverse=True)
    values = np.zeros(len(keys), dtype=data.dtype if len(data) else float)
    np.add.at(values, position.reshape(-1), data)
    keep = values != 0
    return keys[keep], values[keep]


def compare_csr(name, first, second, rows_match, cols_match):
    r"""
    Compare two coboundary matrices given as (data, indices, indptr, shape) tuples, the first in
    the row and column order of rows_match.first / cols_match.first, the second in the order of
    the second export.  Returns None when they agree, otherwise a one-line description of the
    first mismatch (or of the first problem found in the arrays).
    """
    if rows_match.problem or cols_match.problem:
        return f'{name}: cannot match rows ({rows_match.problem or cols_match.problem})'
    nrows, ncols = len(rows_match.first), len(cols_match.first)
    try:
        kA, vA = csr_entries(*first, expected_shape=(nrows, ncols))
    except ValueError as error:
        return f'{name}: invalid matrix in the first export: {error}'
    try:
        kB, vB = csr_entries(*second, expected_shape=(nrows, ncols))
    except ValueError as error:
        return f'{name}: invalid matrix in the second export: {error}'
    # Re-express the second export's entries in the first export's row and column order.
    to_first_rows = np.empty_like(rows_match.to_second)
    to_first_rows[rows_match.to_second] = np.arange(nrows)
    to_first_cols = np.empty_like(cols_match.to_second)
    to_first_cols[cols_match.to_second] = np.arange(ncols)
    rB, cB = np.divmod(kB, ncols)
    kB = to_first_rows[rB] * ncols + to_first_cols[cB]
    order = np.argsort(kB)
    kB, vB = kB[order], vB[order]
    keys = np.union1d(kA, kB)
    xA = np.zeros(len(keys), dtype=np.result_type(vA, vB))
    xB = np.zeros(len(keys), dtype=xA.dtype)
    xA[np.searchsorted(keys, kA)] = vA
    xB[np.searchsorted(keys, kB)] = vB
    bad = np.flatnonzero(xA != xB)
    if not len(bad):
        return None
    r, c = divmod(int(keys[bad[0]]), ncols)
    return (f'{name}: first mismatch at row {rows_match.label(r)}, column '
            f'{cols_match.label(c)}: {xA[bad[0]]} vs {xB[bad[0]]}')


def csr_arrays(arrays, prefix):
    return tuple(arrays[f'{prefix}/{k}'] for k in ('data', 'indices', 'indptr', 'shape'))


def compare_d(name, A, B, p, rows_match, cols_match):
    prefix = f'complex/{p}/d'
    try:
        first, second = csr_arrays(A, prefix), csr_arrays(B, prefix)
    except KeyError as missing:
        return f'{name}: missing array {missing}'
    return compare_csr(name, first, second, rows_match, cols_match)


def agrees(report):
    """Whether every line of a report from compare() says OK."""
    return all(line.endswith(': OK') for line in report)


def compare(A, B, rtol=1e-12):
    r"""
    Compare two exports given as dicts of arrays (as loaded by load(), or built in memory) and
    return the report: one line per quantity, ending in ': OK' or describing the first mismatch.
    """
    report = []
    D, N = int(A['D']), int(A['N'])
    if (D, N) != (int(B['D']), int(B['N'])):
        return [f'(D, N) differ: {(D, N)} vs {(int(B["D"]), int(B["N"]))}']
    if int(A.get('seed', 0)) != int(B.get('seed', 0)):
        report.append(f'seed differs: {int(A["seed"])} vs {int(B["seed"])}; cup factors will differ')

    match = {}
    for p in range(D + 1):
        name = f'complex/{p}/labels'
        match[p] = Matching(A[name], B[name], D)
        report.append(f'{name}: ' + (match[p].problem or 'OK'))
    for p in range(D + 1):
        report.append(compare_values(f'complex/{p}/weights', match[p], A[f'complex/{p}/weights'],
                                     B[f'complex/{p}/weights'], rtol) or f'complex/{p}/weights: OK')
        if p < D:
            report.append(compare_d(f'complex/{p}/d', A, B, p, match[p + 1], match[p]) or f'complex/{p}/d: OK')
    name = f'complex/{D}/orientation'
    report.append(compare_values(name, match[D], A[name], B[name]) or f'{name}: OK')

    cups = sorted({tuple(int(x) for x in k.split('/')[1:4]) for k in A if k.startswith('cup/')})
    other = {tuple(int(x) for x in k.split('/')[1:4]) for k in B if k.startswith('cup/')}
    for p, q, i in cups:
        prefix = f'cup/{p}/{q}/{i}'
        if (p, q, i) not in other:
            report.append(f'{prefix}: only in the first export')
            continue
        out = Matching(A[f'{prefix}/labels'], B[f'{prefix}/labels'], D)
        report.append(compare_values(f'{prefix}/a', match[p], A[f'{prefix}/a'], B[f'{prefix}/a']) or f'{prefix}/a: OK')
        report.append(compare_values(f'{prefix}/b', match[q], A[f'{prefix}/b'], B[f'{prefix}/b']) or f'{prefix}/b: OK')
        report.append(compare_values(f'{prefix}/values', out, A[f'{prefix}/values'], B[f'{prefix}/values'])
                      or f'{prefix}/values: OK')
    for p, q, i in sorted(other - set(cups)):
        report.append(f'cup/{p}/{q}/{i}: only in the second export')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('first')
    parser.add_argument('second')
    parser.add_argument('--rtol', type=float, default=1e-12, help='relative tolerance for the weights')
    args = parser.parse_args()
    report = compare(load(args.first), load(args.second), args.rtol)
    bad = [line for line in report if not line.endswith(': OK')]
    for line in report:
        print(line)
    print(f'{len(report) - len(bad)} of {len(report)} quantities agree')
    sys.exit(0 if agrees(report) else 1)


if __name__ == '__main__':
    main()
