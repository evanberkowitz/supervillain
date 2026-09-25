#!/usr/bin/env python3
"""Write this implementation's interop export of the barycubic complex to one .npz file.

The file holds, at one (D, N):

  * for every degree p, BarycubicLattice.export_complex(p): the canonical labels, the coboundary
    d as CSR arrays (data, indices, indptr, shape), the Hodge weights, and for p = D the
    orientation;
  * for every legal (p, q, i), BarycubicLattice.export_cup(p, q, i, seed): the product's
    canonical labels, the reproducible integer factors a and b, and the values of a cup_i b.

Key layout (all arrays):
    D, N, seed
    complex/{p}/labels, complex/{p}/weights
    complex/{p}/d/data, .../d/indices, .../d/indptr, .../d/shape        (p < D)
    complex/{D}/orientation
    cup/{p}/{q}/{i}/labels, .../a, .../b, .../values

Another implementation only has to write the same keys, in its own row order, for
compare_exports.py to diff the two.  The factors a and b are in the canonical-label order of
complex/{p}/labels and complex/{q}/labels of the same file.

Usage:  python write_export.py D N OUT.npz [--seed SEED]
"""
import argparse

import numpy as np


def export(D, N, seed=0):
    from supervillain.lattice.barycubic import BarycubicLattice

    L = BarycubicLattice(D, N)
    arrays = {'D': np.array(D), 'N': np.array(N), 'seed': np.array(seed)}
    for p in range(D + 1):
        e = L.export_complex(p)
        arrays[f'complex/{p}/labels'] = e['labels']
        arrays[f'complex/{p}/weights'] = e['weights']
        if 'd' in e:
            d = e['d'].tocsr()
            arrays[f'complex/{p}/d/data'] = d.data
            arrays[f'complex/{p}/d/indices'] = d.indices
            arrays[f'complex/{p}/d/indptr'] = d.indptr
            arrays[f'complex/{p}/d/shape'] = np.array(d.shape)
        if 'orientation' in e:
            arrays[f'complex/{p}/orientation'] = e['orientation']
    for p in range(D + 1):
        for q in range(D + 1):
            for i in range(min(p, q) + 1):
                if p + q - i > D:
                    continue
                e = L.export_cup(p, q, i, seed=seed)
                for key, value in e.items():
                    arrays[f'cup/{p}/{q}/{i}/{key}'] = value
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('D', type=int)
    parser.add_argument('N', type=int)
    parser.add_argument('out', help='output .npz file')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    arrays = export(args.D, args.N, args.seed)
    np.savez_compressed(args.out, **arrays)
    print(f'wrote {len(arrays)} arrays to {args.out}')


if __name__ == '__main__':
    main()
