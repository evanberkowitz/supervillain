#!/usr/bin/env python

r"""
Money plot for ``kappa_scan.py`` output: does long-range order trade hands between
``Spin_Spin`` and ``Intersection_Intersection`` as κ falls?

    uv run python kappa_scan_plot.py --dir scan/N8 [--worm free] [--figure scan-N8.pdf]
"""

import argparse
import glob

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

import supervillain

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='scan/N8')
parser.add_argument('--worm', default='free')
parser.add_argument('--figure', default=False)
args = parser.parse_args()

CORRELATORS = ('Spin_Spin_Normalized', 'Intersection_Intersection_Normalized')

points = []
for path in sorted(glob.glob(f'{args.dir}/kappa*-{args.worm}.h5')):
    with h5.File(path, 'r') as f:
        entry = {'kappa': f.attrs['kappa'], 'N': f.attrs['N']}
        for name in CORRELATORS:
            for key in ('onaxis', 'onaxis_err', 'antipode', 'antipode_err'):
                entry[f'{name}/{key}'] = float(f[f'plateaus/{name}/{key}'][()])
            entry[f'{name}/mean'] = f[f'correlators/{name}/mean'][()]
            entry[f'{name}/err'] = f[f'correlators/{name}/err'][()]
        points.append(entry)
points.sort(key=lambda x: x['kappa'])
assert points, f'no {args.worm} h5 files under {args.dir}'
N = int(points[0]['N'])
L = supervillain.lattice.Lattice(4, N)
kappas = np.array([p['kappa'] for p in points])

fig, (ax_plateau, ax_shape) = plt.subplots(2, 1, figsize=(7, 8))

for name, color in zip(CORRELATORS, ('C0', 'C1')):
    short = name.split('_Normalized')[0]
    for key, marker in (('onaxis', 'o'), ('antipode', 's')):
        y = np.array([p[f'{name}/{key}'] for p in points])
        dy = np.array([p[f'{name}/{key}_err'] for p in points])
        ax_plateau.errorbar(kappas, y, dy, label=f'{short} {key}', color=color,
                            marker=marker, linestyle='-' if key == 'onaxis' else '--',
                            markerfacecolor='none')
ax_plateau.set_xscale('log')
ax_plateau.set_xlabel('κ')
ax_plateau.set_ylabel('long-distance plateau')
ax_plateau.axhline(0, color='gray', linewidth=0.5)
ax_plateau.legend()

dx = L.linearize(L.R_squared) ** 0.5
picks = [points[0], points[len(points) // 2], points[-1]]
for p in picks:
    ax_shape.errorbar(dx, L.linearize(p['Intersection_Intersection_Normalized/mean']),
                      L.linearize(p['Intersection_Intersection_Normalized/err']),
                      linestyle='none', marker='o', markerfacecolor='none',
                      label=f"κ={p['kappa']}")
ax_shape.set_yscale('log')
ax_shape.set_xlabel('Δx')
ax_shape.set_ylabel('Intersection_Intersection_Normalized')
ax_shape.legend()

fig.suptitle(f'NoIntersections D=4 N={N} ({args.worm} worm)')
fig.tight_layout()
if args.figure:
    fig.savefig(args.figure)
    print(f'wrote {args.figure}')
else:
    plt.show()
