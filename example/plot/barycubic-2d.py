#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import supervillain

parser = supervillain.cli.ArgumentParser()
parser.add_argument('--pdf', type=str, default='')
args = parser.parse_args()

# Two unit squares of the D=2 hypercubic lattice, subdivided barycentrically.
# Every fine triangle is anchored at the site x of the square that contains it, and its
# vertices are 2x + (the lifted local vertices of its label), in doubled coordinates.
L = supervillain.lattice.BarycubicLattice(2, 4)
anchors = [np.array([0, 0]), np.array([1, 0])]
triangles = L.local_vertices(2)   # (f_2, 3, 2) = (8, 3, 2): rank-ordered, doubled coordinates

rank_colors = ('tab:blue', 'tab:orange', 'tab:green')
rank_names = ('rank 0: sites', 'rank 1: link midpoints', 'rank 2: plaquette centers')

fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

# The fine triangles; the coarse squares are drawn on top in black.
for x in anchors:
    for V in triangles:
        corners = (2 * x + V) / 2          # back to lattice units
        ax.add_patch(Polygon(corners, closed=True, facecolor='none', edgecolor='lightgray', linewidth=1))
for x in anchors:
    ax.add_patch(Polygon(x + np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), closed=True,
                         facecolor='none', edgecolor='black', linewidth=2))

# The fine vertices, colored by rank = number of odd doubled coordinates.
points = {tuple(2 * x + v) for x in anchors for V in triangles for v in V}
for xi in points:
    r = int(np.sum(np.array(xi) % 2))
    ax.plot(*(np.array(xi) / 2), 'o', color=rank_colors[r], markersize=11, zorder=3)
for r, name in enumerate(rank_names):
    ax.plot([], [], 'o', color=rank_colors[r], markersize=11, label=name)

# One fine triangle, with its vertices in rank order v0 -> v1 -> v2.
label = L.label_index[2][((0b00, 0b01, 0b11), (-1, -1))]
highlight = (2 * anchors[0] + triangles[label]) / 2
ax.add_patch(Polygon(highlight, closed=True, facecolor='gold', alpha=0.5, edgecolor='black', linewidth=1.5))
for a in range(2):
    ax.annotate('', xy=highlight[a + 1], xytext=highlight[a],
                arrowprops=dict(arrowstyle='->', color='black', linewidth=2, shrinkA=8, shrinkB=8))
for a, v in enumerate(highlight):
    ax.annotate(f'$v_{a}$', xy=v, xytext=v + np.array([0.04, 0.05]), fontsize=16, zorder=4)

ax.set_xlim(-0.15, 2.15)
ax.set_ylim(-0.15, 1.15)
ax.set_aspect('equal')
ax.set_xticks(np.arange(0, 2.5, 0.5), labels=[f'{t:g}\n$\\xi_0={int(2*t)}$' for t in np.arange(0, 2.5, 0.5)])
ax.set_yticks(np.arange(0, 1.5, 0.5), labels=[f'$\\xi_1={int(2*t)}$  {t:g}' for t in np.arange(0, 1.5, 0.5)])
ax.set_xlabel('0th direction')
ax.set_ylabel('1st direction')
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)

fig.tight_layout()

if args.pdf:
    fig.savefig(args.pdf)
else:
    plt.show()
