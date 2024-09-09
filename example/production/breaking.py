#!/usr/bin/env python

from collections import deque
from itertools import product
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon

import supervillain
import steps
import results


import logging
logger = logging.getLogger(__name__)

# We compute the volume average
#
#    ∑ exp(2πi v/W) / volume
#
# configuration by configuration and make a histogram in the complex plane.
# The histogram lies in the polygon whose points are the Wth roots of unity.

def plot_z3(W, N, ensembles):
    # When W=3 the possible points are laid out in a triangular lattice,
    # and each possibility lies in the center of a hexagonal Voronoi cell.
    # That makes it natural to use hexbin to make the histogram.
    #
    # There is one catch, which is that hexbin naturally has a vertex rather than an edge at the top,
    # the opposite orientation of the natural real and imaginary parts.
    # The best way to reorient is essentially to rotate the data 90˚ and then rotate the figure back at the end.
    #
    #    https://stackoverflow.com/questions/69256188/how-to-get-hexagon-in-matplotlib-hexbin-flat-side-up/
    #
    # We'll rotate the figure when we include it in a paper, so we'll aim for a tall and skinny figure
    # that needs to be rotated by 90˚ clockwise.

    # We'll create one histogram for each ensemble and set aside a little space for the colorbar.

    histogram_size = 8
    colorbar_width = 0.4
    cmap='hot'
    fig, ax = plt.subplots(len(ensembles)+1, 1, figsize=(histogram_size, colorbar_width+len(ensembles)*histogram_size),
                           height_ratios=(8,)*len(ensembles) + (colorbar_width,),
                           )

    for a, (i, E) in zip(ax, enumerate(results.ensembles(ensembles.sort_values(by=['kappa'], ascending=False)))):

        W = E.Action.W
        L = E.Action.Lattice

        rotate = 1j
        c = rotate * np.mean(np.exp(2j*np.pi*E.v/W), axis=(-2,-1))

        vertices = np.exp(2j*np.pi*np.arange(W)/W)
        weights=np.stack(tuple(np.array([x, y, L.sites-x-y]) for x, y in product(range(L.sites), range(L.sites)) if L.sites-x-y >=0))/L.sites

        roots=np.array(((rotate*vertices).real, (rotate*vertices).imag))
        extent=np.array((min(roots[0]), max(roots[0]), min(roots[1]), max(roots[1])+np.sqrt(3)/L.sites))

        # These possibilities can help visualize that we've aligned the hexes correctly
        #possibilities=np.einsum('pa,a->p', weights, rotate*vertices)
        #hb = a.hexbin(possibilities.real, possibilities.imag,
        hb = a.hexbin(c.real, c.imag,
                       gridsize=(L.sites, (1+L.sites)//2),
                       extent=extent,
                       cmap=cmap,
                      )

        #a.scatter(possibilities.real, possibilities.imag, color='black', facecolor='none')

        a.text(-0.75, np.sqrt(3)/4, f'κ={E.Action.kappa}\nτ={E.generator.stride}', fontsize=18, rotation='vertical')
        a.set_aspect('equal')

        # Create a triangular patch and add it to the plot to clip the binning
        polygon = Polygon(np.stack(tuple(np.array([v.real, v.imag]) for v in rotate*vertices)), closed=True, edgecolor='black', fill=False, linewidth=2)
        a.add_patch(polygon)
        hb.set_clip_path(polygon)

        a.axis('off')

    # The colorbar is a bit annoying orientation-wise.
    # High needs to be on the top once rotated, so on the left,
    # which is why we use the _r(eversed) color map
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap+'_r'), cax=ax[-1],
                 orientation='horizontal',
                 ticks=(),
                 #location='bottom',
                 )
    # Also, we want the text to be to the right of the colorbar once rotate,
    # so on top before rotation.  That's easiest to accomplish by having a two-axis plot.
    cbt = ax[-1].twiny()
    cbt.set_xticks(())
    cbt.set_xlabel(r'Increasing Frequency', fontsize=18, rotation=180)
    cbt.xaxis.set_label_coords(.5, 1.5)

    return fig, ax

def visualize(ensembles):
    figs = deque()

    for (W, N), data in ensembles.groupby(['W', 'N']):
        data = data[data['action'] == 'Worldline']

        if W != 3:
            logger.warning(f'Not plotting the breaking for {W=}.')
            continue
        fig, ax = plot_z3(W, N, data)
        ax[0].text(-np.sqrt(3)/2, 1, f'W=3\nN={N}', fontsize=24, rotation='vertical', verticalalignment='bottom',)

        fig.tight_layout()
        figs.append(fig)

    return figs

if __name__ == '__main__':

    import supervillain
    import parallel

    parser = supervillain.cli.ArgumentParser()
    parser.add_argument('input_file', type=supervillain.cli.input_file('input'), default='input.py')
    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument('--pdf', default='', type=str)

    args = parser.parse_args()
    ensembles = args.input_file.ensembles
    if args.parallel:
        import parallel
        ensembles = ensembles.apply(parallel.io_prep, axis=1)

    ensembles = ensembles[ensembles['action']=='Worldline']
    logger.info(ensembles)


    figs = visualize(ensembles)

    if args.pdf:
        results.pdf(args.pdf, figs)
    else:
        plt.show()

