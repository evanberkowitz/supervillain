#!/usr/bin/env python

import matplotlib.colors as colors
import numpy as np

from supervillain.lattice.compact import Lattice, push


class Lattice2D(Lattice):
    r'''
    A two-dimensional square lattice — a thin wrapper around
    :class:`~supervillain.lattice.Lattice` with ``D=2``.

    All lattice machinery (forms, exterior derivative, codifferential,
    Fourier transforms, symmetrize, linearize/coordinatize, …) is inherited
    from :class:`~supervillain.lattice.Lattice`.  ``Lattice2D`` adds only the
    :py:meth:`plot_form` visualisation helper and the ``nt``/``nx`` aliases.

    Parameters
    ----------
    n: int
        The number of sites on a side.
    '''

    def __init__(self, n):
        super().__init__(D=2, N=n)

    @classmethod
    def from_h5(cls, group, strict=True, _top=True):
        from supervillain.h5 import Data
        N = Data.read(group['N'], strict)
        return cls(N)

    @property
    def plaquettes(self):
        """Number of plaquettes (2-cells): $C(2,2) N^2 = N^2$."""
        return self.cells_of_degree[2]

    @property
    def nt(self):
        """Temporal extent N."""
        return self.N

    @property
    def nx(self):
        """Spatial extent N."""
        return self.N

    @property
    def t(self):
        """FFT-convention t-coordinates (first direction), shape (N,)."""
        return self._coord_1d

    @property
    def x(self):
        """FFT-convention x-coordinates (second direction), shape (N,)."""
        return self._coord_1d

    @property
    def T(self):
        """Array of shape (N, N) with the t-coordinate at each site."""
        return np.tile(self.t, (self.N, 1)).T

    @property
    def X(self):
        """Array of shape (N, N) with the x-coordinate at each site."""
        return np.tile(self.x, (self.N, 1))

    def __str__(self):
        return f'Lattice2D({self.nt},{self.nx})'

    def __repr__(self):
        return str(self)

    def plot_form(self, form, axis, label=None, zorder=None,
                  cmap=None, cbar_kw=dict(), norm=colors.CenteredNorm(),
                  pointsize=200, linkwidth=0.025,
                  background='white',
                  markerstyle='o'
                 ):
        r'''
        Plots the form on the axis.  The degree of the form determines
        whether sites, links, or plaquettes are visualized.

        The following figure shows a 0-form plotted on sites, a 1-form on links, and a 2-form on plaquettes.
        See the source for details.

        .. plot:: example/plot/forms.py

        Parameters
        ----------
        form: Form
            The p-form to plot; ``form.degree`` determines the visualization.
        axis: matplotlib.pyplot.axis
            The axis on which to plot.

        Returns
        -------
        matplotlib.image.AxesImage:
            A handle for the data-sensitive part of the plot.

        Other Parameters
        ----------------
        label: string
            If specified, show a colorbar with the title given by the label.
        zorder: float
            If `None` defaults to ``-form.degree`` to layer plaquettes, links, and sites well.
        cmap: string or matplotlib.colors.Colormap
            If a string, it should name `a colormap known to matplotlib <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_.
        cbar_kw: dict
            A dictionary of keyword arguments forwarded to `the colorbar constructor <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar>`_.
        norm: matplotlib.colors.Normalize
            A `matplotlib color normalization <https://matplotlib.org/stable/users/explain/colors/colormapnorms.html>`_.
        '''
        p = form.degree
        zorder = {'zorder': -p if zorder is None else zorder}

        marker_size = {
            's': pointsize,
            'edgecolor': background,
            'linewidth': 2,
            'marker': markerstyle
        }
        marker_color = {
            'cmap': cmap,
            'norm': norm,
        }
        no_arrowhead = {'headwidth': 0, 'headlength': 0, 'headaxislength': 0,}
        linkpadding = {'edgecolor': background, 'linewidth': 4}
        links = {
            'scale_units': 'xy', 'scale': 1,
            'width': linkwidth,
            **no_arrowhead,
            **linkpadding,
            'cmap': cmap,
            'norm': norm,
        }

        if p == 0:
            f = axis.scatter(self.T, self.X, c=form[0], **zorder, **marker_size, **marker_color)

        if p == 1:
            # To get the horizontal links and vertical links to have the same coloring the simplest
            # thing is to combine them into a single quiver.  We'll just completely flatten the 1-form
            # which puts all the 0-direction links first and then all the 1-direction links.
            # So, we need two copies of their starting directions...
            T = np.tile(self.T.flatten(), 2)
            X = np.tile(self.X.flatten(), 2)
            # ... and to say that the first half point in the 0 direction and the latter half in the 1 direction ...
            U = np.concatenate((np.ones_like (self.T.flatten()), np.zeros_like(self.T.flatten())))
            V = np.concatenate((np.zeros_like(self.T.flatten()), np.ones_like (self.T.flatten())))
            # and then we can plot the whole form together.
            f = axis.quiver(T, X, U, V, form.flatten(), **zorder, **links)
            axis.scatter(self.T, self.X, color=background, **zorder, **marker_size)

        if p == 2:
            # Center (0,0) in the plot; imshow expects row-major so transpose.
            data = push(form[0], ((self.nt-1) // 2, (self.nx-1) // 2)).transpose()
            f = axis.imshow(data, **zorder, cmap=cmap,
                        origin='lower', extent=(min(self.t), max(self.t)+1, min(self.x), max(self.x)+1),
                        norm=norm,
                       )
            axis.quiver(self.T, self.X, 1, 0, color='white', **zorder, **links)
            axis.quiver(self.T, self.X, 0, 1, color='white', **zorder, **links)
            axis.scatter(self.T, self.X, color=background, **zorder, **marker_size)
            axis.xaxis.set_zorder(-p)
            axis.yaxis.set_zorder(-p)

        if label:
            cbar = axis.figure.colorbar(f, ax=axis, **cbar_kw)
            cbar.ax.set_title(label)

        axis.set_xlim(min(self.t)-0.5, max(self.t)+1.5)
        axis.set_ylim(min(self.x)-0.5, max(self.x)+1.5)
        axis.set_xlabel('t')
        axis.set_ylabel('x')

        return f


import numba
from numba.experimental import jitclass

@jitclass([
    ('nt', numba.int64),
    ('nx', numba.int64),
    ('dims', numba.int64[:]),
    ('t',  numba.int64[:]),
    ('x',  numba.int64[:]),
    ])
class _Lattice2D:
    r'''
    A numba-accelerated collection of lattice functions.

    .. warning::

       NOT ALL LATTICES METHODS ARE INCLUDED; THEY MAY BE ADDED OVER TIME AS NEEDED.

       Moreover, not all methods have the same signature due to limitations in numba.

       Seriously this is to be used but rarely.  Used only in :class:`~.worldline.ClassicWorm`.

    .. note ::

        Currently `numba jitclasses do not support classmethods <https://numba.readthedocs.io/en/stable/proposals/jit-classes.html>`_.
        In particular, that makes they incompatible with
        our HDF5 infrastructure; they cannot be made :class:`~.ReadWriteable`, for instance.

        Therefore, they should be set as class members; you may need to reconstruct them.
    '''

    def __init__(self, dims):
        self.nt = dims[0]
        self.nx = dims[1]

        self.dims = np.array(dims)

        self.t = self._dimension(self.nt)
        self.x = self._dimension(self.nx)

    def _dimension(self, n):
        return np.concatenate((
            np.arange(0, n // 2 + 1, dtype=np.int64),
            np.arange( - n // 2 + 1, 0, dtype=np.int64),
            ))

    def mod(self, points=np.array([[]])):
        r'''

        .. warning ::
           The return value is unpacked along each dimension.
           This seemed to be a peculiar requirement of numba.

        Parameters
        ----------
        points: 2D np.array
            An n×2 array of points to be modded.

        Returns
        -------
        t: np.array
            The first coordinate of each point modded into the lattice.
        x: np.array
            The second coordinate of each point modded into the lattice.
        '''
        flip = points.T
        t_modded = np.mod(flip[0], self.nt)
        x_modded = np.mod(flip[1], self.nx)
        return self.t[t_modded], self.x[x_modded]

    def neighboring_sites(self, here):
        # east, north, west, south
        return self.mod(here + np.array([[+1,0], [0,+1], [-1,0], [0,-1]]))

    def neighboring_plaquettes(self, here):
        # east, north, west, south
        return self.mod(here + np.array([[0,-1], [+1,0], [0,+1], [-1,0]]))

    def adjacent_links(self, form, site):

        if form == 0:
            # Links to the east, north, west, and south
            t, x = self.neighboring_sites(site)
            east = np.array([t[0], x[0]])
            north= np.array([t[1], x[1]])
            west = np.array([t[2], x[2]])
            south= np.array([t[3], x[3]])

            #     n
            #     |     x
            #   w-h-e   ^
            #     |     |
            #     s     o-->t

            return ((0, site [0], site [1]), # t link to the east
                    (1, site [0], site [1]), # x link to the north
                    (0, west [0], west [1]), # t link to the west
                    (1, south[0], south[1])) # x link to the south

        if form == 2:
            t, x = self.neighboring_plaquettes(site)
            east = np.array([t[0], x[0]])
            north= np.array([t[1], x[1]])
            west = np.array([t[2], x[2]])
            south= np.array([t[3], x[3]])
            #        n
            #       +-+         t
            #      w|h|e        ^
            #       o-+         |
            #        s      x<--o

        return ((0, site [0], site [1]), # t link to the east
                (1, north[0], north[1]), # x link to the north
                (0, west [0], west [1]), # t link to the west
                (1, site [0], site [1])) # x link to the south
