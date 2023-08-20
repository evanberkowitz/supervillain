#!/usr/bin/env python

from functools import cached_property
import numpy as np

from supervillain.h5 import H5able

def _dimension(n):
    '''

    Parameters
    ----------
        n:  int
            size of the dimension

    Returns
    -------
        an FFT-convention-compatible list of coordinates for a dimension of size n,
        ``[0, 1, 2, ... max, min ... -2, -1]``.
    '''
    return np.array(list(range(0, n // 2 + 1)) + list(range( - n // 2 + 1, 0)), dtype=int)

class Lattice2D(H5able):

    def __init__(self, n):
        self.nt = n
        self.nx = n

        self.dims = (self.nx, self.nt)
        r'''
        The dimension sizes in order.

        >>> lattice = Lattice2D(5)
        >>> lattice.dims
        (5, 5)
        '''

        self.dim = len(self.dims)

        self.sites = self.nt * self.nx
        r'''
        The total number of sites.

        >>> lattice = Lattice2D(5)
        >>> lattice.sites
        25
        '''

        self.links = self.dim * self.sites
        self.plaquettes = self.sites # 2D!

        self.t = _dimension(self.nt)
        r'''
        The coordinates in the t direction.

        >>> lattice = Lattice2D(5)
        >>> lattice.t
        array([ 0,  1,  2, -2, -1])
        '''

        self.x = _dimension(self.nx)
        r'''
        The coordinates in the x direction.

        >>> lattice = Lattice2D(5)
        >>> lattice.x
        array([ 0,  1,  2, -2, -1])
        '''

        self.T = np.tile( self.t, (self.nx, 1)).transpose()
        r'''
        An array of size ``dims`` with the t coordinate as a value.

        >>> lattice = Lattice(5)
        >>> lattice.T
        array([[ 0,  0,  0,  0,  0],
               [ 1,  1,  1,  1,  1],
               [ 2,  2,  2,  2,  2],
               [-2, -2, -2, -2, -2],
               [-1, -1, -1, -1, -1]])
        '''
        self.X = np.tile( self.x, (self.nt, 1))
        r'''
        An array of size ``dims`` with the y coordinate as a value.

        >>> lattice = Lattice(5)
        >>> lattice.X
        array([[ 0,  1,  2, -2, -1],
               [ 0,  1,  2, -2, -1],
               [ 0,  1,  2, -2, -1],
               [ 0,  1,  2, -2, -1],
               [ 0,  1,  2, -2, -1]])
        '''

        # We also construct a linearized list of coordinates.
        # The order matches self.X.ravel() and self.Y.ravel()
        self.coordinates = np.stack((self.T.flatten(), self.X.flatten())).transpose()
        '''
        An array of size ``[sites, len(dims)]``.  Each row contains a pair of coordinates.  The order matches ``{T,X}.flatten()``.

        >>> lattice = Lattice(5)
        >>> lattice.coordinates
        >>> lattice.coordinates
        array([[ 0,  0],
               [ 0,  1],
               [ 0,  2],
               [ 0, -2],
               [ 0, -1],
               [ 1,  0],
               [ 1,  1],
               [ 1,  2],
               [ 1, -2],
               [ 1, -1],
               [ 2,  0],
               [ 2,  1],
               [ 2,  2],
               [ 2, -2],
               [ 2, -1],
               [-2,  0],
               [-2,  1],
               [-2,  2],
               [-2, -2],
               [-2, -1],
               [-1,  0],
               [-1,  1],
               [-1,  2],
               [-1, -2],
               [-1, -1]])
        '''

    def __str__(self):
        return f'Lattice2D({self.nt},{self.nx})'

    def __repr__(self):
        return str(self)

    def mod(self, x):
        r'''
        Mod integer coordinates into values on the lattice.

        Parameters
        ----------
            x:  np.ndarray
                Either one coordinate pair of ``.shape==(2,)`` or a set of pairs ``.shape==(*,2)``
                The last dimension should be of size 2.

        Returns
        -------
            np.ndarray
                Each x is identified with an entry of ``coordinates`` by periodic boundary conditions.
                The output is the same shape as the input.
        '''

        modded = np.mod(x, self.dims).transpose()
        return np.stack((
                self.t[modded[0]],
                self.x[modded[1]]
                )).transpose()

    def distance_squared(self, a, b):
        r'''
        .. math::
            \texttt{distance_squared}(a,b) = \left| \texttt{mod}(a - b)\right|^2

        Parameters
        ----------
            a:  np.ndarray
                coordinates that need not be on the lattice
            b:  np.ndarray
                coordinates that need not be on the lattice

        Returns
        -------
            np.ndarray
                The distance between ``a`` and ``b`` on the lattice accounting for the fact that,
                because of periodic boundary conditions, the distance may shorter than naively expected.
                Either ``a`` and ``b`` both hold the same number of coordinate pairs, or one is a singleton.
        '''
        d = self.mod(a-b)
        if d.ndim == 1:
            return np.sum(d**2)

        return np.sum(d**2, axis=(1,))

    def roll(self, data, shift, axes=(-2,-1)):
        
        return np.roll(np.roll(data, shift=shift[0], axis=axes[0]), shift=shift[1], axis=axes[1])

    def coordinatize(self, v, dims=(-1,), center_origin=False):
        r'''
        Unflattens all the dims from a linear superindex to one index for each dimension in ``.dims``.
        
        Parameters
        ----------
            v: np.ndarray
                An array with at least one dimension linearized in space.
            dims: tuple of integers
                The directions you wish to unflatten into a meaningful shape that matches the lattice.
            center_origin: boolean
                If true, each coordinatized dimension is rolled so that the origin is in the center of the two slices.  This is primarily good for making pictures.  :func:`~.linearize` does not provide an inverse of this, because you really should not do it in the middle of a calculation!

                
            
        Returns
        -------
            np.ndarray
                ``v`` but with more, shorter dimensions.  Dimensions specified by ``dims`` are unflattened.
        '''
        
        v_dims  = len(v.shape)

        # We'll build up the new shape by considering each index left-to-right.
        # So, for negative indices we need to mod them by the number of dimensions.
        to_reshape = np.sort(np.remainder(np.array(dims), v_dims))
        
        new_shape = ()
        for i, s in enumerate(v.shape):
            new_shape += ((s,) if i not in to_reshape else self.dims)

        reshaped = v.reshape(new_shape)
        if not center_origin:
            return reshaped
        
        axes = to_reshape + np.arange(len(to_reshape))
        shifts = (self.nt // 2, self.nx // 2)
        for a in axes:
            reshaped = reshaped.roll(shifts, dims=(a,a+1))

        return reshaped

    def linearize(self, v, dims=(-1,)):
        r'''
        Flattens adjacent dimensions of v with shape ``.dims`` into a dimension of size ``.sites``.
        
        Parameters
        ----------
            v:  np.ndarray
            dims: tuples of integers that specify that dimensions *in the result* that come from flattening.
                Modded by the dimension of the resulting array so that any dimension is legal.
                However, one should take care to ensure that no two are the SAME index of the result;
                this causes a RuntimeError.
            
        Returns
        -------
            np.ndarray
                ``v`` but with fewer, larger dimensions

        .. note::
            The ``dims`` parameter may be a bit confusing.  This perhaps-peculiar convention is to make it easier to
            combine with ``coordinatize``.  ``linearize`` and ``coordinatize`` are inverses when they get *the same* 
            dims arguments.

            >>> import numpy as np
            >>> import supervillain
            >>> nx = 5
            >>> dims = (0, -1)
            >>> lattice = supervillain.lattice.Lattice2D(5)
            >>> v = np.arange(nx**(2*3)).reshape(nx**2, nx**2, nx**2)
            >>> u = lattice.coordinatize(v, dims)
            >>> u.shape
            (5, 5, 25, 5, 5)
            >>> w = lattice.linearize(u, dims) # dims indexes into the dimensions of w, not u!
            >>> w.shape
            (25, 25, 25)
            >>> (v == w).all()
            True

        '''

        shape   = v.shape
        v_dims  = len(shape)
        
        dm = set(dims)
        
        future_dims = v_dims - (len(self.dims)-1) * len(dm)
        dm = set(d % future_dims for d in dm)
            
        new_shape = []
        idx = 0
        for i in range(future_dims):
            if i not in dm:
                new_shape += [shape[idx]]
                idx += 1
            else:
                new_shape += [self.sites]
                idx += len(self.dims)
        try:
            return v.reshape(new_shape)
        except RuntimeError as error:
            raise ValueError(f'''
            This happens when two indices to be linearized are accidentally the same.
            For example, for a lattice of size [5,5], if v has .shape [t, x, 5, 5]
            and you linearize(v, (2,-1)) the 2 axis and the -1 axis would refer to
            the same axis in [t, x, 25].
            
            Perhaps this happened with your vector of shape {v.shape} and {dims=}?
            ''') from error

    def form(self, p, count=None, dtype=float):
        r'''
        Parameters
        ----------
            p: integer
                A 2D lattice supports {0, 1, 2}-forms.
            count:
                How many forms to return.
            dtype:
                Data type (float, int, etc.)

        Returns
        -------
            np.ndarray
                If count is none, return an array full of zeros that can hold a p-form.
                If count is not none, return an array that can hold that many p-forms, batch dimension first.

                For example, if we needed to hold 7 forms of each kind for a 3×3 lattice,

                >>> L = Lattice2D(3)
                >>> L.form(0, 7).shape
                (7, 3, 3)
                >>> L.form(1, 7).shape
                (7, 2, 3, 3)
                >>> L.form(2, 7).shape
                (7, 3, 3)

                Notice that the 1-form has an extra dimension compared to the 0 form (because there are 2 links per site in 2 dimensions) and the 2-form has the same shape as sites (which is special to 2D).
                The spacetime dependence is last because :func:`~coordinatize` and :func:`~Lattice2D.linearize` default to the last dimension.
        '''
        if count is None:
            return self.form(p, count=1, dtype=dtype)[0]

        if p == 0:
            return np.zeros((count,) + self.dims, dtype=dtype)
        elif p == 1:
            return np.zeros((count, self.dim) + self.dims, dtype=dtype)
        elif p == 2:
            return np.zeros((count, ) + self.dims, dtype=dtype) # 2D
        else:
            raise ValueError("It's a 2D lattice, you can't have a {p}-form.")

    def d(self, p, form):
        r'''
        The (lattice) exterior derivative.

        Parameters
        ----------
            p: int
                The rank of the form.
            form: np.ndarray
                The data the form.
        '''
        if p == 0:

            return np.stack(tuple(
                np.roll(form, shift=-1, axis=a) - form for a, _ in enumerate(self.dims)
            ))

        elif p == 1:

            return form[0] + np.roll(form[1], shift=-1, axis=1) - np.roll(form[0], shift=-1, axis=0) - form[1]

        elif form == 2:

            return 0

        else:
            raise ValueError("It's a 2D lattice, you can't have a {p}-form.")

    def delta(self, p, form):
        r'''
        Not yet implemented.
        '''
        if p == 0:
            return 0

        elif p == 1:
            raise NotImplemented("δ(1 form) not implemented ... yet.")

        elif p == 2:
            raise NotImplemented("δ(2 form) not implemented ... yet.")

        else:
            raise ValueError("It's a 2D lattice, you can't have a {p}-form.")


