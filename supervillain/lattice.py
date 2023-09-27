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

        self.point_group_operations = np.array((
                    # Matches the order of the (a,b) orbit in docs/D4.rst
                    # That makes it easy to read off the weights
                    ((+1,0),(0,+1)), # identity
                    ((0,+1),(+1,0)), # reflect across y=+x
                    ((0,-1),(+1,0)), # rotate(π/2)
                    ((-1,0),(0,+1)), # reflect across y-axis
                    ((-1,0),(0,-1)), # rotate(π) = inversion
                    ((0,-1),(-1,0)), # reflect across y=-x
                    ((0,+1),(-1,0)), # rotate(3π/2)
                    ((+1,0),(0,-1)), # reflect across x-axis
                ))

        self.point_group_permutations = np.stack(tuple(self._point_group_permutation(o) for o in self.point_group_operations))
        self.point_group_weights = {
            'A1': np.array((+1,+1,+1,+1,+1,+1,+1,+1)) + 0.j,
            'A2': np.array((+1,-1,+1,-1,+1,-1,+1,-1)) + 0.j,
            'B1': np.array((+1,-1,-1,+1,+1,-1,-1,+1)) + 0.j,
            'B2': np.array((+1,+1,-1,-1,+1,+1,-1,-1)) + 0.j,
            ("E", +1): np.array((+1,+1j,+1j,-1,-1,-1j,-1j,+1)),
            ("E", -1): np.array((+1,-1j,-1j,-1,-1,+1j,+1j,+1)),
            ("E'", +1): np.array((+1,-1j,+1j,+1,-1,+1j,-1j,-1)),
            ("E'", -1): np.array((+1,+1j,-1j,+1,-1,-1j,+1j,-1)),
        }
        self.point_group_irreps = self.point_group_weights.keys()
        self.point_group_norm = 1./8

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

        Returns
        -------
            np.ndarray:
                d(0-form) = 1-form, d(1-form) = 2-form, d(2-form) = 0.
        '''
        if p == 0:

            return np.stack(tuple(
                np.roll(form, shift=-1, axis=a) - form for a, _ in enumerate(self.dims)
            ))

        elif p == 1:

            return form[0] + np.roll(form[1], shift=-1, axis=0) - np.roll(form[0], shift=-1, axis=1) - form[1]

        elif p == 2:

            return 0

        else:
            raise ValueError("It's a 2D lattice, you can't have a {p}-form.")

    def delta(self, p, form):
        r'''
        The (lattice) interior derivative / divergence of the p-form.

        Parameters
        ----------
            p: int
                The rank of the form.
            form: np.ndarray
                The data the form.

        Returns
        -------
            np.ndarray:
                δ(2-form) = 1-form, δ(1-form) = 0-form, δ(0-form) = 0.
        '''
        if p == 0:
            return 0

        elif p == 1:
            return self.roll(form[0], (+1,0)) + self.roll(form[1], (0,+1)) - form[0] - form[1]

        elif p == 2:
            return np.stack((
                form - self.roll(form, (0,+1)),
                self.roll(form, (+1,0)) - form,
                ))

        else:
            raise ValueError("It's a 2D lattice, you can't have a {p}-form.")

    δ = delta
    r'''
    Alias for :func:`delta <supervillain.lattice.Lattice2D.delta>`.
    '''

    def t_fft(self, form, axis=-2):
        r'''
        Fourier transforms the form in the time direction,

        .. math ::
            
            F_\nu = \frac{1}{\sqrt{N}} \sum_{t=0}^{N-1} e^{-2\pi i \nu t / N} f_t

        where $\nu$ is the integer frequency, $t$ the integer time coordinate and $N$ is the temporal extent of the lattice.

        Parameters
        ----------
        form: np.array
            The data to transform
        axis: int
            The axis which is the time direction.

        Returns
        -------
        np.array:
            The form is transformed to the frequency domain along the axis.
        '''
        return np.fft.fft(form, axis=axis, norm='ortho')

    def t_ifft(self, form, axis=-2):
        r'''
        Inverse transforms the form in the time direction,

        .. math ::
            
            f_t = \frac{1}{\sqrt{N}} \sum_{\nu=0}^{N-1} e^{+2\pi i \nu t / N} F_\nu

        where $\nu$ is the integer frequency, $t$ the integer time coordinate and $N$ is the temporal extent of the lattice.

        Parameters
        ----------
        form: np.array
            The data to transform
        axis: int
            The axis which is the frequency direction.

        Returns
        -------
        np.array:
            The form is transformed to the time domain along the axis.
        '''
        return np.fft.ifft(form, axis=axis, norm='ortho')

    def t_convolution(self, f, g, axis=-2):
        r'''
        The `convolution <https://en.wikipedia.org/wiki/Convolution>`_ is given by

        .. math ::
            \texttt{t_convolution(f, g)}(t) = (f * g)(t) = \sum_\tau f(\tau) g(t-\tau)

        .. collapse :: The convolution is Fourier accelerated.
            :class: note

            .. math ::

                   \begin{align}
                    (f * g)(t) &= \sum_\tau  f(\tau ) g(t-\tau )
                    \\  &= \sum_{\tau } \left( \frac{1}{\sqrt{N}} \sum_\nu e^{2\pi i \nu \tau  / N} F_\nu \right)\left( \frac{1}{\sqrt{N}} \sum_{\nu'} e^{2\pi i \nu' (t-\tau ) / N} G_{\nu'} \right)
                    \\  &= \sum_{\nu\nu'} e^{2\pi i \nu' t / N} F_\nu G_{\nu'} \left(\frac{1}{N} \sum_{\tau} e^{2\pi i (\nu-\nu') \tau  / N} \right) 
                    \\  &= \sum_{\nu} e^{2\pi i \nu t / N} F_\nu G_\nu
                    \\
                    \texttt{t_convolution(f, g)} &= \sqrt{N} \times \texttt{t_ifft(t_fft(f)t_fft(g))}
                   \end{align}

        Parameters
        ----------
        f: np.array
            A form whose axis is a temporal direction.
        g: np.array
            A form whose axis is a temporal direction.
        axis: int
            The common spatial dimension along which to convolve.

        Returns
        -------
        np.array:
            The convolution of f and g along the axis.


        '''
        return np.sqrt(self.nt) * self.t_ifft( self.t_fft(f, axis=axis) * self.x_fft(g, axis=axis), axis=axis)

    def t_correlation(self, f, g, axis=-1):
        r'''
        The temporal `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ is given by

        .. math ::
            \texttt{t_correlation(f, g)}(t) = (f ⋆ g)(t) = \frac{1}{N} \sum_\tau f(\tau)^* g(\tau-t)

        where $f^*$ is the complex conjugate of $f$.

        .. collapse :: The temporal cross-correlation is Fourier accelerated.
            :class: note

            .. math ::

               \begin{align}
                (f ⋆ g)(t) &= \frac{1}{N} \sum_\tau f(\tau )^* g(\tau -t)
                \\  &= \frac{1}{N} \sum_{\tau } \left( \frac{1}{\sqrt{N}} \sum_\nu e^{2\pi i \nu \tau  / N} F_\nu \right)^* \left( \frac{1}{\sqrt{N}} \sum_{\nu'} e^{2\pi i \nu' (\tau -t) / N} G_{\nu'} \right)
                \\  &= \frac{1}{N} \sum_{\nu\nu'} e^{-2\pi i \nu' t / N} F_\nu^* G_{\nu'} \; \left(\frac{1}{N}\sum_\tau  e^{2\pi i (\nu'-\nu) \tau  / N} = \delta_{\nu'\nu} \right)
                \\  &= \frac{1}{N} \sum_{\nu} e^{-2\pi i \nu t / N} F_\nu^* G_\nu
                \\  &= \frac{1}{\sqrt{N}} \left( \frac{1}{\sqrt{N}} \sum_{\nu} e^{-2\pi i \nu t / N} F_\nu^* G_\nu \right)
                \\
                \texttt{t_correlation(f, g)} &= \texttt{t_fft(conj(t_fft(f))t_fft(g))} / \sqrt{N}
               \end{align}

        .. warning ::
            We have $g(\tau-t)$ whereas `Wikipedia <https://en.wikipedia.org/wiki/Cross-correlation>`_ has $g(\tau+t)$.
            The difference is just the sign on the relative coordinates.

        .. warning ::
            We normalize by the number of time slices, `Wikipedia <https://en.wikipedia.org/wiki/Cross-correlation>`_ does not.


        Parameters
        ----------
        f: np.array
            A form whose axis is a temporal direction.
        g: np.array
            A form whose axis is a temporl direction.
        axis: int
            The common temporal dimension along which to correlate.

        Returns
        -------
        np.array:
            The correlation of f and g along the axis, which is now the relative coordinate.

        '''
        return  self.x_ifft( self.x_fft(f, axis=axis).conj() * self.x_fft(g, axis=axis), axis=axis) / np.sqrt(self.nx)


    def x_fft(self, form, axis=-1):
        r'''
        Fourier transforms the form in the space direction,

        .. math ::
            
            F_k = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} e^{-2\pi i k x / N} f_x

        where $k$ is the integer wavenumber, $x$ the integer space coordinate and $N$ is the spatial volume of the lattice.

        Parameters
        ----------
        form: np.array
            The data to transform
        axis: int
            The axis which is the space direction.

        Returns
        -------
        np.array:
            The form is transformed to the wavenumber domain along the axis.
        '''
        return np.fft.fft(form, axis=axis, norm='ortho')

    def x_ifft(self, form, axis=-1):
        r'''
        Inverse transforms the form in the space direction,

        .. math ::
            
            f_x = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{+2\pi i k x / N} F_k

        where $k$ is the integer wavenumber, $x$ the integer space coordinate and $N$ is the spatial volume of the lattice.

        Parameters
        ----------
        form: np.array
            The data to transform
        axis: int
            The axis which is the wavenumber direction.

        Returns
        -------
        np.array:
            The form is transformed to the space domain along the axis.
        '''
        return np.fft.ifft(form, axis=axis, norm='ortho')

    def x_convolution(self, f, g, axis=-1):
        r'''
        The `convolution <https://en.wikipedia.org/wiki/Convolution>`_

        .. math ::
            (f * g)(x) = \int dy\; f(y) g(x-y)

        on the discretized lattice is given by

        .. math ::
            \texttt{x_convolution(f, g)}(x) = (f * g)(x) = \sum_y f(y) g(x-y)

        .. collapse :: The convolution is Fourier accelerated.
            :class: note

            .. math ::

               \begin{align}
                (f * g)(x) &= \sum_y f(y) g(x-y)
                \\  &= \sum_{y} \left( \frac{1}{\sqrt{N}} \sum_k e^{2\pi i k y / N} F_k \right)\left( \frac{1}{\sqrt{N}} \sum_q e^{2\pi i q (x-y) / N} G_q \right)
                \\  &= \sum_{kq} e^{2\pi i q x / N} F_k G_q \left(\frac{1}{N} \sum_y e^{2\pi i (k-q) y / N} = \delta_{kq} \right)
                \\  &= \sum_{k} e^{2\pi i k x / N} F_k G_k
                \\
                \texttt{x_convolution(f, g)} &= \sqrt{N} \times \texttt{x_ifft(x_fft(f)x_fft(g))}
               \end{align}

        Parameters
        ----------
        f: np.array
            A form whose axis is a spatial direction.
        g: np.array
            A form whose axis is a spatial direction.
        axis: int
            The common spatial dimension along which to convolve.

        Returns
        -------
        np.array:
            The convolution of f and g along the axis.

        '''
        return np.sqrt(self.nx) * self.x_ifft( self.x_fft(f, axis=axis) * self.x_fft(g, axis=axis), axis=axis)

    def x_correlation(self, f, g, axis=-1):
        r'''
        The spatial `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ is given by

        .. math ::
            \texttt{x_correlation(f, g)}(x) = (f ⋆ g)(x) = \frac{1}{N} \sum_y f(y)^* g(y-x)

        where $f^*$ is the complex conjugate of $f$.

        .. collapse :: The spatial cross-correlation is Fourier accelerated.
            :class: note

            .. math ::

               \begin{align}
                (f ⋆ g)(x) &= \frac{1}{N} \sum_y f(y)^* g(y-x)
                \\  &= \frac{1}{N} \sum_{y} \left( \frac{1}{\sqrt{N}} \sum_k e^{2\pi i k y / N} F_k \right)^* \left( \frac{1}{\sqrt{N}} \sum_q e^{2\pi i q (y-x) / N} G_q \right)
                \\  &= \frac{1}{N} \sum_{kq} e^{-2\pi i q x / N} F_k^* G_q \; \left(\frac{1}{N}\sum_y e^{2\pi i (q-k) y / N} = \delta_{qk} \right)
                \\  &= \frac{1}{N} \sum_{k} e^{-2\pi i k x / N} F_k^* G_k
                \\  &= \frac{1}{\sqrt{N}} \left( \frac{1}{\sqrt{N}} \sum_{k} e^{-2\pi i k x / N} F_k^* G_k \right)
                \\
                \texttt{x_correlation(f, g)} &= \texttt{x_fft(conj(x_fft(f))x_fft(g))} / \sqrt{N}
               \end{align}

        .. warning ::
            We have $g(y-x)$ whereas `Wikipedia <https://en.wikipedia.org/wiki/Cross-correlation>`_ has $g(y+x)$.
            The difference is just the sign on the relative coordinates.

        .. warning ::
            We normalize by the spatial volume, `Wikipedia <https://en.wikipedia.org/wiki/Cross-correlation>`_ does not.

        Parameters
        ----------
        f: np.array
            A form whose axis is a spatial direction.
        g: np.array
            A form whose axis is a spatial direction.
        axis: int
            The common spatial dimension along which to correlate.

        Returns
        -------
        np.array:
            The correlation of f and g along the axis, which is now the relative coordinate.

        '''
        return  self.x_ifft( self.x_fft(f, axis=axis).conj() * self.x_fft(g, axis=axis), axis=axis) / np.sqrt(self.nx)

    def fft(self, form, axes=(-2,-1)):
        r'''
        Fourier transforms the form in the space and time directions,

        .. math ::
            
            F_{\nu,k} = \frac{1}{N} \sum_{x,t=0}^{N-1} e^{-2\pi i (\nu t +k x) / N} f_{t,x}

        where $\nu, k$ are the integer frequency and wavenumber, $t, x$ are the integer time adn space coordinates and $N$ is the linear extent of the lattice.

        Parameters
        ----------
        form: np.array
            The data to transform
        axes: (int, int)
            The axes which are the (time, space) directions.

        Returns
        -------
        np.array:
            The form is transformed to the (frequency, wavenumber) domain along the axis.
        '''
        return np.fft.fft2(form, axes=axes, norm='ortho')

    def ifft(self, form, axes=(-2,-1)):
        r'''
        Inverse Fourier transforms the form in the space and time directions,

        .. math ::
            
            f_{t,x} = \frac{1}{N} \sum_{\nu,k=0}^{N-1} e^{-2\pi i (\nu t +k x) / N} F_{\nu,k}

        where $\nu, k$ are the integer frequency and wavenumber, $t, x$ are the integer time adn space coordinates and $N$ is the linear extent of the lattice.

        Parameters
        ----------
        form: np.array
            The data to transform
        axes: (int, int)
            The axes which are the (frequency, wavenumber) directions.

        Returns
        -------
        np.array:
            The form is transformed to the (time, space) domain along the axis.
        '''
        return np.fft.ifft2(form, axes=axes, norm='ortho')

    def convolution(self, f, g, axes=(-2, -1)):
        r'''
        The `convolution <https://en.wikipedia.org/wiki/Convolution>`_ is given by

        .. math ::
            \texttt{convolution(f, g)}(t, x) = (f * g)(t, x) = \sum_{\tau y} f(\tau,y) g(t-\tau, x-y)

        where $f^*$ is the complex-conjugate of $f$.

        .. collapse :: The convolution is Fourier accelerated.
            :class: note

            .. math ::

               \begin{align}
                (f * g)(t,x) &= \sum_{\tau y} f(\tau,y) g(t-\tau, x-y)
                \\ &= \sum_{\tau y}
                    \left(\frac{1}{N} \sum_{\nu,k} e^{-2\pi i (\nu \tau +k y) / N} F_{\nu,k}\right)
                    \left(\frac{1}{N} \sum_{\nu',q} e^{-2\pi i (\nu' (t-\tau) +q (x-y)) / N} G_{\nu',q}\right)
                \\ &= \sum_{\nu, k, \nu', q}
                    e^{-2\pi i (\nu' t + q x) / N} F_{\nu,k}G_{\nu',q}
                    \left(\frac{1}{N^2}\sum_{\tau y}e^{-2\pi i [\tau(\nu-\nu') + y(k-q)] / N} = \delta_{kq} \delta_{\nu\nu'}\right)
                \\ &= N \times \frac{1}{N} \sum_{\nu, k}
                    e^{-2\pi i (\nu t + k x) / N} F_{\nu,k}G_{\nu,k}
                \\
                \texttt{convolution(f, g)} &= N \times \texttt{ifft(fft(f)fft(g))} 
               \end{align}

        Parameters
        ----------
        f: np.array
            A form whose axes are temporal and spatial directions.
        g: np.array
            A form whose axes are temporal and spatial directions.
        axes: (int, int)
            The common temporal and spatial dimensions along which to convolve.

        Returns
        -------
        np.array:
            The convolution of f and g along the axes, which represent the (time, space) separation.

        '''
        return np.sqrt(self.sites) * self.ifft(self.fft(f, axes=axes) * self.fft(g, axes=axes), axes=axes)

    def correlation(self, f, g, axes=(-2,-1)):
        r'''
        The `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ is given by

        .. math ::
            \texttt{correlation(f, g)}(t, x) = (f ⋆ g)(t, x) = \frac{1}{N^2} \sum_{\tau y} f(\tau,y)^* g(\tau-t, y-x)

        where $f^*$ is the complex-conjugate of $f$.

        .. collapse :: The cross-correlation is Fourier accelerated.
            :class: note

            .. math ::

               \begin{align}
                (f ⋆ g)(t,x) &= \frac{1}{N^2} \sum_{\tau y} f(\tau,y)^* g(\tau-t, y-x)
                \\ &= \frac{1}{N^2} \sum_{\tau y}
                    \left(\frac{1}{N} \sum_{\nu,k} e^{-2\pi i (\nu \tau +k y) / N} F_{\nu,k}\right)^*
                    \left(\frac{1}{N} \sum_{\nu',q} e^{-2\pi i (\nu' (\tau-t) +q (y-x)) / N} G_{\nu',q}\right)
                \\ &= \frac{1}{N^2} \sum_{\nu, k, \nu', q}
                    e^{+2\pi i (\nu' t + q x) / N} F_{\nu,k}^*G_{\nu',q}
                    \left(\frac{1}{N^2}\sum_{\tau y}e^{2\pi i [\tau(\nu-\nu') + y(k-q)] / N} = \delta_{kq} \delta_{\nu\nu'}\right)
                \\ &= \frac{1}{N}\times \frac{1}{N} \sum_{\nu, k}
                    e^{+2\pi i (\nu t + k x) / N} F_{\nu,k}^*G_{\nu,k}
                \\
                \texttt{correlation(f, g)} &= \texttt{fft(conj(fft(f))fft(g))} / N
               \end{align}

        .. warning ::
            We have $g(\tau-t, y-x)$ whereas `Wikipedia <https://en.wikipedia.org/wiki/Cross-correlation>`_ has $g(\tau+t, y+x)$.
            The difference is just the sign on the relative coordinates.

        .. warning ::
            We normalize by the spacetime volume, `Wikipedia <https://en.wikipedia.org/wiki/Cross-correlation>`_ does not.


        Parameters
        ----------
        f: np.array
            A form whose axes are temporal and spatial directions.
        g: np.array
            A form whose axes are temporal and spatial directions.
        axes: (int, int)
            The common temporal and spatial dimensions along which to correlate.

        Returns
        -------
        np.array:
            The correlation of f and g along the axes, which represent the (time, space) separation.

        '''
        return  self.fft( self.fft(f, axes=axes).conj() * self.fft(g, axes=axes), axes=axes) / np.sqrt(self.sites)

    def plot_form(self, p, form, axis, zorder=None, pointsize=200, linkwidth=0.025):
        r'''
        Plots the p-form on the axis.

        The following figure shows a 0-form plotted on sites, a 1-form on links, and a 2-form on plaquettes.
        See the source for details.

        .. plot:: examples/plot-forms.py
        
        Parameters
        ----------
        p: int
            The kind of form.
        form: np.array
            The data constituting form.
        axis: matplotlib.pyplot.axis
            The axis on which to plot.
        zorder: float
            If `None` defaults to `zorder=-p` to layer plaquettes, links, and sites well.
        '''
        zorder = -p if zorder is None else zorder
        
        background='white'
        marker = {'s': pointsize, 'edgecolor': background, 'linewidth': 2} 
        no_arrowhead = {'headwidth': 0, 'headlength': 0, 'headaxislength': 0,}
        linkpadding = {'edgecolor': background, 'linewidth': 4}
        
        links = {'scale_units': 'xy', 'scale': 1, 'width': linkwidth, **no_arrowhead, **linkpadding}
        
        if p == 0:
            axis.scatter(self.T, self.X, c=form, zorder=zorder, **marker)
        
        if p == 1:
            axis.quiver(self.T, self.X, 1, 0, form[0], zorder=zorder, **links)
            axis.quiver(self.T, self.X, 0, 1, form[1], zorder=zorder, **links)
            axis.scatter(self.T, self.X, color=background, zorder=zorder, **marker)
            
        if p == 2:
            # We roll the form because the figure should have (0,0) in the middle but the form has (0,0) in the corner.
            # We transpose because imshow goes in the 'other order'.
            form = self.roll(form, (self.nt // 2, self.nx // 2)).transpose()
            axis.imshow(form, zorder=zorder, origin='lower', extent=(min(self.t), max(self.t)+1, min(self.x), max(self.x)+1))
            axis.quiver(self.T, self.X, 1, 0, color='white', zorder=zorder, **links)
            axis.quiver(self.T, self.X, 0, 1, color='white', zorder=zorder, **links)
            axis.scatter(self.T, self.X, color=background, zorder=zorder, **marker)
            axis.xaxis.set_zorder(-p)
            axis.yaxis.set_zorder(-p)
            
        axis.set_xlim(min(self.t)-0.5, max(self.t)+1.5)
        axis.set_ylim(min(self.x)-0.5, max(self.x)+1.5)
        axis.set_xlabel('t')
        axis.set_ylabel('x')

    def x_even(self, form, axis=-1):
        r'''
        Returns the form symmetrized along the spatial axis.
        '''
        return 0.5*(form + np.roll(np.flip(form, axis=axis), 1, axis=axis))

    def x_odd(self, form, axis=-1):
        r'''
        Returns the form antisymmetrized along the spatial axis.
        '''
        return 0.5*(form - np.roll(np.flip(form, axis=axis), 1, axis=axis))

    def t_even(self, form, axis=-2):
        r'''
        Returns the form symmetrized along the temporal axis.
        '''
        return 0.5*(form + np.roll(np.flip(form, axis=axis), 1, axis=axis))

    def t_odd(self, form, axis=-2):
        r'''
        Returns the form antisymmetrized along the temporal axis.
        '''
        return 0.5*(form - np.roll(np.flip(form, axis=axis), 1, axis=axis))

    # TODO: spacetime point group symmetry projection to D4 irreps.

    def _point_group_permutation(self, operator):
        # Since the operations map lattice points to lattice points we know that they are a permutation
        # on the set of coordinates.
        permutation = []
        for i in range(self.sites):
            for j in range(self.sites):
                if (self.coordinates[i] == np.matmul(operator, self.coordinates[j])).all():
                    permutation += [j]
                    continue # since a permutation is one-to-one
        return np.array(permutation)

    def irrep(self, correlator, irrep='A1', conjugate=False, dims=(-1,)):
        r'''

        The point group of a 2D lattice is $D_4$ and the structure and irreps are detailed in `https://two-dimensional-gasses.readthedocs.io/en/latest/computational-narrative/D4.html <the tdg documentation>`_\, where the spatial lattice is 2D.

        .. plot:: examples/plot-D4-irreps.py

        .. note::
            Currently we only know how project scalar correlators that depend on a single spatial separation.

        Parameters
        ----------
            data: np.ndarray
                Data whose `axes` should be symmetrized.
            irrep: one of ``.point_group_irreps``
                The irrep to project to.
            conjugate: `True` or `False`
                The weights are conjugated, which only affects the E representations.
            dims: 
                The latter of an adjacent time/space pair of dimensions.
                The dimensions will be linearized and therefore must be adjacent.

        Returns
        -------
            A complex-valued torch.tensor of the same shape as data, but with the axis projected to the requested irrep.
        '''

        C = self.linearize(correlator, dims=dims)
        temp = np.zeros_like(C) + 0.j

        for p, w in zip(
                self.point_group_permutations,
                self.point_group_weights[irrep] if not conjugate else self.point_group_weights[irrep].conj()
                ):
            temp += w * np.take(C, p, -1)

        return self.coordinatize(temp, dims=dims) * self.point_group_norm
