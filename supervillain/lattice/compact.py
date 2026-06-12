#!/usr/bin/env python
# coding: utf-8
#
# compact.py — Differential forms on a D-dimensional hypercubic lattice,
# stored without wasted space.
#
# MOTIVATION
# ----------
# The interlaced representation (see interlaced.py) places p-form data at
# sites with exactly p odd coordinates in a (2N)^D array.  That is correct
# but sparse: only C(D,p)/2^D of array elements are used.  Here we store a
# p-form compactly as an array
# of shape
#
#     (C(D,p), N, N, ..., N)       [D spatial axes follow]
#
# The first axis indexes the C(D,p) "components", listed in lexicographic
# order by the sorted tuple of directions that are "form directions".
#
# Example — D=3:
#   0-form components:  ()            → shape (1, N, N, N)
#   1-form components:  (0,),(1,),(2,) → shape (3, N, N, N)
#   2-form components:  (0,1),(0,2),(1,2) → shape (3, N, N, N)
#   3-form components:  (0,1,2)       → shape (1, N, N, N)

from functools import cached_property
from itertools import combinations, permutations as _permutations
from math import comb, factorial
import numpy as np

from supervillain.h5 import ReadWriteable
from supervillain.lattice import _dimension


def _hyperoctant_pair_mask(coords, b, D):
    """
    Sites in one pair of opposite hyperoctants, indexed by b in 0..2^(D-1)-1.

    The representative hyperoctant has coords[0] >= 0; bits of b fix the sign
    of coords[1], …, coords[D-1].  The paired hyperoctant flips every sign.
    """
    mask_pos = coords[0] >= 0
    mask_neg = coords[0] < 0
    for k in range(1, D):
        bit = (b >> (k - 1)) & 1
        if bit == 0:
            mask_pos &= coords[k] >= 0
            mask_neg &= coords[k] < 0
        else:
            mask_pos &= coords[k] < 0
            mask_neg &= coords[k] >= 0
    return mask_pos | mask_neg


# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------

class Lattice(ReadWriteable):
    """
    A D-dimensional hypercubic lattice with N sites per direction.

    The lattice knows how to enumerate the components of any p-form and
    can create zero or random p-forms on demand.
    """

    def __init__(self, D, N):
        self.D = D
        self.N = N

        # components[p] — ordered list of component tuples for p-forms.
        # Each tuple is a sorted sequence of p direction indices (0-based).
        # Length = C(D, p).  Order is lexicographic.
        self.components = {
            p: list(combinations(range(D), p))
            for p in range(D + 1)
        }

        # comp_index[p][dirs] — reverse map: tuple of dirs → integer index.
        # Use comp_index[p][(i, j)] to find where component (i,j) lives
        # along axis 0 of a p-form array.
        self.comp_index = {
            p: {comp: idx for idx, comp in enumerate(self.components[p])}
            for p in range(D + 1)
        }

    def to_h5(self, group, _top=True):
        """Write D and N to HDF5; everything else is recomputed on load."""
        from supervillain.h5 import Data
        Data.write(group, 'D', self.D)
        Data.write(group, 'N', self.N)

    @classmethod
    def from_h5(cls, group, strict=True, _top=True):
        """Reconstruct from HDF5 by reading D and N reconstructing the rest."""
        from supervillain.h5 import Data
        D = Data.read(group['D'], strict)
        N = Data.read(group['N'], strict)
        return cls(D=D, N=N)

    @cached_property
    def sites(self):
        """Total number of sites $N^D$."""
        return self.N ** self.D

    @cached_property
    def dims(self):
        """Tuple of side lengths (N, N, ..., N), length D."""
        return (self.N,) * self.D

    @property
    def dim(self):
        """Number of dimensions (alias for D)."""
        return self.D

    @cached_property
    def links(self):
        """Total number of links (1-form components) $D N^D$."""
        return self.D * self.sites

    @cached_property
    def cells_of_degree(self):
        """
        dict[int, int]: p → number of p-cells = $C(D,p)  N^D$."""
        return {p: comb(self.D, p) * self.sites for p in range(self.D + 1)}

    @cached_property
    def cells_of_codegree(self):
        """
        dict[int, int]: q → number of (D−q)-cells = $C(D, D−q) N^D$."""
        return {q: self.cells_of_degree[self.D - q] for q in range(self.D + 1)}

    @cached_property
    def coords(self):
        """FFT-convention coordinate for each lattice site, shape (D, N, …, N)."""
        return np.stack(
            np.meshgrid(*(_dimension(self.N) for _ in range(self.D)), indexing='ij'),
            axis=0,
        )

    @cached_property
    def checkerboarding(self):
        r"""
        Partition lattice sites into colors with no same-color nearest neighbors.

        On a lattice of even size both the sites and plaquettes can be bipartitioned so that no
        simplex of one color has a neighbor of the same color.  On an odd-sized lattice the
        periodic boundary conditions make it impossible to accomplish this with only 2 colors, but
        a similar construction with more colors is possible.

        With even $N$ there are two colors (coordinate-sum parity); with odd $N$ there are $2^{\max(D, 2)}$
        to ensure that no nearest-neighbor sites share a color.
        The figure below shows both cases for D=2:

        .. plot:: example/plot/checkerboarding.py

        Returns a tuple of ``np.where`` index-array tuples, one per color.  Each element selects a
        color's sites from the D spatial axes of a form::

            for i, color in enumerate(L.checkerboarding):
                form[(slice(None), *color)] = i


        .. warning::
            No promise is made about the future sizes of the color partitions.
            For example, it might be wiser for performance to split the odd-N colors less evenly.
            All that is promised is that within each color no site shares a nearest-neighbor edge
            with a site of the same color.
        """
        D, N = self.D, self.N
        coords = self.coords
        parity = np.mod(self.coords.sum(axis=0), 2)

        if N % 2 == 0:
            return tuple(np.where(parity == c) for c in (0, 1))

        colors = []
        n_pairs = 1 << max(D - 1, 1)
        for b in range(n_pairs):
            if D == 1:
                pair = coords[0] >= 0 if b == 0 else coords[0] < 0
            else:
                pair = _hyperoctant_pair_mask(list(coords), b, D)
            for c in (0, 1):
                colors.append(np.where(pair & (parity == c)))
        return tuple(colors)

    # ------------------------------------------------------------------
    # Factory methods

    def zeros(self, p, dtype=float):
        """Return a zero p-form: shape (C(D,p), N,...,N)."""
        shape = (comb(self.D, p),) + (self.N,) * self.D
        return Form(np.zeros(shape, dtype=dtype), degree=p, lattice=self)

    def random(self, p):
        """Return a p-form with random entries uniform in [0, 1)."""
        shape = (comb(self.D, p),) + (self.N,) * self.D
        return Form(np.random.random(shape), degree=p, lattice=self)

    def form(self, p, dtype=float):
        """Return a zero p-form.  Alias for zeros(p, dtype=dtype)."""
        return self.zeros(p, dtype=dtype)

    @cached_property
    def _coord_1d(self):
        return _dimension(self.N)

    @cached_property
    def R_squared(self):
        """Distance-squared from the origin at each site, shape (N,...,N)."""
        return np.sum(self.coords**2, axis=0)

    def distance_squared(self, a, b):
        r"""Squared lattice distance between coordinate vectors ``a`` and ``b``.

        Accounts for periodic boundary conditions: the distance is computed
        via ``mod(a - b)``, so it is the shortest-path distance on the torus.

        Parameters
        ----------
        a, b : array_like
            Coordinate vectors of shape ``(D,)`` for a single pair, or
            ``(..., D)`` for a batch.

        Returns
        -------
        np.ndarray
            Scalar for a single pair, shape ``(...)`` for a batch.
        """
        d = self.mod(np.asarray(a) - np.asarray(b))
        return np.sum(d**2, axis=-1)

    @cached_property
    def coordinates(self):
        """Array of shape (sites, D) listing every site's coordinates."""
        return np.stack(
            [c.flatten() for c in np.meshgrid(*[self._coord_1d] * self.D, indexing='ij')],
            axis=1,
        )

    def mod(self, x):
        """Mod integer coordinates into FFT-convention lattice values."""
        x = np.asarray(x)
        modded = np.mod(x, self.N)
        return self._coord_1d[modded]

    # ------------------------------------------------------------------
    # Fourier methods
    #
    # Convention matches Lattice2D: ortho normalization, spatial axes last.
    # For a p-form the spatial axes are the last D axes; the default axes
    # argument reflects this so callers seldom need to override it.

    def _spatial_axes(self):
        """The last D axes, i.e. the spatial directions of any p-form."""
        return tuple(range(-self.D, 0))

    def fft(self, form, axes=None):
        r"""
        D-dimensional DFT over the spatial axes of ``form``.

        .. math::

            F_{\boldsymbol{k}} = \frac{1}{N^{D/2}}
            \sum_{\boldsymbol{x}} e^{-2\pi i \boldsymbol{k}\cdot\boldsymbol{x}/N} f_{\boldsymbol{x}}

        Parameters
        ----------
        form: np.ndarray
            The data to transform.  Spatial axes are the last D axes.
        axes: tuple of int, optional
            Override which axes to transform.  Defaults to the last D axes.

        Returns
        -------
        np.ndarray
            The Fourier-transformed array (complex).
        """
        return np.fft.fftn(form, axes=(axes if axes is not None else self._spatial_axes()), norm='ortho')

    def ifft(self, form, axes=None):
        r"""
        D-dimensional inverse DFT over the spatial axes of ``form``.

        Parameters
        ----------
        form: np.ndarray
        axes: tuple of int, optional

        Returns
        -------
        np.ndarray
        """
        return np.fft.ifftn(form, axes=(axes if axes is not None else self._spatial_axes()), norm='ortho')

    def convolution(self, f, g, axes=None):
        r"""
        The `convolution <https://en.wikipedia.org/wiki/Convolution>`_ is given by

        .. math::

            \texttt{convolution}(f,g)(\boldsymbol{r})
            = (f * g)(\boldsymbol{r})
            = \sum_{\boldsymbol{x}} f(\boldsymbol{x})\,g(\boldsymbol{r}-\boldsymbol{x})

        .. collapse:: The convolution is Fourier accelerated.
            :class: note

            With the ortho-normalized DFT convention
            :math:`f(\boldsymbol{x}) = N^{-D/2} \sum_{\boldsymbol{k}} F_{\boldsymbol{k}}\, e^{2\pi i\,\boldsymbol{k}\cdot\boldsymbol{x}/N}`,

            .. math::

                \begin{aligned}
                (f * g)(\boldsymbol{r})
                &= \sum_{\boldsymbol{x}}
                    \Bigl(\tfrac{1}{N^{D/2}}\sum_{\boldsymbol{k}}  F_{\boldsymbol{k}}\,
                          e^{2\pi i\,\boldsymbol{k}\cdot\boldsymbol{x}/N}\Bigr)
                    \Bigl(\tfrac{1}{N^{D/2}}\sum_{\boldsymbol{k}'} G_{\boldsymbol{k}'}\,
                          e^{2\pi i\,\boldsymbol{k}'\cdot(\boldsymbol{r}-\boldsymbol{x})/N}\Bigr)
                \\
                &= \frac{1}{N^D}\sum_{\boldsymbol{k},\boldsymbol{k}'}
                    F_{\boldsymbol{k}} G_{\boldsymbol{k}'}\,
                    e^{2\pi i\,\boldsymbol{k}'\cdot\boldsymbol{r}/N}
                    \underbrace{
                        \sum_{\boldsymbol{x}} e^{2\pi i\,(\boldsymbol{k}-\boldsymbol{k}')\cdot\boldsymbol{x}/N}
                    }_{N^D\,\delta_{\boldsymbol{k}\boldsymbol{k}'}}
                \\
                &= \sum_{\boldsymbol{k}} F_{\boldsymbol{k}} G_{\boldsymbol{k}}\,
                    e^{2\pi i\,\boldsymbol{k}\cdot\boldsymbol{r}/N}
                \\
                &= \sqrt{N^D}\times
                    \underbrace{\tfrac{1}{N^{D/2}}\sum_{\boldsymbol{k}}
                        \bigl(F_{\boldsymbol{k}} G_{\boldsymbol{k}}\bigr)\,
                        e^{2\pi i\,\boldsymbol{k}\cdot\boldsymbol{r}/N}
                    }_{\texttt{ifft}(\hat f \hat g)(\boldsymbol{r})}
                \\
                \texttt{convolution}(f,g) &= \sqrt{N^D}\;\texttt{ifft}\!\bigl(\texttt{fft}(f)\cdot\texttt{fft}(g)\bigr)
                \end{aligned}

        Parameters
        ----------
        f, g: np.ndarray
            Forms on this lattice; spatial axes are the last D axes.
        axes: tuple of int, optional

        Returns
        -------
        np.ndarray
        """
        ax = axes if axes is not None else self._spatial_axes()
        return np.sqrt(self.sites) * self.ifft(self.fft(f, axes=ax) * self.fft(g, axes=ax), axes=ax)

    def correlation(self, f, g, axes=None):
        r"""
        The `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ is given by

        .. math::

            \texttt{correlation}(f,g)(\boldsymbol{r})
            = (f \star g)(\boldsymbol{r})
            = \frac{1}{N^D} \sum_{\boldsymbol{x}} f(\boldsymbol{x})^*\,g(\boldsymbol{x}-\boldsymbol{r})

        where :math:`f^*` is the complex conjugate of :math:`f`.

        .. collapse:: The cross-correlation is Fourier accelerated.
            :class: note

            With the ortho-normalized DFT convention
            :math:`f(\boldsymbol{x}) = N^{-D/2} \sum_{\boldsymbol{k}} F_{\boldsymbol{k}}\, e^{2\pi i\,\boldsymbol{k}\cdot\boldsymbol{x}/N}`,

            .. math::

                \begin{aligned}
                (f \star g)(\boldsymbol{r})
                &= \frac{1}{N^D}\sum_{\boldsymbol{x}}
                    \Bigl(\tfrac{1}{N^{D/2}}\sum_{\boldsymbol{k}}  F_{\boldsymbol{k}}\,
                          e^{2\pi i\,\boldsymbol{k}\cdot\boldsymbol{x}/N}\Bigr)^*
                    \Bigl(\tfrac{1}{N^{D/2}}\sum_{\boldsymbol{k}'} G_{\boldsymbol{k}'}\,
                          e^{2\pi i\,\boldsymbol{k}'\cdot(\boldsymbol{x}-\boldsymbol{r})/N}\Bigr)
                \\
                &= \frac{1}{N^{2D}}\sum_{\boldsymbol{k},\boldsymbol{k}'}
                    F_{\boldsymbol{k}}^* G_{\boldsymbol{k}'}\,
                    e^{-2\pi i\,\boldsymbol{k}'\cdot\boldsymbol{r}/N}
                    \underbrace{
                        \sum_{\boldsymbol{x}} e^{2\pi i\,(\boldsymbol{k}'-\boldsymbol{k})\cdot\boldsymbol{x}/N}
                    }_{N^D\,\delta_{\boldsymbol{k}\boldsymbol{k}'}}
                \\
                &= \frac{1}{N^D}\sum_{\boldsymbol{k}}
                    F_{\boldsymbol{k}}^* G_{\boldsymbol{k}}\,
                    e^{-2\pi i\,\boldsymbol{k}\cdot\boldsymbol{r}/N}
                \\
                &= \frac{1}{\sqrt{N^D}}\times
                    \underbrace{\tfrac{1}{N^{D/2}}\sum_{\boldsymbol{k}}
                        \bigl(F_{\boldsymbol{k}}^* G_{\boldsymbol{k}}\bigr)\,
                        e^{-2\pi i\,\boldsymbol{k}\cdot\boldsymbol{r}/N}
                    }_{\texttt{fft}(\hat f^*\!\cdot\hat g)(\boldsymbol{r})}
                \\
                \texttt{correlation}(f,g) &= \texttt{fft}\!\bigl(\texttt{fft}(f)^*\cdot\texttt{fft}(g)\bigr) / \sqrt{N^D}
                \end{aligned}

        .. warning::

            We have :math:`g(\boldsymbol{x}-\boldsymbol{r})` whereas
            `Wikipedia <https://en.wikipedia.org/wiki/Cross-correlation>`_ has
            :math:`g(\boldsymbol{x}+\boldsymbol{r})`.
            The difference is just the sign of the relative coordinate.

        .. warning::

            We normalize by the spacetime volume :math:`N^D`;
            `Wikipedia <https://en.wikipedia.org/wiki/Cross-correlation>`_ does not.

        Parameters
        ----------
        f, g: np.ndarray
            Forms on this lattice; spatial axes are the last D axes.
        axes: tuple of int, optional

        Returns
        -------
        np.ndarray
        """
        ax = axes if axes is not None else self._spatial_axes()
        return self.fft(self.fft(f, axes=ax).conj() * self.fft(g, axes=ax), axes=ax) / np.sqrt(self.sites)

    def linearize(self, v, dims=(-1,)):
        r"""Flatten the D spatial dims of ``v`` into a single axis of size ``sites``.

        Parameters
        ----------
        v : np.ndarray
            Array whose D adjacent spatial dims will be collapsed into one.
        dims : tuple of int
            Axes of the *result* that come from flattening.  Pass the same value
            to :meth:`coordinatize` for a round-trip.

        Returns
        -------
        np.ndarray
        """
        shape = v.shape
        v_dims = len(shape)

        dm = set(dims)
        future_dims = v_dims - (self.D - 1) * len(dm)
        dm = set(d % future_dims for d in dm)

        new_shape = []
        idx = 0
        for i in range(future_dims):
            if i not in dm:
                new_shape.append(shape[idx])
                idx += 1
            else:
                new_shape.append(self.sites)
                idx += self.D
        return v.reshape(new_shape)

    def coordinatize(self, v, dims=(-1,), center_origin=False):
        r"""Unflatten a linear site-index axis back into D spatial axes.

        Parameters
        ----------
        v : np.ndarray
            Array with a ``sites``-sized axis to expand.
        dims : tuple of int
            Axes of ``v`` to unflatten.  Each must have size ``sites``.
        center_origin : bool
            If True, roll each spatial block so that the origin sits in the
            centre of the array.  Useful for visualisation; not invertible.

        Returns
        -------
        np.ndarray
        """
        v_dims = len(v.shape)
        to_reshape = np.sort(np.remainder(np.array(dims), v_dims))

        new_shape = ()
        for i, s in enumerate(v.shape):
            new_shape += ((s,) if i not in to_reshape else self.dims)

        reshaped = v.reshape(new_shape)
        if not center_origin:
            return reshaped

        # Each original axis at position a expands to D axes; subsequent blocks
        # shift right by (D-1) per earlier expansion.
        axes = to_reshape + np.arange(len(to_reshape)) * (self.D - 1)
        for a in axes:
            for d in range(self.D):
                reshaped = np.roll(reshaped, self.N // 2, axis=int(a) + d)

        return reshaped

    @cached_property
    def _hyperoctahedral_permutations(self):
        """Site-index permutations for each of the D!·2^D hyperoctahedral group elements."""
        coords = self.coordinates          # (sites, D)
        coord_to_idx = {tuple(c): k for k, c in enumerate(coords)}
        result = []
        # The hyperoctahedral group B_D = (Z/2Z)^D ⋊ S_D combines the D! permutations
        # of coordinate axes with 2^D independent per-axis sign flips.
        for perm in _permutations(range(self.D)):
            # signs is NOT the sign (parity) of the permutation — it is D independent
            # bits, one per axis, encoding which coordinates get negated.
            for signs in np.ndindex(*([2] * self.D)):
                # Map the 0/1 bits to +1/−1: a signed permutation x → sign_vec * x[perm].
                sign_vec = np.array([1 - 2 * s for s in signs])
                idx_perm = np.array([
                    coord_to_idx[tuple(self.mod(sign_vec * coords[i][list(perm)]))]
                    for i in range(self.sites)
                ])
                result.append(idx_perm)
        return result

    def symmetrize(self, correlator, dims=(-1,)):
        r"""Average ``correlator`` over the hyperoctahedral group (D!·2^D signed permutations).

        Projects onto the totally-symmetric (A₁/trivial) irrep of the lattice point
        group.

        .. plot:: example/plot/symmetrize.py

        Parameters
        ----------
        correlator : np.ndarray
            Spatial shape ``(N,...,N)`` (or any array whose ``dims`` axes span the
            sites of this lattice after linearization).
        dims : tuple of int
            Axes to symmetrize (same convention as :meth:`linearize`).

        Returns
        -------
        np.ndarray
            Same shape as ``correlator``.
        """
        C = self.linearize(correlator, dims=dims)
        v_dims = len(C.shape)
        sites_axis = list(dims)[0] % v_dims
        perms = self._hyperoctahedral_permutations
        result = np.sum([np.take(C, p, axis=sites_axis) for p in perms], axis=0)
        return self.coordinatize(result / len(perms), dims=dims)

    def __repr__(self):
        return f"Lattice(D={self.D}, N={self.N})"


# ---------------------------------------------------------------------------
# Form  (numpy.ndarray subclass)
# ---------------------------------------------------------------------------

class Form(np.ndarray):
    """
    A differential p-form on a Lattice, stored compactly.

    Underlying array shape: (C(D,p), N, N, ..., N).
    - Axis 0 : component index (C(D,p) values, lex order by direction tuple).
    - Axes 1…D : physical lattice site n = (n_0, …, n_{D-1}).

    RELATIONSHIP TO THE INTERLACED LAYOUT
    In the interlaced (2N)^D array a p-form component with odd directions I
    lives at interlaced coordinates x where:
        x_k = 2 n_k      for k ∉ I  (even — a "site" direction)
        x_k = 2 n_k + 1  for k ∈ I  (odd  — a "link/plaquette/…" direction)
    In both cases  x_k // 2 = n_k, so the physical site is always the floor
    of the interlaced coordinate divided by 2.  This is why from_interlaced
    can use slice(0,None,2) and slice(1,None,2) interchangeably to extract
    the same physical-site index range for every component.

    ARITHMETIC
    Element-wise operations (±, *, /, unary −, abs, **, np.sqrt, np.isclose,
    ==, …) return a Form of the same degree when all Form operands share a
    degree.  Reductions (sum, max, …) return plain numpy scalars or arrays.
    Mixed-degree arithmetic is left as a plain ndarray because the degree of
    the result would be ambiguous.

    Numpy subclassing protocol:
    - __new__          : attach degree and lattice when the object is created.
    - __array_finalize__: propagate them when numpy makes an internal view.
    - __array_ufunc__  : intercept every ufunc call to return a typed Form
                         when the result is unambiguous.
    """

    __batch_tag__ = 'Form'

    @classmethod
    def spatial_shape(cls, *, degree, lattice):
        return (comb(lattice.D, degree),) + (lattice.N,) * lattice.D

    def __new__(cls, input_array, *, degree, lattice, dtype=None):
        obj = np.asarray(input_array, dtype=dtype).view(cls)
        obj.degree  = degree
        obj.lattice = lattice
        return obj

    def __array_finalize__(self, obj):
        # obj is None when called from an explicit constructor,
        # otherwise it is the "parent" array being viewed.
        if obj is None:
            return
        self.degree  = getattr(obj, 'degree',  None)
        self.lattice = getattr(obj, 'lattice', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Collect every Form among the inputs and check degree agreement.
        form_inputs = [x for x in inputs if isinstance(x, Form)]
        degrees     = {f.degree for f in form_inputs}

        # Strip Form wrappers so the ufunc operates on raw data.
        raw = tuple(np.asarray(x) for x in inputs)
        out = kwargs.get('out')
        if out is not None:
            kwargs['out'] = tuple(np.asarray(o) for o in out)

        result = getattr(ufunc, method)(*raw, **kwargs)

        # Re-wrap as a Form when the result is unambiguous:
        #   • all Form inputs share the same degree (mixed-degree → plain)
        #   • same shape as the inputs (element-wise, not a reduction)
        # Boolean results (np.isclose, ==, …) are included: "is component k
        # at site n close to zero?" has the same index structure as the form
        # itself, so the degree is still meaningful.
        if (len(degrees) == 1
                and isinstance(result, np.ndarray)
                and result.shape == form_inputs[0].shape):
            return Form(result,
                        degree=form_inputs[0].degree,
                        lattice=form_inputs[0].lattice)

        return result

    # ------------------------------------------------------------------
    # Component access

    def component(self, *dirs):
        """
        View of a single component's spatial data, shape (N,...,N).

        Pass the direction indices as separate arguments or as a tuple:
            f.component(0, 2)   — the (0,2)-component of a 2-form
            f.component((0, 2)) — same

        The return value is a *view*, so writes back to the Form.
        """
        # Accept either f.component(0,2) or f.component((0,2))
        if len(dirs) == 1 and hasattr(dirs[0], '__iter__'):
            dirs = tuple(dirs[0])
        dirs = tuple(sorted(dirs))
        idx = self.lattice.comp_index[self.degree][dirs]
        return self[idx]   # shape (N,...,N), a view

    def to_interlaced(self):
        """
        Embed the compact form into a (2N)^D interlaced array (see
        interlaced.py).  Every site that is not a p-form site is zero.

        Each component indexed by direction tuple `comp` occupies the
        sub-array where direction k uses odd indices (start=1) if k is
        in `comp`, and even indices (start=0) otherwise.

        Returns a plain numpy.ndarray of shape (2N, 2N, ..., 2N).
        """
        lat = self.lattice
        D, N = lat.D, lat.N
        result = np.zeros((2 * N,) * D, dtype=self.dtype)

        for comp_tuple, comp_idx in lat.comp_index[self.degree].items():
            dirs = set(comp_tuple)
            slc = tuple(
                slice(1 if k in dirs else 0, None, 2)
                for k in range(D)
            )
            result[slc] = self[comp_idx]

        return result

    @classmethod
    def from_interlaced(cls, p, data, lattice=None):
        """
        Construct a compact Form from an interlaced (2N)^D array.

        p       — form degree
        data    — interlaced array of shape (2N, 2N, ..., 2N); only the
                  sites with exactly p odd coordinates are read.
        lattice — optional Lattice; inferred from data.shape if omitted.

        This is the left-inverse of to_interlaced():
            Form.from_interlaced(p, f.to_interlaced()) == f   for any p-Form f.

        For a valid interlaced p-form it is also the right-inverse:
            Form.from_interlaced(p, data).to_interlaced() == data
            (provided data is zero at all non-p-form sites).
        """
        D = data.ndim
        N = data.shape[0] // 2
        if lattice is None:
            lattice = Lattice(D, N)

        result = lattice.zeros(p, dtype=data.dtype)
        for comp_tuple, comp_idx in lattice.comp_index[p].items():
            dirs = set(comp_tuple)
            slc  = tuple(slice(1 if k in dirs else 0, None, 2) for k in range(D))
            result[comp_idx] = data[slc]

        return result

    # ------------------------------------------------------------------
    # Face / coface sums  (unsigned incidence, for Metropolis aggregation)

    def face_sum(self):
        """
        Sum this p-form onto its (p-1)-faces, returning a (p-1)-form.

        A p-cell is bounded by (p-1)-faces.  For each such face M at site n,

            g[M, n]  =  Σ_{O ⊃ M}  ( f[O, n]  +  f[O, n − ê_e] )

        where the sum runs over p-cells O = M∪{e}.  Example: a 1-form on
        links summed onto the sites that bound each link gives a 0-form.

        Unlike δ, all contributions enter with the same sign.
        """
        lat = self.lattice
        p   = self.degree
        if p == 0:
            return 0

        result = lat.zeros(p - 1)
        all_dirs = set(range(lat.D))

        for M_comp in lat.components[p - 1]:
            out_idx = lat.comp_index[p - 1][M_comp]
            M_set   = set(M_comp)

            for e in sorted(all_dirs - M_set):
                in_comp = tuple(sorted(M_set | {e}))
                in_idx  = lat.comp_index[p][in_comp]
                spatial = self[in_idx]

                result[out_idx] += spatial
                result[out_idx] += np.roll(spatial, +1, axis=e)

        return result

    def coface_sum(self):
        """
        Sum this p-form onto incident (p+1)-cofaces, returning a (p+1)-form.

        A (p+1)-cell has p-faces; for each such coface O at site n,

            g[O, n]  =  Σ_{M ⊂ O}  ( f[M, n]  +  f[M, n + ê_e] )

        where the sum runs over p-faces M = O\\{o_j} of O.  Example: a 1-form
        on links summed onto plaquettes gives a 2-form (vortex Metropolis).

        Dual to :meth:`face_sum`; unlike d, all contributions enter unsigned.
        """
        lat = self.lattice
        p   = self.degree
        if p == lat.D:
            return 0

        result = lat.zeros(p + 1)

        for O_comp in lat.components[p + 1]:
            out_idx = lat.comp_index[p + 1][O_comp]

            for j, k_j in enumerate(O_comp):
                in_comp = tuple(k for k in O_comp if k != k_j)
                in_idx  = lat.comp_index[p][in_comp]
                spatial = self[in_idx]

                result[out_idx] += spatial
                result[out_idx] += np.roll(spatial, -1, axis=k_j)

        return result

    def __repr__(self):
        return (
            f"Form(degree={self.degree}, "
            f"shape={self.shape}, "
            f"lattice={self.lattice})"
        )


# ---------------------------------------------------------------------------
# Translation operators
# ---------------------------------------------------------------------------

def push(form, shift):
    """Translate form forward: result[..., n + shift] = form[..., n]  (periodic).

    Parameters
    ----------
    form : np.ndarray
        Array whose last ``len(shift)`` axes are the spatial directions.
    shift : sequence of int
        One integer per spatial direction.

    Returns
    -------
    np.ndarray
    """
    result = form
    for i, s in enumerate(shift):
        if s:
            result = np.roll(result, s, axis=i - len(shift))
    return result


def pull(form, shift):
    """Translate form backward: result[..., n] = form[..., n + shift]  (periodic).

    Equivalent to ``push`` with the sign of each component of ``shift`` reversed.
    """
    return push(form, tuple(-s for s in shift))


# ---------------------------------------------------------------------------
# Exterior derivative  d : Ω^p → Ω^{p+1}
# ---------------------------------------------------------------------------

def d(f):
    """
    Exterior derivative of a p-form, returning a (p+1)-form.

    For each output component O = (o_0, …, o_p) the formula is

        d(f)[O, n]  =  Σ_{j=0}^{p}  (-1)^j  ·  Δ_{o_j}  f[O \\ {o_j}]

    where  Δ_k A  is the forward finite difference in direction k:

        Δ_k A[n]  =  A[n + ê_k] − A[n]
                   =  np.roll(A, -1, axis=k) − A

    The sign alternates ±1 over the directions of the target form, matching
    the interlaced convention in interlaced.py.  Periodic boundary conditions
    come for free from np.roll.
    """
    lat = f.lattice
    p   = f.degree
    if p == lat.D:
        return 0

    result = lat.zeros(p + 1)

    for out_comp in lat.components[p + 1]:          # each (p+1)-form component
        out_idx = lat.comp_index[p + 1][out_comp]

        for j, k_j in enumerate(out_comp):
            # Remove direction k_j from out_comp to get the source p-form component.
            in_comp = tuple(k for k in out_comp if k != k_j)
            in_idx  = lat.comp_index[p][in_comp]

            sign = (-1) ** j

            # Forward finite difference of f[in_idx] in direction k_j.
            # np.roll(A, -1, axis=k) shifts data so result[n] = A[n+1 mod N].
            spatial = f[in_idx]   # shape (N,...,N)
            fwd_diff = np.roll(spatial, -1, axis=k_j) - spatial

            result[out_idx] += sign * fwd_diff

    return result


# ---------------------------------------------------------------------------
# Codifferential  δ : Ω^p → Ω^{p-1}
# ---------------------------------------------------------------------------

def delta(f):
    """
    Codifferential (formal adjoint of d) of a p-form, returning a (p-1)-form.

    For each output component M = (m_0, …, m_{p-1}) and each direction
    e ∉ M, let j = #{m ∈ M : m < e} (the position where e would be
    inserted to keep M ∪ {e} sorted).  Then

        δ(F)[M, n]  =  Σ_{e ∉ M}  (-1)^j  ·  ∇*_e  F[M ∪ {e}]

    where  ∇*_e A  is the backward finite difference in direction e:

        ∇*_e A[n]  =  A[n] − A[n − ê_e]
                    =  A − np.roll(A, +1, axis=e)

    The sign (-1)^j equals (-1)^(e−i) where i is the position of e in the
    sorted complement of M in {0,…,D-1}, because e − i = #{m ∈ M : m < e}.
    This matches the interlaced convention in interlaced.py.
    """
    lat = f.lattice
    p   = f.degree
    if p == 0:
        return 0

    result = lat.zeros(p - 1)

    all_dirs = set(range(lat.D))

    for out_comp in lat.components[p - 1]:          # each (p-1)-form component
        out_idx = lat.comp_index[p - 1][out_comp]
        M_set   = set(out_comp)

        for e in sorted(all_dirs - M_set):          # directions not in M
            # Position where e is inserted into sorted(M ∪ {e}).
            j    = sum(1 for m in out_comp if m < e)
            sign = (-1) ** j

            in_comp = tuple(sorted(M_set | {e}))
            in_idx  = lat.comp_index[p][in_comp]

            # Backward finite difference of F[in_idx] in direction e.
            # np.roll(A, +1, axis=e) shifts so result[n] = A[n-1 mod N].
            spatial  = f[in_idx]
            bwd_diff = spatial - np.roll(spatial, +1, axis=e)

            result[out_idx] += sign * bwd_diff

    return result

δ = delta


# ---------------------------------------------------------------------------
# Hodge star  ★ : Ω^p → Ω^{D-p}
# ---------------------------------------------------------------------------

def _perm_sign(seq):
    """Sign of the permutation that sorts a sequence of distinct integers."""
    return (-1) ** sum(
        1 for i in range(len(seq)) for j in range(i + 1, len(seq))
        if seq[i] > seq[j]
    )


def star(f):
    """
    Hodge star of a p-form, returning a (D-p)-form.

    For each output component J (a sorted (D-p)-tuple of directions),
    let I = complement of J in {0,…,D-1}.  Then

        (★f)[J, n]  =  σ(I,J) · f[I, n − ê_I]

    where
        σ(I,J)  = sign of the permutation sorting the concatenation (I, J)
                = (-1)^(#{(i,j) ∈ I×J : i > j})
        ê_I     = Σ_{k ∈ I} ê_k   (sum of unit vectors for the I directions)

    and  f[I, n − ê_I]  is implemented as successive  np.roll(…, +1, axis=k)
    calls for k ∈ I.

    WHY THE SPATIAL SHIFT?
    In the discrete interlaced geometry, a p-form and its Hodge dual are
    "centred" at different lattice positions.  The shift aligns them so that
    the inner-product identity holds pointwise after summing:

        Σ_{n, I}  a_I[n] · b_I[n]  =  Σ_n  (a ∧ ★b)_{0,…,D-1}[n]

    For p=0 and p=D the shift is trivial (|I|=0 or the shifts cancel in the
    wedge), which is why a pure push without signs already works for those
    two degrees.
    """
    lat = f.lattice
    p   = f.degree
    D   = lat.D
    result = lat.zeros(D - p)

    for J_comp in lat.components[D - p]:
        J_set  = set(J_comp)
        I_comp = tuple(k for k in range(D) if k not in J_set)

        sign = _perm_sign(I_comp + J_comp)

        # Roll f[I_comp] by +1 in each direction k ∈ I_comp to
        # evaluate f_I at site n − ê_I.
        spatial = f[lat.comp_index[p][I_comp]]
        for k in I_comp:
            spatial = np.roll(spatial, +1, axis=k)

        result[lat.comp_index[D - p][J_comp]] = sign * spatial

    return result


# ---------------------------------------------------------------------------
# Wedge product  ∧ : Ω^n × Ω^m → Ω^{n+m}
# ---------------------------------------------------------------------------

def wedge(a, b):
    """
    Wedge product of an n-form a and an m-form b, returning an (n+m)-form.

    For each output component O = (o_0, …, o_{n+m-1}) we sum over all
    ways to split O into

        B_dirs  (n directions, the a-component index)
        A_dirs  (m directions, the b-component index)

    with the formula

        (a ∧ b)[O, n]  =  Σ  sign · a[B_dirs, n] · b[A_dirs, n + shift_B]

    where shift_B = Σ_{k ∈ B_dirs} ê_k, implemented as successive
    np.roll(…, -1, axis=k) calls on the b-component array.

    SIGN CONVENTION
    ---------------
    Let ASSIGNMENT be the tuple indexed by position in O where entry j
    is 1 if O[j] ∈ A_dirs (→ shifts a) and 0 if O[j] ∈ B_dirs.
    sign = (-1)^(# inversions in ASSIGNMENT).

    An inversion is a pair (i < j) with ASSIGNMENT[i]=1, ASSIGNMENT[j]=0,
    i.e. an A-direction appearing before a B-direction in O.  Equivalently:

        sign = (-1)^(Σ_{k ∈ B_dirs}  #{j ∈ A_dirs : j < k})

    This matches the interlaced wedge in interlaced.py exactly (verified by
    tracing the push/pull shifts for several (n,m) pairs).
    """
    lat = a.lattice
    n, m = a.degree, b.degree
    if n + m > lat.D:
        raise ValueError(
            f"Cannot wedge a {n}-form and a {m}-form in D={lat.D}: {n}+{m} > {lat.D}"
        )

    result = lat.zeros(n + m)

    for out_comp in lat.components[n + m]:
        out_idx = lat.comp_index[n + m][out_comp]

        # Enumerate all size-m subsets of out_comp → these become A_dirs.
        # The complementary size-n subset becomes B_dirs.
        for A_dirs in combinations(out_comp, m):
            B_dirs = tuple(k for k in out_comp if k not in A_dirs)
            # B_dirs is already sorted because out_comp is sorted and we
            # remove elements while preserving order.

            # Compute the sign: count A-before-B inversions.
            A_set = set(A_dirs)
            inversions = sum(
                1
                for k in B_dirs
                for j in A_dirs
                if j < k
            )
            sign = (-1) ** inversions

            # a is evaluated at site n using component B_dirs.
            a_idx = lat.comp_index[n][B_dirs]
            a_spatial = a[a_idx]   # shape (N,...,N)

            # b is evaluated at site n + shift_B, where shift_B moves
            # +1 in each direction k ∈ B_dirs.
            # Rolling b by -1 in axis k maps b[n] ← b[n+1_k], which is
            # equivalent to evaluating b at the site one step forward in k.
            b_idx = lat.comp_index[m][A_dirs]
            b_spatial = b[b_idx]   # shape (N,...,N)
            for k in B_dirs:
                b_spatial = np.roll(b_spatial, -1, axis=k)

            result[out_idx] += sign * a_spatial * b_spatial

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from itertools import product as iproduct

    parser = argparse.ArgumentParser(description="Test compact differential forms.")
    parser.add_argument('--D', type=int, default=4, help='Spacetime dimension')
    parser.add_argument('--N', type=int, default=6, help='Lattice size per direction')
    args = parser.parse_args()

    D = args.D
    N = args.N
    lat = Lattice(D, N)
    print(f"Testing on {lat}\n")

    # --- Sparsity report ---------------------------------------------------
    # How many numbers do we actually store per lattice site?
    print("STORAGE PER SITE")
    for p in range(D + 1):
        n_comps = comb(D, p)
        old_fraction = n_comps / 2**D
        print(f"  {p}-form: {n_comps} component(s) per site "
              f"(was {old_fraction:.4f} × 2^D = {n_comps} nonzero in old layout)")

    # --- Nilpotency of d ---------------------------------------------------
    print("\nNILPOTENCY OF d")
    for p in range(D + 1):
        a  = lat.random(p)
        da = d(a)
        if p == D:
            assert da == 0, f"d({p}-form) should be scalar 0"
            print(f'  ✅ d({p}-form) = 0')
        else:
            assert not np.isclose(da, 0).all(), f"d({p}-form) is unexpectedly zero"
            dda = d(da)
            # np.asarray handles both the scalar-0 case (p == D-1) and the
            # Form case (p < D-1) uniformly.
            assert np.isclose(np.asarray(dda), 0).all(), f"d² ≠ 0 on {p}-form"
            print(f'  ✅ d² = 0 on {p}-forms')

    # --- Nilpotency of δ ---------------------------------------------------
    print("\nNILPOTENCY OF δ")
    for p in range(D + 1):
        a  = lat.random(p)
        da = delta(a)
        if p == 0:
            assert da == 0, f"δ({p}-form) should be scalar 0"
            print(f'  ✅ δ(0-form) = 0')
        else:
            assert not np.isclose(da, 0).all(), f"δ({p}-form) is unexpectedly zero"
            dda = delta(da)
            assert np.isclose(np.asarray(dda), 0).all(), f"δ² ≠ 0 on {p}-form"
            print(f'  ✅ δ² = 0 on {p}-forms')

    # --- Adjointness  Σ (da)·b = -Σ a·(δb) --------------------------------
    # Inner product of two p-forms: sum over all components and all sites.
    # In compact layout every array element is a genuine DOF, so np.sum()
    # is the correct lattice L² inner product.
    print("\nADJOINTNESS  ⟨da, b⟩ = ⟨a, δb⟩")
    for p in range(D):
        a = lat.random(p)
        b = lat.random(p + 1)
        LHS = +( d(a) * b ).sum()
        RHS = -( a * delta(b) ).sum()
        assert np.isclose(LHS, RHS), f"Adjointness failed for p={p}: {LHS} vs {RHS}"
        print(f"  ✅ ⟨da, b⟩ = ⟨a, δb⟩  for a ∈ Ω^{p}, b ∈ Ω^{p+1}")

    # --- Wedge: bilinearity ------------------------------------------------
    print("\nBILINEARITY")
    for n, m in iproduct(range(D + 1), repeat=2):
        if n + m > D:
            continue
        a = lat.random(n);  b = lat.random(n);  c = lat.random(m);  e = lat.random(m)
        assert np.isclose(wedge(a + b, c), wedge(a, c) + wedge(b, c)).all()
        assert np.isclose(wedge(a, c + e), wedge(a, c) + wedge(a, e)).all()
        print(f"  ✅ ∧ bilinear for {n} ∧ {m}")

    # --- Wedge: Leibniz rule  d(a∧b) = da∧b + (-1)^n a∧db ----------------
    print("\nLEIBNIZ RULE")
    for n, m in iproduct(range(D + 1), repeat=2):
        if n + m + 1 > D:
            continue
        a = lat.random(n);  b = lat.random(m)
        LHS = d(wedge(a, b))
        RHS = wedge(d(a), b) + (-1)**n * wedge(a, d(b))
        assert not np.isclose(LHS, 0).all()
        assert np.isclose(LHS, RHS).all(), f"Leibniz failed for {n}∧{m}"
        print(f"  ✅ d(a∧b) = da∧b + (-1)^{n} a∧db  for a ∈ Ω^{n}, b ∈ Ω^{m}")

    # --- Wedge: associativity  (a∧b)∧c = a∧(b∧c) -------------------------
    print("\nASSOCIATIVITY")
    for n, m, p in iproduct(range(D), repeat=3):
        if n + m + p > D:
            continue
        a = lat.random(n);  b = lat.random(m);  c = lat.random(p)
        LHS = wedge(wedge(a, b), c)
        RHS = wedge(a, wedge(b, c))
        assert not np.isclose(LHS, 0).all()
        assert np.isclose(LHS, RHS).all(), f"Associativity failed for {n},{m},{p}"
        print(f"  ✅ (a∧b)∧c = a∧(b∧c)  for degrees {n}, {m}, {p}")

    # --- Hodge star: inner product identity --------------------------------
    # Σ_{n,I} a_I[n] b_I[n]  =  Σ_n (a ∧ ★b)_{0,…,D-1}[n]
    print("\nHODGE STAR  Σ a·b = Σ (a∧★b)")
    for p in range(D + 1):
        a = lat.random(p)
        b = lat.random(p)
        LHS = (a * b).sum()
        RHS = wedge(a, star(b)).sum()
        assert np.isclose(LHS, RHS), \
            f"Hodge identity failed for p={p}: {LHS} vs {RHS}"
        print(f"  ✅ Σ a·b = Σ (a∧★b)  for p={p}")

    # --- to_interlaced round-trip ------------------------------------------
    # Verify that embedding a compact form into the (2N)^D array places
    # nonzero values only at sites with exactly p odd coordinates.
    print("\nTO_INTERLACED")
    TXYZ = np.mgrid[(slice(0, 2*N),) * D]
    odds = np.mod(TXYZ, 2).sum(axis=0)   # number of odd coords per site
    for p in range(D + 1):
        a = lat.random(p)
        big = a.to_interlaced()           # shape (2N,...,2N)
        # Nonzero only where exactly p coordinates are odd.
        wrong_sites = big[odds != p]
        assert np.all(wrong_sites == 0), f"to_interlaced put values at non-{p}-form sites"
        # All p-form sites are filled (random data is generically nonzero).
        right_sites = big[odds == p]
        assert not np.all(right_sites == 0), f"to_interlaced left {p}-form sites empty"
        print(f"  ✅ {p}-form embeds correctly into (2N)^D array")

    # --- Round-trip tests for from_interlaced / to_interlaced --------------
    print("\nROUND-TRIP: compact → interlaced → compact")
    for p in range(D + 1):
        f = lat.random(p)
        g = Form.from_interlaced(p, f.to_interlaced())
        assert (f == g).all(), f"compact→interlaced→compact failed for p={p}"
        print(f"  ✅ from_interlaced(to_interlaced(f)) == f  for p={p}")

    print("\nROUND-TRIP: interlaced → compact → interlaced")
    from supervillain.lattice import interlaced as il
    il_lat = il.Lattice(D, N)
    for p in range(D + 1):
        data = il_lat.random(p)
        assert (Form.from_interlaced(p, data).to_interlaced() == data).all(), \
            f"interlaced→compact→interlaced failed for p={p}"
        print(f"  ✅ to_interlaced(from_interlaced(data)) == data  for p={p}")

    # --- Cross-validation against interlaced.py ----------------------------
    print("\nCROSS-VALIDATION: compact d vs interlaced d")
    for p in range(D):
        a = lat.random(p)
        path_A = d(a).to_interlaced()
        path_B = il.d(a.to_interlaced())
        assert (path_A == path_B).all(), f"compact d ≠ interlaced d on {p}-forms"
        print(f"  ✅ compact d = interlaced d  on {p}-forms")

    print("\nCROSS-VALIDATION: compact δ vs interlaced δ")
    for p in range(1, D + 1):
        a = lat.random(p)
        path_A = delta(a).to_interlaced()
        path_B = il.delta(a.to_interlaced())
        assert (path_A == path_B).all(), f"compact δ ≠ interlaced δ on {p}-forms"
        print(f"  ✅ compact δ = interlaced δ  on {p}-forms")

    print("\nCROSS-VALIDATION: compact wedge vs interlaced wedge")
    for p, q in iproduct(range(D + 1), repeat=2):
        if p + q > D:
            continue
        a = lat.random(p)
        b = lat.random(q)
        path_A = wedge(a, b).to_interlaced()
        path_B = il.wedge(p, q, a.to_interlaced(), b.to_interlaced())
        # The two paths can take different floating-point arithmetic orderings,
        # so compare with np.isclose.  Other cross-validations should be bit-exact matches.
        assert np.isclose(path_A, path_B).all(), f"compact ∧ ≠ interlaced ∧ for {p}∧{q}"
        print(f"  ✅ compact wedge = interlaced wedge  for {p}∧{q}")

    print("\nCROSS-VALIDATION: compact ★ vs interlaced ★")
    for p in range(D + 1):
        a = lat.random(p)
        path_A = star(a).to_interlaced()
        path_B = il.star(p, a.to_interlaced())
        assert (path_A == path_B).all(), f"compact ★ ≠ interlaced ★ on {p}-forms"
        print(f"  ✅ compact ★ = interlaced ★  on {p}-forms")

    print("\nAll tests passed.")
