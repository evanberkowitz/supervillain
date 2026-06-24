
********
Lattices
********

We work on regular hypercubic lattices in $D$ dimensions with $N$ sites per direction and periodic boundary conditions.

.. note::

   The lattice is specified by a *single* $N$: every direction has the same number of sites.
   This equal-size (hypercubic) assumption is baked into much of the code — modular arithmetic
   on coordinates, the Fourier-transform and correlation helpers, and observables that normalize
   by a length or volume (for example :class:`~.VortexSusceptibilityScaled`, which scales by
   $N^{D}$).  Anisotropic lattices with a different number of sites in different directions are
   **not** supported.

.. _interlaced:

The Interlaced Picture
======================

Fields do not only live on sites.  A scalar field assigns a value to every site, but a gauge field assigns a value to every *link*, a field strength to every *plaquette*, and so on: a $p$-form (see :doc:`form`) assigns a value to every $p$-dimensional cell of the lattice.

Every cell has a natural geometric location.  A site sits at integer coordinates $n = (n_0, \ldots, n_{D-1})$.  The link leaving $n$ in direction $k$ is centered half a step away, at $n + \hat{e}_k/2$.  The plaquette anchored at $n$ spanning directions $j$ and $k$ is centered at $n + (\hat{e}_j + \hat{e}_k)/2$.  In general, the $p$-cell anchored at $n$ spanning the directions $I = (i_1 < i_2 < \cdots < i_p)$ is centered at $n + \frac{1}{2} \sum_{i \in I} \hat{e}_i$.

Doubling every coordinate clears the half-integers.  Measured in half-lattice units, the cell anchored at $n$ spanning the directions $I$ sits at

.. math::

   x = 2n + \mathbb{1}_I

where $\mathbb{1}_I$ has a 1 in the directions the cell spans and a 0 elsewhere.  Now every cell of every degree occupies a distinct integer point of a $(2N)^D$ lattice, and the *parity* of the coordinates says what kind of cell lives there: $x_k$ is odd exactly when the cell extends in direction $k$.  A $p$-cell is a point with exactly $p$ odd coordinates, and its anchoring site is always $n = \lfloor x/2 \rfloor$.

We call this doubled lattice *interlaced* because the cells of every degree are interleaved on a single grid.  The figure below shows the four cells anchored at the origin of a two-dimensional lattice: the site at interlaced coordinates $(0,0)$, the two links at $(1,0)$ and $(0,1)$, and the plaquette at $(1,1)$.

.. plot:: example/plot/lattice/layout.py

The interlaced picture is how the code "thinks" geometrically: which cells are incident to which, which neighbor a value is gathered from, and where the operators of the exterior calculus get their :ref:`shifts and signs <sign-conventions>`.  The *storage*, however, is compact: a point with exactly $p$ odd coordinates is rare (only $\binom{D}{p}/2^D$ of the doubled lattice), so a $p$-form is stored as an array of shape $(\binom{D}{p}, N, \ldots, N)$ whose 0th axis enumerates the components — the sorted direction tuples $I$, in lexicographic order — and whose remaining axes give the anchoring site $n$.  See :class:`~supervillain.lattice.Form`.

A reference implementation that stores forms directly as sparse $(2N)^D$ interlaced arrays is documented with the other :ref:`reference implementations <reference_implementations>`.
But even on small 4-dimensional lattices the interlaced representation is much slower than the compact representation, so we use the compact representation for all practical purposes.

Arbitrary Dimensions
====================

We provide common machinery for working with (hyper)cubic lattices in arbitrary dimensions.
For some purposes (like plotting) it is useful to have a specialization; for example below we provide :class:`~supervillain.lattice.Lattice2D` for two dimensions.

.. autoclass :: supervillain.lattice.Lattice
   :members:

Two Dimensions
==============

.. autoclass :: supervillain.lattice.Lattice2D
   :members:
