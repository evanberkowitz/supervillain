
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

.. _space-group:

Lattice Symmetries
===================

The periodic hypercubic lattice has more structure than translation invariance alone.
Every lattice site can be shifted, each axis can be reflected across the coordinate hyperplane through the origin, and all the axes can be relabelled by a permutation; each of these leaves the lattice looking exactly like itself.
Together these generate the *space group*

.. math ::

   G = \mathbb{Z}_N^D \rtimes \left(\{\pm 1\}^D \rtimes S_D\right),
   \qquad |G| = N^D \cdot 2^D \cdot D!

--- the semidirect product of translations and the hyperoctahedral point group $\{\pm 1\}^D \rtimes S_D$ of sign flips and axis permutations.

.. note ::

   $S_D$ is not the rotation subgroup of the point group.
   An odd permutation has determinant $-1$: the transposition $x_{\mu} \leftrightarrow x_{\nu}$ is the mirror reflection across the diagonal hyperplane $x_{\mu} = x_{\nu}$.
   In even $D$ the point inversion $-I$ has determinant $(-1)^{D} = +1$ and so lies inside the rotation subgroup, and the point group does not factor as rotations $\times \{I, -I\}$ the way it does in three dimensions.

:func:`~supervillain.lattice.translate`, :func:`~supervillain.lattice.reflect`, and :func:`~supervillain.lattice.permute` implement the three factors of $G$ on a :class:`~supervillain.lattice.Form` of any degree.

A $p$-form assigns a value to every $p$-cell, and a cell is specified by an anchoring site together with the directions it spans.
A symmetry can move both.
It carries a cell to another cell, which changes the anchoring site, and it may change which directions the cell spans, which changes the component the value is stored under.
Because components are labelled by sorted direction tuples, and because a cell carries an orientation, that relabelling can also introduce a sign.

Translation is the simple case.
Shifting by $a$ never changes which directions a cell spans, so for every degree

.. math ::

    \omega'(n) = \omega(n - a)

with no component relabelling and no sign.

.. autofunction :: supervillain.lattice.translate

A reflection negating the axes in a set $F$ leaves the direction label $I$ alone, since sign flips do not permute directions, but it changes both the orientation of a cell and where the cell sits.
Each direction in $I \cap F$ is one the cell spans and the reflection reverses, so the orientation flips once per element of $I \cap F$.
The anchor of a cell is its minimal corner, and a reflection sends the minimal corner to the maximal one, so the reflected cell is anchored one step back along each flipped direction it spans:

.. math ::

    \omega'_I(y) = (-1)^{|I \cap F|}\;
    \omega_I\!\left(R\!\left(y + \sum_{\mu \in I \cap F} \hat e_\mu\right)\right)

where $R$ negates each coordinate in $F$ modulo $N$.

The :ref:`interlaced <interlaced>` picture explains where that shift comes from.
A cell anchored at $n$ spanning $I$ sits at interlaced coordinates $x_{k} = 2 n_{k} + [k \in I]$, and $n_{\mu} \to -n_{\mu}$ sends $x_{\mu} \to -x_{\mu}$ whether or not $\mu \in I$: for $\mu \notin I$ the coordinate $2 n_{\mu}$ simply negates, while for $\mu \in I$ the image cell is anchored at $2(-n_{\mu} - 1) + 1 = -(2 n_{\mu} + 1)$.
On the doubled lattice a reflection is one coordinate negation and the anchor moves along with it.
Compact storage separates the site from the component, so the same motion has to be written out as an explicit shift.

.. autofunction :: supervillain.lattice.reflect

A permutation $\pi$ sends axis $\mu$ to axis $\pi(\mu)$, so a cell spanning $I$ maps to one spanning $\pi(I)$.
Since components are stored under sorted tuples, the value belongs to $\mathrm{sort}(\pi(I))$, and putting it in order costs the sign of the permutation that sorts it:

.. math ::

    \omega'_{\mathrm{sort}(\pi(I))}(\pi n) = \varepsilon\, \omega_I(n)

The same $\pi$ acts on the lattice axes and on the component index.
This is the antisymmetry of the wedge basis.
Degrees 0 and 1 have no pair of directions to reorder, so $\varepsilon \equiv +1$ there; from degree 2 up the sign matters.
In the interlaced picture the relabelling is a plain transpose of the doubled axes, carrying the site and the spanned directions together, while the sign --- which records an ordering convention rather than a position --- stays explicit.



.. autofunction :: supervillain.lattice.permute
