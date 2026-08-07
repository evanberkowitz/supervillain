
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
Every lattice site can be shifted, every axis can be reflected through the origin, and every axis can be relabelled by a permutation, and each of these leaves the lattice looking exactly like itself.
Together these generate the *space group*

.. math ::

   G = \mathbb{Z}_N^D \rtimes \left(\{\pm 1\}^D \rtimes S_D\right),
   \qquad |G| = N^D \cdot 2^D \cdot D!

--- translations, semidirect the hyperoctahedral point group $\{\pm 1\}^D \rtimes S_D$ of sign flips and axis permutations.
This is *not* the Poincaré group: there are no boosts, the signature is Euclidean, and the group is discrete rather than continuous.

.. note ::

   $S_D$ is not the rotation subgroup of the point group.
   An odd permutation has determinant $-1$; the transposition $x_{\mu} \leftrightarrow x_{\nu}$, for instance, is the mirror reflection across the diagonal hyperplane $x_{\mu} = x_{\nu}$.
   And in even $D$ the point inversion $-I$ has determinant $(-1)^D = +1$, so it lies *inside* the rotation subgroup rather than pairing with it the way $\{I, -I\}$ does in odd dimensions --- the familiar 3-dimensional factorization "full point group = rotations $\times \{I, -I\}$" does not hold here.

:func:`~supervillain.lattice.translate`, :func:`~supervillain.lattice.reflect`, and :func:`~supervillain.lattice.permute` implement the three factors of $G$ directly on a :class:`~supervillain.lattice.Form` of any degree, taking $\omega \mapsto \omega'$ for a group element $g$.
Because a $p$-form's components are labelled by sorted direction tuples $I$ rather than raw axis indices, applying $g$ is more than moving array data around: a reflection or permutation can also relabel *which* component a value belongs to, and can introduce a sign.

Translation is the simple case: $\varphi'(x) = \varphi(x - a)$ for every degree, with no component relabelling and no sign, because translating a cell never changes which directions it spans.

A reflection negating the axes in a set $F$ keeps a component's direction label $I$ fixed --- sign flips do not permute directions --- but two subtleties appear.
First, an orientation sign $(-1)^{\left|I \cap F\right|}$: the cell spans a direction in $F$ for each element of $I \cap F$, and flipping the direction a cell spans reverses its orientation.
Second, a base-point shift: a cell whose edge runs in a flipped direction lands on the far side of its own image under the flip, so evaluating the transformed form at $y$ means evaluating the original at $y$ shifted by $\hat e_{\mu}$ for each $\mu \in I \cap F$, *before* negating.
Explicitly,

.. math ::

    \omega'_I(y) = (-1)^{|I \cap F|}\;
    \omega_I\!\left(R\!\left(y + \sum_{\mu \in I \cap F} \hat e_\mu\right)\right)

where $R$ negates each coordinate in $F$ modulo $N$.
The :ref:`interlaced <interlaced>` picture is where this rule is *derived*, not merely stated: there a component with directions $I$ sits at $\xi_{k} = 2x_{k} + [k \in I]$, and for both $\mu \notin I$ (even $\xi_{\mu} = 2x_{\mu}$) and $\mu \in I$ (odd $\xi_{\mu} = 2x_{\mu} + 1$) the map $x_{\mu} \to -x_{\mu}$ collapses to the SAME single coordinate negation $\xi_{\mu} \to -\xi_{\mu}$.
The base-point shift that has to be made explicit on the compact array is automatic there; only the orientation sign remains to track, and it is applied at $I$ itself since flips never permute directions.
See :func:`supervillain.lattice.interlaced.reflect`.

A permutation $\pi$ relabels axis $\mu$ to axis $\pi(\mu)$, and because components are always stored under their *sorted* direction tuple, this does two things at once: it moves a component's data to the tuple $\mathrm{sort}(\pi(I))$, and --- whenever $\pi(I)$ is not already sorted --- it multiplies by the sign of the permutation that sorts it,

.. math ::

    \omega'_{\pi(I)}(Rx) = \omega_I(x) \;\Longrightarrow\;
    \omega'_{\mathrm{sort}(\pi(I))} = \varepsilon\, \omega_I

The sign belongs at the DESTINATION component $\mathrm{sort}(\pi(I))$, not at the source $I$: the two coincide whenever $\pi(I)$ happens to already be sorted, which is *always* true when $D = 2$, so a two-dimensional test cannot distinguish correct code from this particular bug --- it only shows up once $D \geq 3$.
The interlaced picture makes the position half of this automatic (a plain axis transpose moves both the site and the parity pattern together) but not the sign, which must still be applied at the destination; see :func:`supervillain.lattice.interlaced.permute`.

.. autofunction :: supervillain.lattice.translate

.. autofunction :: supervillain.lattice.reflect

.. autofunction :: supervillain.lattice.permute
