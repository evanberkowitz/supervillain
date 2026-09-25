.. _barycubic:

*****************
Barycubic Lattice
*****************

The :doc:`hypercubic lattice <lattice>` carries an exterior calculus with an exact $d$, a wedge product that obeys the Leibniz rule, and a Hodge star.
Its wedge is a *cubical* cup product: on each cell it is a sum over shuffles of directions, with one factor displaced along the other's directions.
Subdividing every hypercube into simplices trades that for the *simplicial* products, the Alexander–Whitney cup $\cup$ and Steenrod's higher products $\cup_i$, which read their two factors on faces of a single simplex: for $\cup$ a front face and a back face that share one vertex, and for $\cup_i$ a sum of pairs of faces that share $i+1$ vertices.
The subdivided lattice is the barycubic lattice.
This page explains the construction, the conventions, and the operators, in the same order as the :doc:`hypercubic forms <form>`.

.. contents::
   :local:
   :depth: 1

Why Subdivide
=============

On a simplicial complex whose simplices carry an ordering of their vertices, the Alexander–Whitney cup product is local: $(a \cup b)$ on a simplex reads $a$ on one face and $b$ on another face of that *same* simplex.
It satisfies the Leibniz rule $d(a \cup b) = da \cup b + (-1)^{p} a \cup db$ exactly, and it comes with the Steenrod products $\cup_i$ that measure its failure to be graded-commutative.

For a 2-form $f$ in four dimensions, the intersection charge $q = f \cup f$ is then a single product evaluated on each top simplex,

.. math::

   q(v_0 v_1 v_2 v_3 v_4) = f(v_0 v_1 v_2)\, f(v_2 v_3 v_4),

and nothing has to cancel between different terms inside one constraint cell.
The barycentric subdivision of the hypercubic lattice is a simplicial complex of this kind that refines the hypercubic lattice and keeps all of its symmetries; we call it the barycubic lattice.

The Construction
================

Start from the periodic hypercubic lattice $(\mathbb{Z}/N)^D$, viewed as a cell complex: sites, links, plaquettes, cubes, and so on up to hypercubes.
Its *barycentric subdivision* has

* one fine vertex $b(F)$ at the barycenter of every cell $F$, and
* one fine p-simplex for every **flag** $F_0 \subsetneq F_1 \subsetneq \cdots \subsetneq F_p$ of cells, with vertices $b(F_0), \ldots, b(F_p)$.

The **rank** of the fine vertex $b(F)$ is the dimension of $F$: sites have rank 0, link midpoints rank 1, plaquette centers rank 2, and so on.
The vertices of a fine simplex come from a flag of cells of strictly increasing dimension, so they have *distinct* ranks.
Listing them by rank therefore orders every simplex, and deleting a vertex from a rank-ordered list leaves a rank-ordered list, so the orders are compatible with faces.
This **rank order** is the local ordering that the products need.

The figure shows two unit squares of the two-dimensional lattice, each cut into $2^2 \cdot 2! = 8$ fine triangles.
Every fine triangle has one site, one link midpoint, and one plaquette center; the highlighted one is shown with its vertices in rank order.
(The fine links along the original links, from a site to a link midpoint, lie under the black lines.)

.. plot:: example/plot/barycubic-2d.py

A fine p-simplex is determined by its flag, and counting flags gives the number of fine p-simplices per site,

.. math::

   f_p(D) = \sum_{S_0 \subsetneq \cdots \subsetneq S_p \subseteq \{0..D-1\}} 2^{|S_p \setminus S_0|}
          = \sum_{j=0}^{p} (-1)^{p-j} \binom{p}{j} (2j+2)^D.

Each hypercube is cut into $f_D = 2^D D!$ top simplices, and $f_1 = 4^D - 2^D$.
In four dimensions,

.. list-table::
   :header-rows: 1

   * - $D=4$
     - $p=0$
     - $1$
     - $2$
     - $3$
     - $4$
     - Euler characteristic
   * - $f_p$
     - 16
     - 240
     - 800
     - 960
     - 384
     - 0

so there are 60 times as many fine links as hypercubic links.
These counts are checked against an independent, slow construction that enumerates cells and flags explicitly (:source:`test/barycubic_reference.py`), by ``test_counts`` in :source:`test/test_barycubic_complex.py` and ``test_reference_counts_match_inclusion_exclusion`` in :source:`test/test_barycubic_reference.py`.
For every $N \geq 2$ the subdivision is a genuine simplicial complex after the periodic identification (distinct simplices have distinct vertex sets), as checked by ``test_simplicial_complex_invariant``; its cohomology is that of the torus, as checked block by block in momentum space by ``test_fourier_topology``, both in :source:`test/test_barycubic_complex.py`.

.. autoclass :: supervillain.lattice.barycubic.BarycubicLattice
   :members: sites, dims, dim, cells_of_degree, rank_sets, local_vertices, zeros, random, form
   :show-inheritance:

Doubled Coordinates and Labels
==============================

The barycenter of a cell lies on the half-integer lattice, so, as in :ref:`the interlaced picture <interlaced>`, we double every coordinate.
In doubled coordinates $\xi = 2x$ every fine vertex is an integer point of $(\mathbb{Z}/2N)^D$, and every such point is a fine vertex: the fine vertices are exactly the interlaced lattice, $2^D$ per site.
The parity of $\xi$ tells us which cell it is the barycenter of:

* the **type** $T(\xi) = \{k : \xi_k \text{ odd}\}$ is the set of directions the cell spans;
* the **rank** is $\lvert T(\xi)\rvert$;
* the **anchor site** is $\lfloor \xi / 2 \rfloor$.

A fine p-simplex has rank-ordered vertices $\xi^0, \ldots, \xi^p$ with strictly nested types $S_0 \subsetneq \cdots \subsetneq S_p$.
Its anchor site $x$ is that of its top vertex, $\xi^p = 2x + \mathbb{1}_{S_p}$.
Each lower cell is a face of the top cell, displaced from its center by half a step along every direction it does not span, towards one side or the other.
Recording that side as a sign $\sigma_k = \pm 1$ for each $k \in S_p \setminus S_0$ gives

.. math::

   \xi^a_k = \begin{cases}
      \xi^p_k + \sigma_k & k \in S_p \setminus S_a, \\
      \xi^p_k            & \text{otherwise.}
   \end{cases}

So a fine simplex is its anchor $x$ plus a **local label** $(S_0, \ldots, S_p;\, \sigma)$, which says what the simplex looks like relative to its anchor.
The highlighted triangle in the figure is anchored at the origin with label $(\varnothing, \{0\}, \{0,1\};\, \sigma_0 = \sigma_1 = -1)$: its top vertex is $\xi^2 = (1,1)$, and $\xi^1 = (1, 0)$, $\xi^0 = (0, 0)$.

.. autoclass :: supervillain.lattice.barycubic.LocalLabel
   :members: masks, signs

Storage
-------

Because every fine simplex is a translate of one of the $f_p$ labels anchored at the origin, a p-cochain is stored exactly like a hypercubic :class:`~supervillain.lattice.Form`, with the label in place of the component:
an array of shape $(f_p, N, \ldots, N)$, indexed ``[label, x]``.
Labels are ordered lexicographically by $(\text{mask}(S_0), \ldots, \text{mask}(S_p), \text{bits}(\sigma))$, where $\text{mask}(S) = \sum_{k \in S} 2^k$ and the bits are 0 for $\sigma_k = -1$ and 1 for $\sigma_k = +1$, in increasing $k$; ``lattice.labels[p]`` lists them and ``lattice.label_index[p]`` inverts the list.
The ordering and indexing are checked by ``test_labels_are_ordered_and_indexed`` and the vertex rule by ``test_local_vertices_follow_label_rule``, both in :source:`test/test_barycubic_complex.py`.

The stored value is the cochain evaluated on the vertices **in rank order**, $\omega[\sigma] = \omega(\xi^0 \ldots \xi^p)$.
Evaluating it on the same vertices in another order costs the sign of the permutation, as for any simplicial cochain.

Forms hold signed integers, floats, or complex numbers.
The operators keep signed-integer forms exactly integer, and refuse unsigned-integer and boolean forms with a ``TypeError``, because the signs in $d$ and the products would wrap around; cast such data to a signed or floating type first.
They also refuse anything that is not a whole form: a slice or reshape of a form is still an instance of the class, but it is not a cochain, and passing one raises ``ValueError``.
Degrees outside $0, \ldots, D$ raise ``ValueError`` everywhere.
These rules are checked by ``test_unsigned_and_bool_forms_rejected``, ``test_int32_forms_stay_integer_and_exact``, ``test_operators_reject_wrong_shapes``, and ``test_degree_out_of_range_rejected`` in :source:`test/test_barycubic_operators.py`.

.. autoclass :: supervillain.lattice.barycubic.BarycubicForm
   :members: dual, spatial_shape
   :show-inheritance:

.. _barycubic-common-lift:

The Common Lift
---------------

Stored vertex coordinates are reduced mod $2N$, but every *geometric* quantity (displacements, orientations, volumes, centroids) must be computed from the **common lift**: the unreduced integers $\xi^p = 2x + \mathbb{1}_{S_p}$ with $x \in [0, N)^D$ and $\xi^a = \xi^p + \sum_{k \in S_p \setminus S_a} \sigma_k e_k$.
:meth:`~supervillain.lattice.barycubic.BarycubicLattice.local_vertices` returns exactly these lifted vertices for the simplices anchored at the origin, with entries in $\{0, 1, 2\}$.

The reason is the periodic seam.
Take $D = 1$ and $N = 2$, so the doubled circle has four points $0, 1, 2, 3$.
The link anchored at $x = 1$ with $\sigma = +1$ has top vertex $\xi^1 = 3$ (the midpoint of the link from site 1 to site 2) and lower vertex $\xi^0 = 3 + 1 = 4$, which is stored as $0$.
The reduced coordinates differ by $3 - 0 = 3$, but the true displacement from $\xi^0$ to $\xi^1$ is $3 - 4 = -1$ doubled units.
Using reduced differences would, for example, make the winding cochain $w(e) = \Delta\xi(e)/2$ an exact cochain with period 0 instead of a closed one with period $N$.
The seam case is a regression test, ``test_seam_regression_d1_n2`` in :source:`test/test_barycubic_cup.py`; the winding cochains are checked by ``test_winding_cochains_are_closed_with_period_N`` in the same file.

Orientation and Integration
===========================

Two different orientations are in play, and it helps to keep them apart.

* The **reference orientation** of a fine simplex is its rank order.  It is purely combinatorial, is inherited by faces, and is what the stored values, $d$, and the products refer to.
* The **spacetime orientation** of a top simplex says whether its rank order is positively or negatively oriented in $\mathbb{R}^D$.  This is what integration needs.

In the common lift the rank-ordered edges of a top simplex are single steps: if the type grows by the direction $k_a$ between ranks $a$ and $a+1$, then $\xi^{a+1} - \xi^a = -\sigma_{k_a} e_{k_a}$.
The sign of the determinant of these edges is therefore

.. math::

   \varepsilon_\sigma = \mathrm{sgn}(k_0, \ldots, k_{D-1}) \prod_{k} (-\sigma_k),

exposed as the integer top form :attr:`~supervillain.lattice.barycubic.BarycubicLattice.orientation`.
In the figure, the highlighted triangle steps $+e_0$ then $+e_1$ and has $\varepsilon = +1$; its mirror image across the diagonal steps $+e_1$ then $+e_0$ and has $\varepsilon = -1$.
Half of the top simplices of every hypercube have each sign.

The integral of a top form pairs it with the fundamental cycle $[M] = \sum_\sigma \varepsilon_\sigma \sigma$,

.. math::

   \int f = \sum_\sigma \varepsilon_\sigma f_\sigma .

The fundamental cycle has no boundary, as checked by ``test_fundamental_cycle_is_closed`` in :source:`test/test_barycubic_cup.py`, so $\int df = 0$.
The orientation agrees with a determinant computed from the reference construction, as checked by ``test_orientation_matches_reference`` in :source:`test/test_barycubic_reference.py`.

.. autoproperty :: supervillain.lattice.barycubic.BarycubicLattice.orientation

.. autofunction :: supervillain.lattice.barycubic.integrate

Coboundary
==========

The coboundary is the standard simplicial one, applied to rank-ordered vertex lists,

.. math::

   (d\omega)(v_0 \ldots v_{p+1}) = \sum_{j=0}^{p+1} (-1)^j\, \omega(v_0 \ldots \widehat{v_j} \ldots v_{p+1}).

On the highlighted triangle of the figure, for a 1-form $a$, this reads $(da)(v_0v_1v_2) = a(v_1v_2) - a(v_0v_2) + a(v_0v_1)$.

Deleting a vertex from a rank-ordered list keeps it rank ordered, so each face value is read straight from storage, with no reordering sign.
Only deleting the top vertex can move the anchor, and then only by $+1$ in some directions, so every face of a simplex anchored at $x$ is anchored in $x + \{0,1\}^D$.
That makes $d$ a *local stencil* over labels: the table returned by :meth:`~supervillain.lattice.barycubic.BarycubicLattice.operator_table` lists, for each output label, the input label, the anchor offset, and the sign of each face.

$d \circ d = 0$ exactly, as checked by ``test_d_nilpotent`` in :source:`test/test_barycubic_operators.py`, and $d$ agrees with the reference construction entry by entry, as checked by ``test_d_matches_reference`` in :source:`test/test_barycubic_reference.py`.
Like the hypercubic :func:`~supervillain.lattice.d`, it returns the scalar ``0`` on a top form and keeps integer forms integer (``test_d_top_degree_is_scalar_zero`` and ``test_integer_forms_stay_integer`` in :source:`test/test_barycubic_operators.py`).

.. autofunction :: supervillain.lattice.barycubic.d

.. automethod :: supervillain.lattice.barycubic.BarycubicLattice.operator_table

Weights and the Hodge Star
==========================

The Barycentric Dual Cell
-------------------------

A kinetic term $\sum (d\phi)^2$ needs a metric, which on a simplicial complex enters through a Hodge star.
We use a **diagonal** star: one positive weight $M_p(\sigma)$ per fine simplex,

.. math::

   M_p(\sigma) = \frac{\mathrm{Vol}_{D-p}(\star\sigma)}{\mathrm{Vol}_p(\sigma)},
   \qquad
   \mathrm{Vol}_k = \frac{\sqrt{\det \mathrm{Gram}}}{k!}, \quad \mathrm{Vol}_0 = 1,

where volumes are Euclidean and $\star\sigma$ is the **barycentric dual cell** of $\sigma$.
The dual cell is built from maximal chains of fine simplices that start at $\sigma$, $\sigma = \tau_0 \subsetneq \tau_1 \subsetneq \cdots \subsetneq \tau_{D-p}$: each chain contributes the $(D-p)$-simplex spanned by the centroids $c(\tau_0), \ldots, c(\tau_{D-p})$, and $\star\sigma$ is their union.
These pieces are not coplanar in general, so $\mathrm{Vol}(\star\sigma)$ is the sum of their volumes.
At the extremes, the dual cell of a top simplex is a point, so $M_D = 1/\mathrm{Vol}_D = 2^D D!$, and the dual cell of a vertex is a $D$-dimensional region whose volume is $M_0$.

The hyperoctahedral group and the translations act transitively on the simplices with a given **rank set** $R(\sigma) = \{\lvert S_0\rvert, \ldots, \lvert S_p\rvert\}$, so $M_p$ depends only on $R$.
There are $2^{D+1} - 1$ rank sets, and :attr:`~supervillain.lattice.barycubic.BarycubicLattice.hodge_weight_by_rank_set` computes one weight for each.
The half-diagonal translation (:ref:`below <barycubic-symmetries>`) exchanges rank $r$ with $D - r$ isometrically, so $M(R) = M(D - R)$; in $D = 4$ that leaves 18 distinct values among the 31 rank sets.
The vertex weights have the closed form $M_0(r) = 1/\big((D+1)\binom{D}{r}\big)$, so the vertex dual cells of one site have total volume exactly 1 and tile the torus.

The ten link weights in $D = 4$, in units of the hypercubic lattice spacing, are

.. list-table::
   :header-rows: 1

   * - rank classes $(r, s)$
     - links per site
     - length $\lvert e\rvert$
     - weight $M_1 = \lvert \star e\rvert  / \lvert e\rvert$
   * - $(0,1)$, $(3,4)$
     - 8, 8
     - $1/2$
     - $\sqrt5/10 \approx 0.224$
   * - $(1,2)$, $(2,3)$
     - 24, 24
     - $1/2$
     - $\sqrt6/30 \approx 0.082$
   * - $(0,2)$, $(2,4)$
     - 24, 24
     - $\sqrt2/2$
     - $\sqrt6/60 \approx 0.041$
   * - $(0,3)$, $(1,4)$
     - 32, 32
     - $\sqrt3/2$
     - $1/40$
   * - $(0,4)$
     - 16
     - $1$
     - $\sqrt2/40 \approx 0.035$
   * - $(1,3)$
     - 48
     - $\sqrt2/2$
     - $\sqrt2/60 \approx 0.024$

A link of class $(r, s)$ joins a rank-$r$ vertex to a rank-$s$ vertex and has length $\sqrt{s-r}/2$.

Why Diagonal, and Not Exact
---------------------------

A diagonal star makes the kinetic term a sum of independent squares, $\sum_\sigma M_p(\sigma)\, \omega_\sigma^2$, so that changing one value changes only one term.
It is a consistent discretization but not an exact Riemann sum, because the barycentric dual cell of a fine simplex is not orthogonal to it.
The size of the discrepancy is measured by the **affine normalization**.
For an affine field $\phi = k \cdot x$ the kinetic term per site is $\mathcal{N} \lvert k\rvert^2$, where, by the cubic symmetry,

.. math::

   \mathcal{N} = \sum_{\text{links } e \text{ per site}} M_1(e)\, (\Delta x_0(e))^2
   = \tfrac{1}{10}\big(\sqrt5 + 2\sqrt6 + 3 + 2\sqrt2\big) \approx 1.2963

in $D = 4$, rather than the continuum value $1$.
For slowly varying fields the discrepancy is therefore a constant rescaling of the coupling.

The weight table, the sum rules, and this normalization are checked by ``test_weight_sum_rules``, ``test_boundary_degree_weights``, and ``test_d4_link_weights_and_affine_normalization`` in :source:`test/test_barycubic_complex.py`, and the weights are compared with an independent geometric construction of the dual cells by ``test_weights_match_geometric_reference`` in :source:`test/test_barycubic_reference.py`.

.. autoproperty :: supervillain.lattice.barycubic.BarycubicLattice.hodge_weight_by_rank_set

.. automethod :: supervillain.lattice.barycubic.BarycubicLattice.hodge_weights

Inner Product, Codifferential, and Laplacian
--------------------------------------------

The weights define the inner product

.. math::

   \langle a, b \rangle = \sum_\sigma M_p(\sigma)\, \overline{a_\sigma}\, b_\sigma,

which is Hermitian and positive definite on complex forms.
The codifferential is the adjoint of $d$ in this inner product,

.. math::
   :label: barycubic-d-delta-adjoint

   \delta = M_{p-1}^{-1}\, d^\top M_p,
   \qquad
   \langle d a, b \rangle = \langle a, \delta b \rangle,

the same adjointness convention as the hypercubic :eq:`d-delta-formal-adjoint`; $d^\top$ is the signed sum over cofaces.
The Laplacian is $\Delta = d\delta + \delta d$.
Unlike the hypercubic :func:`~supervillain.lattice.laplacian`, it does not act diagonally on labels, because different simplex classes carry different weights.

The adjointness is checked by ``test_delta_is_adjoint_of_d``, $\delta^2 = 0$ by ``test_delta_nilpotent_and_typed``, the definition of the Laplacian by ``test_laplacian_is_d_delta_plus_delta_d``, and its self-adjointness and positivity by ``test_laplacian_self_adjoint_and_positive``, all in :source:`test/test_barycubic_operators.py`.
Its kernel on the torus is the $\binom{D}{p}$-dimensional space of harmonic forms, as checked by ``test_fourier_topology`` in :source:`test/test_barycubic_complex.py`.

.. autofunction :: supervillain.lattice.barycubic.inner

.. autofunction :: supervillain.lattice.barycubic.delta

.. autofunction :: supervillain.lattice.barycubic.laplacian

Star and Dual Forms
-------------------

The star itself multiplies by the weights and returns a :class:`~supervillain.lattice.barycubic.BarycubicDualForm`, a $(D-p)$-cochain on the dual cells.
Dual cells correspond one-to-one to primal simplices, so a dual form is stored on the primal labels; what changes is its type and its ``degree``.
It exists so that a kinetic term can be written as a pairing, $\langle \xi, \xi \rangle = \sum_\sigma \overline{\xi_\sigma}\, (\star\xi)_{\star\sigma}$, i.e. ``(xi.conj() * star(xi)).sum()``; for real fields the conjugate does nothing.
No dual complex is built, so there is no $d$ on dual forms; the unsupported operations raise ``TypeError``.
These properties are checked by ``test_star_is_typed_dual_form``, ``test_inverse_star_undoes_star``, ``test_dual_arithmetic_and_pairing``, and ``test_unsupported_dual_operations_raise`` in :source:`test/test_barycubic_operators.py`.

.. autofunction :: supervillain.lattice.barycubic.star

.. autofunction :: supervillain.lattice.barycubic.inverse_star

.. autoclass :: supervillain.lattice.barycubic.BarycubicDualForm
   :members: dual, primal_degree, spatial_shape
   :show-inheritance:

Cup and Steenrod Products
=========================

The Cup Product
---------------

The Alexander–Whitney cup product of a p-form $a$ and a q-form $b$ evaluates $a$ on the *front* face and $b$ on the *back* face of each rank-ordered $(p+q)$-simplex, which share the single vertex $v_p$:

.. math::

   (a \cup b)(v_0 \ldots v_{p+q}) = a(v_0 \ldots v_p)\, b(v_p \ldots v_{p+q}).

For two 1-forms on the highlighted triangle of the figure, $(a \cup b)(v_0 v_1 v_2) = a(v_0 v_1)\, b(v_1 v_2)$: $a$ is read on the link from the site to the link midpoint, and $b$ on the link from the link midpoint to the plaquette center.
For a 2-form $f$ in four dimensions, $(f \cup f)(v_0 \ldots v_4) = f(v_0 v_1 v_2)\, f(v_2 v_3 v_4)$, the factorization of :ref:`the introduction <barycubic>`, checked by ``test_plaquette_factorization`` in :source:`test/test_barycubic_complex.py`.

The cup product is bilinear, translation covariant, associative, and obeys the Leibniz rule

.. math::

   d(a \cup b) = da \cup b + (-1)^p\, a \cup db

exactly, as checked by ``test_cup_bilinear_for_every_i``, ``test_cup_translation_covariant``, ``test_cup_associative``, and ``test_cup_leibniz`` in :source:`test/test_barycubic_cup.py`.
Unlike the hypercubic :func:`~supervillain.lattice.wedge`, it is defined through the vertex order alone, with no shuffles and no shifts.
It computes the ring structure of the torus's cohomology: for the winding 1-cochains $w_\mu$, whose period around the $\mu$ cycle is $N$,

.. math::

   \int w_{\pi(0)} \cup \cdots \cup w_{\pi(D-1)} = \mathrm{sgn}(\pi)\, N^D

exactly, as checked by ``test_cohomology_ring`` in :source:`test/test_barycubic_cup.py`.

Steenrod's Higher Products
--------------------------

The cup product is not graded commutative at the level of cochains; $a \cup b$ and $(-1)^{pq} b \cup a$ differ by a coboundary and a correction.
Steenrod's $\cup_1$ product is that correction, $\cup_2$ corrects $\cup_1$, and so on.
The product $a \cup_i b$ of a p-form and a q-form is a $(p+q-i)$-form, defined for $0 \leq i \leq \min(p, q)$.
On a rank-ordered simplex $(v_0 \ldots v_n)$ with $n = p + q - i$,

.. math::

   (a \cup_i b)(v_0 \ldots v_n) = \sum_{J = (j_0 < \cdots < j_i) \subseteq \{0..n\}}
       \varepsilon(J)\; a(v_A)\; b(v_B).

The $i+1$ **cut points** $J$ split $0, 1, \ldots, n$ into $i+2$ intervals that share their endpoints.
The even-numbered intervals form $a$'s face and the odd-numbered ones form $b$'s face,

.. math::

   A = [0, j_0] \cup [j_1, j_2] \cup [j_3, j_4] \cup \cdots,
   \qquad
   B = [j_0, j_1] \cup [j_2, j_3] \cup \cdots,

with the last interval ending at $n$, and terms whose faces have the wrong sizes, $\lvert A\rvert \neq p+1$ or $\lvert B\rvert \neq q+1$, are omitted.
The cut points belong to both faces.
For $i = 0$ there is one cut point, $j_0 = p$, and this is the Alexander–Whitney product.

The sign is

.. math::

   \varepsilon(J) = (-1)^{\binom{p}{2} + \binom{n}{2} + \sum_{u \in U_B} \mathrm{ind}_U(u)},

where $U = \{0..n\} \setminus J$ are the uncut positions, $\mathrm{ind}_U(u) = \#\{w \in U : w < u\}$ is the position of $u$ among them, and $U_B = \{u \in U : u - \mathrm{ind}_U(u) \text{ odd}\}$ are the uncut positions below an odd number of cuts, which appear only in $b$'s face.
With these signs the products obey the Steenrod coboundary formula

.. math::
   :label: barycubic-steenrod

   d(a \cup_i b) = da \cup_i b + (-1)^p\, a \cup_i db
       + (-1)^{p+q-i}\, a \cup_{i-1} b + (-1)^{pq+p+q}\, b \cup_{i-1} a,

with $\cup_{-1} = 0$ and terms of degree above $D$ dropped.
For $i = 0$ it is the Leibniz rule, and for $i \geq 1$ the last two terms say that $\cup_i$ is a homotopy between $\cup_{i-1}$ and its transpose.
The top product $\cup_p$ of two p-forms is the pointwise product, $a \cup_p b = a\, b$.

.. admonition:: Worked example: ∪₁ of two 1-cochains on a triangle

   Write $a_{ij} = a(v_i v_j)$ for a 1-cochain on the rank-ordered triangle $(v_0 v_1 v_2)$.

   * For two 1-cochains, $n = 1$ and the only cut is $J = (0, 1)$, with $A = B = (0, 1)$ and sign $+1$: $a \cup_1 b = a\, b$ on every link, the pointwise top product.
   * For a 2-cochain $F$ and a 1-cochain $b$ ($p = 2$, $q = 1$, $n = 2$), two cuts survive, $J = (0,1)$ with $B = (0,1)$ and $J = (1,2)$ with $B = (1,2)$, both with sign $(-1)^{1 + 1 + 0} = +1$:

     .. math::

        (F \cup_1 b)(v_0 v_1 v_2) = F_{012}\,(b_{01} + b_{12}).

   * In the other order ($p = 1$, $q = 2$), the one surviving cut $J = (0, 2)$ gives $A = (0, 2)$, $B = (0,1,2)$, and $U = \{1\}$ with $1 - 0$ odd, so the sign is $(-1)^{0 + 1 + 0} = -1$:

     .. math::

        (b \cup_1 F)(v_0 v_1 v_2) = -\,b_{02}\, F_{012}.

   Now evaluate :eq:`barycubic-steenrod` for two 1-cochains $a$, $b$ with $i = 1$ on the triangle.
   The left side is $d(ab)(v_0 v_1 v_2) = a_{12} b_{12} - a_{02} b_{02} + a_{01} b_{01}$.
   On the right, with $(da)_{012} = a_{12} - a_{02} + a_{01}$,

   .. math::

      \underbrace{(da)_{012}(b_{01} + b_{12})}_{da \,\cup_1\, b}
      \;\underbrace{+\; a_{02}\,(db)_{012}}_{-\,a \,\cup_1\, db}
      \;\underbrace{-\; a_{01} b_{12}}_{-\,a \cup b}
      \;\underbrace{-\; b_{01} a_{12}}_{-\,b \cup a}
      = a_{12} b_{12} - a_{02} b_{02} + a_{01} b_{01}.

   Every cross term cancels.
   For closed $a$ and $b$ this says $a \cup b + b \cup a = -d(a \cup_1 b)$: the cup product of 1-cocycles is graded commutative up to a coboundary, and $\cup_1$ is the explicit homotopy.
   The same identity is checked on every fine simplex, for random integer cochains, by ``test_steenrod_identity_integers`` in :source:`test/test_barycubic_cup.py`.

Every legal product and every Steenrod identity is checked exactly on integer cochains, and mod 2, by ``test_steenrod_identity_integers`` and ``test_steenrod_identity_mod2``; Steenrod squares of mod-2 cocycles are cocycles, as checked by ``test_steenrod_squares_of_cocycles_are_cocycles``; all in :source:`test/test_barycubic_cup.py`.

.. important::

   The signs $\varepsilon(J)$ are *forced*: with $\cup_0$ fixed to Alexander–Whitney, they are the unique solution of :eq:`barycubic-steenrod` on a simplex of high enough dimension.
   Inside a $D$-dimensional complex, however, one sign escapes: the only product with $i = D$ is $\cup_D$ of two top forms, whose Steenrod identity has degree $D + 1$ and vanishes identically.
   That sign is pinned only by the pointwise identity $a \cup_D b = a\, b$, which is why ``test_top_steenrod_product_is_pointwise`` checks $a \cup_p b = a\,b$ for every $p$ including $p = D$, and why ``test_cup_table_matches_closed_form`` compares every sign in the contraction tables with the closed form; both are in :source:`test/test_barycubic_cup.py`.

The closed form was derived for this code rather than quoted from the literature.
The standalone script :source:`docs/barycubic-lattice/verify_cup_i_signs.py` is its certificate: using only the Python standard library, it checks the Steenrod identity in exact integer arithmetic for every $(p, q, i)$ that fits on its standard simplex (ten vertices by default), confirms that three deliberately wrong sign rules fail, and checks the order-reversal law and the pointwise top product.

Order Reversal
--------------

Reversing the vertex order of every simplex turns front faces into back faces.
For even $i$ the products transform as

.. math::

   \mathrm{rev}(a \cup_i b) = (-1)^{pq + i + \binom{i+1}{2}}\; \mathrm{rev}(b) \cup_i \mathrm{rev}(a),

where $\mathrm{rev}$ is the pullback along the reversal.
For odd $i$ this holds only on the diagonal $p = q = i$, where $a \cup_i b = a\, b$.
The law matters here because the :func:`half-diagonal translation <supervillain.lattice.barycubic.half_shift>` reverses rank order; it is checked by ``test_half_shift_reversal_law`` in :source:`test/test_barycubic_cup.py`.

.. autofunction :: supervillain.lattice.barycubic.cup

.. autofunction :: supervillain.lattice.barycubic.wedge

.. automethod :: supervillain.lattice.barycubic.BarycubicLattice.cup_table

.. _barycubic-symmetries:

Symmetries
==========

The hyperoctahedral group (permutations and reflections of the axes) and the integer translations act on doubled coordinates and map cells to cells of the same dimension.
They preserve rank, and hence the rank order of every simplex, so their pullback is a pure relabelling with no sign.
:func:`~supervillain.lattice.barycubic.transform` implements it.
It commutes with $d$ and every $\cup_i$, preserves the weights, and multiplies the orientation by $\det g$, as checked by ``test_transform_is_pullback``, ``test_transform_commutes_with_d_and_cup``, and ``test_transform_preserves_weights_and_scales_orientation`` in :source:`test/test_barycubic_cup.py`.

The subdivision has one more symmetry, invisible on the hypercubic lattice: the **half-diagonal translation**

.. math::

   \tau: \xi \mapsto \xi + (1, \ldots, 1),

a shift by half a lattice spacing in every direction.
It flips the parity of every coordinate, so it maps a cell of rank $r$ to one of rank $D - r$: sites to hypercube centers, link midpoints to cube centers, and so on.
In the figure it sends the sites to the plaquette centers and exchanges the two kinds of link midpoints.
Because it reverses rank order, the pullback of a p-form carries the sign $(-1)^{\binom{p+1}{2}}$ of reversing $p+1$ vertices.
It commutes with $d$, maps the weight $M(R)$ to $M(D - R)$, which is why those are equal, and obeys the order-reversal law above.
Applying it twice is the unit translation along the diagonal.
These are checked by ``test_half_shift_is_signed_pullback``, ``test_half_shift_twice_is_unit_translation``, ``test_half_shift_commutes_with_d_and_mirrors_weights``, and ``test_half_shift_reversal_law`` in :source:`test/test_barycubic_cup.py`.

.. autofunction :: supervillain.lattice.barycubic.transform

.. autofunction :: supervillain.lattice.barycubic.half_shift

Coarsening to the Hypercubic Lattice
====================================

Every hypercubic p-cell $F$ is tiled by the $2^p p!$ fine p-simplices inside it: those with ranks $0, 1, \ldots, p$ whose top vertex is the barycenter of $F$.
Summing a fine cochain over each cell, with the sign $o(\sigma, F) = \pm 1$ that compares the rank orientation of $\sigma$ with the orientation of $F$ in increasing direction order,

.. math::

   (Sd^\top \omega)(F) = \sum_{\sigma \subseteq F,\ \dim \sigma = p} o(\sigma, F)\, \omega_\sigma,

gives a hypercubic :class:`~supervillain.lattice.Form`.
This is the pullback along the subdivision chain map, so it is a cochain map:

.. math::

   \texttt{coarsen}(d\omega) = d\, \texttt{coarsen}(\omega),

with the hypercubic :func:`~supervillain.lattice.d` on the right, exactly, and it preserves integrals of top forms.
It is checked by ``test_coarsen_is_cochain_map`` and ``test_coarsen_preserves_top_integral`` in :source:`test/test_barycubic_operators.py`.
A map in the other direction, from hypercubic to fine cochains, is not provided.

.. autofunction :: supervillain.lattice.barycubic.coarsen

Rank Coloring
=============

Every fine link joins two vertices of different ranks, so the rank is a proper coloring of the fine vertices with $D + 1$ colors.
It works for every $N$, even or odd, and it is optimal, since the $D+1$ vertices of a top simplex are all joined to each other.

The index format differs from :attr:`Lattice.checkerboarding <supervillain.lattice.Lattice.checkerboarding>` on purpose.
The hypercubic colors select *sites*, and are applied to every component at once with ``form[(slice(None), *color)]``.
Here the rank depends on the label and every site carries vertices of every rank, so a color must select along all $D + 1$ axes of a 0-form, and is applied as ``form[color]``.
This is checked by ``test_rank_checkerboarding`` in :source:`test/test_barycubic_complex.py`.

.. autoproperty :: supervillain.lattice.barycubic.BarycubicLattice.checkerboarding

Dispatch
========

The package-level operators :func:`supervillain.lattice.d`, :func:`~supervillain.lattice.delta`, :func:`~supervillain.lattice.laplacian`, :func:`~supervillain.lattice.star`, :func:`~supervillain.lattice.wedge`, and :func:`~supervillain.lattice.cup` accept either kind of form.
They look at the type of their argument: a :class:`~supervillain.lattice.barycubic.BarycubicForm` goes to the functions on this page, and a hypercubic :class:`~supervillain.lattice.Form` goes to the functions documented in :doc:`form`, unchanged.
The products :func:`~supervillain.lattice.wedge` and :func:`~supervillain.lattice.cup` look at both arguments, and mixing a barycubic form with a hypercubic one (or with any array that is not a barycubic form) raises ``TypeError``, as checked by ``test_mixing_hypercubic_and_barycubic_raises`` and ``test_barycubic_products_reject_foreign_second_argument``.
So code written against the hypercubic forms runs on barycubic forms wherever the operations make sense.
The functions on this page remain importable directly from ``supervillain.lattice.barycubic``.

The hypercubic lattice has only the ordinary product, so :func:`supervillain.lattice.cup` on hypercubic forms is the :func:`~supervillain.lattice.wedge` for $i = 0$ and raises ``NotImplementedError`` for $i > 0$.
The dispatch is checked by ``test_dispatch_routes_barycubic_forms``, ``test_dispatch_routes_barycubic_products``, ``test_dispatch_leaves_hypercubic_forms_unchanged``, and ``test_dispatch_hypercubic_higher_cup_raises`` in :source:`test/test_barycubic_operators.py`.

.. autofunction :: supervillain.lattice.cup

HDF5 and Batches
================

A :class:`~supervillain.lattice.barycubic.BarycubicLattice` is written to HDF5 as its ``D`` and ``N``; everything else is rebuilt on load.
A standalone barycubic form is written by a dedicated :doc:`strategy <h5>`, which stores the data with its degree, its primal or dual kind, and its lattice, and restores the right class on read.
Without it, the form would be caught by the NumPy strategy and read back as a plain array.
Only whole forms are written this way: a slice or reshape of a form is not a cochain, and is written and read back as the plain array it is, as views of hypercubic forms are.
Inconsistent metadata is rejected on read.
Forms also pickle with their degree and lattice (the lattice pickles as its ``D`` and ``N``), so they can be sent between processes.
Barycubic forms can also be the elements of a :class:`~supervillain.batch.Batch` column, under their own tags ``'BarycubicForm'`` and ``'BarycubicDualForm'``, so a hypercubic column and a barycubic column can never be confused when read back.

These round trips, for every degree and for integer, float, and complex data, are checked by ``test_lattice_h5_roundtrip``, ``test_form_h5_roundtrip_standalone``, ``test_dual_form_h5_roundtrip_standalone``, ``test_form_batch_h5_roundtrip``, ``test_dual_form_batch_h5_roundtrip``, and ``test_hypercubic_and_barycubic_columns_never_confused``; the handling of views by ``test_h5_views_are_written_as_plain_arrays`` and ``test_h5_orphaned_form_leaves_no_broken_dataset``; and pickling by ``test_forms_pickle``; all in :source:`test/test_barycubic_operators.py`.

.. automethod :: supervillain.lattice.barycubic.BarycubicLattice.to_h5

.. automethod :: supervillain.lattice.barycubic.BarycubicLattice.from_h5

.. autoclass :: supervillain.h5.strategy.barycubic.BarycubicFormStrategy

Comparing Implementations
=========================

Internal layouts (the label order, the storage shape) are conventions, and another implementation of the same complex may choose differently.
To compare two implementations exactly, :meth:`~supervillain.lattice.barycubic.BarycubicLattice.export_complex` names every fine simplex by its **canonical label**, the rank-ordered list of its vertices in doubled coordinates reduced to $[0, 2N)$.
Since the subdivision is a simplicial complex, that list determines the simplex.
Keyed by canonical labels, the coboundary matrix, the weights, and the orientations of two implementations can be compared entry by entry.

Products are compared on shared cochains.
:meth:`~supervillain.lattice.barycubic.BarycubicLattice.export_cup` builds two integer cochains from a fully specified hash of the canonical labels, so another implementation can rebuild exactly the same factors without sharing any code, and exports $a \cup_i b$ keyed by canonical label.

Two standalone scripts do the comparison.
:source:`docs/barycubic-lattice/write_export.py` writes :meth:`~supervillain.lattice.barycubic.BarycubicLattice.export_complex` for every degree and :meth:`~supervillain.lattice.barycubic.BarycubicLattice.export_cup` for every legal $(p, q, i)$ at one $(D, N)$ into a single ``.npz`` file, in a documented key layout that another implementation can write too.
:source:`docs/barycubic-lattice/compare_exports.py`, which needs only numpy, matches the rows of two such exports by canonical label and reports, for every quantity, agreement or the first mismatch:

.. code-block:: bash

   python docs/barycubic-lattice/write_export.py 3 2 ours.npz
   python docs/barycubic-lattice/compare_exports.py ours.npz theirs.npz

The exports are checked against the reference construction by ``test_export_complex_matches_reference`` and ``test_export_cup_matches_reference`` in :source:`test/test_barycubic_reference.py`, and the cochains' reproducibility by ``test_export_cup_is_deterministic_and_seeded`` in :source:`test/test_barycubic_operators.py`.

.. automethod :: supervillain.lattice.barycubic.BarycubicLattice.canonical_labels

.. automethod :: supervillain.lattice.barycubic.BarycubicLattice.export_complex

.. automethod :: supervillain.lattice.barycubic.BarycubicLattice.export_cup

Performance
===========

Everything combinatorial depends only on $D$, not on $N$: the labels, the incidence tables of $d$ and $d^\top$, the contraction tables of every $\cup_i$, the weights, the orientations, the coarsening matrix, and the label maps of the symmetries.
Each is built the first time it is needed, and then cached and shared by every lattice of that dimension.
The arrays that the accessors return (:meth:`~supervillain.lattice.barycubic.BarycubicLattice.operator_table`, :meth:`~supervillain.lattice.barycubic.BarycubicLattice.cup_table`, :meth:`~supervillain.lattice.barycubic.BarycubicLattice.hodge_weights`, and so on) are read-only views of these caches, so they cannot be corrupted by accident, as checked by ``test_cached_tables_are_read_only`` in :source:`test/test_barycubic_operators.py`.

Applying an operator never loops over sites in Python.
The tables are handed to small compiled (numba) kernels, together with a per-lattice table of neighbor sites for each anchor offset, so no shifted copy of the input is ever made:

* $d$ and $d^\top$ run once over the table rows, accumulating ``out[o, x] += sign * f[i, x + offset]`` into a preallocated output;
* $\delta = M_{p-1}^{-1} d^\top M_p$ is the same pass with the two diagonal scalings folded into one coefficient per row, and the Laplacian accumulates $\delta\, d f$ and $d\, \delta f$ into a single output;
* a $\cup_i$ product runs once over its contraction rows, ``out[o, x] += sign * a[i, x + a_offset] * b[j, x + b_offset]``;
* :func:`~supervillain.lattice.barycubic.transform` and :func:`~supervillain.lattice.barycubic.half_shift` are a single gather through a cached label map.

What an operator allocates, beyond the numba compilation on first use (under a second, then cached on disk):

* its output, which the kernel fills in place;
* the Laplacian also holds $df$ and $\delta f$, and promotes an integer or single-precision input to double precision first (so integer inputs cannot overflow);
* a copy of an input that is not C-contiguous (a transposed or Fortran-order array), since the kernels read the flat ``(labels, sites)`` layout, and in :func:`~supervillain.lattice.barycubic.cup` a converted copy of a factor whose dtype differs from the result's; C-contiguous inputs of the result dtype are used in place;
* small per-call arrays: the table signs converted to the output dtype (one entry per table row), and for :func:`~supervillain.lattice.barycubic.transform` and :func:`~supervillain.lattice.barycubic.half_shift` one flattened-site index array per distinct anchor offset (four for the 2-form transform benchmarked below), rebuilt on each call because they depend on the translation;
* the first time a lattice meets a new set of offsets, its neighbor-site table, which is then cached on the lattice (one integer per site and offset).

Caching the transform index arrays as well was considered and not done: their key includes the translation, so a cache would grow with every distinct shift, while rebuilding them took about 0.5 ms of the 3.2 ms of a $D = 4$, $N = 8$ :func:`~supervillain.lattice.barycubic.transform`.
The kernels are serial: Monte Carlo chains usually already run one per core.

On an Apple M1 Pro laptop (September 2026, while other jobs kept all cores busy), in $D = 4$ with the tables built and the kernels compiled, :source:`benchmark/barycubic.py` measured these costs per fine output simplex:

.. list-table::
   :header-rows: 1

   * - operation
     - $N = 4$
     - $N = 8$
   * - :func:`~supervillain.lattice.barycubic.d`, 1-forms to 2-forms
     - 2.0 ns
     - 3.1 ns
   * - :func:`~supervillain.lattice.barycubic.delta`, 2-forms to 1-forms
     - 5.9 ns
     - 9.3 ns
   * - :func:`~supervillain.lattice.barycubic.laplacian` of a 1-form
     - 16 ns
     - 23 ns
   * - :func:`~supervillain.lattice.barycubic.cup` of two 1-forms, $\cup_0$
     - 1.2 ns
     - 1.2 ns
   * - :func:`~supervillain.lattice.barycubic.cup` of two 2-forms, $\cup_1$
     - 1.8 ns
     - 3.4 ns
   * - :func:`~supervillain.lattice.barycubic.transform` of a 2-form
     - 2.3 ns
     - 1.0 ns
   * - :func:`~supervillain.lattice.barycubic.half_shift` of a 2-form
     - 3.0 ns
     - 1.2 ns
   * - hypercubic :func:`~supervillain.lattice.d`, 1-forms to 2-forms (per hypercubic plaquette)
     - 3.6 ns
     - 2.5 ns

For comparison, at $N = 8$ a 2-form has about 3.3 million fine triangles against 25 thousand hypercubic plaquettes, so a barycubic $d$ on 1-forms took about 10 ms against 0.06 ms for the hypercubic one.
Building every table for $D = 4$ took about 0.4 s in total, 0.3 s of it for the 35 cup-product tables; in $D = 5$ the labels, incidence tables, and weights took about 2 s.
These are measurements on one machine, not guarantees.
