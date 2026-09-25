
.. _reference_implementations:

*************************
Reference Implementations
*************************

Sometimes the obvious implementation can be accelerated dramatically.

It is useful to have a *reference implementation* where things are done the totally obvious way so that smarter or faster approaches have a fixed target for correctness.  

.. note ::
   Everything here can be considered an implementation detail and a user should not need the reference implementations at all.


==========
Generators
==========

Villain NeighborhoodUpdate
==========================
.. autoclass :: supervillain.generator.reference_implementation.villain.NeighborhoodUpdateSlow
   :members:

Villain Classic Worm
====================
.. autoclass :: supervillain.generator.reference_implementation.villain.ClassicWorm
   :members:

Wordline Classic Worm
=====================
.. autoclass :: supervillain.generator.reference_implementation.worldline.ClassicWorm
   :members:



===========
Observables
===========


Spin Correlations
=================
.. autoclass :: supervillain.observable.reference_implementation.spin.Spin_SpinSloppy
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.reference_implementation.spin.Spin_SpinSlow
   :members:
   :show-inheritance:


==================
Differential Forms
==================

.. _interlaced-forms:

Interlaced Forms
================

:ref:`The interlaced picture <interlaced>` can be taken literally: a $p$-form on an $N^D$ lattice is stored in the elements of a $(2N)^D$ array with exactly $p$ odd coordinates.
However, this can be very wasteful since only $\binom{D}{p}/2^D$ of the array elements are nonzero.
Nevertheless, it is useful to have an implementation that realizes this picture directly so that the production :class:`~supervillain.lattice.Form`, which is space-efficient, can be tested against it.

The operators of the interlaced format are independent implementations of the exterior calculus with the same :ref:`sign conventions <sign-conventions>`, giving a fixed target for correctness.

Below, let the unit step $\varepsilon_k$ advance $\xi_k$ by one (half a physical lattice step in direction $k$).

.. autoclass :: supervillain.lattice.interlaced.Lattice
   :members:

Translation
-----------

Forms can be translated around the (periodic) interlaced lattice.
Only a completely even shift maps a $p$-form to another $p$-form. 

.. autofunction :: supervillain.lattice.interlaced.pull

.. autofunction :: supervillain.lattice.interlaced.push

Exterior Derivative
-------------------

At each $(p+1)$-form output site $\xi$ with odd directions $O = (o_0, \ldots, o_p)$:

.. math::

   (df)[\xi] = \sum_{j=0}^{p} (-1)^j \bigl(f[\xi + \varepsilon_{o_j}] - f[\xi - \varepsilon_{o_j}]\bigr)

Each term is a centered (symmetric) difference in interlaced direction $o_j$, spanning one physical lattice step.

.. autofunction :: supervillain.lattice.interlaced.d

This $d$ is nilpotent $d^2 = 0$ holds as checked by ``test_d_nilpotent`` in :source:`test/test_lattice_interlaced.py`.

Codifferential
--------------

The codifferential is the formal adjoint of the exterior derivative,

.. math::
   :label: interlaced-d-delta-adjoint

   \langle da, b \rangle = \langle a, \delta b \rangle

tested by ``test_adjointness`` in :source:`test/test_lattice_interlaced.py`.

At each $(p-1)$-form output site $\xi$ with odd directions $M$ and even directions $E = (e_0, \ldots, e_{D-p})$:

.. math::

   (\delta f)_M[\xi] = -\sum_{i=0}^{D-p} (-1)^{e_i - i} \bigl(f_{M \cup \{e_i\}}[\xi + \varepsilon_{e_i}] - f_{M \cup \{e_i\}}[\xi - \varepsilon_{e_i}]\bigr)

The sign $(-1)^{e_i - i}$ equals $(-1)^j$ where $j = \#\{m \in M : m < e_i\}$ is the insertion position of $e_i$ into $M$.

.. autofunction :: supervillain.lattice.interlaced.delta

The nilpotency $\delta^2 = 0$ holds as checked by ``test_delta_nilpotent`` in :source:`test/test_lattice_interlaced.py`.

The continuum identity $\delta = (-1)^{D(k+1)+1}\,\star\,d\,\star$ holds on the interlaced lattice only up to a translation; see :ref:`the note below <interlaced-star-d-star-shift>`.

Hodge Star
----------

For each output $(D-p)$-form site $\xi$ with odd directions $J$, let $I = \{0,\ldots,D-1\} \setminus J$ be its complement:

.. math::

   (\star f)_J[\xi] = \sigma(I \frown J)\; f_I[\xi - \varepsilon_{\mathrm{all}}]

where $\varepsilon_{\mathrm{all}} = (1, \ldots, 1)$.
The all-direction shift $-\varepsilon_{\mathrm{all}}$ flips every coordinate's parity, mapping the $(D-p)$-form site $\xi$ to the $p$-form site $\xi - \varepsilon_{\mathrm{all}}$.

.. autofunction :: supervillain.lattice.interlaced.star

The Hodge inner-product identity $\langle a, b \rangle = \sum_{\xi} (a \wedge \star b)_{(0,\ldots,D-1)}[\xi]$ holds exactly; it is cross-validated by ``test_compact_star_matches_interlaced`` in :source:`test/test_lattice.py`.

Wedge Product
-------------

At each $(n+m)$-form output site $\xi$ with odd directions $O$, summing over all shuffles $O = A \sqcup B$ ($A$ the $n$ a-directions, $B$ the $m$ b-directions):

.. math::

   (a \wedge b)[\xi] = \sum_{O = A \sqcup B} \sigma(B \frown A)\; a[\xi - \hat{\varepsilon}_A]\; b[\xi + \hat{\varepsilon}_B]

where $\hat{\varepsilon}_A = \sum_{k \in A} \varepsilon_k$ and $\hat{\varepsilon}_B = \sum_{k \in B} \varepsilon_k$.
The sign $\sigma(B \frown A)$ sorts $(B, A)$ back into $O$; it equals $(-1)^{nm}\,\sigma(A \frown B)$.
In contrast to the compact formula (which evaluates $b$ at $x + \hat{e}_A$), the output site $\xi$ lies between $a$ at $\xi - \hat{\varepsilon}_A$ and $b$ at $\xi + \hat{\varepsilon}_B$.

.. collapse:: Why σ(B⌢A) and not σ(A⌢B)? That's what appears in the dense format...
   :class: note

   The compact wedge evaluates $a$ at the output site and shifts $b$ to the far side of the $a$-cell:

   .. math::

      (a \wedge b)_O[x] = \sum_{O = A \sqcup B} \sigma(A \frown B)\; a_A[x]\; b_B\!\left[x + \hat{e}_A\right]

   The interlaced wedge is instead centered: $a$ is pulled back to $\xi - \hat{\varepsilon}_A$ and $b$ pushed forward to $\xi + \hat{\varepsilon}_B$.
   Shifting $a$'s evaluation point from $\xi$ to $\xi - \hat{\varepsilon}_A$ introduces exactly the factor $(-1)^{nm}$ that converts $\sigma(A \frown B)$ into $\sigma(B \frown A)$, so both formulas give the same physical result.

   A direct check for $D=2$, $n=m=1$, $O=(0,1)$:

   .. list-table::
      :header-rows: 1

      * - shuffle
        - compact
        - interlaced (translated to physical)
      * - $A=(0),\,B=(1)$
        - $+a_0[x]\,b_1[x+\hat{e}_0]$
        - $-a_1[x]\,b_0[x+\hat{e}_1]$
      * - $A=(1),\,B=(0)$
        - $-a_1[x]\,b_0[x+\hat{e}_1]$
        - $+a_0[x]\,b_1[x+\hat{e}_0]$

   The rows swap between the two, but the total $a_0[x]\,b_1[x+\hat{e}_0] - a_1[x]\,b_0[x+\hat{e}_1]$ is the same.

.. autofunction :: supervillain.lattice.interlaced.wedge

Bilinearity, associativity, and the Leibniz rule $d(a \wedge b) = da \wedge b + (-1)^n\,a \wedge db$ all hold exactly; they are cross-validated by ``test_compact_wedge_matches_interlaced`` in :source:`test/test_lattice.py`.

Unlike the continuum, however, the lattice wedge product is not anti-commutative; see :ref:`the note below <interlaced-wedge-commutativity-fail>`.

Differences from the Continuum
-------------------------------

In the continuum, on a Riemannian manifold with Euclidean signature, the Hodge star squares to

.. math::

   \star\star a = (-1)^{p(D-p)}\, a \qquad \text{on a } p\text{-form.}

.. _interlaced-star-star-shift:

.. danger::

   On the lattice the identity picks up a spatial shift.  In interlaced coordinates:

   .. math::

      (\star\star f)[\xi] = (-1)^{p(D-p)}\, f[\xi - 2\varepsilon_{\mathrm{all}}]

   a backward shift of $2\varepsilon_{\mathrm{all}} = (2,\ldots,2)$ — one physical lattice step in each direction.
   The shift drops out of any periodic sum.
   This is tested by ``test_star_star`` in :source:`test/test_lattice_interlaced.py`.

In the continuum we expect anti/commutativity

.. math::

   (a \wedge b) = (-1)^{n m} (b \wedge a)

to hold.

.. _interlaced-wedge-commutativity-fail:

.. danger::

   On the lattice this property fails!

In the continuum, on a Riemannian manifold with Euclidean signature, the Hodge star satisfies

.. math::

   \delta = (-1)^{D(k+1)+1}\, \star\, d\, \star \qquad \text{on } k\text{-forms}.

.. _interlaced-star-d-star-shift:

.. danger ::

   On the lattice this identity picks up a spatial shift.  In interlaced coordinates the shift is $2\varepsilon_{\mathrm{all}} = (2, \ldots, 2)$ — one physical step in each direction:

   .. math::

      \delta = (-1)^{D(k+1)+1}\, T_{2\varepsilon_{\mathrm{all}}}\, \star\, d\, \star

   where $T_{\Delta\xi}$ is the translation operator (:func:`~supervillain.lattice.interlaced.pull` by $\Delta\xi$).
   The shift drops out of any periodic sum, which is why :eq:`interlaced-d-delta-adjoint` holds exactly despite it.
   This is tested by ``test_star_d_star_equals_shifted_delta`` in :source:`test/test_lattice_interlaced.py`.

.. _barycentric-forms:

Barycentric Forms
=================

The barycentric subdivision $S = \mathrm{sd}\,K$ of the hypercubic complex $K$ replaces every cell of $K$ --- vertex, edge, plaquette, cube, up to the hypercube itself --- by the simplices spanned by the barycenters of its faces.
Cochains on $S$ are not $p$-forms on the original lattice: they assign a number to each fine simplex, and the simplices of one degree are not congruent to one another.
This module implements the exterior calculus there directly, in the totally obvious way, as a fixed target for correctness.

The fine vertices are the barycenters $b(F)$ of every face $F$ of every hypercube.
A rank-$r$ barycenter has exactly $r$ half-integer coordinates, so in the :ref:`interlaced <interlaced>` doubled coordinates $\xi = 2x$ the fine vertices are exactly the sites of a $(2N)^D$ array, and

.. math::

   \mathrm{rank}\; b(F) = \dim F = \#\{\mu : \xi_{\mu} \text{ odd}\}

which is precisely the interlaced $p$-form-site condition.
A rank-$p$ barycenter and an interlaced $p$-form site are the same array element; the two pictures share a substrate and differ in what lives on it.

A fine $p$-simplex is a strictly nested chain of faces $F_0 \subset \cdots \subset F_p$, and in doubled coordinates such a chain is a *staircase*.
Write $\varepsilon_{\mu}$ for the unit step that advances $\xi_{\mu}$ by one, half a physical lattice step in direction $\mu$.
From the simplex's unique lowest-rank vertex --- its **anchor** --- each step releases a disjoint, nonempty, signed set $\Delta_i$ of directions,

.. math::

   v_i = v_{i-1} + \sum_{\mu \in \Delta_i} \sigma_{\mu}\, \varepsilon_{\mu}

flipping those coordinates from even to odd, so that each vertex has strictly higher rank than the last.
A step releasing several directions at once is a chain that skips ranks.
Every staircase is a nested chain and every nested chain is a staircase, so a fine $p$-simplex is exactly an anchor together with an ordered tuple $(\Delta_1, \ldots, \Delta_p)$, which we call its **type**.

A degree-$p$ cochain is therefore stored as an array of shape $(T_p,) + (2N,)^D$: the leading axis enumerates the types and the rest give the anchor.
A type is realized at an anchor only when every direction it releases is even there, so this array is sparse in the same spirit as the interlaced picture, and :meth:`~supervillain.lattice.barycentric.doubled.Lattice.support` is that mask.
A type releasing $k$ directions lives on $2^{-k}$ of the anchors, so the waste grows with degree: in $D = 4$ the occupancy runs $100\%, 19\%, 11\%, 8\%, 6\%$ from $p = 0$ to $p = 4$.

That is the price of the obvious layout, and it is deliberate.
A space-efficient implementation would group anchors by parity pattern --- each of the $2^D$ patterns admits its own set of types, and those anchors form an $N^D$ sublattice --- storing exactly one entry per simplex.
That layout is a factor of $16$ smaller at the top degree and about $11$ smaller overall, but every operator must then route between $2^D$ blocks with different type sets in each, which is exactly the sort of intricacy worth checking against an implementation that does the obvious thing instead.

.. autoclass :: supervillain.lattice.barycentric.doubled.Lattice
   :members:

.. autoclass :: supervillain.lattice.barycentric.doubled.Form
   :no-index:

Types and Classes
-----------------

The number of degree-$p$ types is the number of ordered tuples of $p$ disjoint nonempty signed direction sets,

.. math::
   :label: barycentric-type-count

   T_p(D) = \sum_{j=0}^{p} (-1)^j \binom{p}{j} \bigl(2(p-j)+1\bigr)^{D}

giving $1, 80, 464, 768, 384$ for $D = 4$, as checked by ``test_type_counts_match_closed_form`` in :source:`test/test_lattice_barycentric.py`.
The top count has a second reading: a maximal chain releases one direction at a time, so it is a choice of corner and an order of release, $2^D D! = 384$ fine four-simplices per hypercube.

A fine link joins a rank-$r$ to a rank-$s$ barycenter, and the pair $(r,s)$ with $r < s$ labels its **class**.
There are $\binom{D+1}{2} = 10$ classes in $D = 4$, carrying $240$ links per original site:

.. list-table::
   :header-rows: 1

   * - class
     - $(0,1)$
     - $(0,2)$
     - $(0,3)$
     - $(0,4)$
     - $(1,2)$
     - $(1,3)$
     - $(1,4)$
     - $(2,3)$
     - $(2,4)$
     - $(3,4)$
   * - links per site
     - 8
     - 24
     - 32
     - 16
     - 24
     - 48
     - 32
     - 24
     - 24
     - 8

checked by ``test_ten_link_classes`` and ``test_two_hundred_forty_links_per_original_site``.
:attr:`~supervillain.lattice.barycentric.doubled.Lattice.link_class` labels each link slot with its class, and classes matter because nothing forces two links of the same class, or even of the same length, to be treated alike.

Measure
-------

On the hypercubic lattice every $p$-cell of a degree is congruent to every other.
Here they are not, and :meth:`~supervillain.lattice.barycentric.doubled.Lattice.measure` gives the $p$-volume of each type as

.. math::

   |\sigma| = \frac{\sqrt{\det G}}{p!}, \qquad G_{ij} = (v_i - v_0)\cdot(v_j - v_0)

in original lattice units.
A class-$(r,s)$ link differs from its partner in $s-r$ coordinates by half a step, so it has length $\sqrt{s-r}/2$; within one degree the measure varies by a factor of two.
Every maximal simplex has volume $1/(D!\,2^D)$ --- $1/384$ in $D = 4$ --- and the eight fine triangles cutting a plaquette each have area $1/8$.
These are checked by ``test_link_length_is_sqrt_rank_difference_over_two``, ``test_measure_of_top_simplex_is_general_formula`` and ``test_primal_triangle_has_area_one_eighth``; that the top measures sum to the volume of the torus is ``test_total_measure_of_top_simplices_is_the_volume``.

Note that equal length does not make two simplices interchangeable: the classes $(0,1)$, $(1,2)$, $(2,3)$ and $(3,4)$ all have length $1/2$ and remain four distinct classes.

Orientation
-----------

Each fine $p$-simplex carries a $p$-vector: its extent on each coordinate $p$-plane, the determinant of its edge vectors restricted to that plane.
:meth:`~supervillain.lattice.barycentric.doubled.Lattice.pvector` returns those components, indexed by :meth:`~supervillain.lattice.barycentric.doubled.Lattice.planes`.

A simplex whose every step releases a single direction spans a coordinate plane and has exactly one nonzero component, $\pm 1$; there are $\binom{D}{p} p!\, 2^p$ such types, checked by ``test_coordinate_aligned_types_have_exactly_one_unit_component``.
A simplex whose steps release several directions at once spans a $p$-plane that is not coordinate-aligned --- a class-$(r,r{+}2)$ link, say, whose single edge vector moves in two directions.
Such a simplex is not unoriented; it has extent in more than one plane, and correspondingly several nonzero components (``test_non_aligned_types_have_several_components_not_none``).
A zero component says the simplex has no extent in that plane.

The components multiply.
Split a top simplex's steps at any position into a front and a back face; whenever each spans a single coordinate plane and the two planes are complementary,

.. math::
   :label: barycentric-pvector-product

   \epsilon_{\sigma} = v_{\mathrm{front}}\; v_{\mathrm{back}}\; \sigma(I_{\mathrm{front}} \frown I_{\mathrm{back}})

which ``test_pvector_components_multiply`` checks at every split point in $D = 2, 3, 4$.

Only the top degree carries a *sign* rather than a vector of components.
A $D$-simplex spans the one coordinate $D$-plane there is, so its $p$-vector has a single entry, and that entry compares the simplex's rank ordering with the ambient orientation; below the top there is no such comparison without a choice of complement.
:attr:`~supervillain.lattice.barycentric.doubled.Lattice.orientation_sign` is that entry, and ``test_top_pvector_is_the_orientation_sign`` ties the two together.

Translation
-----------

Forms can be translated by whole unit cells.
Because one lattice step is two doubled steps, :func:`~supervillain.lattice.barycentric.doubled.push` rolls the array by $2a$ for a shift $a$ given in original lattice units, and :func:`~supervillain.lattice.barycentric.doubled.translate` is the same function under the name used elsewhere in the library.

.. autofunction :: supervillain.lattice.barycentric.doubled.push

.. autofunction :: supervillain.lattice.barycentric.doubled.pull

Only whole-cell shifts and the :ref:`half-diagonal <barycentric-half-diagonal>` below are symmetries; a general half-step is not, as ``test_push_rolls_whole_unit_cells`` and the enumeration in ``test_only_the_all_ones_parity_shift_preserves_the_complex`` record.

Exterior Derivative
-------------------

On a fine $(p+1)$-simplex with vertices $u_0 \cdots u_{p+1}$ in increasing rank, the coboundary is the alternating sum over faces:

.. math::

   (da)[u_0 \cdots u_{p+1}] = \sum_{i=0}^{p+1} (-1)^i\, a[u_0 \cdots \widehat{u_i} \cdots u_{p+1}]

where the hat denotes omission.
Read on staircases the three cases are concrete: omitting $u_0$ deletes the first step and re-anchors at $v_1$, omitting an interior $u_i$ merges the steps $\Delta_i$ and $\Delta_{i+1}$ into one --- legal because their direction sets are disjoint --- and omitting the last vertex deletes the last step.

.. autofunction :: supervillain.lattice.barycentric.doubled.d

Every codimension-two face appears twice with opposite signs, so $d^2 = 0$ exactly; this is checked by ``test_d_squared_is_zero``, with a non-vanishing assertion first so it cannot pass on a degenerate $da$.

Codifferential
--------------

The codifferential is the formal adjoint of the exterior derivative with respect to the pairing of a cochain with a chain,

.. math::
   :label: barycentric-d-delta-adjoint

   \langle da, b \rangle = \langle a, \delta b \rangle, \qquad
   \langle a, b \rangle = \sum_{\sigma} a_{\sigma}\, b_{\sigma}

the plain sum over fine simplices, tested by ``test_adjointness``.
It is implemented as the transpose of the very table $d$ walks: where $d$ gathers from a face, $\delta$ scatters back to it.

.. autofunction :: supervillain.lattice.barycentric.doubled.delta

Adjointness is the sharpest available test that the coboundary is right, and the pairing it is stated with respect to is a choice worth being explicit about; see :ref:`the note below <barycentric-sums>`.

Cup Product
-----------

The Alexander--Whitney cup product evaluates its first factor on the front face of a simplex and its second on the back:

.. math::

   (a \smile c)[v_0 \cdots v_{p+q}] = a[v_0 \cdots v_p]\; c[v_p \cdots v_{p+q}]

On staircases this splits the output type's step sequence at position $p$: $a$ is read at the anchor from the first $p$ steps, and $c$ at the vertex $v_p$ from the last $q$.

.. autofunction :: supervillain.lattice.barycentric.doubled.cup

There is exactly one term per output simplex --- no sum over shuffles.
This is the sharpest difference from :func:`~supervillain.lattice.interlaced.wedge`, and it means a product formed from $\smile$ contributes a single term to each top simplex, so distinct contributions cannot cancel within one.
:func:`~supervillain.lattice.barycentric.doubled.wedge` is an alias, so that expressions written against the hypercubic lattice read the same here.

Bilinearity, associativity and the Leibniz rule $d(a \smile c) = da \smile c + (-1)^{p}\, a \smile dc$ all hold exactly, checked by ``test_cup_is_bilinear``, ``test_cup_is_associative`` and ``test_leibniz``.

The Cup-Square at a Plaquette Centre
------------------------------------

The cup-square of a $2$-cochain has a structure worth stating, because it is a property of the subdivision rather than of anything built on it.

Fix a plaquette $P$ spanning two directions, with centre $b(P)$ of rank two, and let $P^*$ be the dual cell: the square spanning the complementary two directions whose corners are the centres of the hypercubes containing $P$, and whose centre is also $b(P)$.
Each square is cut by the subdivision into eight fine triangles.
Call the eight inside $P$ the **primal** family and the eight inside $P^*$ the **dual** family; a primal triangle runs corner $\subset$ edge $\subset P$, and a dual one runs $P \subset$ cube $\subset$ hypercube.

Every fine four-simplex has exactly one rank-two vertex.
Those whose rank-two vertex is $b(P)$ are precisely the convex hulls of one primal and one dual triangle, so there are $8 \times 8 = 64$ of them, as ``test_sixty_four_simplices_meet_at_a_plaquette_centre`` verifies by enumerating all $T_4$ types rather than by construction.

By :eq:`barycentric-pvector-product` the orientation sign factorizes across that split, and so does the cup-square: summing the $64$ orientation-signed products at $b(P)$ gives

.. math::
   :label: barycentric-plaquette-factorization

   \sum_{\sigma \ni b(P)} \epsilon_{\sigma}\, (f \smile f)[\sigma]
       = \Phi(P)\, \Phi(P^*)

the product of the sums of $f$ over the two triangle families, each taken with its own orientation, and with no shift vector anywhere in it.
This is ``test_cup_square_factorizes_at_a_plaquette_centre``.
Because the $64$ values are the entries of an outer product of two eight-vectors, they vanish together exactly when one of the two families carries nothing --- ``test_cup_square_vanishes_iff_one_family_does``.

Four of the ten link classes, $(0,3)$, $(0,4)$, $(1,3)$ and $(1,4)$, join a rank below two to a rank above it.
No cup-square of a $2$-cochain of the form $dn$ reads them, so changing $n$ there leaves every such product untouched (``test_rank_skipping_links_enter_no_cup_square``).

Symmetries
----------

A signed coordinate permutation maps faces to faces and preserves their dimension, hence preserves rank and the nesting of chains.
The full hypercubic group $(\mathbb{Z}_2)^D \rtimes S_D$ therefore acts, of order $2^D D! = 384$ in $D = 4$, and ``test_point_group_order_is_measured_not_assumed`` measures that order by counting distinct actions rather than counting enumerated elements.

.. autofunction :: supervillain.lattice.barycentric.doubled.reflect

.. autofunction :: supervillain.lattice.barycentric.doubled.permute

Because rank is preserved, so is the rank ordering of every chain, and **no orientation sign is needed** --- the whole transformation is a permutation of the type axis together with an ordinary array operation on the sites.
This is the one respect in which these are simpler than :func:`~supervillain.lattice.interlaced.reflect` and :func:`~supervillain.lattice.interlaced.permute`, which carry a per-site sign.

.. _barycentric-half-diagonal:

The Half-Diagonal Translation
-----------------------------

Translating by half the diagonal, $\tau : x \mapsto x + \tfrac{1}{2}(1, \ldots, 1)$, is a further symmetry, and it exists in every $D$ for a structural reason: it carries $K$ to the dual cubulation $K^*$, and $\mathrm{sd}\,K = \mathrm{sd}\,K^*$ because dual cells share barycenters and reverse nesting.

.. autofunction :: supervillain.lattice.barycentric.doubled.tau

That duality is why $\tau$ reverses rank, $r \mapsto D - r$, and why it carries link class $(r,s)$ to $(D-s, D-r)$ --- a reflection of the class table across its antidiagonal, checked by ``test_tau_maps_link_class_antidiagonally``.
Reversing rank reverses the vertex order of every simplex, so $\tau$ re-anchors each one at the opposite end and carries a sign $(-1)^{p(p+1)/2}$ on $p$-cochains; ``test_tau_moves_a_marked_simplex_to_the_predicted_slot_with_the_predicted_sign`` pins both the placement and the sign, and $\tau^2$ is the unit diagonal translation.

.. autofunction :: supervillain.lattice.barycentric.doubled.translate_diagonal

The other half-diagonals $\tfrac{1}{2}(\pm 1, \ldots, \pm 1)$ are symmetries but not independent ones: $\tfrac{1}{2}(1,\ldots,1,-1) = \tfrac{1}{2}(1,\ldots,1) + (0,\ldots,0,-1)$, so each is $\tau$ composed with a whole-cell translation.
The translation group is generated by the unit translations together with $\tau$.

No other half-integer diagonal is a symmetry: of the $2^D$ shifts by $\tfrac{1}{2}v$ with $v \in \{0,1\}^D$, only $v = 0$ and $v = (1,\ldots,1)$ carry fine links to fine links and fine triangles to fine triangles, as ``test_only_the_all_ones_parity_shift_preserves_the_complex`` checks by enumerating both sets for $D = 2, 3, 4$.

Differences from the Hypercubic Lattice
---------------------------------------

.. _barycentric-sums:

.. danger ::

   There is no single way to sum a cochain.

   On the hypercubic lattice every $p$-cell of a degree is congruent, so summing a form is unambiguous.
   Here three contractions are distinct:

   .. code-block:: python

       a.sum()                             # the pairing with a chain
       (lat.measure(p) * a).sum()          # a Riemann sum
       (weight[lat.link_class] * a).sum()  # a sum weighted class by class

   The first is the pairing of :eq:`barycentric-d-delta-adjoint`, and adjointness is stated with respect to it.
   **Adjointness does not survive measure weighting**: it fails at every degree once the measure is inserted, as ``test_adjointness_is_with_respect_to_the_plain_sum`` asserts in both directions.
   A cochain is already an integrated quantity, and weighting it by the measure again is right only when it stands for a density.

   The lattice therefore supplies the weights --- :meth:`~supervillain.lattice.barycentric.doubled.Lattice.measure` and :attr:`~supervillain.lattice.barycentric.doubled.Lattice.link_class` --- and no privileged way to sum with them.

.. danger ::

   The cup product is not graded-commutative.

   :func:`~supervillain.lattice.barycentric.doubled.cup` evaluates one factor on the front face and the other on the back, an asymmetry that does not wash out, so $a \smile c$ and $(-1)^{pq} c \smile a$ differ simplex by simplex.
   They agree in the cases the :ref:`interlaced wedge <interlaced-wedge-commutativity-fail>` fails too, and for the same underlying reason: a lattice product must pick evaluation points, and no choice is symmetric.

.. danger ::

   This module has no Hodge star, and a star here needs more than an operator.

   A star would have to be a degree-reversing map, and there is no such map from
   $p$-simplices to $(D-p)$-simplices: per original site there are $16$ fine
   vertices against $384$ four-simplices, and $240$ links against $960$
   tetrahedra, where the hypercubic counts are symmetric because
   $\binom{D}{p} = \binom{D}{D-p}$.

   The resolution is that a star does not map within one complex.  Dual blocks
   are in bijection with primal simplices --- $D(\sigma)$ of dimension $D-p$ for
   each $p$-simplex --- so a dual $(D-p)$-cochain holds one number per
   $p$-simplex and the map is index-preserving, $240 \leftrightarrow 240$.  What
   a star needs is therefore a *dual-indexed companion* to :class:`Form`, which
   this module does not have; it is not that the operator is missing but that
   half its type is.

   The mechanism that supplies :func:`~supervillain.lattice.interlaced.star` is
   present here but acts on the wrong index.  In the interlaced picture the rank
   of a site --- its number of odd coordinates --- *is* the form degree, so the
   parity flip $\xi \mapsto \xi + \varepsilon_{\mathrm{all}}$ reverses degree.
   Here rank and degree are different, and
   :func:`~supervillain.lattice.barycentric.doubled.tau`, which is that same parity flip,
   reverses rank while carrying $p$-simplices to $p$-simplices.

   Underneath is that a simplicial complex is not self-dual: the dual cell of a
   fine $p$-simplex is a $(D-p)$-dimensional block, a union of simplices of the
   *second* barycentric subdivision.  Poincaré duality survives in cohomology,
   realized by a map between two complexes rather than by an operator on one.
   Nothing in this module needs a star in any case:
   :func:`~supervillain.lattice.barycentric.doubled.delta` is the transpose of $d$ with
   respect to :eq:`barycentric-d-delta-adjoint`, not $\pm\,\star d\,\star$.
