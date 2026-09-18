
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

The tests cited below live in :source:`test/test_lattice_barycentric.py`;
running the module as a script (``python -m supervillain.lattice.barycentric
--D 4 --N 3``) executes an abridged set of the same checks.

This module implements differential cochains on the barycentric subdivision
$S = \mathrm{sd}\,K$ of a periodic $D$-dimensional hypercubic lattice $K$ ---
the representation behind the collaborator's no-intersection model
(``barycubic-no-intersection-model.pdf``).  It is a standalone reference
implementation, analogous in spirit to
:mod:`supervillain.lattice.interlaced` for the ordinary hypercubic lattice,
and is documented here for the same reason: it is a fixed target for
correctness, not production code.

.. _barycentric-picture:

The Barycentric Picture
------------------------

Subdividing $K$ barycentrically replaces every cell $F$ of every dimension
--- vertices, edges, plaquettes, cubes, ..., up to the top hypercube --- with
its barycenter $b(F)$, and cuts each hypercube into simplices whose vertices
are barycenters of a chain of faces $F_0 \subset F_1 \subset \cdots \subset
F_p$.  Because $F_0 \subset \cdots \subset F_p$ is exactly a *flag*, the fine
simplices of $S$ are indexed by flags of $K$, and a flag's dimension
sequence $(\dim F_0, \ldots, \dim F_p)$ records how many dimensions each
step in the flag adds.

A barycenter's own dimension is its **rank**: $b(F)$ has rank $r = \dim F$.
Since a face of dimension $r$ spans $r$ of the $D$ directions, its
barycenter has exactly $r$ half-integer coordinates and $D - r$ integer
ones.  Doubling every coordinate, $\xi = 2x$, clears the halves: the fine
vertices of $S$ are exactly the sites of a $(2N)^D$ array, and

.. math::

    \mathrm{rank}\, b(F) = \dim F = \#\{\mu : \xi_\mu \text{ odd}\},

which is precisely the :ref:`interlaced <interlaced>` "$p$-form site"
condition.  A rank-$p$ barycenter and an interlaced $p$-form site are the
same array element; the barycentric construction assigns those sites a
second, independent meaning as the vertices of the subdivided complex.

A fine $p$-simplex is a flag $F_0 \subset \cdots \subset F_p$ of length
$p + 1$, i.e. a chain of $p$ steps up in rank.  In doubled coordinates this
is a **staircase**: starting from the simplex's unique lowest-rank vertex
--- its **anchor** $\xi$ --- each step releases a disjoint, nonempty, signed
set of directions $\Delta_i$, flipping those coordinates from even to odd
and raising the rank by $|\Delta_i|$,

.. math::

    v_i = v_{i-1} + \sum_{\mu \in \Delta_i} \sigma_\mu \hat\varepsilon_\mu,
    \qquad v_0 = \xi.

A step releasing more than one direction at once corresponds to a flag that
skips ranks (its chain of faces jumps dimension by more than one at that
step).  So a fine $p$-simplex is completely specified by an anchor and a
**type** $(\Delta_1, \ldots, \Delta_p)$, and conversely every anchor/type
pair whose touched directions are all even at the anchor is a real fine
simplex.

Storage
~~~~~~~

A $p$-cochain is therefore stored as an array of shape ``(T_p,) +
(2N,)*D``: the leading axis enumerates the types, and the remaining $D$ axes
give the anchor in doubled coordinates.  $T_p$, the number of types, and
every table that depends on the type alone (which directions a type
touches, its vertex offsets, its measure, ...) depend only on $D$, not on
$N$, so they are built once per $D$ and cached; the $N$-dependent arrays
(which anchors are real, the rank of each site, ...) hang off
:class:`supervillain.lattice.barycentric.Lattice`.  This mirrors the interlaced module's own
per-$D$ cache, for the same reason: the type tables are combinatorial and
lattice-size-independent, so there is no reason to rebuild them per
instance.

.. autoclass :: supervillain.lattice.barycentric.Lattice
   :members:

.. autoclass :: supervillain.lattice.barycentric.Form
   :no-index:
   :members:
   :show-inheritance:

Exterior Derivative
--------------------

On a fine $(p+1)$-simplex with vertices $u_0 \cdots u_{p+1}$ in increasing
rank, the coboundary sums over omitting one vertex at a time:

.. math::

    (da)[u_0 \cdots u_{p+1}] = \sum_{i=0}^{p+1} (-1)^i\,
        a[u_0 \cdots \hat u_i \cdots u_{p+1}].

On staircases, omitting an interior vertex $u_i$ merges the steps $\Delta_i$
and $\Delta_{i+1}$ into one; omitting $u_0$ deletes the first step and
re-anchors at $v_1$; omitting the last vertex simply deletes the last step.
Every codimension-two face of a fine simplex appears twice in this sum with
opposite signs, so $d^2 = 0$ exactly, at finite lattice spacing --- verified
in ``test_d_squared_is_zero``.

.. autofunction :: supervillain.lattice.barycentric.d

Codifferential
--------------

:func:`supervillain.lattice.barycentric.delta` is the formal adjoint of :func:`supervillain.lattice.barycentric.d`
with respect to the diagonal pairing over fine simplices,
$\langle a, b \rangle = \sum_{\sigma} a_{\sigma} b_{\sigma}$, so that
$\sum (da)\cdot b = \sum a \cdot (\delta b)$ --- the strongest available
check that the coboundary table is right, verified in ``test_adjointness``.
The model itself never needs $\delta$.

.. autofunction :: supervillain.lattice.barycentric.delta

Cup Product
-----------

The Alexander--Whitney cup product evaluates a $p$-cochain on the first
$p+1$ vertices of a simplex and a $q$-cochain on the last $q+1$:

.. math::

    (a \smile c)[v_0 \cdots v_{p+q}] = a[v_0 \cdots v_p]\, c[v_p \cdots v_{p+q}].

There is exactly one term per output simplex, no sum over shuffles --- what
distinguishes this from the hypercubic wedge product of
:func:`supervillain.lattice.interlaced.wedge`, and why the charge built from
it cannot cancel between contributions inside a single constraint cell.
Bilinearity, associativity, and the Leibniz rule
$d(a \smile c) = da \smile c + (-1)^p a \smile dc$ all hold exactly and are
checked directly rather than assumed.

.. autofunction :: supervillain.lattice.barycentric.cup

A Model's Charge
----------------

In $D = 4$, the cup-square of the flux $f = dn$ with the orientation sign
attached is the topological charge of the no-intersection model,

.. math::

    q_\sigma = \epsilon_\sigma\, f_{012}\, f_{234} = (f \smile f)[\sigma^+].

Reading the cup-square in rank order, it selects the triangles $012$ and
$234$ of a fine four-simplex, which meet at the rank-two vertex $v_2$: the
first lies inside an original plaquette $P$ and the second inside its dual
cell $P^*$.  Summed over the 64 fine four-simplices meeting at a plaquette
centre, the total charge factorizes as $\Phi(P)\cdot\Phi(P^*)$, the product
of the primal and dual fluxes through the two triangle families --- so the
constraint $q_\sigma = 0$ everywhere says flux may occupy one family or the
other at a plaquette centre, never both.  Both the factorization and the
occupancy rule it implies are verified directly, in
``test_charge_factorizes_into_primal_and_dual_flux`` and
``test_constraint_collapses_to_the_occupancy_rule``.

The lattice does **not** provide this charge.  It provides the subdivision,
the exterior calculus, and the orientation sign; which product a model forms
out of them, and what that product is taken to mean, belongs to the model.
Building it is two lines:

.. code-block:: python

    import numpy as np
    import supervillain.lattice.barycentric as bary

    lat = bary.Lattice(4, N)
    f   = bary.d(n)                                  # the flux, a 2-cochain
    q   = bary.Form(np.asarray(bary.cup(f, f)) * lat.orientation_sign,
                    degree=4, lattice=lat)           # the model's charge

and the no-intersection constraint is then ``(np.asarray(q) == 0).all()``.
A different model on the same lattice would form a different product, or
none, and nothing in :class:`~supervillain.lattice.barycentric.Lattice`
presumes which.

Symmetries
----------

Ordinary Translations
~~~~~~~~~~~~~~~~~~~~~~

A *whole unit cell* translation rolls the doubled array by twice the shift,
exactly as :func:`supervillain.lattice.interlaced.push` rolls by the shift
itself once every coordinate is already doubled --- but a *raw* half-step
roll is deliberately **not** offered as a public symmetry here.  On the
barycentric complex a half-step in a single direction carries, say, a
rank-one edge midpoint to a rank-two plaquette center, which is not a
symmetry of $S$ at all; the half-step roll survives only as the private
``_gather``, used internally by :func:`supervillain.lattice.barycentric.d` and
:func:`supervillain.lattice.barycentric.cup` to read a neighbouring barycenter.

.. autofunction :: supervillain.lattice.barycentric.push

.. autofunction :: supervillain.lattice.barycentric.pull

The Point Group
~~~~~~~~~~~~~~~~

The full hypercubic point group $\{\pm 1\}^D \rtimes S_D$ --- sign flips and
axis permutations, order $2^D D!$ --- acts on the barycentric complex, not
merely the cubical model's order-48 subgroup.  Because a signed coordinate
permutation preserves rank, it preserves the rank-ordering of every flag, so
unlike :func:`supervillain.lattice.reflect` and
:func:`supervillain.lattice.permute` in the compact representation, **no
orientation sign is needed**: the whole transformation is a permutation of
the type axis (``_Structures.relabel``) together with the
ordinary array flip or transpose on the site axes.  All $2^D D!$ elements
are measured, not assumed, to act as distinct transformations and to
preserve the support, the link-class multiset, and the no-intersection
constraint set.  The charge transforms covariantly under all 384 elements,
$q(g \cdot n) = \det(g) \cdot g \cdot q(n)$, which is verified in
``test_point_group_order_is_measured_not_assumed`` (the order itself) and
``test_whole_point_group_acts_covariantly_on_the_charge`` (covariance, which
implies constraint-set preservation).

.. autofunction :: supervillain.lattice.barycentric.reflect

.. autofunction :: supervillain.lattice.barycentric.permute

The Half-Diagonal Translation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translating by half the diagonal, $\tau : x \mapsto x + \tfrac12
(1, \ldots, 1)$, is a further symmetry that exists in *every* $D$, for a
structural reason: it carries the cubulation $K$ to the dual cubulation
$K^*$, and $\mathrm{sd}\,K = \mathrm{sd}\,K^*$ because dual cells share
barycenters and reverse nesting.  That duality is why $\tau$ reverses rank,
$r \mapsto D - r$, and why it carries link class $(r,s)$ to $(D-s, D-r)$
--- verified in ``test_tau_reverses_rank`` (which exercises the geometry via
``site_rank``, not the ``tau`` operator itself) and
``test_tau_maps_link_class_antidiagonally``.

Of the $2^D$ parity shifts $\tfrac12 v$ with $v \in \{0,1\}^D$, $\tau$ (and
the trivial $v = 0$) is the *only* one that carries the set of fine links
to itself and the set of fine triangles to itself; the mixed shifts permute
the fine vertices but do not carry links to links, even though several of
them happen to preserve individual links --- which is exactly why
``test_only_the_all_ones_parity_shift_preserves_the_complex`` enumerates
the whole link and triangle sets rather than spot-checking.  That
enumeration is run at $D = 2$ ($N = 3$), $D = 3$ ($N = 2$) and $D = 4$
($N = 2$); the claim for general $D$ rests on the structural argument
above, not on the enumeration.

Because rank reverses, so does every simplex's vertex order, so $\tau$
re-anchors each simplex at the opposite end and carries the sign
$(-1)^{p(p+1)/2}$ on $p$-cochains.  Both halves of that --- *where* the
image lands and *which* sign it carries --- are asserted directly in
``test_tau_moves_a_marked_simplex_to_the_predicted_slot_with_the_predicted_sign``,
which marks a single fine $p$-simplex, applies $\tau$, and requires the
result to be the predicted anchor $\xi + v_p + \mathbb{1}$, the type
obtained by reading the staircase backwards, and that sign, for $p = 1, 2,
3$.  The prediction is built from the ``offsets`` table and the closed form
for the sign, not from the implementation's own $\tau$ tables, so it fails
if either the roll direction or the sign is changed.  An even power of
$\tau$ is an ordinary translation and carries neither the rank reversal nor
the sign: ``test_tau_squared_is_the_unit_diagonal_translation`` applies the
odd branch *twice* and compares the composition against
``translate(a, (1,)*D)``, rather than comparing ``tau(a, 2)`` --- which
short-circuits to exactly that call --- against it.

.. autofunction :: supervillain.lattice.barycentric.tau

The other half-diagonals $\tfrac12(\pm 1, \ldots, \pm 1)$ are symmetries but
not independent ones: each is $\tau$ composed with a unit-cell translation,
e.g. $\tfrac12(1,1,1,-1) = \tfrac12(1,1,1,1) + (0,0,0,-1)$.
:func:`supervillain.lattice.barycentric.translate_diagonal` is generic in $D$ and reduces to
exactly that composition, which is asserted rather than assumed in
``test_translate_diagonal_reduces_to_tau_and_a_translation``.

.. autofunction :: supervillain.lattice.barycentric.translate_diagonal

.. seealso ::

    :ref:`The interlaced picture <interlaced>`, whose doubled-coordinate
    array this module reuses, and the other
    :ref:`reference implementations <reference_implementations>`, which
    this module belongs alongside: a fixed, deliberately unoptimized target
    for correctness rather than production code.
