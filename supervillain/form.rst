
******************
Differential Forms
******************

Fields live on the cells of a :class:`~supervillain.lattice.Lattice`.
A 0-form assigns a value to every site, a 1-form to every link, a 2-form to every plaquette, and so on; a p-form in D dimensions has :math:`\binom{D}{p}` components per site.

.. autoclass :: supervillain.lattice.Form
   :members:
   :show-inheritance:

Translation
===========

Forms can be translated around the (periodic) lattice.

.. autofunction :: supervillain.lattice.pull

.. autofunction :: supervillain.lattice.push


.. _sign-conventions:

Sign Conventions
================

A component of a $p$-form is labeled by a strictly increasing tuple of $p$ directions, stored along axis 0 (see :ref:`interlaced`): the component $f_I$ with $I = (i_1 < \cdots < i_p)$ is the coefficient of $dx_{i_1} \wedge \cdots \wedge dx_{i_p}$, with the factors in increasing order.

Every sign in the exterior calculus has the same origin: an operation naturally produces a tuple of directions *out of order*, and restoring sorted order costs the signature of the sorting permutation.
Write $\sigma(t)$ for the sign of the permutation that sorts the tuple $t$, and $\frown$ for concatenation.
Write $\Delta_e A[x] = A[x + \hat{e}_e] - A[x]$ for the forward finite difference in direction $e$, and $\nabla^*_e A[x] = A[x] - A[x - \hat{e}_e]$ for the backward finite difference.

Exterior Derivative
===================

.. math::

   (d f)_{O}[x] = \sum_{e \in O} \sigma\big((e) \frown (O \setminus e)\big)\; \Delta_e f_{O \setminus e}[x]

Adding direction $e$ wedges $dx_e$ onto the front of $dx_{O \setminus e}$; sorting it into place costs $\sigma\big((e) \frown (O \setminus e)\big) = (-1)^{\#\{i \in O \setminus e \;:\; i < e\}}$.

.. autofunction :: supervillain.lattice.d

The exterior derivative is exact, meaning that it satisfies

.. math ::
   
   d^2 = d \circ d = 0,

which is satisfied by :func:`supervillain.lattice.d`.
This identity is tested by ``test_d_nilpotent`` in :source:`test/test_lattice.py`.

Codifferential
==============

The codifferential is defined as the formal adjoint of the exterior derivative,

.. math::
   :label: d-delta-formal-adjoint

   \langle d a, b \rangle = \langle a, \delta b \rangle

where the $\langle \cdot, \cdot \rangle$ is inner product.
This identity is tested by the ``test_compact_adjointness`` test in :source:`test/test_lattice.py`.


.. math::

   (\delta f)_{M}[x] = - \sum_{e \notin M} \sigma\big((e) \frown M\big)\; \nabla^*_e f_{M \cup \{e\}}[x]

The same insertion sign accounts for removing $e$ from the sorted source tuple $M \cup \{e\}$; the overall minus makes $\delta$ the formal adjoint of $d$.

.. autofunction :: supervillain.lattice.delta

The codifferential is nilpotent, meaning that it satisfies

.. math::

   \delta^2 = \delta \circ \delta = 0.

This identity is tested by the ``test_codifferential_nilpotent`` test in :source:`test/test_lattice.py`.

The continuum identity $\delta = (-1)^{D(k+1)+1}\,\star\,d\,\star$ holds on the lattice only up to a translation; see :ref:`the note below <lattice-star-d-star-shift>`.

Laplacian
=========

The Hodge--de Rham Laplacian is the symmetric combination of the exterior derivative and the codifferential,

.. math::

   \Delta = d\delta + \delta d,

mapping a $p$-form to a $p$-form.
Because $\delta$ is the adjoint of $d$ (:eq:`d-delta-formal-adjoint`), the Laplacian is self-adjoint and positive semidefinite,

.. math::

   \langle \Delta f, f \rangle = \langle d f, d f \rangle + \langle \delta f, \delta f \rangle \geq 0,

checked by the ``test_laplacian_self_adjoint`` and ``test_laplacian_positive_semidefinite`` tests in :source:`test/test_lattice.py`.

On the flat periodic lattice $d$ and $\delta$ are constant-coefficient combinations of the commuting shift operators $T_k$, so the Weitzenböck cross-terms cancel through $\{dx_k \wedge,\, \iota_l\} = \delta_{kl}$ and the Laplacian acts *diagonally on each component* $I$ as the negative of the ordinary nearest-neighbor scalar Laplacian,

.. math::

   (\Delta f)_{I}[x] = \sum_{k=0}^{D-1}\big(2 f_I[x] - f_I[x + \hat{e}_k] - f_I[x - \hat{e}_k]\big),

with no mixing between the $\binom{D}{p}$ components.
This diagonal form agrees with the explicit composition $d\delta + \delta d$, as checked by the ``test_laplacian_matches_d_delta`` test in :source:`test/test_lattice.py`.

.. autofunction :: supervillain.lattice.laplacian

Hodge Star
==========

For each output component $J$ (a sorted $(D-p)$-tuple of directions), let $I = \{0,\ldots,D-1\} \setminus J$ be its complement:

.. math::

   (\star f)_{J}[x] = \sigma\big(I \frown J\big)\; f_{I}\!\left[x - {\textstyle\sum_{k \in I}} \hat{e}_k\right]

The sign sorts the concatenation $(I, J)$ into $(0, \ldots, D-1)$, so that $dx_I \wedge [\sigma(I \frown J)\, dx_J] = dx_0 \wedge \cdots \wedge dx_{D-1}$; the shift $-\hat{e}_I$ aligns the dual cell with the original in the interlaced geometry.

.. autofunction :: supervillain.lattice.star

The Hodge inner-product identity

.. math::

   \sum_{x} (a \wedge \star b)_{(0, \ldots, D-1)}[x] = \langle a, b \rangle

holds exactly, which is checked by the ``test_hodge_inner_product`` test in :source:`test/test_lattice.py`.

Wedge Product
=============

On the lattice the wedge product sums over all shuffles $O = A \sqcup B$, where $A$ are the $n$ directions of $a$ and $B$ are the $m$ directions of $b$:

.. math::

   (a \wedge b)_{O}[x] = \sum_{O = A \sqcup B} \sigma\big(A \frown B\big)\; a_A[x]\; b_B\!\left[x + {\textstyle\sum_{k \in A}} \hat{e}_k\right]

Each shuffle contributes the sign that sorts $(A, B)$ back into $O$, and $b$ is evaluated on the far side of the $a$-cell.

.. autofunction :: supervillain.lattice.wedge

The wedge product is bilinear,

.. math::

   (a+b) \wedge c = a \wedge c + b \wedge c
   a \wedge (b+c) = a \wedge b + a \wedge c

which is checked by the ``test_wedge_bilinear`` test in :source:`test/test_lattice.py`,
and it is associative,

.. math::

   (a \wedge b) \wedge c = a \wedge (b \wedge c)

which is checked by the ``test_wedge_associative`` test in :source:`test/test_lattice.py`.
The Leibniz rule

.. math::
   :label: leibniz-rule

   d(a \wedge b) = da \wedge b + (-1)^{\deg a}\, a \wedge db

holds exactly, which is checked by the ``test_leibniz_rule`` test in :source:`test/test_lattice.py`.

The wedge also satisfies the inner product identity

.. math::

   \sum_x a_x b_x = \sum_{x} (a \wedge \star b)_{(0, \ldots, D-1)}[x]

which holds exactly, as checked by the ``test_wedge_hodge_inner_product`` test in :source:`test/test_lattice.py`.

Unlike the continuum, however, the lattice wedge product is not anti-commutative; see :ref:`the note below <lattice-wedge-commutativity-fail>`.

Differences from the Continuum
==============================

In the continuum, on a Riemannian manifold with Euclidean signature, the Hodge star squares to

.. math::

   \star\star a = (-1)^{p(D-p)}\, a \qquad \text{on a } p\text{-form.}

.. _lattice-star-star-shift:

.. danger::

   On the lattice the identity picks up a spatial shift,

   .. math::

      (\star\star f)_I[x] = (-1)^{p(D-p)}\, f_I\!\left[x - \hat{e}_{\mathrm{all}}\right]

   a backward shift of $\hat{e}_{\mathrm{all}} = \sum_\mu \hat{e}_\mu$ — one step in every direction.
   The shift drops out of any periodic sum.
   This is tested by ``test_star_star`` in :source:`test/test_lattice.py`.

In the continuum we expect anti/commutativity

.. math::

   (a \wedge b) = (-1)^{n m} (b \wedge a)

to hold.

.. _lattice-wedge-commutativity-fail:

.. danger::

   On the lattice this property fails!

In the continuum, on a Riemannian manifold with Euclidean signature, the Hodge star satisfies

.. math::

   \delta = (-1)^{D(k+1)+1}\, \star\, d\, \star \qquad \text{on } k\text{-forms}.

.. _lattice-star-d-star-shift:

.. danger ::

   On the lattice the identity picks up a spatial shift,

   .. math::

      \delta = (-1)^{D(k+1)+1}\, T_{\hat{e}_{\mathrm{all}}}\, \star\, d\, \star

   where $\hat{e}_{\mathrm{all}} = \sum_\mu \hat{e}_\mu$ and $T_{\Delta x}$ is the translation operator in :func:`~supervillain.lattice.pull` by $\Delta x$.
   The shift by $\hat{e}_{\mathrm{all}}$ drops out of any periodic sum, which is why the adjoint identity :eq:`d-delta-formal-adjoint` holds exactly despite it.
   This is tested by ``test_star_d_star_equals_shifted_delta`` in :source:`test/test_lattice.py`.

   The translation commutes with the other operations, as tested in ``test_star_d_star_equals_delta_shifted`` in :source:`test/test_lattice.py`.
