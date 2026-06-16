
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

   d(a \wedge b) = da \wedge b + (-1)^{\deg a}\, a \wedge db

holds exactly, which is checked by the ``test_leibniz_rule`` test in :source:`test/test_lattice.py`.

The wedge also satisfies the inner product identity

.. math::

   \sum_x a_x b_x = \sum_{x} (a \wedge \star b)_{(0, \ldots, D-1)}[x]

which holds exactly, as checked by the ``test_wedge_hodge_inner_product`` test in :source:`test/test_lattice.py`.

Unlike the continuum, however, the lattice wedge product is not anti-commutative; see :ref:`the note below <lattice-wedge-commutativity-fail>`.

Differences from the Continuum
==============================

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


