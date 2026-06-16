
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

Differences from the Continuum
-------------------------------

In the continuum we expect anti/commutativity

.. math::

   (a \wedge b) = (-1)^{n m} (b \wedge a)

to hold.

.. danger::
   
   On the lattice this property fails!

In the continuum, on a Riemannian manifold with Euclidean signature, the Hodge star satisfies

.. math::

   \delta = (-1)^{D(k+1)+1}\, \star\, d\, \star \qquad \text{on } k\text{-forms}.

.. danger ::

   On the lattice this identity picks up a spatial shift.  In interlaced coordinates the shift is $2\varepsilon_{\mathrm{all}} = (2, \ldots, 2)$ — one physical step in each direction:

   .. math::

      \delta = (-1)^{D(k+1)+1}\, T_{2\varepsilon_{\mathrm{all}}}\, \star\, d\, \star

   where $T_{\Delta\xi}$ is the translation operator (:func:`~supervillain.lattice.interlaced.pull` by $\Delta\xi$).
   The shift drops out of any periodic sum, which is why :eq:`interlaced-d-delta-adjoint` holds exactly despite it.
   This is tested by ``test_star_d_star_equals_shifted_delta`` in :source:`test/test_lattice_interlaced.py`.
