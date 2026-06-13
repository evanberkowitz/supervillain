
******************
Differential Forms
******************

Fields live on the cells of a :class:`~supervillain.lattice.Lattice`.
A 0-form assigns a value to every site, a 1-form to every link, a 2-form to every plaquette, and so on; a p-form in D dimensions has :math:`\binom{D}{p}` components per site.

.. autoclass :: supervillain.lattice.Form
   :members:
   :show-inheritance:

.. _sign-conventions:

Sign Conventions
================

A component of a $p$-form is labeled by a strictly increasing tuple of $p$ directions, stored along axis 0 (see :ref:`interlaced`): the component $f_I$ with $I = (i_1 < \cdots < i_p)$ is the coefficient of $dx_{i_1} \wedge \cdots \wedge dx_{i_p}$, with the factors in increasing order.

Every sign in the exterior calculus has the same origin: an operation naturally produces a tuple of directions *out of order*, and restoring sorted order costs the signature of the sorting permutation.
Write $\sigma(t)$ for the sign of the permutation that sorts the tuple $t$, and $\frown$ for concatenation.
Then, with $\Delta_e$ the forward and $\nabla^*_e$ the backward finite difference in direction $e$,

.. math::

   \begin{aligned}
   (d f)_{O}[n] &= \sum_{e \in O} \sigma\big((e) \frown (O \setminus e)\big)\; \Delta_e f_{O \setminus e}[n]
   \\
   (\delta F)_{M}[n] &= - \sum_{e \notin M} \sigma\big((e) \frown M\big)\; \nabla^*_e F_{M \cup e}[n]
   \\
   (\star f)_{J}[n] &= \sigma\big(I \frown J\big)\; f_{I}\big[n - {\textstyle\sum_{k \in I}} \hat{e}_k\big] \qquad I = (0, \ldots, D-1) \setminus J
   \\
   (a \wedge b)_{O}[n] &= \sum_{O = B \sqcup A} \sigma\big(B \frown A\big)\; a_B[n] \; b_A\big[n + {\textstyle\sum_{k \in B}} \hat{e}_k\big]
   \end{aligned}

* In $d$, adding direction $e$ wedges $dx_e$ onto the front of $dx_{O \setminus e}$; sorting it into place costs $\sigma\big((e) \frown (O \setminus e)\big) = (-1)^{\#\{i \in O \setminus e \;:\; i < e\}}$.
* In $\delta$, the same insertion sign accounts for removing $e$ from the sorted source tuple $M \cup e$; the *overall* minus is a convention, chosen so that $\delta$ is the formal adjoint of $d$ (below).
* In $\star$, the dual component is the complement $I \mapsto J$ and the sign sorts the concatenation $(I, J)$ into $(0, \ldots, D-1)$; equivalently, $dx_I \wedge \big[\sigma(I \frown J)\, dx_J\big] = dx_0 \wedge \cdots \wedge dx_{D-1}$.
* In $\wedge$, the output tuple $O$ splits into the $a$-directions $B$ and the $b$-directions $A$ in every possible way (a *shuffle*), and each term carries the sign that sorts the concatenation $(B, A)$ back into $O$.

The finite differences and the spatial shifts in $\star$ and $\wedge$ are the :ref:`interlaced <interlaced>` geometry at work: each factor is read from the cell it geometrically touches.
In $a \wedge b$ the factor $b$ is evaluated on the far side of the $a$-cell, and in $\star$ the dual cell is anchored one step back in every direction of $I$.

With these conventions the following identities hold exactly on the periodic lattice, where $\langle u, v \rangle = \sum_{n, I} u_I[n]\, v_I[n]$ is the componentwise inner product:

.. math::

   \begin{aligned}
   d \circ d = 0 \qquad \qquad \delta \circ \delta &= 0
   \\
   \langle d a, b \rangle &= \langle a, \delta b \rangle
   \\
   d(a \wedge b) &= da \wedge b + (-1)^{\deg a}\, a \wedge db
   \\
   \sum_{n} (a \wedge \star b)_{(0, \ldots, D-1)}[n] &= \langle a, b \rangle
   \end{aligned}

.. note ::

   The *relative* signs are pinned by these identities, but some absolute conventions (most visibly the overall sign of $\delta$) are still under scrutiny and may change.

Calculus
========

The exterior derivative ``d`` and the codifferential ``delta`` move between degrees; both are nilpotent (applying either twice gives zero).

.. autofunction :: supervillain.lattice.d

.. autofunction :: supervillain.lattice.delta

Hodge Star and Wedge Product
============================

.. autofunction :: supervillain.lattice.star

.. autofunction :: supervillain.lattice.wedge

Translation
===========

Forms can be translated around the (periodic) lattice.

.. autofunction :: supervillain.lattice.push

.. autofunction :: supervillain.lattice.pull
