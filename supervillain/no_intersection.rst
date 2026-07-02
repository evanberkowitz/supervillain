.. _no_intersection:

*************************
The No-Intersection Model
*************************


Jacobson observed that the standard Villain model supports an interesting modification which suppresses vortex intersection.

.. math ::
   :label: no-intersection

   S = (d\varphi - 2\pi n)^2 + i \theta_p (dn \wedge dn)_p

with $\theta$ a real-valued 4-form that when path-integrated enforces $dn \wedge dn = 0$.
The physical interpretation is that vortices may not intersect in this model.

The model has two $U(1)$ symmetries, the standard Villain $U(1)$ and the shift symmetry of $\theta$.
Remarkably, neither symmetry is anomalous on its own, but they have a mixed anomaly that is an axial-vector-vector anomaly in 4D, just like the ABJ anomaly!

This model must undergo a transition of some kind as you tune $\kappa$ but the character of that transition is unknown.
It may be first order.
There may be multiple boring (BKT-like) transitions that take us from one of the $U(1)$s being broken to the other.

But the most exciting possibility is that there is one continuous transition yielding an interesting CFT.
There are only so many 4D CFTs known with the right AVV anomaly structure, and they all include fermions.
It could be that this model is a back-door strategy for 4D bosonization!

The Action
==========

.. autoclass:: supervillain.action.NoIntersections
   :members:

The Topological Charge and the Constraint
=========================================

The no-intersection constraint asks that the topological-charge density $q_x$
(the same density measured by :class:`~supervillain.observable.TopologicalChargeDensity`)

.. math::

   q_x = (dn \wedge dn)_x = d(n \wedge dn)_x = (dJ)_x
   \qquad J = n \wedge dn

vanish on every hypercube.  The second equality is exact on the lattice (the
Leibniz rule and $d^2 = 0$ both hold), so $q$ is the divergence of the 3-form
current $J$ and is locally conserved and integer-valued.  A localized closed
$F = dn$ carries zero total charge $Q = \sum_x q_x$, so violations of the
constraint always come as a $+1$ / $-1$ dipole --- the fact that the
:class:`~supervillain.generator.no_intersection.ThetaWorm` exploits.

Generators
==========

The No-Intersection generators update $n$ while preserving $q = 0$; combine any
of them with a $\phi$-update (they are bundled with a
:class:`~supervillain.generator.villain.SiteUpdate` in the :func:`Hammer` below).
They are pure-python reference implementations restricted to $D = 4$.

How can we go about updating the fields in a way that obeys the constraint?
First, we can use :class:`~supervillain.generator.villain.SiteUpdate` to update the $\phi$ field which doesn't directly see the constraint (its action is evaluated in exactly the same way on the constraint surface and on the whole space of unconstrained $n$s).

Second, we can use :class:`~supervillain.generator.villain.ExactUpdate` to update the $n$ field.
Because it makes exact updates to $n$ it is guaranteed not to alter $dn$ and therefore cannot change the charge density $q$ anywhere, so it manifestly maintains the constraint.

One thing you might hope is to just use the $W=1$ :class:`~supervillain.generator.villain.LinkUpdate` from the Villain model, but that will, in general, generate constraint violations.
But we can try to do something simple: make :class:`~supervillain.generator.villain.LinkUpdate`-like proposals but reject any that violate the constraint.

.. autoclass:: supervillain.generator.no_intersection.ConstrainedLinkUpdate
   :members:

Just as in the modified Villain model and the worldline formulation we can think of another kind of generator that makes large, coordinate moves: worms!
We can imagine inserting a worm with a head and tail built of exponentials of Lagrange-multiplier fields (in this case the 4-form $\theta$) on the same hypercube and allowing the head to move from hypercube to hypercube by changing $n$.

However, unlike in $D=2$, where the worm lives on plaquettes and crosses a single link to move to a neighboring plaquette, the worm here must cross a 3-dimensional cube to reach a neighboring hypercube (if you prefer, think of the hypercube as a site on the dual lattice and the cube as a dual link).
Therefore, we expect to need to make coordinated moves simultaneously updating 3 links at once to push the topological defect around.
The fact that it is even possible to move $q$ around without a proliferation of constraint violations can be seen as a repercussion of the fact that $q=dJ$ is locally conserved.

.. autoclass:: supervillain.generator.no_intersection.ThetaWorm
   :members:

.. autoclass:: supervillain.generator.no_intersection.WrappingLoopUpdate
   :members:

.. autofunction:: supervillain.generator.no_intersection.Hammer

