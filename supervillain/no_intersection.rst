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

To see precisely what such a worm measures, remember that in the :class:`~.NoIntersections` case we are trying to sample according to

.. math ::

   \begin{aligned}
       Z &= \sum\hspace{-1.33em}\int D\phi\; Dn\; D\theta\; e^{-S[\phi, n, \theta]}
       \\
       S[\phi, n, \theta] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + i \sum_x \theta_x (dn \wedge dn)_x
   \end{aligned}

and that we may directly path-integrate out the Lagrange multiplier $\theta$ in favor of the constraint

.. math ::

   Z = \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n]} \prod_x [(dn \wedge dn)_x = 0].

Just as the :class:`~.Vortex_Vortex` correlator is the object conjugate to the Villain winding constraint, the object conjugate to the no-intersection constraint is the two-point function of the charge-insertion operator $e^{i\theta}$,

.. math ::

   \Theta_{x,y} = \left\langle e^{i(\theta_x - \theta_y)} \right\rangle,

which poses the same tricky problem to evaluate: if we sample configurations of $Z$ we integrate $\theta$ out first and can no longer see the observable.
Instead we absorb the insertion into the action *before* path-integrating out $\theta$.
Because $\theta_x$ multiplies $q_x = (dn \wedge dn)_x$, integrating $\theta_x$ against the extra $e^{i\theta_x}$ shifts the constraint at the insertions

.. math ::
   :name: theta worm constraint

   S[\phi, n, \theta] - i(\theta_x - \theta_y)
   \rightarrow
   \Theta_{x,y} = \frac{1}{Z} \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n]} \prod_p [(dn \wedge dn)_p = \delta_{px} - \delta_{py}]

where now $x$ and $y$ label hypercubes: the insertion demands exactly one unit of topological-charge density at $x$ and a compensating unit at $y$.
Constructing such a configuration by hand hits exactly the overlap problem of :ref:`the Villain vortex correlator <worm constraint>`---we must lay down a whole sheet of $F = dn$ connecting $y$ to $x$, and unless it threads the valley of the action the change in action is enormous and the correlator is tiny except on rare configurations.

Following Prokof'ev and Svistunov :cite:`PhysRevLett.87.160601` we instead introduce defects where the constraint may be broken, propagate them, and celebrate when they meet.
Consider the mixed regular+path integral $G$ with unspecified normalization $N$ (which cancels from everything of interest)

.. math ::

   G = \frac{1}{N} \sum\hspace{-1.33em}\int D\phi\; Dn\; dh\; dt\; e^{-S[\phi, n]} \prod_p [(dn \wedge dn)_p = \delta_{ph} - \delta_{pt}].

A configuration of $G$ carries a $+1$ unit of $q$ at the head hypercube $h$, a $-1$ unit at the tail hypercube $t$, and $q = 0$ everywhere else.
When the head and tail coincide the constraint is restored everywhere and the $G$ configuration is a valid $Z$ configuration appearing with the right relative frequency.
To insert a worm we drop the head and tail on the same randomly-chosen hypercube; the change in action is 0 and the move is automatically accepted.
Moving the head then means changing $q$ on both the departure and destination cells---restoring the constraint at the former and breaking it at the latter---which, as described above, requires a coordinated three-link change of $n$ that extends the dragged sheet of $F = dn$.
Each such move changes the Villain action and so is Metropolis-tested; when the head returns to the tail we may emit the configuration back into the $Z$ chain.

Just as for the vortex correlator, notice that

.. math ::
   :name: theta worm histogram

   \Theta_{x,y} = \frac{\left\langle \delta_{xh} \delta_{yt} \right\rangle_G}{\left\langle \delta_{ht} \right\rangle_G}

where the expectation values are over configurations drawn from $G$ (not $Z$!).
If we draw from the larger space of $G$ configurations and histogram the head$-$tail displacement, normalizing that histogram by its value at zero displacement recovers $\Theta_{x,y}$.
We accumulate the histogram as the worm evolves and save it inline with $\phi$ and $n$ as ``Theta_Theta`` (alongside the ``Worm_Length``), remembering to normalize any :class:`~.DerivedQuantity` built from it by its value at the origin---exactly as for :class:`~.Vortex_Vortex`.

Finally, note that $\Theta_{x,y} = \langle e^{i(\theta_x - \theta_y)}\rangle$ is the correlator of the field *conjugate* to the charge density---the disorder correlator dual to the density--density correlator :class:`~.Topological_Topological` $= \langle q_x q_y\rangle_c$.
The worm therefore supplies information complementary to what the directly-binned :class:`~.TopologicalTwoPoint` measures; it is not simply that observable up to a normalization.

.. autoclass:: supervillain.generator.no_intersection.ThetaWorm
   :members:

.. autoclass:: supervillain.generator.no_intersection.WrappingLoopUpdate
   :members:

.. autofunction:: supervillain.generator.no_intersection.Hammer

