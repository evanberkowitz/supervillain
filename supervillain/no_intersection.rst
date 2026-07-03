.. _no_intersection:

*************************
The No-Intersection Model
*************************


Jacobson observed that the standard Villain model supports an interesting modification which suppresses vortex intersection.

.. math ::
   :label: no-intersection

   S = (d\phi - 2\pi n)^2 + i \theta_p (dn \wedge dn)_p

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
   :show-inheritance:
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
:class:`~supervillain.generator.no_intersection.IntersectionWorm` exploits.

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
For the same reason the :class:`~supervillain.generator.villain.CohomologyUpdate` is also a legal move: it changes $n$ by a *closed* (but not exact) form, so $dn$---and hence $q$---is again untouched, while the torus-wrapping holonomy of $n$ that neither the :class:`~supervillain.generator.villain.ExactUpdate` nor the constraint-preserving $n$-updates below can reach does change.

One thing you might hope is that you could just use the $W=1$ :class:`~supervillain.generator.villain.LinkUpdate` from the Villain model, but that will, in general, generate constraint violations.
But we can try to do something simple: make :class:`~supervillain.generator.villain.LinkUpdate`-like proposals but reject any that violate the constraint.

.. autoclass:: supervillain.generator.no_intersection.ConstrainedLinkUpdate
   :members:

Frozen configurations
======================

The :class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate` *looks* like it should be ergodic --- it is a Metropolis sweep offering every single-link $\pm 1$ move that preserves $q = 0$ --- but it is not.
Because the constraint is *quadratic* in $n$, there exist valid configurations, which we call **frozen**, on which *every* single-link $\pm 1$ move lights up a defect: each link $\ell$ is *blocked* by the background flux $F = dn$ in the planes complementary to $\ell$'s direction, and a frozen configuration is one in which every link is blocked at once.
A frozen configuration is therefore an isolated point of the single-link move graph: since every single-link move off it violates the constraint, the reverse (single-link) move onto it from any neighbor is equally forbidden.
So :class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate` can neither escape a frozen configuration nor reach one, and a single-link-only algorithm is **not ergodic**.

These beasts are not hypothetical.
The example script :source:`example/no-intersection/frozen.py` constructs explicit frozen configurations in closed form --- a single-pair family $F_{01} = a(-1)^{x_{0}}$, $F_{23} = b(-1)^{x_{0}+x_{2}+x_{3}}$, and a genuinely six-plane "delicate cancellation" family $F_{\mu\nu} = A_{\mu\nu}(-1)^{x_{\mu}+x_{\nu}}$ with $\mathrm{Pf}(A) = 0$ --- and verifies by exhaustive search that not one of the $2 D N^{D}$ single-link $\pm 1$ moves preserves $q = 0$.

Escaping (or reaching) a frozen configuration requires a *coordinated* move that changes several links at once.
The key structural fact is that if $\Delta n$ is confined to a single link direction $\mu$ then $d\Delta n \wedge d\Delta n = 0$ identically, so on *any* background the charge change

.. math::

   \Delta q(\Delta n) = F \wedge d\Delta n + d\Delta n \wedge F

is *linear* in $\Delta n$; a coordinated single-direction move that lands in the kernel of this map preserves $q = 0$ exactly, even where every single-link move fails.
The :class:`~supervillain.generator.no_intersection.WrappingLoopUpdate` is one such move: a closed, torus-wrapping loop of single-direction links, proposed and accepted or rejected atomically.

.. autoclass:: supervillain.generator.no_intersection.WrappingLoopUpdate
   :members:
   :show-inheritance:

The companion script :source:`example/no-intersection/unfreeze.py` demonstrates that this works: starting *on* a frozen configuration, interleaving :class:`~supervillain.generator.villain.SiteUpdate`, :class:`~supervillain.generator.no_intersection.WrappingLoopUpdate`, and :class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate` leaves the frozen sector --- the wrapping loops open up single-link moves, the single-link sweep fires, and the valid-move count cascades.

A complementary coordinated move deposits a whole *sheet* of flux at once.
Recall that $\mathrm{Pf}(A) = 0$ is exactly the condition that $A$ be *decomposable* --- $A = u \wedge v$ for two 4-vectors, a single vortex plane that does not self-intersect --- and that over the integers *any* $u, v$ give such an $A$.
The :class:`~supervillain.generator.no_intersection.PlanarFluxUpdate` proposes the staggered sheet $F_{\mu\nu}(x) = A_{\mu\nu}(-1)^{(x-t)_\mu + (x-t)_\nu}$ with $u, v \in \{-1,0,1\}^4$ and a random anchor $t$, verifies it keeps $q = 0$, and Metropolis-tests it.
Because it changes $F$ over the whole lattice it makes large jumps --- a tunneling move well matched to the sheet-like frozen sector (a frozen configuration is itself such a sheet) --- though that same size makes its acceptance low at nonzero $\kappa$.

.. autoclass:: supervillain.generator.no_intersection.PlanarFluxUpdate
   :members:
   :show-inheritance:

Worms and the intersection correlator
=====================================

A worm is another route to large, coordinated moves --- and the one that additionally yields a physical observable --- just as in the modified Villain model and the worldline formulation.
We can imagine inserting a worm with a head and tail built of exponentials of Lagrange-multiplier fields (in this case the 4-form $\theta$) on the same hypercube and allowing the head to move from hypercube to hypercube by changing $n$.

However, unlike in $D=2$, where the worm lives on plaquettes and crosses a single link to move to a neighboring plaquette, the worm here must cross a 3-dimensional cube to reach an orthogonally-adjacent neighboring hypercube (if you prefer, think of the hypercube as a site on the dual lattice and the cube as a dual link).
Therefore, we expect to need to make coordinated moves simultaneously updating 3 links at once to push the topological defect around.
In fact, if you don't require the links to share a corner, you can make coordinated 2-link moves that push the worm orthogonally.
Coordinated 2-link and 4-link moves can also push the worm defect diagonally.

But, the complication is that the above discussion is on an $F=0$ background.
On a nontrivial background, those coordinated moves are not always legal.
And, on a nontrivial background, moves that are not legal on an $F=0$ background can advance the worm defect without violating the constraint!
This can add a lot of complication and furthermore adds concern about the ergodicity of the worm.

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

The two-point function of the charge-insertion operator $e^{i\theta}$ 

.. math ::

   \Theta_{x,y} = \left\langle e^{i(\theta_x - \theta_y)} \right\rangle

is conjugate to the no-intersection constraint and poses a tricky problem to evaluate: if we sample configurations of $Z$ we integrate $\theta$ out first and can no longer see the field needed for the obvious way to compute the observable.
Instead we absorb the insertion into the action *before* path-integrating out $\theta$.
Because $\theta_x$ multiplies $q_x = (dn \wedge dn)_x$, integrating $\theta_x$ against the extra $e^{i\theta_x}$ shifts the constraint at the insertions

.. math ::
   :name: theta worm constraint

   S[\phi, n, \theta] - i(\theta_x - \theta_y)
   \rightarrow
   \Theta_{x,y} = \frac{1}{Z} \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n]} \prod_p [(dn \wedge dn)_p = \delta_{px} - \delta_{py}]

where now $x$ and $y$ label hypercubes: the insertion demands exactly one unit of topological-charge density at $x$ and a compensating unit at $y$.
Constructing such an overlay by hand, as in the :class:`~supervillain.observable.Spin_Spin`'s taxicab Villain-frame implementation, hits a similar overlap problem: we must lay down a whole sheet of $F = dn$ connecting $y$ to $x$, and unless it threads the valley of the action the change in action is enormous and the correlator is tiny except on rare configurations.
In fact it's much more challenging in this model because the $dn \wedge dn$ constraint is quadratic in $n$ and the overlay must be a sheet of $n$s that satisfies the constraint.

Following Prokof'ev and Svistunov :cite:`PhysRevLett.87.160601` we instead introduce defects where the constraint may be broken, propagate them, and celebrate when they meet.
Consider the mixed regular+path integral $G$ with unspecified normalization $N$ (which cancels from everything of interest)

.. math ::

   G = \frac{1}{N} \sum\hspace{-1.33em}\int D\phi\; Dn\; dh\; dt\; e^{-S[\phi, n]} \prod_p [(dn \wedge dn)_p = \delta_{ph} - \delta_{pt}].

A configuration of $G$ carries a $+1$ unit of $q$ at the head hypercube $h$, a $-1$ unit at the tail hypercube $t$, and $q = 0$ everywhere else.
When the head and tail coincide the constraint is restored everywhere and the $G$ configuration is a valid $Z$ configuration appearing with the right relative frequency.
To insert a worm we drop the head and tail on the same randomly-chosen hypercube; the change in action is 0 and the move is automatically accepted.
Moving the head then means changing $q$ on both the departure and destination cells---restoring the constraint at the former and breaking it at the latter---which, as described above, requires a coordinated three-link change of $n$ that extends the dragged sheet of $F = dn$.
Each such move changes the Villain action and so is Metropolis-tested; when the head returns to the tail we may emit the configuration back into the $Z$ chain.

Notice that

.. math ::
   :name: theta worm histogram

   \Theta_{x,y} = \frac{\left\langle \delta_{xh} \delta_{yt} \right\rangle_G}{\left\langle \delta_{ht} \right\rangle_G}

where the expectation values are over configurations drawn from $G$ (not $Z$!).
If we draw from the larger space of $G$ configurations and histogram the head$-$tail displacement, normalizing that histogram by its value at zero displacement recovers $\Theta_{x,y}$.
We accumulate the histogram as the worm evolves and save it inline with $\phi$ and $n$ as ``Intersection_Intersection`` (alongside the ``Worm_Length``), remembering to normalize any :class:`~.DerivedQuantity` built from it by its value at the origin---exactly as for :class:`~.Vortex_Vortex`.

.. autoclass:: supervillain.generator.no_intersection.IntersectionWorm
   :members:
   :show-inheritance:

The worm accumulates its head$-$tail displacement histogram inline as the
:class:`~.Intersection_Intersection` observable, and :class:`~.Intersection_Intersection_Normalized`
divides it by its value at the origin (which can only be done after the
bootstrap).

.. autoclass:: supervillain.observable.Intersection_Intersection
   :members:
   :show-inheritance:

.. autoclass:: supervillain.observable.Intersection_Intersection_Normalized
   :members:
   :show-inheritance:

In the Villain model constraint is linear and therefore we could construct an :class:`~.ExactUpdate` which was in the kernel of the constraint that was essentially a closed 4-plaquette :class:`~supervillain.generator.villain.ClassicWorm`.
Similarly in the Worldline model the :class:`~supervillain.generator.worldline.PlaquetteUpdate` could be understood as the smallest nontrivial worm.
These could be essentially proposed everywhere because they automatically preserve the constraint.
The essential fact was that the constraint is linear.
But here we have a quadratic constraint and therefore it is not always legal to just stamp a tight worm on an existing configuration---it could break the constraint.

The Hammer
==========

We provide the :func:`~supervillain.generator.no_intersection.Hammer` function to :class:`~.Sequentially` combine the various constraint-preserving updates into a single generator.

.. autofunction:: supervillain.generator.no_intersection.Hammer

