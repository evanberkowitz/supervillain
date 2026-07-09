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
We give :ref:`a step-by-step derivation of that anomaly <no_intersection_anomaly>` by gauging one $U(1)$ and tracking the resulting integer-valued obstruction to the other.

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

.. autoclass:: supervillain.observable.Intersection_Intersection
   :members:
   :show-inheritance:

.. autoclass:: supervillain.observable.Intersection_Intersection_Normalized
   :members:
   :show-inheritance:


The enlarged $G$ ensemble also has a precise topological meaning, which is the structural reason to hope a worm mixes where local updates struggle.
Poincaré duality turns the flux $F = dn$ into a closed 2-dimensional *vortex sheet* on the dual lattice, and $q_x = (dn \wedge dn)_x$ into the signed density of the sheet's transverse self-intersection points: in 4D two 2-dimensional sheets generically meet at isolated points, and each crossing carries a sign from comparing orientations --- the same $\epsilon^{\mu\nu\rho\sigma}$ contraction that appears in $F \wedge F$.
A valid $Z$ configuration is an *embedded* sheet, with no self-intersections anywhere; a $G$ configuration, with its $+1$ at the head and $-1$ at the tail, is an *immersed* sheet carrying exactly one pair of opposite-sign double points.

That is exactly the enlargement topologists use to connect embedded surfaces in four dimensions.
A classical fact --- assembled from Whitney's disk construction :cite:`Whitney1944`, Casson's finger moves :cite:`Casson`, and general position, and stated systematically in the standard reference of Freedman and Quinn :cite:`FreedmanQuinn` (chapter 1) --- is that any two embedded surfaces in a 4-manifold which are homotopic to one another are connected by a sequence of just three elementary processes: ambient *isotopies* (deformations that never create an intersection), *finger moves* (pushing a patch of sheet through another, creating a $+/-$ pair of double points --- always possible), and *Whitney moves* (the inverse: annihilating a $+/-$ pair by sliding the sheet across an embedded disk).
A generic path between embedded surfaces passes only through immersed surfaces with isolated $\pm$ double-point pairs --- that is, through $G$.
This matters because in 4D embedded surfaces can *knot*: two sheets in the same homotopy class need not be connected through embeddings alone.
They are, however, connected through immersions, and the worm walks precisely that corridor --- its head/tail dipole is the lattice double-point pair, and the corridor's three ingredients correspond one-to-one to the worm's three kinds of move (the dictionary is spelled out in the class documentation below).

**Worms walk the Freedman--Quinn corridor.**  Poincaré-dually, a valid configuration
is an *embedded* vortex sheet (no transverse self-intersections: $q \equiv 0$) and a $G$
configuration is an *immersed* sheet carrying one $+/-$ pair of double points --- the
head and the tail.  A classical fact of 4-manifold topology (Whitney's disk construction
:cite:`Whitney1944`, Casson's finger moves :cite:`Casson`, and general position; stated
systematically by Freedman and Quinn :cite:`FreedmanQuinn`, whose chapter 1 is the
standard reference and lends the corridor its name here) connects any two homotopic
embedded surfaces in a 4-manifold by exactly three elementary processes --- ambient
isotopies, finger moves (double-point pair creation), and Whitney moves (pair
annihilation) --- and the worm's three aspects implement them one-to-one:

 - **opening and the first step $=$ finger move.**  Dropping head $=$ tail on one
   hypercube ($\Delta S = 0$, automatically accepted) is a double-point pair at zero
   separation; the first accepted head move separates the pair --- a patch of sheet
   piercing another, creating the $+1$ and $-1$ transversally.
 - **transport and closing $=$ Whitney move.**  Each subsequent head move drags the $+1$
   double point, extending the dragged sheet of $F = dn$ (the finger); when the head
   rejoins the tail and the worm closes, the pair annihilates --- the Whitney move, with
   the dragged sheet playing the role of the Whitney disk's neighbourhood.
 - **idle moves $=$ ambient isotopy.**  Accepted 1-link draws with $\Delta q \equiv 0$
   rearrange the sheet while both double points stay put.  Without this leg the corridor
   would be incomplete: head transport alone can never move the sheet out of its own way
   at fixed defects.


Two honest caveats temper the optimism.
Freedman--Quinn's homotopies may require *several* double-point pairs in flight at once --- this is Casson's obstruction :cite:`Casson`: Whitney disks can themselves intersect things, and repairing that creates more pairs --- while the worm carries exactly one.
Whether one pair at a time always suffices turns out to be an open problem in 4-manifold topology, but thankfully the answer does not matter for the correctness of this library, only for its mixing rate.
We give :ref:`a separate step-by-step argument<no_intersection_ergodicity>`; the short version is that knotted sheets untie through *embedded, valid* intermediates by genus fluctuations our legal moves perform, so the worm's corridor is a shortcut rather than a necessity.

.. autoclass:: supervillain.generator.no_intersection.IntersectionWorm
   :members:
   :show-inheritance:

The worm accumulates its head$-$tail displacement histogram inline as the
:class:`~.Intersection_Intersection` observable.

In the Villain model constraint is linear and therefore we could construct an :class:`~.ExactUpdate` which was in the kernel of the constraint that was essentially a closed 4-plaquette :class:`~supervillain.generator.villain.ClassicWorm`.
Similarly in the Worldline model the :class:`~supervillain.generator.worldline.PlaquetteUpdate` could be understood as the smallest nontrivial worm.
These could be essentially proposed everywhere because they automatically preserve the constraint.
The essential fact was that the constraint is linear.
But here we have a quadratic constraint and therefore it is not always legal to just stamp a tight worm on an existing configuration---it could break the constraint.

In fact, the above worm performed absolutely dismally.
On cold (large-$\kappa$) backgrounds, $F=0$ it had the room to move the defect around.
But the multi-link moves were so costly that essentally all were rejected.
On warm (small-$\kappa$) backgrounds, the plaquette flux is nonzero and the :class:`~supervillain.generator.no_intersection.IntersectionWorm`'s dipole stencils wound up breaking the constraint!
So it is a very inefficient update scheme no matter $\kappa$.


The adaptive worm
=================

The two failures of the :class:`~supervillain.generator.no_intersection.IntersectionWorm`---costly rejections on the cold background and constraint-violating stencils on the warm one---share a cause: it commits to a single dipole stencil *before* looking at the flux it has to move through.
The :class:`~supervillain.generator.no_intersection.AdaptiveIntersectionWorm` keeps the head/tail construction, the Freedman--Quinn corridor, and the orthogonal 1-, 2-, and 3-link stencils above, but at each step it *enumerates* the clean set $C$ of every move that would advance the head in the drawn direction on the *current* background $F = dn$, draws one uniformly, and corrects for the state-dependence of that count with the exact Metropolis--Hastings ratio

.. math ::

   A = \min\left(1,\; \frac{\left|C\right|}{\left|C'\right|}\, e^{-\Delta S}\right),

where $\left|C'\right|$ is the size of the clean set for the reverse move at the destination.
Because $F$ is a function of the current configuration and not of the chain's history, this is ordinary state-dependent Metropolis--Hastings.
The head now proposes only among moves that are legal on the sheet it is actually standing on, so on a fluxful background it advances instead of stalling.
The exactness turns on three facts, spelled out in the class documentation: the reverse of every clean move is itself clean (so $\left|C'\right| \geq 1$ and the ratio is well defined), the deduplicated uniform draw makes the proposal symmetric up to $\left|C\right|/\left|C'\right|$, and the open/close accounting that emits the worm is left exactly as in the Prokof'ev--Svistunov prescription.
The head-fixed *idle* moves that supply the corridor's isotopy leg are enumerated the same way.

.. autoclass:: supervillain.generator.no_intersection.AdaptiveIntersectionWorm
   :members: step, report
   :show-inheritance:

Even the adaptive worm draws from a *fixed* library of stencils, and on a sufficiently structured background none of those templates happen to be clean---the head can still stall for want of a move of the right shape.
The :class:`~supervillain.generator.no_intersection.TwoLinkAdaptiveWorm` removes that limitation for the two-link sector by *live-enumerating* it: for the drawn direction it tests every pair of nearby links, with coefficients up to $\left|c\right| = 2$ (which carries the mixed-magnitude solutions the quadratic constraint occasionally forces), and keeps every pair whose combined charge change is exactly the head dipole---together with the analogous two-link *idle* isotopies.
It is a strict superset of the adaptive worm: its clean sets still contain all the library moves, so it advances the head wherever the fixed templates can *and* wherever a bespoke two-link move is the only clean option.
Because this is still the same fixed-family-filtered-by-$F$ construction, closed under inversion, the Metropolis--Hastings exactness carries over unchanged; only the candidate set grows---to thousands of shapes per direction, so the clean-set evaluation is compiled (see the class documentation for both the exactness argument and the acceleration).
It is provided as an opt-in generator and is not part of the default :func:`~supervillain.generator.no_intersection.Hammer`.

.. autoclass:: supervillain.generator.no_intersection.TwoLinkAdaptiveWorm
   :members: step_reference, step, report
   :show-inheritance:

Beyond worms: the grand-canonical defect gas
============================================

Even the live-enumerated worm starves, and measurement says why.
On thermalized small-$\kappa$ backgrounds the sheet is so dense that the *median* hypercube admits **no** clean mover of any one- or two-link shape at all --- enriching the candidate family (wider coefficients, farther-flung pairs, millions of shapes) does not help, because the jam is structural: on a background with multi-unit flux everywhere, an exact unit-dipole $\Delta q$ demands cancellations that small templates simply cannot arrange.
At moderate $\kappa$ the movers exist but the action suppresses them, and the worm's all-or-nothing structure compounds the problem: every step must be perfectly clean, so one unlucky draw ends the excursion.
Both failures share a root: the worm insists that the constraint be *exactly* repaired at every single move.

So we stop insisting, and *price the mess instead*.
Enlarge the ensemble with a per-defect fugacity $\zeta$,

.. math ::

   \Pi = \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n]}\, \zeta^{D(n)},
   \qquad
   D(n) = \sum_x \left|q_x(n)\right|,

and sample it with the humblest move there is: a single link and $c = \pm 1$, drawn uniformly, accepted with $\min(1, e^{-\Delta S} \zeta^{\Delta D})$.
The very move that was poison for the constrained model --- a single-link change smears $q$ over its neighborhood --- is now merely *expensive*: whatever charge it scatters is a legal state, discounted by $\zeta^{\Delta D}$, and the reverse move that cleans it up is *rewarded* by the same factor.
Every link is always proposable, so there is no jam and no frozen sector: the frozen configurations that isolate the :class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate` are frozen only for moves that must keep $q = 0$ exactly; the defect gas walks straight through them, paying the toll on the way in and collecting it on the way out.
Ergodicity, which every constrained move set above had to argue for case by case, is manifest.

What does this buy us physically?  Recall :ref:`the worm's shifted constraint <theta worm constraint>`: inserting $e^{i(\theta_x - \theta_y)}$ demands $q = \delta_x - \delta_y$.
The worm's $G$ ensemble is the *two*-defect sector of $\Pi$, carrying weight $\zeta^2$ --- one factor of $\zeta$ per insertion of the charge operator $e^{\pm i\theta}$, which is exactly the sense in which $\Pi$ is the *grand-canonical* worm ensemble: it sums over any number of worms in flight, with fugacity $\zeta$ per endpoint.
The correlator is then read off by pure bookkeeping, no steering required.
Tally, after every proposal, which sector the chain sits in:

.. math ::

   \Theta_{x,y}
   = \frac{\left\langle \prod_p [q_p = \delta_{px} - \delta_{py}] \right\rangle_\Pi}
          {\zeta^2 \left\langle \prod_p [q_p = 0] \right\rangle_\Pi},

the ratio of the time spent in the exact single-pair sector to the time spent in the vacuum, with the known price $\zeta^2$ divided back out.
Two properties are worth internalizing.
First, $\Theta_{x,x} = 1$ *identically* --- a coincident pair *is* the vacuum --- so this estimator is **absolutely normalized**: where the worm histogram must be normalized by its origin bin, the defect gas measures $\Theta$ outright.
Second, $\Theta$ is **independent of** $\zeta$, because the intermediate sectors' weights cancel from the ratio entirely; $\zeta$ tunes only the variance.
Running twice at different $\zeta$ and comparing is therefore a sharp end-to-end exactness test that comes for free.

The fugacity also has a clean meaning in terms of the Lagrange multiplier we integrated out.
Since $q$ is integer-valued,

.. math ::

   \zeta^{\left|q\right|} = \int_{-\pi}^{\pi} \frac{d\theta}{2\pi}\; \frac{1 - \zeta^2}{1 - 2\zeta\cos\theta + \zeta^2}\; e^{i\theta q},

the Poisson kernel: the defect gas is the theory in which the *flat* $\theta$ measure (whose integration produced the hard constraint) is replaced by a Poisson-kernel prior on every hypercube.
$\zeta \to 0$ collapses the kernel to the flat measure's delta function and recovers the hard constraint; $\zeta \to 1$ removes the constraint altogether.

Topologically, the defect gas finishes the story the worm began.
The Freedman--Quinn caveat above was that homotopies between embedded sheets may require *several* double-point pairs in flight at once --- Casson's obstruction --- while the worm carries exactly one.
The defect gas carries **any number**: a finger move is a pair creation (priced $\zeta^2$), a Whitney move is a pair annihilation (rewarded $\zeta^{-2}$), an isotopy is a $\Delta q$-neutral rearrangement (free), and higher-multiplicity double points ($\left|q_x\right| \geq 2$) are ordinary states of the gas rather than special cases needing bespoke repair moves.
It walks the full immersed corridor, not the one-pair shortcut.

The :class:`~supervillain.generator.no_intersection.DefectGas` is deliberately *not* named a worm --- nothing walks, nothing is steered; defects appear, diffuse, and annihilate on their own schedule, and the physics is read off from where the chain happens to sit.
As a :class:`~supervillain.generator.Generator` it nonetheless slots into the machinery above through a simple device: a :meth:`~supervillain.generator.no_intersection.DefectGas.step` advances the gas until a prescribed number of returns to the vacuum sector and emits *that* configuration.
The visits of a reversible chain to a subset of its states form the *trace chain* on that subset, reversible with respect to the restricted measure --- and $\Pi$ restricted to the vacuum sector is exactly $e^{-S}$ on the constraint surface.
So, exactly as for the walking worms, the invalid states live only *inside* a step; every emitted configuration satisfies $q \equiv 0$; and because $\phi$ is frozen within a step the update is $n$-only and composes with a $\phi$-update in a :class:`~supervillain.generator.combining.Sequentially` like every other generator here.
The pair-sector dwell rides along as the inline observables ``Theta_Theta`` and ``Vacuum_Ticks``, from which $\Theta$ is the ratio of ensemble means.

The one knob that must be handled with respect is $\zeta$ itself.
Entropy pushes $D$ up --- each defect may live anywhere, and the denser the sheet the more charge a single link scatters --- so the well-tuned $\zeta$ shrinks with the volume and with $1/\kappa$, and a badly-large $\zeta$ condenses the gas into a defect soup that never revisits the measured sectors (the failure is loud: the step raises rather than hang).
But this tuning pressure is itself a diagnostic, because *physical* defect condensation is precisely $\theta$ long-range order --- pairs costing $O(1)$ at any separation --- the phase the mixed anomaly is hunting for.
A tuned $\zeta$ that must fall like $1/V$, and a pair-sector dwell spreading flat in separation, are that phase announcing itself: signal, not failure.

.. autoclass:: supervillain.generator.no_intersection.DefectGas
   :members:
   :show-inheritance:

.. autoclass:: supervillain.observable.Theta_Theta
   :members:
   :show-inheritance:

.. autoclass:: supervillain.observable.Vacuum_Ticks
   :members:
   :show-inheritance:

.. autoclass:: supervillain.observable.IntersectionSusceptibility
   :members:
   :show-inheritance:

Irreducibility by construction
==============================

The generators above are motivated case by case, and the frozen configurations show how easily a plausible-looking move set can fail to connect the constraint surface.
The :class:`~supervillain.generator.no_intersection.ScattershotUpdate` settles the reachability question wholesale: it proposes a *joint, atomic* change of every link at once, drawn from a symmetric distribution whose support is every integer-valued $\Delta n$, so any valid configuration is proposed from any other in a single step with positive probability and the chain is manifestly irreducible on the constraint surface.
A typical draw touches only a couple of scattered links --- and, unlike a single-link sweep, those few links are accepted or rejected *together*, so it also supplies small coordinated moves of arbitrary geometry --- while the ergodicity guarantee lives in the tail.
What remains empirical is only the *rate* of mixing, which still belongs to the structured moves.

.. autoclass:: supervillain.generator.no_intersection.ScattershotUpdate
   :members:
   :show-inheritance:

The Hammer
==========

We provide the :func:`~supervillain.generator.no_intersection.Hammer` function to :class:`~.Sequentially` combine the various constraint-preserving updates into a single generator.

.. autofunction:: supervillain.generator.no_intersection.Hammer

