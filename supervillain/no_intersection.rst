.. _no_intersection:

*************************
The No-Intersection Model
*************************


Jacobson observed that the standard Villain model supports an interesting modification which suppresses vortex intersection.

.. math ::
   :label: no-intersection

   S = \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + i \sum_h \theta_h (dn \wedge dn)_h

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

The no-intersection constraint asks that the topological-charge density $q_h$
(the same density measured by :class:`~supervillain.observable.TopologicalChargeDensity`)

.. math::

   q_h = (dn \wedge dn)_h = d(n \wedge dn)_h = (dj)_h
   \qquad j = n \wedge dn

vanish on every hypercube.  The second equality is exact on the lattice (the
Leibniz rule and $d^2 = 0$ both hold), so $q$ is the divergence of the 3-form
current $j$ and is locally conserved and integer-valued.  A localized closed
$F = dn$ carries zero total charge $Q = \sum_h q_h$, so violations of the
constraint always come as a $+1$ / $-1$ dipole --- the fact that the
:class:`~supervillain.generator.no_intersection.IntersectionWorm` and the :class:`~.DefectGas` exploits.

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

Just as the Villain :class:`~supervillain.generator.villain.LinkUpdate` has an exact-heatbath
counterpart in the :class:`~supervillain.generator.villain.LinkHeatbath`, so does the
:class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate`.  The single-link charge
change is *exactly linear* in the shift---because the self-wedge $d\delta_\ell \wedge d\delta_\ell \equiv 0$,

.. math ::

   \Delta q = c\,\bigl(F \wedge d\delta_\ell + d\delta_\ell \wedge F\bigr) \equiv c\, L_\ell(F),

and $L_\ell(F)$ reads $F$ only on the planes complementary to $\ell$'s direction, which $n_\ell$
never touches, so it does not depend on $n_\ell$ itself.  Cleanliness ($\Delta q = 0 \Leftrightarrow L_{\ell}(F) = 0$)
is therefore a property of the background alone, *independent of the shift $c$*, and it splits
every link cleanly in two: a **clean** link ($L_\ell(F) = 0$) admits *every* integer shift, so its
exact conditional is the unconstrained Villain discrete Gaussian and we can heatbath it outright;
a **frozen** link ($L_\ell(F) \ne 0$) admits *only* $c = 0$ and is left fixed.  There is no middle
case---no link ever has a bounded, nontrivial legal window---so the 'heatbath' is simply: resample
the clean links from the same discrete Gaussian the :class:`~supervillain.generator.villain.LinkHeatbath`
uses, and skip the frozen ones.

.. autoclass:: supervillain.generator.no_intersection.ConstrainedLinkHeatbath
   :members:

Be clear about what this does and does not buy, since the very same linearity that *makes* the
heatbath exact also *bounds its value*.  Because cleanliness is independent of $c$, the heatbath and
the :class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate` reach the **same set of
configurations**---the reachability graph has identical connected components for any shift magnitude.
So the heatbath is not the ergodic fix: it is trapped by exactly the frozen configurations described
next, and it cannot cross between the topological sectors of the constraint surface any more than the
Metropolis version can.  Its only advantage is that it jumps clean links to a fresh discrete-Gaussian
draw in one hit instead of diffusing them $\pm 1$, a modest mixing gain confined to small $\kappa$
where that conditional (width $\sigma = 1/2\pi\sqrt{\kappa}$) is wide enough for the difference to
matter.  Escaping a frozen configuration, or changing sector, still needs a *coordinated* move.

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
       S[\phi, n, \theta] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + i \sum_h \theta_h (dn \wedge dn)_h
   \end{aligned}

and that we may directly path-integrate out the Lagrange multiplier $\theta$ in favor of the constraint

.. math ::

   Z = \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n]} \prod_h [(dn \wedge dn)_h = 0].

The two-point function of the charge-insertion operator $e^{i\theta}$ 

.. math ::
   :label: defect correlator

   \Theta_{x,y} = \left\langle e^{i(\theta_x - \theta_y)} \right\rangle

is conjugate to the no-intersection constraint and poses a tricky problem to evaluate: if we sample configurations of $Z$ we integrate $\theta$ out first and can no longer see the field needed for the obvious way to compute the observable.
Instead we absorb the insertion into the action *before* path-integrating out $\theta$.
Because $\theta_h$ multiplies $q_h = (dn \wedge dn)_h$, integrating $\theta_x$ against the extra $e^{i\theta_x}$ shifts the constraint at the insertions

.. math ::
   :name: theta worm constraint

   S[\phi, n, \theta] - i(\theta_x - \theta_y)
   \rightarrow
   \Theta_{x,y} = \frac{1}{Z} \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n]} \prod_h [(dn \wedge dn)_h = \delta_{hx} - \delta_{hy}]

where $x$ and $y$ label hypercubes: the insertion demands exactly one unit of topological-charge density at $x$ and a compensating unit at $y$.
Constructing such an overlay by hand, as in the :class:`~supervillain.observable.Spin_Spin`'s taxicab Villain-frame implementation, hits a similar overlap problem: we must lay down a whole sheet of $F = dn$ connecting $y$ to $x$, and unless it threads the valley of the action the change in action is enormous and the correlator is tiny except on rare configurations.
In fact it's much more challenging in this model because the $dn \wedge dn$ constraint is quadratic in $n$ and the overlay must be a sheet of $n$s that satisfies the constraint.

Following Prokof'ev and Svistunov :cite:`PhysRevLett.87.160601` we instead introduce defects where the constraint may be broken, propagate them, and celebrate when they meet.
Consider the mixed regular+path integral $G$ with unspecified normalization $N$ (which cancels from everything of interest)

.. math ::
   :label: intersection G correlator

   G = \frac{1}{N} \sum\hspace{-1.33em}\int D\phi\; Dn\; dh\; dt\; e^{-S[\phi, n]} \prod_p [(dn \wedge dn)_p = \delta_{ph} - \delta_{pt}].

A configuration of $G$ carries a $+1$ unit of $q$ at the head hypercube $h$, a $-1$ unit at the tail hypercube $t$, and $q = 0$ everywhere else.
When the head and tail coincide the constraint is restored everywhere and the $G$ configuration is a valid $Z$ configuration appearing with the right relative frequency.
To insert a worm we drop the head and tail on the same randomly-chosen hypercube; the change in action is 0 and the move is automatically accepted.
Moving the head then means changing $q$ on both the departure and destination cells---restoring the constraint at the former and breaking it at the latter---which, as described above, requires a coordinated three-link change of $n$ that extends the dragged sheet of $F = dn$.
Each such move changes the Villain action and so is Metropolis-tested; when the head returns to the tail we may emit the configuration back into the $Z$ chain.

Notice that the defect correlator :eq:`defect correlator` is equal to

.. math ::
   :name: theta worm histogram

   \Theta_{x,y} = \frac{\left\langle \delta_{xh} \delta_{yt} \right\rangle_G}{\left\langle \delta_{ht} \right\rangle_G}

where the expectation values are over configurations drawn from $G$ (not $Z$!).
If we draw from the larger space of $G$ configurations and histogram the head$-$tail displacement, normalizing that histogram by its value at zero displacement recovers $\Theta_{x,y}$.
We accumulate the histogram as the worm evolves and save it inline with $\phi$ and $n$ as ``IntersectionTwoPoint`` (alongside the ``Worm_Length``), remembering to normalize any :class:`~.DerivedQuantity` built from it by its value at the origin---exactly as for :class:`~.Vortex_Vortex`.

.. autoclass:: supervillain.observable.IntersectionTwoPoint
   :members:
   :show-inheritance:

The enlarged $G$ ensemble also has a precise topological meaning, which is the structural reason to hope a worm mixes where local updates struggle.
Poincaré duality turns the flux $F = dn$ into a closed 2-dimensional *vortex sheet* on the dual lattice, and $q = (dn \wedge dn)$ into the signed density of the sheet's transverse self-intersection points: in 4D two 2-dimensional sheets generically meet at isolated points, and each crossing carries a sign from comparing orientations --- the same $\epsilon^{\mu\nu\rho\sigma}$ contraction that appears in $F \wedge F$.
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
Moreover, the whole ergodicity issue is obviated by the :class:`~.DefectGas` which we develop below.

.. autoclass:: supervillain.generator.no_intersection.IntersectionWorm
   :members:
   :show-inheritance:

The worm accumulates its head$-$tail displacement histogram inline as the
:class:`~.IntersectionTwoPoint` observable.

In the Villain model constraint is linear and therefore we could construct an :class:`~.ExactUpdate` which was in the kernel of the constraint that was essentially a closed 4-plaquette :class:`~supervillain.generator.villain.ClassicWorm`.
Similarly in the Worldline model the :class:`~supervillain.generator.worldline.PlaquetteUpdate` could be understood as the smallest nontrivial worm.
These could be essentially proposed everywhere because they automatically preserve the constraint.
The essential fact was that the constraint is linear.
But here we have a quadratic constraint and therefore it is not always legal to just stamp a tight worm on an existing configuration---it could break the constraint.

In fact, the above worm performs absolutely dismally, and tallying its behavior says why.
The right statistic is the rate at which it draws a *clean mover* at all --- a proposal that
is legal on the background it is standing on.  Averaged over its whole stencil library:

.. list-table::
   :header-rows: 1

   * - $\kappa$
     - clean movers drawn / drawn
   * - 0.01
     - $0$ / $1146$ --- $2$ / $2502$   (:math:`\lesssim 0.08\%`)
   * - 0.03
     - $14$ / $2824$   ($0.50\%$)
   * - 0.1
     - $22$ / $2209$, $27$ / $2360$   ($1.0$--$1.1\%$)

Clean-mover availability falls roughly tenfold as the coupling drops into the jammed phase.
And the *multi-link* stencils are essentially always constraint-violating on nontrivial backgrouns: at
$\kappa = 0.01$, every single ``ortho3``, ``ortho2`` and ``same4`` shape drawn broke the
constraint ($\texttt{unclean}/\texttt{drawn} = 1.0000$) in every run we measured --- not one
clean multi-link move in thousands of draws.  Above the jammed phase the library is only
*nearly* dead ($\texttt{unclean}/\texttt{drawn} = 0.955$--$1.000$ at $\kappa = 0.1$).  What
little transport the worm achieves rides on 1-link moves alone, because when $\kappa$ is big the multi-link moves cost a lot of action and are often rejected.

.. Neither ``Worm_Length`` nor the off-origin weight of the head--tail histogram measures
   this worm's transport.  It tallies head--tail dwell on *every* iteration, including
   rejected stay-puts, and it does not auto-close --- so a single accepted mover pins the
   head away from the tail and every later rejection inflates both numbers.  At
   $\kappa = 0.01$ the off-origin dwell reads $0.000$ on three independent backgrounds and
   $0.213$ on a fourth, the outlier produced by exactly *two* accepted movers out of $2502$
   draws, which then contributed $533$ stalled ticks.  Only the clean-mover draw rate is
   free of dwell.

The census is :source:`example/no-intersection/worm_jam.py`.  Because every generator in the
:func:`~supervillain.generator.no_intersection.Hammer` roster seeds its own RNG, the
thermalized background varies run to run; the numbers above are the observed ranges, and the
jam is stable across every seed we measured.


The adaptive worms, and why they were retired
---------------------------------------------

.. note ::

   The ``AdaptiveIntersectionWorm`` and ``TwoLinkAdaptiveWorm`` described here **no longer
   exist** in the library; they were removed once measurement showed they jam.  Their code
   lives in the history at commit ``45244c3`` and its ancestors.  The
   :class:`~supervillain.generator.no_intersection.FreeTargetWorm` survives, opt-in, as the
   culmination of the idea --- and as the experiment that separates the two ways a worm can
   fail.

The naive worm commits to a dipole stencil *before* looking at the flux it must move
through.  The **adaptive worm** fixed that: it kept the head/tail construction, the
Freedman--Quinn corridor, and the orthogonal 1-, 2-, and 3-link stencils, but at each step
it *enumerated* the clean set $C$ of every move that would advance the head in the drawn
direction on the *current* background $F = dn$, drew one uniformly, and corrected for the
state-dependence of that count with the exact Metropolis--Hastings ratio

.. math ::

   A = \min\left(1,\; \frac{\left|C\right|}{\left|C'\right|}\, e^{-\Delta S}\right),

where $\left|C'\right|$ is the size of the clean set for the reverse move at the
destination.  Because $F$ is a function of the current configuration and not of the chain's
history, this is ordinary state-dependent Metropolis--Hastings.

It still starved.  The **two-link adaptive worm** then *live-enumerated* the two-link
sector --- every pair of nearby links with coefficients up to $\left|\Delta n\right| = 2$, keeping
every pair whose combined charge change is exactly the head dipole --- a strict superset of
the adaptive worm's moves.  It starved too.

Both worms drew a **direction** first, and then asked what could move the head that way.
That is the flaw.  Measured on thermalized backgrounds, as the fraction of
(head, direction, sign) proposals with *no legal move at all*:

.. list-table::
   :header-rows: 1

   * - $\kappa$
     - adaptive
     - two-link
     - free-target
     - free-target median $\left|C\right|$
   * - 0.01
     - **0.977**
     - **0.948**
     - **0.488**
     - **1**
   * - 0.03
     - 0.947
     - 0.848
     - 0.038
     - 450
   * - 0.05
     - 0.869
     - 0.739
     - 0.000
     - 594
   * - 0.1
     - 0.814
     - 0.572
     - 0.008
     - 1208
   * - 0.5
     - 0.497
     - 0.004
     - 0.000
     - 192

Recall that small $\kappa$ is the *warm, dense, jammed* phase; it lies below
$\kappa \approx 0.02$.  The adaptive worm is dead in at least half of its proposals
everywhere we measured, only just dipping under one-half at the largest coupling we
tried ($0.497$ at $\kappa = 0.5$), and in the jammed phase it is dead 98% of the time
with a maximum clean set of **one** move, which often means the worm can take a step
and then step back.  Live-enumerating the two-link sector helps --- but never enough.

Dropping the direction requirement is what unjams the clean set.  The
:class:`~supervillain.generator.no_intersection.FreeTargetWorm` enumerates one
head-anchored family, computes each placement's $\Delta q$ on the current $F$, and lets the
transport go where it wants: $\Delta q \equiv 0$ is an idle, a clean dipole is a mover to
wherever the dipole lands.  Above the jammed phase its clean set explodes --- a median of
$1208$ movers at $\kappa = 0.1$, where the adaptive worm is dead $81\%$ of the time.

.. autoclass:: supervillain.generator.no_intersection.FreeTargetWorm
   :members: step_reference, step, report
   :show-inheritance:

And it fails anyway.  It is squeezed from both sides, and no family engineering escapes the
squeeze:

* **In the jammed phase the jam is structural.**  On a background with multi-unit flux
  everywhere, an exact unit-dipole $\Delta q$ demands cancellations that local templates
  cannot arrange.  At $\kappa = 0.01$ even the free-target family leaves $48.8\%$ of
  hypercubes with no mover at all, and a median clean set of one.  Enrichment halves the
  dead fraction ($97.7\% \to 48.8\%$); it does not rescue the worm.

* **Above the jammed phase the moves exist but are too expensive.**  The free-target clean
  union is $90.2\%$ *idles*, and the worm auto-closes the instant an accepted idle leaves
  the head on the tail.  Of the movers, $99.8\%$ are two-link exact repairs whose cost grows
  with $\kappa$: median $\Delta S = 6.01$ at $\kappa = 0.1$ and $21.06$ at $\kappa = 0.5$,
  where **not one of $4756$ movers had $\Delta S \le 0$**.  The probability of transporting
  at all on a given opening is $4.1 \times 10^{-3}$ at $\kappa = 0.1$ and $2.3 \times 10^{-10}$
  at $\kappa = 0.5$.

The result is that the free-target worm transports **essentially never**, at any coupling we
measured.  Here, unlike for the naive worm, the worm length *is* an honest transport
statistic: the free-target worm auto-closes the moment the head returns to the tail, so a
worm of length $1$ is one that never moved the defect at all.  Across runs, the fraction of
zero-length worms is $0.95$--$1.00$ --- $1.0000$ over $200$ worms at $\kappa = 0.01$ and over
$160$ worms at $\kappa = 0.1$ --- and the median worm length is $1$ everywhere.  Rare long
excursions do occur; when one does it dominates any dwell ratio, which is why the dwell
ratio is not quoted here.

Both failures come from one insistence: that the constraint be *exactly repaired at every
single move*.  In the jammed phase no local exact-repair *family suffices* --- often no
repair exists at all, and where one does it is a lone move, a median clean set of one,
which is no way to transport a defect.  Outside the jammed phase the repairs are plentiful
and too expensive to accept.  Two failures, mirror images of each other, produced by the
same demand.

The grand-canonical defect gas
==============================

Every worm starves, and measurement says exactly where.  Enriching the candidate family
buys real ground --- and buys it where the worm was already losing.  It does not buy enough:
in the jammed phase no local exact-repair family suffices, and outside the jammed phase the
movers that exist are too expensive to accept.  Both failures share a root: the worm insists
that the constraint be *exactly* repaired at every single move.

The worm enlarges the configuration space it samples by adding a defect pair.
In this case, we tell the same joke again and again and it gets funnier: rather than
a fixed number of defects, we allow any number of defects to appear and disappear.
Enlarge the ensemble with a per-defect fugacity $\zeta$,

.. math ::

   \Pi = \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n]}\, \zeta^{D(n)},
   \qquad
   D(n) = \sum_h \left|q_h(n)\right|,

and sample it with a plain single link $n\to n\pm 1$ sampled uniformly and accepted with $\min\left(1, e^{-\Delta S} \zeta^{\Delta D}\right)$.
The very move that was poison for the constrained model --- a single-link change that can populate charge defects in its neighborhood --- is now merely *expensive*: whatever charge it scatters is a legal state, with cost inflated by $\zeta^{\Delta D}$, and the reverse move that cleans it up is *discounted* by the same factor.
Every link is always proposable, so there is no jam and no frozen sector: the frozen configurations that isolate the :class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate` are frozen only for moves that must keep $q = 0$ exactly; the defect gas walks straight through them, paying the toll on the way in and collecting it on the way out.
Ergodicity, which every constrained move set above had to argue for case by case, is manifest.

What does this buy us physically?  Recall :ref:`the worm's shifted constraint <theta worm constraint>`: inserting $e^{i(\theta_x - \theta_y)}$ demands $q = \delta_x - \delta_y$.
The worm's $G$ ensemble is the *two*-defect sector of $\Pi$, carrying weight $\zeta^2$ --- one factor of the fugacity $\zeta$ per insertion of the charge operator $e^{\pm i\theta}$, which is exactly the sense in which $\Pi$ is the *grand-canonical* worm ensemble: it sums over any number of worms in flight, with fugacity $\zeta$ per endpoint.
The correlator is then read off by pure bookkeeping.
Tally, after every proposal, which sector the chain sits in.  The two-point defect correlator :eq:`defect correlator` is the ratio of the time spent in the single-pair sector to the time spent in the vacuum,

.. math ::
   :label: theta-defect-correlator

   \Theta_{x,y}
   = \frac{\left\langle \prod_h [q_h = \delta_{hx} - \delta_{hy}] \right\rangle_\Pi}
          {\zeta^2 \left\langle \prod_h [q_h = 0] \right\rangle_\Pi},

with $[\cdots]$ the Iverson bracket, and the factor of $\zeta^2$ dividing out the fugacity price of the insertion.

.. autoclass:: supervillain.observable.Theta_Theta
   :members:
   :show-inheritance:

.. autoclass:: supervillain.observable.VacuumTicks
   :members:
   :show-inheritance:


Two properties are worth internalizing.
First, $\Theta_{x,x} = 1$ *identically* --- a coincident pair *is* the vacuum --- so this estimator is *absolutely normalized*: where the worm histogram must be normalized by its origin bin, the defect gas measures $\Theta$ outright.
Second, $\Theta$ is *independent of* $\zeta$, because the intermediate sectors' weights cancel from the ratio entirely; $\zeta$ tunes only the variance.
Running twice at different $\zeta$ and comparing is therefore a sharp end-to-end exactness test.

The fugacity also has a clean meaning in terms of the Lagrange multiplier we integrated out.
Since $q$ is integer-valued,

.. math ::

   \zeta^{\left|q\right|} = \int_{-\pi}^{\pi} \frac{d\theta}{2\pi}\; \frac{1 - \zeta^2}{1 - 2\zeta\cos\theta + \zeta^2}\; e^{i\theta q},

the Poisson kernel: the defect gas is the theory in which the *flat* $\theta$ measure (whose integration produced the hard constraint) is replaced by a Poisson-kernel prior on every hypercube.
Taking the fugacity $\zeta \to 0$ collapses the kernel to the flat measure's delta function and recovers the hard constraint; $\zeta \to 1$ removes the constraint altogether.

Topologically, the defect gas finishes the story the worm began.
The Freedman--Quinn caveat above was that homotopies between embedded sheets may require *several* double-point pairs in flight at once --- Casson's obstruction --- while the worm carries exactly one.
The defect gas carries *any number*: a finger move is a pair creation (priced by the fugacity $\zeta^2$), a Whitney move is a pair annihilation (rewarded $\zeta^{-2}$), an isotopy is a $\Delta q$-neutral rearrangement (free in terms of fugacity), and higher-multiplicity double points ($\left|q_h\right| \geq 2$) are ordinary states of the gas rather than special cases needing bespoke repair moves.
It walks the full immersed corridor, not the one-pair shortcut.

The :class:`~supervillain.generator.no_intersection.DefectGas` is *not* a worm --- nothing walks, nothing is steered; defects appear, diffuse, and annihilate on their own schedule, and the physics is read off when the system revisits the vacuum sector, which is distributed exactly according to the expected constraint-satisfying Villain model.
Importantly, a negative defect from one step may annihilate a positive defect created far away in another step rather than the positive defect it was originally created with.
Exactly as for the walking worms, the invalid states live only *inside* a step; every emitted configuration satisfies $q \equiv 0$.

The fugacity $\zeta$ must be handled with care.
Entropy pushes $D$ up --- each defect may live anywhere, and the denser the sheet the more charge a single link scatters --- so the well-tuned fugacity $\zeta$ shrinks with the volume and with $1/\kappa$, and a badly-large fugacity condenses the gas into a defect soup that never revisits the measured sectors: the sampling will hang.
But this tuning pressure is itself a diagnostic, because *physical* defect condensation is precisely $\theta$ long-range order.
A tuned fugacity that must fall like $1/V$, and a pair-sector dwell spreading flat in separation, are that phase announcing itself: signal, not failure.

.. autoclass:: supervillain.generator.no_intersection.DefectGas
   :members:
   :show-inheritance:

Because it can be hard to guess a good fugacity in practice we provide a tuner that finds a good set of run parameters for the :class:`~supervillain.generator.no_intersection.DefectGas`.

.. autoclass:: supervillain.generator.no_intersection.DefectGasFugacityTuner
   :members:

At strongly decaying couplings, however, there may be *no* working fugacity at all.
The geometric price is one knob controlling every sector at once: $\zeta^{D}$ fixes all the adjacent-sector dwell *ratios* simultaneously, while the sectors' entropies --- roughly $\lambda^{k}/k!$ for $k$ pairs free to roam the volume --- grow at different rates.
A $\zeta$ small enough to keep the top sectors from condensing can squeeze the pair sector's dwell below anything measurable, and the window between *condensed* and *silent* can close entirely; at $N = 6$, $\kappa = 0.05$ we found it shut.

The cure is to abandon the geometric form and price each sector on its own,

.. math ::

   \zeta^{D(n)} \longrightarrow w_{D(n)/2},
   \qquad w_{0} = 1,

a *multicanonical* weight table with one entry per pair sector up to the cap $K = $ ``max_defects`` $/2$.
The right target is **flat sector occupancy**: with every sector dwelling equally the vacuum is revisited a fraction $1/(K+1)$ of the time --- a fixed *polynomial* price for full multi-pair traffic, against the $e^{-\langle D/2 \rangle}$ suppression *some* sector must suffer under any geometric pricing.
And, since the intermediate sectors' weights cancel from every published estimator exactly as $\zeta$ did, the table costs nothing in exactness: running twice with materially different tables and comparing is the same free end-to-end test as running at two fugacities.

Flatness has a physical identity worth respecting: the flat table sits at the *sector-coexistence point* of the auxiliary ensemble, the same knife's edge on which the fugacity ladder kept slipping into metastability --- probe a candidate briefly and it looks healthy, run it longer and it condenses.
The :class:`DefectGasWeightTuner` is built around that hazard.
Its probes are *sweep-budgeted* rather than vacuum-anchored, so a condensing candidate produces a lopsided histogram --- a measurement the recursion corrects on the next iteration --- instead of a hang; sectors the probe never visited are left untouched, because extrapolating into unmeasured territory is how multicanonical recursions blow up; per-iteration updates are damped and clipped; and convergence demands visits everywhere, flatness, half-vs-half stationarity, *and* completed vacuum--top--vacuum round trips, never flatness alone.
After the recursion freezes, the table is deliberately *lightened* ($w_{k} \to w_{k}/\texttt{lighten}^{k}$), stepping production off the coexistence point onto its vacuum side: slightly less multi-pair traffic, bought back many times over in stability.

.. autoclass:: supervillain.generator.no_intersection.DefectGasWeightTuner
   :members:

We can build a derived quantity from the defect gas's dwell ratios to compute the intersection susceptibility $\chi_\theta$. The pair-sector dwell rides along as the :class:`~.DefectGas`'s inline observable ``Theta_Theta``, while the defect-free dwell is measured by the inline observable ``VacuumTicks``, from which $\Theta$ :eq:`theta-defect-correlator` is the ratio of ensemble means.


.. autoclass:: supervillain.observable.Intersection_Intersection
   :members:
   :show-inheritance:

The intersection susceptibility is then the sum of the correlator over all separations.

.. autoclass:: supervillain.observable.Intersection_Intersection_Normalized
   :members:
   :show-inheritance:

.. autoclass:: supervillain.observable.IntersectionSusceptibility
   :members:
   :show-inheritance:


Beyond the $\Theta$ correlator, the defect gas can yield quantities with some
simple bookkeeping.  If we are interested in studying a transition that involves
$U(1)_\theta$ symmetry breaking we might be interested in the Binder cumulant,
built from the fourth moment of the order parameter and the square of the second moment.

.. autoclass:: supervillain.observable.IntersectionBinderCumulant
   :members:
   :show-inheritance:

Both moments reduce to sector dwell.  We've already seen how to get the two-point function by tallying the time the defect gas spends in the single-pair sector. 


The second moment is given by the :class:`~.IntersectionSusceptibility` observable,

.. math ::

   \left\langle \left|M\right|^2 \right\rangle = \left\langle \left|\sum_h e^{i\theta_h}\right|^2 \right\rangle = V \chi_\theta,
   \qquad
   \chi_\theta = \sum_{\Delta x} \Theta_{\Delta x}.

Like every susceptibility it obeys the standard trichotomy:
it goes to a *constant* in the thermodynamic limit when the
$\theta$ correlations are short-ranged, grows like $L^{D - 2\Delta_\theta}$ at a
critical point, and grows like $\left|\langle e^{i\theta}\rangle\right|^2 V$
if the defects condense --- so its behavior as the volume gets large reveals the ordering.

The fourth moment 

.. math::
   :label: fourth moment theta

   |M|^4 = \sum_{x_1x_2y_1y_2} e^{i(\theta_{x_1} + \theta_{x_2} - \theta_{y_1} - \theta_{y_2})}

requires two defect pairs, and this is where the :class:`~.DefectGas`'s
grand-canonical formulation really shines.
Despite obviously including 4 insertions, if those insertions coincide at a single hypercube the charge can sum or cancel and we need to account for combinatorial factors and time spent in sectors with less charge.

We can evaluate $\langle \left|M\right|^4 \rangle_Z$ by counting the time the defect gas spends in each sector and dividing out the fugacity $\zeta$ of the insertions.  
But unlike the second moment, there are complications because of the possibility of cancellations and double-counting.
These are akin to the :class:`~.Theta_Theta` observable not explicitly having its origin bin as 1.


.. math::

   \langle |M|^4 \rangle_Z = \frac{\sum_{x_1x_2y_1y_2} \left\langle \prod_h [q_h = \delta_{x_1h} + \delta_{x_2h} - \delta_{y_1h} - \delta_{y_2h}] \right\rangle_\Pi}{\zeta^4 \left\langle \prod_h [q_h = 0] \right\rangle_\Pi},

.. collapse:: We can understand and count the four-point sectors.
   :class: note

   Let's imagine expanding the sum :eq:`fourth moment theta`.
   Depending on the values of $x_1, x_2, y_1, y_2$ we may have some charge cancellation.

   First, if $x_1 = y_1$ and $x_2 = y_2$ we have no net charge.
   But also, if $x_1 = y_2$ and $x_2 = y_1$ we have no net charge.
   Therefore $|M|^4$ contains a $2V^2$ term.
   But this overcounts the case where all four insertions coincide at a single hypercube, so that we should subtract $V$.

   Next let's count the 1-pair sector, where one $+/-$ pair cancels and the
   surviving pair nets $+1$ on one hypercube and $-1$ on another.
   The cancellation can happen by a triple coincidence.  There are
   $V(V-1)$ ways to pick $x_1 \neq x_2 = y_1 = y_2$,
   $V(V-1)$ ways to pick $x_2 \neq x_1 = y_1 = y_2$,
   $V(V-1)$ ways to pick $y_1 \neq x_1 = x_2 = y_2$, and
   $V(V-1)$ ways to pick $y_2 \neq x_1 = x_2 = y_1$.
   But the cancelling pair can also sit together on its own hypercube, away from
   both survivors.  There are
   $V(V-1)(V-2)$ ways to pick $x_1 = y_1$ with $x_2$ and $y_2$ on two further
   distinct hypercubes, and likewise for $x_1 = y_2$, for $x_2 = y_1$, and for
   $x_2 = y_2$: $4V(V-1)(V-2)$ more.
   Altogether $4V(V-1) + 4V(V-1)(V-2) = 4V(V-1)^2$ terms of the sum
   :eq:`fourth moment theta` land in the 1-pair sector: each of the $V(V-1)$
   arrangements of one pair of charges is created $4(V-1)$ ways, and that
   per-arrangement multiplicity is what multiplies the 1-pair dwell below.

   Two-pair patterns are everything left in the sum :eq:`fourth moment theta`.  The
   multiplicity is the number of ordered assignments for a charge placement; a $+2$ forces
   $x_1 = x_2$ and eats a factor of $2$, for example

    .. list-table::
        :header-rows: 1

        * - class
          - net charges
          - ordered assignments
        * - $\{+1, +1, -1, -1\}$
          - four distinct hypercubes
          - $2 \times 2 = 4$
        * - $\{+2, -1, -1\}$
          - $x_1 = x_2$
          - $1 \times 2 = 2$
        * - $\{+1, +1, -2\}$
          - $y_1 = y_2$
          - $2 \times 1 = 2$
        * - $\{+2, -2\}$
          - both $x_1 = x_2$ and $y_1 = y_2$
          - $1 \times 1 = 1$

   As a check, freeze $\theta$ so that every dwell ratio is $1$: the strata must then
   sum to $V^4$ term by term,

   .. math ::

      (2V^2 - V) + 4V(V-1)^2 + V(V-1)(V-2)(V-3) + 2V(V-1)(V-2) + V(V-1) = V^4,

   which gives $16$ at $V = 2$ and $81$ at $V = 3$: every term of
   :eq:`fourth moment theta` is counted exactly once.

The punchline is that we can express the fourth moment as a set of tallied histograms normalized by the fugacity $\zeta$ of the insertions and by the vacuum tally.

.. math ::

   \begin{aligned}
      \langle |M|^4 \rangle_Z
      =&
      (2V^2 - V)
      \nonumber\\
      &+ 4(V-1) \sum_{x\neq y} \frac{\left\langle \prod_h [q_h = \delta_{xh} - \delta_{yh}] \right\rangle_\Pi}{\zeta^2 \left\langle \prod_h [q_h = 0] \right\rangle_\Pi}
      \nonumber\\
      &+ \Bigg(
         4 \sum_{x_1, x_2, y_1, y_2 \text{ all different}}
         + 2 \sum_{x_1=x_2, y_1\neq y_2}
         + 2 \sum_{x_1\neq x_2, y_1=y_2}
         + 1 \sum_{x_1=x_2, y_1=y_2}
         \Bigg)\Bigg[
      \nonumber\\
      &\quad\quad\quad \frac{\left\langle \prod_h [q_h = \delta_{x_1h} + \delta_{x_2h} - \delta_{y_1h} - \delta_{y_2h}] \right\rangle_\Pi}{\zeta^4 \left\langle \prod_h [q_h = 0] \right\rangle_\Pi} \Bigg]
   \end{aligned}

which can be directly translated into 

.. math ::

   \left\langle \left|M\right|^{4} \right\rangle
   = (2V^{2} - V)
   + 4(V-1)\, V\, \frac{\sum_{\Delta x} \left\langle \texttt{Theta\_Theta}_{ \Delta x} \right\rangle}{\left\langle \texttt{VacuumTicks} \right\rangle}
   + \frac{\left\langle \texttt{FourDefects} \right\rangle}{\left\langle \texttt{VacuumTicks} \right\rangle}

The explicit factor of $V$ appears in translation because ``Theta_Theta`` is
translation-averaged --- it carries $1/V\zeta^{2}$ --- so the site sum
$\sum_{x \neq y}$ is $V \sum_{\Delta x \neq 0}$; and because the raw histogram's
origin bin is empty by construction, the unrestricted $\sum_{\Delta x}$ is
automatically the $\Delta x \neq 0$ sum.

.. autoclass:: supervillain.observable.FourDefects
   :members:
   :show-inheritance:


Two practical notes.  All sector prices divide out, so the :class:`~.IntersectionBinderCumulant` is independent of the fugacity or weight table ---
the same free exactness test as for $\Theta$ --- but the *statistics* are not: under geometric pricing the
quartic-sector dwell scales like $\zeta^4$, so the Binder cumulant measurement wants the
largest healthy fugacity (exactly what :meth:`~supervillain.generator.no_intersection.DefectGasFugacityTuner.tune_edge`
selects) or, better, the :class:`DefectGasWeightTuner`'s flat table, which visits the
quartic sector a fixed $1/(K+1)$ fraction of the time no matter how expensive $\zeta^{4}$
would have made it.  An under-visited quartic sector shows up as impossible values ($U > 1/2$
violates Cauchy--Schwarz) with
underestimated :class:`~.Bootstrap` errors --- loud, like every other failure mode of
this sampler.

The pair-separation umbrella
----------------------------

The sector table controls how *often* the pair sector is visited, not where the pair goes once inside: within $D = 2$, visits to separation $r$ arrive in proportion to $\Theta(r)$ itself, so in a decaying phase the far bins are starved by exactly the physics we are trying to measure.
The remedy is a second learned table, an *umbrella* $w_{2}(r)$ over the distinct minimal-image $r^{2}$ shells (translation- and hypercubic-invariant by construction), which multiplies the enlarged-ensemble weight,

.. math ::

   W = 1 \; (D = 0,\ D \geq 6,\ \text{doubled charges}),
   \qquad
   W = w_{2}(\left|x - y\right|) \; (D = 2),

.. math ::

   W = \tfrac{1}{2}\left[
       w_{2}(\left|x_{1} - y_{1}\right|)\, w_{2}(\left|x_{2} - y_{2}\right|)
     + w_{2}(\left|x_{1} - y_{2}\right|)\, w_{2}(\left|x_{2} - y_{1}\right|)
   \right] \; (D = 4,\ \{+1,+1,-1,-1\}),

the $D = 4$ form a Wick-inspired symmetrization: two defects' heads cannot be told apart from their tails, so the weight must be bosonic --- invariant under relabelings --- and averaging the two contractions makes the all-ones table exactly neutral in every sector.
Since any deterministic $W$ is exactness-preserving, the estimators only need their bookkeeping adjusted: ``Theta_Theta`` bins are pure in $r$ and are divided binwise by $w_{2}$ at emission, while the $D = 4$ weight varies *within* the class, so the four-defect tally accumulates $1/W$ per tick.
Independence of the published physics from the $w_{2}$ table --- on $\Theta$ *and* on the Binder cumulant --- generalizes the fugacity- and $w$-independence exactness tests.

The umbrella is learned by :meth:`~.DefectGasWeightTuner.tune_umbrella` as a second stage of the same damped, clipped recursion with the sector table frozen, renormalized each iteration so the total pair-sector weight is unchanged --- $w$ owns sector traffic, $w_{2}$ owns the within-sector profile.
Because unvisited shells are never extrapolated into, steeply decaying couplings want the warm start $w_{2} \sim 1/\hat\Theta$ from a previous run or stored correlator.

.. warning ::

   The umbrella reshapes the *stationary distribution*, not the *kinetics*, and in production the kinetics can be the binding constraint.
   A pair made flat in $r$ random-walks the whole lattice and must diffuse back to adjacency before it can annihilate: flatten the shell dwell aggressively at a steeply decaying coupling and the vacuum round trips die, which is defect condensation by another road.
   (At the susceptibility peak the opposite obtains: the natural far-shell dwell is only a small factor below flat, and the umbrella has little to buy.)
   The defect-adjacent proposal targeting below is that transport move: with it, the umbrella's flattened stationary distribution and the targeted kinetics compose.

Defect-adjacent proposals
-------------------------

The transport bottleneck has a proposal-side cure: instead of proposing links uniformly --- so that a separated pair is touched with probability $\sim 48/4V$ per tick --- the gas can, with probability $1 - \gamma_k$ in sector $k = D/2$, pick a live defect cell with probability $\left|q_c\right|/D$ and one of its 32 edges uniformly.
The proposal density at link $\ell$,

.. math ::

   p(\ell \mid n) = \frac{\gamma_k}{4V}
   + (1 - \gamma_k)\, \frac{s(\ell, q)}{32 D},
   \qquad s(\ell, q) = \sum_{c \ni \ell} \left|q_c\right|,

(the sum over the 8 hypercubes containing $\ell$; $\sum_\ell s = 32 D$ exactly) is asymmetric, so the accept test carries the full Metropolis--Hastings factor $p(\ell \mid n') / p(\ell \mid n)$ --- and then *every estimator is exactly unchanged*, with $\gamma$-independence joining fugacity-, $w$-, and $w_2$-independence in the family of free end-to-end exactness tests.

Be clear about what targeting buys: for a reversible creation/annihilation pair the Hastings factor exactly cancels the proposal concentration, so net creation and annihilation *fluxes* are invariant.
The gain is *in-sector transport* --- both endpoint states carry the adjacency term, the factor is $O(1)$, and the pair's $r$-space diffusion rate rises by $\sim (1 - \gamma)\, V/(8\langle s\rangle)$ --- exactly the kinetics the umbrella needs and the volume scaling that starves the far bins at large $N$.
The :class:`DefectGasWeightTuner` uses the same $\gamma$ in its probes and in the production chain it builds, so all its measurements are self-consistent.

Detecting $\theta$-sector symmetry breaking
-------------------------------------------

The mixed 't Hooft anomaly forbids a trivially gapped phase symmetric under
both $U(1)$s, but it does *not* insist on a $\theta$ condensate: the matching
can be saturated by symmetry breaking (a Goldstone) or by gapless degrees of
freedom (a symmetric gapped TQFT cannot match a continuous-symmetry anomaly
of this type).  The charge-1 correlator
:class:`~.Intersection_Intersection` struggles to distinguish these at
accessible volumes --- a condensate $\left|\langle e^{i\theta}\rangle\right|^2
\sim e^{-2 S_{\text{core}}}$ can hide below any per-bin floor --- and it is
blind to a further possibility: a condensate in a *different charge sector*.

If single defects are bound while pairs condense --- $U(1)_\theta \to
\mathbb{Z}_2$ --- the charge-1 correlator decays exponentially forever while
the charge-2 correlator plateaus.  Its spacetime integral is available from
tallies the gas already keeps: the $\{+2,-2\}$ class of
:class:`~.FourDefectDistribution` is exactly the sector with the quantum
numbers of $e^{2i\theta_x} e^{-2i\theta_y}$.

.. autoclass:: supervillain.observable.DoubleIntersectionSusceptibility
   :members:
   :show-inheritance:

The sharper instrument does not involve defect insertions at all.
$U(1)_\theta$'s conserved current is the integer 3-form $j = n \wedge dn$,
with $dj = q$ *exactly* (lattice Leibniz and $d^{2} = 0$), so on the
constraint surface $j$ is identically divergence-free and its flux $J_\mu$ through
any 3-torus is a topological integer --- computable on every stored
configuration, no enlarged ensemble required.  Fluctuations of this winding
are the $\theta$-sector *stiffness*, the analog of detecting a superfluid by
its helicity modulus rather than its order parameter: nonzero in the
condensed phase, exponentially suppressed in a gapped symmetric phase, and
scale-invariant at a critical point --- separating the anomaly matchings the
two-point function cannot.

Which current, though, needs care, and the distinction is worth stating before
the classes below.  The Chern--Simons form $\mathrm{CS}(n) = n \wedge dn$ is
the natural thing to write --- but it is **not gauge invariant pointwise**:
under $n \to n + dm$ it shifts by the exact form $d(m \wedge dn)$.  A period
over a closed 3-cycle is a sum with no boundary, so discrete Stokes kills that
shift and the fluxes $J_\mu$ survive; but *anything local* built from
$\mathrm{CS}(n)$ --- a correlator at separated points, a structure factor at
nonzero momentum --- measures the gauge the configuration was stored in.  That
is why :func:`~supervillain.observable.chern_simons_form` is a plain function
rather than an observable: it exists to make the fluxes, in exact integer
arithmetic, and nothing else.

The observable current is the pointwise-invariant

.. math ::

   j = \frac{(d\phi - 2\pi n) \wedge dn}{-2\pi},
   \qquad dj = q,

built from the invariant combination $(d\phi - 2\pi n)$ and the invariant
$dn$.  It differs from $\mathrm{CS}(n)$ by the improvement term
$d(\phi\, dn)/2\pi$, so it carries exactly the same periods while being safe to
correlate at any momentum.

Second caution: $J$ is a *topological* integer, which is exactly what makes it
slow.  In the constrained ensemble it moves only when a defect pair winds the
torus, so a chain can report $\langle J^{2}\rangle = 0$ with zero error ---
censored, not measured.  Because $j$ is closed on shell its dual is transverse,
so the $k \to 0$ intercept of its structure factor gives the same stiffness
with no winding move at all.

.. autofunction:: supervillain.observable.chern_simons_form

.. autoclass:: supervillain.observable.IntersectionCurrent
   :members:
   :show-inheritance:

.. autoclass:: supervillain.observable.IntersectionWinding
   :members:
   :show-inheritance:

.. autoclass:: supervillain.observable.IntersectionWindingSquared
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

The surface worm gas
====================

Every sampler above updates $n$, and that choice has a structural consequence: a move that
preserves $q = 0$ exactly can never change the topological winding $J$ of the current
$j = n \wedge dn$, because for any closed change the shift of $j$ is exact and its periods are
untouched.  The $\theta$-sector stiffness of :class:`~supervillain.observable.IntersectionWindingSquared`
is therefore *censored, not measured* by the whole $n$-space roster: the sector only moves when an
open surface's boundary sweeps a non-contractible cycle while charge rides the slack it provides,
and the constraint-preserving moves never open a surface at all.

The :class:`~supervillain.generator.no_intersection.SurfaceWormGas` tells the defect-gas joke one
more time, in the variable where the mechanism lives.  It works in **F-space** --- the integer
2-form $F$ that stands in for $dn$ --- and relaxes *both* physical constraints at once, pricing
them instead of forbidding them:

.. math ::

   \pi_{\text{ext}}(F) \propto e^{-2\pi^{2}\kappa\, C(F)}\; w(D(F))\;
   \eta_{q}^{Q(F)}\; Z_{\text{wind}}(F)\; w_{2}(r),

with $C(F)$ the coexact norm (the exact $\phi$-marginal of the Villain action), $D$ the number of
open-surface cells ($dF \neq 0$), $Q$ the number of intersecting hypercubes ($F \wedge F \neq 0$),
$Z_{\text{wind}}$ the winding-sector sum that makes the sampled marginal the physical one (folded
into the chain unconditionally --- emitted configurations are physical *raw*, and there is no
importance weight to double-count), and $w_{2}$ a pair-separation umbrella on the charge-$\pm1$
sector.  The moves are a uniform (or defect-targeted) plaquette toggle with manifest detailed
balance and an exact coboundary heatbath --- a local move, the 6-plaquette coboundary of one unit
link translated to a random anchor.

Physical configurations are emitted from the **legal vacuum**: $D = 0$, $Q = 0$, *and* all six
2-cycle periods of $F$ zero.  The last condition is easy to forget and load-bearing: on the torus
closed is not exact, a worm that recloses around a non-contractible 2-cycle leaves $F$ with a
nontrivial $H^{2}$ class for which *no* $n$ with $dn = F$ exists, and emission reconstructs
$(n, \phi)$ from $F$ --- an integer staircase primitive, the harmonic winding resampled from its
exact conditional, and $\phi$ drawn from its exact Gaussian conditional.  Because open surfaces and
intersections are ordinary priced states of the gas, the sector-changing excursions the $n$-space
roster cannot make are just traffic here: at $N = 4$ the emitted stream's $J$ random-walks with
transitions on most rows, where the Hammer started cold stays imprisoned at $J = 0$.

.. autoclass:: supervillain.generator.no_intersection.SurfaceWormGas
   :members:
   :show-inheritance:

Tuning the gas
--------------

Three tuners, three objectives; they are not interchangeable, and the two $w(D)$ tuners
deliberately optimize *conflicting* goals.  The
:class:`~supervillain.generator.no_intersection.SectorWeightTuner` flattens $w(D)$ at a fixed,
moderate cap --- the default for $\Theta$-correlator work and pilots, since the correlator's
statistics are bought with closed-shell dwell.  The
:class:`~supervillain.generator.no_intersection.PairUmbrellaTuner` learns the orthogonal
$w_{2}(r)$ table that un-censors the correlator's large-separation bins.  The
:class:`~supervillain.generator.no_intersection.TransportTuner` instead chooses the **cap** by the
$J$-transport physics --- growing it stage by stage and keeping the design with the most
transport-corner dwell subject to a vacuum-return floor --- spending exactly the dwell the
correlator wants in order to buy the deep charged excursions that flip the sector.  Reach for it
only when $J$ flips per unit compute is the figure of merit, at volumes where flips demonstrably
fire.

.. autoclass:: supervillain.generator.no_intersection.SectorWeightTuner
   :members:
   :show-inheritance:

.. autoclass:: supervillain.generator.no_intersection.PairUmbrellaTuner
   :members:
   :show-inheritance:

.. autoclass:: supervillain.generator.no_intersection.TransportTuner
   :members:
   :show-inheritance:

