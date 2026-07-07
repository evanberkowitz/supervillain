:orphan:

.. _no_intersection_ergodicity:

**************************************************
Appendix: Ergodicity and the Topology of 2-Knots
**************************************************

This appendix explains, step by step, why the ergodicity of the
:ref:`no-intersection model's <no_intersection>` constrained Monte Carlo is *not*
held hostage by the theory of knotted surfaces in four dimensions --- even though
knotted surfaces genuinely inhabit the model's configuration space, and even though
one of the natural questions raised by our worm algorithm turns out to be an open
problem in 4-manifold topology.  The argument matters for the correctness of the
library, so every step is either cited to the topology literature or verified by a
script in :source:`example/no-intersection`.  No topology background is assumed
beyond what the steps build up.

Step 0: What could go wrong
===========================

A Metropolis chain samples correctly if each move satisfies detailed balance (proven
generator by generator in the class documentation) *and* the moves connect the whole
configuration space.  Connectivity is the dangerous half.  It can fail two ways:

* **Kinetic traps**: valid configurations that a particular move set cannot enter or
  leave, even though richer moves connect them.  These are *proven to exist* here:
  the frozen configurations of :source:`example/no-intersection/frozen.py` are
  isolated points of the single-link move graph, escaped only by the coordinated
  global moves (demonstrated in :source:`example/no-intersection/unfreeze.py`).
* **True sectors**: configurations separated by a conserved quantity that *no*
  local dynamics can change.  This is the serious worry, and in four dimensions the
  candidate conserved quantity has a name: the *knot type* of the vortex sheet.

This appendix shows that knot type is not a conserved quantity of the model's
dynamics, explains exactly which mathematical results carry that conclusion, and
states honestly which questions remain --- and why they concern *rates*, not
correctness.

Step 1: From integers to sheets
===============================

Ergodicity in $n$ reduces to ergodicity in the field strength $F = dn$: the
:class:`~supervillain.generator.villain.ExactUpdate` and
:class:`~supervillain.generator.villain.CohomologyUpdate` connect all $n$ with the
same $dn$, so what remains is to connect the possible $F$'s.

$F$ has a geometric identity.  Poincaré duality trades the closed integer 2-form
$F$ for a closed 2-dimensional surface on the dual lattice --- the **vortex sheet**
(the worldsheet that the model's vortex loops sweep out in time: slice the lattice
at fixed $x_0$ and the sheet's cross-sections are ordinary vortex loops in three
dimensions).  In this language

.. math::

   q_x = (dn \wedge dn)_x = \text{signed density of transverse self-intersections
   of the sheet,}

because in four dimensions two 2-dimensional surfaces generically meet at isolated
*points*, each carrying a sign ($2 + 2 = 4$; the sign is the
$\epsilon^{\mu\nu\rho\sigma}$ comparison of the two tangent planes --- the same
contraction that defines $F \wedge F$).  So the constraint surface has a crisp
description:

  **a valid configuration is an embedded sheet** --- a surface with no
  self-intersections at all.

One structural fact we use repeatedly: because $F = dn$ is exact, the sheet is
null-homologous ($[F] = 0$), for every configuration.  Its self-intersection number
$Q = \sum_x q_x = [F] \cdot [F]$ therefore vanishes identically, which is why
constraint violations only ever appear in $+/-$ pairs.

Step 2: Four dimensions is where surfaces knot
==============================================

In three dimensions closed *curves* knot.  In four dimensions curves always unknot
--- but closed *surfaces* take over the job.  Artin constructed knotted 2-spheres in
1925 by *spinning* classical knots :cite:`Artin1925`, and the subject has been rich
ever since (Zeeman's twist-spinning :cite:`Zeeman1965` produces knotted spheres that
are provably not of Artin's type :cite:`Cochran1983`).  "Knotted" means exactly what
a physicist expects: the sphere is embedded, but no deformation through embedded
surfaces relaxes it to a standard round sphere.  (For surfaces of any genus,
"unknotted" means the surface bounds a solid handlebody.)

This is not hypothetical for us.  The scripts
:source:`example/no-intersection/torus_knotted.py` and
:source:`example/no-intersection/spun_sphere.py` build *exactly valid* lattice
configurations whose sheets are knotted tori and knotted (spun) 2-spheres, and
:source:`example/no-intersection/alexander.py` certifies the knotting by exact
computation: the equatorial cross-section of the lattice spun trefoil measures
Alexander polynomial $(t^2 - t + 1)^2$ --- the square knot, precisely as the
continuum construction demands.

So the worry of Step 0 is concrete: if knot type were conserved by our moves, the
knotted and unknotted sectors would never mix, expectation values would depend on
the initial configuration, and the algorithm would be wrong.

Step 3: Genus is not conserved --- and that is the key
======================================================

The escape hatch is that our ensemble is *not* an ensemble of spheres.  The sheet
of a valid configuration may have any genus (and any number of components), and ---
this is the crucial, machine-verified fact --- **legal local moves change the
genus**: :source:`example/no-intersection/genus.py` exhibits a single-link move,
exactly of the kind :class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate`
proposes, that turns a spherical sheet into a torus (and another that merges two
components).  Attaching such a trivial handle is called a **stabilization** in the
topology literature; removing one is a destabilization.  Genus fluctuates in
equilibrium like any other unprotected quantity.

Why does that dissolve knotting?  Because of a beautiful pair of theorems:

* **Hosokawa--Kawauchi** :cite:`HosokawaKawauchi`: every knotted closed orientable
  surface in $S^4$ becomes *unknotted* after finitely many stabilizations.  The
  intuition is worth internalizing: the surface bounds a 3-dimensional "Seifert
  solid"; stabilizing along the solid's 1-handles leaves a surface bounding a
  handlebody --- unknotted by definition.
* **Baykur--Sunukjian** :cite:`BaykurSunukjian`: in any compact orientable
  4-manifold (our $T^4$ included), any two embedded surfaces that are *homologous*
  become smoothly *isotopic* after finitely many stabilizations of each.  Step 1
  showed all our sheets are null-homologous, so the theorem applies to every pair
  of valid configurations.

Both theorems have a property that will be decisive in Step 6: **the entire path
stays embedded.**  Stabilize (a local move), deform (local moves), destabilize
(a local move) --- at every intermediate moment the configuration is an ordinary
valid state of our ensemble, just with a few units more genus.  In the continuum,
the valid set is *connected* under moves our generators implement in principle.
Knot type is not a conserved quantity; it is a slow coordinate.

Step 4: The worm's corridor --- immersions
==========================================

The :class:`~supervillain.generator.no_intersection.IntersectionWorm` samples a
second, independent corridor.  Its enlarged $G$ ensemble --- constraint satisfied
everywhere except a $+1$ at the head and $-1$ at the tail --- is precisely the
space of *immersed* sheets with one pair of opposite-sign double points, and the
classical toolkit of 4-manifold topology (Whitney's disk construction
:cite:`Whitney1944`, Casson's finger moves :cite:`Casson`, assembled in chapter 1
of Freedman--Quinn :cite:`FreedmanQuinn`) says that homotopic embedded surfaces are
connected through exactly this space: ambient isotopies, plus **finger moves** (pair
creation --- the worm opening and taking its first step), plus **Whitney moves**
(pair annihilation --- the worm closing).  The worm's own documentation spells out
the dictionary move by move.

The worm, however, carries exactly **one** pair at a time.  Does that suffice?

Step 5: Length versus width --- and an open problem
===================================================

This question has been sharpened beautifully in recent literature, and it pays to
state it carefully.

1. Any two embedded 2-spheres in $S^4$ are *regularly homotopic* (Smale
   :cite:`Smale1958`, Hirsch :cite:`Hirsch1959`): connected through immersions,
   with double points appearing and disappearing only in finger/Whitney pairs.
2. Joseph--Klug--Ruppik--Schwartz :cite:`JosephKlugRuppikSchwartz` define the
   **Casson--Whitney number** $u_{cw}(K)$: the minimal *total* number of
   finger/Whitney pairs in a regular homotopy from the 2-knot $K$ to the unknot.
   It is finite for every 2-knot.  Call this the **length** of the unknotting.
3. But beware the normal form: a regular homotopy can always be rearranged so that
   *all the finger moves happen first --- simultaneously --- followed by all the
   Whitney moves* (a folklore fact; see Quinn :cite:`Quinn1986`, section 4.1).  So
   "$u_{cw}(K) = k$" is, as stated, a statement about an immersion carrying $k$
   pairs *at once*.
4. The quantity the single worm cares about is different: the **width** --- the
   maximum number of pairs alive *simultaneously*, minimized over all unknotting
   homotopies.  This is Singh's invariant $d_{sing}$ :cite:`Singh2020`.  A 2-knot
   has $d_{sing} = 1$ exactly when it can be unknotted by repeated *single*-pair
   excursions: finger, wander, Whitney, embedded again --- one worm at a time.
5. **Whether $d_{sing}$ ever exceeds 1 is an open problem** --- posed explicitly as
   Question 7.3 of :cite:`JosephKlugRuppikSchwartz`.  No 2-knot is known to require
   two fingers at once, and no theorem guarantees one at a time always suffices.
   The reason the question is open is instructive: every known lower-bound
   technique (fundamental-group quotients, Fox colorings, the Nakanishi index of
   the Alexander module) counts *total* relations --- length --- and is blind to
   width.
6. Width 1 is *proven* for large families, because $u_{cw} = 1$ there: spun knots
   of unknotting-number-one knots, and every nontrivial twist-spin of a 2-bridge
   knot (:cite:`JosephKlugRuppikSchwartz`, Theorem 4.7) --- including all
   twist-spun trefoils, which by Cochran :cite:`Cochran1983` are not even ribbon.
   Connected sums of width-1 knots unknot one pair at a time, factor by factor,
   even though their length $u_{cw}$ grows.

So there is a genuinely open mathematical question sitting next to our worm: *can
every knotted sheet be undone one worm excursion at a time?*

Step 6: Why the open problem does not threaten correctness
==========================================================

Here is the punchline, in small steps.  Suppose the worst: some 2-knot $K$ is
discovered with $d_{sing}(K) \geq 2$, so no sequence of single-pair worm excursions
traverses the immersed corridor from $K$ to the unknot.

1. The **embedded corridor of Step 3 is still open**.  Indeed
   :cite:`JosephKlugRuppikSchwartz` (Theorem 1.1) proves the stabilization number
   obeys $u_{st}(K) \leq u_{cw}(K) + 1$: finitely many handles *always* suffice,
   and the embedded detour costs at most one handle more than the immersed
   shortcut would have.
2. Stabilization has a width invariant too ($d_{st}$, also not known to exceed 1
   :cite:`Singh2020`), but --- and this is the decisive structural point --- **for
   our Monte Carlo, stabilization width is not a constraint at all.**  A worm must
   hold its double-point pairs *in suspension*: they are defects of the enlarged
   $G$ ensemble, and holding $k$ of them requires a $k$-pair worm algorithm.  Extra
   handles are not like that.  A sheet with $m$ extra handles is an *ordinary valid
   configuration* of genus $g + m$ --- a state, not a defect.  The ensemble
   "holds" it for free, and the machine-verified genus moves of Step 3 walk in and
   out of it one handle at a time.
3. Therefore a $d_{sing} \geq 2$ discovery would close a *shortcut*: certain
   passages would no longer be achievable by worm excursions alone and the dynamics
   would take the handle detour, paying whatever action cost the higher-genus
   intermediates carry.  That is a statement about **mixing rates**.  It cannot
   disconnect the valid set.

The multiworm (a $k$-pair extended ensemble) is therefore *rate insurance* --- a
way to keep the immersed shortcut open if the open problem resolves badly --- and
not a correctness requirement.

Step 7: The honest ledger
=========================

What, then, does the correctness of the constrained Monte Carlo actually rest on?

* **Detailed balance**: proven per generator (see each class's documentation).
* **Fiber coverage** ($n$ at fixed $F$): exactness arguments for
  :class:`~supervillain.generator.villain.ExactUpdate` and
  :class:`~supervillain.generator.villain.CohomologyUpdate`.
* **Continuum connectivity of the valid set**: the stabilization theorems of Step 3
  :cite:`HosokawaKawauchi,BaykurSunukjian`, applicable because every sheet is
  null-homologous.  Knot type is not conserved; the (open) width question of Step 5
  affects only how efficiently the worm shortcuts the handle detour.
* **Lattice realizability** of the continuum paths by *this finite move library* on
  *every* background: this is the part that is genuinely empirical, and it is where
  our effort goes.  The frozen configurations prove that kinetic traps are real and
  that the global moves (:class:`~supervillain.generator.no_intersection.WrappingLoopUpdate`,
  :class:`~supervillain.generator.no_intersection.PlanarFluxUpdate`) are necessary;
  :source:`example/no-intersection/torus_dismantle.py` gives an *executed
  certificate* that the existing generators unknot the knotted-torus family;
  :source:`example/no-intersection/corridors.py` and
  :source:`example/no-intersection/exotic.py` census the local move structure on
  adversarial backgrounds.  Two exhaustive verdicts close the question for every
  background we know how to construct.  First,
  :source:`example/no-intersection/isolated.py` sweeps the *complete* known trap
  families (all three single-pair plane pairs with all nonzero coefficients in
  $[-3,3]^2$, and all 2,312 valid six-plane members with $\mathrm{Pf}(A) = 0$ in
  $[-2,2]^6$): of 2,456 valid configurations, 1,948 are frozen and **every one has
  legal global escapes** --- no "super-frozen" configuration exists in the families,
  and the script decides any future suspect.  Second,
  :source:`example/no-intersection/corridor_bfs.py` takes the dangerous
  continuum-adjacent transitions --- the coordinated clean moves provably not
  decomposable into unit steps --- and verifies that **all 86 of them** found on the
  knotted backgrounds are realized by two-step worm excursions from the committed
  library, every stage checked exactly.  What remains unenumerable ("every valid
  configuration") is guarded by the standing decision procedures and, wholesale, by
  the :class:`~supervillain.generator.no_intersection.ScattershotUpdate` below.
* **A caveat to keep in view**: the lattice $q$ is a cup-product density --- a
  cousin of the geometric self-intersection count, not the thing itself --- and
  lattice sheets may carry multiplicity or junction lines that the smooth theorems
  do not speak about.  Those configurations *enlarge* the state space (extra
  corridors, if anything), but they are the reason the continuum theorems guide
  expectations rather than substitute for lattice verification.
* **The wholesale answer**: independent of all topology, the
  :class:`~supervillain.generator.no_intersection.ScattershotUpdate` makes the full
  chain irreducible *by construction* --- its proposal distribution has full
  support --- converting every question above into a question about rates.  We keep
  the topological analysis anyway, because rates are physics: the slow coordinates
  identified here (knot type, entanglement of sheet components, wrapping sectors)
  are exactly where autocorrelations will live.

Summary
=======

Valid configurations are embedded vortex sheets; sheets in four dimensions can
knot; but genus is not conserved by our legal moves (machine-verified), and by
Hosokawa--Kawauchi and Baykur--Sunukjian finitely many genus fluctuations undo any
knot through *embedded, valid* intermediates.  The worm adds an immersed shortcut
whose one-pair-at-a-time sufficiency is equivalent to an open problem in 4-manifold
topology (is Singh's $d_{sing}$ ever $> 1$?) --- but because extra handles are
ordinary states while worm defects are algorithmic burdens, that open problem can
only tax the mixing rate, never the correctness.  What remains to verify is not
topology but lattice kinetics: that this finite library realizes the continuum
corridors on every background --- the census and certificate programs of the
example scripts.
