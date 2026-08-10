

********
Sampling
********

:class:`~.Ensemble`\ s can be :py:meth:`generated <supervillain.ensemble.Ensemble.generate>` using a *Generator*.
A generator is any object which has a class with a :meth:`~.Generator.step` method that takes a configuration as a dictionary and returns a new configuration as a dictionary.
Some generators can measure observables as the step is taken and return them inside the step, in which case the :class:`~.Configurations` need a place to store them, which is created with the :meth:`~.Generator.inline_observables` method.

.. autoclass :: supervillain.generator.Generator
   :members:

For example, a very dumb generator is

.. literalinclude:: generator/example.py
   :pyobject: DoNothing

Then, if you

.. code:: python

   L = supervillain.Lattice2D(5)
   S = supervillain.Villain(L, kappa=0.1)
   e = supervillain.Ensemble(S).generate(17, DoNothing(), start='cold')

the result will be an ensemble ``e`` that has 17 identical configurations (because ``DoNothing`` makes no updates).
Obviously, for good sampling, DoNothing is the worst imaginable algorithm!
It has an infinitely bad ergodicity problem!

-----------------------
The Villain Formulation
-----------------------

A less dumb algorithm is a local update, which changes only fields in some small area of the lattice.

As an example, we can formulate an update scheme offering localized changes to the $\phi$ and $n$ fields in the :class:`~.Villain` formulation.

Picking a site $x$ at random and proposing a change 

.. math ::
    
    \begin{aligned}
    \Delta\phi_x    &\sim \text{uniform}(-\texttt{interval\_phi}, +\texttt{interval\_phi})
    \\
    \Delta n_\ell   &\sim [-\texttt{interval\_n}, +\texttt{interval\_n}]
    \end{aligned}

for the $\phi$ on $x$ and $n$ on links $\ell$ which touch $x$ is ergodic (once swept over the lattice) and satisfies detailed balance so long as we accept the proposal based on the change of action.
The :class:`NeighborhoodUpdateSlow <supervillain.generator.reference_implementation.villain.NeighborhoodUpdateSlow>` generator implements this update algorithm but suffers from a variety of defects.

First, its *implementation* makes a lot of calls.
We could 'unfactor' the proposal, single-site accept/reject, and sweep of the lattice for some benefits in speed.
Moreover, to compute the change in action it evaluates the action for both configurations and takes the difference.
But, we know that's silly, most of the $\phi$s and $n$s haven't changed at all and those links will contribute to both actions equally, giving zero difference.
We could reduce the arithmetic substantially by computing the difference directly.
Finally, for ease of thinking, each :func:`proposal <supervillain.generator.villain.NeighborhoodUpdateSlow.proposal()>` reckons locations relative the origin and therefore moves all the fields around in order to update ergodically.
All of that data movement adds cost, especially as the lattice gets large.

It was easy to write the :class:`NeighborhoodUpdateSlow <supervillain.generator.villain.NeighborhoodUpdateSlow>` but we can do better for production.

.. autoclass :: supervillain.generator.villain.NeighborhoodUpdate
   :members:

One reason the NeighborhoodUpdate suffers is that offering updates to 5 fields at onces ($\phi$ and the 4 ns that touch it) means that the change in action can be large.
We can decouple these proposals.

.. autoclass :: supervillain.generator.villain.SiteUpdate
   :members:

.. autoclass :: supervillain.generator.villain.LinkUpdate
   :members:

When $W=1$ the combination of the :class:`~.SiteUpdate` and :class:`~.LinkUpdate` are ergodic.
But when $W>1$ the :class:`~.LinkUpdate` only offers changes to $n$ by multiples of $W$ to preserve the constraint $dn = 0 \text{ mod }W$.
For an ergodic algorithm when $W>1$ we need to offer ways to change $n$ by 1 (less than $W$) while maintaining the constraint.
We need to make closed updates to $n$, which can be broken up into :class:`~.villain.ExactUpdate`\ s (which are automatically closed) and :class:`~.villain.CohomologyUpdate`\ s (which change the global winding sector).

.. autoclass :: supervillain.generator.villain.ExactUpdate
   :members:

.. autoclass :: supervillain.generator.villain.CohomologyUpdate
   :members:

The combination of the :class:`~.SiteUpdate`, :class:`~.LinkUpdate`, :class:`~.ExactUpdate`, and :class:`~.CohomologyUpdate` is ergodic even when $W>1$.
But it can be slow to decorrelate.
As mentioned, the :class:`~.CohomologyUpdate` often rejects because it touches a macroscopic number of variables ($N^{D-1}$ links per direction).
A major issue is that the route across the torus is very rigid: it's just a straight shot.
Smarter *worm algorithms* can make high-acceptance updates to many variables across the lattice, which can help overcome *critical slowing down*.

Remember that in the :class:`~.Action.Villain` case we are trying to sample according to

.. math ::

   \begin{aligned}
       Z &= \sum\hspace{-1.33em}\int D\phi\; Dn\; Dv\; e^{-S[\phi, n, v]}
       \\
       S[\phi, n, v] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + 2\pi i \sum_p v_p (dn)_p / W
   \end{aligned}

and that we may directly path-integrate out the Lagrange multiplier $v$ in favor of a constraint

.. math ::

   Z = \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n, v]} \prod_p [dn_p \equiv 0 \text{ mod }{W}]

One observable of interest is the :class:`~.Vortex_Vortex` correlation function,

.. math ::

   V_{x,y} = \left\langle e^{2\pi i (v_x - v_y) / W} \right\rangle

which poses a tricky problem to evaluate, since if we sample configurations of $Z$ we integrate $v$ out first and cannot easily compute the observable.
Instead, we can think of adding an insertion of the vortex creation and annihilation operators, and we can absorb them into the action *before* path-integrating out $v$ and use the insertions to shift the constraint at the insertions

.. math ::
   :name: worm constraint

   S[\phi, n, v] - 2\pi i (v_x - v_y) / W
   \rightarrow
   V_{x,y} = \frac{1}{Z} \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n, v]} \prod_p [dn_p \equiv \delta_{px} - \delta_{py} \text{ mod }{W}]

where $x$ and $y$ label plaquettes.
The built-in :class:`~.Vortex_Vortex` observable measures this by picking a path between $x$ and $y$ on which to change $n$ to satisfy the new shifted constraint.
Depending on the parameters this inevitably hits an overlap problem, where we must change so many links between $x$ and $y$ that the change in action is very big and the correlator is naturally very small, except on rare configurations where it is big.

To address overlap problems of this kind Prokof'ev and Svistunov invented worm algorithms :cite:`PhysRevLett.87.160601` which operate by introducing defects
where constraints may be broken, propagating those defects, and when they meet and annihilate the configuration again obeys the constraint.
Consider the mixed regular+path integral $G$ with unspecified normalization $N$ (that will cancel from all interesting quatities that follow) that integrates over all sectors of possible constraints

.. math ::

   G = \frac{1}{N} \sum\hspace{-1.33em}\int D\phi\; Dn\; dh\; dt\; e^{-S[\phi, n, v]} \prod_p [dn_p \equiv \delta_{ph} - \delta_{pt} \text{ mod }{W}].

A configuration of $G$ consists of two plaquettes $h$ and $t$, $\phi\in\mathbb{R}$ on sites and $n\in\mathbb{Z}$ on links which satisfy the $(h, t)$ constraint.

Worm algorithms sample from the larger space of configurations of $G$ and, when the *head* $h$ and *tail* $t$ coincide, each $G$ configuration is a valid $Z$ configuration appearing with the right relative frequency.
This is very different from thinking that we need to carefully design updates to always maintain the constraint.
No!
Violate the constraint as part of the evolution!

If we want to use a worm as an update in our Markov chain, we need to emit configurations that satisfy the original :ref:`winding constraint <winding constraint>`.
We can imagine constructing a Markov chain to sample configurations of both $Z$ and $G$.
To go from a $Z$ configuration insert the head and tail on the same randomly-chosen location (in this case, a plaquette); the change in action is 0 and this is automatically accepted.
This configuration now 'has a worm' even though the fields have not changed.

Next we evolve the configuration in the larger set of all $G$ configurations which need not satisfy the original :ref:`winding constraint <winding constraint>` but instead violates it in exactly two places (the head and the tail), in opposite ways.
Starting from a 'diagonal' configuration which has $h=t$, we can move the head by one plaquette.
As the head moves it changes the $n$ on the link it crosses so that it changes $dn$ on *both* adjacent plaquettes.
With a clever choice we can ensure that as the head leaves a plaquette it restores the constraint there and breaks it on the destination plaquette.
In this way the head moves around the lattice.
This change of $n$ changes the action $S$, and so the moves need to be Metropolis-tested in some sense.

Finally, when the head reaches the tail, the constraint is restored everywhere.
We might allow the worm to keep evolving, but since it satisfies the :ref:`winding constraint <winding constraint>` it is possible to reach a $Z$ configuration too.
To go from a $G$ configuration to a $Z$ configuration we require the head and tail to coincide and satisfy the winding constraint; the change in action is 0 and the acceptance is likewise automatically accepted if this change is proposed.
Now we have a $Z$ configuration and add it to our Markov chain (or update it with different generators first).
Inside the worm algorithm we need not provide a direct route from any $Z$ configuration to any other; the only way between $Z$ configurations is through $G$, though nothing prevents us from using other generators to go directly from $Z$ configuration to $Z$ configuration.

Notice that

.. math ::
   :name: worm histogram

   V_{x,y} = \frac{\left\langle \delta_{xh} \delta_{yt} \right\rangle_G}{\left\langle \delta_{ht} \right\rangle_G}

where the expectation values are with respect to configurations drawn from $G$ (not $Z$!).
Let us understand this expression for the :ref:`vortex correlator <worm histogram>`.
It says that if we draw from the larger space of $G$ configurations and make a histogram in $x$ and $y$, we can normalize that histogram by its value at zero displacement to get the $V_{x,y}$.
We can measure the histogram as we go and report with the $\phi$ and $n$.
The reason to save the histogram as a sample rather than to just accumulate a histogram for the whole worm's evolution is that once we reach a $Z$ configuration we update the other variables; that histogram is conditional on those variables; when they change the histogram will too.
So, the worm's displacement histogram can be saved inline as :class:`~.Vortex_Vortex`, as long as we remember to normalize any :class:`~.DerivedQuantity` that depends on it by the element of the expectation at the origin.

.. autoclass :: supervillain.generator.villain.worm.ClassicWorm
   :members:

The worm is not ergodic on its own---it doesn't update $\phi$, for example, and it cannot change a link by ±W.
But, in combination of with :class:`~.villain.SiteUpdate` and :class:`~.villain.LinkUpdate` it is ergodic;
the worm can replace the combination of :class:`~.villain.ExactUpdate` and :class:`~.villain.CohomologyUpdate`.
The :class:`~.villain.ExactUpdate` can be understood as a very simple worm that takes the tightest nontrivial path,
while the :class:`~.villain.CohomologyUpdate` can be understood as a worm that goes once around the world in a straight shot.
The worm offers *dynamically determined constraint-preserving updates* and is much more flexible.
In can change the holonomy, for example, by finding a route around the torus that isn't a straight shot but
runs through the valley of the action.

Finally, we provide a convenience function which provides an ergodic generator.

.. autofunction :: supervillain.generator.villain.Hammer

-------------------------
The Worldline Formulation
-------------------------

In the :class:`~.Worldline` formulation the constraint $\delta m = 0$ restricts which kinds of updates we could propose.
For example, changing only a single link is *guaranteed* to break the constraint on both ends.
So, we need clever generators to maintain the constraint.

.. autoclass :: supervillain.generator.worldline.PlaquetteUpdate
   :members:

We can decouple the proposals in the :class:`~.worldline.PlaquetteUpdate`, and update the vortex fields and the worldlines independently.

.. autoclass :: supervillain.generator.worldline.VortexUpdate
   :members:

.. autoclass :: supervillain.generator.worldline.CoexactUpdate
   :members:

To have a fully ergodic algorithm we will also need to update the :class:`~.TorusWrapping` of the worldlines.

.. autoclass :: supervillain.generator.worldline.WrappingUpdate
   :members:

The combination of the :class:`~.worldline.VortexUpdate`, :class:`~.CoexactUpdate`, and :class:`~.WrappingUpdate` are ergodic.
However, the may be suboptimal.  In particular the :class:`~.worldline.WrappingUpdate` touches a large number of links which often leads to very large changes in action
and rejection.  Just as in the Villain case we can make smarter updates to a dynamically-determined set of variables with high acceptance by using a *worm algorithm*.

Unlike the Villain formulation, the Worldline formulation has a constraint even when :math:`W=1`, :math:`\delta m = 0` everywhere, from path-integrating $\phi$

.. math ::
   \begin{aligned}
   Z &=  (2\pi\kappa)^{-|\ell|/2}\sum\hspace{-1.33em}\int D\phi\; Dm\; Dv\; e^{-S[\phi, m, v]}
   \\
   S[\phi, m, v] &= \frac{1}{2\kappa} \sum_\ell \left(m - \frac{\delta v}{W} \right)_\ell^2 - i \sum_x \left(\delta m\right)_x \phi_x
   \end{aligned}

Thinking about the :class:`~.Spin_Spin` correlation function, we want to insert $e^{i(\phi_x-\phi_y)}$, which shifts the constraint to

.. math ::

    (\delta m)_s = \delta_{sx} - \delta_{sy}

and we can perform updates in the mixed regular+path integral $G$ that integrates over all sectors of possible constraints

.. math ::

   G = \frac{1}{N} (2\pi\kappa)^{-|\ell|/2} \sum\hspace{-1.33em}\int Dm\; Dv\; dh\; dt\; e^{-S[m, v]} \prod_s [\delta m = \delta_{sh} - \delta_{st}]

As in the Villain case, when the head $h$ and tail $t$ coincide a configuration of $G$ is a valid $Z$ configuration and each appears with the correct relative frequency.
Again, the philosophy is to evolve in a larger space with the constraint lifted and celebrate when we receive a constraint-satisfying configuration.

Just like the Villain case we can measure a two-point correlator inline, but in this case the constraint is $\delta m = 0$ everywhere and constraint-violating insertions are of $e^{\pm i \phi}$ (as in :meth:`Spin_Spin <supervillain.observable.Spin_Spin.Worldline>`),

.. math ::

   S_{x,y} = \frac{\left\langle \delta_{xh} \delta_{yt} \right\rangle}{\left\langle \delta_{ht} \right\rangle}

which amounts to constructing a normalized histogram after ensemble averaging.

.. autoclass :: supervillain.generator.worldline.worm.ClassicWorm
   :members:

Finally, we provide a convenience function which provides an ergodic generator.

.. autofunction :: supervillain.generator.worldline.Hammer

------------------
Lattice Symmetries
------------------

Every formulation lives on the same periodic hypercubic lattice, and that lattice has :ref:`a symmetry group <space-group>` of its own --- translations, axis reflections, and axis permutations --- independent of $\phi$, $n$, $m$, or any other field content.
Drawing an element of that group and applying it to every field is therefore a generator that works for any formulation.
It carries a configuration to a physically identical one, applying :func:`~supervillain.lattice.translate`, :func:`~supervillain.lattice.reflect`, and :func:`~supervillain.lattice.permute` to each :class:`~supervillain.lattice.Form` the action declares.

These are exact symmetries rather than proposals to be tested.
$\Delta S = 0$ identically, so they are always accepted.
Since nothing in the sampling statistics would register a symmetry applied wrongly, an action must declare which symmetries it admits; one is never assumed on its behalf.

Which elements are admissible depends on what the Boltzmann weight is built from, because the lattice operators are not all equally well behaved.
Writing $R$ for the action of a group element on forms, an operator $\mathcal{O}$ is *equivariant* under that element when it commutes with $R$ up to whatever sign its continuum counterpart carries,

.. math ::

    R\left(\mathcal{O}\omega\right) = \pm\, \mathcal{O}\left(R\omega\right).

The table gives that sign, and records which factors of the space group each operator is equivariant under.

.. list-table::
   :header-rows: 1

   * - operator
     - sign
     - translations
     - permutations
     - sign flips
   * - :func:`~supervillain.lattice.d`, :func:`~supervillain.lattice.delta`, :func:`~supervillain.lattice.laplacian`
     - $+1$
     - yes
     - yes
     - yes
   * - :func:`~supervillain.lattice.star`
     - $\det R$
     - yes
     - yes
     - no
   * - :func:`~supervillain.lattice.wedge`, $a \wedge b$
     - $+1$
     - yes
     - yes
     - no
   * - :func:`~supervillain.lattice.wedge`, $a \wedge a$
     - $+1$
     - yes
     - yes
     - the full inversion only

$d$ is the coboundary of the cell complex --- pure combinatorics of which cell borders which --- so it commutes with any relabelling of cells, and $\delta$ is its adjoint  with respect to the cell-wise inner product :eq:`d-delta-formal-adjoint`, which the space group leaves invariant.
The wedge and the star instead carry base-point shifts tied to the *sense* of a direction, and a sign flip reverses that sense.
Note that orientation is not what matters: odd permutations are orientation-reversing and both operators handle them.

So a Boltzmann weight assembled from $d$, $\delta$, and the inner product admits the whole space group, while one containing a wedge may not --- and it makes no difference whether the wedge sits in a constraint or in the action itself.
Each action declares the consequence through its ``admissible_symmetries``.

Among the built-in formulations only :class:`~supervillain.action.NoIntersections` is restricted, its constraint $q = dn \wedge dn = 0$ being a cup square.
Its admissible sign flips are therefore the identity and the full inversion $x \to -x$.
Those two are themselves a subgroup, the inversion being central, so uniform draws, closure under inverses, and detailed balance are all unaffected; what shrinks is the reach of a single move.

.. note ::

   Two lattice identities are worth knowing here, since neither is quite the continuum statement.
   ${\star}{\star}$ is $(-1)^{p(D-p)}$ --- the continuum sign --- times a translation by one step in *every* direction.
   And $\sum_x (a \wedge {\star} b) = \langle a, b \rangle$ exactly, so the inner product is recovered even though neither ${\star}$ nor $\wedge$ is equivariant under a sign flip on its own.
   That is why $\delta$ is unaffected despite being expressible as ${\star} d {\star}$.

.. autoclass :: supervillain.generator.symmetry.Symmetry
   :members:

.. autofunction :: supervillain.generator.symmetry.all_flip_sets

.. autoclass :: supervillain.generator.symmetry.LatticeSymmetry
   :members:

.. autoclass :: supervillain.generator.symmetry.Translation
   :members:

:class:`~supervillain.generator.symmetry.Reflection` refuses an action whose admissible flip sets are a proper subset of all $2^D$, rather than applying only the few that survive.

.. autoclass :: supervillain.generator.symmetry.Reflection
   :members:

.. autoclass :: supervillain.generator.symmetry.AxisPermutation
   :members:

:class:`~supervillain.generator.symmetry.SpaceGroup` instead narrows to the largest subgroup the action admits, and its ``report`` names the group it samples.

.. autoclass :: supervillain.generator.symmetry.SpaceGroup
   :members:

Charge conjugation acts on the field values rather than on lattice coordinates, so it is not built from a :class:`~supervillain.generator.symmetry.Symmetry`.
It is admissible for every action, :class:`~supervillain.action.NoIntersections` included.
The constraint $q = F \wedge F$, with $F = dn$, is quadratic in $F$, so $n \to -n$ sends $F \to -F$ and leaves $q$ invariant outright --- conjugation never touches the lattice map the cup product depends on.

.. autoclass :: supervillain.generator.symmetry.Conjugation
   :members:

--------------------
Combining Generators
--------------------

You can combine generators.
One simple combination is just the sequential application.

.. autoclass :: supervillain.generator.combining.Sequentially
   :members:

Another (trivial) combination is a decorrelating application.

.. autoclass :: supervillain.generator.combining.KeepEvery
   :members:

--------
Monitors
--------

We can make pass-through generators to perform specific tasks, like logging.

.. autoclass :: supervillain.generator.monitor.Logger
   :members:

One could construct monitors that, for example, measured observables inline.
