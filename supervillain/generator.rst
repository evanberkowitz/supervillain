

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


Metropolis Updates
------------------

A less dumb algorithm is a local Metropolis-tested update, which changes only fields in some small area of the lattice.

As an example, we can formulate an update scheme offering localized changes to the $\phi$ and $n$ fields in the :class:`~.Villain` formulation.

Picking a site $x$ at random and proposing a change 

.. math ::
    
    \begin{aligned}
    \Delta\phi_x    &\sim \text{uniform}(-\texttt{interval\_phi}, +\texttt{interval\_phi})
    \\
    \Delta n_\ell   &\sim [-\texttt{interval\_n}, +\texttt{interval\_n}]
    \end{aligned}

for the $\phi$ on $x$ and $n$ on links $\ell$ which touch $x$ is ergodic (once swept over the lattice) and satisfies detailed balance so long as we accept the proposal based on the change of action---the so-called Metropolis acceptance criterion.
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

Heatbath Updates
----------------

Because of the simplicity of the :class:`~.Villain` action, these decoupled proposals can be sampled *exactly* rather than Metropolis-tested.
At fixed $n$ the action is quadratic in $\phi$, so a single $\phi$ has a Gaussian conditional; at fixed $\phi$ each $n_\ell$ enters the action independently, so a single $n$ has a (discrete) Gaussian conditional on the constraint-preserving coset (when the constraint integer $W>1$).
The :class:`~.SiteHeatbath` and :class:`~.LinkHeatbath` draw these conditionals directly.
They are drop-in heatbath replacements for the :class:`~.SiteUpdate` and :class:`~.LinkUpdate`.

.. autoclass :: supervillain.generator.villain.SiteHeatbath
   :members: step, report

The :class:`~.SiteOverrelaxation` is the microcanonical partner of the :class:`~.SiteHeatbath`: rather than drawing the $\phi$ conditional it *reflects* $\phi$ about the conditional mean, an action-preserving, rejection-free move that decorrelates the Gaussian $\phi$ sector faster than the heatbath alone.
It is not ergodic by itself --- it never leaves the action shell --- and is run interleaved with the heatbath.
Neither is in the :func:`~.villain.Hammer` any more: both are superseded by the one-shot draw below.

.. autoclass :: supervillain.generator.villain.SiteOverrelaxation
   :members: step, report

Sweeping single-site conditionals is stochastic Gauss--Seidel, which relaxes a massless field's long-wavelength modes only diffusively.
But at fixed $n$ the *joint* conditional of $\phi$ is Gaussian too, and $\delta d$ (equal to the Laplacian $\Delta$ on the 0-form $\phi$) is diagonal in Fourier space, so the whole field can be drawn at once.
The :class:`~.FourierSiteHeatbath` splits $\phi = \bar\phi + \eta$ into the solution of $\Delta\bar\phi = 2\pi\delta n$ and a fluctuation whose Fourier modes are independent, and produces it in three fast Fourier transforms.
Its successive $\phi$ are statistically *independent* rather than merely correlated less, so the $\phi$ sector has no autocorrelation at all and there is nothing left for an overrelaxation to improve.
It is therefore what both the :func:`~.villain.Hammer` and the :func:`~.no_intersection.Hammer` use for $\phi$, in place of the :class:`~.SiteHeatbath` + :class:`~.SiteOverrelaxation` pair; those Hammers consequently no longer take an ``overrelax`` argument.

.. autoclass :: supervillain.generator.villain.FourierSiteHeatbath
   :members: step, conditional_mean, report

As mentioned we can also update the $n$ fields with a heatbath.

.. autoclass :: supervillain.generator.villain.LinkHeatbath
   :members: step, report

Just as the :class:`~.SiteHeatbath` and :class:`~.LinkHeatbath` are exact-conditional versions of the :class:`~.SiteUpdate` and :class:`~.LinkUpdate`, the closed updates admit heatbath kernels too.
Holding everything else fixed, the action is quadratic in a single site's integer shift $z_x$ (for :class:`~.ExactUpdate`, $\Delta n = dz$) or in the integer holonomy shift $h_\mu$ on a slice (for :class:`~.CohomologyUpdate`), so each conditional is a one-dimensional *discrete* Gaussian which the :class:`~.ExactHeatbath` and :class:`~.CohomologyHeatbath` draw directly (by truncated enumeration and a Gumbel-max pick), never rejecting.
Being discrete, neither has a microcanonical overrelaxation partner (a reflection about the real mean leaves the integer lattice).

.. autoclass :: supervillain.generator.villain.ExactHeatbath
   :members: step, report

.. autoclass :: supervillain.generator.villain.CohomologyHeatbath
   :members: step, report


Worm and Cluster Algorithms
---------------------------

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

A different nonlocal move for the Villain spin sector is the Wolff reflection cluster :cite:`PhysRevLett.62.361`.
Reflecting $\phi \rightarrow 2r - \phi$ on a cluster of sites and $n \rightarrow -n$ on its internal links sends the gauge-invariant link $(d\phi - 2\pi n) \rightarrow -(d\phi - 2\pi n)$ internally, leaving every internal-bond energy invariant; a whole-lattice reflection is the exact symmetry $(d\phi - 2\pi n) \rightarrow -(d\phi - 2\pi n)$ (an O(2) reflection combined with charge conjugation $n \rightarrow -n$).
Growing the cluster across boundary bonds with the probability $q = 1 - \exp(-[E_b^R - E_b]_+)$ makes the whole reflection rejection-free while satisfying detailed balance.

.. autoclass :: supervillain.generator.villain.ReflectionCluster
   :members: step, report

Like the worm the cluster is not ergodic on its own.
A torus-wrapping cluster can change the winding holonomy ${w}_\mu$ (the $H^1$ class) but the sign do not allow magnitude fluctuations---only sign flips.
Combined with the local updates its value is decorrelating the long-wavelength spin modes near criticality.

The Hammer
----------

Finally, we provide a convenience function which provides an ergodic generator.

.. autofunction :: supervillain.generator.villain.Hammer

-------------------------
The Worldline Formulation
-------------------------

Metropolis Updates
------------------

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

Heatbath Updates
----------------

As in the Villain formulation these decoupled proposals can be sampled *exactly*.
Writing the gauge-invariant link one-form $f = m - \delta v / \bar{W}$, a single plaquette's shift of $v$ or of the integer two-form $t$ (with $\Delta m = \delta t$) has a Gaussian conditional, so the :class:`~.VortexHeatbath` and :class:`~.CoexactHeatbath` are drop-in heatbath replacements for the :class:`~.worldline.VortexUpdate` and :class:`~.CoexactUpdate` that take no proposal width and never reject.
The :class:`~.CoexactHeatbath` (and the :class:`~.VortexHeatbath` at finite $W$) draws a *discrete* Gaussian; at $W=\infty$ the vortex field $v$ is real, so the :class:`~.VortexHeatbath` conditional is continuous and admits a microcanonical partner, the :class:`~.VortexOverrelaxation`, which reflects $v$ about its conditional mean.
The wrapping sector admits a heatbath too: each torus cycle's action is quadratic in a single integer winding shift $\Delta$, and the $D N^{D-1}$ cycles touch disjoint links, so the :class:`~.WrappingHeatbath` draws every cycle's shift from its exact *discrete* Gaussian simultaneously --- the rejection-free drop-in for the :class:`~.worldline.WrappingUpdate`, curing exactly the large-action-change acceptance problem noted above.  Like the :class:`~.worldline.WrappingUpdate` it is not ergodic on its own.

.. autoclass :: supervillain.generator.worldline.VortexHeatbath
   :members: step, report

.. autoclass :: supervillain.generator.worldline.VortexOverrelaxation
   :members: step, report

.. autoclass :: supervillain.generator.worldline.CoexactHeatbath
   :members: step, report

.. autoclass :: supervillain.generator.worldline.WrappingHeatbath
   :members: step, report


Worm Algorithms
---------------

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
