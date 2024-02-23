

********
Sampling
********

:class:`~.Ensemble`\ s can be :py:meth:`generated <supervillain.ensemble.Ensemble.generate>` using a *generator*.
A generator is any object which has a class with a method that takes a configuration as a dictionary and returns a new configuration as a dictionary.

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
    
    \begin{align}
    \Delta\phi_x    &\sim \text{uniform}(-\texttt{interval_phi}, +\texttt{interval_phi})
    \\
    \Delta n_\ell   &\sim [-\texttt{interval_n}, +\texttt{interval_n}]
    \end{align}

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

.. autoclass :: supervillain.generator.villain.FlatUpdate
   :members:

But, also, as an *algorithm* the neighborhood update also suffers because it can only make small changes in a local area.  Smarter algorithms can make high-acceptance updates to many variables across the lattice, which can help overcome *critical slowing down*.


^^^^^^^^^^^^^^^
Worm Algorithms
^^^^^^^^^^^^^^^

With the constraint integer $W$, the flux is required to satisfy $dn \equiv 0 \text{ mod }W$ on every plaquette.
This can make it difficult to make local updates.  For example, changing any link by $1$ can change a plaquette which satisfies the constraint to one that breaks it.
The :class:`~.NeighborhoodUpdate` handles the constraint by only proposing changes to $n$ that would leave the constraint satisfied.
But this is not the only way to think.  Prokof'ev and Svistunov invented *worm algorithms* :cite:`PhysRevLett.87.160601` which operate by introducing defects
where constraints may be broken, propagating those defects, and when they meet and annihilate the configuration again obeys the constraint.

The way to think is that there are two sets of configurations, sometimes called $\{z\}$ (which satisfy the constraint and contribute to the path integral $Z$) and $\{g\}$ (which need not obey the constraint but contribute to a two-point Green's function).
A worm algorithm produces a Markov process on the set $\{z\} \cup \{g\}$ weighted by the unconstrained action; when the Markov chain visits a $z$ configuration we add that to the samples that will contribute to the path integral $Z$.

Starting from some $z$ we put a worm on a single location (in this case, a plaquette), taking us to an identical configuration of fields but now in the $g$ sector 'with a worm' which makes no change to the action at first.
The worm has a head which will move around and a tail which will stay fixed for simplicity.
The head can be thought of as inserting $\exp(-2\pi i v_h/W)$ into the path integral while the tail inserts $\exp(+2\pi i v_t/W)$ at locations $h$ and $t$ respectively.
These insertions shift :ref:`the winding constraint <winding constraint>` to account for those defects,

.. math ::
   :name: worm constraint

    dn_p \equiv (+ \delta_{ph} - \delta_{pt}) \text{ mod } W.

The exception is that when $h$ and $t$ coincide, the configuration with the worm accidentally satisfies the constraint, which is why we can always go from a configuration in $z$ to a configuration in $g$: just add the worm, changing nothing physical.
So, $g$ contains the set of constraint-satisfying configurations as well as configurations that violate the constraint in exactly two places, in opposite ways.

The $g$ configurations include all configurations so long as they satisfy the winding constraint except at the head and the tail, where they violate it by ±1, which we will call :ref:`the worm constraint <worm constraint>`.
Starting from a 'diagonal' configuration which has $h=t$, we can move the head by one plaquette.
As the head moves it changes the $n$ on the link it crosses so that it changes $dn$ on *both* adjacent plaquettes.
This change of $n$ changes the action, and so the moves need to be Metropolis-tested in some sense.

With a clever choice we can ensure that as the head leaves a plaquette it restores the constraint there and breaks it on the destination plaquette.
In this way the head moves around the lattice.
Finally, when the head reaches the tail, the constraint is restored everywhere, and we have a $g$ configuration in the diagonal sector.
Now we might possibly transition to a $z$ configuration, finally adding that configuration to our samples of for the path integral $Z$.
*There is typically no direct way to go from any $z$ configuration to any other; the only way is through $g$.*

.. autoclass :: supervillain.generator.villain.worm.Geometric
   :members:

-------------------------
The Worldline Formulation
-------------------------

In the :class:`~.Worldline` formulation the constraint $\delta m = 0$ restricts which kinds of updates we could propose.
For example, changing only a single link is *guaranteed* to break the constraint on both ends.
So, we need clever generators to maintain the constraint.

.. autoclass :: supervillain.generator.worldline.PlaquetteUpdate
   :members:

To have a fully ergodic algorithm we will also need to update the :class:`~.TorusWrapping` of the worldlines.

.. autoclass :: supervillain.generator.worldline.WrappingUpdate
   :members:


^^^^^^^^^^^^^^^
Worm Algorithms
^^^^^^^^^^^^^^^

Unlike the Villain formulation, the Worldline formulation has a constraint even when :math:`W=1`, :math:`\delta m = 0` everywhere.
As in the Villain case, we can formulate a worm algorithm through configurations which purposefully and explicitly break the constraint.
Worm algorithms are purportedly less punishing with regards to autocorrelation times, and are also efficent tools for calculating *correlations* at the same time as generating configurations.

.. autoclass :: supervillain.generator.worldline.worm.Geometric
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
