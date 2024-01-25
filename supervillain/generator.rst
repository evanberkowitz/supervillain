

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

But, also, as an *algorithm* the neighborhood update suffers because it can only make small changes in a local area.  Smarter algorithms can make high-acceptance updates to many variables across the lattice, which can help overcome *critical slowing down*.

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

Another class of constraint-maintaining algorithms are *worm algorithms*.
They explicitly allow the configuration to change in the larger space of constraint-violating configurations and when the worm closes the constraint is once again obeyed.
Worm algorithms are purportedly less punishing with regards to autocorrelation times, and are also efficent tools for calculating *correlations* at the same time as generating configurations.

.. autoclass :: supervillain.generator.worm.UndirectedWorm
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
