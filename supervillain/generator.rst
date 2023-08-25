

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

A less dumb algorithm is a local update,

.. autoclass :: supervillain.generator.metropolis.SlowNeighborhoodUpdate
   :members:

The :class:`SlowNeighborhoodUpdate <supervillain.generator.metropolis.SlowNeighborhoodUpdate>` generator suffers from a variety of defects.

First, its *implementation* makes a lot of calls.  We could 'unfactor' the proposal, single-site accept/reject, and sweep of the lattice for some benefits in speed.  Moreover, to compute the change in action it evaluates the action for both configurations and takes the difference.  But, we know that's silly, most of the $\phi$s and $n$s haven't changed at all and those links will contribute to both actions equally, giving zero difference.  We could reduce the arithmetic substantially by computing the difference directly.

.. autoclass :: supervillain.generator.NeighborhoodUpdate
   :members:

But, also, as an *algorithm* the neighborhood update suffers because it can only make small changes in a local area.  Smarter algorithms can make high-acceptance updates to many variables across the lattice, which can help overcome *critical slowing down*.
