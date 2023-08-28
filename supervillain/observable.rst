
***********
Observables
***********

Observables are physical quantities that can be measured on ensembles.
We distinguish between *primary observables* and *derived quantities*, using language from Ref. :cite:`Wolff:2003sm`.
A primary observable can be measured directly on a single configuration.
A derived quantity is a generally nonlinear function of primary observables which can only be estimated using expectation values from a whole ensemble.

The same observable might be computed differently using different :ref:`actions <action>`.

You can construct observables by writing a class that inherits from the ``supervillain.observable.Observable`` class.

.. autoclass :: supervillain.observable.Observable
   :members:

Your observable can provide different implementations for different actions.
Implementations are `staticmethod`_\ s named for their corresponding action that take an action object and a single configuration of fields needed to evaluate the action.

A simple example, since :ref:`actions <action>` are already callable, is

.. literalinclude:: observable/energy.py

Under the hood ``Observable``\ s are attached to the :class:`~.Ensemble` class.
In particular, you can evaluate the observable for *every* configuration in an ensemble by just calling for the ensemble's property with the name of the observable.
The result is cached and repeated calls for that ensemble require no further computation.

For example, to evaluate the action density you would ask for ``ensemble.ActionDensity``.
If the ensemble was constructed from a :class:`~.Villain` action you will get the ``Villain`` implementation above; if it was constructed from a :class:`~.Worldline` action you will get the ``Worldline`` implementation.

All of these nice features are accomplished using the `Descriptor`_ protocol but the implementation is unimportant.

If the observable does not provide an implementation for the ensemble's action, asking for it will raise a `NotImplemented`_ exception.

.. _primary observables:

-------------------
Primary Observables
-------------------

.. autoclass :: supervillain.observable.InternalEnergyDensity
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.WindingSquared
   :members:
   :show-inheritance:

.. _staticmethod: https://docs.python.org/3/library/functions.html#staticmethod
.. _Descriptor: https://docs.python.org/3/howto/descriptor.html
.. _NotImplemented: https://docs.python.org/3/library/exceptions.html#NotImplementedError
