
***********
Observables
***********

Observables are physical quantities that can be measured on ensembles.
We distinguish between *primary observables* and *derived quantities*, using language from Ref. :cite:`Wolff:2003sm`.
A primary observable can be measured directly on a single configuration.
A derived quantity is a generally nonlinear function of primary observables which can only be estimated using expectation values from a whole ensemble.

The same observable or derived quantity might be computed differently using different :ref:`actions <action>`.

.. _primary observables:

===================
Primary Observables
===================

You can construct observables by writing a class that inherits from the ``supervillain.observable.Observable`` class.

.. autoclass :: supervillain.observable.Observable
   :members:

Your observable can provide different implementations for different actions.
Implementations are `staticmethod`_\ s named for their corresponding action that take an action object and a single configuration of fields needed to evaluate the action.

A simple example, since :ref:`actions <action>` are already callable, is

.. literalinclude:: observable/energy.py
   :pyobject: InternalEnergyDensity


Under the hood ``Observable``\ s are attached to the :class:`~.Ensemble` class.
In particular, you can evaluate the observable for *every* configuration in an ensemble by just calling for the ensemble's property with the name of the observable.
The result is cached and repeated calls for that ensemble require no further computation.

For example, to evaluate the action density you would ask for ``ensemble.ActionDensity``.
If the ensemble was constructed from a :class:`~.Villain` action you will get the ``Villain`` implementation above; if it was constructed from a :class:`~.Worldline` action you will get the ``Worldline`` implementation.

All of these nice features are accomplished using the `Descriptor`_ protocol but the implementation is unimportant.

If the observable does not provide an implementation for the ensemble's action, asking for it will raise a `NotImplemented`_ exception.


-----------------------
Internal Energy Density
-----------------------

.. autoclass :: supervillain.observable.InternalEnergyDensity
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.InternalEnergyDensitySquared
   :members:
   :show-inheritance:

------
Action
------

.. autoclass :: supervillain.observable.ActionDensity
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.Action_Action
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.ActionTwoPoint
   :members:
   :show-inheritance:

-------
Winding
-------

.. autoclass :: supervillain.observable.WindingSquared
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.Winding_Winding
   :members:
   :show-inheritance:

-----------------
Spin Correlations
-----------------

.. autoclass :: supervillain.observable.Spin_Spin
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.SpinSusceptibility
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.SpinSusceptibilityScaled
   :members:
   :show-inheritance:


.. _derived quantities:

==================
Derived Quantities
==================

Like the primary observables, derived quantities also inherit from a common :class:`supervillain.observable.DerivedQuantity` class.

.. autoclass :: supervillain.observable.DerivedQuantity
   :members:

Just like observables, derived quantities can provide different implementations for different actions.
However, because derived quantities are (possibly-)nonlinear combinations of expectation values of primary observables, they cannot be measured on single configurations and therefore are not attached to :class:`~.Ensemble`\ s but to :class:`~.Bootstrap`\ s, which automatically provide resampled expectation values of primary obervables.

DerivedQuantity implementations are `staticmethod`_\ s named for their corresponding action that take an action object and a bootstrap-resampled expectation value of primary observables or other derived quantities.
Because DerivedQuantities are often reductions of other primary Observables or DerivedQuantities, the implementation may be shared between different actions; you can provide a common ``default`` implementation to fall back to that can be overridden by action-specific implementations.
In other words, where an :class:`~.Observable` takes the action and the field variables, a :class:`~.DerivedQuantity` takes the action and potentially other quantities.
The arguments are the action and the exact names of the needed observables or derived quantities.

.. note ::
   The arguments' names matter and have to exactly match the needed expectation values.

The implementations are automatically threaded over the bootstrap samples, maintaining all correlations.


.. _staticmethod: https://docs.python.org/3/library/functions.html#staticmethod
.. _Descriptor: https://docs.python.org/3/howto/descriptor.html
.. _NotImplemented: https://docs.python.org/3/library/exceptions.html#NotImplementedError
