
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
Implementations are `staticmethod`_\ s named for their corresponding action.
Implementations *always* take the action first, and then can take field variables or other primary observables.

.. note ::
   Implementations always take the action ``S`` first.

.. note ::
   The names of the arguments matter; they're used to look up the correct field variables or other observables.

A simple example, since :ref:`actions <action>` are already callable, is

.. literalinclude:: observable/energy.py
   :pyobject: InternalEnergyDensity


Under the hood ``Observable``\ s are attached to the :class:`~.Ensemble` class.
In particular, you can evaluate the observable for *every* configuration in an ensemble by just calling for the ensemble's property with the name of the observable.
The result is cached and repeated calls for that ensemble require no further computation.

For example, to evaluate the action density you would ask for ``ensemble.ActionDensity``.
If the ensemble was constructed from a :class:`~.Villain` action you will get the ``Villain`` implementation; if it was constructed from a :class:`~.Worldline` action you will get the ``Worldline`` implementation.

All of these nice features are accomplished using the `Descriptor`_ protocol but the implementation is unimportant.

If the observable does not provide an implementation for the ensemble's action, asking for it will raise a `NotImplemented`_ exception.
However, some observables can provide a ``default`` implementation, which is particularly useful for simple functions of other primary observables.
For example, the :class:`~.SpinSusceptibility` is just the sum of the :class:`~.Spin_Spin` two-point function.

.. literalinclude:: observable/spin.py
   :pyobject: SpinSusceptibility

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
Just like an :class:`~.Observable`, a :class:`~.DerivedQuantity` takes the action, primary observables, and potentially other derived quantities, though it is almost certainly a mistake for a derived quantity to depend directly on field variables.

.. note ::
   Implementations always take the action ``S`` first.

.. note ::
   The arguments' names matter and have to exactly match the needed expectation values.

The implementations are automatically threaded over the bootstrap samples, maintaining all correlations.

.. literalinclude:: observable/action.py
   :pyobject: Action_Action

.. _physical quantities:

===================
Physical Quantities
===================

When possible we implement observables for different formulations.
Since the equivalence between formulations is link-by-link, when measured on the same size lattice the different formulations should give equal results in the infinite-statistics limit.
Which formulation is more efficient of course depends on the action parameters and on the lattice size.

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



.. _staticmethod: https://docs.python.org/3/library/functions.html#staticmethod
.. _Descriptor: https://docs.python.org/3/howto/descriptor.html
.. _NotImplemented: https://docs.python.org/3/library/exceptions.html#NotImplementedError
