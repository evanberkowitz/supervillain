
***********
Observables
***********

When possible we implement observables for different formulations.
Since the equivalence between formulations is link-by-link, when measured on the same size lattice the different formulations should give equal results in the infinite-statistics limit.
Which formulation is more efficient of course depends on the action parameters and on the lattice size.

=====
Links
=====

.. autoclass :: supervillain.observable.Links
   :members:
   :show-inheritance:

=======================
Internal Energy Density
=======================

.. autoclass :: supervillain.observable.InternalEnergyDensity
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.InternalEnergyDensitySquared
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.InternalEnergyDensityVariance
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.SpecificHeatCapacity
   :members:
   :show-inheritance:

======
Action
======

.. autoclass :: supervillain.observable.ActionDensity
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.Action_Action
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.ActionTwoPoint
   :members:
   :show-inheritance:

=======
Winding
=======

.. autoclass :: supervillain.observable.WindingSquared
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.Winding_Winding
   :members:
   :show-inheritance:

========
Wrapping
========

.. autoclass :: supervillain.observable.TorusWrapping
   :members:
   :show-inheritance:

Calculations of the :func:`~.autocorrelation_time` are easiest for scalars.
These decouple the two components of the wrapping.

.. warning ::
   Like :class:`~.TorusWrapping` these are motivated differently for the different formulations.

.. autoclass :: supervillain.observable.TWrapping
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.XWrapping
   :members:
   :show-inheritance:



=================
Spin Correlations
=================

.. autoclass :: supervillain.observable.Spin_Spin
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.SpinSusceptibility
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.SpinSusceptibilityScaled
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.SpinCriticalMoment
   :members:
   :show-inheritance:

===================
Vortex Correlations
===================

.. autoclass :: supervillain.observable.Vortex_Vortex
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.VortexSusceptibility
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.VortexSusceptibilityScaled
   :members:
   :show-inheritance:



.. _staticmethod: https://docs.python.org/3/library/functions.html#staticmethod
.. _Descriptor: https://docs.python.org/3/howto/descriptor.html
.. _NotImplemented: https://docs.python.org/3/library/exceptions.html#NotImplementedError
