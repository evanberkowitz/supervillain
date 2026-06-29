
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

==================
Topological Charge
==================

The absolute topological-charge density can be measured alongside the action
density on any generated four-dimensional, :math:`W=1` Villain ensemble:

.. code-block:: python

   import supervillain

   L = supervillain.lattice.Lattice(D=4, N=4)
   S = supervillain.action.Villain(L, kappa=0.05, W=1)
   G = supervillain.generator.villain.NeighborhoodUpdate(S)

   ensemble = supervillain.Ensemble(S).generate(
       1_000,
       G,
       start='cold',
   )
   measurements = ensemble.measure([
       'ActionDensity',
       'AbsoluteTopologicalChargeDensity',
   ])
   absolute_charge_density = measurements['AbsoluteTopologicalChargeDensity']

``absolute_charge_density`` contains one scalar value per generated
configuration.  Accessing ``ensemble.AbsoluteTopologicalChargeDensity`` gives
the same cached measurement.

.. autoclass :: supervillain.observable.AbsoluteTopologicalChargeDensity
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


.. autoclass :: supervillain.observable.WrappingSquared
   :members:
   :show-inheritance:



=================
Spin Correlations
=================

.. autoclass :: supervillain.observable.Spin_Spin
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.Spin_Spin_Normalized
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

.. autoclass :: supervillain.observable.Vortex_Vortex_Normalized
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.VortexSusceptibility
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.VortexSusceptibilityScaled
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.VortexCriticalMoment
   :members:
   :show-inheritance:


.. _staticmethod: https://docs.python.org/3/library/functions.html#staticmethod
.. _Descriptor: https://docs.python.org/3/howto/descriptor.html
.. _NotImplemented: https://docs.python.org/3/library/exceptions.html#NotImplementedError
