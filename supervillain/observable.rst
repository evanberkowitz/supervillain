
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

In four dimensions we can define a simple 4-form topological-charge density $q_x$ in the Villain frame,

.. math ::

   \begin{aligned}
      Q &= \sum_x q_x
      &
      q_x = (dn \wedge dn)_x.
   \end{aligned}

In D=4 the density is a top-form.
The charge density is exact, since

.. math ::

   \begin{aligned}
      q_x &= dn \wedge dn = d(n \wedge dn) = (dJ)_x
      &
      J &= n \wedge dn
   \end{aligned}

and the Leibniz rule :eq:`leibniz-rule` holds exactly.
This means that the charge is locally conserved, and the total charge $Q$ vanishes configuration by configuration.
(Of course you could equally well say that $J \sim dn \wedge n$.)


.. autoclass :: supervillain.observable.TopologicalChargeDensity
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.TopologicalCharge
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.TopologicalChargeDensitySquared
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.TopologicalTwoPoint
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.Topological_Topological
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
