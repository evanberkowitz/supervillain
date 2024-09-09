
.. _reference_implementations:

*************************
Reference Implementations
*************************

Sometimes the obvious implementation can be accelerated dramatically.

It is useful to have a *reference implementation* where things are done the totally obvious way so that smarter or faster approaches have a fixed target for correctness.  

.. note ::
   Everything here can be considered an implementation detail and a user should not need the reference implementations at all.


==========
Generators
==========

Villain NeighborhoodUpdate
==========================
.. autoclass :: supervillain.generator.reference_implementation.villain.NeighborhoodUpdateSlow
   :members:

Wordline Classic Worm
=====================
.. autoclass :: supervillain.generator.reference_implementation.worldline.ClassicWorm
   :members:



===========
Observables
===========


Spin Correlations
=================
.. autoclass :: supervillain.observable.reference_implementation.spin.Spin_SpinSloppy
   :members:
   :show-inheritance:

.. autoclass :: supervillain.observable.reference_implementation.spin.Spin_SpinSlow
   :members:
   :show-inheritance:

