
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

Villain Classic Worm
====================
.. autoclass :: supervillain.generator.reference_implementation.villain.ClassicWorm
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


==================
Differential Forms
==================

Interlaced Forms
================

:class:`~supervillain.lattice.Form` stores a $p$-form compactly, with one slot for every $p$-cell and no wasted space.
The interlaced reference implementation instead realizes :ref:`the interlaced picture <interlaced>` literally: a $p$-form on an $N^D$ lattice is a $(2N)^D$ array, zero except at points with exactly $p$ odd coordinates.
Its operators are independent implementations of the exterior calculus with the same :ref:`conventions <sign-conventions>`, giving the compact versions a fixed target for correctness.

.. automodule :: supervillain.lattice.interlaced
   :members: Lattice, push, pull, d, delta, wedge, star

