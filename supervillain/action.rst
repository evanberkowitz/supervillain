.. _action:

*******
Actions
*******

An *action* assigns a weight to a configuration.
Everything else in the library is organized around one: a :doc:`generator <generator>` proposes configurations distributed according to it, an :class:`~supervillain.Ensemble` holds the configurations it produced, and an :doc:`observable <observable>` is a function of them.

The same physics can be written down in more than one way.
The :ref:`Villain formulation <villain>` samples a real 0-form $\varphi$ and an integer 1-form $n$.
The :ref:`worldline formulation <worldline>` samples an integer 1-form $m$ and a 2-form $v$ instead, and the constrained version is naturally free of the sign problem while the obvious modified Villain model is not.
They differ in what a configuration *is*, so they cannot share a single evaluation signature --- but they agree on the structure of everything else.

:class:`~supervillain.action.Action` records that agreement.
It implements nothing; each formulation supplies its own physics.
What it fixes is the programming surface: an action lives on a :class:`~supervillain.lattice.Lattice`, it can be evaluated, it declares what fields a configuration is made of, it says which configurations are legal, and it says which :ref:`lattice symmetries <space-group>` it admits.

.. note ::

   Because a configuration is a dictionary, code that does not know the
   formulation evaluates an action as ``S(**configuration)``.  Entries the
   configuration carries beyond the fields --- inline observables, or an
   :ref:`importance weight <importance-weights>` --- are absorbed by each
   action's ``**kwargs`` and ignored.

.. autoclass :: supervillain.action.Action
   :members:
   :exclude-members: Lattice
