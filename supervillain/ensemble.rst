

*********
Ensembles
*********

We perform Markov-chain Monte Carlo and estimate expectation values stochastically.

Ensembls are made up of configurations.

.. autoclass :: supervillain.configurations.Configurations
   :members:
   :special-members: __getitem__, __setitem__



.. autoclass :: supervillain.ensemble.Ensemble
   :no-special-members:
   :members: Action, from_configurations, generate, continue_from, cut, every
   :show-inheritance:

