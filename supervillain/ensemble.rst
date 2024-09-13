

*********
Ensembles
*********

We perform Markov-chain Monte Carlo and estimate expectation values stochastically.

Ensembls are made up of configurations.

.. autoclass :: supervillain.configurations.Configurations
   :members:
   :special-members: __getitem__, __setitem__
   :show-inheritance:



.. autoclass :: supervillain.ensemble.Ensemble
   :no-special-members:
   :members: Action, from_configurations, generate, measure, measured, autocorrelation_time, continue_from, cut, every, plot_history
   :show-inheritance:

