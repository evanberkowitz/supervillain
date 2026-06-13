

****************************
Configurations and Ensembles
****************************

We perform Markov-chain Monte Carlo and estimate expectation values stochastically.
Each step in the Markov Chain is called a 'configuration'; a set of configurations is called an 'ensemble'.

We need to store data the same way for each configuration.
A batch is a place to store data for a single physical field for every configuration.

.. autoclass :: supervillain.batch.Batch
   :members:
   :special-members: __getitem__, __setitem__, __len__, __iter__
   :show-inheritance:

Each set of Configurations contains a Batch for each physical field.

.. autoclass :: supervillain.configurations.Configurations
   :members:
   :special-members: __getitem__, __setitem__
   :show-inheritance:

Ensembles are made up of configurations and also have other physics information---not just the stored fields themselves.

.. autoclass :: supervillain.ensemble.Ensemble
   :no-special-members:
   :members: Action, from_configurations, generate, measure, measured, autocorrelation_time, continue_from, cut, every, plot_history
   :show-inheritance:

