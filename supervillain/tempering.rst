******************
Parallel Tempering
******************

At small κ single Markov chains can get trapped in metastable basins.
Parallel tempering runs a ladder of actions that differ only in κ, applying every rung's local updates in lockstep and letting adjacent rungs exchange whole configurations with a Metropolis swap; healthy configurations generated where the chain mixes well diffuse to the rungs where it does not.

Because the :class:`Villain <supervillain.action.Villain>`-family actions are linear in κ and the constraints are κ-independent, the swap decision needs only the scalar $E = S/\kappa$ from each rung, and a configuration valid on one rung is valid on every rung.

.. autoclass :: supervillain.tempering.ParallelTempering
   :members:
   :show-inheritance:

.. autofunction :: supervillain.tempering.swap_accepted

.. autoclass :: supervillain.tempering.EvenOddPairs
   :members:
   :show-inheritance:

.. autoclass :: supervillain.tempering.TemperedRung
   :members:
   :show-inheritance:

Tuning the ladder
=================

Where the rungs sit is the tempering-specific tuning problem; it is action-agnostic because the swap acceptance depends only on $\Delta\kappa$ and the distributions of $E$.
Proposal machinery (fugacities and the like) is tuned per rung by the per-action tuners, between tempering legs.

.. autoclass :: supervillain.tempering.ParallelTemperingTuner
   :members:
   :show-inheritance:
