Recent Changes
==============

Unreleased
----------

* Added :mod:`supervillain.generator.no_intersection.surface_worm`, the
  :class:`~supervillain.generator.no_intersection.surface_worm.gas.SurfaceWormGas`:
  an $F$-space extended-ensemble sampler for the Jacobson No Intersections
  model that relaxes and prices *both* physical constraints (closedness and
  self-intersection-freedom) instead of transporting defects at fixed $dn =
  0$.  It only ever emits a configuration on the exactness-gated joint
  vacuum $D = Q = 0$ with every $H^2$ period zero --- closed alone is not
  exact on $T^4$, so that third condition is what guarantees every emitted
  row corresponds to a genuine Villain $(n, \varphi)$.  The torus-winding
  tilt $Z_\text{wind}$ is unconditionally part of the sampled distribution
  (no ``windingInSampler`` toggle), so emitted rows need no importance
  reweighting.  :class:`~supervillain.generator.no_intersection.SectorWeightTuner`
  and :class:`~supervillain.generator.no_intersection.PairUmbrellaTuner`
  learn the multicanonical tables that keep the closed shell populated.

.. git_commit_detail::
   :branch:
   :commit:

.. git_changelog::
   :detailed-message-pre: True
