#!/usr/bin/env python

r"""Generators for the Jacobson No Intersections model.  Alongside the
:class:`DefectGas` roster :func:`Hammer` assembles --- which transports
defects at fixed $dn = 0$ --- the :class:`SurfaceWormGas` is an $F$-space
extended-ensemble alternative that relaxes and prices *both* constraints,
emitting exactness-gated physical configurations; see
:mod:`~.surface_worm` for its full account.
"""

from .worm import IntersectionWorm
from .free_target_worm import FreeTargetWorm
from .defect_gas import DefectGas, DefectGasFugacityTuner, DefectGasWeightTuner
from .link import ConstrainedLinkUpdate
from .link_heatbath import ConstrainedLinkHeatbath
from .wrapping import WrappingLoopUpdate
from .planar import PlanarFluxUpdate
from .scattershot import ScattershotUpdate
from .surface_worm import (SurfaceWormGas, SectorWeightTuner, ChargeWeightTuner,
                           PairUmbrellaTuner, TransportTuner)

import supervillain.generator.villain as _villain
import supervillain.generator.combining as _combining


def Hammer(S, fugacity=None):
    r'''
    Syntactic sugar for an ergodic :class:`~.Sequentially` combination of the
    No-Intersection generators.  It may change from version to version as new
    generators become available or get improved.

    The :class:`DefectGas` and :class:`ConstrainedLinkHeatbath` both update $n$ only; a
    :class:`~supervillain.generator.villain.FourierSiteHeatbath` is included to update
    $\phi$ (exactly, since at fixed $n$ the action is Gaussian in $\phi$ --- and *jointly*
    so, which is why the whole field is drawn at once rather than swept site by site and
    then overrelaxed; the draw leaves $n$ untouched, so $dn\wedge dn = 0$ survives
    trivially).  The local
    single-link move is the :class:`ConstrainedLinkHeatbath` rather than the Metropolis
    :class:`ConstrainedLinkUpdate` --- they share the same connectivity (a link is *clean*
    or *frozen* independently of the shift), so the heatbath simply resamples the clean
    links from their exact discrete-Gaussian conditional instead of a $\pm 1$ step, using
    the heatbath variant here exactly as we do for $\phi$ and the closed-$n$ moves below.  We also reuse the Villain
    :class:`~supervillain.generator.villain.ExactUpdate` and
    :class:`~supervillain.generator.villain.CohomologyUpdate`: both change $n$ by a
    *closed* form, so they leave $dn$ (and hence the charge density $q = dn\wedge dn$)
    untouched and manifestly preserve the constraint.  The
    :class:`~supervillain.generator.villain.ExactUpdate` moves the exact part of $n$,
    while the :class:`~supervillain.generator.villain.CohomologyUpdate` changes the
    torus-wrapping holonomy of $n$ at fixed $dn$ --- a sector the other $n$-updates do
    not reach.  The :class:`PlanarFluxUpdate` contributes a large,
    whole-lattice tunneling move (deposit a decomposable flux sheet), complementing the
    local single-link moves.  (The :class:`WrappingLoopUpdate` --- the coordinated
    torus-wrapping loop that escapes frozen configurations --- is **omitted for now**: it is
    still a slow pure-python reference implementation, and since the
    :class:`ScattershotUpdate` and :class:`DefectGas` below already guarantee ergodicity, its
    structured mixing is a bonus not yet worth its cost.  Re-add it once it is compiled.)  The
    :class:`ScattershotUpdate` proposes a joint,
    atomic change of every link at once from a symmetric full-support distribution,
    which upgrades the combination's ergodicity on the constraint surface from a
    plausible hope to a one-line **theorem**: every valid configuration is proposed from
    every other with positive probability, so the chain is manifestly irreducible.

    Finally the :class:`DefectGas` supplies the defect transport that **no worm could**.
    It occupies the slot a worm would, and measures the intersection correlator inline as
    :class:`~.Theta_Theta` and :class:`~.VacuumTicks` --- absolutely normalized, where a
    worm histogram needs its origin bin.  See :ref:`the no-intersection docs
    <no_intersection>` for the measurements that retired the worms.

    Parameters
    ----------
    S: a NoIntersections action
    fugacity: float or None
        The per-defect fugacity handed to the :class:`DefectGas`.  ``None`` (the
        default) delegates to a :class:`DefectGasFugacityTuner` built with this
        roster as its companions, which runs short Monte-Carlo probes down a ladder
        before sampling begins; pass an explicit value to skip that cost.  The
        emitted configurations satisfy the constraint for **any** fugacity
        $\zeta \in (0, 1]$ --- it tunes only the variance.  The tuned gas inherits the
        tuner's default ``max_defects = 8`` cap (an explicit ``fugacity`` leaves ``max_defects``
        uncapped as before).

    Returns
    -------
    An ergodic generator for updating No-Intersection configurations.
    '''
    # The exact closed-n moves (ExactHeatbath: Δn=dz; CohomologyHeatbath: a constant on a
    # slice) and the φ draw all leave dn untouched, so dn∧dn=0 is preserved exactly;
    # LinkHeatbath is still excluded here since its W-coset move would break the constraint.
    # WrappingLoopUpdate is omitted for now: it is a slow pure-python reference implementation,
    # and ScattershotUpdate/DefectGas already cover ergodicity.  Re-add it once it is compiled.
    roster = (
        _villain.FourierSiteHeatbath(S),
        _villain.ExactHeatbath(S),
        _villain.CohomologyHeatbath(S),
        ConstrainedLinkHeatbath(S),
        PlanarFluxUpdate(S),
        ScattershotUpdate(S),
    )
    if fugacity is None:
        return DefectGasFugacityTuner(S, companions=roster).generator()
    return _combining.Sequentially((*roster, DefectGas(S, fugacity)))
