#!/usr/bin/env python

from .worm import IntersectionWorm
from .free_target_worm import FreeTargetWorm
from .defect_gas import DefectGas, DefectGasFugacityTuner
from .link import ConstrainedLinkUpdate
from .wrapping import WrappingLoopUpdate
from .planar import PlanarFluxUpdate
from .scattershot import ScattershotUpdate

import supervillain.generator.villain as _villain
import supervillain.generator.combining as _combining


def Hammer(S, fugacity=None):
    r'''
    Syntactic sugar for an ergodic :class:`~.Sequentially` combination of the
    No-Intersection generators.  It may change from version to version as new
    generators become available or get improved.

    The :class:`DefectGas`, :class:`ConstrainedLinkUpdate`, and
    :class:`WrappingLoopUpdate` all update $n$ only; a
    :class:`~supervillain.generator.villain.SiteUpdate` is included to update
    $\phi$.  We also reuse the Villain
    :class:`~supervillain.generator.villain.ExactUpdate` and
    :class:`~supervillain.generator.villain.CohomologyUpdate`: both change $n$ by a
    *closed* form, so they leave $dn$ (and hence the charge density $q = dn\wedge dn$)
    untouched and manifestly preserve the constraint.  The
    :class:`~supervillain.generator.villain.ExactUpdate` moves the exact part of $n$,
    while the :class:`~supervillain.generator.villain.CohomologyUpdate` changes the
    torus-wrapping holonomy of $n$ at fixed $dn$ --- a sector the other $n$-updates do
    not reach.  The :class:`PlanarFluxUpdate` contributes a large,
    whole-lattice tunneling move (deposit a decomposable flux sheet), complementing the
    local loop moves.  The :class:`ScattershotUpdate` proposes a joint,
    atomic change of every link at once from a symmetric full-support distribution,
    which upgrades the combination's ergodicity on the constraint surface from a
    plausible hope to a one-line **theorem**: every valid configuration is proposed from
    every other with positive probability, so the chain is manifestly irreducible.

    Finally the :class:`DefectGas` supplies the defect transport that **no worm could**.
    It occupies the slot a worm would, and measures the intersection correlator inline as
    :class:`~.Theta_Theta` and :class:`~.Vacuum_Ticks` --- absolutely normalized, where a
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
        tuner's default ``D_max = 8`` cap (an explicit ``fugacity`` leaves ``D_max``
        uncapped as before).

    Returns
    -------
    An ergodic generator for updating No-Intersection configurations.
    '''
    roster = (
        _villain.SiteUpdate(S),
        _villain.ExactUpdate(S),
        _villain.CohomologyUpdate(S),
        ConstrainedLinkUpdate(S),
        WrappingLoopUpdate(S),
        PlanarFluxUpdate(S),
        ScattershotUpdate(S),
    )
    if fugacity is None:
        return DefectGasFugacityTuner(S, companions=roster).generator()
    return _combining.Sequentially((*roster, DefectGas(S, fugacity)))
