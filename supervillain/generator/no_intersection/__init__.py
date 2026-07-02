#!/usr/bin/env python

from .worm import IntersectionWorm
from .link import ConstrainedLinkUpdate
from .wrapping import WrappingLoopUpdate

import supervillain.generator.villain as _villain
import supervillain.generator.combining as _combining


def Hammer(S):
    r'''
    Syntactic sugar for an ergodic :class:`~.Sequentially` combination of the
    No-Intersection generators.  It may change from version to version as new
    generators become available or get improved.

    The :class:`IntersectionWorm`, :class:`ConstrainedLinkUpdate`, and
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
    not reach.

    Parameters
    ----------
    S: a NoIntersections action

    Returns
    -------
    An ergodic generator for updating No-Intersection configurations.
    '''
    return _combining.Sequentially((
        _villain.SiteUpdate(S),
        _villain.ExactUpdate(S),
        _villain.CohomologyUpdate(S),
        ConstrainedLinkUpdate(S),
        WrappingLoopUpdate(S),
        IntersectionWorm(S),
    ))
