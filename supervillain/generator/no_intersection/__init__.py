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
    $\phi$, so the combination is ergodic.

    Parameters
    ----------
    S: a NoIntersections action

    Returns
    -------
    An ergodic generator for updating No-Intersection configurations.
    '''
    return _combining.Sequentially((
        _villain.SiteUpdate(S),
        ConstrainedLinkUpdate(S),
        WrappingLoopUpdate(S),
        IntersectionWorm(S),
    ))
