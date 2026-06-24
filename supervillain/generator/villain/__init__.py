
from .site import SiteUpdate
from .link import LinkUpdate
from .exact import ExactUpdate
from .cohomology import CohomologyUpdate
from .neighborhood import NeighborhoodUpdate
from .worm import ClassicWorm as Worm

import supervillain.generator.combining as _combining

def Hammer(S, worms=1):
    r'''
    The Hammer is just syntactic sugar for a :class:`~.Sequentially` applied ergodic
    combination of generators.  It may change from version to version as new generators
    become available or get improved.

    .. note ::

        When $W=\infty$ we only include updates that leave $dn=0$ (**NOT** $\text{mod }W$!).

    .. note ::

        The :class:`~.ClassicWorm` is currently only implemented for $D=2$.  In higher
        dimensions the Hammer omits it, so the returned combination is not ergodic on its
        own until a $D>2$ worm is available.

    Parameters
    ----------

    S: a Villain action
    worms: int
        A positive integer saying how many worms to do per iteration.

    Returns
    -------

    An ergodic generator for updating Villain configurations.

    '''

    # We omit the NeighborhoodUpdate since it is a simple combination of the Vortex and CoexactUpdates.

    # The ClassicWorm is only implemented for D=2; in higher dimensions we omit it.
    if S.Lattice.D == 2:
        W = Worm(S)
        if worms > 1:
            W = _combining.KeepEvery(worms, W)
        worm = (W,)
    else:
        worm = ()

    if S.W < float('inf'):
        return _combining.Sequentially((
                SiteUpdate(S),
                LinkUpdate(S),  # <-- changes dn by W, omitted below.
                ExactUpdate(S),
                CohomologyUpdate(S),
                ) + worm)

    return _combining.Sequentially((
            SiteUpdate(S),
            ExactUpdate(S),
            CohomologyUpdate(S),
            ) + worm)
