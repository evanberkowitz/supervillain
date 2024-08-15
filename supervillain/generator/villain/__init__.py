
from .site import SiteUpdate
from .link import LinkUpdate
from .exact import ExactUpdate
from .holonomy import HolonomyUpdate
from .neighborhood import NeighborhoodUpdate
from .worm import Classic as Worm

import supervillain.generator.combining as _combining

def Hammer(S):
    r'''
    The Hammer is just syntactic sugar for a :class:`~.Sequentially` applied ergodic
    combination of generators.  It may change from version to version as new generators
    become available or get improved.

    Parameters
    ----------

    S: a Villain action

    Returns
    -------

    An ergodic generator for updating Villain configurations.

    '''
    return _combining.Sequentially((
            SiteUpdate(S),
            LinkUpdate(S),
            ExactUpdate(S),
            HolonomyUpdate(S),
            Worm(S),
            ))
