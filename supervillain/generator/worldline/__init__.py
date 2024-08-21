
from .wrapping import WrappingUpdate
from .plaquette import PlaquetteUpdate
from .vortex import VortexUpdate
from .coexact import CoexactUpdate
from .worm import Classic as Worm

import supervillain.generator.combining as _combining

def Hammer(S):
    r'''
    The Hammer is just syntactic sugar for a :class:`~.Sequentially` applied ergodic
    combination of generators.  It may change from version to version as new generators
    become available or get improved.

    Parameters
    ----------

    S: a Worldline action

    Returns
    -------

    An ergodic generator for updating Villain configurations.

    '''

    # We omit the PlaquetteUpdate since it is a simple combination of the Vortex and CoexactUpdates.

    return _combining.Sequentially((
            VortexUpdate(S), # <-- Handles the finite-W or infinite-W case
            CoexactUpdate(S),
            WrappingUpdate(S),
            Worm(S),
            ))
