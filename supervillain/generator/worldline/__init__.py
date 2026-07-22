
from .wrapping import WrappingUpdate
from .plaquette import PlaquetteUpdate
from .vortex import VortexUpdate
from .vortex_heatbath import VortexHeatbath
from .vortex_overrelaxation import VortexOverrelaxation
from .coexact import CoexactUpdate
from .coexact_heatbath import CoexactHeatbath
from .worm import ClassicWorm as Worm

import supervillain.generator.combining as _combining

def Hammer(S, worms=1, overrelax=3):
    r'''
    The Hammer is just syntactic sugar for a :class:`~.Sequentially` applied ergodic
    combination of generators.  It may change from version to version as new generators
    become available or get improved.

    The $v$ and coexact-$m$ moves are drawn from their exact conditionals by the
    :class:`~.VortexHeatbath` and :class:`~.CoexactHeatbath` (rejection-free replacements for
    the :class:`~.worldline.VortexUpdate` and :class:`~.CoexactUpdate`).  At $W=\infty$ the
    vortex field $v$ is continuous, so a microcanonical :class:`~.VortexOverrelaxation` is
    interleaved after its heatbath to accelerate decorrelation; at finite $W$ (discrete $v$)
    it has no reflection partner and is omitted.

    Parameters
    ----------

    S: a Worldline action
    worms: int
        A positive integer saying how many worms to do per iteration.
    overrelax: int
        How many $v$ overrelaxation sweeps (:class:`~.VortexOverrelaxation`) to interleave
        after the vortex heatbath at $W=\infty$.  Must be a positive integer ($\geq 1$);
        ignored at finite $W$.

    Returns
    -------

    An ergodic generator for updating Worldline configurations.

    '''

    # We omit the PlaquetteUpdate since it is a simple combination of the Vortex and CoexactUpdates.

    if (not isinstance(overrelax, int)) or (overrelax < 1):
        raise ValueError(f"overrelax must be a positive integer (>= 1), not {overrelax}")

    W = Worm(S)
    if worms > 1:
        W = _combining.KeepEvery(worms, W)

    # VortexOverrelaxation is only defined at W=∞, where v is continuous.
    vorx = (VortexOverrelaxation(S, applications=overrelax),) if S.W == float('inf') else ()

    return _combining.Sequentially((
            VortexHeatbath(S),
            ) + vorx + (
            CoexactHeatbath(S),
            WrappingUpdate(S),
            W,
            ))
