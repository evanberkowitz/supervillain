
from .site import SiteUpdate
from .site_heatbath import SiteHeatbath
from .site_fourier_heatbath import FourierSiteHeatbath
from .site_overrelaxation import SiteOverrelaxation
from .link import LinkUpdate
from .link_heatbath import LinkHeatbath
from .exact import ExactUpdate
from .exact_heatbath import ExactHeatbath
from .cohomology import CohomologyUpdate
from .cohomology_heatbath import CohomologyHeatbath
from .neighborhood import NeighborhoodUpdate
from .worm import ClassicWorm as Worm
from .cluster import ReflectionCluster

import supervillain.generator.combining as _combining

def Hammer(S, worms=1):
    r'''
    The Hammer is just syntactic sugar for a :class:`~.Sequentially` applied ergodic
    combination of generators.  It may change from version to version as new generators
    become available or get improved.

    .. note ::

        When $W=\infty$ we only include updates that leave $dn=0$ (**NOT** $\text{mod }W$!).

    .. note ::

        The :class:`~supervillain.generator.villain.worm.ClassicWorm` is currently only
        implemented for $D=2$.  In higher dimensions the Hammer omits it, so the returned
        combination is ergodic but could be very slow to update the torus-wrapping modes.

    Parameters
    ----------

    S: a Villain action
    worms: int
        A positive integer saying how many worms to do per iteration.

    Returns
    -------

    An ergodic generator for updating Villain configurations.

    '''

    # We omit the NeighborhoodUpdate since it is a simple combination of the SiteUpdate and ExactUpdate.

    # The ClassicWorm is only implemented for D=2; in higher dimensions we omit it.
    if S.Lattice.D == 2:
        W = Worm(S)
        if worms > 1:
            W = _combining.KeepEvery(worms, W)
        worm = (W,)
    else:
        worm = ()

    # φ is drawn by the FourierSiteHeatbath rather than by a SiteHeatbath sweep followed by
    # SiteOverrelaxation: at fixed n the *joint* conditional of φ is Gaussian and diagonal in
    # Fourier space, so the whole field is drawn at once and successive φ are independent.
    # Nothing is then left for an overrelaxation to decorrelate, which is why `overrelax` is
    # gone rather than merely defaulted — passing it now raises loudly instead of silently
    # doing nothing.
    if S.W < float('inf'):
        return _combining.Sequentially((
                FourierSiteHeatbath(S),
                LinkHeatbath(S),  # <-- changes dn by W, omitted below.
                ExactHeatbath(S),
                CohomologyHeatbath(S),
                ) + worm)

    return _combining.Sequentially((
            FourierSiteHeatbath(S),
            ExactHeatbath(S),
            CohomologyHeatbath(S),
            ) + worm)
