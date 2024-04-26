from .observable import Observable, Scalar, Constrained
from .observable import OnlyVillain, NotVillain, OnlyWorldline, NotWorldline
from .derived import DerivedQuantity

from .links import Links
from .energy import InternalEnergyDensity, InternalEnergyDensitySquared, InternalEnergyDensityVariance, SpecificHeatCapacity
from .action import ActionDensity, ActionTwoPoint, Action_Action
from .winding import WindingSquared, Winding_Winding
from .wrapping import TorusWrapping, TWrapping, XWrapping
from .spin import Spin_Spin, SpinSusceptibility, SpinSusceptibilityScaled, SpinCriticalMoment
from .vortex import Vortex_Vortex, VortexSusceptibility, VortexSusceptibilityScaled

def progress(iterable, **kwargs):
    r'''
    Like `tqdm <https://tqdm.github.io/docs/tqdm/#tqdm-objects>`_, but requires the iterable.

    The default progress bar is a no-op that forwards the iterable.  You can overwrite it simply.

    .. code:: python

        from tqdm import tqdm
        import supervillain
        supervillain.observable.progress=tqdm
    '''
    return iterable
