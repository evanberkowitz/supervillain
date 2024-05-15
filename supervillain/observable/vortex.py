import numpy as np
from supervillain.observable import Observable, Scalar, Constrained, NotVillain
import supervillain.action

class Vortex_Vortex(NotVillain, Constrained, Observable):
    r'''

    In the constrained model the vortex correlations are given by

    .. math ::
        V_{x,y} = \left\langle e^{2\pi i (v_x - v_y) / W}\right\rangle

    and we can use translational invariance to reduce to a single relative coordinate

    .. math ::
        \texttt{Vortex_Vortex}_{\Delta x} = V_{\Delta x} = \frac{1}{\Lambda} \sum_x V_{x,x-\Delta x}
    '''

    @staticmethod
    def Worldline(S, v):
        r'''
        $v$ is accessible only in the Worldline formulation.
        '''

        L = S.Lattice

        vortex = np.exp(2j*np.pi * v / S.W)

        return L.correlation(vortex, vortex)

    @staticmethod
    def CriticalScalingDimension(W):
        r'''
        The critical scaling dimension of the winding-$w$ operator is $R^2 w^2 / 2$.
        With the constraint that only charge $W$ vortices are allowed as propagating excitations,
        at the phase transition $R^2 W^2 / 2 = 2$, so that the critical $R=2/W$.

        What we want to know is the scaling dimension of the $w=1$ operator, which at the phase transition is $\Delta = (1R)^2/2 = 2/W^2$.

        This is the critical scaling dimension of a *single* insertion, so the two-point :class:`~.Vortex_Vortex` scales with twice this dimension at the critical point.
        '''

        return 2/W**2



class VortexSusceptibility(NotVillain, Constrained, Scalar, Observable):
    r'''
    The *vortex susceptibility* is the spacetime integral of the :class:`~.Vortex_Vortex` correlator $V_{\Delta x}$,

    .. math::

        \texttt{VortexSusceptibility} = \chi_V = \int d^2r\; V(r).
    '''

    @staticmethod
    def default(S, Vortex_Vortex):
        return np.sum(Vortex_Vortex.real)


class VortexSusceptibilityScaled(VortexSusceptibility):
    r'''
    At the critical point and in the CFT the :class:`~.VortexSusceptibility` has a known expected scaling that comes from the scaling dimension $\Delta$ of $e^{2\pi i v/W}$.

    .. math ::

        \chi_V \sim L^{2-2\Delta(\kappa, W)}

    where the scaling at the critical coupling $\kappa_c$ is known and depends on the constraint integer $W$.

    So, we scale the susceptibility,

    .. math::

        \texttt{VortexSusceptibilityScaled} = \chi_V / L^{2-2\Delta(\kappa_c, W)}

    so that at the critical coupling the infinite-volume limit of :class:`~.VortexSusceptibilityScaled` will be a constant.

    .. note::
        The 2 depends on being in 2 dimensions, while the $2\Delta$ comes from the fact that the :class:`~.Vortex_Vortex` correlator is a two-point function.
    '''

    @staticmethod
    def default(S, VortexSusceptibility):

        L = S.Lattice.nx
        # NOTE: implicitly assumes that the lattice is square!
        return VortexSusceptibility / L**(2-2*supervillain.observable.Vortex_Vortex.CriticalScalingDimension(S.W))

class VortexCriticalMoment(NotVillain, Constrained, Scalar, Observable):
    r'''
    The *critical moment* of the vortex correlator :math:`C_V` is the volume-average of the correlator multiplied by its long-distance critical behavior,

    .. math::
        C_V = \frac{1}{L^2} \int d^2r\; r^{2\Delta_V(\kappa_c, W)}\; V(r)

    At the critical $\kappa$ the long-distance behavior of the :class:`~.Vortex_Vortex` correlator :math:`V` decays with exactly the required power to cancel the explicit power of $r$ and the integral cancels the normalization, giving 1 in the large-$L$ limit.

    In the gapped phase $V$ decays exponentially with $r$ and the integral converges, so $C_V$ goes to 0 in the large-$L$ limit.

    In the CFT, $V$ decays polynomially, but faster than the weight from the moment grows.  The integral scales with a power less than 2 and $C_V$ again goes to 0 in the large-$L$ limit.
    '''
 
    @staticmethod
    def default(S, Vortex_Vortex):

        L = S.Lattice
        return np.sum(L.R_squared**(supervillain.observable.Vortex_Vortex.CriticalScalingDimension(S.W)) * Vortex_Vortex.real) / L.sites
