import numpy as np
from supervillain.observable import Observable, Scalar, Constrained, NotVillain
import supervillain.action

class Vortex_Vortex(Constrained, Observable):
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

    _change_n = dict()
    _changed_links = dict()

    @staticmethod
    def Villain(S, Links):
        r'''
        We can write $V_{x,y}$ as the ratio of two partition functions,

        .. math ::
            \begin{align}
                V_{x,y} &= Z_0[x,y] / Z_0
                \\
                Z_J[x,y] &= \sum\hspace{-1.33em}\int D\phi\; Dn\; Dv\; e^{-S_J[\phi, n, v]} e^{2\pi i (v_x - v_y) / W}
                \\
                S_J[\phi, n, v] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + 2\pi i \sum_p \left(v/W + J/2\pi \right)_p (dn)_p
            \end{align}

        where $Z$ is the standard :class:`~.Villain` partition function with action $S$.
        The difference between $Z[x,y]$ and $Z$ is the insertion of the two-point exponential.
        We can absorb that difference into the $v$ term in $S$,

        .. math ::
            
            \begin{align}
                S_0'[\phi, n, v]
                    &=
                \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + 2\pi i \sum_p \left(v/W\right)_p (dn)_p + 2\pi i (v_x-v_y)/W
                    \\
                &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + 2\pi i \sum_p \left(v/W\right)_p (dn_p + \delta_{xp} - \delta_{yp} )
            \end{align}

        so integrating out $v$ would now change the :ref:`winding constraint <winding constraint>` to $[dn_p \equiv \delta_{yp} - \delta_{xp} \text{ mod } W]$.

        Now define $\hat{n} = n_{\ell} - [P_{xy}]_{\ell}$ for $n$ satisfying the modified constraint and $P$ defined as follows.
        Let $[\mathcal{P}_{xy}]_{\tilde{\ell}}$ trace any fixed path at all whatsoever on the dual lattice from $\star y$ to $\star x$, accumulating +1 when it traces along a dual link and -1 when it traces against a dual link.
        Then $[P_{xy}]_{\star\tilde{\ell}} = [\star\mathcal{P}_{xy}]_{\tilde{\ell}}$ and $\hat{n}$ satisfies the original constraint.

        Then we can change the integration variables from $n$ to $\hat{n}$ as long as  we also change the action,

        .. math :: 
            \begin{align}
                Z_J[x,y] &= \sum\hspace{-1.33em}\int D\phi\; D\hat{n}\; e^{-S_J[\phi, \hat{n} + P]} [d\hat{n} \equiv 0 \text{ mod } W]
            \end{align}

        with the same $S_J$.  Since in the hatted variables the constraint is satisfied, we can calculate this using constraint-obeying configurations by reweighting,

        .. math :: 

            \begin{align}
                Z_J[x,y] &= \sum\hspace{-1.33em}\int D\phi\; D\hat{n}\; e^{-S_J[\phi, \hat{n}]} e^{-(S_J[\phi, \hat{n} + P]-S_J[\phi, \hat{n}])} [d\hat{n} \equiv 0 \text{ mod } W]
            \end{align}

        Or, in other words, we measure

        .. math ::

             \begin{align}
                V_{x,y} &= \left\langle \hat{V}_{x,y} = e^{-(S_J[\phi, \hat{n} + P_{xy}]-S_J[\phi, \hat{n}])} \right\rangle
            \end{align}

        in our standard ensemble.
        '''

        L = S.Lattice
        correlator = L.form(0)

        for Δt, Δx in L.coordinates:
            if (Δt, Δx) == (0, 0):
                correlator[0,0] = 1
                continue

            key = (L.nt, L.nx, Δt, Δx)

            try:
                change_n = Vortex_Vortex._change_n[key]
                changed_links = Vortex_Vortex._changed_links[key]
            except KeyError:

                length = np.abs([Δt, Δx]).sum()

                stencil = L.form(1, dtype=int)

                if Δt > 0:
                    stencil[1][:Δt, 0] = +1
                elif Δt < 0:
                    stencil[1][Δt:, 0] = -1
                stencil[1] = L.roll(stencil[1], (1, 0))

                if Δx > 0:
                    stencil[0][Δt, :Δx] = -1
                elif Δx < 0:
                    stencil[0][Δt, Δx:] = +1
                stencil[0] = L.roll(stencil[0], (0, 1))

                big_change = np.stack([L.roll(stencil, x0) for x0 in L.coordinates])
                changed_links = np.where(big_change)
                changed_links = tuple(c.reshape(-1, length) for c in changed_links)
                change_n = big_change[changed_links]
                changed_links = changed_links[1:]

                Vortex_Vortex._change_n[key] = change_n
                Vortex_Vortex._changed_links[key] = changed_links

            dS = -2*np.pi*S.kappa * ((change_n * Links[changed_links]).sum(axis=1) - np.pi*change_n.shape[1])
            correlator[Δt, Δx] = np.exp(-dS).mean(axis=0)

        return correlator


class VortexSusceptibility(Constrained, Scalar, Observable):
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

class VortexCriticalMoment(Constrained, Scalar, Observable):
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
