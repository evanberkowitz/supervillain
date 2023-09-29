import numpy as np
from supervillain.observable import Observable, DerivedQuantity

_stencils = dict()

class SloppySpin_Spin(Observable):
    r'''

    This performs the same measurement as the non-Sloppy version but does not get all the juice out of every Worldline configuration.


    See the :class:`~.Spin_Spin` documentation for detailed descriptions.
    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        The same as in the :class:`~.Spin_Spin`.
        '''

        L = S.Lattice

        exp_i_phi = np.exp(1.j * phi)

        return L.correlation(exp_i_phi, exp_i_phi)

    @staticmethod
    def Worldline(S, m):
        r'''

        See the :class:`Spin_Spin` documentation.
        '''

        L = S.Lattice
        kappa = S.kappa

        result = L.linearize(L.form(0))

        # For every displacment we will take the taxicab route, as dumb as possible.
        # Just go Δt in time first and then Δx in space.
        for i, (Δt, Δx)  in enumerate(L.coordinates):
            P = L.form(1)

            if Δt >= 0:
                # Follow the links in the positive t direction.
                # Therefore increment P by 1
                P[0][:Δt,0] = +1
            else:
                # Follow the links in the negative t direction.
                # Therefore decrement P by 1
                P[0][Δt:,0] = -1

            if Δx >= 0:
                # Follow the links in the positive x direction.
                P[1][Δt,:Δx] = +1
            else:
                # Follow the links in the negative x direction.
                P[1][Δt,Δx:] = -1

            # The difference between the Sloppy and full versions is that here we only
            # overlay the stencil on the configuration one time.  That always puts the
            # defect at the absolute origin and (Δt, Δx), as opposed to summing over all
            # the possible origins.
            result[i] += np.exp(-1/(2*kappa) * (P* (2*m + P)).sum())

        return L.coordinatize(result)

class Spin_Spin(Observable):
    r'''

    We can deform $Z_J \rightarrow Z_{J}[x,y]$ to include the creation of a boson at $y$ and the destruction of a boson at $x$ in the action.
    We define the expectation value

    .. math ::
        S_{x,y} = \frac{1}{Z_J} Z_J[x,y]

    and reduce to a single relative coordinate

    .. math ::
        \texttt{Spin_Spin}_{\Delta x} = S_{\Delta x} = \frac{1}{\Lambda} \sum_x S_{x,x-\Delta x}

    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        In the :class:`~.Villain` formulation the correlator is just

        .. math ::
            S_{xy} = \left\langle e^{i(\phi_x - \phi_y)} \right\rangle


        '''

        L = S.Lattice

        exp_i_phi = np.exp(1.j * phi)

        return L.correlation(exp_i_phi, exp_i_phi)

    @staticmethod
    def Worldline(S, m):
        r'''
        By starting in the Villain formulation with a modified action

        .. math ::
            S' = \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_{\ell}^2 + i \sum_p J_p dn_p + i \phi_x - i \phi_y

        we can Poisson resum $n \rightarrow m$ as usual but the offset by the operator changes the constraint.
        Rather than requiring $\delta m = 0$ everywhere we get $(\delta m)_z = \delta_{y,z} - \delta_{x,z}$,

        .. math ::
           \begin{align}
               Z_J[x,y] &= \sum Dm\; e^{-S_J[m]} \left[\delta m = 0 \text{ not on }x, y\right]\left[(\delta m)_x = -1 \right]\left[(\delta m)_y = +1 \right]
               \\
               S_J[m] &= \frac{1}{2\kappa} \sum_\ell \left(m - \frac{\delta J}{2\pi}\right)_\ell^2 + \frac{|\ell|}{2} \ln (2\pi \kappa) - |x| \ln 2\pi
           \end{align}

        Now define $\hat{m}_\ell = m_{\ell} - [P_{xy}]_\ell$ where $P_{xy}$ traces any fixed path at all whatsoever from $x$ to $y$ and on any link $P$ accumulates $+1$ for every time the path traces along the link and $-1$ every time the path traces against the link.
        For sites visited in the middle of the path $P$ the constraint is maintained while at the endpoints it is violated in exactly the desired way.

        Then we can change the integration variables from $m$ to $\hat{m}$ as long as we also change the action,

        .. math ::
           \begin{align}
               Z_J[x,y] &= \sum D\hat{m}\; e^{-S_J[\hat{m} + P_{xy}]} \left[\delta \hat{m} = 0 \text{ not on }x, y\right]\left[(\delta \hat{m})_x = 0 \right]\left[(\delta \hat{m})_y = 0 \right]
           \end{align}

        with the same $S_J$.  Since in the hatted variables the constraint is satisfied, we can calculate this using constraint-obeying configurations and measuring the operator

        .. math ::

            \hat{V}_{xy} = \exp{\left[ - \frac{1}{2\kappa} \sum_{\ell \in P_{xy}} \left\{(\hat{m} + P_{xy})_\ell^2 - \hat{m}_\ell^2 \right\}\right]}

        .. note ::
            The actual path $P_{xy}$ used is irrelevant in expectation, though of course on a fixed configuration you get different measurements if you pick different paths.
            The point is that any other path can be reached by making a combination of :class:`~.PlaquetteUpdate`\s and :class:`~.HolonomyUpdate`\s.

        '''

        # Note: for a substantially similar but slightly simpler implementation
        # which leaves a lot of information on the table, see the SloppySpin_Spin
        # observable.

        L = S.Lattice
        kappa = S.kappa

        result = L.linearize(L.form(0))

        # For every displacment we will take the taxicab route, as dumb as possible.
        # Just go Δt in time first and then Δx in space.
        for i, (Δt, Δx)  in enumerate(L.coordinates):
            try:
                P = _stencils[(L.nt, L.nx, Δt, Δx)]
            except KeyError:
                # Each stencil will hold the path starting on the origin, AND a copy
                # for every other starting point.
                #
                # We construct the one from the origin first, as it is easiest to think about...
                P = L.form(1, L.sites)

                if Δt >= 0:
                    # Follow the links in the positive t direction.
                    # Therefore increment P by 1
                    P[0, 0][:Δt,0] = +1
                else:
                    # Follow the links in the negative t direction.
                    # Therefore decrement P by 1
                    P[0, 0][Δt:,0] = -1

                if Δx >= 0:
                    # Follow the links in the positive x direction.
                    P[0, 1][Δt,:Δx] = +1
                else:
                    # Follow the links in the negative x direction.
                    P[0, 1][Δt,Δx:] = -1

                # ... and then we just roll it around and store every possible translation.
                for j, shift in enumerate(L.coordinates):
                    P[j] = L.roll(P[0], shift)

                _stencils[(L.nt, L.nx, Δt, Δx)] = P

            # In the full measurement we overlay the stencil on the configuration 
            # in all possible ways and sum.  Then we have measured the dependence on Δx
            # as efficiently as possible for each configuration, summing each displacement
            # over all possible starting points.
            #
            # Since we have stored every translation, we can use the numpy broadcasting rules
            #
            #    https://numpy.org/doc/stable/user/basics.broadcasting.html
            #
            # to eliminate some python for loops, which were explicit in previous implementations.
            result[i] = np.exp(-1/(2*kappa) * (P * (2*m + P)).sum(
                axis=(1,2,3)    # the 0th axis is the broadcast axis, 1,2, and 3 are the vector index, time, and space.
                )).mean()       # <-- we should average over the different starting points.

        return L.coordinatize(result)


class SpinSusceptibility(Observable):
    r'''
    The *spin susceptibility* is the spacetime integral of the :class:`~.Spin_Spin` correlator $S_{\Delta x}$,

    .. math::
        
        \texttt{SpinSusceptibility} = \chi_S = \int d^2r\; S(r).
    '''

    @staticmethod
    def default(S, Spin_Spin):
        return np.sum(Spin_Spin.real)
    
def _CriticalSpinScalingDimension(W):
    r'''
    W is the constraining integer which controls the allowed vorticity.
    '''
    # TODO: cache?
    # TODO: W != 1
    if W == 1:  # The BKT case, Δ = 1/8
        return 0.125

    else:
        raise NotImplementedError(f'The constrained W≠1 scaling is not yet implemented so {W=} cannot be computed.')

class SpinSusceptibilityScaled(Observable):
    r'''
    At the critical point and in the CFT the :class:`~.SpinSusceptibility` has a known expected scaling that comes from the scaling dimension $\Delta$ of $e^{i\phi}$

    .. math::
        
        \chi_S \sim L^{2-2\Delta(\kappa)}.

    where the scaling dimension at the critical coupling $\kappa_c$ is known and depends on the constraint integer $W$.

    So, we scale the susceptibility,

    .. math::
        \texttt{SpinSusceptibilityScaled} = \chi_S / L^{2-2\Delta(\kappa_c)}

    so that at the critical coupling the infinite-volume limit of :class:`~.SpinSusceptibilityScaled` will be a constant.

    .. note::
        The 2 depends on being in 2 dimensions, while the $2\Delta$ comes from the fact that the :class:`~.Spin_Spin` is a two-point function.
    '''

    @staticmethod
    def default(S, SpinSusceptibility):

        L = S.Lattice.nx
        # NOTE: implicitly assumes that the lattice is square!
        # TODO: Since we don't currently have any constraint implemented we hard-code W=1.
        return SpinSusceptibility / L**(2-2*_CriticalSpinScalingDimension(1))

