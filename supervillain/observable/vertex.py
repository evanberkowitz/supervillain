import numpy as np
from supervillain.observable import Observable

_stencils = dict()

class SloppyVertex_Vertex(Observable):
    r'''

    This performs the same measurement as the non-Sloppy version but does not get all the juice out of every Worldline configuration.


    See the :class:`~.Vertex_Vertex` documentation for detailed descriptions.
    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        The same as in the :class:`~.Vertex_Vertex`.
        '''

        L = S.Lattice

        exp_i_phi = np.exp(1.j * phi)

        return L.correlation(exp_i_phi, exp_i_phi)

    @staticmethod
    def Worldline(S, m):
        r'''

        See the :class:`Vertex_Vertex` documentation.
        '''

        L = S.Lattice
        kappa = S.kappa

        result = L.linearize(L.form(0))

        # For every displacment we will take the taxicab route, as dumb as possible.
        # Just go Δt in time first and then Δx in space.
        for i, (Δt, Δx)  in enumerate(L.coordinates):
            try:
                P = _stencils[(L.nt, L.nx, Δt, Δx)]
            except KeyError:
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

                _stencils[(L.nt, L.nx, Δt, Δx)] = P

            # The difference between the Sloppy and full versions is that here we only
            # overlay the stencil on the configuration one time.  That always puts the
            # defect at the absolute origin and (Δt, Δx), as opposed to summing over all
            # the possible origins.
            result[i] += np.exp(-1/(2*kappa) * (P* (2*m + P)).sum())

        return L.coordinatize(result)

class Vertex_Vertex(Observable):
    r'''

    We can deform $Z_J \rightarrow Z_{J}[x,y]$ to include the creation of a boson at $y$ and the destruction of a boson at $x$ in the action.
    We define the expectation value

    .. math ::
        V_{x,y} = \frac{1}{Z_J} Z_J[x,y]

    and reduce to a single relative coordinate

    .. math ::
        \texttt{Vertex_Vertex}_{\Delta x} = \frac{1}{\Lambda} \sum_x V_{x,x-\Delta x}

    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        In the :class:`~.Villain` formulation the correlator is just

        .. math ::
            V_{xy} = \left\langle e^{i(\phi_x - \phi_y)} \right\rangle


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
        # which leaves a lot of information on the table, see the SloppyVertex_Vertex
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

                _stencils[(L.nt, L.nx, Δt, Δx)] = P

            # In the full measurement we overlay the stencil on the configuration 
            # in all possible ways and sum.  Then we have measured the dependence on Δx
            # as efficiently as possible for each configuration, summing each displacement
            # over all possible starting points.
            for shift in L.coordinates:
                # Warning: adds an extra factor of the volume in cost at least!
                p = L.roll(P, shift)
                result[i] += np.exp(-1/(2*kappa) * (p * (2*m + p)).sum())

        return L.coordinatize(result) / L.sites
