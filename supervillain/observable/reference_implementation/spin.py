import numpy as np
from supervillain.observable import Observable

class Spin_SpinSloppy(Observable):
    r'''

    This performs the same measurement as the non-Sloppy version but does not get all the juice out of every Worldline configuration.


    See the :class:`~.Spin_Spin` documentation for detailed descriptions.
    '''

    @staticmethod
    def Villain(S, phi):
        r'''
        The same as in the :class:`~.Spin_Spin`.
        '''

        L = S.Lattice

        exp_i_phi = np.exp(1.j * phi)

        return L.correlation(exp_i_phi, exp_i_phi)

    @staticmethod
    def Worldline(S, Links):
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
            result[i] += np.exp(-1/(2*kappa) * (P* (2*Links + P)).sum())

        return L.coordinatize(result)

class Spin_SpinSlow(Observable):
    r'''

    We can deform $Z_J \rightarrow Z_{J}[x,y]$ to include the creation of a boson at $y$ and the destruction of a boson at $x$ in the action.
    We define the expectation value

    .. math ::
        S_{x,y} = \frac{1}{Z_J} Z_J[x,y]

    and reduce to a single relative coordinate

    .. math ::
        \texttt{Spin_Spin}_{\Delta x} = S_{\Delta x} = \frac{1}{\Lambda} \sum_x S_{x,x-\Delta x}

    .. seealso::
        Compared to :class:`~.reference_implementation.spin.Spin_SpinSloppy` this implementation gets more juice from each configuration.
        In other words, for a fixed configuration their results will differ, but they will agree in expectation.

        In contrast, this observable produces the same numerical values as the production implementation :class:`~.Spin_Spin`, which is much faster.
'''

    @staticmethod
    def Villain(S, phi):
        r'''
        The same as in  :class:`~.Spin_Spin`.
        '''

        L = S.Lattice

        exp_i_phi = np.exp(1.j * phi)

        return L.correlation(exp_i_phi, exp_i_phi)


    _stencils = dict()

    @staticmethod
    def Worldline(S, Links):
        r'''
        Computes the same result as :class:`~.Spin_Spin` but more slowly.
        Compared to :class:`~.Spin_SpinSloppy` we measure the same correlator but get more juice from each configuration by averaging over translations.
        '''

        # Note: for a substantially similar but slightly simpler implementation
        # which leaves a lot of information on the table, see the Spin_SpinSloppy
        # observable.

        L = S.Lattice
        kappa = S.kappa

        result = L.linearize(L.form(0))

        # For every displacment we will take the taxicab route, as dumb as possible.
        # Just go Δt in time first and then Δx in space.
        for i, (Δt, Δx)  in enumerate(L.coordinates):
            try:
                P = Spin_SpinSlow._stencils[(L.nt, L.nx, Δt, Δx)]
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

                Spin_SpinSlow._stencils[(L.nt, L.nx, Δt, Δx)] = P

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
            result[i] = np.exp(-1/(2*kappa) * (P * (2*Links + P)).sum(
                axis=(1,2,3)    # the 0th axis is the broadcast axis, 1,2, and 3 are the vector index, time, and space.
                )).mean()       # <-- we should average over the different starting points.

        return L.coordinatize(result)


