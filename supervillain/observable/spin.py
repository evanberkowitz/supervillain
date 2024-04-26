import numpy as np
from supervillain.observable import Scalar, Observable
import supervillain.action

class Spin_Spin(Observable):
    r'''

    We can deform $Z_J \rightarrow Z_{J}[x,y]$ to include the creation of a boson at $y$ and the destruction of a boson at $x$ in the action.
    We define the expectation value

    .. math ::
        S_{x,y} = \frac{1}{Z_J} Z_J[x,y]

    and reduce to a single relative coordinate

    .. math ::
        \texttt{Spin_Spin}_{\Delta x} = S_{\Delta x} = \frac{1}{\Lambda} \sum_x S_{x,x-\Delta x}

    .. seealso:: 

        The Worldline formulation of this observable is the trickiest observable we supply,
        and its implementation is nontrivial.

        If you need a simpler-to-understand implementation see the reference implementation :class:`~.reference_implementation.spin.Spin_SpinSlow`.
    '''

    @staticmethod
    def Villain(S, phi):
        r'''
        In the :class:`~.Villain` formulation the correlator is just

        .. math ::
            S_{xy} = \left\langle e^{i(\phi_x - \phi_y)} \right\rangle


        '''

        L = S.Lattice

        spin = np.exp(1.j * phi)

        return L.correlation(spin, spin)

    _signs = dict()
    _directions = dict()
    _coordinates = dict()


    @staticmethod
    def Worldline(S, Links):
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

        with the same $S_J$.  Since in the hatted variables the constraint is satisfied, we can calculate this using constraint-obeying configurations (setting $J=2\pi v/W$ for the constraint) and measuring the operator

        .. math ::

            \hat{V}_{xy} = \exp{\left[ - \frac{1}{2\kappa} \sum_{\ell \in P_{xy}} \left\{(\hat{m} - \delta v / W + P_{xy})_\ell^2 - (\hat{m} - \delta v / W)_\ell^2 \right\}\right]}

        which is what we need to reweight to sampling according to $S[\hat{m}]$ with no defect.

        .. note ::
            The actual path $P_{xy}$ used is irrelevant in expectation, though of course on a fixed configuration you get different measurements if you pick different paths.
            An implementation detail is that the fixed chosen path is the taxicab path that first covers the whole time separation and then the whole space separation.
            The point is that any other path can be reached by making a combination of :class:`~.PlaquetteUpdate`\s and :class:`~.WrappingUpdate`\s.

        '''

        # Note: for a substantially similar but slower implementation see the Spin_SpinSlow observable.

        L = S.Lattice
        kappa = S.kappa

        result = L.form(0)

        # For every displacment we will take the taxicab route, as dumb as possible.
        # In Spin_SpinSlow we create stencils that are 0 and ±1, one value for every link.
        # Then we multiply m by the stencil, which selects links on the taxicab route, and sum.
        #
        # That's a perfectly correct algorithm, but the issue is there is a lot of wasted effort.
        # For every displacement we have to do a whole volume's worth of multiplications, while
        # most of the products, not being on the taxicab route, are zero.  A lot of wasted effort.
        # In fact, the amount of waste is like L^2 while the amount of true work is L.  So the waste
        # is bad enough that we worsen the scaling of the algorithm.
        #
        # Instead, let's just take the links we need and evaluate ΔS.
        #
        # Just go Δt in time first and then Δx in space.
        for i, (Δt, Δx)  in enumerate(L.coordinates):
            if (Δt, Δx) == (0, 0):
                result[0,0] = 1
                continue

            key = (L.nt, L.nx, Δt, Δx)
            T = np.abs(Δt)
            X = np.abs(Δx)
            length = T+X

             # The main idea is that the change in action on an included link needs
            # a sign (depending on the orientation the link is traced), 
            # the coordinates where the link is stored, and
            # the direction the link is pointing.
           
            try:
                # We can use the precomputed path if we have it.
                sign = Spin_Spin._signs[key]
                coordinates = Spin_Spin._coordinates[key]
                direction = Spin_Spin._directions[key]
            except KeyError:
                # Otherwise we need to figure it out.
                # The sign and direction are both 1 number per included link.
                # The coordinates are two because we are two dimensions.
                sign = np.zeros(length, dtype=int)
                direction = np.zeros(length, dtype=int)
                coordinates = np.zeros((length, L.dim), dtype=int)
                
                if Δt > 0:
                    # If we trace along a link it counts towards ΔS.
                    sign[:Δt] = +1
                    # For the taxicab route we go in time first.
                    direction[:Δt] = 0
                    # The first steps are just sequential steps in time along the x-axis.
                    coordinates[:Δt, 0] = np.arange(Δt)
                    # The the remaining steps don't change the time at all.
                    coordinates[Δt:, 0] = Δt
                elif Δt < 0:
                    # If we trace against a link it counts against ΔS.
                    direction[:-Δt] = 0
                    # For the taxicab route we go in time first.
                    coordinates[:-Δt, 0] = np.arange(-1,Δt-1,-1)
                    # The first steps are just sequential steps in time along the x-axis.
                    coordinates[-Δt:, 0] = Δt
                    # The the remaining steps don't change the time at all.
                    sign[:-Δt] = -1
                
                if Δx > 0:
                    # If we trace along a link it counts towards ΔS.
                    sign[T:] = +1
                    # After all the temporal steps we take spatial steps.
                    direction[T:] = 1
                    # The spatial steps are just off of the t-axis.
                    coordinates[T:, 1] = np.arange(Δx)
                elif Δx < 0:
                    # If we trace a against link it counts against ΔS.
                    sign[T:] = -1
                    # After all the temporal steps we take spatial steps.
                    direction[T:] = 1
                    # The spatial steps are just off of the t-axis.
                    coordinates[T:, 1] = np.arange(-1,Δx-1,-1)

                # Now something a bit tricky.  We want to average over all possible starting locations.
                # We can do that by adding to the coordinates of the path every starting location in L.coordinates
                # and then modding back into the lattice.
                #
                # To avoid writing a python for loop, however, we manually broadcast.
                coordinates = L.mod(np.broadcast_to(coordinates, (L.sites,length, L.dim)).transpose((1,0,2)) + L.coordinates).T
                # The directions don't change but will be broadcast together in the indexing into m.
                direction  = np.broadcast_to(direction, (L.sites, length))

                Spin_Spin._signs[key] = sign
                Spin_Spin._directions[key] = direction
                Spin_Spin._coordinates[key] = coordinates

            # Rather than compute (m+P)^2 - m^2 we can save some arithmetic by opening up the parens.
            #
            #   (m+P)^2 - m^2 = 2Pm + P^2
            #
            # (m really means m - δv/W in the constrained case.)
            # On our non-looping taxicab route P^2 = |P|,
            Psq = T+X
            # because P is ±1 on every nonzero link.
            #
            # We can use the threaded indexing in numpy to pull out only the needed links
            # in the correct directions at the correct coordinates, and account for them with the
            # appropriate signs.
            Pm = (sign * Links[(direction, *coordinates)]).sum(axis=1)
            #
            # We summed over the links but we still have the volume averaging to accomplish.
            # However, the averaging has to be of the observable, meaning that we have to
            # average AFTER computing the reweighting factor,
            result[Δt,Δx]= np.exp(-1/(2*kappa) * (2*Pm + Psq)).mean(axis=0)

        return result

    @staticmethod
    def CriticalScalingDimension(W):
        r'''
        Setting the scaling dimension $(WR)^2 / 2$ of a charge-W vortex operator to 2 yields $R=2/W$.
        The corresponding scaling dimension of the spin operator $e^{i\phi}$ is $\Delta = (1R)^{-2}/2 = W^2/8$.

        This is the critical scaling dimension of a *single* insertion, so the two-point :class:`~.Spin_Spin` scales with twice this dimension at the critical point.
        '''

        return W**2 / 8


class SpinSusceptibility(Scalar, Observable):
    r'''
    The *spin susceptibility* is the spacetime integral of the :class:`~.Spin_Spin` correlator $S_{\Delta x}$,

    .. math::
        
        \texttt{SpinSusceptibility} = \chi_S = \int d^2r\; S(r).
    '''

    @classmethod
    def autocorrelation(cls, ensemble):
        r'''
        As it currently stands, even though this is a scalar observable, in the Worldline case
        the measurement cost is very high.  Since in all cases we've seen so far it fluctuates quickly
        compared to the slower observables, it is okay to omit to save computational time.

        Once we have a worm algorithm that measures the :class:`~.Spin_Spin` correlator on the fly, we can restore that case.
        '''
        
        if isinstance(ensemble.Action, supervillain.action.Worldline):
            return False
        
        return True

    @staticmethod
    def default(S, Spin_Spin):
        return np.sum(Spin_Spin.real)
    
class SpinSusceptibilityScaled(SpinSusceptibility):
    r'''
    At the critical point and in the CFT the :class:`~.SpinSusceptibility` has a known expected scaling that comes from the scaling dimension $\Delta$ of $e^{i\phi}$

    .. math::
        
        \chi_S \sim L^{2-2\Delta(\kappa)}.

    where the scaling dimension at the critical coupling $\kappa_c$ is known and depends on the constraint integer $W$.

    So, we scale the susceptibility by the :py:meth:`~.Spin_Spin.CriticalScalingDimension`,

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
        return SpinSusceptibility / L**(2-2*Spin_Spin.CriticalScalingDimension(S.W))

