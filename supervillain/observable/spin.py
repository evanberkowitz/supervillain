import numpy as np
from supervillain.observable import Scalar, Observable, DerivedQuantity, OnlyVillain
import supervillain.action

class Spin_Spin(Observable):
    r'''

    We can deform $Z_J \rightarrow Z_{J}[x,y]$ to include the creation of a boson at $y$ and the destruction of a boson at $x$ in the action.
    We define the expectation value

    .. math ::
        S_{x,y} = \frac{1}{Z_J} Z_J[x,y]

    and reduce to a single relative coordinate

    .. math ::
        \texttt{Spin\_Spin}_{\Delta x} = S_{\Delta x} = \frac{1}{\Lambda} \sum_x S_{x,x-\Delta x}

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

        spin = np.exp(1.j * phi[0])

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
           \begin{aligned}
               Z_J[x,y] &= \sum Dm\; e^{-S_J[m]} \left[\delta m = 0 \text{ not on }x, y\right]\left[(\delta m)_x = -1 \right]\left[(\delta m)_y = +1 \right]
               \\
               S_J[m] &= \frac{1}{2\kappa} \sum_\ell \left(m - \frac{\delta J}{2\pi}\right)_\ell^2 + \frac{|\ell|}{2} \ln (2\pi \kappa) - |x| \ln 2\pi
           \end{aligned}

        Now define $\hat{m}_\ell = m_{\ell} - [P_{xy}]_\ell$ where $P_{xy}$ traces any fixed path at all whatsoever from $x$ to $y$ and on any link $P$ accumulates $+1$ for every time the path traces along the link and $-1$ every time the path traces against the link.
        For sites visited in the middle of the path $P$ the constraint is maintained while at the endpoints it is violated in exactly the desired way.

        Then we can change the integration variables from $m$ to $\hat{m}$ as long as we also change the action,

        .. math ::
           \begin{aligned}
               Z_J[x,y] &= \sum D\hat{m}\; e^{-S_J[\hat{m} + P_{xy}]} \left[\delta \hat{m} = 0 \text{ not on }x, y\right]\left[(\delta \hat{m})_x = 0 \right]\left[(\delta \hat{m})_y = 0 \right]
           \end{aligned}

        with the same $S_J$.  Since in the hatted variables the constraint is satisfied, we can calculate this using constraint-obeying configurations (setting $J=2\pi v/W$ for the constraint) and measuring the operator

        .. math ::

            \hat{S}_{xy} = \exp{\left[ - \frac{1}{2\kappa} \sum_{\ell \in P_{xy}} \left\{(\hat{m} - \delta v / W + P_{xy})_\ell^2 - (\hat{m} - \delta v / W)_\ell^2 \right\}\right]}

        which is what we need to reweight to sampling according to $S[\hat{m}]$ with no defect.

        .. note ::
            The actual path $P_{xy}$ used is irrelevant in expectation, though of course on a fixed configuration you get different measurements if you pick different paths.
            An implementation detail is that the fixed chosen path is the taxicab path that first covers the whole time separation and then the whole space separation.
            The point is that any other path can be reached by making a combination of :class:`~.PlaquetteUpdate`\s and :class:`~.WrappingUpdate`\s.

        Clearly $S_{xx}=1$, and we can normalize so that $\texttt{Spin\_Spin}_{\Delta x = 0} = 1$.
        The method provided in this observable are already naturally normalized.
        However, inline measurements like those provided by the :class:`worm <supervillain.generator.worldline.worm.Classic>` are not,
        and can only be normalized *after* the bootstrap, which is why anything that depends on this observable is a :class:`~.DerivedQuantity`.

        .. note ::
            This taxicab-path measurement is only implemented for $D=2$ and raises ``NotImplementedError``
            otherwise.  The :class:`~.Villain` measurement and the inline worm histogram are
            dimension-general and work in any $D$.
        '''

        if S.Lattice.D != 2:
            raise NotImplementedError(
                'The Worldline Spin_Spin measurement traces a taxicab path on the lattice and is '
                'only implemented for D=2.  (In the Villain formulation Spin_Spin is measured '
                'directly and works in any D.)'
            )

        # Note: for a substantially similar but slower implementation see the Spin_SpinSlow observable.

        L = S.Lattice
        kappa = S.kappa

        result = np.zeros(L.dims)

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
                result[L.origin] = 1
                continue

            # Include D in the key so caches for different dimensions never collide
            # (the measurement is D=2-only today, but the key should not assume it).
            key = (L.D, L.N, Δt, Δx)
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
    def CriticalScalingDimension(S):
        r'''
        Setting the scaling dimension $(WR)^2 / 2$ of a charge-W vortex operator to 2 yields $R=2/W$.
        The corresponding scaling dimension of the spin operator $e^{i\phi}$ is $\Delta = (1R)^{-2}/2 = W^2/8$.

        This is the critical scaling dimension of a *single* insertion, so the two-point :class:`~.Spin_Spin_Normalized` scales with twice this dimension at the critical point.

        When $W=\infty$ every $\kappa>0$ is critical and $\Delta_S = 2/R^2 = 2/2\pi \kappa = 1/\pi \kappa$.
        '''

        W = S.W
        if W < float('inf'):
            return W**2 / 8

        return 1/S.kappa/np.pi

class Spin_Spin_Normalized(DerivedQuantity):
    r'''
    The :class:`~.Spin_Spin` correlator $S_{\Delta x}$ normalized by its value at zero separation,

    .. math::

        \texttt{Spin\_Spin\_Normalized}_{\Delta x} = \frac{S_{\Delta x}}{S_0},

    so that $\texttt{Spin\_Spin\_Normalized}_0 = 1$. 

    In the Villain formulation the :class:`~.Spin_Spin` correlator is automatically normalized to 1 at the origin,
    but in the Worldline formulation the inline worm measurement needs to be normalized by the expectation value
    of the worm's histogram at the origin and therefore cannot be done configuration-by-configuration.

    On the Villain correlator this observable is essentially a no-op but it is a meaningful rescaling of the
    Worldline correlator required to match the two formulations.
    '''

    @staticmethod
    def default(S, Spin_Spin):
        return Spin_Spin / Spin_Spin[S.Lattice.origin]

class SpinSusceptibility(DerivedQuantity):
    r'''
    The *spin susceptibility* is the spacetime integral of the :class:`~.Spin_Spin_Normalized` correlator $S_{\Delta x}$,

    .. math::
        
        \texttt{SpinSusceptibility} = \chi_S = \int d^Dr\; S(r).
    '''

    @staticmethod
    def default(S, Spin_Spin_Normalized):
        return np.sum(Spin_Spin_Normalized.real)

class SpinStiffness(DerivedQuantity):
    r'''
    The *spin stiffness* (helicity modulus) $\Upsilon_\phi$ measures how much
    the free energy resists a global twist of the $U(1)_\phi$ symmetry: impose
    a boundary condition that rotates $\phi$ by a constant $a$ per step across
    the lattice in one direction and ask for the curvature of the free energy
    in $a$,

    .. math ::

        \Upsilon_\phi = \frac{1}{V} \left.\frac{\partial^2 F}{\partial a^2}\right|_{a=0},

    averaged over the directions of the twist.
    It is the order parameter for whether $U(1)_\phi$ is *rigid*.  A phase that
    spontaneously breaks $U(1)_\phi$ (or is critical) pays an energy $\propto a^2$
    to accommodate the twist, so $\Upsilon_\phi > 0$; a symmetric, disordered
    phase relaxes the twist away for free and $\Upsilon_\phi \to 0$ in the
    thermodynamic limit.  This is the field-theory analogue of a superfluid
    density: it is nonzero precisely when the would-be Goldstone mode is present.

    Two exact limits fix the interpretation.  As $\kappa \to \infty$ the field
    $n$ is frozen and the twist is absorbed rigidly, $\Upsilon_\phi \to \kappa$
    (the tree-level stiffness), so $\Upsilon_\phi/\kappa \to 1$.  In a phase
    where vortices proliferate and screen the twist completely,
    $\Upsilon_\phi \to 0$.  In the unconstrained :class:`~.Villain` model
    $\Upsilon_\phi/\kappa$ passes from $\approx 0$ below the critical coupling to
    $\approx 1$ above it, tracking the ordering transition; an intermediate
    value signals partial screening.

    .. note ::

        A positive stiffness establishes that $U(1)_\phi$ is **not disordered** ---
        the phase is rigid --- but it does not by itself distinguish true
        long-range order (a condensate with a Goldstone boson) from a
        critical/power-law phase; both are rigid.  Separating those requires a
        two-point summary such as the :class:`~.SpinSusceptibility` and its
        finite-size scaling.

    .. note ::

        Unlike the :class:`~.Spin_Spin` correlator, whose amplitude is
        multiplied by an exponentially small vortex-core factor and so sinks
        below the statistical floor at small $\kappa$, the stiffness is built
        from a topological winding sum that every local update moves freely.
        It is therefore not censored by the sampler: it stays informative
        exactly where the correlator goes dark.

    '''

    @staticmethod
    def Villain(S, WrappingSquared, TorusWrapping):
        r'''
        In the Villain frame the twist enters the action through
        $(d\phi - 2\pi n - a\,\hat\mu)^2$, and $\phi$ drops out of the response
        entirely: on the periodic lattice $\sum_{\ell \in \mu} d\phi_\ell = 0$
        configuration by configuration, so the second
        derivative of the free energy sees only the integer link sum
        ${M}_\mu = \sum_{\ell \in \mu} {n}_\ell$ --- which is exactly the
        :class:`~.TorusWrapping`.  The helicity modulus reduces to the
        fluctuation of that winding,

        .. math ::

            \begin{aligned}
                \Upsilon_\phi &= \kappa - \frac{(2\pi\kappa)^2}{V} \times \frac{1}{D} \sum_\mu \left(\langle {M}_\mu^2\rangle - \langle M_\mu \rangle^2\right)
                \nonumber\\
                &= \kappa - \frac{(2\pi\kappa)^2}{D\,V} \left(\langle \texttt{WrappingSquared} \rangle - \sum_\mu\langle \texttt{TorusWrapping}_\mu \rangle^2\right).
            \end{aligned}

        though the :class:`~.TorusWrapping` is 0 by symmetry.
        '''
        # <M_mu^2> summed over directions minus the disconnected piece sum_mu <M_mu>^2
        # (the latter vanishes in expectation by symmetry, but subtracting it removes
        # the finite-sample bias in <M>); WrappingSquared and TorusWrapping arrive as
        # bootstrap-resampled means, so this whole expression is per-resample.
        connected = WrappingSquared - (TorusWrapping**2).sum(axis=-1)
        L = S.Lattice
        # divide by D to average the per-direction stiffness over the D twist directions.
        return S.kappa - (2 * np.pi * S.kappa)**2 * connected / (L.D * L.sites)

class SpinSusceptibilityScaled(SpinSusceptibility):
    r'''
    At the critical point and in the CFT the :class:`~.SpinSusceptibility` has a known expected scaling that comes from the scaling dimension $\Delta$ of $e^{i\phi}$

    .. math::
        
        \chi_S \sim L^{D-2\Delta(\kappa)}.

    where the scaling dimension at the critical coupling $\kappa_c$ is known and depends on the constraint integer $W$.

    So, we scale the susceptibility by the :py:meth:`~.Spin_Spin.CriticalScalingDimension`,

    .. math::
        \texttt{SpinSusceptibilityScaled} = \chi_S / L^{D-2\Delta(\kappa_c)}

    so that at the critical coupling the infinite-volume limit of :class:`~.SpinSusceptibilityScaled` will be a constant.

    .. note::
        $2\Delta$ comes from the fact that the :class:`~.Spin_Spin` is a two-point function.
    '''

    @staticmethod
    def default(S, SpinSusceptibility):

        L = S.Lattice
        # NOTE: implicitly assumes that the lattice is square!
        return SpinSusceptibility / L.N**(L.D-2*Spin_Spin.CriticalScalingDimension(S))

class SpinMagnetizationSquared(OnlyVillain, Scalar, Observable):
    r'''
    The squared modulus of the volume-averaged spin,

    .. math::
        \texttt{SpinMagnetizationSquared} = |m|^2
        \qquad
        m = \frac{1}{\Lambda} \sum_x e^{i\phi_x},

    measured configuration by configuration.  The global $O(2)$ symmetry
    $\phi \rightarrow \phi + c$ guarantees $\langle m \rangle = 0$ on the torus, so no
    disconnected subtraction arises; $\Lambda \left\langle \left|m\right|^2 \right\rangle$ is the (finite-volume)
    spin susceptibility, equal in expectation to the :class:`~.SpinSusceptibility`
    computed from the :class:`~.Spin_Spin` correlator.

    Only implemented in the :class:`~.Villain` formulation, where $\phi$ is part of the field content.
    '''

    @staticmethod
    def Villain(S, phi):
        m = np.exp(1.j * phi[0]).mean()
        return np.abs(m)**2

class SpinMagnetizationQuartic(OnlyVillain, Scalar, Observable):
    r'''
    The fourth power of the modulus of the volume-averaged spin,

    .. math::
        \texttt{SpinMagnetizationQuartic} = |m|^4
        \qquad
        m = \frac{1}{\Lambda} \sum_x e^{i\phi_x},

    measured configuration by configuration; the quartic moment needed for the
    :class:`~.SpinBinderCumulant`.

    Only implemented in the :class:`~.Villain` formulation, where $\phi$ is part of the field content.
    '''

    @staticmethod
    def Villain(S, phi):
        m = np.exp(1.j * phi[0]).mean()
        return np.abs(m)**4

class SpinBinderCumulant(DerivedQuantity):
    r'''
    The Binder ratio of the spin order parameter $m = \frac{1}{\Lambda}\sum_x e^{i\phi_x}$,

    .. math::
        \texttt{SpinBinderCumulant} = U =
        1 - \frac{\left\langle |m|^4 \right\rangle}{2 \left\langle |m|^2 \right\rangle^2},

    the same convention as the :class:`~.IntersectionBinderCumulant`: a dimensionless,
    exponent-free diagnostic with $U \to 0$ (complex Gaussian) deep in the symmetric
    phase and $U \to 1/2$ in the spin-ordered phase, so that curves of $U(\kappa; L)$
    at different volumes cross at a critical point without knowledge of any scaling
    dimension.  (Same 2026-07-13 convention change as the
    :class:`~.IntersectionBinderCumulant`: previously $\langle|m|^4\rangle /
    \langle|m|^2\rangle^2$; convert stored values as $U_{\text{new}} = 1 -
    U_{\text{old}}/2$.)
    '''

    @staticmethod
    def default(S, SpinMagnetizationSquared, SpinMagnetizationQuartic):
        return 1 - SpinMagnetizationQuartic / (2 * SpinMagnetizationSquared**2)

class SpinCriticalMoment(DerivedQuantity):
    r'''
    The *critical moment* of the spin correlator :math:`C_S` is the volume-average of the correlator multiplied by its long-distance critical behavior,

    .. math::
        C_S = \frac{1}{L^D} \int d^Dr\; r^{2\Delta_S(\kappa_c, W)}\; S(r)

    At the critical $\kappa$ the long-distance behavior of the :class:`~.Spin_Spin` correlator :math:`S` decays with exactly the required power to cancel the explicit power of $r$ and the integral cancels the normalization, giving 1 in the large-$L$ limit.

    In the gapped phase $S$ decays exponentially with $r$ and the integral converges, so $C_S$ goes to 0 in the large-$L$ limit.

    In the CFT, $S$ decays polynomially, but slower than the weight from the moment grows.  The integral scales with a power larger than 2 and $C_S$ diverges in the large-$L$ limit.
    
    '''

    @staticmethod
    def default(S, Spin_Spin_Normalized):

        L = S.Lattice
        return np.sum(L.R_squared**(supervillain.observable.Spin_Spin.CriticalScalingDimension(S)) * Spin_Spin_Normalized.real) / L.sites

