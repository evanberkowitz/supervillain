import numpy as np
from supervillain.observable import Observable, DerivedQuantity
import supervillain.action


class IntersectionTwoPoint(Observable):
    r'''
    The raw head$-$tail displacement histogram accumulated inline by a worm on the
    :class:`~supervillain.action.NoIntersections` model --- the ingredient from which
    :class:`~.Intersection_Intersection` would be built, exactly as
    :class:`~.ActionTwoPoint` is the ingredient of :class:`~.Action_Action`.

    **There is no closed-form estimator**, and this histogram is *not* normalized: the
    normalization is by its expectation value at the origin, which cannot be applied
    configuration-by-configuration.

    .. warning ::

        Every worm we built jams --- see :ref:`the no-intersection docs
        <no_intersection>`.  This observable exists for the opt-in
        :class:`~supervillain.generator.no_intersection.IntersectionWorm` and
        :class:`~supervillain.generator.no_intersection.FreeTargetWorm`, neither of which
        is in the default :func:`~supervillain.generator.no_intersection.Hammer`.  For the
        physical correlator use :class:`~.Intersection_Intersection`, which the
        :class:`~supervillain.generator.no_intersection.DefectGas` measures with absolute
        normalization.
    '''


class Intersection_Intersection(DerivedQuantity):
    r'''
    The intersection--intersection correlator in the
    :class:`~supervillain.action.NoIntersections` model,

    .. math ::
        \Theta_{\Delta x} = \left\langle e^{i(\theta_x - \theta_{x - \Delta x})} \right\rangle,

    the two-point function of the operator $e^{i\theta}$ that inserts a unit of
    vortex-sheet self-intersection $q = (dn \wedge dn)$.

    '''

    @staticmethod
    def NoIntersections(S, Theta_Theta, Vacuum_Ticks):
        r'''
        .. note ::
        
            While you can build this quantity from the :class:`~.IntersectionTwoPoint` correlator, it is much more efficiently measured by the :class:`~.DefectGas`.

        Measured by the :class:`~supervillain.generator.no_intersection.DefectGas` as the
        ratio of sector dwells,
        
        .. math ::

            \Theta_{\Delta x} = \frac{\left\langle\texttt{Theta\_Theta}_{\Delta x}\right\rangle_{\Pi}}{\left\langle\texttt{Vacuum\_Ticks}\right\rangle_{\Pi}}.

        and is therefore absolutely normalized: no
        origin-bin division is needed, and $\Theta$ is independent of the fugacity $\zeta$.

        The origin bin is *written*, not divided out: $\Theta_0 = 1$ identically (a coincident
        pair is the vacuum), so the gas never visits it and
        :class:`~.Theta_Theta`'s origin bin is empty by construction.
        '''
        theta = np.array(Theta_Theta / Vacuum_Ticks)
        theta[S.Lattice.origin] = 1.0
        return theta


class Intersection_Intersection_Normalized(DerivedQuantity):
    r'''
    The :class:`~.Intersection_Intersection` correlator normalized by its value at zero
    separation.  Since :class:`~.Intersection_Intersection` is already absolutely
    normalized ($\Theta_0 = 1$), this is the identity; it is retained so that analysis
    code written against the :class:`~.Vortex_Vortex_Normalized` /
    :class:`~.Spin_Spin_Normalized` idiom keeps working.
    '''

    @staticmethod
    def default(S, Intersection_Intersection):
        return Intersection_Intersection / Intersection_Intersection[S.Lattice.origin]

class IntersectionSusceptibility(DerivedQuantity):
    r"""
    The *intersection susceptibility* is the spacetime integral of the
    absolutely-normalized intersection correlator,

    .. math ::

        \texttt{IntersectionSusceptibility} = \chi_\theta = \int d^Dr\; \Theta(r)
        = \sum_{\Delta x} \Theta_{\Delta x},

    a plain sum over :class:`~.Intersection_Intersection_Normalized`.  

    When the $\theta$ correlations are short-ranged, $\chi_\theta$ approaches a
    constant in the thermodynamic limit.  If there is $\theta$ long-range order
    it instead grows with the volume.
    """

    @staticmethod
    def default(S, Intersection_Intersection_Normalized):
        return np.sum(Intersection_Intersection_Normalized.real)


class Theta_Theta(Observable):
    r"""
    The absolutely-normalized intersection correlator measured by the
    :class:`~supervillain.generator.no_intersection.DefectGas`: per step, the pair-sector
    dwell histogram scaled by its known fugacity price,
    $H_{\text{pair}}(\Delta x) / V \zeta^{2}$.  Its ensemble mean, divided by the mean
    of :class:`~.Vacuum_Ticks`, is the correlator
    $\Theta_{\Delta x} = \left\langle e^{i(\theta_x - \theta_y)} \right\rangle$.

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas`
    only; there is no ex-post-facto estimator.

    .. seealso ::

        This is one of the :class:`~.DefectGas` :meth:`~supervillain.generator.no_intersection.DefectGas.inline_observables`.
    """


class Vacuum_Ticks(Observable):
    r"""
    The vacuum-sector dwell of the :class:`~supervillain.generator.no_intersection.DefectGas`
    per step: how many of the step's Monte-Carlo clock ticks sat at $q \equiv 0$.  The
    denominator of the sector-dwell estimator (see :class:`~.Theta_Theta`).

    .. seealso ::

        This is one of the :class:`~.DefectGas` :meth:`~supervillain.generator.no_intersection.DefectGas.inline_observables`.
    """


class Pair_Excursions(Observable):
    r"""
    The number of pair *excursions* --- maximal stretches of nonvacuum ticks of the
    :class:`~supervillain.generator.no_intersection.DefectGas` chain --- completed
    during the step.  The effective sample count behind the far bins of
    :class:`~.Theta_Theta`: those bins are fed only by excursions whose relative walk
    survives to large separation, so a small excursion count means the far bins are
    *transport-censored*, not measured.

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas` only.
    """


class Max_Pair_RSq(Observable):
    r"""
    The largest min-image separation squared $\left|\Delta x\right|^{2}$ reached by any
    single $\pm$ pair during the step --- the step's *transport ceiling*.  Bins of
    :class:`~.Theta_Theta` beyond $\sqrt{\texttt{Max\_Pair\_RSq}}$ were never even
    visited: a zero there is a censored value, bounded by transport, and carries no
    information about $\theta$ long-range order.

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas` only.
    """


class Excursion_Lengths(Observable):
    r"""
    Histogram of completed excursion lengths (in Monte-Carlo ticks) during the step,
    in saturating power-of-two bins: entry $b$ counts excursions whose length had
    bit-length $b$, i.e.\ lengths in $[2^{b-1}, 2^{b})$.  Far-separation dwell lives
    in the extreme tail of this distribution.

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas` only.
    """



class FourDefectDistribution(Observable):
    r"""
    The :class:`~supervillain.generator.no_intersection.DefectGas`'s per-step dwell in
    the four $D = 4$ sector classes, scaled by the known fugacity price $1/\zeta^{4}$
    (only the generator knows $\zeta$, so the price must be divided out at emission):
    index 0 counts $\{+1,+1,-1,-1\}$ (four distinct hypercubes), 1 counts
    $\{+2,-1,-1\}$, 2 counts $\{+1,+1,-2\}$, and 3 counts $\{+2,-2\}$.

    The class-resolved bins are what is stored because inline quantities cannot be
    re-measured from stored configurations, and because a class whose bin is
    identically zero across an ensemble was never *visited* --- a censored stratum,
    not a measurement of zero.  Physics quantities should consume the
    multiplicity-weighted combination :class:`~.FourDefects` instead.

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas` only.
    """


class FourDefects(Observable):
    r"""
    The two-pair-sector dwell weighted by the ordered-assignment multiplicities of the
    four $D = 4$ classes,

    .. math ::

        \texttt{FourDefects} = (4, 2, 2, 1) \cdot \texttt{FourDefectDistribution},

    so that its ensemble mean divided by that of :class:`~.Vacuum_Ticks` is the
    two-pair-sector contribution to the fourth moment
    $\left\langle \left|M\right|^{4} \right\rangle$ of the $\theta$-shift order
    parameter (see :class:`~.ThetaBinderCumulant`).
    """

    @staticmethod
    def default(S, FourDefectDistribution):
        return np.array([4., 2., 2., 1.]) @ FourDefectDistribution.real


class ThetaBinderCumulant(DerivedQuantity):
    r"""
    The Binder ratio of the $\theta$-shift order parameter
    $M = \sum_{x} e^{i\theta_{x}}$,

    .. math ::

        \texttt{ThetaBinderCumulant} = U =
        \frac{\left\langle \left|M\right|^{4} \right\rangle}
             {\left\langle \left|M\right|^{2} \right\rangle^{2}},

    a dimensionless, exponent-free diagnostic: $U \to 2$ (complex Gaussian) deep in the
    symmetric phase, $U \to 1$ in a $\theta$-ordered phase, and curves of $U(\kappa; L)$
    at different volumes cross at a critical point without knowledge of any scaling
    dimension.

   """

    @staticmethod
    def default(S, Theta_Theta, Vacuum_Ticks, FourDefects):
        V = int(np.prod(S.Lattice.dims))
        S1 = np.sum(Theta_Theta.real) / Vacuum_Ticks
        M2 = V * (1 + S1)
        M4 = (2 * V**2 - V) + 4 * (V - 1) * V * S1 + FourDefects / Vacuum_Ticks
        return M4 / M2**2
