import numpy as np
from supervillain.observable import Observable, Scalar, DerivedQuantity
from supervillain.lattice import d, wedge
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
        \Theta_{\Delta x} = \frac{1}{V} \sum_x \left\langle e^{i(\theta_x - \theta_{x - \Delta x})} \right\rangle,

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

        '''
        # The origin bin is *written*, not divided out: $\Theta_0 = 1$ identically 
        # (a coincident pair is the vacuum), so the gas never visits it and
        # :class:`~.Theta_Theta`'s origin bin is empty by construction.
        Theta = np.array(Theta_Theta / Vacuum_Ticks)
        Theta[S.Lattice.origin] = 1.0
        return Theta


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
    Tallies the two-pair sector dwell weighted by the ordered-assignment multiplicities
    of the four $D = 4$ classes in ``FourDefectDistribution`` that is produced as one of the :class:`~.DefectGas`
    :meth:`~supervillain.generator.no_intersection.DefectGas.inline_observables`,

    .. math ::

        \texttt{FourDefects} = (4, 2, 2, 1) \cdot \texttt{FourDefectDistribution},

    so that its ensemble mean divided by that of :class:`~.Vacuum_Ticks` is the
    two-pair-sector contribution to the fourth moment
    $\left\langle \left|M\right|^{4} \right\rangle$ of the $\theta$-shift order
    parameter (see :class:`~.IntersectionBinderCumulant`).
    """

    @staticmethod
    def default(S, FourDefectDistribution):
        return np.array([4., 2., 2., 1.]) @ FourDefectDistribution.real


class DoubleIntersectionSusceptibility(DerivedQuantity):
    r"""
    The *charge-2 intersection susceptibility*: the spacetime integral of the
    two-point function of the doubly-charged defect operator $e^{2i\theta}$,

    .. math ::

        \texttt{DoubleIntersectionSusceptibility} = \chi_{2}
        = \sum_{\Delta x} \left\langle e^{2i(\theta_x - \theta_{x-\Delta x})}
          \right\rangle.

    **Why measure it.**  The mixed 't Hooft anomaly forbids a trivially gapped
    $U(1)_\phi \times U(1)_\theta$-symmetric phase, but it does not insist the
    matching condensate carry $\theta$-charge one: if single defects are bound
    while *pairs* condense --- $U(1)_\theta \to \mathbb{Z}_2$ rather than fully
    broken --- the charge-1 correlator :class:`~.Intersection_Intersection`
    decays exponentially forever while the charge-2 correlator plateaus.  A
    paired condensate is the natural suspect in this model, whose $D = 4$ dwell
    is utterly dominated by the pair class.  The diagnostic is the volume
    scaling of $\chi_2 - 1 \sim V \left|\langle e^{2i\theta}\rangle\right|^2$.

    **Estimator.**  The only defect sector with the quantum numbers of
    $e^{2i\theta_x} e^{-2i\theta_y}$ at $x \neq y$ is $q = 2\delta_x -
    2\delta_y$: the $\{+2, -2\}$ class, index 3 of
    :class:`~.FourDefectDistribution` (and $D = \sum\left|q\right| = 4$ keeps
    it under the cap).  Following the same dwell-ratio logic as
    :class:`~.IntersectionBinderCumulant` --- the class tally is *not*
    translation-averaged, so one factor of the volume divides out, and the
    tally already carries its $1/\zeta^4$ (or sector-weight) price ---

    .. math ::

        \chi_2 = 1
        + \frac{\left\langle \texttt{FourDefectDistribution}_3 \right\rangle}
               {V \left\langle \texttt{Vacuum\_Ticks} \right\rangle},

    the $1$ being the coincident term: a coincident $\pm 2$ pair *is* the
    vacuum, exactly as $\Theta_0 = 1$ normalizes the charge-1 correlator.

    .. note ::

        The pair-separation umbrella does not disturb this estimator: the
        $w_2$ weight touches only the unit-charge classes, so the
        $\{+2, -2\}$ class always carries $W = 1$.

    .. warning ::

        The $\{+2,-2\}$ class is a rare stratum (:math:`< 0.4\%` of the
        $D = 4$ dwell at the couplings surveyed), so expect honest but wide
        errors; a bin identically zero across an ensemble means the class was
        never *visited* --- censored, not measured (see
        :class:`~.FourDefectDistribution`).
    """

    @staticmethod
    def NoIntersections(S, FourDefectDistribution, Vacuum_Ticks):
        V = int(np.prod(S.Lattice.dims))
        return 1 + FourDefectDistribution.real[3] / (V * Vacuum_Ticks)


class IntersectionCurrent(Observable):
    r"""
    The $\theta$-sector current: the integer-valued 3-form

    .. math ::

        \texttt{IntersectionCurrent} = j = n \wedge dn,

    whose divergence is *exactly* the topological-charge density,

    .. math ::

        dj = d(n \wedge dn) = dn \wedge dn - n \wedge d(dn) = q,

    by the lattice Leibniz rule and $d^2 = 0$ --- both exact, so the identity
    holds configuration by configuration, not just in expectation.

    **Physical meaning.**  $U(1)_\theta$ shifts $\theta$ by a constant; its
    charged objects are the defects created by $e^{i\theta}$, and $J$ is the
    conserved current that transports that charge: in the constrained ensemble
    ($q \equiv 0$) it is identically divergence-free.  It is the
    $\theta$-sector analog of the vorticity current of $U(1)_\phi$, and it is
    computable on every *stored* configuration --- no defect insertions, no
    enlarged ensemble --- so it opens the $\theta$ sector to plain
    re-analysis.  Its topological slice sums are the
    :class:`~.IntersectionWinding`, whose fluctuations
    (:class:`~.IntersectionWindingSquared`) are the stiffness diagnostic of
    $U(1)_\theta$ symmetry breaking.

    Requires a four-dimensional lattice.  On the unconstrained Villain model it
    is still measurable, but $dj = q \neq 0$, so only its *constrained*
    ($q \equiv 0$) slice sums are topological.
    """

    @staticmethod
    def Villain(S, n):
        r'''Measure the 3-form $j = n \wedge dn$, shape ``(4,) + L.dims``.'''
        L = S.Lattice
        if L.D != 4:
            raise NotImplementedError(
                'IntersectionCurrent requires a four-dimensional lattice.')
        return np.asarray(wedge(n, d(n)))


class IntersectionWinding(Observable):
    r"""
    The topological winding of the :class:`~.IntersectionCurrent` around each
    direction of the torus: for each $\mu$, the flux of the 3-form $j$ through
    the 3-torus transverse to $\hat\mu$,

    .. math ::

        \texttt{IntersectionWinding}_\mu = J_\mu
        = \sum_{x \,:\, x_\mu = c} j_{\bar\mu}(x),

    with $\bar\mu$ the 3-form component spanning the other three directions.
    Because $dj = q$ and the constrained ensemble has $q \equiv 0$, the sum is
    independent of the slice position $c$ --- a topological integer per
    configuration per direction.  We evaluate it as the lattice average over
    slices, $\frac{1}{N}\sum_x j_{\bar\mu}(x)$, which coincides with any single
    slice when the charge vanishes and degrades gracefully (to the
    slice-averaged flux) when it does not.

    Nonzero fluctuations of $J$ are how a $\theta$ condensate carries
    supercurrent around the torus; see :class:`~.IntersectionWindingSquared`.
    """

    @staticmethod
    def Villain(S, IntersectionCurrent):
        r'''The slice-averaged flux of $j$ per direction, shape ``(4,)``.'''
        L = S.Lattice
        W = np.empty(4)
        for mu in range(4):
            comp = tuple(k for k in range(4) if k != mu)
            W[mu] = IntersectionCurrent[L.comp_index[3][comp]].sum() / L.N
        return W


class IntersectionWindingSquared(Scalar, Observable):
    r"""
    The direction-averaged square of the :class:`~.IntersectionWinding`,

    .. math ::

        \texttt{IntersectionWindingSquared}
        = \frac{1}{4} \sum_\mu J_\mu^2.

    **This is the sharp finite-volume diagnostic of $U(1)_\theta$ symmetry
    breaking** --- the $\theta$-sector stiffness, playing the role the helicity
    modulus plays for a superfluid.  $\theta$ has no kinetic term (it enters
    only as $i\theta q$), so its stiffness is generated entirely by the matter
    it constrains; a $\theta$ condensate supports supercurrents of defect
    charge winding the torus, giving $\langle J^2 \rangle \neq 0$ with
    the characteristic superfluid volume scaling, while a gapped symmetric
    phase suppresses it exponentially and a critical (gapless) phase shows
    scale-invariant fluctuations.  It therefore separates the two anomaly
    matchings --- condensate versus gapless --- that the charge-1 correlator
    :class:`~.Intersection_Intersection` cannot distinguish at accessible
    volumes.

    Because the winding changes only through topology-shifting moves, check
    its autocorrelation time before trusting error bars: a chain can render it
    *frozen* rather than measured.
    """

    @staticmethod
    def Villain(S, IntersectionWinding):
        r'''Measure $\frac{1}{4}\sum_\mu J_\mu^2$.'''
        return np.mean(IntersectionWinding ** 2)


class IntersectionBinderCumulant(DerivedQuantity):
    r"""
    The Binder ratio of the $\theta$-shift order parameter
    $M = \sum_{x} e^{i\theta_{x}}$,

    .. math ::

        \texttt{IntersectionBinderCumulant} = U =
        1 - \frac{\left\langle \left|M\right|^{4} \right\rangle}
                 {2 \left\langle \left|M\right|^{2} \right\rangle^{2}},

    a dimensionless, exponent-free diagnostic: $U \to 0$ deep in the symmetric phase
    (complex Gaussian; the finite-volume value is $1/2V$), $U \to 1/2$ in a
    $\theta$-ordered phase, and curves of $U(\kappa; L)$ at different volumes cross at
    a critical point without knowledge of any scaling dimension.

   """

    @staticmethod
    def default(S, Theta_Theta, Vacuum_Ticks, FourDefects):
        V = int(np.prod(S.Lattice.dims))
        S1 = np.sum(Theta_Theta.real) / Vacuum_Ticks
        M2 = V * (1 + S1)
        M4 = (2 * V**2 - V) + 4 * (V - 1) * V * S1 + FourDefects / Vacuum_Ticks
        return 1 - M4 / (2 * M2**2)
