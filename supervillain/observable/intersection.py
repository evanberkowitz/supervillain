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

    Measured by the :class:`~supervillain.generator.no_intersection.DefectGas` as the
    ratio of sector dwells,
    $\Theta_{\Delta x} = \overline{\texttt{Theta\_Theta}}_{\Delta x} /
    \overline{\texttt{Vacuum\_Ticks}}$, and therefore **absolutely normalized**: no
    origin-bin division is needed, and $\Theta$ is independent of the fugacity $\zeta$.

    The origin bin is *written*, not divided out: $\Theta_0 = 1$ identically (a coincident
    pair is the vacuum), so the gas never visits it and
    :class:`~.Theta_Theta`'s origin bin is empty by construction.
    '''

    @staticmethod
    def NoIntersections(S, Theta_Theta, Vacuum_Ticks):
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


class Theta_Theta(Observable):
    r"""
    The absolutely-normalized intersection correlator measured by the
    :class:`~supervillain.generator.no_intersection.DefectGas`: per step, the pair-sector
    dwell histogram scaled by its known fugacity price,
    $H_{\text{pair}}(\Delta x) / V \zeta^{2}$.  Its ensemble mean, divided by the mean
    of :class:`~.Vacuum_Ticks`, is the correlator
    $\Theta_{\Delta x} = \left\langle e^{i(\theta_x - \theta_y)} \right\rangle$
    with **no origin normalization required** ($\Theta_0 = 1$ identically; the origin
    bin of the histogram is empty by construction --- a coincident pair *is* the
    vacuum).

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas`
    only; there is no ex-post-facto estimator.
    """


class Vacuum_Ticks(Observable):
    r"""
    The vacuum-sector dwell of the :class:`~supervillain.generator.no_intersection.DefectGas`
    per step: how many of the step's Monte-Carlo clock ticks sat at $q \equiv 0$.  The
    denominator of the sector-dwell estimator (see :class:`~.Theta_Theta`).

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas` only.
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


class IntersectionSusceptibility(DerivedQuantity):
    r"""
    The *intersection susceptibility* is the spacetime integral of the
    absolutely-normalized intersection correlator,

    .. math ::

        \texttt{IntersectionSusceptibility} = \chi_\theta = \int d^Dr\; \Theta(r)
        = 1 + \sum_{\Delta x \neq 0} \Theta_{\Delta x},

    where the $1$ is the identically-unit origin ($\Theta_0 = 1$: coincident insertions
    are the identity) and the rest comes from the
    :class:`~supervillain.generator.no_intersection.DefectGas`'s sector-dwell estimator,
    $\Theta_{\Delta x} = \overline{\texttt{Theta\_Theta}}_{\Delta x} /
    \overline{\texttt{Vacuum\_Ticks}}$.

    When the $\theta$ correlations are short-ranged, $\chi_\theta$ approaches a
    **constant** in the thermodynamic limit.  $\theta$ long-range order --- the defect
    condensate conjugate to the no-intersection constraint, one way the mixed anomaly
    could be matched --- instead makes it **grow with the volume**,
    $\chi_\theta \sim \left|\left\langle e^{i\theta} \right\rangle\right|^2 V$,
    so the volume dependence of $\chi_\theta$ is a clean order-parameter diagnostic
    even though $\left\langle e^{i\theta} \right\rangle$ itself vanishes identically
    on the torus.

    Requires the inline observables of the
    :class:`~supervillain.generator.no_intersection.DefectGas`.
    """

    @staticmethod
    def default(S, Theta_Theta, Vacuum_Ticks):
        return 1 + np.sum(Theta_Theta.real) / Vacuum_Ticks


class Four_Defect(Observable):
    r"""
    The :class:`~supervillain.generator.no_intersection.DefectGas`'s per-step dwell in
    the four $D = 4$ sector classes, scaled by the known fugacity price $1/\zeta^{4}$:
    index 0 counts $\{+1,+1,-1,-1\}$ (four distinct hypercubes), 1 counts
    $\{+2,-1,-1\}$, 2 counts $\{+1,+1,-2\}$, and 3 counts $\{+2,-2\}$.  The raw
    material of the fourth moment of the $\theta$-shift order parameter (see
    :class:`~.ThetaBinderCumulant`).

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas` only.
    """


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

    Both moments reduce to sector dwell.  With $\Theta(0) = 1$ identically and
    $S_{1} = \sum_{r \neq 0} \Theta(r) =
    \overline{\texttt{Theta\_Theta}}\,\Sigma / \overline{\texttt{Vacuum\_Ticks}}$,

    .. math ::

        \left\langle \left|M\right|^{2} \right\rangle = V (1 + S_{1}).

    For the fourth moment, classify the ordered insertion 4-tuples
    $(x_{1}, x_{2}; y_{1}, y_{2})$ by their net charge pattern: vacuum patterns
    contribute the contact combinatorics $2V^{2} - V$; single-pair patterns contribute
    $4(V-1)\, V\, S_{1}$ (a multiset count: $\{x_{1}, x_{2}, b\} = \{y_{1}, y_{2}, a\}$
    has $4(V-1)$ ordered solutions per $(a, b)$); and the genuine two-pair sectors enter
    with ordered multiplicities $4, 2, 2, 1$ for the classes
    $\{+1,+1,-1,-1\}$, $\{+2,-1,-1\}$, $\{+1,+1,-2\}$, $\{+2,-2\}$ counted by
    :class:`~.Four_Defect`:

    .. math ::

        \left\langle \left|M\right|^{4} \right\rangle
        = (2V^{2} - V) + 4(V-1)\, V\, S_{1}
        + \frac{\overline{(4, 2, 2, 1) \cdot \texttt{Four\_Defect}}}
               {\overline{\texttt{Vacuum\_Ticks}}}.

    Checks: at $V = 1$ the formula returns $1$ exactly; deep in the symmetric phase the
    contact term alone gives $U = 2 - 1/V$; and $U$ is independent of the fugacity $\zeta$ (all
    fugacity prices are divided back out), so runs at two fugacities test the
    implementation end to end.  Note $\langle M \rangle = \langle M^{2} \rangle = 0$
    exactly on the torus (charge neutrality), so no disconnected subtractions arise.

    Requires the inline observables of the
    :class:`~supervillain.generator.no_intersection.DefectGas`.
    """

    @staticmethod
    def default(S, Theta_Theta, Vacuum_Ticks, Four_Defect):
        V = int(np.prod(S.Lattice.dims))
        S1 = np.sum(Theta_Theta.real) / Vacuum_Ticks
        M2 = V * (1 + S1)
        M4 = (2 * V**2 - V) + 4 * (V - 1) * V * S1 \
            + np.sum(np.array([4., 2., 2., 1.]) * Four_Defect.real) / Vacuum_Ticks
        return M4 / M2**2
