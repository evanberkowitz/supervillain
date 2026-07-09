import numpy as np
from supervillain.observable import Observable, DerivedQuantity
import supervillain.action


class Intersection_Intersection(Observable):
    r'''
    The intersection--intersection correlator in the :class:`~supervillain.action.NoIntersections`
    model,

    .. math ::
        \Theta_{x,y} = \left\langle e^{i(\theta_x - \theta_y)} \right\rangle,

    the two-point function of the operator $e^{i\theta}$ that inserts a unit of
    vortex-sheet self-intersection $q = (dn \wedge dn)$, reduced by translation
    invariance to a single relative coordinate

    .. math ::
        \texttt{Intersection\_Intersection}_{\Delta x} = \Theta_{\Delta x} = \frac{1}{\Lambda} \sum_x \Theta_{x, x-\Delta x}.

    **There is no closed-form estimator** because the constraint can obstruct any straightforward way to compute the correlator.
    The correlator is measured *inline* as the head$-$tail displacement histogram of the :class:`~supervillain.generator.no_intersection.IntersectionWorm`.
    The inline histogram is not normalized to $1$ at the origin, and that normalization can only be applied *after* the ensemble average.
    Therefore, the :class:`~.Intersection_Intersection_Normalized` is a :class:`~.DerivedQuantity`.

    The observable is only ever produced inline by the worm, so it is available on
    the :class:`~supervillain.action.NoIntersections` model only; there is no
    generic :class:`~.Villain` or :class:`~.Worldline` implementation.

    .. note ::

        In fact, because of the constraint issue there is no ex-post-facto observable at all!
        This is a stub placeholder.
    '''


class Intersection_Intersection_Normalized(DerivedQuantity):
    r'''
    The :class:`~.Intersection_Intersection` correlator $\Theta_{\Delta x}$ normalized by
    its value at zero separation,

    .. math ::

        \texttt{Intersection\_Intersection\_Normalized}_{\Delta x} = \frac{\Theta_{\Delta x}}{\Theta_0},

    so that $\texttt{Intersection\_Intersection\_Normalized}_0 = 1$.

    The inline worm histogram must be normalized by the *expectation value* of the
    histogram at the origin, which cannot be done configuration-by-configuration,
    so this is a :class:`~.DerivedQuantity`.  We provide a default implementation,
    but notice that only the :class:`~supervillain.action.NoIntersections` even supports the idea
    and the requisite observable must be measured by the :class:`~supervillain.generator.no_intersection.IntersectionWorm`.
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
