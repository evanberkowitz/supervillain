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
