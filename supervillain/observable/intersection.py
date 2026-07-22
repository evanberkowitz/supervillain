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
    def NoIntersections(S, Theta_Theta, VacuumTicks):
        r'''
        .. note ::
        
            While you can build this quantity from the :class:`~.IntersectionTwoPoint` correlator, it is much more efficiently measured by the :class:`~.DefectGas`.

        Measured by the :class:`~supervillain.generator.no_intersection.DefectGas` as the
        ratio of sector dwells,
        
        .. math ::

            \Theta_{\Delta x} = \frac{\left\langle\texttt{Theta\_Theta}_{\Delta x}\right\rangle_{\Pi}}{\left\langle\texttt{VacuumTicks}\right\rangle_{\Pi}}.

        and is therefore absolutely normalized: no
        origin-bin division is needed, and $\Theta$ is independent of the fugacity $\zeta$.

        '''
        # The origin bin is *written*, not divided out: $\Theta_0 = 1$ identically 
        # (a coincident pair is the vacuum), so the gas never visits it and
        # :class:`~.Theta_Theta`'s origin bin is empty by construction.
        Theta = np.array(Theta_Theta / VacuumTicks)
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
    of :class:`~.VacuumTicks`, is the correlator
    $\Theta_{\Delta x} = \left\langle e^{i(\theta_x - \theta_y)} \right\rangle$.

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas`
    only; there is no ex-post-facto estimator.

    .. seealso ::

        This is one of the :class:`~.DefectGas` :meth:`~supervillain.generator.no_intersection.DefectGas.inline_observables`.
    """


class VacuumTicks(Observable):
    r"""
    The vacuum-sector dwell of the :class:`~supervillain.generator.no_intersection.DefectGas`
    per step: how many of the step's Monte-Carlo clock ticks sat at $q \equiv 0$.  The
    denominator of the sector-dwell estimator (see :class:`~.Theta_Theta`).

    .. seealso ::

        This is one of the :class:`~.DefectGas` :meth:`~supervillain.generator.no_intersection.DefectGas.inline_observables`.
    """


class PairExcursions(Observable):
    r"""
    The number of pair *excursions* --- maximal stretches of nonvacuum ticks of the
    :class:`~supervillain.generator.no_intersection.DefectGas` chain --- completed
    during the step.  The effective sample count behind the far bins of
    :class:`~.Theta_Theta`: those bins are fed only by excursions whose relative walk
    survives to large separation, so a small excursion count means the far bins are
    *transport-censored*, not measured.

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas` only.
    """


class MaxPairSeparationSquared(Observable):
    r"""
    The largest min-image separation squared $\left|\Delta x\right|^{2}$ reached by any
    single $\pm$ pair during the step --- the step's *transport ceiling*.  Bins of
    :class:`~.Theta_Theta` beyond $\sqrt{\texttt{MaxPairSeparationSquared}}$ were never even
    visited: a zero there is a censored value, bounded by transport, and carries no
    information about $\theta$ long-range order.

    Produced inline by the :class:`~supervillain.generator.no_intersection.DefectGas` only.
    """


class ExcursionLengths(Observable):
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

    so that its ensemble mean divided by that of :class:`~.VacuumTicks` is the
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
               {V \left\langle \texttt{VacuumTicks} \right\rangle},

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
    def NoIntersections(S, FourDefectDistribution, VacuumTicks):
        V = int(np.prod(S.Lattice.dims))
        return 1 + FourDefectDistribution.real[3] / (V * VacuumTicks)


def chern_simons_form(n):
    r"""
    The abelian Chern--Simons 3-form of the integer connection $n$,

    .. math ::

        \mathrm{CS}(n) = n \wedge dn,
        \qquad
        d\,\mathrm{CS}(n) = dn \wedge dn - n \wedge d(dn) = q,

    by the lattice Leibniz rule and $d^2 = 0$ --- both exact, so the identity
    holds configuration by configuration, not just in expectation.  Integer
    valued, and computed in exact integer arithmetic, which is what makes the
    periods :class:`~.IntersectionWinding` exact integers rather than floats
    that happen to round.

    .. warning ::

        **This is not gauge invariant, and that is intrinsic to a
        Chern--Simons form rather than a defect to be fixed.**  Under the
        integer gauge transformation $\phi \to \phi + 2\pi m$, $n \to n + dm$
        (what :class:`~supervillain.generator.villain.ExactUpdate` performs),

        .. math ::

            n \wedge dn \;\longrightarrow\; n \wedge dn + dm \wedge dn
            \;=\; n \wedge dn + d(m \wedge dn),

        an *exact* shift.  Only quantities insensitive to an exact 3-form
        survive --- the periods over closed 3-cycles, which is exactly why
        :class:`~.IntersectionWinding` is well defined (a period is a sum over
        a cycle with no boundary, so discrete Stokes annihilates the shift;
        note this needs only closedness of the *cycle*, so it holds even off
        the constraint surface).  **Anything local --- a two-point function at
        separated points, a structure factor at $k \neq 0$, the pointwise mean
        --- is a property of the gauge the configuration happens to be stored
        in, not an observable.**

        This is deliberately a plain function and *not* an
        :class:`~.Observable`: registering it would make it an
        :class:`~supervillain.Ensemble` attribute that
        :meth:`~supervillain.Ensemble.measure` computes by default and writes
        to disk, which is both a $V$-fold storage waste (its whole invariant
        content is four integers per configuration) and an invitation to
        correlate it.  It was an ``Observable`` until 2026-07-20; the stored
        arrays were deleted by ``no-intersections/migrate_h5_names.py``, whose
        ``DELETIONS`` section records why.

        When a *local* $\theta$ current is wanted, use
        :class:`~.IntersectionCurrent`, which is gauge invariant pointwise.

    Parameters
    ----------
    n : supervillain.lattice.Form
        An integer-valued 1-form on a four-dimensional lattice.

    Returns
    -------
    supervillain.lattice.Form
        The integer 3-form $n \wedge dn$, degree 3, shape ``(4,) + L.dims``.
    """
    if n.lattice.D != 4:
        raise NotImplementedError(
            'The Chern-Simons form requires a four-dimensional lattice.')
    return wedge(n, d(n))


class IntersectionCurrent(Observable):
    r"""
    The $U(1)_\theta$ current: the **gauge-invariant** 3-form

    .. math ::

        \texttt{IntersectionCurrent} = j
        = \frac{(d\phi - 2\pi n) \wedge dn}{-2\pi},

    normalized so that its divergence is *exactly* the topological-charge
    density, $dj = q$.  Both factors are inert under
    $\phi \to \phi + 2\pi m$, $n \to n + dm$ --- $(d\phi - 2\pi n)$ is the
    invariant combination the action is built from, and $dn$ is invariant
    because $d^2 = 0$ --- so unlike the Chern--Simons form
    :func:`~.chern_simons_form` this is an observable **pointwise**, and its
    correlators and structure factors at $k \neq 0$ mean something.

    .. note ::

        The two currents differ by an *improvement term*,

        .. math ::

            j = n \wedge dn - \frac{d(\phi\, dn)}{2\pi},

        using $d\phi \wedge dn = d(\phi\, dn)$.  An exact form has vanishing
        periods, so the two carry the **same** :class:`~.IntersectionWinding`
        and differ only by the piece that was gauge-ambiguous in the first
        place.  Improvement terms are $\partial$ of a local antisymmetric
        object, contributing only terms regular at $p = 0$, so they do not
        move the residue of a massless pole --- but if you ever extract a
        *nonzero* stiffness this way, check it against the choice of
        representative before believing it.

    **Physical meaning.**  $U(1)_\theta$ shifts $\theta$ by a constant; its
    charged objects are the defects created by $e^{i\theta}$, and $j$ is the
    conserved current that transports that charge: in the constrained ensemble
    ($q \equiv 0$) it is identically divergence-free.  It is computable on
    every *stored* configuration --- no defect insertions, no enlarged
    ensemble, no worm --- so it opens the $\theta$ sector to plain
    re-analysis.  Because it is closed on shell its dual is transverse, which
    makes $\left\langle \lvert\hat{\jmath}(k)\rvert^2 \right\rangle$ the transverse
    current correlator whose $k \to 0$ intercept is the $\theta$-sector
    helicity modulus --- the stiffness diagnostic
    (:class:`~.IntersectionWindingSquared`) measured *without* needing the
    sampler to move a winding.

    .. warning ::

        This is a $V$-sized float field, so measuring it across a campaign
        stores $4 V$ numbers per configuration.  Prefer computing it on demand,
        or reducing it (to a structure factor) before storing.

    Requires a four-dimensional lattice.  On the unconstrained Villain model it
    is still measurable, but $dj = q \neq 0$, so only its *constrained*
    ($q \equiv 0$) slice sums are topological.
    """

    @staticmethod
    def Villain(S, phi, n):
        r'''Measure $j = (d\phi - 2\pi n) \wedge dn / (-2\pi)$ as a degree-3
        :class:`~supervillain.lattice.Form`, shape ``(4,) + L.dims``.'''
        L = S.Lattice
        if L.D != 4:
            raise NotImplementedError(
                'IntersectionCurrent requires a four-dimensional lattice.')
        A = d(phi) - 2 * np.pi * n
        return wedge(A, d(n)) / (-2 * np.pi)


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
    independent of the slice position $c$ --- a topological **integer** per
    configuration per direction, and integer *exactly*: $j$ is integer-valued
    and a period is a sum of integers over a cycle, so no rounding enters
    anywhere.  We evaluate a single slice and return it in an integer dtype.

    Nonzero fluctuations of $J$ are how a $\theta$ condensate carries
    supercurrent around the torus; see :class:`~.IntersectionWindingSquared`.

    **What $J$ is, topologically.**  $dj = q$, so the constraint $q \equiv 0$
    says exactly that $j$ is a *closed* 3-form, and it therefore carries a
    cohomology class

    .. math ::

        [j] \in H^3(T^4; \mathbb{Z}) \cong \mathbb{Z}^4,

    of which the four $J_\mu$ are the periods over the coordinate 3-cycles.
    This is the *secondary* (Chern--Simons / Hopf) invariant: it is born
    precisely where the primary invariant $q$ dies, and it exists on the
    constrained manifold and nowhere else.  (In the unconstrained model $j$ is
    not closed, the slice sums depend on $c$, and there is no class.)  Nothing
    here requires the vortex sheet to be multiplicity-free or a manifold, so
    unlike surface invariants such as the genus or the Alexander polynomial,
    $J$ is computable on *every* stored configuration --- including the generic
    ones with $|F| \ge 2$ --- in $O(V)$, with no diagram, projection, slice, or
    desingularization.

    **$J$ depends on the field strength alone.**  Although written with $n$, it
    is a function of $F = dn$: for any *closed* 1-form $z$ (a gauge
    transformation $dm$, or a harmonic winding $h$),
    $z \wedge dn = -d(z \wedge n)$ is exact, so $[z \wedge F] = 0$ and
    $[j]$ cannot move.  Both gauge and winding drop out.

    Read that carefully: it is the *class* $[j]$ --- equivalently the periods
    $J_\mu$ --- that is invariant, because an exact shift integrates to zero
    over a closed cycle.  The current $j$ itself is **not** invariant
    pointwise, so this paragraph licenses $J_\mu$ and nothing finer.  That is
    why :func:`~.chern_simons_form` is a plain function rather than an
    observable, and why a *local* $\theta$ current means the gauge-invariant
    :class:`~.IntersectionCurrent`.

    **Geometrically it is a self-linking (framing) number, not a knot
    invariant.**  By Poincare duality $H^3 \cong H_1$, so $[j]$ measures
    *loop--sheet* linking --- the self-linking of the vortex sheet.  Two
    cautions follow, both easy to get backwards:

    * It is blind to *intrinsic* topology.  A wild sheet of genus 217 can carry
      $J = 0$.
    * It does **not** detect knotting.  For the $x_0$-independent spatial
      configuration whose sheet is the torus knot $T(k, g-k)$ one finds
      $J = (-k(g-k), 0, 0, 0)$ --- verified for $T(2,3)$ (trefoil, $-6$),
      $T(2,5)$ ($-10$), $T(3,4) = 8_{19}$ ($-12$), and, decisively, the
      **unknot** $T(1,4)$, which carries $J = -4 \neq 0$.  What $J$ reads is
      the framing $k(g-k)$, which an unknot has just as well as a knot.

    Such $x_0$-independent spatial configurations ($n_0 = 0$, no $x_0$
    dependence) are also the cheapest way to build a nonzero-$J$ background:
    every $dn$ component carrying a 0-index vanishes, so $q \equiv 0$
    identically and the configuration is valid by construction, with
    $J = (\mathrm{CS}(a), 0, 0, 0)$ given by the 3D abelian Chern--Simons sum
    $\mathrm{CS}(a) = \sum_{T^3} a \wedge da$ of the spatial 1-form.  Nonzero
    $J$ is generic among these, already at $N = 6$.
    """

    @staticmethod
    def Villain(S, n):
        r'''The flux of $\mathrm{CS}(n)$ through one 3-cycle per direction, shape ``(4,)``.

        Computed from :func:`~.chern_simons_form` rather than from
        :class:`~.IntersectionCurrent`, and by summing a *single* slice rather
        than averaging over all $N$ of them.  Both choices are about
        exactness: $\mathrm{CS}(n)$ is integer-valued, so a period is a sum of
        integers and is **an exact integer** --- as a secondary characteristic
        class must be --- and this returns it in the integer dtype it deserves,
        with no division to round and no $V$-sized intermediate.  Averaging the
        $N$ slices would divide an integer by $N$ and hand back a float; the
        gauge-invariant representative would be worse still, delivering these
        integers only to floating precision.

        On the constraint surface every slice carries the same flux, so the
        choice of slice is immaterial and this *is* the invariant.  Off it the
        observable is not topological at all (:math:`q \neq 0` makes the flux
        slice-dependent), and this reports the $c = 0$ cycle.
        '''
        L = S.Lattice
        j = chern_simons_form(n)
        return np.array([j[L.comp_index[3][tuple(k for k in range(4) if k != mu)]]
                         .take(0, axis=mu).sum() for mu in range(4)])


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

    No disconnected piece is subtracted: reflecting any direction spanned by
    the transverse 3-cycle flips $J_\mu$'s orientation, so
    $\langle {J}_\mu \rangle = 0$ exactly and $\langle J^2 \rangle$ *is* the
    susceptibility --- subtracting a noisy sample mean would only bias small
    ensembles.  (Restore the subtraction if you ever simulate in a
    reflection-breaking background.)

    Because the winding changes only through topology-shifting moves, check
    its autocorrelation time before trusting error bars: a chain can render it
    *frozen* rather than measured.

    .. note ::

        That freezing is not hypothetical --- it is the normal situation in the
        constrained ensemble, where $J$ moves only when a defect pair winds the
        torus, an excursion whose acceptance falls exponentially in the linear
        size.  A chain can then report $\langle J^2 \rangle = 0$ with *zero*
        error, which is a censored value and not a measurement of the
        stiffness.

        The censoring-free alternative measures the same stiffness away from
        the zero mode: because $j$ is closed on the constraint surface, its
        dual is transverse, so the structure factor of the *gauge-invariant*
        :class:`~.IntersectionCurrent` is the transverse current correlator,
        and its $k \to 0$ intercept is the helicity modulus.  It is computable
        on stored configurations and needs no winding move.  Beware
        fit-window curvature bias in that extrapolation: a wide window
        manufactures a spurious nonzero intercept that shrinks as the window
        does.
    """

    @classmethod
    def autocorrelation(cls, ensemble):
        r'''
        Only included for four-dimensional
        :class:`~supervillain.action.NoIntersections` ensembles: elsewhere the
        slice sums of $j$ are undefined ($D \neq 4$) or not topological
        ($q \neq 0$), and including this observable by default would silently
        change long-standing unconstrained-Villain
        :meth:`~supervillain.Ensemble.autocorrelation_time` computations.
        '''
        S = ensemble.Action
        return (isinstance(S, supervillain.action.NoIntersections)
                and S.Lattice.D == 4
                and super().autocorrelation(ensemble))

    @staticmethod
    def Villain(S, IntersectionWinding):
        r'''Measure $\frac{1}{4}\sum_\mu {J}_\mu^2$.'''
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
    def default(S, Theta_Theta, VacuumTicks, FourDefects):
        V = int(np.prod(S.Lattice.dims))
        S1 = np.sum(Theta_Theta.real) / VacuumTicks
        M2 = V * (1 + S1)
        M4 = (2 * V**2 - V) + 4 * (V - 1) * V * S1 + FourDefects / VacuumTicks
        return 1 - M4 / (2 * M2**2)
