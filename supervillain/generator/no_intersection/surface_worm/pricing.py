r"""How the :class:`~.SurfaceWormGas` prices a defect count.

The gas's extended weight is a product of *prices*, one per relaxed constraint:

.. math ::

    \pi_\text{ext}(F) \propto e^{-2\pi^2\kappa C(F)}\cdot w_{dF}\big(D(F)\big)\cdot w_q\big(Q(F)\big)

with $D = \#\{dF \neq 0\}$ and $Q = \#\{q \neq 0\}$.  A price is any function of one
count; this module names the interface (:class:`Pricing`) and supplies the *simple* kind
(:class:`Fugacity`), while the *general* kind is the tuned table
:class:`~.weights.SectorWeights`, re-exported here as :data:`WeightTable` because nothing
about it is specific to the open-surface axis.

**Why the distinction earns a class.**  A fugacity is $\log w(n) = n\log\eta$ ---
**linear in the exponent**, so it can *shift* a count's distribution but never *broaden*
it.  Crossing a suppressed intermediate region is the multicanonical problem and needs a
non-linear $\log w$: more than one number, by necessity rather than by choice.  The
open-surface axis learned this and got :class:`~.weights.SectorWeights`; the intersection
axis kept a bare scalar, even though the measurements say it is the binding one (a
torus-wrapping sheet's saddle at $N=6$, $\kappa=0.03$ is ~88% intersection price and ~1%
action).  Naming the interface is what lets the two axes be priced by different kinds
without the gas caring which it holds.

.. note ::

    **A per-axis price cannot express a joint one.**  This interface is a function of a
    *single* count, so a roster of them is by construction a product
    $w_{dF}(D)\,w_q(Q)$ --- factorized.  Tuning both to flatten their **marginals** is
    therefore the most this design can reach, and a genuine $w(D,Q)$ that boosts the
    *corner* (large $D$ **and** large $Q$ together, which is where a wrapping sheet
    lives) is outside it.  That is a deliberate first step, not an oversight: the
    factorized form is cheap to tune ($\mathcal{O}(\text{cap})$ bins per axis against
    $\mathcal{O}(\text{cap}^2)$), and it should be measured before a joint table is
    built.  A joint price would enter as a *different* protocol --- one taking a tuple of
    counts --- and the seam is here.

.. seealso ::

    :class:`~.weights.SectorWeights` for the tuned-table kind and its regularization
    (:meth:`~.weights.SectorWeights.smoothed`,
    :meth:`~.weights.SectorWeights.interpolated`), and the tuners that fit it.
"""

import numpy as np

from supervillain.h5 import ReadWriteable

from .weights import SectorWeights

#: The general kind of :class:`Pricing` --- an arbitrary tabulated $\log w$.  This is
#: :class:`~.weights.SectorWeights` under a name that does not presume the open-surface
#: axis; the class is index-agnostic and is used for both $D$ and $Q$.
WeightTable = SectorWeights


class Pricing(ReadWriteable):
    r"""The interface the gas requires of a price on one defect count.

    Implementations must provide

    ``__call__(n)``
        $\log w(n)$, broadcasting over an array of counts.
    ``change(old, new)``
        $\log w(\texttt{new}) - \log w(\texttt{old})$ as a float --- what an acceptance
        needs, and the only method the sampler's hot path calls.  Not ``delta``: $\delta$
        is the **codifferential** everywhere else here
        (:func:`supervillain.lattice.delta`), and a weight table's finite difference is
        not that.
    ``arrays()``
        ``(logWeight, tailSlope, hardWall)``, the plain form the compiled kernel takes.
        Every price is representable this way: a table on $[0, \texttt{cap}]$ plus a rule
        past it.
    ``price(n)``
        $w(n)/w(0)$, the *linear-scale* price of $n$ defects.  Measurements that undo the
        pricing to recover a physical dwell ratio divide by this rather than by a power of
        a fugacity, which is what makes them correct for either kind.
    ``cap``
        The largest count the table resolves.

    :class:`Fugacity` and :data:`WeightTable` are the two kinds.  **This is a documented
    protocol, not an enforced base class**: the gas duck-types, and
    :class:`~.weights.SectorWeights` cannot inherit from here because this module imports
    *it* (the general kind is the tuned table, so the dependency runs that way and a base
    class would close the cycle).  Implement the five members above and the gas will take
    it.
    """


class Fugacity(WeightTable, Pricing):
    r"""The simple kind of price: a single number, $\log w(n) = n\log\eta$.

    Stored *as a table* (:class:`WeightTable` with slope $\log\eta$ and no wall) so the
    compiled kernel has one code path and the fugacity is exactly the affine special
    case --- the same device
    :meth:`SectorWeights.fugacity <.weights.SectorWeights.fugacity>` used when the
    open-surface axis migrated.

    .. warning ::

        A tuned copy of a fugacity is **not** a fugacity.  :meth:`~.weights.SectorWeights.smoothed`
        and :meth:`~.weights.SectorWeights.interpolated` therefore hand back a plain
        :data:`WeightTable`: once the table stops being affine there is no $\eta$ to
        report, and pretending otherwise would leave :attr:`eta` describing a curve it no
        longer generates.

    Parameters
    ----------
    eta: float
        The fugacity $\eta \in (0, 1]$ --- one defect's price.
    cap: int
        How far to tabulate.  Beyond it the linear tail continues with the same slope, so
        unlike a tuned table the cap is bookkeeping only and does not bound the count.
    """

    def __init__(self, eta, cap=64):
        eta = float(eta)
        if not eta > 0:
            raise ValueError(f'a fugacity must be positive, got {eta}.')
        lg = np.log(eta)
        super().__init__(lg * np.arange(cap + 1), lg, hardWall=False)
        self.eta = eta

    def __str__(self):
        return f'Fugacity(eta={self.eta:g}, cap={self.cap})'

    def price(self, n):
        r"""$\eta^n$ --- computed as a power, not as $e^{n\log\eta}$.

        The difference is a rounding in the last place, and it is the difference between a
        measurement that reproduces the pre-pricing sampler exactly and one that
        reproduces it nearly.  The accumulator divides its sector dwells by this.
        """
        return np.asarray(self.eta, dtype=float) ** np.asarray(n)
