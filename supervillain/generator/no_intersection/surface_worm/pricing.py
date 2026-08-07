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


class JointWeightTable(ReadWriteable):
    r"""A price on **both** counts at once, $\log w(D, Q)$ --- what the factorized
    :class:`Pricing` roster provably cannot express.

    A roster of per-axis prices is a product $w_{dF}(D)\,w_q(Q)$, and flattening the two
    marginals flattens the joint only if the counts are *independent*.  They are not:
    opening a surface on a background at sheet occupancy $\approx 0.44$ **makes**
    intersections.  Measured consequence (q-pricing-2026-08-06 findings 6--8): both
    marginal tables anchor $\log w(0) = 0$ and each flattens its own axis against that
    zero, so their product over-favours the corner they both call cheapest, $(0,0)$ ---
    and neither tuner can see it, because each measures one axis.  The $D$ half then sits
    on one bin, where the multicanonical increment $-\log(H/\bar H)$ is identically zero,
    and its table comes back byte-identical forever.  That is a trap, not slow
    convergence, and no amount of alternation escapes it.

    The object this has to reach is the **corner**: a torus-wrapping sheet lives at
    $D \approx 12$, $Q \approx 16$ *together* (h2-relaxation-2026-08-06 finding 6).

    .. note ::
        :meth:`from_marginals` builds the joint that **is** a given factorized pair, so a
        gas holding it reproduces the factorized sampler to machine precision and the
        joint's introduction is testable rather than a leap --- the same device
        :meth:`~.weights.SectorWeights.fugacity` used for $D$ and :class:`Fugacity` for
        $Q$.  It is machine precision rather than bit-identity because
        $w(D,Q) - w(D_0,Q_0)$ sums the two marginals before differencing where the
        factorized path differences before summing, and float addition is not
        associative.

    Parameters
    ----------
    logWeight: numpy.ndarray
        $\log w(D, Q)$, shape ``(capD + 1, capQ + 1)``.  Normalized to
        $\log w(0,0) = 0$ on construction, since only differences matter.
    tailSlopeD, tailSlopeQ: float
        Continuation slopes past each cap when ``hardWall`` is false.
    hardWall: bool
        Close the window: $w = 0$ beyond either cap, so a move proposing outside it is
        rejected.  Detailed balance survives because $w$ is still a genuine function of
        $(D, Q)$ and the reverse of a forbidden move is equally forbidden.
    """

    def __init__(self, logWeight, tailSlopeD, tailSlopeQ, hardWall=True):
        self.logWeight = np.asarray(logWeight, dtype=float).copy()
        if self.logWeight.ndim != 2:
            raise ValueError(f'a joint table is 2D; got shape {self.logWeight.shape}.')
        self.logWeight -= self.logWeight[0, 0]
        self.tailSlopeD = float(tailSlopeD)
        self.tailSlopeQ = float(tailSlopeQ)
        self.hardWall = bool(hardWall)
        self.capD = self.logWeight.shape[0] - 1
        self.capQ = self.logWeight.shape[1] - 1

    @classmethod
    def from_marginals(cls, sectorWeights, chargeWeights, capD=None, capQ=None):
        r"""The joint that *is* the factorized pair: $\log w(D,Q) = \log w_{dF}(D) +
        \log w_q(Q)$, evaluated on the grid.  Seed a joint tuning from here."""
        capD = sectorWeights.cap if capD is None else int(capD)
        capQ = chargeWeights.cap if capQ is None else int(capQ)
        D = np.arange(capD + 1)
        Q = np.arange(capQ + 1)
        table = np.asarray(sectorWeights(D))[:, None] + np.asarray(chargeWeights(Q))[None, :]
        return cls(table, sectorWeights.tailSlope, chargeWeights.tailSlope,
                   hardWall=bool(sectorWeights.hardWall or chargeWeights.hardWall))

    def __str__(self):
        return (f'JointWeightTable(capD={self.capD}, capQ={self.capQ}, '
                f'range={self.logWeight.min():.2f}..{self.logWeight.max():.2f})')

    def __call__(self, D, Q):
        r"""$\log w(D, Q)$, broadcasting; $-\infty$ outside the window under a hard wall,
        else the two linear tails."""
        D = np.asarray(D)
        Q = np.asarray(Q)
        out = (self.logWeight[np.minimum(D, self.capD), np.minimum(Q, self.capQ)]
               + np.maximum(D - self.capD, 0) * self.tailSlopeD
               + np.maximum(Q - self.capQ, 0) * self.tailSlopeQ)
        if self.hardWall:
            out = np.where((D > self.capD) | (Q > self.capQ), -np.inf, out)
        return out

    def change(self, oldD, oldQ, newD, newQ):
        r"""$\log w(\text{new}) - \log w(\text{old})$ --- what an acceptance needs.

        Not ``delta``: $\delta$ is the codifferential everywhere else here."""
        return float(self(newD, newQ)) - float(self(oldD, oldQ))

    def price(self, D, Q):
        r"""$w(D,Q)/w(0,0)$ on the linear scale --- what a measurement divides out."""
        return np.exp(np.asarray(self(D, Q), dtype=float) - float(self(0, 0)))

    def arrays(self):
        r"""``(logWeight, tailSlopeD, tailSlopeQ, hardWall)``, the plain form the compiled
        kernel takes."""
        return (np.ascontiguousarray(self.logWeight), self.tailSlopeD,
                self.tailSlopeQ, self.hardWall)

    def smoothed(self, length=2.0, rate=0.2, reachable=None):
        r"""A copy relaxed against its own roughness, in **both** directions.

        The 2D analogue of :meth:`~.weights.SectorWeights.smoothed`, and needed for the
        same reason: multicanonical flattening estimates $\log w$ bin by bin from finite
        histograms, so it accumulates high-frequency noise, and a curvature of a few log
        units between adjacent bins is an $e^{\text{few}}$ wall the chain cannot random-walk
        back across --- indistinguishable, in the tuner's own diagnostics, from a converged
        table.  It bites *harder* in 2D, where a $(\text{cap}_D{+}1)(\text{cap}_Q{+}1)$
        grid is filled from the same budget that filled two rows.

        ``reachable`` is a 2D boolean mask of the $(D,Q)$ cells that are geometrically
        possible.  Relaxing across impossible cells averages meaningless bins into their
        neighbours; in 1D those were $D = 1,2,3$, and in 2D the impossible set is large
        and not known in advance, so it is *learned* (the union of everything the chain
        has ever visited) exactly as the 1D tuner learns its own.
        """
        passes = max(1, int(np.ceil(length ** 2 / (4 * rate))))
        work = self.logWeight.copy()
        mask = (np.ones_like(work, dtype=bool) if reachable is None
                else np.asarray(reachable, dtype=bool))
        finite = np.isfinite(work) & mask
        for _ in range(passes):
            lap = np.zeros_like(work)
            lap[1:-1, :] += work[:-2, :] - 2 * work[1:-1, :] + work[2:, :]
            lap[:, 1:-1] += work[:, :-2] - 2 * work[:, 1:-1] + work[:, 2:]
            step = np.where(finite, rate * lap, 0.0)
            work = work + step
        return JointWeightTable(np.where(finite, work, self.logWeight),
                                self.tailSlopeD, self.tailSlopeQ, self.hardWall)
