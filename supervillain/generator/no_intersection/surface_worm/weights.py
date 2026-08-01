r"""Weight tables for the Surface Worm Gas sampler on the Jacobson No Intersections model.

This module provides two weight tables: one for managing the open-surface sector
and one for managing pair separations in the charge sectors.
"""

import numpy as np

from supervillain.h5 import ReadWriteable


class SectorWeights(ReadWriteable):
    r"""$\log w(D)$ on $D = 0 \ldots \texttt{cap}$, closed by a hard wall (or a linear tail).

    A sector weight table $w(D)$ for the `SurfaceWormGas`, indexed by the number of
    open-surface cells $D = \#\{dF \neq 0\}$.

    The gas prices open surfaces with a single fugacity, $\eta_{dF}^D$ --- which is a weight
    **linear in the exponent**, $\log w(D) = D\log\eta_{dF}$.  A linear weight can only shift
    the $D$-distribution, never broaden it, and measurement at $N=8$, $\kappa=0.03$ showed
    that is fatal: a factor 1.1 in $\eta_{dF}$ drops the closed-shell residency by $\geq10^4$
    (0.09 gives $\langle D\rangle=3.7$, $P(D{=}0)=0.33$; 0.10 gives $\langle D\rangle=18.6$,
    $P(D{=}0)<5\times10^{-5}$), and $P(D{=}0)=0.33$ at $\langle D\rangle=3.7$ is nothing like
    Poisson's $e^{-3.7}=0.025$.  So the $D$-distribution is strongly bimodal --- isolated
    bubbles versus a system-spanning open network --- and the gas must cross that to reach the
    configurations whose boundary can wrap the torus and change the topological sector.
    Crossing a suppressed intermediate region is the multicanonical problem, and it needs a
    *non-linear* $\log w(D)$: more than one number, by necessity rather than by choice.

    Whatever a move proposes, $w$ must be a genuine function of $D$ or the acceptance is
    undefined; both settings satisfy that, one by extrapolating and one by vanishing.

    .. note ::
        **A weight on $D$ cannot bias $\Theta$.**  The intersection correlator is a ratio of two
        sector dwells taken **both** on the closed shell $D = 0$ --- the pair sector's divided
        by its price $V\eta_q^2$, the vacuum's by 1 --- so the common factor $w(0)$ cancels
        exactly, for the same reason $\eta_{dF}$ cancels.  What the table changes is which
        configurations the chain *travels through*, never the distribution it samples on the
        shell where the measurement lives.

    .. warning ::
        The default is now the **hard wall**, which changes the sampled ensemble rather than
        only the tuning bookkeeping. See the PROVISIONAL note in ``__init__`` for why the
        linear tail failed and what would have to be rechecked to go back to it.

        $D$ is an imperfect collective variable, and knowingly so.  The $dF\neq0$ cells form the
        **boundary of an open surface** --- an extended, closed object --- so two configurations
        with equal $D$ can be one long torus-wrapping loop or several small bubbles (Evan).  A
        $w(D)$ table averages over exactly the distinction that governs wrapping.  It is the
        right first attempt because it is cheap, learnable, and matches the `DefectGas`
        `sectorWeights` precedent; if flat-in-$D$ sampling still fails to wrap, the collective
        variable is the thing to change, not the tuning.

    .. seealso ::
        ``tune_sector_weights.py`` learns a table by iterative histogram flattening rather than
        by hand.

    Parameters
    ----------
    logWeight: numpy.ndarray
        $\log w(D)$ for $D = 0, \ldots, \texttt{cap}$.  Only *differences* matter, so the
        overall additive constant is irrelevant; it is normalized to $\log w(0) = 0$ on
        construction so tables from different tuning runs are directly comparable.
    tailSlope: float
        $d\log w/dD$ used beyond ``cap`` when ``hardWall`` is false.  For the untuned gas
        this is $\log \eta_{dF}$, which continues the original pricing.
    hardWall: bool
        Close the window at ``cap`` ($w = 0$ beyond) instead of extrapolating.  Default
        true; see ``__init__`` for the measurement that motivated it and the caveats.
    """

    def __init__(self, logWeight, tailSlope, hardWall=True):
        self.logWeight = np.asarray(logWeight, dtype=float).copy()
        self.logWeight -= self.logWeight[0]      # only differences matter; anchor at D=0
        self.tailSlope = float(tailSlope)
        self.cap = len(self.logWeight) - 1
        # PROVISIONAL -- THIS MAY NEED CHANGING.  With hardWall the window [0, cap] is
        # closed: w = 0 beyond it, so any move proposing D > cap is rejected outright.
        #
        # Why it was introduced: the linear tail was an ESCAPE HATCH.  The tuner boosts
        # under-visited bins, so log w rises near the cap; past the cap nothing flattens and
        # the weight merely extrapolates with the fixed, modest slope log(eta_dF) ~ -2.1.
        # Once the edge is boosted high enough that tail stops confining, the walk drifts
        # out and never comes back -- measured at N=6, cap=12: by iteration 49 ALL 3000
        # sampled ticks had D > cap and the interior histogram was empty, so flatness was
        # inf forever and the empty interior bins kept getting the maximum boost.  `cap`
        # was not acting as a cap at all, which also explains the 33-67% `beyond` fractions
        # and P(D=0) coming out identical across different caps.
        #
        # Why it is safe for THIS measurement: a wall is just another choice of w, and
        # Theta is a ratio of two dwells taken BOTH at D = 0, so any common w(0) cancels
        # exactly -- the same argument that makes eta_dF and the learned table harmless.
        #
        # Why it may need changing: it genuinely alters the sampled ensemble rather than
        # only the tuning bookkeeping.  Anything measured OFF the closed shell, or any
        # observable that is not such a ratio, does not enjoy that cancellation.  It also
        # hard-forbids the large-D excursions that the wrapping mechanism needs, so the cap
        # must now be escalated deliberately rather than leaked past -- if wrapping requires
        # D beyond any cap whose window still returns to D = 0, the wall converts a silent
        # leak into an explicit NO WINDOW verdict, which is the honest outcome but a
        # different one. A steeply negative tailSlope is the less invasive alternative and
        # keeps w positive everywhere; it was not chosen only because it leaves a knob that
        # can be set too shallow, which is exactly how this failed.
        self.hardWall = bool(hardWall)

    def __str__(self):
        return (f'SectorWeights(cap={self.cap}, '
                f'range={self.logWeight.min():.2f}..{self.logWeight.max():.2f}, '
                f'tailSlope={self.tailSlope:.4f})')

    @classmethod
    def fugacity(cls, openSurfaceFugacity, cap=64):
        r"""The table equivalent to a bare fugacity, $\log w(D) = D\log\eta_{dF}$.

        This is what the gas already does, written as a table --- so a gas constructed with
        it reproduces the untuned sampler exactly, which is what makes the table's
        introduction testable.
        """
        lg = np.log(openSurfaceFugacity)
        return cls(lg * np.arange(cap + 1), lg)

    def __call__(self, D):
        r"""$\log w(D)$ --- $-\infty$ beyond ``cap`` under a hard wall, else the linear tail."""
        D = np.asarray(D)
        capped = np.minimum(D, self.cap)
        out = (self.logWeight[capped]
               + np.maximum(D - self.cap, 0) * self.tailSlope)
        if self.hardWall:
            out = np.where(D > self.cap, -np.inf, out)
        return out

    def delta(self, old, new):
        r"""$\log w(\texttt{new}) - \log w(\texttt{old})$ --- what an acceptance needs."""
        return float(self(new)) - float(self(old))

    def arrays(self):
        r"""``(logWeight, tailSlope, hardWall)`` in the plain form the numba kernel takes."""
        return np.ascontiguousarray(self.logWeight), self.tailSlope, self.hardWall

    def flatness(self, histogram):
        r"""How flat a visit histogram is over the bins that were visited at all.

        The tuning target: ``max/min`` over occupied bins, which is 1 for a perfectly flat
        random walk in $D$.  Returns ``inf`` if any bin in $[0, \texttt{cap}]$ was never
        visited, since an unvisited bin is a barrier the table has not yet flattened.
        """
        h = np.asarray(histogram, dtype=float)[:self.cap + 1]
        if (h <= 0).any():
            return np.inf
        return float(h.max() / h.min())

    def interpolated(self, visited):
        r"""A copy with $\log w$ filled in across bins the tuning never visited.

        The tuner can only learn $\log w(D)$ where the chain went, and it leaves every
        other bin alone --- which makes an unvisited bin **self-perpetuating**: it is never
        boosted, so it is never reached, so it stays unvisited.  A $D$ sitting behind a
        barrier the chain did not happen to cross is then permanently indistinguishable
        from one that is geometrically impossible, and the states that matter (the large-$D$
        configurations whose boundary can wrap) are exactly the ones most likely to be
        missed.

        Linear interpolation between the learned neighbours is the right repair because
        $\log w \approx -\log\Omega(D)$ and the density of states is smooth in $D$: a gap
        gets the value its neighbours imply rather than a stale one, so the chain is offered
        the move and the *sampling* decides whether the state exists.  Bins past the last
        visited one keep the linear tail, which is already how the table extrapolates.

        .. note ::
            This cannot bias $\Theta$.  The correlator is a ratio of two dwells taken both
            on the closed shell, so any $w$ cancels there --- a wrong interpolation costs
            efficiency, never correctness.

        Parameters
        ----------
        visited: array of bool
            Which $D$ the tuning actually sampled.  Interpolation happens across the rest.
        """
        visited = np.asarray(visited, dtype=bool)[:self.cap + 1]
        known = np.flatnonzero(visited)
        if len(known) < 2:
            # Nothing to interpolate between; hand back an unchanged copy rather than
            # inventing a curve from one point.
            return SectorWeights(self.logWeight, self.tailSlope, self.hardWall)
        D = np.arange(self.cap + 1)
        filled = np.interp(D, known, self.logWeight[known])
        # Past the last learned bin, np.interp would hold the endpoint flat, which is a
        # *different* extrapolation from the tail this table promises everywhere else.
        # Continue the tail slope instead so log w stays one consistent function of D.
        last = known[-1]
        filled[last + 1:] = self.logWeight[last] + (D[last + 1:] - last) * self.tailSlope
        return SectorWeights(filled, self.tailSlope, self.hardWall)


class PairUmbrella(ReadWriteable):
    r"""$\log w_2(r^2)$ on the $\pm1$ pair sector; identically 1 everywhere else.

    A pair-separation umbrella $w_2(r)$ for the charge-$\pm1$ sector of the SurfaceWormGas.

    The correlator's large-$|\Delta x|$ bins are censored because a defect pair at separation $r$
    has equilibrium weight $\propto \Theta(r)$, which is small --- and *how* small is the very
    thing being measured, so no cost argument here assumes a functional form.  The fix is the
    same one used on $D$: bias toward the rare states and divide the bias out.

    Indexed by the **squared** minimal-image separation $r^2$, which is the integer the lattice
    actually produces and what ``MaxPairSeparationSquared`` already reports; $r^2$ runs
    $0 \ldots N^2$.

    .. warning ::
        **$w_2$ does not cancel from $\Theta$, unlike $w(D)$.**  $w(D)$ divides out because the
        pair dwell and the vacuum dwell both sit at $D=0$, so a common factor cancels in the
        ratio.  $w_2$ is $r$-dependent while the numerator is $r$-resolved, so it must be divided
        out **bin by bin** in the accumulator.  Getting that wrong does not look like a bug --- it
        looks like a correlator *shape*.

    .. note ::
        **Do not punish leaving the pair sector** (Evan).  If $w_2 > 1$ at large $r$ --- the whole
        point --- then a configuration out at large separation carries a large weight and any move
        *out* of the $Q=2$ sector costs $1/w_2(r)$.  The chain pins at large $r$ and stops
        returning to $Q=0$, which is fatal rather than slow: the vacuum dwell is $\Theta$'s
        denominator, so the measurement destroys itself exactly when the umbrella starts working.
        :meth:`normalized` fixes the sector's *mean* weight to 1 so entering and leaving are
        unbiased on average, and ``VacuumReturns`` is the run-time check that it worked.

    Parameters
    ----------
    logWeight: numpy.ndarray
        $\log w_2$ for $r^2 = 0, \ldots, N^2$.  Only *differences within the sector* carry
        meaning, exactly as for :class:`~.SectorWeights`.
    N: int
        Linear lattice size, so the table's span can be checked against the geometry.
    """

    def __init__(self, logWeight, N):
        self.N = int(N)
        self.logWeight = np.asarray(logWeight, dtype=float).copy()
        expected = self.N ** 2 + 1
        if len(self.logWeight) != expected:
            raise ValueError(
                f'PairUmbrella needs one entry per r^2 in [0, N^2] = {expected} for N={N}; '
                f'got {len(self.logWeight)}.  The table is indexed by SQUARED minimal-image '
                'separation, not by r.')

    @classmethod
    def off(cls, N):
        r"""The identity umbrella.  A gas built with this must reproduce the un-umbrella'd
        sampler **exactly**, which is gate 1."""
        return cls(np.zeros(N ** 2 + 1), N)

    def __str__(self):
        return (f'PairUmbrella(N={self.N}, '
                f'range={self.logWeight.min():.2f}..{self.logWeight.max():.2f})')

    def logW(self, r2):
        r"""$\log w_2(r^2)$; 0 when ``r2`` is None, i.e. not in the $\pm1$ pair sector."""
        if r2 is None:
            return 0.0
        return float(self.logWeight[r2])

    def delta(self, oldR2, newR2):
        r"""$\log w_2(\text{new}) - \log w_2(\text{old})$ for an acceptance.

        Either argument may be None (the move enters or leaves the pair sector), and the
        missing side contributes 0 --- i.e. the weight really is 1 outside the sector, which
        is what makes it a state function and keeps detailed balance intact.
        """
        return self.logW(newR2) - self.logW(oldR2)

    def weight(self, r2):
        r"""$w_2$ itself --- what the accumulator divides by, per bin."""
        return float(np.exp(self.logW(r2)))

    def normalized(self, occupancy):
        r"""Rescale so the sector's mean weight is 1 under the measured ``occupancy``.

        ``occupancy[r2]`` is the observed number of pair-sector ticks at that separation.
        Setting $\sum \Omega(r^2)\,w_2(r^2) = \sum \Omega(r^2)$ leaves the $Q=2$ sector's
        total weight unchanged relative to everything else, so the umbrella reshapes the
        distribution over $r$ without making the sector as a whole more or less attractive.
        Without it, a table that enhances large $r$ also enhances *being in the pair sector at
        all*, which is the pinning failure above.
        """
        occ = np.asarray(occupancy, dtype=float)
        if occ.shape != self.logWeight.shape:
            raise ValueError(f'occupancy has shape {occ.shape}, table {self.logWeight.shape}')
        total = occ.sum()
        if total <= 0:
            return PairUmbrella(self.logWeight, self.N)
        mean = float((occ * np.exp(self.logWeight)).sum() / total)
        if not np.isfinite(mean) or mean <= 0:
            return PairUmbrella(self.logWeight, self.N)
        return PairUmbrella(self.logWeight - np.log(mean), self.N)


    def shifted(self, delta):
        r"""The same shape with $\log w_2$ moved by a constant.

        A constant cancels from $\Theta$ (numerator and denominator both carry $1/Z_w$), so
        this changes *only* how attractive the pair sector is relative to everything else --
        which is the one knob that enforces the don't-punish-leaving constraint.
        """
        return PairUmbrella(self.logWeight + float(delta), self.N)

    def rebalanced(self, sectorTicks, totalTicks, targetOdds):
        r"""Shift the table so the pair sector's occupancy odds return to ``targetOdds``.

        :meth:`normalized` is a *one-shot* rescale against a previously measured occupancy,
        and it is not enough: the umbrella exists to change that occupancy, so the sector
        ends up more attractive than the rescale anticipated.  Measured at $N=8$,
        $\kappa=0.03$ once the coboundary heatbath also carried $w_2$, a `normalized` table
        drove the $Q=2$ occupancy from 30% to **88%** and the vacuum dwell from 1707 ticks
        to **134** --- the pinning failure, arriving through the sector's *total* weight
        rather than through its shape.

        Shifting by $-\log(\text{odds}_\text{now}/\text{odds}_\text{target})$ is the
        first-order correction: to the extent the sector's weight acts like a fugacity, the
        occupancy odds respond exponentially to a constant in $\log w_2$.  It is a control
        loop, so it is iterated rather than trusted in one step.
        """
        inSector = float(sectorTicks)
        out = float(totalTicks) - inSector
        if inSector <= 0 or out <= 0 or targetOdds <= 0:
            return PairUmbrella(self.logWeight, self.N)
        return self.shifted(-np.log((inSector / out) / targetOdds))


def pair_separation_squared(chargeSites, N):
    r"""Squared minimal-image separation of a $\pm1$ pair, or None if not in that sector.

    Deliberately the **same** condition ``CorrelatorAccumulator.tick`` uses --- exactly two
    charged cells carrying $+1$ and $-1$.  If the umbrella weighted a sector the accumulator
    does not bin (or vice versa), the division would not undo the bias.
    """
    if chargeSites is None or len(chargeSites) != 2:
        return None
    (h1, c1), (h2, c2) = chargeSites.items()
    if {c1, c2} != {1, -1}:
        return None
    return int(sum(min((a - b) % N, (b - a) % N) ** 2 for a, b in zip(h1, h2)))
