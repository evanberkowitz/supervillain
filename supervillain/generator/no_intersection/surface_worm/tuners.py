#!/usr/bin/env python

r"""Multicanonical tuners for the two weight tables the
:class:`~.gas.SurfaceWormGas` reads: :class:`~.weights.SectorWeights` (the
open-surface count $D$) and :class:`~.weights.PairUmbrella` (the $\pm1$ pair
separation $r^2$).

Both tuners are histogram-flattening control loops: run the gas, look at
where it went, boost the bins it under-visited, repeat.  Neither can ever
bias the physical answer the gas exists to compute --- :class:`SectorWeights`
cancels from $\Theta$ because both the pair and vacuum dwells sit on the
closed shell $D = 0$ (its docstring), and a :class:`PairUmbrella` is divided
back out bin-by-bin by :class:`~.accumulator.CorrelatorAccumulator` (its
docstring's warning).  A badly tuned table only costs efficiency, never
correctness --- which is what lets these tuners run unattended and be judged
purely on convergence diagnostics rather than against a reference.

.. note ::

    A straight port of the audited standalone scripts
    (``tune_sector_weights.py``'s ``visit_histogram``/``tune``,
    ``tune_pair_umbrella.py``'s ``achievable_shells``/``measure``/``update``/
    ``OffsetBisector``/``coverage``, and ``transport_tuner.py``'s
    ``stage``/``score_flips``/``__main__`` decision loop, in the
    no-intersections lab notebook's ``j-vacuum-2026-07-31`` snapshot), with
    the free functions folded into methods of three tuner classes and the
    dict-keyed ``cfg`` re-expressed as :class:`~.state.FState`.  See
    ``test_surface_worm_tuners.py`` for the gates.
"""

import time

import numpy as np

from .gas import SurfaceWormGas
from .state import FState
from .pricing import Fugacity, JointWeightTable
from .weights import SectorWeights, PairUmbrella


class SectorWeightTuner:
    r"""Learn a :class:`~.weights.SectorWeights` table by iterative
    histogram flattening.

    The goal is a $w(D)$ under which the gas performs a **random walk in
    $D$** over $[0, \texttt{cap}]$ instead of sitting in one lobe of a
    bimodal distribution (see the "PROVISIONAL" note in
    :class:`~.weights.SectorWeights.__init__` for the measurement that shows
    a bare fugacity is bimodal). That is what lets a single chain hold both
    the closed shell $D = 0$ --- where the correlator is measured --- and
    the large-$D$ configurations whose boundary can wrap the torus and
    change the topological sector.

    The update is the standard multicanonical one.  Under weight $w$ the
    visit histogram is $H(D) \propto \Omega(D)\,w(D)$ with $\Omega$ the
    (unknown) density of states, so

    .. math ::
        \log w_{\text{new}}(D) = \log w(D) - \log H(D)

    leaves $\Omega\,w_\text{new}$ flat to the accuracy of $H$.  Iterating
    converges on $w \approx 1/\Omega$.  Only *differences* of $\log w$
    matter, so each iteration is anchored at $\log w(0) = 0$
    (:class:`~.weights.SectorWeights` does this on construction).

    .. note ::
        **A learned table cannot bias $\Theta$**, however badly it is
        tuned.  The correlator is a ratio of two dwells taken **both** on
        the closed shell, so the common factor $w(0)$ cancels exactly ---
        the same argument that makes the bare open-surface fugacity
        harmless. A bad table costs efficiency, never correctness.

    .. warning ::
        Convergence is judged on the **occupied** range only, and an
        unvisited bin inside $[0, \texttt{cap}]$ is reported rather than
        silently averaged over --- it is a barrier the table has not yet
        flattened. ``flatness`` (see :meth:`tune`'s per-iteration history)
        is ``inf`` in that case.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The action; sets $N$ and $\kappa$.
    intersectionFugacity: float
        Self-intersection fugacity $\eta_q$, left alone --- only the
        open-surface sector is retuned.
    cap: int
        Flatten over $D \in [0, \texttt{cap}]$.  This is the knob that
        decides how far open the chain is encouraged to go, and therefore
        whether it can wrap.
    openSurfaceFugacity: float
        Starting fugacity for the initial (linear) table
        (``SectorWeights.fugacity(openSurfaceFugacity, cap=cap)``).
    iterations, ticks, stride: int
        Flattening iterations, samples per iteration, and compiled moves
        between samples.
    damping: float
        Fraction of $-\log H$ applied per iteration.  1.0 is the plain
        update; below 1 trades convergence speed for stability when the
        histogram is thin.
    targetFlatness: float
        Stop when ``max/min`` over the reachable range falls below this.
    minimumReachable: int
        How many distinct $D$ must be known before a flat histogram counts
        as converged.  Flatness over one visited bin is 1.0 *by
        definition*, so without this a first iteration that never leaves
        $D=0$ declares victory and returns the seed table untouched.
    pCob: float
        Coboundary-move probability passed to every :meth:`~.gas.SurfaceWormGas.sweep`.
    targetFraction: float
        Defect-adjacent proposal targeting, passed to the internal gas.  A
        table is learned from one chain's visit histogram, so tuning at one
        ``targetFraction`` and sampling at another learns the wrong table
        --- this **must match** the value the tuned table will be used
        with.
    seed: int, optional
        Seeds both the compiled kernel's RNG (``seed=``) and the python
        reference stream (``rng=numpy.random.default_rng(seed)``) of the
        internal gas --- passing only one seeds the kernel alone and leaves
        the tuner's own draws (none here, but any future python-side move)
        non-reproducible; see ``test_surface_worm_sweep.py`` for the gap
        this closes.

    Attributes
    ----------
    history: list of dict
        One entry per completed iteration:
        ``iteration``, ``flatness`` (over the reachable set), ``visited``
        (bins occupied this iteration), ``occupiedFlatness`` (flatness over
        just this iteration's occupied bins), ``reachable`` (bins ever
        seen), ``beyond`` (ticks that landed past ``cap``), ``histogram``
        (the raw counts), ``logWeight`` (the table in force during this
        iteration), ``seconds``.
    """

    #: Which count this tuner flattens.  Everything else in the class --- the
    #: multicanonical increment, the learned reachable set, the smoothing, the
    #: frontier-static convergence test --- is axis-agnostic, so a tuner for the
    #: other axis is these four hooks and nothing more (:class:`ChargeWeightTuner`).
    axis = 'D'

    def _count(self, state):
        r"""The count this tuner histograms, read off the chain's state."""
        return int(state.D)

    def _set_price(self, gas, weights):
        r"""Install the learned table as the gas's price on :attr:`axis`."""
        return gas.setSectorWeights(weights)

    def _gas_kwargs(self, weights):
        r"""How to build the internal gas with ``weights`` on :attr:`axis` and the
        *other* axis held at whatever this tuner was given.

        The other axis may be pinned to a learned ``chargeWeights`` table rather than a
        fugacity, which is what the **alternation** of the two marginals needs: each
        tuner must see the other's current table, or the two halves are each tuned
        against a background the composed sampler never has.
        """
        if self.chargeWeights is not None:
            return dict(sectorWeights=weights, chargeWeights=self.chargeWeights)
        return dict(sectorWeights=weights,
                    intersectionFugacity=self.intersectionFugacity,
                    sectorWeightCap=self.cap)

    def _seed_table(self):
        r"""The table to flatten from when no ``seedWeights`` is given: the bare
        fugacity written out, so iteration 0 **is** the untuned sampler."""
        return SectorWeights.fugacity(self.openSurfaceFugacity, cap=self.cap)

    def __init__(self, S, intersectionFugacity, cap, openSurfaceFugacity=0.09,
                 iterations=20, ticks=2000, stride=200, damping=0.7,
                 targetFlatness=1.5, minimumReachable=3, pCob=0.6,
                 targetFraction=0.8, seed=None, gasFactory=None,
                 seedWeights=None, smoothUpdate=2.0, smoothTable=0.0, smoothFinal=2.0,
                 warmStart=None, chargeWeights=None):
        # gasFactory: a SurfaceWormGas-compatible callable (e.g. a subclass, or
        # functools.partial with extra knobs preset).  A tuner must tune the
        # sampler that will actually run -- a table tuned against plain-SWG
        # kinetics is mistuned for a sampler with different kinetics.
        self.gasFactory = gasFactory if gasFactory is not None else SurfaceWormGas
        # warmStart: a configuration dict to begin the internal F-chain FROM,
        # instead of the trivial vacuum F = 0.  A tuner measures its histogram
        # on whatever background its chain reaches, so a chain that cannot
        # nucleate out of the empty lattice within the tuning budget tunes a
        # table for a background that does not exist -- the cold-start
        # pathology the vortex-sector work named "cold-start excursion
        # physics".  Inside the hysteresis band, or at any (N, kappa) where
        # nucleation is slow, hand this an equilibrated configuration.
        self.warmStart = warmStart
        # chargeWeights: pin the OTHER axis to a learned table instead of the bare
        # intersectionFugacity.  Only meaningful when alternating the two marginals;
        # None keeps the historical behaviour exactly.
        self.chargeWeights = chargeWeights
        # seedWeights: an initial table to flatten FROM, instead of the bare
        # fugacity table.  Wide-range (large-cap) flattening is the standard
        # multicanonical range problem; staging cap upward, seeding each stage
        # from the previous stage's learned table (extended by its tail
        # slope), is how a cap far beyond the bare table's reach is tuned.
        self.seedWeights = seedWeights
        # Relaxation passes applied to each iteration's INCREMENT (not to the
        # accumulated table): the increment is where the fresh estimator noise
        # lives, and one part of it is discontinuous BY CONSTRUCTION -- bins
        # reachable but unvisited this iteration get a flat boost equal to the
        # largest correction any visited bin got, so neighbours can differ by
        # that whole boost in one step.  Smoothing the increment denoises the
        # new information without repeatedly filtering what the table already
        # learned.  Rough tables also feed back: a kink distorts where the
        # chain explores next iteration, making the next histogram worse.
        # both in BINS (a diffusion length), not pass counts -- see
        # SectorWeights.smoothed; ~2 bins damps the bin-scale estimator noise
        # and the by-construction discontinuity in the unvisited-bin boost,
        # while leaving real structure alone.  0 disables.
        self.smoothUpdate = float(smoothUpdate)
        # smoothTable: relax the ACCUMULATED table every iteration too, not just
        # the increment (regularized multicanonical).  Keeps the chain sampling
        # under a smooth table throughout, so every histogram is gathered on a
        # well-behaved landscape -- the stronger form of breaking the
        # roughness feedback loop.  The cost is that repeated filtering
        # compounds: the fixed point becomes "as flat as the smoothing allows",
        # so the table cannot represent structure finer than this length.  Safe
        # while that length stays well below the scale on which log rho varies.
        self.smoothTable = float(smoothTable)
        self.smoothFinal = float(smoothFinal)
        self.S = S
        self.intersectionFugacity = float(intersectionFugacity)
        self.cap = int(cap)
        self.openSurfaceFugacity = float(openSurfaceFugacity)
        self.iterations = int(iterations)
        self.ticks = int(ticks)
        self.stride = int(stride)
        self.damping = float(damping)
        self.targetFlatness = float(targetFlatness)
        self.minimumReachable = int(minimumReachable)
        self.pCob = float(pCob)
        self.targetFraction = float(targetFraction)
        self.seed = seed
        self.history = []

    def visit_histogram(self, gas, state, ticks, stride, cap, pCob=0.6):
        r"""Histogram of :attr:`axis` over ``ticks`` samples spaced ``stride``
        compiled moves apart.

        Uses the compiled :meth:`~.gas.SurfaceWormGas.sweep`, so the tuner is
        affordable: flattening needs many iterations and each needs enough
        statistics that $\log H$ is not itself noise.

        Parameters
        ----------
        gas: supervillain.generator.no_intersection.surface_worm.gas.SurfaceWormGas
            The (already-constructed) gas to advance.
        state: supervillain.generator.no_intersection.surface_worm.state.FState
            Mutated in place; the chain carries its state across calls.
        ticks: int
            Number of samples.
        stride: int
            Compiled moves between samples.
        cap: int
            A count above this is tallied in ``beyond`` instead of the histogram.
        pCob: float
            Coboundary-move probability.

        Returns
        -------
        (numpy.ndarray, int)
            ``(hist, beyond, vacuum)`` --- counts for the tuned count
            $0 \ldots \texttt{cap}$, how many samples landed beyond the cap, and how
            many sat in the joint vacuum $D = Q = 0$ (where emission happens).
        """
        hist = np.zeros(cap + 1, dtype=np.int64)
        beyond = vacuum = 0
        for _ in range(ticks):
            gas.sweep(state, stride, pCob=pCob)
            n = self._count(state)
            if n <= cap:
                hist[n] += 1
            else:
                beyond += 1
            # The joint vacuum is where emission happens, so a table that opens its own
            # axis and never comes back is WORSE than the fugacity it replaced -- the
            # cap-affordability tension the TransportTuner measures on D (cap 30: 20%
            # corner dwell, zero vacuum returns).  Counted here so no flattening run can
            # report success without it having been looked at.
            vacuum += (state.D == 0 and state.Q == 0)
        return hist, beyond, vacuum

    def tune(self, log=print):
        r"""Run the flattening loop; return the learned table.

        Builds **one** internal gas and reuses it across iterations via
        :meth:`~.gas.SurfaceWormGas.setSectorWeights` (the usual
        multicanonical warm start) --- rebuilding the gas every iteration
        would repay the $O(6V^2)$ ``windingSensitivity`` precompute each
        time, which dominates the whole run and puts $N \geq 8$ out of
        reach (see :meth:`~.gas.SurfaceWormGas.setSectorWeights`'s
        docstring).

        Parameters
        ----------
        log: callable, optional
            Called with one string per iteration (and at convergence /
            completion); defaults to :func:`print`.

        Returns
        -------
        SectorWeights
            The learned table, interpolated across any bin in
            $[0, \texttt{cap}]$ the chain never visited (see
            :meth:`~.weights.SectorWeights.interpolated`).
        """
        weights = (self.seedWeights if self.seedWeights is not None
                   else self._seed_table())
        if weights.cap != self.cap:
            raise ValueError(
                f'seedWeights has cap {weights.cap} but the tuner was built with '
                f'cap {self.cap}; extend the seed table (tail slope) before seeding.')
        self.history = []
        # Which D are REACHABLE at all is LEARNED as the union of everything ever
        # seen, and flatness is judged on that set alone -- see SectorWeights'
        # module docstring / the source script's comment for why treating a
        # geometrically-empty bin as an unflattened barrier destabilizes the table.
        everSeen = np.zeros(self.cap + 1, dtype=bool)
        sinceExpansion = 0          # iterations since the reachable set last grew

        gas = self.gasFactory(self.S, targetFraction=self.targetFraction,
                              measure=False, seed=self.seed,
                              rng=np.random.default_rng(self.seed),
                              **self._gas_kwargs(weights))
        state = (FState.from_configuration(self.S, self.warmStart)
                 if self.warmStart is not None else FState(self.S))

        for it in range(self.iterations):
            self._set_price(gas, weights)
            t0 = time.time()
            gas.sweep(state, 5 * self.stride, pCob=self.pCob)     # brief settle
            hist, beyond, vacuum = self.visit_histogram(gas, state, self.ticks,
                                                        self.stride, self.cap,
                                                        pCob=self.pCob)
            grew = bool((~everSeen & (hist > 0)).any())
            everSeen |= hist > 0
            sinceExpansion = 0 if grew else sinceExpansion + 1
            reachable = hist[everSeen]
            flat = (float(reachable.max() / reachable.min()) if (reachable > 0).all()
                    else np.inf)
            seenNow = hist > 0
            occupied = int(seenNow.sum())
            occFlat = (float(hist[seenNow].max() / hist[seenNow].min())
                       if seenNow.any() else np.inf)
            d2 = np.diff(np.diff(weights.logWeight))
            self.history.append(dict(
                iteration=it, flatness=flat, occupiedFlatness=occFlat,
                roughness=float(np.abs(d2).mean()), maxKink=float(np.abs(d2).max()),
                visited=occupied, reachable=int(everSeen.sum()), beyond=beyond,
                histogram=hist.copy(), logWeight=weights.logWeight.copy(),
                vacuumFraction=vacuum / max(1, self.ticks),
                seconds=time.time() - t0))
            log(f'  iter {it:>3d}  occupied {occupied:>3d}/{int(everSeen.sum())} reachable  '
                f'beyond cap {beyond:>5d}  flatness {flat:>9.3g}  '
                f'(this iter {occFlat:>7.2f})  vacuum {vacuum / max(1, self.ticks):>6.3f}  '
                f'({time.time() - t0:.0f}s)')

            known = int(everSeen.sum())
            if flat < self.targetFlatness and known >= self.minimumReachable and sinceExpansion >= 2:
                log(f'  converged: flatness {flat:.3g} < {self.targetFlatness} over '
                    f'{known} reachable D, frontier static for {sinceExpansion} iterations')
                break
            if flat < self.targetFlatness and (known < self.minimumReachable or sinceExpansion < 2):
                log(f'  flat ({flat:.3g}) but NOT converged: {known} reachable D '
                    f'(need {self.minimumReachable}), frontier static {sinceExpansion} '
                    '(need 2) -- continuing')

            # log w -= log H on the visited bins; unvisited-but-reachable bins get
            # the largest boost any visited bin received (never bins never seen at
            # all -- boosting those forever is what made the table oscillate).
            update = np.zeros(self.cap + 1)
            seen = hist > 0
            update[seen] = -np.log(hist[seen] / hist[seen].mean())
            missed = everSeen & ~seen
            if missed.any():
                update[missed] = update[seen].max() if seen.any() else 1.0
            if self.smoothUpdate and len(update) >= 11:
                lo = int(np.argmax(everSeen))
                hi = int(len(everSeen) - np.argmax(everSeen[::-1]))
                seg = update[lo:hi].copy()
                for _ in range(max(1, int(np.ceil(self.smoothUpdate ** 2 / 0.5)))):
                    if len(seg) < 3:
                        break
                    lap = np.zeros_like(seg)
                    lap[1:-1] = seg[:-2] - 2 * seg[1:-1] + seg[2:]
                    seg += 0.25 * lap
                update[lo:hi] = seg
            step = self.damping / (1.0 + sinceExpansion / 3.0)
            newLog = weights.logWeight + step * update
            weights = SectorWeights(newLog, weights.tailSlope, weights.hardWall)
            if self.smoothTable:
                weights = weights.smoothed(length=self.smoothTable, reachable=everSeen)

        if self.smoothFinal:
            # only among reachable D: D = 1,2,3 (and 5) are geometrically
            # impossible, and smoothing across them corrupts w(0)
            weights = weights.smoothed(length=self.smoothFinal, reachable=everSeen)
        unreachable = np.flatnonzero(~everSeen)
        if len(unreachable):
            log(f'  {self.axis} never visited in [0, {self.cap}]: {list(unreachable)} '
                '(interpolated, NOT assumed empty -- see SectorWeights.interpolated)')
        return weights.interpolated(everSeen)


class ChargeWeightTuner(SectorWeightTuner):
    r"""Learn a $w(Q)$ table on the **intersection** count, by the same flattening
    :class:`SectorWeightTuner` runs on the open-surface count.

    Everything of substance is inherited: the multicanonical increment, the *learned*
    reachable set, the damped step, the smoothed increment and table, the
    frontier-static convergence test, and the refusal to boost bins never seen at all
    (which is what stops the table oscillating).  Only four hooks differ --- which count
    to read, which price to set, how to build the gas, and what to flatten from.

    **Why this axis needs a table at all.**  $Q$ was priced by a single fugacity, which
    is linear in the exponent and so can *shift* the $Q$ distribution but never *broaden*
    it --- the argument :class:`~.weights.SectorWeights` was created to answer one axis
    over.  And $Q$ is the axis that binds: growing a torus-wrapping sheet on a production
    $N=6$, $\kappa=0.03$ background has a saddle that is ~88% intersection price and ~1%
    action, needing $Q \approx 16$ against a fugacity that holds the chain at $Q \le 2$.

    .. warning ::
        **Watch ``vacuumFraction`` in the history, not just flatness.**  Emission is
        gated on the joint vacuum $D = Q = 0$, so a $w(Q)$ that opens its own axis and
        does not come back is *worse* than the fugacity it replaced.  This is the exact
        analogue of the cap-affordability tension :class:`TransportTuner` measures on
        $D$ (at cap 30, 20% transport-corner dwell with **zero** vacuum returns), and it
        is not part of the convergence test --- flatness in $Q$ can be perfect while the
        chain never returns.

    .. note ::
        This tunes ONE marginal, holding the other axis fixed.  A roster of per-axis
        prices is a product $w_{dF}(D)\,w_q(Q)$ by construction (see
        :mod:`~.pricing`), so alternating this tuner with
        :class:`SectorWeightTuner` flattens the two **marginals** --- which is the most
        the factorized form can reach, and strictly less than a joint $w(D,Q)$ that
        boosts the corner where large $D$ and large $Q$ occur *together*.  Measure
        whether the marginals suffice before building the joint.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The action.
    sectorWeights: supervillain.generator.no_intersection.surface_worm.weights.SectorWeights
        The open-surface table to **pin** while $Q$ is flattened.  Required, and not
        defaulted to a fugacity: tuning $Q$ against an untuned $D$ measures a background
        the production sampler does not have.
    intersectionFugacity: float
        The price to flatten *from*, so iteration 0 is the untuned sampler.
    cap: int
        Top $Q$ the table resolves.
    **kw:
        As :class:`SectorWeightTuner` (``iterations``, ``ticks``, ``stride``,
        ``damping``, ``targetFlatness``, ``minimumReachable``, ``pCob``,
        ``targetFraction``, ``seed``, ``gasFactory``, ``seedWeights``, the smoothing
        lengths, ``warmStart``).
    """

    axis = 'Q'

    def __init__(self, S, sectorWeights, intersectionFugacity, cap, **kw):
        kw.pop('openSurfaceFugacity', None)     # the D price is pinned, not seeded
        super().__init__(S, intersectionFugacity, cap, **kw)
        self.sectorWeights = sectorWeights

    def _count(self, state):
        return int(state.Q)

    def _set_price(self, gas, weights):
        return gas.setChargeWeights(weights)

    def _gas_kwargs(self, weights):
        return dict(sectorWeights=self.sectorWeights, chargeWeights=weights)

    def _seed_table(self):
        return Fugacity(self.intersectionFugacity, cap=self.cap)


class JointWeightTuner(SectorWeightTuner):
    r"""Learn a joint $w(D, Q)$ by 2D histogram flattening.

    The factorized roster cannot reach the object that matters.  Measured
    (q-pricing-2026-08-06 findings 6--8): alternating the two marginals **stalls**, and
    not for want of tuning care --- small-step alternation and a starved coboundary move
    both fail the same way.  Both marginal tables anchor $\log w(0) = 0$ and each
    flattens its own axis against that zero, so their product over-favours the corner
    they both call cheapest, $(0,0)$, and neither tuner can see it because each measures
    one axis.  Once the chain is pinned at $D = 0$ the increment $-\log(H/\bar H)$ over a
    single occupied bin is identically zero and the $D$ table comes back byte-identical
    forever.  Flattening two marginals flattens the joint only if the counts are
    independent, and they are not: opening a surface at sheet occupancy $\approx0.44$
    *makes* intersections.

    So this flattens the 2D histogram directly, and the target is the **corner**: a
    torus-wrapping sheet lives at $D \approx 12$, $Q \approx 16$ *together*.

    Seeding is :meth:`~.pricing.JointWeightTable.from_marginals`, so iteration 0 **is**
    the factorized sampler and the joint's introduction is testable rather than a leap.

    .. warning ::
        The cost of the joint is bins: $(\text{cap}_D{+}1)(\text{cap}_Q{+}1)$ against
        $\text{cap}_D + \text{cap}_Q + 2$, filled from the same budget.  Expect to need
        far more ticks per iteration than either marginal tuner, and watch
        ``vacuumFraction`` --- emission is gated on $D = Q = 0$, so a table that opens
        both axes and does not come back is worse than the fugacities it replaced.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
    sectorWeights, chargeWeights: the marginal prices to seed the joint from.
    capD, capQ: int
        The window the table resolves.
    **kw:
        As :class:`SectorWeightTuner`, except that ``smoothTable`` is the 2D smoothing
        length (see :meth:`~.pricing.JointWeightTable.smoothed`).
    """

    axis = '(D,Q)'

    def __init__(self, S, sectorWeights, chargeWeights, capD, capQ,
                 boostFloor=0.0, rangeStart=None, rangeEnd=None,
                 targetDecay=0.0, targetDecayQ=None, targetDecayQEnd=None, **kw):
        kw.pop('openSurfaceFugacity', None)
        # boostFloor: the MINIMUM correction a reachable-but-unvisited cell receives.
        # Without it such a cell gets `update[seen].max()` -- the largest correction any
        # VISITED cell got -- so the frontier advances at O(1) per iteration while the
        # range a 2D table needs is O(100) log units.  Measured: 60 iterations left the
        # vacuum with zero visits under a table already favouring it by e^134, because it
        # was still converging (q-pricing-2026-08-06 finding 14).  A floor is
        # Wang--Landau in spirit: the frontier gets a generous fixed step.  0 keeps the
        # 1D tuner's behaviour exactly.
        self.boostFloor = float(boostFloor)
        # ADIABATIC RANGE RAMP (Evan).  The failure mode this cures: iteration 0 samples
        # under the production seed and is vacuum-rich, the first update expands the
        # frontier, the chain is pulled out and never returns -- so the vacuum is never
        # re-sampled and the table can never learn to bring it back.  Measured at 8x12:
        # 841 -> 55 -> 0 vacuum returns in three iterations.
        #
        # The cure is to stop the table running away from the chain.  After each update
        # clip log w to [-R, 0], with R ramped from `rangeStart` to `rangeEnd` across the
        # run.  Small R is a nearly-uniform table -- the chain behaves like production and
        # samples the vacuum well; growing R lets the far cells be suppressed a little
        # more each iteration, so the chain is always in equilibrium with a table only
        # slightly more aggressive than the one it just equilibrated to.
        #
        # This is the adiabatic version of the `alpha` knob, without alpha's defect of
        # presupposing a converged target table to interpolate towards: here the target
        # IS the table being learned.  The upper clip at 0 encodes the one thing we know
        # a priori -- the vacuum should be the most-weighted cell, since it is where
        # emission happens.  None disables the ramp entirely.
        self.rangeStart = None if rangeStart is None else float(rangeStart)
        self.rangeEnd = None if rangeEnd is None else float(rangeEnd)
        # TARGET DISTRIBUTION (Evan).  Flatness is the wrong objective here, and not by a
        # little: a flat target over (D, Q) has no interior fixed point, because the
        # density of states keeps growing outward, so the tuner keeps pulling the chain
        # away from the vacuum -- measured, at cap 108 the chain still spent 15% of its
        # time beyond D = 108.  Worse, the vacuum is a single extreme cell that flatness
        # deliberately visits only 1/N_cells of the time, and the chain loses it entirely.
        #
        # So target a MILD PREFERENCE for the vacuum instead of flatness:
        #
        #     pi_target(D, Q)  proportional to  exp(-targetDecay * (D + Q)),
        #
        # which is normalizable on the unbounded domain (a fixed point exists), has its
        # maximum at (0,0) (recurrence, hence emission, by construction), and still
        # reaches the corner at a known, tunable suppression -- at targetDecay = 0.15 the
        # wrapping-sheet corner D+Q ~ 28 sits at 1.5% of the vacuum's occupancy, which is
        # perfectly samplable.  targetDecay = 0 recovers exact flatness.
        #
        # The update generalizes accordingly: pi ~ Omega*w = target, and H ~ Omega*w, so
        # log w_new = log w + log target - log H.  With a constant target that is the
        # familiar -log(H/mean).
        self.targetDecay = float(targetDecay)
        # A LOOSENED FUGACITY, not flatness (Evan).  The two axes are not alike and should
        # not get the same target.  Production already achieves a FLAT D-distribution (that
        # is what its w(D) table is for) and returns to the vacuum perfectly well from
        # D = 83 -- 2555 vacuum visits in 2e8 moves at cap 120.  What production does NOT
        # flatten is Q: the fugacity gives log w = Q log(eta_q), a decay of
        # lambda_Q = -log(0.012) = 4.42 per intersection, and that steep restoring force is
        # what keeps the chain near Q = 0 and hence able to return.
        #
        # Flattening Q removes that force entirely, which is what killed every run here.
        # (My first attempt at a decaying target used lambda = 0.15-0.30 -- 15x to 29x
        # LOOSER than the fugacity, i.e. essentially flat.  It failed for exactly this
        # reason.)  So: target flat in D, and in Q a *loosened* fugacity,
        #
        #     pi_target(D, Q)  proportional to  exp(-lambda_Q(t) * Q),
        #
        # with lambda_Q ramped from the production value down to a chosen floor.  At
        # lambda_Q = 1 reaching Q = 16 costs 16 instead of 71 -- an e^55 relaxation -- while
        # <Q>_target stays 0.58, so the vacuum remains the modal state and emission
        # survives.  None keeps the isotropic `targetDecay` behaviour.
        self.targetDecayQ = None if targetDecayQ is None else float(targetDecayQ)
        self.targetDecayQEnd = (self.targetDecayQ if targetDecayQEnd is None
                                else float(targetDecayQEnd))
        super().__init__(S, getattr(chargeWeights, 'eta', 0.3), capD, **kw)
        self.sectorWeights = sectorWeights
        self.chargeWeights = chargeWeights
        self.capD = int(capD)
        self.capQ = int(capQ)

    def _seed_table(self):
        return JointWeightTable.from_marginals(self.sectorWeights, self.chargeWeights,
                                               capD=self.capD, capQ=self.capQ)

    def _set_price(self, gas, weights):
        return gas.setJointWeights(weights)

    def _gas_kwargs(self, weights):
        return dict(sectorWeights=self.sectorWeights, chargeWeights=self.chargeWeights,
                    jointWeights=weights)

    def visit_histogram(self, gas, state, ticks, stride, cap, pCob=0.6):
        r"""2D visit counts over $(D, Q)$, plus how many samples fell outside the window
        and how many sat in the joint vacuum."""
        hist = np.zeros((self.capD + 1, self.capQ + 1), dtype=np.int64)
        beyond = vacuum = 0
        # Which axis overflows is the interesting question, not that one did: a chain
        # that leaves the window in D is doing something quite different from one that
        # leaves in Q, and a single `beyond` cannot tell them apart.  Recorded separately
        # so a window can be widened on the axis that actually binds.
        self.beyondD = self.beyondQ = 0
        self.returns = 0
        wasVacuum = False
        for _ in range(ticks):
            gas.sweep(state, stride, pCob=pCob)
            D, Q = int(state.D), int(state.Q)
            if D <= self.capD and Q <= self.capQ:
                hist[D, Q] += 1
            else:
                beyond += 1
                self.beyondD += (D > self.capD)
                self.beyondQ += (Q > self.capQ)
            isVacuum = (D == 0 and Q == 0)
            # Count RETURNS (entries into the vacuum), not dwell samples: dwell measures
            # how long the chain sits there and returns measure how often it gets back,
            # and it is the second that sets the emission rate.  Same edge-counting the
            # TransportTuner uses for its return floor.  Both are strided, so a visit
            # shorter than `stride` is invisible and a zero bounds the rate rather than
            # proving there were none.
            self.returns += (isVacuum and not wasVacuum)
            wasVacuum = isVacuum
            vacuum += isVacuum
        return hist, beyond, vacuum

    def tune(self, log=print):
        r"""Flatten the 2D histogram; return the learned :class:`~.pricing.JointWeightTable`.

        The same loop as :meth:`SectorWeightTuner.tune` with the histogram, the learned
        reachable set and the smoothing all one dimension larger.  The convergence test
        is deliberately the same: flatness over the *ever-reachable* cells, which is
        ``inf`` while any of them goes unvisited --- and in 2D that is a much stronger
        demand, so expect it to be the budget that binds.
        """
        weights = self.seedWeights if self.seedWeights is not None else self._seed_table()
        self.history = []
        everSeen = np.zeros((self.capD + 1, self.capQ + 1), dtype=bool)
        sinceExpansion = 0

        gas = self.gasFactory(self.S, targetFraction=self.targetFraction,
                              measure=False, seed=self.seed,
                              rng=np.random.default_rng(self.seed),
                              **self._gas_kwargs(weights))
        state = (FState.from_configuration(self.S, self.warmStart)
                 if self.warmStart is not None else FState(self.S))

        for it in range(self.iterations):
            self._set_price(gas, weights)
            t0 = time.time()
            gas.sweep(state, 5 * self.stride, pCob=self.pCob)
            hist, beyond, vacuum = self.visit_histogram(gas, state, self.ticks,
                                                        self.stride, self.capD,
                                                        pCob=self.pCob)
            grew = bool((~everSeen & (hist > 0)).any())
            everSeen |= hist > 0
            sinceExpansion = 0 if grew else sinceExpansion + 1
            # Success is now "H/target is constant", not "H is constant".
            if self.targetDecayQ is not None:
                frac = it / max(1, self.iterations - 1)
                lamQ = self.targetDecayQ + frac * (self.targetDecayQEnd - self.targetDecayQ)
                logTarget = -lamQ * np.broadcast_to(np.arange(self.capQ + 1),
                                                    (self.capD + 1, self.capQ + 1)).copy()
            else:
                logTarget = -self.targetDecay * (np.add.outer(np.arange(self.capD + 1),
                                                              np.arange(self.capQ + 1)))
            ratio = hist / np.exp(logTarget - logTarget.max())
            reach = ratio[everSeen]
            flat = float(reach.max() / reach.min()) if (reach > 0).all() else np.inf
            self.history.append(dict(
                iteration=it, flatness=flat, histogram=hist.copy(),
                logWeight=weights.logWeight.copy(), beyond=beyond,
                visited=int((hist > 0).sum()), reachable=int(everSeen.sum()),
                vacuumFraction=vacuum / max(1, self.ticks),
                returns=int(self.returns), beyondD=int(self.beyondD),
                beyondQ=int(self.beyondQ), seconds=time.time() - t0))
            log(f'  iter {it:>3d}  occupied {int((hist > 0).sum()):>4d}/'
                f'{int(everSeen.sum()):>4d} reachable  beyond {beyond:>5d}  '
                f'flatness {flat:>9.3g}  vacuum {vacuum / max(1, self.ticks):>6.3f}  '
                f'returns {self.returns:>5d}  beyond D/Q {self.beyondD}/{self.beyondQ}  '
                f'({time.time() - t0:.0f}s)')
            if flat < self.targetFlatness and int(everSeen.sum()) >= self.minimumReachable \
                    and sinceExpansion >= 2:
                log(f'  converged: flatness {flat:.3g} over {int(everSeen.sum())} cells')
                break

            # log w -= log H on the visited cells; reachable-but-unvisited cells get the
            # largest correction any visited cell got.  NEVER cells never seen at all --
            # boosting those forever is what makes a multicanonical table oscillate.
            update = np.zeros_like(weights.logWeight)
            seen = hist > 0
            lt = logTarget[seen]
            update[seen] = ((lt - lt.mean())
                            - (np.log(hist[seen]) - np.log(hist[seen]).mean()))
            missed = everSeen & ~seen
            if missed.any():
                base = float(update[seen].max()) if seen.any() else 1.0
                update[missed] = max(base, self.boostFloor)
            step = self.damping / (1.0 + sinceExpansion / 3.0)
            newLog = weights.logWeight + step * update
            if self.rangeStart is not None:
                frac = it / max(1, self.iterations - 1)
                R = self.rangeStart + frac * ((self.rangeEnd if self.rangeEnd is not None
                                               else self.rangeStart) - self.rangeStart)
                newLog = np.clip(newLog - newLog[0, 0], -abs(R), 0.0)
            weights = JointWeightTable(newLog, weights.tailSlopeD, weights.tailSlopeQ,
                                       weights.hardWall)
            # Fill the unreached cells EVERY iteration, not only at the end as the 1D
            # tuner does.  In 1D the unvisited set is a few bins behind the frontier; in
            # 2D it is most of the grid, and the corner that matters sits behind it, so a
            # table that is only repaired at the end never offers the chain the moves
            # that would have reached it.  Measured without this: D spanning 0-24 while Q
            # never passed 6, and zero corner visits.
            weights = weights.interpolated(everSeen)
            if self.smoothTable:
                weights = weights.smoothed(length=self.smoothTable, reachable=everSeen)

        weights = weights.interpolated(everSeen)
        if self.smoothFinal:
            weights = weights.smoothed(length=self.smoothFinal, reachable=everSeen)
        unseen = int((~everSeen).sum())
        if unseen:
            log(f'  (D,Q) cells never visited: {unseen} of {everSeen.size}')
        return weights


class _OffsetBisector:
    r"""Choose a :class:`~.weights.PairUmbrella`'s additive constant by
    **bisection**, not by prediction.

    The constant controls how attractive the pair sector is, and the
    occupancy's response to it is monotonic but violently nonlinear ---
    measured local gains from 1.3 to 8 across the transition, with a
    vacuum-dominated and a pair-dominated phase either side.  Every
    *predictive* scheme tried in the source notebook failed on the same
    point: reweighting estimates the new sector dwell as
    $\sum_r \Omega(r)e^{\text{step}(r)}$, a **perturbative** answer to a
    question about a **stationary distribution**, so it under-predicts the
    enhancement and the chain lands in the collapsed phase.

    Bisection needs no model of the response --- only monotonicity, which
    is measured. Each call to :meth:`update` brackets the target: an
    occupancy above target (or a collapse) lowers the ceiling, one below
    raises the floor. Before a bracket exists it steps geometrically to
    find one.

    .. note ::
        The constant is tracked **separately** from the shape (see
        :meth:`PairUmbrellaTuner.tune`), so a correction to the
        normalization never disturbs the flattening. That separation is
        what makes a collapse cost one iteration rather than the
        accumulated tuning.
    """

    def __init__(self, offset=0.0, step=1.0):
        self.offset = float(offset)
        self.step = float(step)
        self.lo = None                 # highest offset seen BELOW target occupancy
        self.hi = None                 # lowest offset seen ABOVE target (or collapsed)

    def update(self, tooHigh):
        r"""Record this iteration's verdict and return the next offset.

        Parameters
        ----------
        tooHigh: bool
            True when the pair sector took more than its share ---
            including a collapse, which is simply a very emphatic "too
            high".

        Returns
        -------
        float
            The next offset to try.
        """
        if tooHigh:
            self.hi = self.offset if self.hi is None else min(self.hi, self.offset)
        else:
            self.lo = self.offset if self.lo is None else max(self.lo, self.offset)
        if self.lo is not None and self.hi is not None:
            self.offset = 0.5 * (self.lo + self.hi)
        elif tooHigh:
            self.offset -= self.step   # no floor yet: step down to find one
        else:
            self.offset += self.step   # no ceiling yet: step up
        return self.offset

    def __str__(self):
        b = (f'[{self.lo:.2f}, {self.hi:.2f}]'
             if self.lo is not None and self.hi is not None else 'unbracketed')
        return f'offset {self.offset:+.2f} {b}'


class PairUmbrellaTuner:
    r"""Iterative multicanonical tuning of the pair-separation umbrella
    $w_2(r^2)$.

    The goal is **every shell reached reliably**: the walker should visit
    each achievable squared separation often enough that $\Theta$ has real
    statistics there, out to the box maximum $r^2 = N^2$, instead of dying
    at the frontier the plain sampler happens to reach.

    .. note ::
        **One update handles both the shape and the constant, on purpose.**
        Chasing the pair-sector occupancy with a separate control loop on
        an additive constant is ill-conditioned: the occupancy is
        *bistable* in the constant, so it oscillates however it is damped.
        Predicting the step's effect and compensating in the same update
        also fails, for the same reason :class:`_OffsetBisector`'s
        docstring gives: reweighting is a perturbative answer to a
        stationary-distribution question.  What works is to **stop
        modelling the response at all**: track the shape and the constant
        separately, flatten the shape from the histogram
        (:meth:`update`), and set the constant by :class:`_OffsetBisector`,
        which needs only monotonicity.

    .. warning ::
        An iteration whose vacuum dwell has collapsed (fewer than
        ``excursionFloor`` returns to $Q=0$) is **not a measurement** ---
        there is no denominator to normalize $\Theta$ --- so its histogram
        is discarded rather than learned from. It still informs the
        bisection bracket, since a collapse is an emphatic "the sector is
        too attractive", but it must never reach the shape update.

    .. warning ::
        **Known limitation** (observed 2026-07-31): the bisection tuner
        oscillates at $N=4$ between a high-occupancy table and a
        dead-sector one --- the bracket narrows but the *shape* update
        keeps nudging the achieved occupancy back and forth across
        ``targetOdds`` faster than the bisection converges at this small a
        volume. Inspect :attr:`history` and prefer the best-scoring
        iteration (largest shell coverage, then largest minimum count)
        rather than trusting the last one; :meth:`tune` already does this
        internally and returns that best iteration's table, but a caller
        chasing the oscillation further should look at the history
        directly rather than assume monotonic improvement.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The action; sets $N$ and $\kappa$.
    sectorWeights: supervillain.generator.no_intersection.surface_worm.weights.SectorWeights
        The (already-tuned) open-surface table, held fixed throughout ---
        this tuner only retunes the pair umbrella.
    intersectionFugacity: float
        Self-intersection fugacity $\eta_q$.
    targetFraction: float
        Defect-adjacent proposal targeting, forwarded to every internal
        gas; must match the value production sampling will use, for the
        same reason as :class:`SectorWeightTuner`'s ``targetFraction``.
    pCob: float
        Coboundary-move probability.
    iterations, ticks, stride: int
        Tuning iterations, measured ticks per iteration, and compiled moves
        between ticks.
    equilibrate: int
        Compiled moves to burn in before each iteration's measurement ---
        each iteration constructs a **fresh** gas and chain (see
        :meth:`measure`'s docstring for why), so this pays for mixing every
        time rather than once.
    targetOdds: float
        Pair-sector odds to hold while the shape is tuned; 1.0 is an even
        split with everything outside, which keeps the vacuum dwell alive.
    seed: int, optional
        Base seed; iteration ``it`` measures with ``seed + it`` (both
        ``seed=`` and ``rng=numpy.random.default_rng(...)``, the same
        reproducibility guard as :class:`SectorWeightTuner`), so successive
        iterations are decorrelated but the whole run is reproducible from
        one number.

    Attributes
    ----------
    history: list of dict
        One entry per iteration: ``iteration``, ``shellsHit``,
        ``minCount``, ``occupancy`` (pair-sector fraction of all ticks),
        ``excursions`` (vacuum returns), ``collapsed``, ``maxSep2``,
        ``offset`` (the bisector's offset used for this iteration's
        measurement), ``score`` (``None`` when collapsed), ``table`` (the
        :class:`~.weights.PairUmbrella` measured this iteration), and
        ``converged`` (whether this iteration hit every achievable shell
        with at least ``minCount`` visits).
    """

    #: Pseudo-count constants for the shape update and the collapse test.
    #: Not exposed on the constructor (the brief's Interfaces block does not
    #: list them as tuner knobs); kept as class attributes rather than
    #: buried literals so a subclass or a script can still override them.
    damping = 0.5
    maxStep = 1.2
    unvisited = 0.5
    excursionFloor = 25
    minCount = 25

    def __init__(self, S, sectorWeights, intersectionFugacity, targetFraction, pCob,
                 iterations=20, ticks=6000, stride=100, equilibrate=2_000_000,
                 targetOdds=0.5, seed=None, gasFactory=None, warmStart=None):
        # see SectorWeightTuner: tune the sampler that will actually run
        self.gasFactory = gasFactory if gasFactory is not None else SurfaceWormGas
        # see SectorWeightTuner.warmStart
        self.warmStart = warmStart
        self.S = S
        self.sectorWeights = sectorWeights
        self.intersectionFugacity = float(intersectionFugacity)
        self.targetFraction = float(targetFraction)
        self.pCob = float(pCob)
        self.iterations = int(iterations)
        self.ticks = int(ticks)
        self.stride = int(stride)
        self.equilibrate = int(equilibrate)
        self.targetOdds = float(targetOdds)
        self.seed = seed
        self.history = []

    @staticmethod
    def achievable_shells(N):
        r"""The squared minimal-image separations a $\pm1$ **pair** can
        actually occupy.

        Two filters, and both are needed.  Not every integer in
        $[0, N^2]$ is a sum of four squares of integers in $[0, N/2]$, so
        counting coverage against ``range(N**2+1)`` would report a frontier
        that can never be reached and the tuner would chase it forever.

        .. warning ::
            $r^2 = 0$ is excluded. A coincident pair is **not** a two-charge
            state --- the two charges would sit on one cell and cancel,
            which is the vacuum --- so
            :func:`~.weights.pair_separation_squared` can never return 0
            and $\Theta_0 \equiv 1$ is written outright by the observable.
            Including it makes ``hit == len(shells)`` unsatisfiable no
            matter how well the tuning goes, and drives $\log w_2(0)$
            without bound since a permanently-unvisited shell is boosted
            every iteration.

        Parameters
        ----------
        N: int
            Linear lattice size.

        Returns
        -------
        numpy.ndarray
            Sorted achievable $r^2$ values, excluding 0.
        """
        m = np.arange(N // 2 + 1) ** 2
        s = {0}
        for _ in range(4):
            s = {a + b for a in s for b in m}
        return np.array(sorted(s - {0}))

    @staticmethod
    def coverage(occupancy, shells):
        r"""How many achievable shells were hit, the smallest hit count, and
        the total pair-sector ticks.

        Parameters
        ----------
        occupancy: numpy.ndarray
            ``PairSeparationTicks`` from a harvest.
        shells: numpy.ndarray
            The achievable $r^2$ values (:meth:`achievable_shells`).

        Returns
        -------
        (int, float, int)
            ``(hit, lo, tot)`` --- shells with at least one visit, the
            smallest nonzero visit count among them (0.0 if none were hit),
            and the total visits summed over ``shells``.
        """
        occ = np.asarray(occupancy, dtype=float)[shells]
        hit = occ > 0
        if not hit.any():
            return 0, 0.0, 0
        return int(hit.sum()), float(occ[hit].min()), int(occ[hit].sum())

    def measure(self, umbrella, seed, ticks, stride, equilibrate):
        r"""Burn in and measure one fresh gas under ``umbrella``.

        A **new** :class:`~.gas.SurfaceWormGas` and cold :class:`~.state.FState`
        every call, ported faithfully from the source script's ``measure``
        --- unlike :class:`SectorWeightTuner`, which warm-starts one chain
        across iterations. The pair umbrella changes the *sector* the chain
        spends its time in far more violently than a sector-weight update
        changes $D$ (see the class docstring's bistability note), so a
        stale chain from the previous, differently-tuned umbrella is not a
        representative starting point for the next measurement; a fresh
        burn-in is the safer (if costlier) choice the source made.

        Parameters
        ----------
        umbrella: supervillain.generator.no_intersection.surface_worm.weights.PairUmbrella
            The table to measure under.
        seed: int or None
            Seeds both the compiled kernel and the python RNG stream of the
            fresh gas.
        ticks, stride: int
            Measured ticks and compiled moves per tick.
        equilibrate: int
            Compiled moves to burn in before measuring.

        Returns
        -------
        dict
            The harvested :class:`~.accumulator.CorrelatorAccumulator`
            observables.
        """
        gas = self.gasFactory(self.S, sectorWeights=self.sectorWeights,
                              intersectionFugacity=self.intersectionFugacity,
                              sectorWeightCap=self.sectorWeights.cap,
                              targetFraction=self.targetFraction,
                              pairUmbrella=umbrella, measure=True,
                             seed=seed, rng=np.random.default_rng(seed))
        state = (FState.from_configuration(self.S, self.warmStart)
                 if self.warmStart is not None else FState(self.S))
        gas.sweep(state, equilibrate, pCob=self.pCob)
        gas.accumulator.reset()
        gas.sweep_measured(state, ticks, stride=stride, pCob=self.pCob)
        return gas.accumulator.harvest()

    def update(self, umbrella, occupancy, totalTicks, shells):
        r"""One multicanonical step toward "every achievable shell equally
        populated", holding the sector's overall odds at :attr:`targetOdds`.

        Parameters
        ----------
        umbrella: supervillain.generator.no_intersection.surface_worm.weights.PairUmbrella
            The current table (its shape only --- the caller is expected to
            pass one with the additive constant zeroed, since the constant
            is :class:`_OffsetBisector`'s job, not this method's).
        occupancy: numpy.ndarray
            ``PairSeparationTicks`` --- visits per $r^2$ over **all**
            ticks, not just the closed shell, because the separation only
            moves while $dF \neq 0$.
        totalTicks: int
            ``Ticks`` from the same harvest.
        shells: numpy.ndarray
            The achievable $r^2$ values; bins outside this set are never
            targeted.

        Returns
        -------
        PairUmbrella
            The updated shape (constant not re-set; see the class
            docstring's note on tracking shape and constant separately).
        """
        occ = np.asarray(occupancy, dtype=float)
        inSector = occ.sum()
        out = max(float(totalTicks) - inSector, 1.0)
        n = len(shells)
        target = out / n                       # even split with everything outside
        counts = np.where(occ[shells] > 0, occ[shells], self.unvisited)
        step = np.clip(-self.damping * np.log(counts / target), -self.maxStep, self.maxStep)
        # Compensate the occupancy the step is about to cause, rather than letting
        # it float -- see the class docstring for why a perturbative prediction
        # alone (without the bisector) was not enough.
        predicted = (float((occ[shells] * np.exp(step)).sum())
                     + self.unvisited * np.exp(self.maxStep))
        shift = -np.log(max(predicted, 1e-300) / (self.targetOdds * out))
        logW = umbrella.logWeight.copy()
        logW[shells] += step + shift
        logW[shells] -= logW[shells].min()     # keep the table from drifting to -inf
        return PairUmbrella(logW, umbrella.N)

    def tune(self, log=print):
        r"""Run the bisection + shape-flattening loop; return the best
        table seen.

        Parameters
        ----------
        log: callable, optional
            Called with one string per iteration; defaults to
            :func:`print`.

        Returns
        -------
        PairUmbrella
            The highest-scoring non-collapsed iteration's table (shells hit
            first, then minimum visit count, both capped at
            :attr:`minCount` so an already-adequate shell cannot outbid
            genuine coverage gains elsewhere) --- see the class docstring's
            known-limitation warning for why "best", not "last".  If no
            iteration ever avoided collapse, the last table tried is
            returned instead (with a logged warning), matching the source
            script's fallback.
        """
        N = self.S.Lattice.N
        shells = self.achievable_shells(N)
        targetOcc = self.targetOdds / (1.0 + self.targetOdds)

        def table(shape, offset):
            lw = np.zeros_like(shape)
            lw[shells] = shape[shells] + offset
            return PairUmbrella(lw, N)

        # Shape and normalization are tracked SEPARATELY -- `shape` is mean-zero
        # over the achievable shells and carries the flattening; `bisector` owns
        # the additive constant that decides how attractive the sector is. See
        # the class docstring for why this separation is what keeps a collapse
        # from costing more than one iteration.
        off = PairUmbrella.off(N)
        shape = off.logWeight.copy()
        shape[shells] -= shape[shells].mean()
        bisector = _OffsetBisector(offset=float(off.logWeight[shells].mean()), step=1.0)
        umbrella = table(shape, bisector.offset)

        self.history = []
        best, bestScore = None, -1
        for it in range(self.iterations):
            seed = None if self.seed is None else self.seed + it
            h = self.measure(umbrella, seed, self.ticks, self.stride, self.equilibrate)
            hit, lo, tot = self.coverage(h['PairSeparationTicks'], shells)
            totalTicks = int(h['Ticks'])
            occ = tot / max(totalTicks, 1)
            excursions = int(h['VacuumReturns'])
            collapsed = excursions < self.excursionFloor

            record = dict(iteration=it, shellsHit=hit, minCount=lo, occupancy=occ,
                          excursions=excursions, collapsed=collapsed,
                          maxSep2=int(h['MaxPairSeparationSquared']),
                          offset=bisector.offset, table=umbrella, score=None,
                          converged=False)

            if collapsed:
                log(f'  it {it:2d}: COLLAPSED (sector {100*occ:.1f}%, {excursions} '
                    f'excursions < {self.excursionFloor}) -- histogram discarded')
            else:
                score = hit * 10_000 + min(lo, self.minCount)
                record['score'] = score
                if score > bestScore:
                    best, bestScore = umbrella, score
                shape = self.update(table(shape, 0.0), h['PairSeparationTicks'],
                                    h['Ticks'], shells).logWeight
                shape[shells] -= shape[shells].mean()
                log(f'  it {it:2d}: shells {hit:3d}/{len(shells)}  min count {lo:6.0f}  '
                    f'sector {100*occ:5.1f}%  vacuum {int(h["VacuumTicks"]):5d} '
                    f'({excursions} excursions)  maxsep2 {record["maxSep2"]:3d}')
                record['converged'] = hit == len(shells) and lo >= self.minCount
                if record['converged']:
                    log(f'  converged at iteration {it}: every achievable shell has '
                        f'>= {self.minCount} visits')
                    self.history.append(record)
                    # Rebuild from the JUST-UPDATED shape (not `umbrella`, which was
                    # measured under the PRE-update shape): the source's convergence
                    # branch deliberately hands back the table incorporating this
                    # iteration's flattening refinement, not the one that produced
                    # the convergent measurement. The non-converged path above keeps
                    # `umbrella` on purpose (that IS what was measured, and its score
                    # is what the comparison is against); only the converged exit
                    # gets the extra refinement because there is no next iteration
                    # left to measure it under.
                    best, bestScore = table(shape, bisector.offset), score
                    break

            self.history.append(record)
            bisector.update(collapsed or occ > targetOcc)
            umbrella = table(shape, bisector.offset)
            log(f'          {bisector}')

        if best is None:
            log('  WARNING: no healthy iteration ever completed; returning the last table')
            best = umbrella
        return best


class TransportTuner:
    r"""Grow the $w(D)$ cap against the transport physics, not against
    $\Theta$-measurement health.

    Finding 7 (the source notebook's ``NOTES.md``): every $J$-changing
    excursion --- a self-intersection winding change --- carries charge
    through a wide-open surface, and every one of them hit the hard $w(D)$
    cap.  :class:`SectorWeightTuner` chooses that cap for $\Theta$
    measurement health (occupied-range flatness), which throttles exactly
    the excursions that transport $J$.  This tuner chooses the cap instead
    by staging: grow the cap, flatten $w(D)$ at each stage with an internal
    :class:`SectorWeightTuner`, measure what the stage bought, and stop by
    the transport physics rather than by a pressure heuristic.

    Recipe (v4), earned by three documented failures in the source notebook:

    * **untargeted tuning** (v2's lesson): the internal
      :class:`SectorWeightTuner` at each stage flattens with
      ``targetFraction=tuneTargetFraction`` (0.0 by default) even though the
      stage *measurement* and eventual production sampling use
      ``targetFraction``.  v1 and v2 tuned at the production
      ``targetFraction=0.8`` and both chose designs scoring 0.00 / 0.53
      flips/Mmove against 3.52--5.40 for tables tuned untargeted --- pushing
      the proposal toward already-open cells during *tuning* degrades the
      histogram the table learns from.
    * **corner-dwell over pressure** (v1's lesson): v1 grew the cap while
      top-of-range *pressure* (dwell in the top ``pressureFrac`` of the $D$
      range) persisted and stopped when it vanished --- and was fooled: a
      badly-flattened stage table blocks the approach to the cap, which
      reads as zero pressure while transporting *nothing* (v1's first run
      chose such a stage and scored 0 flips/40M).  Low pressure is ambiguous
      between "expansion exhausted" and "tune failed"; the **transport
      corner dwell** ($D \geq$ ``cornerFrac`` $\cdot$ cap with $Q > 0$, the
      finding-7 proxy --- an 18% flip-conversion rate per excursion at
      $N=4$) is not, so it is what :meth:`tune` actually decides on.
      ``pressureFrac``/``pressureEps`` are still accepted and still recorded
      in :attr:`history` (a stage's ``pressure``/``topFloor`` fields), but
      --- as in the source --- play no role in the stop decision.
    * **multi-seed mean-corner / min-returns** (finding 12): the chain is
      bistable, so a single equilibration lands in a basin and a one-seed
      measurement scores the basin draw, not the table.  Each stage
      equilibrates ``stageSeeds`` independent chains; transport is judged by
      the **mean** corner dwell across seeds and the constraint by the
      **minimum** returns (the floor must hold whichever basin production
      happens to fall into).
    * **retries only when transport looks dead**: a stage whose corner dwell
      falls below half the running best is re-tuned (up to ``retries``
      extra attempts, different seeds) before being accepted --- a healthy
      stage is never retried, since retrying a merely-lower-but-live stage
      would waste budget chasing noise rather than a bad flatten.

    :meth:`tune` keeps the best stage **by corner dwell** among those
    meeting the return floor, and stops on one of three verdicts:
    constraint-binds (a stage's minimum returns fell below ``returnFloor``),
    turnover (two consecutive caps scored below half the running best), or
    ``capMax`` exhaustion.  Unlike the source script, it raises
    :class:`RuntimeError` rather than calling ``SystemExit`` when no cap
    ever satisfies the floor.

    .. warning ::
        **Known limitation** (v4, observed 2026-07-31): large-cap stages
        need *more* tune budget than small ones to flatten reliably, and
        this tuner does not scale ``tuneIterations``/``tuneTicks`` with
        ``cap`` --- v4 chose cap 12 at fixed budget while a brute-force
        cap-24 flattening (generous, hand-tuned budget) reached 5.40
        flips/Mmove against v4's 4.20.  Budget scaling with cap is the next
        lever on this design, not a change to the decision logic above.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The action; sets $N$ and $\kappa$.
    intersectionFugacity: float
        Self-intersection fugacity $\eta_q$, forwarded to every internal
        gas and tuner unchanged.
    openSurfaceFugacity: float
        Starting fugacity for each stage's internal
        :class:`SectorWeightTuner` (its own ``openSurfaceFugacity``).
    targetFraction: float
        Defect-adjacent proposal targeting used for stage **measurement**
        and (implicitly) production sampling --- see ``tuneTargetFraction``
        for why tuning itself uses a different value.
    pCob: float
        Coboundary-move probability, forwarded everywhere.
    pairUmbrella: supervillain.generator.no_intersection.surface_worm.weights.PairUmbrella, optional
        Held fixed throughout; defaults to the identity
        (:meth:`~.weights.PairUmbrella.off`), matching
        :class:`~.gas.SurfaceWormGas`'s own default.
    cap0, capStep, capMax: int
        The cap ladder: start at ``cap0``, grow by ``capStep`` each accepted
        stage, never exceed ``capMax``.
    pressureFrac: float
        Top fraction of the $D$ range whose dwell defines a stage's
        ``pressure`` diagnostic (recorded, not decided on --- see the class
        docstring).
    pressureEps: float
        Retained for the design-npz schema and as a documented diagnostic
        threshold; **not read by** :meth:`tune`'s decision loop (the
        corner-dwell logic superseded it --- see the class docstring's
        "corner-dwell over pressure" note).
    cornerFrac: float
        The transport corner is $D \geq$ ``cornerFrac`` $\cdot$ cap with
        $Q > 0$ --- the finding-7 proxy :meth:`tune` actually decides on.
    returnFloor: int
        Minimum vacuum returns a stage's **worst** seed must show (the
        $\Theta$-health constraint); below this the cap growth stops with a
        constraint-binds verdict.
    stageSeeds: int
        Independent equilibrations per stage measurement (finding 12).
    retries: int
        Extra tune attempts per cap when a stage's corner dwell looks dead
        (below half the running best).  Must be $\geq 0$ --- ``tune()``'s
        attempt loop is ``range(retries + 1)``, so a negative value would
        leave that range empty and the loop's ``attempt_best`` unset;
        raises :class:`ValueError` here instead of failing confusingly deep
        inside :meth:`tune`.
    tuneTargetFraction: float
        ``targetFraction`` used **only** while flattening $w(D)$ at each
        stage; see the class docstring's "untargeted tuning" note for why
        this is 0.0 by default even though ``targetFraction`` (production)
        is not.
    tuneIterations, tuneTicks, stride, damping: int, int, int, float
        Forwarded to each stage's internal :class:`SectorWeightTuner`.
    measureTicks: int
        Compiled-move samples per stage measurement chain.
    equilibrate: int
        Compiled moves to burn in before each stage measurement chain, and
        before :meth:`score_flips`'s validation chain.
    seed: int, optional
        Base seed.  When ``None``, one integer is drawn here so every
        derived seed below is reproducible from :attr:`seed` alone even
        though the caller never supplied one.  Stage seeds are derived
        exactly as the source script derived them: the internal
        :class:`SectorWeightTuner` at cap ``cap``, attempt ``attempt`` uses
        ``seed + cap + 7919*attempt``; the ``s``-th measurement chain at
        that stage uses ``seed + cap + 271*s + 7919*attempt`` (both ``seed=``
        and ``rng=numpy.random.default_rng(...)``, the same reproducibility
        guard :class:`SectorWeightTuner` and :class:`PairUmbrellaTuner` use).

    Attributes
    ----------
    history: list of dict
        One entry per cap actually staged (in increasing cap order):
        ``cap``, ``pressure`` (mean top-of-range dwell fraction), ``returns``
        (minimum vacuum returns across seeds), ``cornerFrac`` (mean corner
        dwell fraction), ``cornerSpread`` (its standard deviation across
        seeds), ``vacuumFrac`` (mean $D=0$ dwell fraction), ``topFloor``
        (the $D$ threshold ``pressure`` was measured above).
    verdict: str
        The stopping reason: a constraint-binds message, a turnover
        message, or ``"capMax {capMax} reached"``.
    best: dict
        The chosen stage's diagnostic dict, an element of :attr:`history`.
    """

    def __init__(self, S, intersectionFugacity, openSurfaceFugacity=0.09,
                 targetFraction=0.8, pCob=0.2, pairUmbrella=None,
                 cap0=12, capStep=6, capMax=64, pressureFrac=0.15,
                 pressureEps=0.02, cornerFrac=0.75, returnFloor=50,
                 stageSeeds=3, retries=2, tuneTargetFraction=0.0,
                 tuneIterations=20, tuneTicks=3000, measureTicks=4000,
                 stride=200, equilibrate=2_000_000, damping=0.8, seed=None,
                 gasFactory=None, warmStart=None):
        # see SectorWeightTuner: tune the sampler that will actually run.
        # Forwarded to the per-stage internal SectorWeightTuner too, so the
        # whole pipeline (flattening, stage measurement, score_flips) tunes
        # and scores ONE sampler.
        self.gasFactory = gasFactory if gasFactory is not None else SurfaceWormGas
        # see SectorWeightTuner.warmStart; forwarded to the per-stage internal
        # SectorWeightTuner so every stage measures the same background
        self.warmStart = warmStart
        # the previous cap stage's learned table, seeding the next stage's
        # flattening (see _stage) -- None until the first stage completes
        self._stageWeights = None
        self.S = S
        self.intersectionFugacity = float(intersectionFugacity)
        self.openSurfaceFugacity = float(openSurfaceFugacity)
        self.targetFraction = float(targetFraction)
        self.pCob = float(pCob)
        self.pairUmbrella = (pairUmbrella if pairUmbrella is not None
                             else PairUmbrella.off(S.Lattice.N))
        self.cap0 = int(cap0)
        self.capStep = int(capStep)
        self.capMax = int(capMax)
        self.pressureFrac = float(pressureFrac)
        self.pressureEps = float(pressureEps)
        self.cornerFrac = float(cornerFrac)
        self.returnFloor = int(returnFloor)
        self.stageSeeds = int(stageSeeds)
        self.retries = int(retries)
        if self.retries < 0:
            # range(retries + 1) is what tune()'s attempt loop actually iterates;
            # a negative value makes that range empty, so attempt_best/attempt_bestW
            # never get set and tune() dies on a confusing "NoneType is not
            # subscriptable" deep inside the loop instead of here, at construction.
            raise ValueError(f'retries must be >= 0, got {self.retries}.')
        self.tuneTargetFraction = float(tuneTargetFraction)
        self.tuneIterations = int(tuneIterations)
        self.tuneTicks = int(tuneTicks)
        self.measureTicks = int(measureTicks)
        self.stride = int(stride)
        self.equilibrate = int(equilibrate)
        self.damping = float(damping)
        # A bare None here would make every "seed + cap + ..." derivation below
        # raise; drawing one integer up front keeps every derived seed
        # reproducible from `self.seed` alone, exactly as if the caller had
        # passed it explicitly.
        self.seed = (int(np.random.default_rng().integers(2 ** 31)) if seed is None
                     else int(seed))
        self.history = []
        self.verdict = None
        self.best = None
        self._bestWeights = None

    def _stage(self, cap, attempt):
        r"""One flatten-and-measure stage at a fixed ``cap``.

        Flattens $w(D)$ over $[0, \texttt{cap}]$ with a fresh internal
        :class:`SectorWeightTuner` (untargeted --- see the class docstring),
        then measures the frozen table with :attr:`stageSeeds` independent
        chains, aggregating the transport corner dwell by its **mean** and
        the vacuum returns by their **minimum** across seeds (finding 12).

        Parameters
        ----------
        cap: int
            The $D$ cap to flatten and measure at.
        attempt: int
            Which retry this is (0 for the first attempt); folded into every
            seed derived here so a retry is decorrelated from the attempt it
            follows rather than repeating it.

        Returns
        -------
        (SectorWeights, dict)
            The stage's flattened table, and its diagnostic dict (the same
            shape appended to :attr:`history`).
        """
        tuneSeed = self.seed + cap + 7919 * attempt
        # Seed each stage's flattening from the PREVIOUS stage's learned table,
        # extended along its own tail slope.  Flattening a wide range from the
        # bare fugacity table is the multicanonical range problem -- at large
        # cap the chain never reaches the top bins, so the table there stays
        # unlearned and the stage measures a barrier of its own making.  The
        # previous stage's table already carries the compensation up to its
        # own cap, so each stage only has to learn the increment.
        seed_table = None
        if self._stageWeights is not None and self._stageWeights.cap < cap:
            prev = self._stageWeights
            extra = prev.logWeight[-1] + prev.tailSlope * np.arange(1, cap - prev.cap + 1)
            seed_table = SectorWeights(np.concatenate([prev.logWeight, extra]),
                                       prev.tailSlope, hardWall=True)
        weights = SectorWeightTuner(
            self.S, self.intersectionFugacity, cap,
            openSurfaceFugacity=self.openSurfaceFugacity,
            iterations=self.tuneIterations, ticks=self.tuneTicks,
            stride=self.stride, damping=self.damping, pCob=self.pCob,
            targetFraction=self.tuneTargetFraction, seed=tuneSeed,
            gasFactory=self.gasFactory, seedWeights=seed_table,
            warmStart=self.warmStart,
        ).tune(log=lambda *a, **k: None)
        self._stageWeights = weights

        cornerFloor = max(1, int(np.ceil(self.cornerFrac * cap)))
        topFloor = max(1, int(np.floor((1.0 - self.pressureFrac) * cap)))
        corners, returnss, pressures, vacs = [], [], [], []
        for s in range(self.stageSeeds):
            gasSeed = self.seed + cap + 271 * s + 7919 * attempt
            gas = self.gasFactory(
                self.S, sectorWeights=weights,
                intersectionFugacity=self.intersectionFugacity,
                sectorWeightCap=cap, targetFraction=self.targetFraction,
                pairUmbrella=self.pairUmbrella, measure=False,
                seed=gasSeed, rng=np.random.default_rng(gasSeed))
            state = (FState.from_configuration(self.S, self.warmStart)
                     if self.warmStart is not None else FState(self.S))
            gas.sweep(state, self.equilibrate, pCob=self.pCob)
            hist = np.zeros(cap + 1, dtype=np.int64)
            corner = returns = 0
            prevVac = False
            for _ in range(self.measureTicks):
                gas.sweep(state, self.stride, pCob=self.pCob)
                D, Q = state.D, state.Q
                if D <= cap:
                    hist[D] += 1
                if D >= cornerFloor and Q > 0:
                    corner += 1
                vac = (D == 0 and Q == 0)
                if vac and not prevVac:
                    returns += 1
                prevVac = vac
            open_dwell = hist[1:].sum()
            corners.append(corner / self.measureTicks)
            returnss.append(returns)
            pressures.append(hist[topFloor:].sum() / open_dwell if open_dwell else 0.0)
            vacs.append(hist[0] / self.measureTicks)

        diag = dict(cap=cap, pressure=float(np.mean(pressures)),
                    returns=int(min(returnss)),
                    cornerFrac=float(np.mean(corners)),
                    cornerSpread=float(np.std(corners)),
                    vacuumFrac=float(np.mean(vacs)), topFloor=topFloor)
        return weights, diag

    def tune(self, log=print):
        r"""Grow the cap, staging by staging; return the chosen table.

        The v4 decision loop, content-ported from the source script's
        ``__main__``: per cap, tune up to ``retries + 1`` times (only
        re-tuning when the previous attempt's corner dwell looked dead ---
        below half the running best) and keep the attempt with the largest
        corner dwell; then apply the return floor, keep-best-by-corner, and
        two-strike turnover logic described in the class docstring.

        Parameters
        ----------
        log: callable, optional
            Called with one string per accepted stage and at the verdict;
            defaults to :func:`print`.

        Returns
        -------
        SectorWeights
            The chosen stage's table; its ``.cap`` carries the chosen cap
            (the same convention as everywhere else in this module --- no
            separate cap return).

        Raises
        ------
        RuntimeError
            If no cap ever satisfies ``returnFloor`` (the source script's
            ``SystemExit``, replaced so a caller can catch it).
        """
        self.history = []
        best, bestW = None, None
        cap, weak_streak = self.cap0, 0
        verdict = f'capMax {self.capMax} reached'
        while cap <= self.capMax:
            attempt_best, attempt_bestW = None, None
            for attempt in range(self.retries + 1):
                weights, diag = self._stage(cap, attempt)
                if attempt_best is None or diag['cornerFrac'] > attempt_best['cornerFrac']:
                    attempt_best, attempt_bestW = diag, weights
                # A healthy stage needs no retry; only re-tune when transport looks dead.
                if best is None or diag['cornerFrac'] >= 0.5 * best['cornerFrac']:
                    break
            diag, weights = attempt_best, attempt_bestW
            self.history.append(diag)
            log(f'  cap {cap:3d}: pressure {diag["pressure"]:.3f}  returns(min) '
                f'{diag["returns"]:4d}  corner dwell {diag["cornerFrac"]:.4f}'
                f'±{diag["cornerSpread"]:.4f}  P(vac) {diag["vacuumFrac"]:.3f}')
            if diag['returns'] < self.returnFloor:
                verdict = (f'constraint binds: returns {diag["returns"]} < floor '
                           f'{self.returnFloor} at cap {cap}')
                break
            if best is None or diag['cornerFrac'] > best['cornerFrac']:
                best, bestW = diag, weights
                weak_streak = 0
            elif diag['cornerFrac'] < 0.5 * best['cornerFrac']:
                weak_streak += 1
                if weak_streak >= 2:
                    verdict = (f'turnover: corner dwell below half the best '
                               f'({best["cornerFrac"]:.4f} at cap {best["cap"]}) '
                               f'for two consecutive caps')
                    break
            else:
                weak_streak = 0
            cap += self.capStep

        if best is None:
            raise RuntimeError(
                f'no cap satisfied the return floor {self.returnFloor}; '
                'lower cap0 or the floor')
        self.verdict = verdict
        self.best = best
        self._bestWeights = bestW
        log(f'  VERDICT: {verdict}')
        log(f'  chosen cap {best["cap"]}: pressure {best["pressure"]:.3f}, returns '
            f'{best["returns"]}, corner dwell {best["cornerFrac"]:.4f}')
        return bestW

    def score_flips(self, warmStartConfiguration=None, budget=40_000_000):
        r"""Validate the chosen design directly, in $J$ flips per move.

        Runs a gas at the chosen (cap, table) design in 20-move batches,
        and at every :attr:`~.state.FState.legal_vacuum` visit reads $J$ via
        :meth:`~.state.FState.intersection_winding` --- replacing the source
        script's primitive-reconstruction-plus-``IntersectionWinding.Villain``
        route with the same integers computed directly from $F$ (audited by
        :meth:`~.state.FState.intersection_winding`'s own docstring) --- and
        counts how often $J$ differs from the previous vacuum visit.

        Parameters
        ----------
        warmStartConfiguration: dict, optional
            A Villain configuration (as ``S.configurations(...)`` produces)
            to start from via :meth:`~.state.FState.from_configuration`. A
            cold start (default) must first *nucleate* the transporting
            network from $F = 0$, a stochastic wait that confounded every
            earlier single-seed cold score in the source notebook (finding
            13: "0 flips" and "0.53/Mmove" scores there were nucleation
            roulette as much as table quality) --- a cold score here carries
            the same caveat: it measures the basin draw, not just the
            table.
        budget: int
            Minimum compiled moves to advance; the loop stops once at least
            this many moves have run (in units of the 20-move batch, so the
            actual count may exceed ``budget`` slightly).

        Returns
        -------
        (int, int, float)
            ``(flips, moves, chargedFraction)`` --- the number of $J$
            changes observed across vacuum visits, the moves actually run,
            and the fraction of 20-move batches ending with $Q > 0$ (the
            source script's ``P(Q>0)`` diagnostic).

        Raises
        ------
        RuntimeError
            If called before :meth:`tune`.
        """
        if self.best is None:
            raise RuntimeError('score_flips requires tune() to have been called first')
        cap = self.best['cap']
        seed = self.seed + 1000
        gas = self.gasFactory(
            self.S, sectorWeights=self._bestWeights,
            intersectionFugacity=self.intersectionFugacity,
            sectorWeightCap=cap, targetFraction=self.targetFraction,
            pairUmbrella=self.pairUmbrella, measure=False,
            seed=seed, rng=np.random.default_rng(seed))
        state = (FState.from_configuration(self.S, warmStartConfiguration)
                 if warmStartConfiguration is not None else FState(self.S))
        gas.sweep(state, self.equilibrate, pCob=self.pCob)

        moves = flips = charged = batches = 0
        Jprev = None
        while moves < budget:
            gas.sweep(state, 20, pCob=self.pCob)
            moves += 20
            batches += 1
            if state.Q > 0:
                charged += 1
            if state.legal_vacuum:
                J = state.intersection_winding()
                if Jprev is not None and np.any(J != Jprev):
                    flips += 1
                Jprev = J
        chargedFraction = charged / batches if batches else 0.0
        return flips, moves, chargedFraction
