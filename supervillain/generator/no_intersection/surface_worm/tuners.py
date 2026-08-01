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
    (``tune_sector_weights.py``'s ``visit_histogram``/``tune`` and
    ``tune_pair_umbrella.py``'s ``achievable_shells``/``measure``/``update``/
    ``OffsetBisector``/``coverage`` in the no-intersections lab notebook's
    ``j-vacuum-2026-07-31`` snapshot), with the free functions folded into
    methods of two tuner classes and the dict-keyed ``cfg`` re-expressed as
    :class:`~.state.FState`.  See ``test_surface_worm_tuners.py`` for the
    gates.
"""

import time

import numpy as np

from .gas import SurfaceWormGas
from .state import FState
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

    def __init__(self, S, intersectionFugacity, cap, openSurfaceFugacity=0.09,
                 iterations=20, ticks=2000, stride=200, damping=0.7,
                 targetFlatness=1.5, minimumReachable=3, pCob=0.6,
                 targetFraction=0.8, seed=None):
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
        r"""Histogram of $D$ over ``ticks`` samples spaced ``stride`` compiled
        moves apart.

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
            $D$ above this is tallied in ``beyond`` instead of the histogram.
        pCob: float
            Coboundary-move probability.

        Returns
        -------
        (numpy.ndarray, int)
            ``(hist, beyond)`` --- counts for $D = 0 \ldots \texttt{cap}$,
            and how many samples landed at $D > \texttt{cap}$.
        """
        hist = np.zeros(cap + 1, dtype=np.int64)
        beyond = 0
        for _ in range(ticks):
            gas.sweep(state, stride, pCob=pCob)
            D = int(state.D)
            if D <= cap:
                hist[D] += 1
            else:
                beyond += 1
        return hist, beyond

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
        weights = SectorWeights.fugacity(self.openSurfaceFugacity, cap=self.cap)
        self.history = []
        # Which D are REACHABLE at all is LEARNED as the union of everything ever
        # seen, and flatness is judged on that set alone -- see SectorWeights'
        # module docstring / the source script's comment for why treating a
        # geometrically-empty bin as an unflattened barrier destabilizes the table.
        everSeen = np.zeros(self.cap + 1, dtype=bool)
        sinceExpansion = 0          # iterations since the reachable set last grew

        gas = SurfaceWormGas(self.S, sectorWeights=weights,
                             intersectionFugacity=self.intersectionFugacity,
                             sectorWeightCap=self.cap,
                             targetFraction=self.targetFraction,
                             measure=False, seed=self.seed,
                             rng=np.random.default_rng(self.seed))
        state = FState(self.S)

        for it in range(self.iterations):
            gas.setSectorWeights(weights)
            t0 = time.time()
            gas.sweep(state, 5 * self.stride, pCob=self.pCob)     # brief settle
            hist, beyond = self.visit_histogram(gas, state, self.ticks, self.stride,
                                                self.cap, pCob=self.pCob)
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
            self.history.append(dict(
                iteration=it, flatness=flat, occupiedFlatness=occFlat,
                visited=occupied, reachable=int(everSeen.sum()), beyond=beyond,
                histogram=hist.copy(), logWeight=weights.logWeight.copy(),
                seconds=time.time() - t0))
            log(f'  iter {it:>3d}  occupied {occupied:>3d}/{int(everSeen.sum())} reachable  '
                f'beyond cap {beyond:>5d}  flatness {flat:>9.3g}  '
                f'(this iter {occFlat:>7.2f})  ({time.time() - t0:.0f}s)')

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
            step = self.damping / (1.0 + sinceExpansion / 3.0)
            newLog = weights.logWeight + step * update
            weights = SectorWeights(newLog, weights.tailSlope, weights.hardWall)

        unreachable = np.flatnonzero(~everSeen)
        if len(unreachable):
            log(f'  D never visited in [0, {self.cap}]: {list(unreachable)} '
                '(interpolated, NOT assumed empty -- see SectorWeights.interpolated)')
        return weights.interpolated(everSeen)


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
                 targetOdds=0.5, seed=None):
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
        gas = SurfaceWormGas(self.S, sectorWeights=self.sectorWeights,
                             intersectionFugacity=self.intersectionFugacity,
                             sectorWeightCap=self.sectorWeights.cap,
                             targetFraction=self.targetFraction,
                             pairUmbrella=umbrella, measure=True,
                             seed=seed, rng=np.random.default_rng(seed))
        state = FState(self.S)
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
                    best, bestScore = umbrella, score
                    break

            self.history.append(record)
            bisector.update(collapsed or occ > targetOcc)
            umbrella = table(shape, bisector.offset)
            log(f'          {bisector}')

        if best is None:
            log('  WARNING: no healthy iteration ever completed; returning the last table')
            best = umbrella
        return best
