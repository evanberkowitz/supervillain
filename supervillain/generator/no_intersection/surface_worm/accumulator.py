#!/usr/bin/env python

r"""Sector-dwell accumulation for the :class:`~.gas.SurfaceWormGas`, feeding
the library's ``Intersection_Intersection`` correlator.

Everything here is measured on the **closed-F subensemble** ($D = 0$): the $\theta$
operator inserts $q$ into a valid $n$-configuration, so open-surface ticks are
sampler scaffolding and contribute only to ``Ticks``.

The correlator is a ratio of sector dwells,
$\Theta_{\Delta x} = \langle\texttt{Theta\_Theta}\rangle / \langle\texttt{VacuumTicks}\rangle$,
absolutely normalized and independent of the fugacity: the pair sector's dwell is
divided by its known price $V\eta_q^2$ here, and the vacuum sector's price is 1.

.. note ::

    A straight port of the audited reference implementation (``correlator.py``
    in the no-intersections lab notebook's ``swg-audit-2026-07-31`` snapshot),
    with ``etaQ`` renamed :attr:`~CorrelatorAccumulator.intersectionFugacity`
    and :meth:`~CorrelatorAccumulator.tick` reading an
    :class:`~.state.FState` instead of a dict-keyed ``cfg``.  See
    ``test_surface_worm_accumulator.py`` for the gates.
"""

import numpy as np

from .weights import pair_separation_squared


class CorrelatorAccumulator:
    r"""Accumulate sector dwells between emissions.

    .. note ::
        The $\Delta x = 0$ bin of ``Theta_Theta`` is empty by construction --- a
        coincident pair has $q \equiv 0$, which *is* the vacuum, and its ticks are
        counted by ``VacuumTicks``. The observable writes $\Theta_0 = 1$ outright.

    Parameters
    ----------
    N: int
        Linear lattice size.
    intersectionFugacity: float
        The self-intersection fugacity $\eta_q$; the pair sector's price is $\eta_q^2$.
    absoluteChargeCap, squaredChargeCap: int
        Top (saturating) bin of the charge distributions.
    chargeBinWidth: int
        Uniform bin width for both charge distributions.
    sectorCap: int
        Top (saturating) bin of the $Q$ sector histograms (``SectorTicks``,
        ``ClosedSectorTicks``).
    """

    def __init__(self, N, intersectionFugacity, openCap=64,
                 absoluteChargeCap=64, squaredChargeCap=64,
                 chargeBinWidth=1, sectorCap=64):
        self.N = int(N)
        self.V = self.N ** 4
        self.intersectionFugacity = float(intersectionFugacity)
        self.absoluteChargeCap = int(absoluteChargeCap)
        self.squaredChargeCap = int(squaredChargeCap)
        self.chargeBinWidth = int(chargeBinWidth)
        # Sector histogram over Q = #{cells with q != 0}, after the DefectGas's
        # ``SectorTicks`` -- which its source calls "the weight tuner's input and the
        # flat-histogram health check".  The umbrella tuner needs exactly that, and the
        # measurement it replaces was an ad-hoc probe whose two runs disagreed 7% vs 29% on
        # the pair-sector occupancy because one was under-equilibrated.  An inline histogram
        # recorded by every production run cannot drift from the run it describes.
        self.sectorCap = int(sectorCap)
        # D = #{cells with dF != 0}, the open-surface boundary.  Capped at the
        # gas's own sector-weight cap: with a hard wall D cannot exceed it, and
        # one extra bin catches the soft-wall case.
        self.openCap = int(openCap)
        # Set by SurfaceWormGas to the SAME object its acceptance uses.  Left as None here
        # so a bare accumulator behaves exactly as before the umbrella existed.
        self.pairUmbrella = None
        self.reset()

    def reset(self):
        N = self.N
        self.ticks = 0
        self.closedTicks = 0
        self.vacuumTicks = 0
        # float, not int: the umbrella contributes 1/w_2 per tick.
        self.pair = np.zeros((N,) * 4, dtype=float)
        # Sum of SQUARED tick weights, per bin.  Without it the umbrella'd correlator has no
        # error: the estimator is a weighted sum whose variance is dominated by the rarest,
        # largest-weight contributions, so a bin's count says nothing about its precision and
        # an umbrella-vs-baseline comparison cannot separate a wrong 1/w_2 division from
        # Poisson noise.  Also gives the effective sample size (sum w)^2 / sum w^2.
        self.pairWeightSquared = np.zeros((N,) * 4, dtype=float)
        self.pairCounts = np.zeros((N,) * 4, dtype=np.int64)
        self.fourDefect = np.zeros(4, dtype=np.int64)
        self.absoluteCharge = np.zeros(self.absoluteChargeCap + 1, dtype=np.int64)
        self.squaredCharge = np.zeros(self.squaredChargeCap + 1, dtype=np.int64)
        self.maxAbsoluteCharge = 0
        self.maxSquaredCharge = 0
        self.maxPairSeparationSquared = 0
        self.sectorTicks = np.zeros(self.sectorCap + 1, dtype=np.int64)
        # D histograms, the open-surface analogue of sectorTicks: one over ALL
        # ticks (where the chain lives in D) and one restricted to the
        # CHARGE-FREE sector Q = 0.  Both are needed and they answer different
        # questions -- finding 7 of the j-vacuum campaign measured that
        # charge-free open surfaces never flip J (0 flips in 8290 excursions),
        # so the difference between these two histograms isolates the
        # population that can actually transport J.
        self.openTicks = np.zeros(self.openCap + 2, dtype=np.int64)
        self.neutralOpenTicks = np.zeros(self.openCap + 2, dtype=np.int64)
        self.closedSectorTicks = np.zeros(self.sectorCap + 1, dtype=np.int64)
        # Pair separation over ALL ticks -- the umbrella tuner's input, and deliberately
        # NOT restricted to the closed shell: the separation only moves while dF != 0
        # (NOTES 2026-07-30, probe_pair_transport), so a closed-shell histogram would
        # describe a coordinate that does not move under the weight being tuned.
        self.pairSeparationTicks = np.zeros(N ** 2 + 1, dtype=np.int64)
        self.vacuumReturns = 0
        self.wasCharged = False
        self.nontrivialClassTicks = 0

    def tick(self, state):
        r"""Record one Monte-Carlo tick.

        Parameters
        ----------
        state: supervillain.generator.no_intersection.surface_worm.state.FState
            The chain state to record.  Not mutated.
        """
        self.ticks += 1

        # The sector histograms are recorded BEFORE the closed-shell early return, and
        # twice: once over every tick and once over closed ticks only.  Both are needed and
        # they answer different questions.  The all-tick one is the umbrella tuner's input,
        # since the pair separation only moves while dF != 0 -- a histogram restricted to the
        # closed shell would describe a coordinate that is not moving.  The closed-shell one
        # is what the correlator actually samples.  Their ratio is the sampler's efficiency:
        # measured at N=8, kappa=0.04 under the tuned table, only ~4% of ticks are closed.
        Q = int(state.Q)
        self.sectorTicks[min(Q, self.sectorCap)] += 1
        # D histograms, recorded here for the same reason the sector histograms
        # are: before the closed-shell early return, so they describe the whole
        # chain rather than only the D = 0 shell (where D is 0 by definition).
        Dbin = min(int(state.D), self.openCap + 1)
        self.openTicks[Dbin] += 1
        if Q == 0:
            self.neutralOpenTicks[Dbin] += 1
        allSep = pair_separation_squared(state.chargeSites, self.N)
        if allSep is not None:
            self.pairSeparationTicks[allSep] += 1
        if Q == 0 and self.wasCharged:
            self.vacuumReturns += 1          # returns to the defect-free sector
        self.wasCharged = Q != 0

        if state.D != 0:
            return
        # Closed is not exact: a nonzero H^2 class (nonzero component totals at dF = 0)
        # means this "vacuum-shaped" state corresponds to NO (n, phi) configuration, so it
        # must not enter the closed-shell dwells -- neither the vacuum denominator nor the
        # pair-bin numerators (the estimator's sector prices are relative weights of
        # PHYSICAL sectors).  Counted, not silently dropped, so a parameter regime where
        # the chain starts wandering in class is visible in the harvest.
        #
        # Unlike the audit's dict-keyed cfg (where 'periods' could be absent), periods is
        # always present on FState -- built by _rebuild_all and maintained incrementally by
        # every move -- so the cfg.get('periods')-style None-guard collapses to a direct
        # truthiness check.
        if np.any(state.periods):
            self.nontrivialClassTicks += 1
            return
        self.closedTicks += 1
        self.closedSectorTicks[min(Q, self.sectorCap)] += 1

        absolute = int(state.absoluteCharge)
        squared = int(state.squaredCharge)
        self.maxAbsoluteCharge = max(self.maxAbsoluteCharge, absolute)
        self.maxSquaredCharge = max(self.maxSquaredCharge, squared)
        self.absoluteCharge[min(absolute // self.chargeBinWidth,
                                self.absoluteChargeCap)] += 1
        self.squaredCharge[min(squared // self.chargeBinWidth,
                               self.squaredChargeCap)] += 1

        Q = state.Q
        if Q == 0:
            self.vacuumTicks += 1
            return
        if Q == 2:
            (h1, c1), (h2, c2) = state.chargeSites.items()
            if {c1, c2} == {1, -1}:
                plus, minus = (h1, h2) if c1 == 1 else (h2, h1)
                dx = tuple((plus[i] - minus[i]) % self.N for i in range(4))
                # ONE bin per tick, at the +/- displacement.  Do NOT also bin -dx:
                # the DefectGas convention this must match counts each pair once,
                # H_pair(dh) = sum_h [q = delta_{h+dh} - delta_h], and the opposite
                # arrangement fills the -dx bin on its own ticks.  Double-binning
                # would put every bin a factor 2 high and destroy the absolute
                # normalization that makes Theta_0 = 1 meaningful.
                sep = sum(min(d, self.N - d) ** 2 for d in dx)
                # Divide the umbrella out, PER BIN.  This is the one place w_2 differs
                # structurally from w(D): w(D) cancels from Theta because the pair and
                # vacuum dwells both sit at D = 0, but w_2 is r-dependent against an
                # r-resolved numerator, so it survives the ratio and must be removed here.
                # Omit this and every bin is scaled by its own factor -- which reads as a
                # correlator SHAPE, not as a bug.
                wgt = (1.0 if self.pairUmbrella is None
                       else 1.0 / self.pairUmbrella.weight(sep))
                self.pair[dx] += wgt
                self.pairWeightSquared[dx] += wgt * wgt
                self.pairCounts[dx] += 1
                self.maxPairSeparationSquared = max(self.maxPairSeparationSquared, sep)
        if Q <= 4:
            self._four_defect(state)

    def _four_defect(self, state):
        r"""Classify a total-charge-4 state into the four ordered classes the
        library's :class:`~.FourDefects` weights as $(4, 2, 2, 1)$."""
        charges = sorted(state.chargeSites.values())
        if charges == [-1, -1, 1, 1]:
            self.fourDefect[0] += 1
        elif charges == [-1, -1, 2]:
            self.fourDefect[1] += 1
        elif charges == [-2, 1, 1]:
            self.fourDefect[2] += 1
        elif charges == [-2, 2]:
            self.fourDefect[3] += 1

    def harvest(self):
        r"""Return the inline observables and reset. Names follow the library's
        existing observables so they feed ``Intersection_Intersection`` unchanged.

        Returns
        -------
        dict
            The accumulated sector-dwell observables since the last :meth:`reset`.
        """
        out = {
            'Ticks': int(self.ticks),
            'ClosedTicks': int(self.closedTicks),
            'VacuumTicks': int(self.vacuumTicks),
            'Theta_Theta': self.pair / (self.V * self.intersectionFugacity ** 2),
            'Theta_Theta_WeightSquared': (self.pairWeightSquared
                                          / (self.V * self.intersectionFugacity ** 2) ** 2),
            'Theta_Theta_Counts': self.pairCounts.copy(),
            # Per class, not a uniform / eta_q**4: the SWG prices intersections PER
            # OCCUPIED CELL (Q counts cells with q != 0), and the four ordered classes
            # occupy different numbers of cells -- [-1,-1,1,1] 4 cells, [-1,-1,2] and
            # [-2,1,1] 3 cells each, [-2,2] 2 cells -- so they dwell at eta_q**4,
            # eta_q**3, eta_q**3, eta_q**2 respectively.  This is NOT the same
            # normalization as the DefectGas's identically-named 'FourDefectDistribution'
            # column: the DefectGas prices per unit |charge|, and every one of these
            # classes has Sum|q| = 4, so its uniform / zeta**4 is correct as written for
            # that pricing convention.  Only the SWG's per-cell convention needs the
            # per-class exponents below.  After this fix both gases' harvests are the
            # physically-normalized distribution and can be compared directly.
            #
            # Upstream-fix note: the notebook audit toolchain (no-intersections repo,
            # correlator.py) still carries the uncorrected uniform / etaQ**4 version this
            # replaces; port this fix there too before trusting its FourDefectDistribution.
            'FourDefectDistribution': self.fourDefect / self.intersectionFugacity ** np.array([4, 3, 3, 2]),
            'AbsoluteIntersectionChargeDistribution': self.absoluteCharge.copy(),
            'SquaredIntersectionChargeDistribution': self.squaredCharge.copy(),
            'MaxAbsoluteIntersectionCharge': int(self.maxAbsoluteCharge),
            'MaxSquaredIntersectionCharge': int(self.maxSquaredCharge),
            'MaxPairSeparationSquared': int(self.maxPairSeparationSquared),
            # After the DefectGas's SectorTicks/RoundTrips.  SectorTicks spans ALL ticks
            # (the umbrella tuner's input); ClosedSectorTicks only the closed shell (what
            # the correlator samples); VacuumReturns counts returns to Q = 0, which is the
            # quantity the umbrella must not destroy -- a table that pins the chain at large
            # separation starves the vacuum dwell that is Theta's denominator.
            'PairSeparationTicks': self.pairSeparationTicks.copy(),
            'SectorTicks': self.sectorTicks.copy(),
            # D occupancy over all ticks, and over charge-free ticks only.  Their
            # DIFFERENCE is the charged open-surface population -- the one that
            # carries J (j-vacuum finding 7), and the one a transport tuner
            # should be maximizing at the D scale flips actually need.
            'OpenSurfaceTicks': self.openTicks.copy(),
            'NeutralOpenSurfaceTicks': self.neutralOpenTicks.copy(),
            'ClosedSectorTicks': self.closedSectorTicks.copy(),
            'VacuumReturns': int(self.vacuumReturns),
            # Closed-but-non-exact ticks excluded from the closed-shell dwells above.
            # Expected 0 at current parameters (audit: 0/267 emissions, 0 ticks); a
            # nonzero value flags that the chain has started wandering in H^2 class.
            'NontrivialClassTicks': int(self.nontrivialClassTicks),
        }
        self.reset()
        return out
