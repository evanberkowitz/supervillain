#!/usr/bin/env python

r"""The :class:`SurfaceWormGas` --- a grand-canonical extended-ensemble sampler of
the $F$-space No-Intersections model in which BOTH physical constraints are
relaxed and **priced** (not forbidden):

.. math ::

    \pi_\text{ext}(F) \propto e^{-2\pi^2\kappa C(F)}
        \cdot w\big(D(F)\big) \cdot \eta_q^{Q(F)}

with

* $C(F)$ the coexact norm (the $\phi$-marginalized action, $\kappa \neq 0$ part;
  $O(1)$ per toggle via the incremental Green potential),
* $D(F) = \#\{\text{cubes with } dF \neq 0\}$, the closedness (open-surface)
  defect count, priced by the open-surface sector table $w(D)$
  (:class:`~.weights.SectorWeights`), and
* $Q(F) = \#\{\text{hypercubes with } q = F\wedge F \neq 0\}$, the
  self-intersection defect count, priced by the fugacity
  $\eta_q = \texttt{intersectionFugacity}$.

Physical configurations are the joint vacuum $D = Q = 0$ with every period
zero (:attr:`~.state.FState.legal_vacuum`); a later task emits only there.

**Detailed balance is MANIFEST**, which is the whole point: the plaquette
proposal is uniform (component, site, sign all drawn uniformly, or --- with
probability ``targetFraction`` --- targeted at a plaquette incident on an open
cell, with the resulting asymmetry corrected by an explicit Hastings ratio),
so Metropolis against the extended weight above is exactly detailed-balanced
by construction: no adjacency bookkeeping, no orphans. The "gas"/"worm"
character is emergent --- toggles create, move, merge, and annihilate a
fluctuating population of $dF$ and $q$ defects; $w(D)$ and $\eta_q$ tune
their densities.

State lives in an :class:`~.state.FState`: $F$ (the 2-form), $dF$ (its
closure defect), $q$ (its self-intersection density), $G = \Delta^{-1}F$
(the incremental Green potential), and the scalar defect counts $D$, $Q$,
all maintained incrementally per accepted move --- $O(1)$ plus the $O(V)$
Green-column patch.

This module is the python reference sweep (:meth:`SurfaceWormGas.sweep_reference`,
correctness-clear); a compiled inner loop follows in a later task, validated
against THIS reference rather than trusted on seed-independence alone.

.. note ::

    A straight port of the audited reference implementation (``gas.py`` in
    the no-intersections lab notebook's ``swg-audit-2026-07-31`` snapshot),
    with its dict-keyed ``cfg`` state re-expressed as :class:`~.state.FState`
    attribute access, and the ``windingInSampler`` toggle removed --- the
    winding tilt is now unconditionally physical.  See
    ``test_surface_worm_reference.py`` for the gates.
"""

import numpy as np

from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import d, wedge

from .kernel import scalar_green, pot, stencils
from .staircase import primitive_2form
from .weights import SectorWeights, PairUmbrella, pair_separation_squared


class SurfaceWormGas(ReadWriteable, Generator):
    r"""Grand-canonical extended-ensemble sampler of the $F$-space No
    Intersections model (see the module docstring for the extended weight
    $\pi_\text{ext}$).

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The action, used for its lattice and $\kappa$.
    openSurfaceFugacity: float, optional
        The bare open-surface price $\eta_{dF}$, written internally as the
        linear table ``SectorWeights.fugacity(openSurfaceFugacity, cap=sectorWeightCap)``
        --- so an untuned gas is *exactly* the bare-fugacity sampler, which is
        what makes the table's introduction testable rather than a leap.
        Exactly one of ``openSurfaceFugacity`` and ``sectorWeights`` is
        required.
    sectorWeights: supervillain.generator.no_intersection.surface_worm.weights.SectorWeights, optional
        A (possibly tuned, non-linear) open-surface weight table, an
        alternative to ``openSurfaceFugacity``.  A non-linear table is what
        lets the chain cross the suppressed intermediate $D$ region between
        isolated bubbles and a spanning network; a linear one (any fugacity)
        can only shift the distribution, never broaden it.
    intersectionFugacity: float
        The self-intersection price $\eta_q$.
    sectorWeightCap: int
        The cap passed to ``SectorWeights.fugacity`` when ``openSurfaceFugacity``
        is given directly (ignored when ``sectorWeights`` is given).
    targetFraction: float
        Defect-adjacent proposal targeting.  A uniform proposal over all $6V$
        plaquettes touches the open boundary vanishingly rarely once $D$ is
        appreciable, so an edge heals only by diffusion and its tension never
        gets offered the move that would shut it.  With probability
        ``targetFraction`` the proposal instead picks an OPEN cell uniformly
        and then one of the 6 plaquettes incident on it.  This breaks
        proposal symmetry, so the Metropolis test acquires a Hastings ratio
        (see :meth:`_log_proposal_density`).  Default 0.0 reproduces the
        uniform sampler exactly.  Must lie in $[0, 1)$: at exactly 1 the
        uniform component vanishes, so a plaquette touching no open cell has
        proposal probability exactly zero --- $\log q = -\infty$ makes the
        Hastings ratio undefined, and, worse, those plaquettes become
        unreachable, so the sampler is no longer ergodic.  The uniform
        component is what keeps every plaquette proposable.
    pairUmbrella: supervillain.generator.no_intersection.surface_worm.weights.PairUmbrella, optional
        A pair-separation umbrella on the $\pm1$ sector.  Defaults to the
        identity (``PairUmbrella.off(N)``), so a gas built without one is
        *exactly* the un-umbrella'd sampler.
    ticksPerStep: int
        Compiled moves per accumulator tick on the fast (numba) path (a later
        task).
    stride: int
        Compiled moves between consecutive accumulator ticks on the fast
        path (a later task).
    pCob: float
        Default coboundary-move probability for :meth:`sweep_reference` (and
        the fast path, in a later task).
    maxWaitTicks: int
        Safety horizon on the fast path (a later task).
    hardWaitFactor: int
        Safety-horizon multiplier on the fast path (a later task).
    measure: bool
        Whether this gas accumulates correlator statistics.  Stored here;
        the accumulator itself is constructed in a later task (this task's
        moves never tick one).
    seed: int, optional
        Seeds the compiled kernel's own RNG stream in a later task
        (``_build_nb``); kept as plain data here.
    rng: numpy.random.Generator, optional
        The stream the python reference moves draw from.  Defaults to
        ``numpy.random.default_rng()``.
    absoluteChargeCap: int
        Forwarded to the accumulator constructed in a later task.
    squaredChargeCap: int
        Forwarded to the accumulator constructed in a later task.
    chargeBinWidth: int
        Forwarded to the accumulator constructed in a later task.
    """

    def __init__(self, S, openSurfaceFugacity=None, sectorWeights=None,
                 intersectionFugacity=0.3, sectorWeightCap=64, targetFraction=0.0,
                 pairUmbrella=None, ticksPerStep=1000, stride=200, pCob=0.5,
                 maxWaitTicks=200000, hardWaitFactor=10, measure=True,
                 seed=None, rng=None, absoluteChargeCap=64, squaredChargeCap=64,
                 chargeBinWidth=1):
        # openSurfaceFugacity and sectorWeights are two spellings of the same thing --
        # the open-surface price -- since SectorWeights.fugacity(openSurfaceFugacity)
        # IS the linear table.  Accepting both silently lets one win and reads to the
        # caller as two independent knobs.  The DefectGas raises on the identical
        # ambiguity: 'exactly one of fugacity and sectorWeights is required.'
        if (openSurfaceFugacity is None) == (sectorWeights is None):
            raise ValueError(
                'exactly one of openSurfaceFugacity and sectorWeights is required; '
                f'got openSurfaceFugacity={openSurfaceFugacity!r} and '
                f'sectorWeights={sectorWeights!r}.  A tuned table already carries the '
                'open-surface price -- pass openSurfaceFugacity only to '
                'SectorWeights.fugacity() when seeding a tuning run.')
        self.S = S
        self._nb_seed_val = seed        # seeds numba's njit RNG in _build_nb
        self.N = S.Lattice.N
        self.V = self.N ** 4
        self.kappa = float(S.kappa)
        self.intersectionFugacity = float(intersectionFugacity)
        self.lg_q = np.log(self.intersectionFugacity)
        # Sector weight on the open-surface count D.  Defaults to the bare fugacity
        # written as a table, so an untuned gas is EXACTLY the old sampler -- which is
        # what makes the table's introduction testable rather than a leap.
        self.sectorWeights = (sectorWeights if sectorWeights is not None
                              else SectorWeights.fugacity(openSurfaceFugacity, cap=sectorWeightCap))
        # Kept for reporting only.  The acceptance reads the TABLE (dLogSector);
        # nothing multiplies by log(openSurfaceFugacity) anywhere, so this is not a
        # second, competing price.
        self.openSurfaceFugacity = (float(openSurfaceFugacity)
                                     if openSurfaceFugacity is not None else None)
        self._sectorLogWeight, self._sectorTailSlope, self._sectorHardWall = self.sectorWeights.arrays()
        # Pair-separation umbrella on the +/-1 sector.  Defaults to the identity, so a
        # gas built without one is EXACTLY the un-umbrella'd sampler.
        self.pairUmbrella = pairUmbrella if pairUmbrella is not None else PairUmbrella.off(self.N)
        # Defect-adjacent proposal targeting; see the class docstring.
        self.targetFraction = float(targetFraction)
        if not 0.0 <= self.targetFraction < 1.0:
            raise ValueError(
                f'targetFraction must be in [0, 1), got {self.targetFraction}.  At exactly 1 '
                'the uniform component vanishes, so a plaquette touching no open cell has '
                'proposal probability exactly zero: log q = -inf makes the Hastings ratio '
                'undefined, and -- worse -- those plaquettes become unreachable, so the '
                'sampler is no longer ergodic.  The uniform component is what keeps every '
                'plaquette proposable.')
        # How often the targeted branch actually FIRES, which is not targetFraction: it
        # needs D > 0, so the effective rate is targetFraction * P(D>0) and collapses to
        # zero exactly where the surface is usually closed.  Counted rather than assumed.
        self.targetedProposals = 0
        self.closedProposals = 0
        # Incidence, verified by construction: every plaquette meets 4 cells and every
        # cell meets 6 plaquettes, uniformly over components -- so the targeted branch's
        # normalization is the constant 6 rather than a per-cell count.
        self.plaquettesPerCell = 6
        self.cellsPerPlaquette = 4
        # The winding tilt is unconditionally physical: 2*pi^2*kappa/V multiplies every
        # ||M0 + c N^3||^2 term in the winding partition function everywhere it is used
        # (acceptance, heatbath, emission).  There is no "off" switch -- disabling the
        # tilt only ever made sense as a debugging cross-check, and keeping the single
        # physical coefficient means the acceptance and the emitted log-weight are
        # never at risk of disagreeing about which coefficient is "the" one.
        self._windingCoefficient = 2 * np.pi ** 2 * self.kappa / self.V
        self.rng = rng if rng is not None else np.random.default_rng()
        self.g0, self.self_energy = scalar_green(self.N)
        self.dsten, wsten = stencils(self.N)
        for c in range(6):
            if len(self.dsten[c]) != self.cellsPerPlaquette:
                raise ValueError(
                    f'plaquette component {c} meets {len(self.dsten[c])} cells, expected '
                    f'{self.cellsPerPlaquette}; the targeted proposal normalization assumes '
                    'uniform incidence')
        # wedge groups: per plaquette comp c, {hrel: [(pc, rel, v)]} giving the
        # change Delta q[x+hrel] = s*Sum v*F[pc, x+rel] from toggling F[c,x] by s.
        self.wgroups = []
        for c in range(6):
            g = {}
            for (pc, rel, hrel, v) in wsten[c]:
                g.setdefault(hrel, []).append((pc, rel, v))
            self.wgroups.append(g)
        # Winding sensitivity: the winding 4-vector M0(F) = Sum primitive(F) is LINEAR
        # in F, so probing with unit plaquettes gives the whole functional in one
        # O(6V^2) one-time precompute and every move updates it in O(1).  (The
        # staircase primitive is a linear map, so this is well defined even though a
        # unit plaquette is not closed.)
        probe = np.zeros((6,) + (self.N,) * 4, dtype=np.int64)
        self.windingSensitivity = np.zeros((4, 6 * self.V), dtype=np.int64)
        for c in range(6):
            for flat in range(self.V):
                x = np.unravel_index(flat, (self.N,) * 4)
                probe[(c,) + x] = 1
                self.windingSensitivity[:, c * self.V + flat] = (
                    primitive_2form(probe).reshape(4, -1).sum(axis=1))
                probe[(c,) + x] = 0
        # coboundary (relaxation) stencils: da = d(e_mu) is 6 plaquettes, exact
        # (Delta D=0), reshapes closed F locally; Kcob = its coexact self-energy.
        self.cob = []
        self.Kcob = []
        for mu in range(4):
            a = S.Lattice.form(1); aa = np.asarray(a); aa[:] = 0
            aa[(mu,) + (0, 0, 0, 0)] = 1
            Fc = np.asarray(d(a))
            ent = [(int(z[0]), tuple(int(v) % self.N for v in z[1:]), int(Fc[tuple(z)]))
                   for z in np.argwhere(Fc != 0)]
            self.cob.append(ent)
            Ff = np.zeros((6,) + (self.N,) * 4)
            for (pc, off, sg) in ent:
                Ff[(pc,) + off] = sg
            Ft = np.fft.fftn(Ff, axes=(1, 2, 3, 4))
            k = 2 * np.pi * np.fft.fftfreq(self.N)
            K = np.meshgrid(k, k, k, k, indexing='ij')
            k2 = 4 * sum(np.sin(Ki / 2) ** 2 for Ki in K)
            k2s = np.where(k2 == 0, 1.0, k2)
            self.Kcob.append(float((np.abs(Ft) ** 2 / k2s * (k2 != 0)).sum()) / self.V)
        # GATE-ONLY anchor identity, NOT sampler state: the winding shift per unit
        # Delta of a coboundary move ANCHORED AT THE ORIGIN. M0 is linear but its
        # cone-from-origin primitive construction is not translation-invariant, so
        # this per-mu constant is only the shift for a move applied at y=0; the
        # sampler never reads it (_coboundary_log_weights computes the
        # position-resolved shift from windingSensitivity instead, exact at every y).
        # Kept so a gate can check the origin identity directly.
        self.coboundaryWindingShift = np.zeros((4, 4), dtype=np.int64)
        for mu in range(4):
            probe = np.zeros((6,) + (self.N,) * 4, dtype=np.int64)
            for (pc, off, sign) in self.cob[mu]:
                probe[(pc,) + off] += sign
            self.coboundaryWindingShift[mu] = self.winding_of(probe)
        self.windowEdgeWeightMax = 0.0
        self.windowHalfWidthMax = 0
        self.proposed = 0
        self.accepted = 0
        self.cob_proposed = 0
        self.cob_accepted = 0
        # Cadence knobs for the compiled fast path (a later task); stored here as
        # plain data so the constructor is the single source of truth for them.
        self.ticksPerStep = int(ticksPerStep)
        self.stride = int(stride)
        self.pCob = float(pCob)
        self.maxWaitTicks = int(maxWaitTicks)
        self.hardWaitFactor = int(hardWaitFactor)
        # Accumulator plumbing: 'measure' is stored now, but the accumulator itself
        # is built in a later task -- this task's moves never tick one.
        self.measure = bool(measure)
        self.accumulator = None
        self.absoluteChargeCap = int(absoluteChargeCap)
        self.squaredChargeCap = int(squaredChargeCap)
        self.chargeBinWidth = int(chargeBinWidth)

    def winding_of(self, F):
        r"""The winding 4-vector $M_0(F) = \sum_x \left[\text{primitive}(F)\right]_\mu(x)$,
        recomputed from scratch.  Linear in ``F``; the incremental route through
        :attr:`windingSensitivity` is what the sampler actually uses."""
        return primitive_2form(np.asarray(F, dtype=np.int64)).reshape(4, -1).sum(axis=1)

    def _log_winding_weight(self, winding):
        r"""$\log Z_\text{wind}$, the log of the winding partition function
        $Z_\text{wind}(F) = \sum_c e^{-2\pi^2\kappa\|M_0 + cN^3\|^2/V}$ summed over
        the winding coset, from the winding 4-vector.

        .. note ::
            The sum is dominated by the minimum-norm coset representative --- the
            adjacent-sector gap is $2\pi^2\kappa N^2$, e.g. $\approx 63$ at $N=4$,
            $\kappa=0.2$ --- but the full window is summed so that nothing is assumed
            about the frozen sector at small $\kappa$, where the gap closes."""
        quantum = self.N ** 3
        a = self._windingCoefficient
        total = 0.0
        for mu in range(4):
            centre = int(round(-winding[mu] / quantum))
            cs = np.arange(centre - 4, centre + 5)
            lw = -a * (winding[mu] + cs * quantum) ** 2
            m = lw.max()
            total += m + np.log(np.exp(lw - m).sum())
        return float(total)

    def _log_extended_weight(self, F):
        r"""Globally recomputed $\log \pi_\text{ext}(F)$ --- the coexact norm, both
        defect prices, and the winding weight --- from ``F`` alone.

        For gates only: it is $O(V\log V)$ and exists so the incremental acceptance
        exponent can be checked against an independent computation."""
        N = self.N
        C = sum(float((F[c] * pot(F[c], N)).sum()) for c in range(6))
        f = self.S.Lattice.form(2); np.asarray(f)[...] = F
        dFg = np.asarray(d(f)).astype(np.int64)
        qg = np.asarray(wedge(f, f)).astype(np.int64).reshape((N,) * 4)
        return (-2 * np.pi ** 2 * self.kappa * C
                + float(self.sectorWeights(int((dFg != 0).sum())))
                + int((qg != 0).sum()) * self.lg_q
                + self._log_winding_weight(self.winding_of(F)))

    # ---- plaquette toggle: the "worm" move (opens surfaces; changes dF and q)
    def _log_proposal_density(self, openCells, D):
        r"""$\log q$ for proposing one specific plaquette, up to the $\pm$ sign factor
        (which is symmetric and cancels).

        The proposal is a mixture: with probability $1-f$ a uniform plaquette, and with
        probability $f$ an open cell drawn uniformly from the $D$ of them followed by one
        of its 6 incident plaquettes.  A given plaquette $P$ is therefore reachable through
        the targeted branch once per open cell it touches, so

        .. math ::
            q(P) = \frac{1-f}{6V} + f\,\frac{m(P)}{6D},

        with $m(P)$ the number of $P$'s 4 cells that are open.

        .. note ::
            When $D = 0$ the targeted branch has no cell to draw, so the proposal is
            purely uniform and $q = 1/6V$.  This state dependence is fine --- and must
            be applied in **both** directions --- because Hastings needs the true $q$ of
            each state, not a single formula.
        """
        sites = 6 * self.V
        if D == 0 or self.targetFraction == 0.0:
            return -np.log(sites)
        return np.log((1.0 - self.targetFraction) / sites
                      + self.targetFraction * openCells / (self.plaquettesPerCell * D))

    def _propose_plaquette(self, state):
        r"""Draw a plaquette ``(c, x)``: uniform, or --- with probability
        ``targetFraction`` and when the surface is open --- one incident on a randomly
        chosen open cell."""
        N = self.N
        if state.D == 0:
            self.closedProposals += 1
        if self.targetFraction > 0.0 and state.D > 0 and self.rng.random() < self.targetFraction:
            self.targetedProposals += 1
            open_ = np.argwhere(state.dF != 0)
            pick = open_[self.rng.integers(len(open_))]
            cc, h = int(pick[0]), tuple(int(v) for v in pick[1:])
            # Invert the incidence: toggling plaquette (c, h - off) moves cell (cc, h).
            candidates = [(c, tuple((h[i] - off[i]) % N for i in range(4)))
                          for c in range(6)
                          for (ccc, off, _) in self.dsten[c] if ccc == cc]
            return candidates[self.rng.integers(len(candidates))]
        c = int(self.rng.integers(6))
        return c, tuple(int(v) for v in self.rng.integers(N, size=4))

    def _plaquette_log_acceptance(self, state, c, x, s):
        r"""The Metropolis log-acceptance for toggling ``state.F[c,x]`` by ``s``,
        together with the incremental bookkeeping the accept branch needs.

        Split out of :meth:`_plaquette_move` so a gate can compare this exponent
        against a globally recomputed change in $\log \pi_\text{ext}$ --- the move is
        otherwise untestable except through its accept statistics.

        Returns ``(lnA, dD, dQ, cube_new, q_new, idx)``."""
        N = self.N
        F, dF, q, G = state.F, state.dF, state.q, state.G
        twopi2k = 2 * np.pi ** 2 * self.kappa
        dC = 2 * s * G[(c,) + x] + self.self_energy
        dD = 0
        cube_new = []
        for (cc, off, sign) in self.dsten[c]:
            cell = (cc,) + tuple((x[i] + off[i]) % N for i in range(4))
            old = dF[cell]; new = old + s * sign
            dD += (1 if new != 0 else 0) - (1 if old != 0 else 0)
            cube_new.append((cell, new))
        dQ = 0
        q_new = []
        for hrel, terms in self.wgroups[c].items():
            h = tuple((x[i] + hrel[i]) % N for i in range(4))
            dq = s * sum(v * F[(pc,) + tuple((x[i] + rel[i]) % N for i in range(4))]
                         for (pc, rel, v) in terms)
            if dq == 0:
                continue
            old = q[h]; new = old + dq
            dQ += (1 if new != 0 else 0) - (1 if old != 0 else 0)
            q_new.append((h, new))
        idx = c * self.V + int(np.ravel_multi_index(x, (N,) * 4))
        dLogWinding = (self._log_winding_weight(state.winding + s * self.windingSensitivity[:, idx])
                       - self._log_winding_weight(state.winding))
        # The open-surface price is w(D), not a bare fugacity^D: read the table at the
        # OLD and NEW total D rather than multiplying the increment by a constant.  For
        # the default fugacity table the two agree identically, since log w is then
        # linear in D.
        dLogSector = self.sectorWeights.delta(state.D, state.D + dD)
        # Hastings ratio q(x'->x)/q(x->x').  Identically zero for the uniform proposal,
        # so targetFraction=0 leaves the old acceptance untouched.
        openBefore = sum(1 for (cell, _) in cube_new if dF[cell] != 0)
        openAfter = sum(1 for (_, new) in cube_new if new != 0)
        dLogProposal = (self._log_proposal_density(openAfter, state.D + dD)
                        - self._log_proposal_density(openBefore, state.D))
        # Pair-separation umbrella.  The move may enter, leave, or move within the +/-1
        # sector; `delta` treats "not in the sector" as weight 1 on that side, so w_2 is
        # a genuine state function and detailed balance is untouched.  Identically zero
        # for the default (off) umbrella, so this line cannot change an un-umbrella'd run.
        dLogUmbrella = self.pairUmbrella.delta(
            pair_separation_squared(state.chargeSites, N),
            pair_separation_squared(self._charge_sites_after(state, q_new), N))
        lnA = (-twopi2k * dC + dLogSector + dQ * self.lg_q + dLogWinding + dLogProposal
               + dLogUmbrella)
        return (lnA, dD, dQ, cube_new, q_new, idx)

    @staticmethod
    def _charge_sites_after(state, q_new):
        """The charged-site map this move would produce, without mutating anything.

        Needed because the umbrella weight depends on the pair separation AFTER the
        move, and the acceptance is computed before the move is applied.  Cells driven
        to zero must be REMOVED, not left at zero: `pair_separation_squared` counts
        entries, so a stale zero would make a two-defect state look like three and
        silently switch the umbrella off for exactly the configurations it exists to
        weight.
        """
        sites = dict(state.chargeSites)
        for (h, new) in q_new:
            if new:
                sites[h] = int(new)
            else:
                sites.pop(h, None)
        return sites

    def _plaquette_move(self, state):
        r"""Propose and Metropolis-test one plaquette toggle against ``state``."""
        self.proposed += 1
        c, x = self._propose_plaquette(state)
        s = 1 if self.rng.random() < 0.5 else -1
        lnA, dD, dQ, cube_new, q_new, idx = self._plaquette_log_acceptance(state, c, x, s)
        if np.log(self.rng.random()) < lnA:
            self.accepted += 1
            self._apply_plaquette(state, c, x, s, dD, dQ, cube_new, q_new, idx)

    def _apply_plaquette(self, state, c, x, s, dD, dQ, cube_new, q_new, idx):
        r"""Commit an accepted plaquette toggle: the incremental bookkeeping for $F$,
        $dF$, $q$, the Green potential, the winding, the periods, and the defect
        counts.

        Split out of :meth:`_plaquette_move` so a gate can drive a *chosen* move
        rather than waiting for the sampler to propose it -- which is what the
        detailed-balance check needs, and it must exercise this code rather than a
        re-derivation of it."""
        state.F[(c,) + x] += s
        state.periods[c] += s
        state.winding = state.winding + s * self.windingSensitivity[:, idx]
        for (cell, new) in cube_new:
            state.dF[cell] = new
        for (h, new) in q_new:
            old = state.q[h]
            state.absoluteCharge += abs(int(new)) - abs(int(old))
            state.squaredCharge += int(new) ** 2 - int(old) ** 2
            if new != 0:
                state.chargeSites[h] = int(new)
            else:
                state.chargeSites.pop(h, None)
            state.q[h] = new
        state.G[c] += s * np.roll(self.g0, x, axis=(0, 1, 2, 3))
        state.counts['D'] += dD
        state.counts['Q'] += dQ

    # ---- coboundary HEATBATH: the relaxation move F += Delta*d(e_mu), Delta in Z
    #      sampled from its full conditional. Exact => DeltaD=0. C is quadratic in
    #      Delta (discrete Gaussian) and, since da^da=0, q is LINEAR in Delta (so the
    #      intersection count is a step function) -- enumerate a window, softmax-
    #      sample. DB is exact up to the tail tolerance: centering on round(-L/K)
    #      makes the reverse window's mean shift by exactly -Delta, but the adaptive
    #      half-width H can in principle differ between the forward and reverse
    #      draws; any resulting normalization mismatch is bounded by ~2*(edge
    #      tolerance), i.e. <~5e-12, and has never been observed to matter in
    #      practice.
    def _coboundary_log_weights(self, state, mu, y, edge_tol=1e-14, H_cap=256, H0=None):
        r"""The candidate offsets and their log-weights for a $\mu$-coboundary move at
        ``y``, plus the bookkeeping the accept branch needs.

        Split out of :meth:`_coboundary_heatbath` so a gate can compare these weights
        against a globally recomputed change in $\log \pi_\text{ext}$ --- the tilt is
        otherwise untestable except through its sampling statistics.

        ``edge_tol``, ``H_cap``, and ``H0`` default to the production tolerance, cap,
        and initial half-width (``None`` meaning "use the usual floor formula"); a
        gate overrides them (e.g. an artificially tiny ``H0``) to force the
        window-doubling loop, and its cap-exceeded assert, to execute at least once
        each --- production behaviour is unchanged when they are left at their
        defaults.

        Returns ``(deltas, logw, plq, aff, shift)``."""
        N = self.N
        F, q, G = state.F, state.q, state.G
        twopi2k = 2 * np.pi ** 2 * self.kappa
        plq = [(pc, tuple((y[i] + off[i]) % N for i in range(4)), sign)
               for (pc, off, sign) in self.cob[mu]]           # unit da: (pc, xp, sign)
        Kc = self.Kcob[mu]
        L = sum(sign * G[(pc,) + xp] for (pc, xp, sign) in plq)   # <F,da>_C
        # per-unit q change dq1[h] = 2(F^da)[h], accumulated over the 6 plaquettes
        dq1 = {}
        for (pc, xp, sign) in plq:
            for hrel, terms in self.wgroups[pc].items():
                h = tuple((xp[i] + hrel[i]) % N for i in range(4))
                c1 = sign * sum(
                    v * F[(pcc,) + tuple((xp[i] + rel[i]) % N for i in range(4))]
                    for (pcc, rel, v) in terms)
                if c1:
                    dq1[h] = dq1.get(h, 0) + c1
        aff = [(h, q[h], dq1h) for h, dq1h in dq1.items() if dq1h != 0]
        # enumeration window centered on the Gaussian mean; half-width H is
        # STATE-DEPENDENT (widened by the tolerance loop below, not fixed)
        mean = -L / Kc
        center = int(np.round(mean))
        sigma = 1.0 / np.sqrt(2 * twopi2k * Kc)                  # var of the Delta-Gaussian
        # Winding shift per unit Delta, AT THIS y.  self.coboundaryWindingShift[mu] is
        # a gate-only anchor identity (the shift for a coboundary at y=0), NOT sampler
        # state: the primitive's cone-from-origin construction is not translation-
        # invariant, so a shifted copy's true M0 differs from the origin value by an
        # exact integer multiple of the winding quantum N^3 at ~99% of positions
        # (verified by direct enumeration). That multiple is invisible to
        # _log_winding_weight, which is exactly periodic under such shifts, so the
        # heatbath's acceptance ratios would still come out right -- but the raw
        # state.winding integer would drift from winding_of(F) by a growing number of
        # quanta over many moves. M0 IS linear (verified in gate_winding.py), so
        # summing the exact, position-resolved windingSensitivity contributions of
        # this y-translated pattern's plaquettes is exact at every y, at the same
        # O(1) cost.
        shift = sum(sign * self.windingSensitivity[:, pc * self.V + int(np.ravel_multi_index(xp, (N,) * 4))]
                    for (pc, xp, sign) in plq)
        base = self._log_winding_weight(state.winding)
        H = H0 if H0 is not None else max(4, int(np.ceil(6 * sigma)) + 2)
        while True:
            deltas = np.arange(center - H, center + H + 1)
            logw = np.empty(len(deltas))
            for i, Delta in enumerate(deltas):
                dQ = 0
                for (h, q0, dq1h) in aff:
                    dQ += (1 if q0 + Delta * dq1h != 0 else 0) - (1 if q0 != 0 else 0)
                logw[i] = (-twopi2k * (2 * Delta * L + Delta * Delta * Kc)
                           + dQ * self.lg_q
                           + self._log_winding_weight(state.winding + Delta * shift) - base)
            edge = np.exp(max(logw[0], logw[-1]) - logw.max())
            if edge < edge_tol or H > H_cap:
                break
            H *= 2
        self.windowEdgeWeightMax = max(self.windowEdgeWeightMax, float(edge))
        self.windowHalfWidthMax = max(self.windowHalfWidthMax, int(H))
        assert H <= H_cap, f'coboundary window failed to reach tolerance by H={H_cap}'
        return deltas, logw, plq, aff, shift

    def _coboundary_heatbath(self, state):
        r"""One coboundary heatbath move: draw $\mu$ and an anchor $y$ uniformly, then
        the shift $\Delta$ from its exact full conditional (:meth:`_coboundary_log_weights`),
        and commit it."""
        self.cob_proposed += 1
        N = self.N
        F, q, G = state.F, state.q, state.G
        mu = int(self.rng.integers(4))
        y = tuple(int(v) for v in self.rng.integers(N, size=4))
        deltas, logw, plq, aff, shift = self._coboundary_log_weights(state, mu, y)
        p = np.exp(logw - logw.max()); p /= p.sum()
        D = int(deltas[self.rng.choice(len(deltas), p=p)])
        if D == 0:
            return
        self.cob_accepted += 1
        dQ = 0
        for (h, q0, dq1h) in aff:
            new = q0 + D * dq1h
            dQ += (1 if new != 0 else 0) - (1 if q0 != 0 else 0)
            state.absoluteCharge += abs(int(new)) - abs(int(q0))
            state.squaredCharge += int(new) ** 2 - int(q0) ** 2
            if new != 0:
                state.chargeSites[h] = int(new)
            else:
                state.chargeSites.pop(h, None)
            q[h] = new
        for (pc, xp, sign) in plq:
            F[(pc,) + xp] += D * sign
            # da is exact, so these six updates cancel per component -- maintained
            # explicitly anyway, so the invariant never relies on that argument.
            state.periods[pc] += D * sign
            G[pc] += (D * sign) * np.roll(self.g0, xp, axis=(0, 1, 2, 3))
        state.counts['Q'] += dQ
        state.winding = state.winding + D * shift

    # ---- interleaved sweep (both move types are pi-invariant => mixture is DB)
    def sweep_reference(self, state, nmoves, pCob=None):
        r"""Python REFERENCE sweep: each of ``nmoves`` is a coboundary HEATBATH
        (relaxation) move with probability ``pCob``, else a plaquette (worm) $\pm1$
        move.  A compiled ``sweep`` (a later task) reproduces this; validate it
        against THIS, not against seed-independence.

        Parameters
        ----------
        state: supervillain.generator.no_intersection.surface_worm.state.FState
            Mutated in place.
        nmoves: int
            How many moves to make.
        pCob: float, optional
            Coboundary-move probability; defaults to the constructor's ``pCob``.

        Returns
        -------
        FState
            ``state``, for chaining.
        """
        if pCob is None:
            pCob = self.pCob
        for _ in range(nmoves):
            if self.rng.random() < pCob:
                self._coboundary_heatbath(state)
            else:
                self._plaquette_move(state)
        return state

    def setSectorWeights(self, weights):
        r"""Swap the open-surface weight table $w(D)$ in place.

        Everything else the gas holds --- the Green's function, the wedge stencils,
        the coboundary tables, and above all ``windingSensitivity`` --- is a function
        of the lattice alone and is unchanged by a new table.  Rebuilding the whole
        object to change $w$ costs an $O(6V^2)$ ``windingSensitivity`` precompute for
        nothing: **238 s measured at $N=6$**, and it scales as $V^2$, so ~256x that at
        $N=12$.  A multicanonical tuner changes the table every iteration, so paying
        construction each time dominates the entire tuning run and made $N \geq 8$
        unreachable.

        The compiled kernel (a later task) takes ``sectorLogWeight``/``sectorTailSlope``
        as arguments on every call, so nothing needs recompiling either.

        .. note ::
            Only the table changes; the chain's state lives in the caller's
            :class:`~.state.FState` and is untouched, so a tuner that reuses one gas
            continues its walk under the new weights rather than restarting cold.
            That is the usual multicanonical warm start and is what makes successive
            iterations cheap.

        Parameters
        ----------
        weights: supervillain.generator.no_intersection.surface_worm.weights.SectorWeights
            The replacement table.

        Returns
        -------
        SurfaceWormGas
            ``self``, for chaining.
        """
        self.sectorWeights = weights
        self._sectorLogWeight, self._sectorTailSlope, self._sectorHardWall = weights.arrays()
        return self
