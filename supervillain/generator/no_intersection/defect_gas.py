#!/usr/bin/env python

import logging
from types import SimpleNamespace

import numpy as np

from supervillain.generator import Generator
from supervillain.generator.combining import Sequentially
from supervillain.h5 import ReadWriteable
from supervillain.batch import Batch
from supervillain.lattice import Form, d
from supervillain.generator.villain.site import SiteUpdate
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge
from supervillain.generator.no_intersection import defect_gas_kernel
import supervillain.action

logger = logging.getLogger(__name__)


class DefectGas(ReadWriteable, Generator):
    r"""
    Grand-canonical defect sampler for the $q = dn \wedge dn = 0$ constraint in 4D,
    with the inline estimator of the Lagrange-multiplier correlator
    $\left\langle e^{+i\theta_{x}} e^{-i\theta_{y}} \right\rangle$.

    The update starts from a constraint-satisfying defect-free configuration and
    makes updates by sampling in the enlarged ensemble of configurations with any arbitrary number of defects, with each defect weighted by the fugacity $\zeta$,

    .. math ::
        \Pi = \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n]}\, \zeta^{D(n)},
        \qquad D(n) = \sum_{x} \left|q_{x}(n)\right|,
        \quad q = dn \wedge dn,

    The number of defects $D$ is always even since $Q=\sum_h q_h = \sum_h dJ = 0$
    on every configuration, so $D/2$ counts the $\pm$ pairs in flight --- the number of
    worms the grand-canonical ensemble runs at once.
    Though, to be perfectly clear: there are no dedicated 'worms' which need to be advanced using any special worm moves.
    We just update links at random and they do what they want: idling, building up flux, creating and annihilating defects, and transporting charge.

    Standard single-link updates $n\to n \pm 1$ are Metropolis tested with the 
    fugacity included in the weight and are accepted with probability $\min\!\left(1, e^{-\Delta S}\,
    \zeta^{\Delta D}\right)$.  Every link can always receive a proposal but a proposal with
    a messy $\Delta q$ is exponentially discounted by the fugacity $\zeta < 1$, and defect-annihilating moves
    are correspondingly incentivized.  The higher-defect sectors are the corridors through
    jammed backgrounds that clean worms lack, so the sampler cannot jam; single-link
    moves connect every $n$ with nonzero acceptance, making ergodicity on the enlarged
    space manifest.

    The :meth:`step` advances the enlarged chain until its
    ``emit_every``-th visit to the vacuum sector and emits that configuration.
    Restricted to the vacuum sector the enlarged weight is $e^{-S_{V}} \zeta^{0} = e^{-S_{V}}$,
    so the emitted ensemble is exactly the constrained theory.  Like the worms,
    invalid states live only *inside* a step, and every emitted configuration satisfies
    $q \equiv 0$ and therefore this generator can be combined with other constrained generators.

    In a spirit similar to the worms, we can read off defect correlation functions by
    tallying the state of the chain after every proposal.
    Each positive defect amounts to an insertion of $e^{+i\theta}$ and each negative defect amounts to an insertion of $e^{-i\theta}$; for more details see :meth:`~supervillain.generator.no_intersection.DefectGas.inline_observables`.

    .. danger ::

        However, the fugacity $0 < \zeta < 1$ must be handled with care.
        Too large and the defects will proliferate and never return to the vacuum sector.
        Too small and the defects will become exceedingly rare and the samples will not explore the full grand-canonical ensemble.

        In other words, the limit $\zeta\to 1$ is the unconstrained theory while $\zeta\to 0$ restricts to defect-free configurations.
        
    So, we need to think about how to tune the fugacity $\zeta$.
    Entropy pushes $D$ upward (each defect may live anywhere, and the denser
    the sheet the larger a single link's $\left|\Delta D\right|$), so the right $\zeta$
    shrinks with volume and with $1/\kappa$.  Symptoms of a bad choice are loud: too
    large and $D$ pins at ``D_max`` with the vacuum never revisited (:meth:`step` then
    raises rather than hang), too small and pair excursions become needlessly rare.
    The :class:`DefectGasFugacityTuner` automates the choice.

    Parameters
    ----------
    S: a NoIntersections action
        Supplies $S_{V}$, $\kappa$, and the lattice.
    fugacity: float, optional
        The per-defect fugacity $\zeta \in (0, 1]$.  Exactly one of ``fugacity``
        and ``weights`` is required.
    D_max: int or None
        Hard cap on $D$; ``None`` uncaps (geometric path only --- ``weights`` pins it).
    emit_every: int, optional
        Vacuum ticks per :meth:`step` emission; defaults to $4 N^{4}$ (about one
        sweep's worth of vacuum time).
    max_step_sweeps: int
        Safety horizon: :meth:`step` raises after this many sweeps without reaching
        ``emit_every`` vacuum ticks (the defect-condensation symptom).
    rng: numpy Generator, optional
    weights: sequence of floats, optional
        The per-sector weights $w_k$ for $k = D/2 = 0, \ldots, K$, an alternative to
        the geometric ``fugacity`` ($w_k = \zeta^{2k}$).  Normalized so $w_0 = 1$;
        the length pins ``D_max`` $= 2K$.  Chosen well (see
        :class:`DefectGasWeightTuner`) the sector occupancies flatten, so the chain
        shuttles freely between the vacuum and the multi-pair sectors instead of
        paying $e^{-\langle D/2 \rangle}$ for them.

    .. warning ::

        Restricted to $D = 4$.  As a :class:`~supervillain.generator.Generator` this
        updates $n$ only, so it is not ergodic on its own; at least combine it with a
        $\phi$-update such as :class:`~supervillain.generator.villain.SiteUpdate`.
    """

    # D == 4 sector classes for the fourth moment, keyed by the sorted charge values:
    # index 0: {+1,+1,-1,-1}, 1: {+2,-1,-1}, 2: {+1,+1,-2}, 3: {+2,-2}.
    _FOUR = {(-1, -1, 1, 1): 0, (-1, -1, 2): 1, (-2, 1, 1): 2, (-2, 2): 3}

    def __init__(self, S, fugacity=None, D_max=None, emit_every=None, max_step_sweeps=500,
                 rng=None, weights=None):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('DefectGas requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('DefectGas is only implemented for D = 4.')
        if (fugacity is None) == (weights is None):
            raise ValueError('exactly one of fugacity and weights is required.')
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.ndim != 1 or len(w) < 2:
                raise ValueError(f'weights must be a 1D sequence of at least two sectors; got shape {w.shape}.')
            if not np.all(w > 0):
                raise ValueError('weights must be positive.')
            w = w / w[0]
            if D_max is None:
                D_max = 2 * (len(w) - 1)
            elif D_max != 2 * (len(w) - 1):
                raise ValueError(f'weights of length {len(w)} pin D_max = {2 * (len(w) - 1)}; got D_max={D_max}.')
        else:
            if not (0 < fugacity <= 1):
                raise ValueError(f'fugacity must be in (0, 1]; got {fugacity}.')
            w = np.zeros(0, dtype=np.float64)   # empty table: the kernel's geometric path

        self.S = S
        self.L = S.Lattice
        self.N = self.L.N
        self.kappa = S.kappa
        self.fugacity = None if fugacity is None else float(fugacity)
        self.w = w
        # The tally prices unify the two paths: the D = 2 histogram is divided by _w1
        # and the D = 4 classes by _w4 at emission.
        if self.fugacity is not None:
            self._w1, self._w4 = self.fugacity**2, self.fugacity**4
        else:
            self._w1 = float(w[1])
            self._w4 = float(w[2]) if len(w) > 2 else 1.0
        self.D_max = D_max
        self.emit_every = emit_every if emit_every is not None else 4 * self.N**4
        self.max_step_sweeps = max_step_sweeps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.D_trace = []           # per-sweep defect count (diagnostic)
        self.accepted = 0
        self.proposed = 0
        # The persistent chain state used by step(); rebuilt whenever the incoming
        # configuration does not match (first call, or another generator moved it).
        self._state = None

    def __str__(self):
        if self.fugacity is not None:
            return f'DefectGas(fugacity={self.fugacity}, D_max={self.D_max})'
        return f'DefectGas(weights={np.array2string(self.w, precision=4)}, D_max={self.D_max})'

    # ---------------------------------------------------------------- chain internals

    def _init_state(self, phi, n):
        # The whole state the moves need, maintained incrementally so a proposal costs
        # O(1): the field strength F = dn (the per-link charge stencils read it), the
        # charge density q = dn∧dn only through its NONZERO cells (the sparse `defects`
        # dict -- on a well-tuned chain almost all of q is zero), and the scalar defect
        # count D = Σ|q| that the fugacity prices.  (The TOTAL charge Q = Σ q is
        # identically zero for every n -- q is exact -- so every sector is neutral.)
        L = self.L
        st = SimpleNamespace()
        st.n = np.asarray(n).astype(np.int64).copy()
        st.phi = np.asarray(phi).astype(float).copy()
        # 0-forms carry a leading singleton component axis, (1,) + dims; accept bare
        # (N, ..., N) arrays too (the cold default) by restoring it.
        if st.phi.shape == tuple(L.dims):
            st.phi = st.phi.reshape((1,) + tuple(L.dims))
        st.F = np.asarray(d(Form(st.n, degree=1, lattice=L))).astype(np.int64)
        q_arr = np.asarray(charge(Form(st.n, degree=1, lattice=L))).astype(np.int64)
        # defects: nonzero hypercubes, keyed by the 4-tuple cell (component axis stripped).
        st.defects = {tuple(int(x) for x in z[1:]): int(q_arr[tuple(z)])
                      for z in np.argwhere(q_arr != 0)}
        st.D = int(np.abs(q_arr).sum())
        # Flat views and the dense charge state for the compiled tick loop (step); the
        # sparse dict above serves the pure-python step_reference.  n2/F2 are VIEWS, so
        # kernel writes keep st.n/st.F current; q/nzc mirror the dict and are kept in
        # sync by BOTH paths so step and step_reference may be freely interleaved.
        st.F2 = st.F.reshape(st.F.shape[0], -1)
        st.n2 = st.n.reshape(4, -1)
        st.q = np.ascontiguousarray(q_arr.ravel())
        st.nzc = np.zeros(self.N**4, dtype=np.int64)
        nz = np.flatnonzero(st.q)
        st.nzc[:len(nz)] = nz
        st.nnz = int(len(nz))
        # Transport instrumentation, shared by both paths: [current excursion length,
        # completed excursions, max single-pair min-image separation squared] and the
        # power-of-two excursion-length histogram.  These turn empty far bins into
        # honest transport-censoring statements instead of silent zeros.
        st.tstate = np.zeros(3, dtype=np.int64)
        st.exc_hist = np.zeros(32, dtype=np.int64)
        st.dphi = np.ascontiguousarray(d(Form(st.phi, degree=0, lattice=L)))
        st.dphi2 = st.dphi.reshape(4, -1)
        st.n_links = 4 * self.N**4
        self._draw_batch(st)
        return st

    def _draw_batch(self, st):
        # Draw a sweep's worth of proposals up front (numpy batching); the proposal
        # distribution -- uniform link, uniform c = ±1 -- is SYMMETRIC, so plain
        # Metropolis needs no Hastings factor.  This is the whole move set: no
        # templates, no clean sets, no directions.
        rng, N = self.rng, self.N
        st.mus = rng.integers(0, 4, size=st.n_links)
        st.sites = rng.integers(0, N, size=(st.n_links, 4))
        st.cs = rng.choice((-1, 1), size=st.n_links)
        st.us = rng.uniform(0, 1, size=st.n_links)
        st.i = 0

    def _tick(self, st):
        r"""One clock tick of the enlarged chain: a single-link Metropolis proposal
        ($\phi$ frozen for the whole step).  Returns
        ``(vacuum, pair_displacement)`` classifying the sector the chain sits in at
        this tick --- the raw material of every estimate this class makes."""
        if st.i == st.n_links:
            # Sweep boundary.  φ is FROZEN for the whole step -- the chain then
            # preserves the conditional π_ζ(n | φ), whose vacuum trace preserves
            # π(n | φ, q ≡ 0), so the step composes Gibbs-style with a φ-update
            # (SiteUpdate) in a Sequentially, exactly like the other n-only updates.
            self.D_trace.append(st.D)
            self._draw_batch(st)
        i = st.i
        st.i += 1
        mu = int(st.mus[i])
        site = (int(st.sites[i, 0]), int(st.sites[i, 1]),
                int(st.sites[i, 2]), int(st.sites[i, 3]))
        c = int(st.cs[i])
        # The link's charge response on the CURRENT background, from the same local
        # stencils the clean worms use -- but here a messy Δq is not a rejection, it is
        # a price: ΔD counts how many units of |q| the move creates (+) or
        # annihilates (-), summed over the touched hypercubes.
        dq = local_charge.charge_change_from_link(st.F, mu, site, c, self.N)
        dD = 0
        for cell, dv in dq.items():
            q0 = st.defects.get(cell, 0)
            dD += abs(q0 + dv) - abs(q0)
        self.proposed += 1
        if self.D_max is None or st.D + dD <= self.D_max:
            # Metropolis on the ENLARGED weight e^{-S_V} ζ^D: the Villain ΔS is local
            # to this one link, and the constraint enters only through ζ^ΔD --
            # defect-annihilating moves (ΔD < 0) are REWARDED, which is what lets the
            # mess clean itself up.  (The D_max cap above is just a truncated state
            # space: proposals past it are ordinary zero-weight rejections.)
            link = (mu,) + site
            A = st.dphi[link] - 2 * np.pi * st.n[link]
            dS = (self.kappa / 2) * ((A - 2 * np.pi * c)**2 - A**2)
            if st.us[i] < np.exp(-dS) * self.fugacity**dD:
                st.n[link] += c
                local_charge.apply_link_to_F(st.F, mu, site, c, self.N)
                N = self.N
                for cell, dv in dq.items():
                    q1 = st.defects.get(cell, 0) + dv
                    if q1:
                        st.defects[cell] = q1
                    else:
                        st.defects.pop(cell, None)
                    # Mirror into the dense q and compact nonzero list the compiled
                    # step uses, so the two paths can be freely interleaved.
                    rav = ((cell[0] * N + cell[1]) * N + cell[2]) * N + cell[3]
                    if st.q[rav] == 0:
                        st.nzc[st.nnz] = rav
                        st.nnz += 1
                    elif q1 == 0:
                        for b in range(st.nnz):
                            if st.nzc[b] == rav:
                                st.nzc[b] = st.nzc[st.nnz - 1]
                                st.nnz -= 1
                                break
                    st.q[rav] = q1
                st.D += dD
                self.accepted += 1
        # ---- classify the sector at this tick (accepted or not).  This is where the
        # physics is read off: the estimator is pure bookkeeping of WHERE the chain
        # happens to sit.  A rejection is a genuine self-loop and must be counted, or
        # the dwell-time ratio is biased.
        if st.D == 0:
            # Vacuum sector: a valid q ≡ 0 configuration -- one tick of Z.  Close any
            # excursion: power-of-two length bin (bit_length), top bin saturating.
            if st.tstate[0] > 0:
                st.exc_hist[min(31, int(st.tstate[0]).bit_length())] += 1
                st.tstate[1] += 1
                st.tstate[0] = 0
            return True, None, None
        st.tstate[0] += 1
        if st.D == 2 and len(st.defects) == 2:
            # Exactly the worm's G-sector: a single ±1 pair.  (D == 2 alone is not
            # enough -- one cell with |q| = 2 also has D = 2.)
            (c1, v1), (c2, v2) = st.defects.items()
            if v1 == -v2 and abs(v1) == 1:
                plus, minus = (c1, c2) if v1 == 1 else (c2, c1)
                disp = tuple((plus[k] - minus[k]) % self.N for k in range(4))
                # Min-image separation squared: the transport ceiling.
                rsq = sum(min(d, self.N - d)**2 for d in disp)
                if rsq > st.tstate[2]:
                    st.tstate[2] = rsq
                return (False, disp, None)
        if st.D == 4:
            # The two-pair sectors that feed <|M|^4> for the Binder cumulant.  Net-zero
            # D = 4 patterns come in exactly four geometric classes, keyed by the sorted
            # charge multiset; their ordered-insertion multiplicities (4, 2, 2, 1) enter
            # the Binder formula, not the tally.
            cls = self._FOUR.get(tuple(sorted(st.defects.values())))
            if cls is not None:
                return False, None, cls
        # Every other sector is scaffolding: legal states that carry the chain THROUGH
        # jammed backgrounds but never enter the estimators.
        return False, None, None

    # ---------------------------------------------------------------- Generator API

    def inline_observables(self, steps):
        r"""
        We tally ``Vacuum_Ticks``, how often the chain visits the vacuum sector,

        .. math ::

            \texttt{Vacuum\_Ticks} = \sum_{\text{ticks}} \prod [q = 0],

        and ``Theta_Theta``, the pair-sector dwell histogram (translation averaged,
        scaled by its fugacity price $\zeta^2$),

        .. math ::

            \texttt{Theta\_Theta}_{\Delta h} = \frac{1}{V \zeta^2} \sum_{h}
                \prod [q = \delta_{h+\Delta h} - \delta_{h}]

        whose ratio give the :class:`~.Intersection_Intersection` correlator $\Theta$.

        ``Ticks`` counts the total number of proposals the step consumed, so the step's vacuum
        dwell is ``Vacuum_Ticks / Ticks``.

        The ``FourDefectDistribution`` (see :class:`~.FourDefects` and its derivation) is the :class:`~supervillain.generator.no_intersection.DefectGas`'s per-step dwell in
        the four $D = 4$ sector classes, scaled by the known fugacity price $1/\zeta^{4}$.
        Index 0 counts all the charges being on four distinct hypercubes,
        index 1 counts there being a $+2$ charge and two distinct $-1$ charges,
        index 2 counts there being two distinct $+1$ charges and a $-2$ charge, and
        index 3 counts there being both a $+2$ and a $-2$ charge.

        Physical observables probably should consume the
        multiplicity-weighted combination :class:`~.FourDefects`.

        .. note ::

            The $\Delta x = 0$ bin of the ``Theta_Theta`` histogram is exactly zero in every
            configuration --- not because $\Theta_0$ vanishes, but because the chain
            *cannot dwell* there: a coincident $\pm$ pair has $q \equiv 0$, so that
            "sector" is the vacuum itself and its ticks are tallied by
            :class:`~.Vacuum_Ticks` instead.  Physically $\Theta_{0} = 1$ identically
            (coincident insertions are the identity), which is exactly what makes the
            estimator absolutely normalized.  :class:`~.Intersection_Intersection`
            writes that origin value outright, so read $\Theta$ (or sum it, as
            :class:`~.IntersectionSusceptibility` does) from there rather than patching
            the raw histogram.

        Per step these ride along with ``Pair_Excursions``,
        ``Max_Pair_RSq``, and ``Excursion_Lengths``.

        Returns initialized :class:`~supervillain.batch.Batch` storage for each; counters are integers but ``Theta_Theta`` and ``FourDefectDistribution`` are floats because they are scaled by powers of the fugacity.
        """
        return {
            'Theta_Theta': Batch(steps, shape=self.L.dims),
            'Vacuum_Ticks': Batch(steps, shape=(), dtype=np.int64),
            'Ticks': Batch(steps, shape=(), dtype=np.int64),
            'FourDefectDistribution': Batch(steps, shape=(4,), dtype=float),
            'Pair_Excursions': Batch(steps, shape=(), dtype=np.int64),
            'Max_Pair_RSq': Batch(steps, shape=(), dtype=np.int64),
            'Excursion_Lengths': Batch(steps, shape=(32,), dtype=np.int64),
        }

    def _kernel_ticks(self, st, tally, vac_stop, H_pair, H_four):
        # One compiled pass over the remainder of the current proposal batch (stopping
        # early at vac_stop vacuum ticks when positive); mutates the dense chain state
        # in place and returns the number of vacuum ticks seen.
        i0 = st.i
        i, D, nnz, acc, vac = defect_gas_kernel.tick_batch(
            st.F2, st.n2, st.dphi2, st.q, st.nzc, st.D, st.nnz,
            st.mus, st.sites, st.cs, st.us, i0,
            self.kappa, self.fugacity,
            -1 if self.D_max is None else int(self.D_max), self.N,
            *defect_gas_kernel.stencil_pack(),
            H_pair, H_four, tally, vac_stop, st.tstate, st.exc_hist)
        st.i = int(i)
        st.D = int(D)
        st.nnz = int(nnz)
        self.proposed += st.i - i0
        self.accepted += int(acc)
        return int(vac)

    def _step_body(self, configuration, ticker):
        # Shared frame of step and step_reference: state (re)build, the advance-until-
        # emit loop via `ticker`, and the emission.  `ticker` advances the chain and
        # returns (vacuum ticks seen, ticks consumed), tallying into its arguments.
        L, V = self.L, self.N**4
        n_in = np.asarray(configuration['n']).astype(np.int64)
        phi_in = np.asarray(configuration['phi']).astype(float)
        # Rebuild the persistent state if the incoming configuration is not the one we
        # left behind (first call, or another generator in a Sequentially moved it).
        st = self._state
        if st is None or not (np.array_equal(st.n, n_in)
                              and np.array_equal(st.phi, phi_in)):
            st = self._state = self._init_state(phi_in, n_in)
        H_pair = np.zeros(V, dtype=np.int64)
        H_four = np.zeros(4, dtype=np.int64)
        # Per-step transport bookkeeping: the max separation resets each step; the
        # excursion count and length histogram are emitted as this step's increments.
        st.tstate[2] = 0
        exc0 = int(st.tstate[1])
        hist0 = st.exc_hist.copy()
        vacuum = 0
        ticks = 0
        cap = self.max_step_sweeps * st.n_links
        while vacuum < self.emit_every:
            if ticks >= cap:
                raise RuntimeError(
                    f'no {self.emit_every} vacuum ticks in {self.max_step_sweeps} sweeps: '
                    f'fugacity={self.fugacity} is likely too large for this volume/kappa (defect '
                    f'condensation).  Retune (DefectGasFugacityTuner), lower fugacity, or note that '
                    f'a genuinely condensed theta phase requires fugacity ~ 1/V.')
            v, t = ticker(st, self.emit_every - vacuum, H_pair, H_four)
            vacuum += v
            ticks += t
        # Emit AT a vacuum tick: st.n is exactly valid here.
        # φ was frozen for the whole step, so it passes through unchanged; only n is
        # re-emitted (a copy, so the chain's working array stays private).
        return configuration | {
            'n': Form(st.n.copy(), degree=1, lattice=L),
            'Theta_Theta': H_pair.reshape(tuple(L.dims)) / (V * self.fugacity**2),
            'Vacuum_Ticks': int(vacuum),
            'Ticks': int(ticks),
            'FourDefectDistribution': H_four / self.fugacity**4,
            'Pair_Excursions': int(st.tstate[1] - exc0),
            'Max_Pair_RSq': int(st.tstate[2]),
            'Excursion_Lengths': st.exc_hist - hist0,
        }

    def step(self, configuration):
        r"""
        Advance the enlarged chain until its ``emit_every``-th vacuum tick and emit
        that configuration --- the trace of the chain on the constraint surface, so
        every emitted configuration satisfies $q \equiv 0$ exactly and the emitted
        ensemble is the constrained theory.  The pair-sector dwell accumulated along
        the way rides along as the inline ``Theta_Theta`` (already scaled by
        $1/V\zeta^{2}$) and ``Vacuum_Ticks``.

        The tick loop runs in a compiled kernel
        (:func:`~supervillain.generator.no_intersection.defect_gas_kernel.tick_batch`)
        consuming the same pre-drawn proposals as the pure-python
        :meth:`step_reference`, which it reproduces bit-for-bit.
        """
        def ticker(st, vac_stop, H_pair, H_four):
            if st.i == st.n_links:
                self.D_trace.append(st.D)
                self._draw_batch(st)
            i0 = st.i
            vac = self._kernel_ticks(st, True, vac_stop, H_pair, H_four)
            return vac, st.i - i0
        out = self._step_body(configuration, ticker)
        # Mirror the dense charge state back into the sparse dict so step_reference
        # can pick up where the kernel left off.
        st, dims = self._state, tuple(self.L.dims)
        st.defects = {
            tuple(int(x) for x in np.unravel_index(int(cell), dims)): int(st.q[cell])
            for cell in st.nzc[:st.nnz]}
        return out

    def step_reference(self, configuration):
        r"""
        The plain, obviously-correct :meth:`step`: the same advance-until-emit loop
        driven tick-by-tick through the pure-python :meth:`_tick` (sparse defect dict,
        per-proposal stencil dicts).  Kept as the correctness oracle the compiled
        :meth:`step` is validated against --- same proposals, same accept test, same
        tallies, bit-for-bit.
        """
        def ticker(st, vac_stop, H_pair, H_four):
            vac, disp, cls4 = self._tick(st)
            if vac:
                return 1, 1
            if disp is not None:
                N = self.N
                H_pair[((disp[0] * N + disp[1]) * N + disp[2]) * N + disp[3]] += 1
            elif cls4 is not None:
                H_four[cls4] += 1
            return 0, 1
        return self._step_body(configuration, ticker)

    def report(self):
        r"""A short summary: acceptance and the pairs-in-flight trace."""
        # D = Σ|q| is always EVEN (|q| ≡ q mod 2 per site and the total charge Q = Σq
        # vanishes identically), so D/2 -- the number of ±pairs in flight -- is the
        # natural human-facing count.
        Dt = np.array(self.D_trace)
        return '\n'.join([
            f'proposals {self.proposed}  acceptance {self.accepted/max(1,self.proposed):.4f}',
            (f'pairs in flight D/2: mean {Dt.mean()/2:.2f}  max {int(Dt.max())//2}'
             if len(Dt) else 'no sweeps')])


class DefectGasFugacityTuner:
    r"""
    Picks the :class:`DefectGas` fugacity $\zeta$ by short Monte-Carlo probes down (or
    up) a ladder, driving each probe through the ordinary
    :class:`~supervillain.Ensemble` route: every rung runs a throwaway
    ``Sequentially((*companions, DefectGas(S, fugacity)))`` chain and reads the vacuum
    dwell from the emitted inline quantities as ``Vacuum_Ticks / Ticks``.  A rung whose
    chain stops returning to the vacuum (the :meth:`DefectGas.step` ``RuntimeError``)
    is rejected --- that is the defect-condensation signature, not an error.

    A tuner is not a generator: it has no ``step`` and never rides into an
    ensemble.  It runs experiments to decide *which* generator to build;
    :meth:`generator` returns the production-ready chain with fugacity $\zeta$ and
    ``emit_every`` matched by construction.

    Parameters
    ----------
    S: a NoIntersections action
    companions: iterable of generators, optional
        Interleaved with the probe gas (and in the 
        :meth:`generator`).  Defaults to a single
        :class:`~supervillain.generator.villain.SiteUpdate` --- $\phi$ must fluctuate
        or the Villain weights are sampled at frozen $d\phi$.  For honest dwell,
        pass the companions production will run.
    D_max: int or None
        Handed to every probe (and production) :class:`DefectGas`.
    rng: numpy Generator, optional
    """

    def __init__(self, S, companions=None, D_max=8, rng=None):
        self.S = S
        self.D_max = D_max
        self.rng = rng if rng is not None else np.random.default_rng()
        if companions is not None:
            self.companions = tuple(companions)
        else:
            default = SiteUpdate(S)
            default.rng = self.rng
            self.companions = (default,)

    def _probe(self, fugacity, start, steps, emit_every, max_step_sweeps):
        # One rung: a throwaway Generator-route chain.  Returns the per-step
        # (Vacuum_Ticks, Ticks) arrays, or None if the chain condensed (the step
        # RuntimeError) -- an unhealthy rung, not an error.
        import supervillain.ensemble
        gas = DefectGas(self.S, fugacity, D_max=self.D_max, emit_every=emit_every,
                        max_step_sweeps=max_step_sweeps, rng=self.rng)
        chain = Sequentially((*self.companions, gas))
        try:
            e = supervillain.ensemble.Ensemble(self.S).generate(steps, chain,
                                                                start=start)
        except RuntimeError:
            return None
        return np.asarray(e.Vacuum_Ticks), np.asarray(e.Ticks)

    def tune(self, start='cold', ladder=(0.1, 0.05, 0.02, 0.01, 0.005, 0.002),
             steps=120, target=0.15):
        r"""
        Descend the ladder and keep the first fugacity $\zeta$ whose vacuum dwell exceeds
        ``target`` (falling back to the smallest rung if none does).  Because the
        estimator is $\zeta$-independent, tuning affects only the variance, never the
        answer.  Entropy pushes the defect count up, so the right $\zeta$ shrinks with
        volume and with $1/\kappa$.

        Each rung probes ``steps`` configurations from ``start`` with a small
        ``emit_every`` (a healthy step then costs about a sweep); the first half is
        per-rung equilibration and the dwell is read off the second half.

        Parameters
        ----------
        start: 'cold', or a configuration as a dictionary
            Where each rung's probe chain starts; anything
            :meth:`~supervillain.Ensemble.generate` accepts.  Probes from a
            thermalized valid configuration measure the equilibrium dwell; a fresh
            start's early dwell is optimistic.
        ladder: iterable of floats
            Candidate fugacities in descending order; the first rung whose dwell
            exceeds ``target`` wins.
        steps: int
            Configurations per probe; the first half is per-rung equilibration and
            the dwell is read off the second half.
        target: float
            The vacuum-dwell fraction a rung must exceed to be kept.

        Returns
        -------
        float
            The chosen $\zeta$.
        """
        V = self.S.Lattice.N ** 4
        fugacity = ladder[-1]
        condensed = True
        for z in ladder:
            fugacity = z
            probe = self._probe(z, start, steps,
                                emit_every=max(1, round(target * 4 * V)),
                                max_step_sweeps=25)
            if probe is None:
                condensed = True
                continue                        # condensed: descend
            vac, ticks = (a[steps // 2:] for a in probe)
            condensed = False
            if vac.sum() / max(1.0, ticks.sum()) > target:
                break
        if condensed:
            logger.warning(
                'tune: every ladder rung condensed; returning the smallest rung %g, '
                'which itself condensed -- production will likely raise.  Investigate '
                'D_trace / lower the ladder.', fugacity)
        return fugacity

    def tune_edge(self, start='cold',
                  ladder=(0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05,
                          0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3),
                  steps=200, min_vacuum_ticks=500, max_probe_sweeps=4000,
                  step_sweeps=25, floor=None):
        r"""
        Ride the edge: ascend the ladder and keep the *largest* fugacity $\zeta$ at which
        the chain still demonstrably returns to the vacuum --- pair transport scales
        like $\zeta^{2}$, so a conservatively small $\zeta$ silently censors exactly the
        large-separation bins that diagnose $\theta$ order.

        A rung is accepted iff its probe completes (each completed step *is*
        ``emit_every`` vacuum ticks, so completing the tallied half collects
        ``min_vacuum_ticks``) **and** the dwell is stationary across the tallied half
        (its second half at least a quarter of its first: a collapsing dwell is
        condensation in progress).  An explicit ``floor`` may be imposed on top.

        Parameters
        ----------
        start: 'cold', or a configuration as a dictionary
            Where each rung's probe chain starts; anything
            :meth:`~supervillain.Ensemble.generate` accepts.
        ladder: iterable of floats
            Candidate fugacities in ascending order; the largest measurable,
            stationary rung wins and the first failing rung stops the ascent.
        steps: int
            Configurations per probe; the first half is per-rung equilibration and
            the tallied second half must collect ``min_vacuum_ticks``.
        min_vacuum_ticks: int
            Vacuum ticks the tallied half must collect for the rung to count as
            measurable; sets each probe step's ``emit_every``.
        max_probe_sweeps: int
            Sweep budget per rung, divided across its ``steps`` as each probe step's
            horizon; a rung that exhausts a step's share is rejected as condensed.
        step_sweeps: int
            Target sweeps per production step; sizes the returned ``emit_every``
            from the measured dwell.
        floor: float, optional
            An explicit minimum dwell imposed on top of measurability.

        Returns
        -------
        (float, int)
            The chosen $\zeta$ and a matched ``emit_every`` sized so one production
            step costs about ``step_sweeps`` sweeps at the measured dwell.
        """
        if not ladder:
            raise RuntimeError(
                'tune_edge: an empty ladder has no rung to probe (all candidate '
                'fugacities already condensed?).  Treat as signal and investigate D_trace.')
        n_links = 4 * self.S.Lattice.N ** 4
        best = None
        for z in ladder:
            probe = self._probe(
                z, start, steps,
                emit_every=max(1, -(-min_vacuum_ticks // max(1, steps // 2))),
                max_step_sweeps=max(1, max_probe_sweeps // steps))
            if probe is None:
                break                           # past the edge; dwell falls monotonically
            vac, ticks = (a[steps // 2:] for a in probe)
            half = len(vac) // 2
            dwell = vac.sum() / max(1.0, ticks.sum())
            d1 = vac[:half].sum() / max(1.0, ticks[:half].sum())
            d2 = vac[half:].sum() / max(1.0, ticks[half:].sum())
            if (vac.sum() >= min_vacuum_ticks and d2 >= d1 / 4
                    and (floor is None or dwell > floor)):
                best = (z, dwell)
            else:
                break
        if best is None:
            raise RuntimeError(
                f'tune_edge: even the smallest ladder rung fugacity={ladder[0]} never '
                f'completed its probe; the chain cannot emit here (defect '
                f'condensation?).  Treat as signal and investigate D_trace.')
        fugacity, dwell = best
        return fugacity, max(1, int(round(dwell * n_links * step_sweeps)))

    def generator(self, start='cold', edge=False, **kwargs):
        r"""
        The normal way to consume a tune: probe from ``start`` (with :meth:`tune`, or
        :meth:`tune_edge` when ``edge``; ``kwargs`` forward), then return the
        production-ready ``Sequentially((*companions, DefectGas(...)))`` with $\zeta$ and
        ``emit_every`` matched by construction.  The chain carries ``fugacity``
        and ``emit_every`` as plain metadata for introspection.

        Parameters
        ----------
        start: 'cold', or a configuration as a dictionary
            Where the tuning probes start; anything
            :meth:`~supervillain.Ensemble.generate` accepts.
        edge: bool
            Tune with :meth:`tune_edge` (largest measurable $\zeta$, matched
            ``emit_every``) instead of :meth:`tune` (dwell target, default
            ``emit_every``).
        kwargs:
            Forwarded to :meth:`tune` or :meth:`tune_edge`.

        Returns
        -------
        :class:`~supervillain.generator.combining.Sequentially`
            The production-ready chain, its tuned :class:`DefectGas` last.
        """
        if edge:
            fugacity, emit_every = self.tune_edge(start=start, **kwargs)
        else:
            fugacity = self.tune(start=start, **kwargs)
            emit_every = None
        gas = DefectGas(self.S, fugacity, D_max=self.D_max, emit_every=emit_every,
                        rng=self.rng)
        chain = Sequentially((*self.companions, gas))
        chain.fugacity = gas.fugacity
        chain.emit_every = gas.emit_every
        return chain
